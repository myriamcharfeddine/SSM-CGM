"""Scenario pathway diagnostics for AI-READI stream predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .diagnostics import ANCHOR_KEY, ROW_KEY, Q50, aggregate_metrics, forecast_only, save_table


def scenario_prediction_deltas(predictions: pd.DataFrame, metrics_dir, figures_dir, *, inactive_epsilon: float = 1e-3) -> tuple[List[str], List[str]]:
    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    warnings: List[str] = []
    base = forecast_only(predictions)[ROW_KEY + ["horizon_minutes", Q50]].rename(columns={Q50: "q50_forecast_only"})
    rows = []
    for mode in sorted(m for m in predictions["scenario_mode"].dropna().unique() if m != "forecast_only"):
        sdf = predictions[predictions["scenario_mode"] == mode]
        merged = sdf.merge(base, on=ROW_KEY + ["horizon_minutes"], how="inner", validate="one_to_one")
        if merged.empty:
            continue
        merged["delta"] = pd.to_numeric(merged[Q50], errors="coerce") - pd.to_numeric(merged["q50_forecast_only"], errors="coerce")
        mode_metrics = aggregate_metrics(sdf)
        overall = {
            "scenario_mode": mode,
            "horizon_step": 0,
            "horizon_minutes": 0,
            "scope": "overall",
            **mode_metrics,
            "mean_abs_delta_vs_forecast_only": float(merged["delta"].abs().mean()),
            "median_abs_delta_vs_forecast_only": float(merged["delta"].abs().median()),
            "p95_abs_delta_vs_forecast_only": float(merged["delta"].abs().quantile(0.95)),
            "mean_signed_delta_vs_forecast_only": float(merged["delta"].mean()),
        }
        rows.append(overall)
        for (hstep, hmin), g in merged.groupby(["horizon_step", "horizon_minutes"], dropna=False):
            gm = aggregate_metrics(sdf[(sdf["horizon_step"] == hstep) & (sdf["horizon_minutes"] == hmin)])
            rows.append({
                "scenario_mode": mode,
                "horizon_step": int(hstep),
                "horizon_minutes": int(hmin),
                "scope": "horizon",
                **gm,
                "mean_abs_delta_vs_forecast_only": float(g["delta"].abs().mean()),
                "median_abs_delta_vs_forecast_only": float(g["delta"].abs().median()),
                "p95_abs_delta_vs_forecast_only": float(g["delta"].abs().quantile(0.95)),
                "mean_signed_delta_vs_forecast_only": float(g["delta"].mean()),
            })
    out = pd.DataFrame(rows)
    csv_path = metrics_dir / "scenario_prediction_deltas.csv"
    save_table(out, csv_path)
    overall = out[out["scope"] == "overall"] if not out.empty else pd.DataFrame()
    if not overall.empty:
        max_delta = float(overall["mean_abs_delta_vs_forecast_only"].max())
        print("[diagnostics] scenario mean_abs_delta_vs_forecast_only:")
        for r in overall.itertuples(index=False):
            print(f"  {r.scenario_mode}: {r.mean_abs_delta_vs_forecast_only:.6f}")
        if max_delta < inactive_epsilon:
            warning = "Scenario pathway appears inactive or not exercised; do not interpret proxy scenario effects."
            print(f"[diagnostics] WARNING: {warning}")
            warnings.append(warning)
    fig_path = figures_dir / "scenario_delta_by_horizon.png"
    try:
        import matplotlib.pyplot as plt
        htab = out[out["scope"] == "horizon"].copy()
        fig, ax = plt.subplots(figsize=(7, 4))
        for mode, g in htab.groupby("scenario_mode"):
            ax.plot(g["horizon_minutes"], g["mean_abs_delta_vs_forecast_only"], marker="o", label=mode)
        ax.set_xlabel("Horizon minutes")
        ax.set_ylabel("Mean |delta q50| vs forecast_only (mg/dL)")
        ax.set_title("Scenario pathway delta by horizon")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        files = [str(csv_path), str(fig_path)]
    except Exception as exc:
        fail = figures_dir / "scenario_delta_by_horizon_failed.txt"
        fail.write_text(str(exc))
        files = [str(csv_path), str(fail)]
    return files, warnings


def scenario_pathway_audit(predictions: pd.DataFrame, streams, feature_spec, metrics_dir) -> List[str]:
    metrics_dir = Path(metrics_dir)
    fdf = forecast_only(predictions)
    needed: Dict[Tuple[str, int], set] = {}
    for r in fdf[ANCHOR_KEY].drop_duplicates().itertuples(index=False):
        needed.setdefault((str(r.participant_id), int(r.segment_id)), set()).add(int(r.anchor_time_idx))
    names = list(feature_spec.scenario_reals)
    stats = {
        name: {
            "scenario_variable": name,
            "n_anchor_horizon_values": 0,
            "mask_sum": 0.0,
            "mask_nonzero": 0,
            "value_sum": 0.0,
            "value_sumsq": 0.0,
            "value_min": np.inf,
            "value_max": -np.inf,
            "anchors_with_mask": 0,
            "n_anchors": 0,
        }
        for name in names
    }
    H = int(feature_spec.horizon_steps)
    for stream in streams:
        skey = (str(stream.participant_id), int(stream.segment_id))
        anchors = needed.get(skey)
        if not anchors:
            continue
        values = stream.scenario_values.detach().cpu().numpy() if hasattr(stream.scenario_values, "detach") else np.asarray(stream.scenario_values)
        masks = stream.scenario_mask.detach().cpu().numpy() if hasattr(stream.scenario_mask, "detach") else np.asarray(stream.scenario_mask)
        time_idx = stream.time_idx.detach().cpu().numpy() if hasattr(stream.time_idx, "detach") else np.asarray(stream.time_idx)
        pos_by_time = {int(t): i for i, t in enumerate(time_idx.tolist())}
        for anchor_t in anchors:
            pos = pos_by_time.get(int(anchor_t))
            if pos is None or pos + H >= len(values):
                continue
            fut = slice(pos + 1, pos + 1 + H)
            v = values[fut]
            m = masks[fut]
            for j, name in enumerate(names):
                st = stats[name]
                vv = v[:, j].astype(float)
                mm = m[:, j].astype(float)
                st["n_anchor_horizon_values"] += int(vv.size)
                st["n_anchors"] += 1
                st["mask_sum"] += float(mm.sum())
                st["mask_nonzero"] += int((mm != 0).sum())
                st["value_sum"] += float(vv.sum())
                st["value_sumsq"] += float((vv ** 2).sum())
                st["value_min"] = min(st["value_min"], float(np.nanmin(vv)))
                st["value_max"] = max(st["value_max"], float(np.nanmax(vv)))
                st["anchors_with_mask"] += int((mm != 0).any())
    rows = []
    for name in names:
        st = stats[name]
        n = max(int(st["n_anchor_horizon_values"]), 1)
        mean = st["value_sum"] / n
        var = max(st["value_sumsq"] / n - mean ** 2, 0.0)
        rows.append({
            "scenario_variable": name,
            "n_anchors": int(st["n_anchors"]),
            "n_anchor_horizon_values": int(st["n_anchor_horizon_values"]),
            "mask_mean": st["mask_sum"] / n,
            "mask_nonzero_pct": st["mask_nonzero"] / n,
            "value_mean": mean,
            "value_std": float(np.sqrt(var)),
            "value_min": np.nan if st["value_min"] == np.inf else st["value_min"],
            "value_max": np.nan if st["value_max"] == -np.inf else st["value_max"],
            "anchors_with_scenario_available": int(st["anchors_with_mask"]),
            "anchor_available_pct": st["anchors_with_mask"] / max(int(st["n_anchors"]), 1),
        })
    csv_path = metrics_dir / "scenario_pathway_audit.csv"
    save_table(pd.DataFrame(rows), csv_path)
    return [str(csv_path)]
