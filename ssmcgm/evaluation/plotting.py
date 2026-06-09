"""Plotting helpers for AI-READI diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .diagnostics import save_table, safe_level, summarize_rows


def matched_personalization(predictions: pd.DataFrame, metrics_dir, figures_dir, warmup_hours=(0, 6, 12, 24, 48)) -> List[str]:
    from .diagnostics import aggregate_metrics, forecast_only
    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    df = forecast_only(predictions)
    max_warmup = max(float(x) for x in warmup_hours)
    anchor_cols = ["participant_id", "segment_id", "anchor_time_idx"]
    eligible = df[df["hours_since_start"] >= max_warmup][anchor_cols].drop_duplicates()
    key = eligible.assign(_eligible=True)
    matched = df.merge(key, on=anchor_cols, how="inner")
    rows = []
    for hours in warmup_hours:
        rec = {"warmup_hours": float(hours), "matched_anchor_max_warmup_hours": max_warmup}
        rec.update(aggregate_metrics(matched))
        rec["n_anchors"] = int(eligible.shape[0])
        rec["n_participants"] = int(matched["participant_id"].nunique()) if not matched.empty else 0
        rows.append(rec)
    out = pd.DataFrame(rows)
    csv_path = metrics_dir / "personalization_matched_anchor.csv"
    save_table(out, csv_path)
    fig_path = figures_dir / "personalization_matched_anchor.png"
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(out["warmup_hours"], out["mae"], marker="o")
        ax.set_xlabel("Warm-up hours")
        ax.set_ylabel("MAE on matched anchors (mg/dL)")
        ax.set_title("Matched-anchor personalization comparison")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        return [str(csv_path), str(fig_path)]
    except Exception as exc:
        fail = figures_dir / "personalization_matched_anchor_failed.txt"
        fail.write_text(str(exc))
        return [str(csv_path), str(fail)]


def subgroup_outputs(participant_df: pd.DataFrame, metrics_dir, figures_dir) -> List[str]:
    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    files: List[str] = []
    work = participant_df[participant_df["scenario_mode"] == "forecast_only"].copy() if "scenario_mode" in participant_df.columns else participant_df.copy()
    if work.empty:
        return files
    if "hba1c_percent_baseline" in work.columns:
        work["hba1c_quartile"] = pd.qcut(pd.to_numeric(work["hba1c_percent_baseline"], errors="coerce"), q=4, duplicates="drop").astype(str)
    if "bmi_baseline" in work.columns:
        work["bmi_quartile"] = pd.qcut(pd.to_numeric(work["bmi_baseline"], errors="coerce"), q=4, duplicates="drop").astype(str)
    specs = [
        ("participants_study_group", "subgroup_mae_study_group.png"),
        ("hba1c_quartile", "subgroup_mae_hba1c.png"),
        ("med_insulin", "subgroup_mae_med_insulin.png"),
        ("med_any_diabetes_drug", "subgroup_mae_med_any_diabetes_drug.png"),
        ("participants_clinical_site", "subgroup_mae_site.png"),
        ("bmi_quartile", "subgroup_mae_bmi.png"),
    ]
    rows = []
    for col, _ in specs:
        if col not in work.columns:
            continue
        for level, g in work.groupby(col, dropna=False, observed=False):
            rec = {"subgroup": col, "level": safe_level(level), "n_participants": int(len(g))}
            for metric in ["mae", "bias", "tir_gap", "coverage"]:
                vals = pd.to_numeric(g[metric], errors="coerce").dropna() if metric in g.columns else pd.Series(dtype=float)
                rec[f"{metric}_mean"] = float(vals.mean()) if not vals.empty else float("nan")
                rec[f"{metric}_median"] = float(vals.median()) if not vals.empty else float("nan")
            rows.append(rec)
    csv_path = metrics_dir / "subgroup_participant_level_metrics.csv"
    save_table(pd.DataFrame(rows), csv_path)
    files.append(str(csv_path))
    try:
        import matplotlib.pyplot as plt
        for col, fname in specs:
            if col not in work.columns:
                continue
            tab = work.groupby(col, dropna=False, observed=False)["mae"].mean().reset_index().sort_values("mae")
            if tab.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(tab)), 4))
            ax.bar([safe_level(x) for x in tab[col]], tab["mae"])
            ax.set_ylabel("Participant-level MAE (mg/dL)")
            ax.set_title(col)
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            path = figures_dir / fname
            fig.savefig(path, dpi=150)
            plt.close(fig)
            files.append(str(path))
    except Exception as exc:
        fail = figures_dir / "subgroup_plots_failed.txt"
        fail.write_text(str(exc))
        files.append(str(fail))
    return files
