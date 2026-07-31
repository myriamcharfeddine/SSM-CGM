"""Evaluation utilities for AI-READI stateful stream checkpoints."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..data.aireadi import AireadiFeatureSpec, AireadiParticipantStream
from ..models.aireadi_stream import AireadiStreamModel


def qcol(level: float) -> str:
    return f"q{round(float(level) * 100):02d}"


def _valid_anchors(stream: AireadiParticipantStream, horizon: int, stride: int,
                   warmup_steps: int = 0, max_anchors: Optional[int] = None) -> List[int]:
    anchors = []
    last = -10 ** 9
    for t in range(0, stream.n_steps - horizon):
        if t + 1 < warmup_steps:
            continue
        if (t - last) < stride:
            continue
        if bool(stream.observed[t]) and bool(stream.observed[t + 1:t + 1 + horizon].all()):
            anchors.append(t)
            last = t
            if max_anchors is not None and len(anchors) >= max_anchors:
                break
    return anchors


def _apply_proxy_scenario(model: AireadiStreamModel, values: torch.Tensor, mask: torch.Tensor,
                          mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    names = model.feature_spec.scenario_reals
    if mode == "forecast_only":
        return values, torch.zeros_like(mask)
    if mode == "factual_future":
        return values, mask
    values = values.clone()
    out_mask = torch.zeros_like(mask)
    groups = model.feature_spec.scenario_groups
    if mode == "meal_proxy":
        selected = groups.get("meal_proxy", [])
        for name in selected:
            if name in names:
                j = names.index(name)
                values[..., j] = 1.0
                out_mask[..., j] = 1.0
    elif mode == "activity_proxy":
        selected = groups.get("activity_proxy", [])
        for name in selected:
            if name in names:
                j = names.index(name)
                values[..., j] = values[..., j] + 1.0
                out_mask[..., j] = 1.0
    elif mode == "sleep_rest_proxy":
        selected = groups.get("sleep_rest_proxy", [])
        for name in selected:
            if name in names:
                j = names.index(name)
                out_mask[..., j] = 1.0
                if name in {"activity_stage_sedentary", "sleep_stage_light", "sleep_stage_deep", "sleep_stage_rem"}:
                    values[..., j] = values[..., j] + 1.0
    else:
        raise ValueError(f"unknown scenario mode {mode!r}")
    return values, out_mask


@torch.no_grad()
def evaluate_aireadi_streams(
    model: AireadiStreamModel,
    streams: Sequence[AireadiParticipantStream],
    *,
    scenario_mode: str = "forecast_only",
    anchor_stride_steps: int = 3,
    warmup_steps: int = 0,
    max_anchors_per_stream: Optional[int] = None,
    device: str = "cpu",
    record_latency: bool = False,
) -> Dict[str, object]:
    device_obj = torch.device(device)
    model.to(device_obj).eval()
    rows: List[dict] = []
    memory_rows: List[dict] = []
    update_ms: List[float] = []
    decode_ms: List[float] = []
    H = model.feature_spec.horizon_steps
    qlevels = model.quantiles

    for stream_idx, raw_stream in enumerate(streams):
        stream = raw_stream.to(device_obj)
        sctx = model.encode_static(stream.static_cat, stream.static_cont)
        state = model.init_stream(sctx)
        anchors = _valid_anchors(stream, H, anchor_stride_steps, warmup_steps, max_anchors_per_stream)

        if record_latency and stream_idx == 0:
            sample_n = min(stream.n_steps, 64)
            for t in range(sample_n):
                t0 = time.perf_counter()
                state = model.update_stream(state, stream.dynamic[t])
                if device_obj.type == "cuda":
                    torch.cuda.synchronize()
                update_ms.append((time.perf_counter() - t0) * 1000.0)
        sctx = model.encode_static(stream.static_cat, stream.static_cont)
        state = model.init_stream(sctx)
        state, out = model.scan_chunk(stream.dynamic, sctx, state)
        if not anchors:
            memory_rows.append(_memory_row(stream, state, 0))
            continue

        pos = torch.tensor(anchors, dtype=torch.long, device=device_obj)
        fut = pos[:, None] + 1 + torch.arange(H, device=device_obj)[None, :]
        values, masks = _apply_proxy_scenario(
            model,
            stream.scenario_values[fut],
            stream.scenario_mask[fut],
            scenario_mode,
        )
        t0 = time.perf_counter()
        raw = model.decode_horizon(out[0, pos], sctx, stream.time_features[fut], values, masks)
        if device_obj.type == "cuda":
            torch.cuda.synchronize()
        if record_latency:
            decode_ms.append((time.perf_counter() - t0) * 1000.0 / max(len(anchors), 1))
        pred = stream.target[pos].view(-1, 1, 1) + raw
        pred = pred.cpu().numpy()
        target = stream.target.cpu().numpy()
        time_idx = stream.time_idx.cpu().numpy()
        for ai, anchor in enumerate(anchors):
            elapsed_h = (anchor + 1) * model.feature_spec.bin_minutes / 60.0
            base = {
                "participant_id": stream.participant_id,
                "segment_id": stream.segment_id,
                "split": stream.split,
                "anchor_time_idx": int(time_idx[anchor]),
                "anchor_timestamp": pd.Timestamp(stream.timestamps[anchor]),
                "steps_since_start": int(anchor + 1),
                "hours_since_start": float(elapsed_h),
                "scenario_mode": scenario_mode,
            }
            base.update(stream.metadata)
            for h in range(H):
                row = dict(base)
                row.update({
                    "horizon_step": h + 1,
                    "horizon_minutes": model.feature_spec.bin_minutes * (h + 1),
                    "target": float(target[anchor + 1 + h]),
                    "observed": True,
                })
                for qi, level in enumerate(qlevels):
                    row[qcol(level)] = float(pred[ai, h, qi])
                rows.append(row)
        memory_rows.append(_memory_row(stream, state, len(anchors)))

    pred_df = pd.DataFrame(rows)
    return {
        "predictions": pred_df,
        "memory": pd.DataFrame(memory_rows),
        "hardware": summarize_hardware(update_ms, decode_ms, device_obj),
        "n_participants": len({s.participant_id for s in streams}),
    }


def _memory_row(stream: AireadiParticipantStream, state, n_anchors: int) -> dict:
    bytes_total = 0
    for tensor in list(state.layer_states) + list(state.conv_states or []):
        if tensor is not None:
            bytes_total += int(tensor.numel() * tensor.element_size())
    if state.last_output is not None:
        bytes_total += int(state.last_output.numel() * state.last_output.element_size())
    return {
        "participant_id": stream.participant_id,
        "segment_id": stream.segment_id,
        "n_steps": stream.n_steps,
        "n_anchors": n_anchors,
        "state_bytes": bytes_total,
        "state_kib": bytes_total / 1024.0,
    }


def summarize_hardware(update_ms: Sequence[float], decode_ms: Sequence[float], device: torch.device) -> dict:
    def stats(xs):
        arr = np.asarray(list(xs), dtype="float64")
        if arr.size == 0:
            return {"n": 0, "mean_ms": np.nan, "median_ms": np.nan, "p95_ms": np.nan}
        return {
            "n": int(arr.size),
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
        }

    out = {
        "latency_per_update": stats(update_ms),
        "latency_per_1h_forecast": stats(decode_ms),
        "device": str(device),
        "peak_gpu_memory_mb": None,
        "cpu_memory_rss_mb": None,
    }
    if device.type == "cuda":
        out["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / 1e6)
    try:
        import psutil
        import os

        out["cpu_memory_rss_mb"] = float(psutil.Process(os.getpid()).memory_info().rss / 1e6)
    except Exception:
        pass
    return out


def aggregate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan,
            "tir_true": np.nan, "tir_predicted": np.nan, "tir_gap": np.nan,
            "coverage": np.nan, "coverage80": np.nan, "interval_coverage80": np.nan,
            "p90_abs_error": np.nan, "p95_abs_error": np.nan, "p99_abs_error": np.nan,
        }
    pred = df[qcol(0.5)].astype(float)
    target = df["target"].astype(float)
    err = pred - target
    abs_err = err.abs()
    tir_true = ((target >= 70.0) & (target <= 180.0)).mean()
    tir_pred = ((pred >= 70.0) & (pred <= 180.0)).mean()
    lo = df[qcol(0.1)].astype(float) if qcol(0.1) in df else pred
    hi = df[qcol(0.9)].astype(float) if qcol(0.9) in df else pred
    coverage = float(((target >= lo) & (target <= hi)).mean())
    return {
        "n": int(len(df)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "tir_true": float(tir_true),
        "tir_predicted": float(tir_pred),
        "tir_gap": float(tir_pred - tir_true),
        "coverage": coverage,
        "coverage80": coverage,
        "interval_coverage80": coverage,
        "p90_abs_error": float(abs_err.quantile(0.90)),
        "p95_abs_error": float(abs_err.quantile(0.95)),
        "p99_abs_error": float(abs_err.quantile(0.99)),
    }


def metrics_by(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols or df.empty:
        return pd.DataFrame()
    rows = []
    for key, group in df.groupby(cols, dropna=False, observed=False):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(cols, key))
        rec.update(aggregate_metrics(group))
        rows.append(rec)
    return pd.DataFrame(rows)


def clinical_safety_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    pred = df[qcol(0.5)].astype(float)
    target = df["target"].astype(float)

    def event(threshold, op):
        if op == "lt":
            yt = target < threshold
            yp = pred < threshold
        else:
            yt = target > threshold
            yp = pred > threshold
        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}

    metrics = aggregate_metrics(df)
    metrics.update({
        "hypoglycemia_detection": event(70.0, "lt"),
        "hyperglycemia_detection": event(180.0, "gt"),
    })
    return metrics


def personalization_sweep(df: pd.DataFrame, warmup_hours: Sequence[float]) -> pd.DataFrame:
    rows = []
    for hours in warmup_hours:
        rec = {"warmup_hours": float(hours)}
        rec.update(aggregate_metrics(df[df["hours_since_start"] >= float(hours)]))
        rows.append(rec)
    return pd.DataFrame(rows)


def _correct_by_bias(df: pd.DataFrame, bias: float) -> pd.DataFrame:
    out = df.copy()
    for col in [c for c in [qcol(0.1), qcol(0.5), qcol(0.9)] if c in out.columns]:
        out[col] = out[col] - bias
    return out


def bias_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw = aggregate_metrics(df)
    raw["bias_mode"] = "raw"
    rows.append(raw)
    global_bias = raw.get("bias", 0.0)
    corrected = aggregate_metrics(_correct_by_bias(df, global_bias))
    corrected["bias_mode"] = "offset-corrected"
    rows.append(corrected)

    personalized = df.copy()
    if not personalized.empty:
        early = personalized[personalized["hours_since_start"] <= 6.0]
        per_pid = (early[qcol(0.5)] - early["target"]).groupby(early["participant_id"]).mean()
        personalized["_bias"] = personalized["participant_id"].map(per_pid).fillna(0.0)
        for col in [c for c in [qcol(0.1), qcol(0.5), qcol(0.9)] if c in personalized.columns]:
            personalized[col] = personalized[col] - personalized["_bias"]
    rec = aggregate_metrics(personalized.drop(columns=["_bias"], errors="ignore"))
    rec["bias_mode"] = "personalized"
    rows.append(rec)
    rec = aggregate_metrics(_correct_by_bias(personalized.drop(columns=["_bias"], errors="ignore"), global_bias))
    rec["bias_mode"] = "personalized+offset-corrected"
    rows.append(rec)
    return pd.DataFrame(rows)


def subgroup_metrics(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "hba1c_percent_baseline" in work.columns:
        work["HbA1c_quartile"] = pd.qcut(pd.to_numeric(work["hba1c_percent_baseline"], errors="coerce"),
                                         q=4, duplicates="drop")
    if "bmi_baseline" in work.columns:
        work["BMI_quartile"] = pd.qcut(pd.to_numeric(work["bmi_baseline"], errors="coerce"),
                                       q=4, duplicates="drop")
    groups = [
        "participants_study_group", "HbA1c_quartile", "BMI_quartile",
        "participants_clinical_site", "med_insulin", "med_any_diabetes_drug",
    ]
    frames = []
    for col in groups:
        tab = metrics_by(work, [col])
        if not tab.empty:
            tab.insert(0, "subgroup", col)
            frames.append(tab.rename(columns={col: "level"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



def _write_basic_figures(fig_dir: Path, horizon_tab: pd.DataFrame, warmup_tab: pd.DataFrame) -> None:
    """Write lightweight diagnostic figures when matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not horizon_tab.empty and {"horizon_minutes", "mae"}.issubset(horizon_tab.columns):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(horizon_tab["horizon_minutes"], horizon_tab["mae"], marker="o")
            ax.set_xlabel("Horizon minutes")
            ax.set_ylabel("MAE (mg/dL)")
            ax.set_title("AI-READI stream horizon error")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_dir / "mae_by_horizon.png", dpi=150)
            plt.close(fig)
        if not warmup_tab.empty and {"warmup_hours", "mae"}.issubset(warmup_tab.columns):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(warmup_tab["warmup_hours"], warmup_tab["mae"], marker="o")
            ax.set_xlabel("Warm-up hours")
            ax.set_ylabel("MAE (mg/dL)")
            ax.set_title("AI-READI stream personalization sweep")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_dir / "personalization_sweep.png", dpi=150)
            plt.close(fig)
    except Exception as exc:
        (fig_dir / "figure_generation_failed.txt").write_text(str(exc))

def write_evaluation_bundle(
    result: Dict[str, object],
    output_dir,
    *,
    warmup_hours: Sequence[float] = (0, 6, 12, 24, 48),
    config: Optional[dict] = None,
) -> Dict[str, object]:
    out = Path(output_dir)
    for sub in ["predictions", "metrics", "figures", "hardware"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    pred = result["predictions"]
    pred_path = out / "predictions" / "predictions.parquet"
    pred.to_parquet(pred_path, index=False)
    result["memory"].to_csv(out / "hardware" / "stream_state_memory.csv", index=False)
    with (out / "hardware" / "hardware_metrics.json").open("w") as f:
        json.dump(result["hardware"], f, indent=2)
    overall = aggregate_metrics(pred)
    with (out / "metrics" / "overall_metrics.json").open("w") as f:
        json.dump(overall, f, indent=2)
    metrics_by(pred, ["scenario_mode"]).to_csv(out / "metrics" / "scenario_metrics.csv", index=False)
    horizon_tab = metrics_by(pred, ["horizon_step", "horizon_minutes"])
    horizon_tab.to_csv(out / "metrics" / "horizon_metrics.csv", index=False)
    warmup_tab = personalization_sweep(pred, warmup_hours)
    warmup_tab.to_csv(out / "metrics" / "personalization_sweep.csv", index=False)
    _write_basic_figures(out / "figures", horizon_tab, warmup_tab)
    bias_diagnostic(pred).to_csv(out / "metrics" / "bias_diagnostic.csv", index=False)
    subgroup_metrics(pred).to_csv(out / "metrics" / "subgroup_metrics.csv", index=False)
    with (out / "metrics" / "clinical_safety.json").open("w") as f:
        json.dump(clinical_safety_metrics(pred), f, indent=2)
    if config is not None:
        with (out / "config_resolved.yaml").open("w") as f:
            try:
                import yaml

                yaml.safe_dump(config, f, sort_keys=False)
            except Exception:
                json.dump(config, f, indent=2)
    return {"overall": overall, "prediction_file": str(pred_path), "n_rows": int(len(pred))}
