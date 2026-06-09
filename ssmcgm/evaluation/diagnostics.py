"""Shared helpers for AI-READI evaluation diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Q10 = "q10"
Q50 = "q50"
Q90 = "q90"
ANCHOR_KEY = ["participant_id", "segment_id", "anchor_time_idx"]
ROW_KEY = ["participant_id", "segment_id", "anchor_time_idx", "horizon_step"]


def ensure_output_dirs(output_dir) -> Dict[str, Path]:
    out = Path(output_dir)
    dirs = {
        "root": out,
        "metrics": out / "metrics",
        "figures": out / "figures",
        "tables": out / "tables",
        "logs": out / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def load_predictions(eval_dir) -> pd.DataFrame:
    path = Path(eval_dir) / "predictions" / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Prediction parquet not found: {path}")
    df = pd.read_parquet(path)
    required = set(ROW_KEY + ["scenario_mode", "horizon_minutes", "target", Q50])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction parquet is missing required columns: {missing}")
    return df


def forecast_only(df: pd.DataFrame) -> pd.DataFrame:
    if "scenario_mode" not in df.columns:
        return df.copy()
    return df[df["scenario_mode"] == "forecast_only"].copy()


def anonymize_participants(ids: Iterable[object]) -> Dict[str, str]:
    vals = sorted({str(x) for x in ids})
    return {pid: f"P{i + 1:05d}" for i, pid in enumerate(vals)}


def aggregate_metrics(df: pd.DataFrame, pred_col: str = Q50, lo_col: str = Q10, hi_col: str = Q90) -> Dict[str, float]:
    if df.empty:
        return {
            "n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan,
            "tir_true": np.nan, "tir_predicted": np.nan, "tir_gap": np.nan,
            "coverage": np.nan, "coverage80": np.nan,
            "p90_abs_error": np.nan, "p95_abs_error": np.nan, "p99_abs_error": np.nan,
        }
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    target = pd.to_numeric(df["target"], errors="coerce")
    valid = pred.notna() & target.notna()
    pred = pred[valid]
    target = target[valid]
    if len(pred) == 0:
        return aggregate_metrics(pd.DataFrame())
    err = pred - target
    abs_err = err.abs()
    tir_true = ((target >= 70.0) & (target <= 180.0)).mean()
    tir_pred = ((pred >= 70.0) & (pred <= 180.0)).mean()
    if lo_col in df.columns and hi_col in df.columns:
        lo = pd.to_numeric(df.loc[valid, lo_col], errors="coerce")
        hi = pd.to_numeric(df.loc[valid, hi_col], errors="coerce")
        coverage = ((target >= lo) & (target <= hi)).mean()
    else:
        coverage = np.nan
    return {
        "n": int(len(pred)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "tir_true": float(tir_true),
        "tir_predicted": float(tir_pred),
        "tir_gap": float(tir_pred - tir_true),
        "coverage": float(coverage) if pd.notna(coverage) else np.nan,
        "coverage80": float(coverage) if pd.notna(coverage) else np.nan,
        "p90_abs_error": float(abs_err.quantile(0.90)),
        "p95_abs_error": float(abs_err.quantile(0.95)),
        "p99_abs_error": float(abs_err.quantile(0.99)),
    }


def summarize_rows(rows: Sequence[Mapping[str, object]], metric_cols: Sequence[str], group_cols: Sequence[str] = ()) -> List[dict]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    out: List[dict] = []
    groups = df.groupby(list(group_cols), dropna=False) if group_cols else [((), df)]
    for key, g in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key_tuple)) if group_cols else {}
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            rec = dict(base)
            rec["metric"] = col
            rec["n_participants"] = int(vals.size)
            if vals.empty:
                rec.update({"mean": np.nan, "median": np.nan, "std": np.nan, "sem": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "iqr_low": np.nan, "iqr_high": np.nan})
            else:
                std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                sem = std / math.sqrt(vals.size) if vals.size else np.nan
                mean = float(vals.mean())
                rec.update({
                    "mean": mean,
                    "median": float(vals.median()),
                    "std": std,
                    "sem": sem,
                    "ci95_low": mean - 1.96 * sem if pd.notna(sem) else np.nan,
                    "ci95_high": mean + 1.96 * sem if pd.notna(sem) else np.nan,
                    "iqr_low": float(vals.quantile(0.25)),
                    "iqr_high": float(vals.quantile(0.75)),
                })
            out.append(rec)
    return out


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if pd.isna(obj):
        return None
    return str(obj)


def save_table(df: pd.DataFrame, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def maybe_copy_report_outputs(output_dir, generated_files: Sequence[str]) -> List[str]:
    copied: List[str] = []
    out = Path(output_dir)
    root = Path.cwd()
    tables_dir = root / "report" / "tables" / "generated"
    figures_dir = root / "report" / "figures" / "generated"
    for file in generated_files:
        src = Path(file)
        if not src.exists():
            continue
        if src.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
            dst_dir = figures_dir
        elif src.suffix.lower() in {".csv", ".json", ".txt"}:
            dst_dir = tables_dir
        else:
            continue
        if dst_dir.exists():
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def write_summary(output_dir, *, tasks: Sequence[str], generated_files: Sequence[str], warnings: Sequence[str], copied_files: Sequence[str], extra: Optional[dict] = None) -> str:
    out = Path(output_dir)
    existing_files = []
    for sub in ["metrics", "figures", "tables", "logs"]:
        d = out / sub
        if d.exists():
            existing_files.extend(str(p) for p in sorted(d.rglob("*")) if p.is_file())
    merged_generated = sorted(set([str(Path(p)) for p in generated_files] + existing_files))
    path = out / "diagnostics_summary.json"
    previous_tasks = []
    previous_warnings = []
    previous_copied = []
    if path.exists():
        try:
            with path.open() as f:
                previous = json.load(f)
            previous_tasks = list(previous.get("tasks", []))
            previous_warnings = list(previous.get("warnings", []))
            previous_copied = list(previous.get("copied_report_files", []))
        except Exception:
            pass
    payload = {
        "tasks": sorted(set(previous_tasks + list(tasks))),
        "generated_files": merged_generated,
        "warnings": sorted(set(previous_warnings + list(warnings))),
        "copied_report_files": sorted(set(previous_copied + list(copied_files))),
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)
    return str(path)


def safe_level(value) -> str:
    if pd.isna(value):
        return "missing"
    return str(value).replace("/", "_").replace(" ", "_")
