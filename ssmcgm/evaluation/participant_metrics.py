"""Participant-level diagnostic metrics."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from .diagnostics import Q10, Q50, Q90, aggregate_metrics, anonymize_participants, save_table, summarize_rows, write_json

SUBGROUP_COLS = [
    "participants_study_group", "hba1c_percent_baseline", "bmi_baseline",
    "participants_clinical_site", "med_insulin", "med_any_diabetes_drug",
]
PRIVACY_BINNED_CONTINUOUS_COLS = {
    "hba1c_percent_baseline": "hba1c_quartile",
    "bmi_baseline": "bmi_quartile",
}
METRIC_COLS = ["mae", "rmse", "bias", "tir_true", "tir_predicted", "tir_gap", "coverage", "p90_abs_error", "p95_abs_error", "p99_abs_error"]


def participant_level_metrics(predictions: pd.DataFrame, metrics_dir) -> tuple[pd.DataFrame, List[str]]:
    metrics_dir = Path(metrics_dir)
    labels = anonymize_participants(predictions["participant_id"].astype(str))
    rows = []
    group_cols = ["participant_id", "scenario_mode"] if "scenario_mode" in predictions.columns else ["participant_id"]
    for key, group in predictions.groupby(group_cols, dropna=False, observed=False):
        key = key if isinstance(key, tuple) else (key, "forecast_only")
        pid, scenario = str(key[0]), str(key[1])
        rec = {"participant_label": labels[pid], "scenario_mode": scenario}
        rec.update(aggregate_metrics(group))
        for col in SUBGROUP_COLS:
            if col in group.columns:
                vals = group[col].dropna().unique()
                rec[col] = vals[0] if len(vals) else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    for source_col, binned_col in PRIVACY_BINNED_CONTINUOUS_COLS.items():
        if source_col in out.columns:
            vals = pd.to_numeric(out[source_col], errors="coerce")
            try:
                out[binned_col] = pd.qcut(vals, q=4, duplicates="drop").astype(str)
            except ValueError:
                out[binned_col] = "nan"
            out = out.drop(columns=[source_col])

    metrics_path = metrics_dir / "participant_level_metrics.csv"
    save_table(out, metrics_path)
    summary_rows = summarize_rows(out, METRIC_COLS, group_cols=["scenario_mode"] if "scenario_mode" in out.columns else [])
    summary = {
        "scope": "participant-level metrics; participant IDs anonymized; continuous subgroup fields binned; rows grouped by participant_label and scenario_mode",
        "metrics": summary_rows,
    }
    summary_path = metrics_dir / "participant_level_summary.json"
    write_json(summary_path, summary)
    return out, [str(metrics_path), str(summary_path)]
