"""Quantile alarm diagnostics for hypo/hyperglycemia."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .diagnostics import Q10, Q50, Q90, forecast_only, save_table


def _binary_metrics(true_event: pd.Series, alarm: pd.Series) -> dict:
    true_event = true_event.astype(bool)
    alarm = alarm.astype(bool)
    tp = int((true_event & alarm).sum())
    tn = int((~true_event & ~alarm).sum())
    fp = int((~true_event & alarm).sum())
    fn = int((true_event & ~alarm).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and precision + recall else np.nan
    return {
        "n": int(len(true_event)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "prevalence": float(true_event.mean()) if len(true_event) else np.nan,
        "precision": precision, "ppv": precision, "recall": recall, "sensitivity": recall,
        "specificity": specificity, "f1": f1, "npv": npv,
    }


def event_detection_quantile_alarms(predictions: pd.DataFrame, metrics_dir, figures_dir) -> List[str]:
    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    df = forecast_only(predictions)
    rows = []
    rules = [
        ("hypoglycemia", "median_q50_lt_70", df["target"] < 70.0, df[Q50] < 70.0),
        ("hypoglycemia", "risk_q10_lt_70", df["target"] < 70.0, df[Q10] < 70.0),
        ("hyperglycemia", "median_q50_gt_180", df["target"] > 180.0, df[Q50] > 180.0),
        ("hyperglycemia", "risk_q90_gt_180", df["target"] > 180.0, df[Q90] > 180.0),
    ]
    for event, rule, true_event, alarm in rules:
        rec = {"scope": "overall", "horizon_step": 0, "horizon_minutes": 0, "event": event, "rule": rule}
        rec.update(_binary_metrics(true_event, alarm))
        rows.append(rec)
        for (hstep, hmin), g in df.groupby(["horizon_step", "horizon_minutes"], dropna=False):
            if event == "hypoglycemia":
                te = g["target"] < 70.0
                al = g[Q50] < 70.0 if "median" in rule else g[Q10] < 70.0
            else:
                te = g["target"] > 180.0
                al = g[Q50] > 180.0 if "median" in rule else g[Q90] > 180.0
            rec = {"scope": "horizon", "horizon_step": int(hstep), "horizon_minutes": int(hmin), "event": event, "rule": rule}
            rec.update(_binary_metrics(te, al))
            rows.append(rec)
    out = pd.DataFrame(rows)
    csv_path = metrics_dir / "event_detection_quantile_alarms.csv"
    save_table(out, csv_path)
    fig_path = figures_dir / "hypo_alarm_tradeoff.png"
    try:
        import matplotlib.pyplot as plt
        overall = out[(out["scope"] == "overall") & (out["event"] == "hypoglycemia")].copy()
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(overall))
        ax.bar(x - 0.18, overall["precision"], width=0.36, label="precision")
        ax.bar(x + 0.18, overall["recall"], width=0.36, label="recall")
        ax.set_xticks(x)
        ax.set_xticklabels(overall["rule"], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Hypoglycemia alarm tradeoff")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        return [str(csv_path), str(fig_path)]
    except Exception as exc:
        fail_path = figures_dir / "hypo_alarm_tradeoff_failed.txt"
        fail_path.write_text(str(exc))
        return [str(csv_path), str(fail_path)]
