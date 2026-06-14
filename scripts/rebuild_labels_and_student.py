#!/usr/bin/env python
"""Block 4 - rebuild pseudo-labels and the causal student with the insulin
quarantine and the soft onset-distance rescue, reusing the cached teacher.

This regenerates only the CPU phases (labels -> student). It does NOT re-run the
GPU teacher; it reuses ``teacher_predictions.parquet`` and the causal features
already cached in ``causal_student_predictions.parquet``. Originals are backed up
to ``*.prequarantine.parquet`` before being overwritten.

Changes versus the previous student:
  * insulin users are quarantined from label training (label_train_eligible=0),
  * the previously discarded uncertain rows are kept with a soft / ordinal
    onset-distance target, and the student is fit as a regressor on that target,
  * ``student_meal_probability`` is rewritten with the new model.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ssmcgm.meal_transfer import pseudo_labels as PL          # noqa: E402
from ssmcgm.meal_transfer import student as ST                # noqa: E402
from ssmcgm.meal_transfer.config import PseudoLabelConfig, StudentConfig  # noqa: E402
from ssmcgm.meal_transfer.features import CAUSAL_FEATURE_COLUMNS  # noqa: E402

MEAL_DIR = REPO / "outputs" / "no_log_scenarios" / "meal_transfer"


def _backup(path: Path) -> None:
    bak = path.with_suffix(".prequarantine.parquet")
    if path.exists() and not bak.exists():
        bak.write_bytes(path.read_bytes())


def main() -> int:
    t0 = time.time()
    plcfg = PseudoLabelConfig()
    stcfg = StudentConfig()

    student = pd.read_parquet(MEAL_DIR / "causal_student_predictions.parquet")
    student["participant_id"] = student["participant_id"].astype(str)
    pseudo = pd.read_parquet(MEAL_DIR / "meal_pseudo_labels.parquet")
    pseudo["participant_id"] = pseudo["participant_id"].astype(str)

    key = ["participant_id", "ds"]
    pcols = key + ["meal_pseudo_label", "pseudo_label_confidence", "future_peak_rise_120"]
    df = student.merge(pseudo[pcols], on=key, how="left")
    if "segment_id" not in df.columns:
        df = df.merge(pseudo[key + ["segment_id"]], on=key, how="left")
    df = df.sort_values(["participant_id", "ds"]).reset_index(drop=True)

    # Block 4 soft onset-distance labels + insulin quarantine flag
    df = PL.add_soft_onset_labels(df, plcfg)
    n_uncertain = int((df["meal_pseudo_label"] == "uncertain").sum())
    n_quarantine = int(df["insulin_quarantine"].sum())
    n_rescued = int(((df["meal_pseudo_label"] == "uncertain") &
                     (df["meal_soft_target"] > 0) & (df["label_train_eligible"] == 1)).sum())

    # train soft causal student on non-insulin eligible rows of the train split
    feats = [c for c in CAUSAL_FEATURE_COLUMNS if c in df.columns]
    train_mask = df["split"] == "train"
    art = ST.train_student_soft(df[train_mask], stcfg, feature_cols=feats)
    df["student_meal_probability_prequarantine"] = df["student_meal_probability"]
    df["student_meal_probability"] = ST.predict_student(df, art)

    # persist
    _backup(MEAL_DIR / "meal_pseudo_labels.parquet")
    _backup(MEAL_DIR / "causal_student_predictions.parquet")

    pl_out = pseudo.merge(
        df[key + ["meal_soft_target", "label_train_eligible", "insulin_quarantine"]],
        on=key, how="left")
    pl_out.to_parquet(MEAL_DIR / "meal_pseudo_labels.parquet", index=False)

    out_cols = [c for c in student.columns]  # keep original schema
    df_out = df[out_cols].copy()
    df_out.to_parquet(MEAL_DIR / "causal_student_predictions.parquet", index=False)

    summary = {
        "uncertain_rows": n_uncertain,
        "uncertain_rows_rescued_with_soft_target": n_rescued,
        "insulin_rows_quarantined_from_training": n_quarantine,
        "train_rows_used": art.train_rows,
        "student_is_regressor": True,
        "val_average_precision": art.val_ap,
        "val_auc": art.val_auc,
        "student_prob_mean_before": float(df["student_meal_probability_prequarantine"].mean()),
        "student_prob_mean_after": float(df["student_meal_probability"].mean()),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (MEAL_DIR / "label_student_rebuild_summary.json").write_text(json.dumps(summary, indent=2))
    print("[rebuild]", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
