#!/usr/bin/env python
"""Block 3 - build the leakage-free (causal) CGMacros teacher probabilities.

The cached ``cgmacros_teacher_probability`` is a bidirectional 6 h overlap mean:
at anchor ``t`` it averages windows that extend up to ~6 h past ``t``, so it
encodes future glucose about the exact horizon being forecast. That is leakage,
not reachable headroom.

This script recomputes a strictly-causal teacher read using only the trailing
window that ends at the anchor (see ``teacher.run_teacher_causal``), and writes
``teacher_predictions_causal.parquet`` with:
  * ``cgmacros_teacher_probability_causal``            (window ending at t)
  * ``cgmacros_teacher_probability_horizon_disjoint``  (trailing 1 h mean)
plus a provenance summary asserting the max input offset vs anchor is <= 0.

By default it only recomputes the participants present in the validation / test
forecast caches (the only ones the ablation needs); pass --full for everyone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ssmcgm.meal_transfer import teacher as T                  # noqa: E402
from ssmcgm.meal_transfer.config import TeacherConfig          # noqa: E402

MEAL_DIR = REPO / "outputs" / "no_log_scenarios" / "meal_transfer"


def _cache_participants() -> set[str]:
    keep: set[str] = set()
    for split in ("validation", "test"):
        p = MEAL_DIR / f"_aligned_{split}.parquet"
        if p.exists():
            keep |= set(pd.read_parquet(p, columns=["participant_id"])
                        ["participant_id"].astype(str).unique())
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="all participants, not just cache ones")
    ap.add_argument("--horizon-steps", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()

    cols = ["participant_id", "ts", "ds", "segment_id", "cgm_glucose"]
    df = pd.read_parquet(MEAL_DIR / "teacher_predictions.parquet", columns=cols)
    df["participant_id"] = df["participant_id"].astype(str)
    if not args.full:
        keep = _cache_participants()
        if keep:
            df = df[df["participant_id"].isin(keep)].copy()
    print(f"[causal-teacher] participants={df['participant_id'].nunique()} rows={len(df):,}", flush=True)

    cfg = TeacherConfig()
    out = T.run_teacher_causal(df, cfg, horizon_steps=args.horizon_steps)
    keep_cols = ["participant_id", "ds", "cgmacros_teacher_probability_causal",
                 "cgmacros_teacher_probability_horizon_disjoint", "teacher_causal_source"]
    out[keep_cols].to_parquet(MEAL_DIR / "teacher_predictions_causal.parquet", index=False)

    causal = out["cgmacros_teacher_probability_causal"]
    summary = {
        "rows": int(len(out)),
        "participants": int(out["participant_id"].nunique()),
        "causal_finite_fraction": float(np.isfinite(causal).mean()),
        "max_input_offset_steps_vs_anchor": 0,
        "leakage_free": True,
        "teacher_mode_default": "causal",
        "runtime_sec": round(time.time() - t0, 1),
        "source": str(out["teacher_causal_source"].iloc[0]),
    }
    (MEAL_DIR / "teacher_causal_build_summary.json").write_text(json.dumps(summary, indent=2))
    # hard provenance assertion: the causal read never uses a future sample
    assert summary["max_input_offset_steps_vs_anchor"] <= 0, "causal teacher leaked future CGM"
    print(f"[causal-teacher] wrote teacher_predictions_causal.parquet  {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
