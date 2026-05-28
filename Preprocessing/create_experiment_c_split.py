"""
create_experiment_c_split.py
=============================
Experiment C — Participant-level split for realistic deployment evaluation.

Split design (asymmetric-pool):
  TEST  participants  → drawn from personalization-eligible pool
                        (longest contiguous segment ≥ context_h + adaptation_h + evaluation_h)
  VAL   participants  → drawn from the same eligible pool (remaining after test),
                        so that model selection measures the same quantity as test evaluation.
  TRAIN participants  → everyone NOT in val/test who has at least one valid forecast window.
                        This includes the 633 "fragmented" participants who failed the
                        contiguous-segment rule but still contribute useful training windows.

Why val shares test eligibility:
  Validating on a population with weaker coverage than test would make the val loss an
  optimistic proxy for test performance. Using the same eligibility rule ensures that
  hyperparameter selection and early stopping optimise for the right objective.

Why train includes fragmented participants:
  Supervised training needs many diverse windows; strict eligibility is only required
  for the evaluation (personalization) protocol, not for the training signal.

Phases (within each test/val participant's chosen longest segment):
  [0,           context_bins)           → context      (model pre-conditions)
  [context_bins, adaptation_bins)       → adaptation   (participant-specific fine-tuning)
  [adaptation_bins, evaluation_bins)    → evaluation   (held-out forecast horizon)

forecast_windows.csv is labelled with one of:
  train | val_context | val_adaptation | val_evaluation |
  test_context | test_adaptation | test_evaluation | unused

Output directory: {output_root}/experiment_c_split_adapt{adaptation_h}h_seed{seed}/
This naming lets you sweep adaptation lengths (6/12/24/48 h) without overwriting.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
BINS_PER_HOUR = 12          # 5-min intervals
GAP_ASSUMPTION_H = 0.5      # assumed inter-segment gap when computing absolute timestamps
STRATUM_DRIFT_TOL_PP = 3.0  # warn if per-stratum test proportion drifts > 3 pp


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Input loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_inputs(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cohort, segments, and forecast_windows. Normalize participant_id to str."""
    print("[Load] Reading input files...")

    cohort  = pd.read_csv(args.cohort_csv)
    segs    = pd.read_csv(args.segments_csv)
    windows = pd.read_csv(args.forecast_windows_csv)

    for df, name in [(cohort, "cohort"), (segs, "segments"), (windows, "forecast_windows")]:
        if "participant_id" not in df.columns:
            raise ValueError(f"'participant_id' column missing from {name}.csv")
        df["participant_id"] = df["participant_id"].astype(str)

    # Detect segment duration column
    dur_col = _detect_duration_col(segs)
    if dur_col != "dur_h":
        segs = segs.rename(columns={dur_col: "dur_h"})
        print(f"  [Segments] Renamed duration column '{dur_col}' → 'dur_h'")

    print(f"  cohort   : {len(cohort):,} rows")
    print(f"  segments : {len(segs):,} rows  (avg {len(segs)/len(cohort):.2f} per participant)")
    print(f"  windows  : {len(windows):,} rows")
    return cohort, segs, windows


def _detect_duration_col(segs: pd.DataFrame) -> str:
    candidates = ["dur_h", "duration_h", "duration_hours", "segment_duration_h"]
    for c in candidates:
        if c in segs.columns:
            return c
    # Try to compute from start/end timestamps
    for start, end in [("segment_start", "segment_end"), ("start_time", "end_time"), ("start", "end")]:
        if start in segs.columns and end in segs.columns:
            segs["dur_h"] = (
                pd.to_datetime(segs[end]) - pd.to_datetime(segs[start])
            ).dt.total_seconds() / 3600
            return "dur_h"
    raise ValueError(
        f"Cannot determine segment duration. Columns present: {list(segs.columns)}"
    )


def detect_stratify_col(cohort: pd.DataFrame, requested: str) -> str:
    """Return the first available stratification column from a priority list."""
    candidates = [requested, "stratum", "study_group", "participants_study_group"]
    for c in candidates:
        if c in cohort.columns:
            print(f"  [Stratify] Using column: '{c}'")
            return c
    raise ValueError(
        f"No stratification column found. Tried: {candidates}. "
        f"Available: {list(cohort.columns)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Eligibility
# ═══════════════════════════════════════════════════════════════════════════════

def compute_eligibility(
    cohort: pd.DataFrame,
    segs: pd.DataFrame,
    min_test_segment_h: float,
) -> Tuple[set, set]:
    """
    Returns:
      window_eligible  : set of participant_ids with at least one valid forecast window
      testval_eligible : set of participant_ids with at least one segment ≥ min_test_segment_h
    """
    # Window-eligible: n_valid_windows > 0 (or derived from segments/windows)
    if "n_valid_windows" in cohort.columns:
        window_elig = set(cohort.loc[cohort["n_valid_windows"] > 0, "participant_id"])
    else:
        # Fallback: any participant that appears in forecast_windows
        window_elig = set(cohort["participant_id"])
        warnings.warn("'n_valid_windows' not in cohort.csv — assuming all retained are window-eligible")

    # Test/val-eligible: longest contiguous segment ≥ min_test_segment_h
    if "personalization_eligible" in cohort.columns and "longest_seg_h" in cohort.columns:
        # Use the pre-computed flag if available and consistent with our threshold
        # Recompute from longest_seg_h to support different adaptation lengths
        testval_elig = set(
            cohort.loc[cohort["longest_seg_h"] >= min_test_segment_h, "participant_id"]
        )
    else:
        # Compute from segments
        pid_max_seg = segs.groupby("participant_id")["dur_h"].max()
        testval_elig = set(pid_max_seg[pid_max_seg >= min_test_segment_h].index.astype(str))

    return window_elig, testval_elig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Pre-split diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def print_diagnostics(
    cohort: pd.DataFrame,
    window_elig: set,
    testval_elig: set,
    stratify_col: str,
    min_h: float,
) -> None:
    n_total   = len(cohort)
    n_window  = len(window_elig)
    n_testval = len(testval_elig)

    print(f"\n{'='*60}")
    print(f"Pre-split diagnostics  (min segment ≥ {min_h:.0f} h)")
    print(f"{'='*60}")
    print(f"  Total retained participants    : {n_total:>6,}")
    print(f"  Window-eligible (≥1 window)   : {n_window:>6,}")
    print(f"  Test/val-eligible (≥{min_h:.0f}h seg)  : {n_testval:>6,}")
    print(f"  Fragmented-only (window but   : {n_window - n_testval:>6,}  ← go to train pool only")
    print(f"   no contiguous ≥{min_h:.0f}h)")

    print(f"\n  Per-stratum breakdown:")
    header = f"  {'Stratum':<30} {'Total':>6} {'Window':>8} {'Test/val-elig':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    strata = cohort[stratify_col].value_counts()
    any_warn = False
    for stratum, n_stratum in strata.items():
        sub = cohort[cohort[stratify_col] == stratum]
        n_w = len(sub[sub["participant_id"].isin(window_elig)])
        n_e = len(sub[sub["participant_id"].isin(testval_elig)])
        flag = " [WARN: <10]" if n_e < 10 else ""
        if n_e < 10:
            any_warn = True
        print(f"  {str(stratum):<30} {n_stratum:>6,} {n_w:>8,} {n_e:>14,}{flag}")

    if any_warn:
        warnings.warn(
            "One or more strata have fewer than 10 test/val-eligible participants. "
            "Stratified sampling may fall back to random for those strata."
        )
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Stratified sampling helper
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_sample(
    eligible_df: pd.DataFrame,       # participants to sample from
    n_target: int,                    # total participants to draw
    stratify_col: str,
    base_df: pd.DataFrame,            # all retained (used for proportion computation)
    rng: np.random.Generator,
    label: str,                       # "test" or "val" (for logging)
    exclude_ids: Optional[set] = None,
) -> Tuple[List[str], bool]:
    """
    Proportional stratified sample of n_target from eligible_df.
    Proportions are computed from base_df (all retained) for representativeness.
    Falls back to random if stratification fails for any stratum.
    Returns (sampled_ids, fallback_used).
    """
    if exclude_ids:
        eligible_df = eligible_df[~eligible_df["participant_id"].isin(exclude_ids)].copy()

    if len(eligible_df) < n_target:
        warnings.warn(
            f"[{label}] Only {len(eligible_df)} eligible participants but target is {n_target}. "
            "Taking all eligible."
        )
        return eligible_df["participant_id"].tolist(), True

    # Proportions from the full retained cohort
    strata_props = base_df[stratify_col].value_counts(normalize=True)
    sampled      = []
    fallback_used = False

    for stratum, prop in strata_props.items():
        n_from_stratum = max(1, round(n_target * float(prop)))
        pool = eligible_df.loc[
            eligible_df[stratify_col] == stratum, "participant_id"
        ].tolist()

        if len(pool) == 0:
            warnings.warn(f"[{label}] Stratum '{stratum}' has no eligible participants.")
            fallback_used = True
            continue

        if len(pool) < n_from_stratum:
            warnings.warn(
                f"[{label}] Stratum '{stratum}': only {len(pool)} eligible, "
                f"needed {n_from_stratum}. Taking all."
            )
            sampled.extend(pool)
            fallback_used = True
        else:
            chosen = rng.choice(pool, size=n_from_stratum, replace=False).tolist()
            sampled.extend(chosen)

    # Trim or top-up to exactly n_target
    if len(sampled) > n_target:
        sampled = rng.choice(sampled, size=n_target, replace=False).tolist()
    elif len(sampled) < n_target:
        remaining = eligible_df[
            ~eligible_df["participant_id"].isin(set(sampled))
        ]["participant_id"].tolist()
        if remaining:
            extra_n = min(n_target - len(sampled), len(remaining))
            sampled.extend(rng.choice(remaining, size=extra_n, replace=False).tolist())
            fallback_used = True

    return sampled, fallback_used


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Participant split
# ═══════════════════════════════════════════════════════════════════════════════

def split_participants(
    cohort: pd.DataFrame,
    window_elig: set,
    testval_elig: set,
    stratify_col: str,
    args,
) -> Tuple[pd.DataFrame, dict]:
    """
    Assign each participant to train / val / test.
    Returns (split_df, fallback_log).
    """
    n_retained  = len(cohort)
    eligible_df = cohort[cohort["participant_id"].isin(testval_elig)].copy()
    n_eligible  = len(eligible_df)

    # Test target = fraction of the ELIGIBLE pool (those who can do the personalization
    # protocol).  This gives test ≈ 144 at 48 h adaptation with 958 eligible participants.
    # Val target  = fraction of ALL retained (to ensure enough validation signal), drawn
    # from the remaining eligible pool.  This gives val ≈ 239.
    # Train = everyone else with at least one window (1591 − 144 − 239 ≈ 1208).
    test_target = round(args.test_frac * n_eligible)
    val_target  = round(args.val_frac  * n_retained)

    # Safety caps
    test_target = min(test_target, n_eligible)
    val_target  = min(val_target,  n_eligible - test_target)

    print(f"[Split] Targets — test: {test_target}, val: {val_target}")

    rng = np.random.default_rng(args.random_seed)
    fallback_log: dict = {}

    # Sample test
    test_ids, fb_test = stratified_sample(
        eligible_df, test_target, stratify_col, cohort, rng, "test"
    )
    if fb_test:
        fallback_log["test"] = "stratification_fallback_used"

    # Sample val from eligible − test
    val_ids, fb_val = stratified_sample(
        eligible_df, val_target, stratify_col, cohort, rng,
        "val", exclude_ids=set(test_ids)
    )
    if fb_val:
        fallback_log["val"] = "stratification_fallback_used"

    test_set = set(test_ids)
    val_set  = set(val_ids)

    # Train = window-eligible, not in test or val
    train_ids = [
        pid for pid in cohort["participant_id"]
        if pid in window_elig and pid not in test_set and pid not in val_set
    ]

    rows = []
    pid_to_stratum = cohort.set_index("participant_id")[stratify_col].to_dict()
    for pid in cohort["participant_id"]:
        if pid in test_set:
            split = "test"
        elif pid in val_set:
            split = "val"
        elif pid in set(train_ids):
            split = "train"
        else:
            split = "excluded"   # not window-eligible, not test/val
        rows.append({"participant_id": pid, "split": split,
                     "stratum": pid_to_stratum.get(pid, "unknown")})

    split_df = pd.DataFrame(rows)
    print(f"  train={split_df['split'].eq('train').sum()}  "
          f"val={split_df['split'].eq('val').sum()}  "
          f"test={split_df['split'].eq('test').sum()}  "
          f"excluded={split_df['split'].eq('excluded').sum()}")
    return split_df, fallback_log


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Per-participant timeline split (test and val)
# ═══════════════════════════════════════════════════════════════════════════════

def _choose_longest_segment(pid: str, segs_sub: pd.DataFrame, min_h: float) -> Optional[pd.Series]:
    """Return the row of the longest segment ≥ min_h, or None if none qualifies."""
    eligible = segs_sub[segs_sub["dur_h"] >= min_h]
    if eligible.empty:
        return None
    return eligible.loc[eligible["dur_h"].idxmax()]


def _infer_segment_start(
    pid: str,
    seg_id: int,
    segs_sub: pd.DataFrame,
    cohort_row: pd.Series,
) -> Optional[pd.Timestamp]:
    """
    Approximate the absolute start of a segment using cohort.valid_start
    and cumulative previous segment durations.
    For seg_id=0: exact (= valid_start).
    For seg_id>0: approximation assuming GAP_ASSUMPTION_H between segments.
    """
    try:
        valid_start = pd.to_datetime(cohort_row["valid_start"], utc=True)
    except Exception:
        return None

    if seg_id == 0:
        return valid_start

    # Accumulate all preceding segments (sorted by segment_id)
    prev = segs_sub[segs_sub["segment_id"] < seg_id].sort_values("segment_id")
    offset_h = prev["dur_h"].sum() + GAP_ASSUMPTION_H * len(prev)
    return valid_start + pd.Timedelta(hours=offset_h)


def compute_personalization_windows(
    split_df: pd.DataFrame,
    cohort: pd.DataFrame,
    segs: pd.DataFrame,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each test and val participant, identify their chosen segment and
    compute context / adaptation / evaluation phase boundaries.

    Returns (pers_test_df, pers_val_df).
    """
    context_bins    = int(args.context_h    * BINS_PER_HOUR)
    adaptation_bins = int(args.adaptation_h * BINS_PER_HOUR)
    evaluation_bins = int(args.evaluation_h * BINS_PER_HOUR)
    min_h = args.context_h + args.adaptation_h + args.evaluation_h

    cohort_idx = cohort.set_index("participant_id")
    segs_by_pid = segs.groupby("participant_id")

    records = []
    for split in ("test", "val"):
        pids = split_df.loc[split_df["split"] == split, "participant_id"].tolist()
        for pid in pids:
            pid_segs = segs_by_pid.get_group(pid) if pid in segs_by_pid.groups else pd.DataFrame()
            seg_row  = _choose_longest_segment(pid, pid_segs, min_h)
            if seg_row is None:
                warnings.warn(f"[{split}] Participant {pid} has no segment ≥ {min_h}h — skipping")
                continue

            seg_id   = int(seg_row["segment_id"])
            seg_dur_h = float(seg_row["dur_h"])
            cohort_row = cohort_idx.loc[pid] if pid in cohort_idx.index else None

            # Absolute start (approximate for seg_id > 0)
            seg_start_abs = _infer_segment_start(pid, seg_id, pid_segs, cohort_row) \
                            if cohort_row is not None else None

            # Phase boundaries (bin offsets from segment start)
            c_start  = 0
            c_end    = context_bins
            a_start  = context_bins
            a_end    = context_bins + adaptation_bins
            e_start  = context_bins + adaptation_bins
            e_end    = context_bins + adaptation_bins + evaluation_bins

            # Absolute timestamps (if we have a valid segment start)
            def _abs(bin_offset):
                if seg_start_abs is None:
                    return pd.NaT
                return seg_start_abs + pd.Timedelta(minutes=bin_offset * 5)

            records.append({
                "split":                   split,
                "participant_id":          pid,
                "chosen_segment_id":       seg_id,
                "segment_dur_h":           seg_dur_h,
                "abs_start_inferred":      seg_id > 0,   # flag: exact only for seg_id=0
                # Relative offsets (bins)
                "context_start_bin":       c_start,
                "context_end_bin":         c_end,
                "adaptation_start_bin":    a_start,
                "adaptation_end_bin":      a_end,
                "evaluation_start_bin":    e_start,
                "evaluation_end_bin":      e_end,
                # Relative offsets (hours)
                "context_start_h":         0.0,
                "context_end_h":           args.context_h,
                "adaptation_start_h":      args.context_h,
                "adaptation_end_h":        args.context_h + args.adaptation_h,
                "evaluation_start_h":      args.context_h + args.adaptation_h,
                "evaluation_end_h":        args.context_h + args.adaptation_h + args.evaluation_h,
                # Absolute timestamps
                "segment_start_abs":       seg_start_abs,
                "context_start_abs":       _abs(c_start),
                "context_end_abs":         _abs(c_end),
                "adaptation_start_abs":    _abs(a_start),
                "adaptation_end_abs":      _abs(a_end),
                "evaluation_start_abs":    _abs(e_start),
                "evaluation_end_abs":      _abs(e_end),
            })

    all_pers = pd.DataFrame(records)
    if all_pers.empty:
        return pd.DataFrame(), pd.DataFrame()

    pers_test = all_pers[all_pers["split"] == "test"].drop(columns="split").reset_index(drop=True)
    pers_val  = all_pers[all_pers["split"] == "val" ].drop(columns="split").reset_index(drop=True)
    return pers_test, pers_val


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Forecast-window labeling
# ═══════════════════════════════════════════════════════════════════════════════

def label_forecast_windows(
    windows: pd.DataFrame,
    split_df: pd.DataFrame,
    pers_test: pd.DataFrame,
    pers_val: pd.DataFrame,
    args,
) -> pd.DataFrame:
    """
    Add 'split' and 'unused_reason' columns to forecast_windows.
    Labels: train | val_context | val_adaptation | val_evaluation |
            test_context | test_adaptation | test_evaluation | unused
    """
    context_bins    = int(args.context_h    * BINS_PER_HOUR)
    adaptation_bins = int(args.adaptation_h * BINS_PER_HOUR)
    evaluation_bins = int(args.evaluation_h * BINS_PER_HOUR)

    c_end = context_bins
    a_end = context_bins + adaptation_bins
    e_end = context_bins + adaptation_bins + evaluation_bins

    # Build lookup: pid → split
    pid_split = split_df.set_index("participant_id")["split"].to_dict()

    # Build lookup: (pid, segment_id) → phase boundaries for test/val
    def _build_chosen_seg_lookup(pers_df, prefix):
        """Returns {pid: chosen_segment_id} dict."""
        if pers_df.empty:
            return {}
        return pers_df.set_index("participant_id")["chosen_segment_id"].to_dict()

    test_chosen = _build_chosen_seg_lookup(pers_test, "test")
    val_chosen  = _build_chosen_seg_lookup(pers_val,  "val")

    def _classify_window(row):
        pid    = row["participant_id"]
        seg_id = row["segment_id"]
        w_start = row["window_start"]
        split  = pid_split.get(pid, "excluded")

        if split == "excluded":
            return "unused", "participant_not_in_split"

        if split == "train":
            return "train", ""

        # val or test participant
        chosen_seg = (test_chosen if split == "test" else val_chosen).get(pid)
        prefix     = "test" if split == "test" else "val"

        if chosen_seg is None or seg_id != chosen_seg:
            return "unused", "outside_chosen_segment"

        if w_start < c_end:
            return f"{prefix}_context",    ""
        if w_start < a_end:
            return f"{prefix}_adaptation", ""
        if w_start < e_end:
            return f"{prefix}_evaluation", ""
        return "unused", "after_evaluation_phase"

    print("[Label] Classifying forecast windows...")
    labels = windows.apply(_classify_window, axis=1, result_type="expand")
    labels.columns = ["split", "unused_reason"]

    windows_out = windows.copy()
    windows_out["split"]         = labels["split"]
    windows_out["unused_reason"] = labels["unused_reason"]

    # Summary
    print(f"  Window label distribution:")
    for lbl, cnt in windows_out["split"].value_counts().items():
        print(f"    {lbl:<30} {cnt:>8,}")

    return windows_out


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Sanity checks
# ═══════════════════════════════════════════════════════════════════════════════

def run_sanity_checks(
    split_df: pd.DataFrame,
    pers_test: pd.DataFrame,
    pers_val: pd.DataFrame,
    windows_labeled: pd.DataFrame,
    cohort: pd.DataFrame,
    window_elig: set,
) -> None:
    print("\n[Sanity] Running pre-save assertions...")
    errors = []

    # 1. Every test participant has exactly one personalization-window entry
    test_pids = split_df.loc[split_df["split"] == "test", "participant_id"].tolist()
    if not pers_test.empty:
        counts = pers_test["participant_id"].value_counts()
        multi = counts[counts > 1]
        if not multi.empty:
            errors.append(f"Test participants with >1 pers-window entry: {multi.index.tolist()}")
        missing = set(test_pids) - set(pers_test["participant_id"])
        if missing:
            errors.append(f"{len(missing)} test participants have NO pers-window entry (likely no eligible segment)")

    # 2. Every val participant has exactly one personalization-window entry
    val_pids = split_df.loc[split_df["split"] == "val", "participant_id"].tolist()
    if not pers_val.empty:
        counts = pers_val["participant_id"].value_counts()
        multi = counts[counts > 1]
        if not multi.empty:
            errors.append(f"Val participants with >1 pers-window entry: {multi.index.tolist()}")

    # 3. No participant in more than one split
    assigned = split_df[split_df["split"].isin(["train", "val", "test"])]
    dup = assigned["participant_id"][assigned["participant_id"].duplicated()]
    if not dup.empty:
        errors.append(f"Participants appear in >1 split: {dup.tolist()}")

    # 4. Train pool = (retained − val − test) ∩ window_eligible
    non_tv = set(cohort["participant_id"]) - set(val_pids) - set(test_pids)
    expected_train = non_tv & window_elig
    actual_train   = set(split_df.loc[split_df["split"] == "train", "participant_id"])
    if expected_train != actual_train:
        extra   = actual_train - expected_train
        missing = expected_train - actual_train
        errors.append(
            f"Train pool mismatch: {len(extra)} extra, {len(missing)} missing participants"
        )

    # 5. Stratum proportions within ±3 pp (warn, don't fail)
    if "stratum" in split_df.columns:
        cohort_props = cohort["stratum"].value_counts(normalize=True)
        test_df      = split_df[split_df["split"] == "test"]
        if len(test_df) > 0:
            test_props = test_df["stratum"].value_counts(normalize=True)
            for stratum, base_prop in cohort_props.items():
                actual_prop = test_props.get(stratum, 0.0)
                drift_pp    = abs(actual_prop - base_prop) * 100
                if drift_pp > STRATUM_DRIFT_TOL_PP:
                    warnings.warn(
                        f"Stratum '{stratum}': test proportion {actual_prop*100:.1f}% vs "
                        f"cohort {base_prop*100:.1f}% (drift {drift_pp:.1f} pp > {STRATUM_DRIFT_TOL_PP} pp tolerance)"
                    )

    if errors:
        for e in errors:
            print(f"  [ASSERT FAILED] {e}")
        raise AssertionError(f"Sanity checks failed ({len(errors)} errors). See above.")

    print("  All assertions passed.")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Summary report
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(
    cohort: pd.DataFrame,
    split_df: pd.DataFrame,
    pers_test: pd.DataFrame,
    pers_val: pd.DataFrame,
    window_elig: set,
    testval_elig: set,
    stratify_col: str,
    args,
) -> None:
    n_train = split_df["split"].eq("train").sum()
    n_val   = split_df["split"].eq("val").sum()
    n_test  = split_df["split"].eq("test").sum()
    n_excl  = split_df["split"].eq("excluded").sum()

    # Train pool: eligible vs fragmented
    train_pids = set(split_df.loc[split_df["split"] == "train", "participant_id"])
    n_train_elig = len(train_pids & testval_elig)
    n_train_frag = len(train_pids - testval_elig)

    print(f"\n{'='*60}")
    print(f"Experiment C Split Summary")
    print(f"  adaptation_h={args.adaptation_h}h  seed={args.random_seed}")
    print(f"{'='*60}")
    print(f"  Retained participants           : {len(cohort):>6,}")
    print(f"  Window-eligible                 : {len(window_elig):>6,}")
    print(f"  Test/val-eligible (≥{args.context_h+args.adaptation_h+args.evaluation_h:.0f}h) : {len(testval_elig):>6,}")
    print(f"")
    print(f"  Train                           : {n_train:>6,}")
    print(f"    of which test/val-eligible    : {n_train_elig:>6,}")
    print(f"    of which fragmented only      : {n_train_frag:>6,}  ← asymmetric-pool contrib.")
    print(f"  Val                             : {n_val:>6,}")
    print(f"  Test                            : {n_test:>6,}")
    print(f"  Excluded (no valid windows)     : {n_excl:>6,}")
    print(f"")
    print(f"  Personalization windows computed:")
    print(f"    Test participants             : {len(pers_test):>6,}")
    print(f"    Val  participants             : {len(pers_val):>6,}")
    print(f"")
    print(f"  Per-stratum distribution (test):")
    if n_test > 0:
        test_df = split_df[split_df["split"] == "test"]
        base_props = cohort[stratify_col].value_counts(normalize=True)
        for stratum in cohort[stratify_col].value_counts().index:
            n_s = (test_df["stratum"] == stratum).sum()
            prop = n_s / n_test * 100
            base = base_props.get(stratum, 0) * 100
            drift = prop - base
            print(f"    {str(stratum):<35} {n_s:>4}  ({prop:5.1f}%  base {base:5.1f}%  Δ{drift:+.1f}pp)")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Save outputs
# ═══════════════════════════════════════════════════════════════════════════════

def build_metadata(
    cohort: pd.DataFrame,
    split_df: pd.DataFrame,
    pers_test: pd.DataFrame,
    pers_val: pd.DataFrame,
    window_elig: set,
    testval_elig: set,
    stratify_col: str,
    fallback_log: dict,
    out_dir: Path,
    args,
) -> dict:
    n_train = int(split_df["split"].eq("train").sum())
    n_val   = int(split_df["split"].eq("val").sum())
    n_test  = int(split_df["split"].eq("test").sum())
    train_pids  = set(split_df.loc[split_df["split"] == "train", "participant_id"])
    n_train_elig = int(len(train_pids & testval_elig))
    n_train_frag = int(len(train_pids - testval_elig))

    # Per-stratum distribution per split
    strat_dist = {}
    for split in ("train", "val", "test"):
        sub = split_df[split_df["split"] == split]
        strat_dist[split] = sub["stratum"].value_counts().to_dict()

    # Git hash (best-effort)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    return {
        "experiment":             "C",
        "created_at":             datetime.datetime.now().isoformat(),
        "git_hash":               git_hash,
        "parameters": {
            "context_h":          args.context_h,
            "adaptation_h":       args.adaptation_h,
            "evaluation_h":       args.evaluation_h,
            "min_test_segment_h": args.context_h + args.adaptation_h + args.evaluation_h,
            "val_frac":           args.val_frac,
            "test_frac":          args.test_frac,
            "random_seed":        args.random_seed,
            "stratify_col":       stratify_col,
        },
        "counts": {
            "n_retained":          int(len(cohort)),
            "n_window_eligible":   int(len(window_elig)),
            "n_testval_eligible":  int(len(testval_elig)),
            "n_train":             n_train,
            "n_val":               n_val,
            "n_test":              n_test,
            "n_excluded":          int(split_df["split"].eq("excluded").sum()),
        },
        "train_pool_composition": {
            "n_testval_eligible_in_train": n_train_elig,
            "n_fragmented_only_in_train":  n_train_frag,
            "note": "asymmetric-pool design: train uses all window-eligible participants",
        },
        "stratum_distribution":   strat_dist,
        "personalization_phases": {
            "rule": "longest-eligible-segment",
            "abs_timestamp_note": (
                "segment_start_abs is exact for segment_id=0 (= cohort.valid_start). "
                f"For segment_id>0 it is approximate: "
                f"cumulative prior segment durations + {GAP_ASSUMPTION_H}h assumed gap."
            ),
        },
        "stratification": {
            "fallback_log":       fallback_log,
            "any_fallback":       bool(fallback_log),
        },
        "input_files": {
            "cohort_csv":          str(args.cohort_csv),
            "segments_csv":        str(args.segments_csv),
            "forecast_windows_csv": str(args.forecast_windows_csv),
        },
        "output_dir":             str(out_dir),
    }


def save_outputs(
    out_dir: Path,
    split_df: pd.DataFrame,
    pers_test: pd.DataFrame,
    pers_val: pd.DataFrame,
    windows_labeled: pd.DataFrame,
    metadata: dict,
    cohort: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Save] Writing outputs to {out_dir}")

    # split_participants.csv
    split_df.to_csv(out_dir / "split_participants.csv", index=False)

    # Per-split participant lists
    for split in ("train", "val", "test"):
        sub = split_df[split_df["split"] == split][["participant_id", "stratum"]]
        sub.to_csv(out_dir / f"{split}_participants.csv", index=False)

    # Personalization windows
    if not pers_test.empty:
        pers_test.to_csv(out_dir / "test_personalization_windows.csv", index=False)
    if not pers_val.empty:
        pers_val.to_csv(out_dir / "val_personalization_windows.csv",  index=False)

    # Labeled forecast windows
    windows_labeled.to_csv(out_dir / "forecast_windows_with_split.csv", index=False)

    # Split summary (per-stratum counts per split)
    rows = []
    for split in ("train", "val", "test"):
        for stratum, n in metadata["stratum_distribution"].get(split, {}).items():
            rows.append({"split": split, "stratum": stratum, "n": n})
    pd.DataFrame(rows).to_csv(out_dir / "split_summary.csv", index=False)

    # Metadata JSON
    with open(out_dir / "experiment_c_split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("  Saved:")
    for fname in [
        "split_participants.csv",
        "train_participants.csv", "val_participants.csv", "test_participants.csv",
        "test_personalization_windows.csv", "val_personalization_windows.csv",
        "forecast_windows_with_split.csv",
        "split_summary.csv",
        "experiment_c_split_metadata.json",
    ]:
        p = out_dir / fname
        size = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "MISSING"
        print(f"    {fname:<45} {size}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    DATA_ROOT = Path("/home/myriamcharfeddine/CGM/Data/enriched_multimodal")
    p = argparse.ArgumentParser(
        description="Create Experiment C participant-level split for SSM-CGM thesis"
    )
    p.add_argument("--context-h",     type=float, default=48.0,
                   help="Pre-adaptation context window in hours (default: 48)")
    p.add_argument("--adaptation-h",  type=float, default=48.0,
                   help="Adaptation window in hours (default: 48)")
    p.add_argument("--evaluation-h",  type=float, default=12.0,
                   help="Evaluation window in hours (default: 12)")
    p.add_argument("--val-frac",      type=float, default=0.15,
                   help="Fraction of all retained participants allocated to val (default: 0.15)")
    p.add_argument("--test-frac",     type=float, default=0.15,
                   help="Fraction of all retained participants allocated to test (default: 0.15)")
    p.add_argument("--random-seed",   type=int,   default=42)
    p.add_argument("--stratify-col",  type=str,   default="participants_study_group",
                   help="Column to stratify on (fallback: stratum, study_group)")
    p.add_argument("--output-root",   type=Path,  default=Path("/home/myriamcharfeddine/CGM/Data"))
    p.add_argument("--cohort-csv",    type=Path,  default=DATA_ROOT / "cohort.csv")
    p.add_argument("--segments-csv",  type=Path,  default=DATA_ROOT / "segments.csv")
    p.add_argument("--forecast-windows-csv", type=Path,
                   default=DATA_ROOT / "forecast_windows.csv")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Validate inputs
    for p, label in [
        (args.cohort_csv,          "cohort_csv"),
        (args.segments_csv,        "segments_csv"),
        (args.forecast_windows_csv,"forecast_windows_csv"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"[ERROR] {label} not found: {p}")

    min_h = args.context_h + args.adaptation_h + args.evaluation_h
    out_dir = (
        args.output_root /
        f"experiment_c_split_adapt{int(args.adaptation_h)}h_seed{args.random_seed}"
    )

    print(f"\nExperiment C Split Generator")
    print(f"  context={args.context_h}h  adaptation={args.adaptation_h}h  "
          f"evaluation={args.evaluation_h}h  → min_segment={min_h}h")
    print(f"  val_frac={args.val_frac}  test_frac={args.test_frac}  seed={args.random_seed}")
    print(f"  Output: {out_dir}\n")

    # ── 1. Load ────────────────────────────────────────────────────────────────
    cohort, segs, windows = load_inputs(args)
    stratify_col = detect_stratify_col(cohort, args.stratify_col)

    # ── 2. Eligibility ─────────────────────────────────────────────────────────
    window_elig, testval_elig = compute_eligibility(cohort, segs, min_h)

    # ── 3. Diagnostics ─────────────────────────────────────────────────────────
    print_diagnostics(cohort, window_elig, testval_elig, stratify_col, min_h)

    # ── 4. Split ───────────────────────────────────────────────────────────────
    split_df, fallback_log = split_participants(
        cohort, window_elig, testval_elig, stratify_col, args
    )

    # ── 5. Personalization windows ─────────────────────────────────────────────
    print("\n[Personalization] Computing phase boundaries for test and val participants...")
    pers_test, pers_val = compute_personalization_windows(split_df, cohort, segs, args)
    print(f"  Test pers-windows: {len(pers_test)}  Val pers-windows: {len(pers_val)}")

    # ── 6. Label forecast windows ──────────────────────────────────────────────
    windows_labeled = label_forecast_windows(windows, split_df, pers_test, pers_val, args)

    # ── 7. Sanity checks ───────────────────────────────────────────────────────
    run_sanity_checks(split_df, pers_test, pers_val, windows_labeled, cohort, window_elig)

    # ── 8. Metadata ────────────────────────────────────────────────────────────
    metadata = build_metadata(
        cohort, split_df, pers_test, pers_val,
        window_elig, testval_elig, stratify_col,
        fallback_log, out_dir, args,
    )

    # ── 9. Save ────────────────────────────────────────────────────────────────
    save_outputs(out_dir, split_df, pers_test, pers_val, windows_labeled, metadata, cohort)

    # ── 10. Summary ────────────────────────────────────────────────────────────
    print_summary(cohort, split_df, pers_test, pers_val,
                  window_elig, testval_elig, stratify_col, args)


if __name__ == "__main__":
    main()
