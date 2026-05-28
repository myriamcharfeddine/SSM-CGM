#!/usr/bin/env python3
"""Cohort selection pipeline for the AI-READI CGM dataset.

Reads the enriched multimodal parquet produced by create_multimodal_with_clinical.py
plus the participant_static_features table, applies the 4-stage cohort selection
procedure, and writes versioned artifacts consumed by training/evaluation code.

Stage 1 — Pre-filter: duration floor → boundary trim → gap splitting → imputation
Stage 2 — Characterization: window counting, sensitivity sweeps
Stage 3 — Validation: stratum representation, distribution drift, benchmark, artifact

Outputs (all in --output-dir):
  cohort.csv              one row per retained participant
  segments.csv            one row per retained segment
  forecast_windows.csv    one row per valid 48 h → 1 h forecasting window
  sweep_results.csv       joint duration × min-windows sweep grid
  cohort_selection_metadata.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration — mirrors the notebook (cell 26)
# ---------------------------------------------------------------------------

MAX_CONTEXT_H  = 48
ADAPTATION_H   = 48
MIN_EVAL_H     = 12
MIN_DURATION_H = MAX_CONTEXT_H + ADAPTATION_H + MIN_EVAL_H  # 108 h

CONTEXT_H = 48
TARGET_H  = 1
STRIDE_H  = 1

BIN_MINUTES = 5

GAP_THRESHOLDS_MIN  = {"cgm": 30, "hr": 60, "rr": 60, "activity": 60}
GAP_THRESHOLDS_BINS = {k: v // BIN_MINUTES for k, v in GAP_THRESHOLDS_MIN.items()}

TRIM_WINDOW_BINS    = 12    # 1-hour rolling window
TRIM_MISS_THRESHOLD = 0.20

CONTEXT_BINS  = CONTEXT_H * 60 // BIN_MINUTES   # 576
TARGET_BINS   = TARGET_H  * 60 // BIN_MINUTES   # 12
STRIDE_BINS   = STRIDE_H  * 60 // BIN_MINUTES   # 12
MIN_SEGMENT_H    = MAX_CONTEXT_H + TARGET_H      # 49 h
MIN_SEGMENT_BINS = MIN_SEGMENT_H * 60 // BIN_MINUTES

COMPLETENESS_WEIGHTS = {"cgm": 0.40, "hr": 0.20, "rr": 0.20, "activity": 0.20}
SCORE_WEIGHTS        = {"participants": 0.35, "windows": 0.65}

MIN_WINDOWS_GRID_BASE = [12, 24, 48, 96, 192, 384]
KNEE_PCT         = 0.95
STRATUM_TOL_PP   = 5.0
STRATUM_MIN_N    = 30
DRIFT_THRESHOLD  = 0.10

# study_group value → canonical stratum label
STRATUM_MAP: Dict[str, str] = {
    "healthy":                                                             "normoglycemia",
    "pre_diabetes_lifestyle_controlled":                                   "prediabetes",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "T2D-oral",
    "insulin_dependent":                                                   "T2D-insulin",
}
ALL_STRATA     = list(STRATUM_MAP.values()) + ["unclassified"]
GLYCEMIC_STRATA = list(STRATUM_MAP.values())

# Core modalities: key → column name in the time-series parquet
CORE_COLS: Dict[str, str] = {
    "cgm":      "cgm_glucose_mean",
    "hr":       "heart_rate_mean",
    "rr":       "respiratory_rate_mean",
    "activity": "activity_steps_per_min",
}

DEFAULT_DATA_DIR   = Path("/home/myriamcharfeddine/CGM/Data/enriched_multimodal")
DEFAULT_OUTPUT_DIR = Path("/home/myriamcharfeddine/CGM/Data/enriched_multimodal")

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _find_latest_parquet(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def _load_timeseries(path: Path) -> pd.DataFrame:
    print(f"Loading time-series: {path}")
    df = pd.read_parquet(path)
    ts_col = next((c for c in ("timestamp", "timestamp_local") if c in df.columns), None)
    if ts_col and not isinstance(df.index, pd.DatetimeIndex):
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if "participant_id" in df.columns:
        df["participant_id"] = df["participant_id"].astype(str)
    print(f"  {df['participant_id'].nunique()} participants | {len(df):,} bins")
    return df


def _load_static_features(path: Path) -> pd.DataFrame:
    print(f"Loading static features: {path}")
    sf = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    sf["participant_id"] = sf["participant_id"].astype(str)
    return sf


# ---------------------------------------------------------------------------
# Metadata building — map new enriched column names → canonical names
# ---------------------------------------------------------------------------

def build_meta(sf: pd.DataFrame) -> pd.DataFrame:
    """Extract participant metadata from participant_static_features.

    Handles both enriched (participants_* prefix) and legacy column names.
    """
    col_map = {
        # enriched names           → canonical names used internally
        "participants_study_group":    "study_group",
        "participants_clinical_site":  "clinical_site",
        "participants_age":            "age",
        "bmi_baseline":                "BMI",
        "hba1c_percent_baseline":      "HbA1c",
        # legacy fallbacks
        "study_group":                 "study_group",
        "clinical_site":               "clinical_site",
        "age":                         "age",
        "BMI":                         "BMI",
        "HbA1c":                       "HbA1c",
    }
    rename = {src: dst for src, dst in col_map.items() if src in sf.columns and src != dst}
    meta = sf.rename(columns=rename).copy()

    if "study_group" not in meta.columns:
        print("WARNING: study_group not found in static features — all participants marked 'unclassified'")
        meta["study_group"] = "unclassified"

    meta["stratum"] = meta["study_group"].map(STRATUM_MAP).fillna("unclassified")
    keep = ["participant_id", "stratum", "study_group"] + [
        c for c in ["clinical_site", "age", "BMI", "HbA1c"] if c in meta.columns
    ]
    return meta[[c for c in keep if c in meta.columns]].copy()


# ---------------------------------------------------------------------------
# Stage 1.1 — minimum duration floor
# ---------------------------------------------------------------------------

def stage1_duration_floor(df: pd.DataFrame, meta: pd.DataFrame,
                           min_duration_h: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute raw duration per participant and apply the minimum duration floor."""
    dur = (
        df.groupby("participant_id")
        .apply(lambda g: pd.Series({
            "duration_h": (g.index.max() - g.index.min()).total_seconds() / 3600,
            "n_bins_raw": len(g),
        }), include_groups=False)
        .reset_index()
    )
    dur["participant_id"] = dur["participant_id"].astype(str)

    cohort = dur.merge(meta, on="participant_id", how="left")
    cohort["stratum"] = cohort["stratum"].fillna("unclassified")

    before = len(cohort)
    cohort_s1 = cohort[cohort["duration_h"] >= min_duration_h].copy()
    after = len(cohort_s1)

    print(f"\nStage 1.1 — Duration floor: {min_duration_h:.0f} h")
    print(f"  Before: {before}  |  After: {after}  |  Removed: {before - after}")
    _print_stratum_table(cohort_s1, cohort, "1.1")
    return cohort_s1, cohort


# ---------------------------------------------------------------------------
# Stage 1.5 — boundary trimming
# ---------------------------------------------------------------------------

def _trim_participant(grp: pd.DataFrame, core_cols: List[str],
                      window_bins: int, miss_thr: float) -> Tuple[pd.DataFrame, dict]:
    g       = grp.sort_index()
    n       = len(g)
    is_miss = g[core_cols].isna().any(axis=1).astype(float)
    roll    = is_miss.rolling(window_bins, min_periods=1).mean()
    valid   = roll < miss_thr

    if not valid.any():
        return g.iloc[0:0], dict(n_raw=n, n_trimmed=0, lead=n, trail=0, driver="all")

    first_idx = valid.idxmax()
    last_idx  = valid[::-1].idxmax()
    trimmed   = g.loc[first_idx:last_idx]
    lead_cut  = g.index.get_loc(first_idx)
    trail_cut = n - g.index.get_loc(last_idx) - 1

    driver = "none"
    if lead_cut + trail_cut > 0:
        cut_rows = pd.concat([g[core_cols].iloc[:lead_cut],
                              g[core_cols].iloc[n - trail_cut:]])
        if len(cut_rows):
            driver = cut_rows.isna().sum().idxmax()

    return trimmed, dict(n_raw=n, n_trimmed=len(trimmed),
                         lead=lead_cut, trail=trail_cut, driver=driver)


def stage1_trim(df: pd.DataFrame, cohort_s1: pd.DataFrame,
                window_bins: int = TRIM_WINDOW_BINS,
                miss_thr: float = TRIM_MISS_THRESHOLD) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trim leading/trailing bins with high missingness; re-apply duration floor."""
    core_cols_list = list(CORE_COLS.values())
    valid_pids = set(cohort_s1["participant_id"].astype(str))
    df_sub = df[df["participant_id"].astype(str).isin(valid_pids)]

    trim_log, trimmed_pieces = [], {}
    for pid, grp in tqdm(df_sub.groupby("participant_id"), desc="Trimming", unit="participant"):
        t, log = _trim_participant(grp, core_cols_list, window_bins, miss_thr)
        trimmed_pieces[str(pid)] = t
        log["participant_id"] = str(pid)
        trim_log.append(log)

    df_trimmed   = pd.concat(trimmed_pieces.values()).sort_index()
    trim_log_df  = pd.DataFrame(trim_log)

    dur_post = (
        df_trimmed.groupby("participant_id")
        .apply(lambda g: pd.Series({
            "valid_start": g.index.min(),
            "valid_end":   g.index.max(),
            "duration_h_trimmed": (g.index.max() - g.index.min()).total_seconds() / 3600,
        }), include_groups=False)
        .reset_index()
    )
    dur_post["participant_id"] = dur_post["participant_id"].astype(str)

    cohort_s1 = cohort_s1.merge(dur_post, on="participant_id", how="left")
    cohort_s1 = cohort_s1.merge(
        trim_log_df[["participant_id", "lead", "trail", "driver"]],
        on="participant_id", how="left",
    ).rename(columns={"lead": "bins_trimmed_lead", "trail": "bins_trimmed_trail",
                       "driver": "driving_modality"})

    cohort_trim = cohort_s1[cohort_s1["duration_h_trimmed"] >= MIN_DURATION_H].copy()

    print(f"\nStage 1.5 — Boundary trimming (window={window_bins} bins, thr={miss_thr:.0%}):")
    print(f"  Before: {len(cohort_s1)}  |  After re-applying floor: {len(cohort_trim)}"
          f"  |  Dropped: {len(cohort_s1) - len(cohort_trim)}")
    total_trim = (trim_log_df["lead"] + trim_log_df["trail"])
    print(f"  Bins trimmed — mean: {total_trim.mean():.1f}  max: {total_trim.max()}")
    print(f"  Modality driving trim:\n{trim_log_df['driver'].value_counts().to_string()}")
    return cohort_trim, df_trimmed, trim_log_df


# ---------------------------------------------------------------------------
# Stage 1.3 — gap-based segmentation (split, not exclude)
# ---------------------------------------------------------------------------

def _find_long_gap_mask(series: pd.Series, threshold_bins: int) -> pd.Series:
    is_null = series.isna()
    if not is_null.any():
        return pd.Series(False, index=series.index)
    run_id   = (~is_null).cumsum()
    run_lens = is_null.groupby(run_id).sum()
    bad_runs = set(run_lens[run_lens > threshold_bins].index)
    if not bad_runs:
        return pd.Series(False, index=series.index)
    return is_null & run_id.isin(bad_runs)


def _segment_participant(grp: pd.DataFrame,
                          gap_thr_bins: Dict[str, int] = GAP_THRESHOLDS_BINS,
                          min_seg_bins: int = MIN_SEGMENT_BINS) -> List[pd.DataFrame]:
    grp = grp.sort_index()
    bad = pd.Series(False, index=grp.index)
    for k, col in CORE_COLS.items():
        bad |= _find_long_gap_mask(grp[col], gap_thr_bins[k])

    good = ~bad
    if not good.any():
        return []
    seg_label = (good & (good != good.shift(fill_value=False))).cumsum()
    seg_label[bad] = 0
    segments = []
    for _, seg_grp in grp[good].groupby(seg_label[good]):
        if len(seg_grp) >= min_seg_bins:
            segments.append(seg_grp)
    return segments


def _compute_max_gap_bins(series: pd.Series) -> int:
    is_null = series.isna()
    if not is_null.any():
        return 0
    run_id = (~is_null).cumsum()
    return int(is_null.groupby(run_id).sum().max())


def stage1_segment(df_trimmed: pd.DataFrame,
                   cohort_trim: pd.DataFrame) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Split recordings at long gaps; keep segments >= MIN_SEGMENT_BINS."""
    valid_pids    = set(cohort_trim["participant_id"].astype(str))
    df_trim_filt  = df_trimmed[df_trimmed["participant_id"].astype(str).isin(valid_pids)]

    seg_meta_rows, all_segments = [], {}
    old_excluded = 0

    for pid, grp in tqdm(df_trim_filt.groupby("participant_id"),
                         desc="Segmenting", unit="participant"):
        pid_str = str(pid)
        if any(_compute_max_gap_bins(grp[col]) > GAP_THRESHOLDS_BINS[k]
               for k, col in CORE_COLS.items()):
            old_excluded += 1

        segs = _segment_participant(grp)
        for i, seg in enumerate(segs):
            all_segments[(pid_str, i)] = seg
            seg_meta_rows.append(dict(
                participant_id=pid_str, segment_id=i,
                valid_start=seg.index.min(), valid_end=seg.index.max(),
                n_bins=len(seg),
                duration_h=(seg.index.max() - seg.index.min()).total_seconds() / 3600,
            ))

    segments_meta = pd.DataFrame(seg_meta_rows)
    pids_with_segs = set(segments_meta["participant_id"])
    cohort_s2 = cohort_trim[cohort_trim["participant_id"].astype(str).isin(pids_with_segs)].copy()

    segs_per_pid = segments_meta.groupby("participant_id")["segment_id"].count()
    n_multi = (segs_per_pid > 1).sum()
    print(f"\nStage 1.3 — Gap segmentation (split, not exclude):")
    print(f"  Participants in: {len(cohort_trim)}")
    print(f"  Participants with ≥1 segment: {len(pids_with_segs)}")
    print(f"  Participants with 0 segments (lost): {len(cohort_trim) - len(pids_with_segs)}")
    print(f"  Total segments: {len(segments_meta)}")
    print(f"  Participants with >1 segment: {n_multi}")
    print(f"  Mean segments/participant: {segs_per_pid.mean():.2f}")
    print(f"  Would-have-excluded (old approach): {old_excluded}")
    print(f"  Recovered by segmentation: {old_excluded - (len(cohort_trim) - len(pids_with_segs))}")
    return cohort_s2, all_segments, segments_meta


# ---------------------------------------------------------------------------
# Stage 1.4 — imputation within segments
# ---------------------------------------------------------------------------

def _impute_segment(seg: pd.DataFrame) -> pd.DataFrame:
    g = seg.copy()
    g["cgm_glucose_mean"]       = g["cgm_glucose_mean"].interpolate("linear", limit=IMPUTE_LIMITS["cgm"])
    g["heart_rate_mean"]        = g["heart_rate_mean"].interpolate("linear", limit=IMPUTE_LIMITS["hr"])
    g["respiratory_rate_mean"]  = g["respiratory_rate_mean"].interpolate("linear", limit=IMPUTE_LIMITS["rr"])
    g["activity_steps_per_min"] = g["activity_steps_per_min"].fillna(0.0)
    return g

IMPUTE_LIMITS = {k: GAP_THRESHOLDS_BINS[k] for k in CORE_COLS}


def stage1_impute(all_segments: dict) -> dict:
    """Linear interpolation for CGM/HR/RR; zero-fill for activity. Per segment."""
    all_imputed = {}
    for (pid, seg_id), seg in tqdm(all_segments.items(),
                                    desc="Imputing", unit="segment"):
        imp = _impute_segment(seg).copy()
        imp["participant_id"] = str(pid)
        imp["segment_id"]     = int(seg_id)
        all_imputed[(str(pid), int(seg_id))] = imp

    df_s3 = pd.concat(all_imputed.values()).sort_index()

    post_null = {k: df_s3[col].isna().sum() for k, col in CORE_COLS.items()}
    print(f"\nStage 1.4 — Post-imputation NaN in core modalities:")
    for k, n in post_null.items():
        status = "✓ 0" if n == 0 else f"WARNING: {n} remain (boundary bins, excluded by window filter)"
        print(f"  {k:<12}: {status}")
    print(f"  df_s3 → {df_s3['participant_id'].nunique()} participants | "
          f"{len(all_imputed)} segments | {len(df_s3):,} bins")
    return all_imputed


# ---------------------------------------------------------------------------
# Stage 2.0 — window counting and completeness
# ---------------------------------------------------------------------------

def _count_forecastable_windows(arr: np.ndarray) -> int:
    n = len(arr)
    if n < CONTEXT_BINS + TARGET_BINS:
        return 0
    count = 0
    for start in range(0, n - CONTEXT_BINS - TARGET_BINS + 1, STRIDE_BINS):
        if not np.isnan(arr[start: start + CONTEXT_BINS + TARGET_BINS]).any():
            count += 1
    return count


def stage2_window_count(all_imputed: dict, cohort_s2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Count valid 48h→1h windows per segment; aggregate per participant."""
    seg_rows = []
    for (pid, seg_id), seg in tqdm(all_imputed.items(),
                                    desc="Counting windows", unit="segment"):
        arr   = seg["cgm_glucose_mean"].values
        n_w   = _count_forecastable_windows(arr)
        compl = {k: seg[col].notna().mean() for k, col in CORE_COLS.items()}
        wc    = sum(COMPLETENESS_WEIGHTS[k] * compl[k] for k in CORE_COLS)
        seg_rows.append(dict(participant_id=str(pid), segment_id=seg_id,
                             n_valid_windows=n_w, n_bins=len(seg),
                             weighted_completeness=wc,
                             **{f"compl_{k}": compl[k] for k in CORE_COLS}))

    seg_stats = pd.DataFrame(seg_rows)

    agg_rows = []
    for pid, grp in seg_stats.groupby("participant_id"):
        total_bins = grp["n_bins"].sum()
        wc_agg = (grp["weighted_completeness"] * grp["n_bins"]).sum() / total_bins
        agg_rows.append(dict(
            participant_id=str(pid),
            n_valid_windows=int(grp["n_valid_windows"].sum()),
            n_segments=len(grp),
            total_clean_bins=int(total_bins),
            weighted_completeness=float(wc_agg),
            **{f"compl_{k}": float((grp[f"compl_{k}"] * grp["n_bins"]).sum() / total_bins)
               for k in CORE_COLS},
        ))
    agg_df = pd.DataFrame(agg_rows)

    cohort_s2 = cohort_s2.merge(
        agg_df[["participant_id", "n_valid_windows", "weighted_completeness",
                "n_segments", "total_clean_bins"]],
        on="participant_id", how="left",
    )

    print(f"\nStage 2.0 — Window counting:")
    print(f"  Participants: {len(cohort_s2)}")
    print(f"  Total valid windows: {cohort_s2['n_valid_windows'].sum():,}")
    print(f"  Mean windows/participant: {cohort_s2['n_valid_windows'].mean():.1f}")
    print(f"  Mean weighted completeness: {cohort_s2['weighted_completeness'].mean()*100:.1f}%")
    return cohort_s2, seg_stats


# ---------------------------------------------------------------------------
# Stage 2.1 — sensitivity to duration floor
# ---------------------------------------------------------------------------

def stage2_duration_sensitivity(cohort_s2: pd.DataFrame) -> pd.DataFrame:
    dur_range = np.arange(72, 240 + 6, 6)
    rows = []
    for dur_h in dur_range:
        mask = (cohort_s2["duration_h_trimmed"] >= dur_h) & \
               (cohort_s2["n_valid_windows"].fillna(0) >= 1)
        cand = cohort_s2[mask]
        n    = len(cand)
        row  = dict(dur_floor_h=float(dur_h), dur_floor_d=dur_h / 24,
                    n_participants=n,
                    n_valid_windows=int(cand["n_valid_windows"].sum()) if n else 0)
        for s in ALL_STRATA:
            row[f"pct_{s}"] = 100 * (cand["stratum"] == s).sum() / n if n else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 2.2 — joint sensitivity: duration × min-windows
# ---------------------------------------------------------------------------

def stage2_joint_sweep(cohort_s2: pd.DataFrame) -> pd.DataFrame:
    dur_thresholds = np.arange(MIN_DURATION_H / 24, 14.0 + 0.25, 0.25)
    win_grid       = MIN_WINDOWS_GRID_BASE
    total_pids     = len(cohort_s2)
    total_windows  = int(cohort_s2["n_valid_windows"].fillna(0).sum())

    rows = []
    for dur_d in dur_thresholds:
        for min_w in win_grid:
            mask = (cohort_s2["duration_h_trimmed"] / 24 >= dur_d) & \
                   (cohort_s2["n_valid_windows"].fillna(0) >= min_w)
            cand = cohort_s2[mask]
            n    = len(cand)
            w    = int(cand["n_valid_windows"].sum()) if n else 0
            score = ((n / total_pids) ** SCORE_WEIGHTS["participants"] *
                     (w / max(1, total_windows)) ** SCORE_WEIGHTS["windows"])
            rows.append(dict(dur_thr=float(dur_d), min_windows_thr=int(min_w),
                             n_participants=n, n_windows=w, score=float(score)))

    sweep_df = pd.DataFrame(rows)
    best     = sweep_df.loc[sweep_df["score"].idxmax()]
    print(f"\nStage 2 joint sweep: {len(sweep_df)} candidates")
    print(f"  Max-score candidate: dur={best['dur_thr']:.2f}d | "
          f"min_windows={int(best['min_windows_thr'])} | "
          f"N={int(best['n_participants'])} | "
          f"windows={int(best['n_windows']):,} | score={best['score']:.4f}")
    return sweep_df


# ---------------------------------------------------------------------------
# Stage 2.3 — post-selection diagnostics
# ---------------------------------------------------------------------------

def stage2_diagnostics(all_imputed: dict, cohort_s2: pd.DataFrame) -> pd.DataFrame:
    """Compute completeness, longest segment, personalization eligibility per participant."""
    PERSONALIZATION_H = MAX_CONTEXT_H + ADAPTATION_H + MIN_EVAL_H
    ref_pids = set(cohort_s2["participant_id"].astype(str))

    diag_rows = []
    for (pid, seg_id), seg in all_imputed.items():
        if str(pid) not in ref_pids:
            continue
        n_bins = len(seg)
        dur_h  = n_bins * BIN_MINUTES / 60
        wc     = sum(COMPLETENESS_WEIGHTS[k] * seg[col].notna().mean()
                     for k, col in CORE_COLS.items())
        diag_rows.append(dict(
            participant_id=str(pid), segment_id=int(seg_id),
            dur_h=dur_h, n_bins=n_bins, post_weighted_completeness=wc,
            **{f"post_compl_{k}": seg[col].notna().mean() for k, col in CORE_COLS.items()},
        ))

    diag_seg_df = pd.DataFrame(diag_rows)
    pid_rows = []
    for pid, grp in diag_seg_df.groupby("participant_id"):
        total_bins  = grp["n_bins"].sum()
        total_dur_h = grp["dur_h"].sum()
        longest_h   = grp["dur_h"].max()
        wc_agg      = (grp["post_weighted_completeness"] * grp["n_bins"]).sum() / total_bins
        pid_rows.append(dict(
            participant_id=pid,
            total_clean_dur_h=total_dur_h,
            longest_seg_h=longest_h,
            post_weighted_completeness=wc_agg,
            personalization_eligible=longest_h >= PERSONALIZATION_H,
        ))
    pid_diag = pd.DataFrame(pid_rows)

    n_pers = pid_diag["personalization_eligible"].sum()
    print(f"\nStage 2.3 — Post-selection diagnostics (N={len(pid_diag)})")
    print(f"  Weighted completeness: {pid_diag['post_weighted_completeness'].mean()*100:.2f}%")
    print(f"  Longest segment — mean {pid_diag['longest_seg_h'].mean():.1f}h  "
          f"median {pid_diag['longest_seg_h'].median():.1f}h")
    print(f"  Total clean duration — mean {pid_diag['total_clean_dur_h'].mean():.1f}h  "
          f"({pid_diag['total_clean_dur_h'].mean()/24:.1f}d)")
    print(f"  Personalization-eligible (≥{PERSONALIZATION_H}h segment): "
          f"{n_pers}/{len(pid_diag)} ({100*n_pers/len(pid_diag):.1f}%)")
    return pid_diag, diag_seg_df


# ---------------------------------------------------------------------------
# Stage 3.1 — stratum representation
# ---------------------------------------------------------------------------

def stage3_stratum_representation(cohort_s2: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    orig_counts = meta[meta["stratum"].isin(GLYCEMIC_STRATA)]["stratum"].value_counts()
    orig_total  = sum(orig_counts.get(s, 0) for s in GLYCEMIC_STRATA)
    ret_total   = int(cohort_s2["stratum"].isin(GLYCEMIC_STRATA).sum())

    print(f"\nStage 3.1 — Stratum representation (N retained classified={ret_total})")
    print(f"  {'Stratum':<22}  {'Orig':>5}  {'Ret':>5}  {'Rate':>8}  "
          f"{'Orig%':>7}  {'Ret%':>7}  {'Shift':>8}")
    print("  " + "-" * 70)

    rows = []
    for s in GLYCEMIC_STRATA:
        orig_n   = int(orig_counts.get(s, 0))
        ret_n    = int((cohort_s2["stratum"] == s).sum())
        rate     = 100 * ret_n / orig_n if orig_n else 0.0
        orig_pp  = 100 * orig_n / orig_total if orig_total else 0.0
        ret_pp   = 100 * ret_n  / ret_total  if ret_total  else 0.0
        shift    = ret_pp - orig_pp
        flag     = "  ◀ non-trivial" if abs(shift) > STRATUM_TOL_PP else ""
        print(f"  {s:<22}  {orig_n:>5}  {ret_n:>5}  {rate:>7.1f}%  "
              f"{orig_pp:>6.1f}%  {ret_pp:>6.1f}%  {shift:>+7.1f}pp{flag}")
        rows.append(dict(stratum=s, orig_n=orig_n, ret_n=ret_n, retention_pct=rate,
                         orig_pct=orig_pp, ret_pct=ret_pp, shift_pp=shift,
                         non_trivial=abs(shift) > STRATUM_TOL_PP))

    print(f"\n  Total classified retained: {ret_total}/{orig_total} "
          f"({100*ret_total/orig_total:.1f}%)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 3.2 — distribution comparison vs full release
# ---------------------------------------------------------------------------

def stage3_distribution_comparison(df_s3: pd.DataFrame, meta: pd.DataFrame,
                                    cohort_s2: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy import stats as scipy_stats
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        print("Stage 3.2 skipped — scipy/statsmodels not available.")
        return pd.DataFrame()

    cgm_metrics = (
        df_s3.groupby("participant_id")["cgm_glucose_mean"]
        .agg(
            mean_glucose=lambda x: x.mean(),
            tir_70_180   =lambda x: ((x >= 70) & (x <= 180)).mean() * 100,
            cv_pct       =lambda x: (x.std() / x.mean() * 100) if x.mean() > 0 else np.nan,
        )
        .reset_index()
    )

    full_clin = meta.merge(cgm_metrics, on="participant_id", how="left")
    ret_pids  = set(cohort_s2["participant_id"].astype(str))
    ret_clin  = full_clin[full_clin["participant_id"].isin(ret_pids)]
    non_ret   = full_clin[~full_clin["participant_id"].isin(ret_pids)]

    CONT_VARS = [c for c in ["age", "BMI", "HbA1c", "mean_glucose", "tir_70_180", "cv_pct"]
                 if c in full_clin.columns]
    CAT_VARS  = [c for c in ["clinical_site", "stratum"] if c in full_clin.columns]

    rows = []
    for col in CONT_VARS:
        a = ret_clin[col].dropna().values
        b = non_ret[col].dropna().values
        if len(a) < 5 or len(b) < 5:
            continue
        ks_s, ks_p = scipy_stats.ks_2samp(a, b)
        mw_s, mw_p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(variable=col, test="KS+MW", stat=float(ks_s),
                         p_value=float(min(ks_p, mw_p))))

    for col in CAT_VARS:
        a = ret_clin[col].dropna()
        b = non_ret[col].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        cats = sorted(set(a) | set(b))
        obs  = np.array([[(a == c).sum(), (b == c).sum()] for c in cats])
        chi2, p, _, _ = scipy_stats.chi2_contingency(obs)
        rows.append(dict(variable=col, test="chi2", stat=float(chi2), p_value=float(p)))

    if not rows:
        return pd.DataFrame()

    test_df = pd.DataFrame(rows)
    _, adj_p, _, _ = multipletests(test_df["p_value"], method="fdr_bh")
    test_df["p_adj_BH"] = adj_p
    test_df["flagged"]  = test_df["p_adj_BH"] < 0.05

    print(f"\nStage 3.2 — Distribution comparison (N retained={len(ret_clin)}, non-retained={len(non_ret)})")
    print(f"  {'Variable':<18}  {'Test':>6}  {'Stat':>9}  {'p(raw)':>10}  {'p(BH)':>10}  Flag")
    print("  " + "-" * 65)
    for _, r in test_df.iterrows():
        flag = "  ▲" if r["flagged"] else ""
        print(f"  {r['variable']:<18}  {r['test']:>6}  {r['stat']:>9.3f}"
              f"  {r['p_value']:>10.4f}  {r['p_adj_BH']:>10.4f}{flag}")
    n_flagged = int(test_df["flagged"].sum())
    print(f"\n  {n_flagged}/{len(test_df)} variables flagged (BH p < 0.05).")
    return test_df


# ---------------------------------------------------------------------------
# Stage 3.3 — benchmark against published SSM-CGM statistics
# ---------------------------------------------------------------------------

def stage3_benchmark(df_s3: pd.DataFrame, cohort_s2: pd.DataFrame,
                     ref_parquet: Optional[Path] = None) -> dict:
    ret_bins = df_s3[df_s3["participant_id"].isin(
        set(cohort_s2["participant_id"].astype(str))
    )]["cgm_glucose_mean"].dropna()

    coh_mean_g = float(ret_bins.mean())
    coh_tir    = float(((ret_bins >= 70) & (ret_bins <= 180)).mean() * 100)
    coh_cv     = float(ret_bins.std() / ret_bins.mean() * 100)

    # Published AI-READI statistics from Isaac & Collin et al., NeurIPS 2025, Table 2
    # Used as fallback when no --benchmark-parquet is provided
    PUBLISHED_AIREADI = {"mean_g": 133.12, "tir": 88.46, "cv": 32.92}
    ref_mean_g = PUBLISHED_AIREADI["mean_g"]
    ref_tir    = PUBLISHED_AIREADI["tir"]
    ref_cv     = PUBLISHED_AIREADI["cv"]
    if ref_parquet and ref_parquet.exists():
        ref_df = pd.read_parquet(ref_parquet)
        if "cgm_glucose_mean" in ref_df.columns:
            rb       = ref_df["cgm_glucose_mean"].dropna()
            ref_mean_g = float(rb.mean())
            ref_tir  = float(((rb >= 70) & (rb <= 180)).mean() * 100)
            ref_cv   = float(rb.std() / rb.mean() * 100)

    bench = {
        "Mean glucose (mg/dL)": (coh_mean_g, ref_mean_g),
        "TIR 70-180 (%)":       (coh_tir,    ref_tir),
        "CV (%)":               (coh_cv,     ref_cv),
    }

    print(f"\nStage 3.3 — Benchmark against{'reference parquet' if ref_parquet else 'N/A (no --benchmark-parquet)'}:")
    print(f"  {'Metric':<24}  {'Cohort':>8}  {'Reference':>10}  {'|Drift|':>9}  Flag")
    print("  " + "-" * 58)
    for metric, (coh_val, ref_val) in bench.items():
        if np.isnan(ref_val):
            print(f"  {metric:<24}  {coh_val:>8.2f}  {'N/A':>10}  {'—':>9}")
        else:
            drift = abs(coh_val - ref_val) / abs(ref_val) if ref_val != 0 else 0
            flag  = "  ▲ >10%" if drift > DRIFT_THRESHOLD else ""
            print(f"  {metric:<24}  {coh_val:>8.2f}  {ref_val:>10.2f}  {drift:>8.1%}{flag}")

    return {m: {"cohort": v[0], "reference": v[1]} for m, v in bench.items()}


# ---------------------------------------------------------------------------
# Stage 3.4 — versioned artifact
# ---------------------------------------------------------------------------

def stage3_artifact(cohort_s2: pd.DataFrame, pid_diag: pd.DataFrame,
                    diag_seg_df: pd.DataFrame, all_imputed: dict,
                    output_dir: Path) -> str:
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    try:
        git_hash = subprocess.check_output(
            ["git", "-C", str(output_dir), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"
    tag = f"{timestamp}_{git_hash}"
    print(f"\nStage 3.4 — Writing versioned artifact (tag: {tag})")

    # cohort.csv
    cohort_out = cohort_s2.merge(pid_diag, on="participant_id", how="left")
    cohort_out["artifact_tag"] = tag
    cohort_path = output_dir / "cohort.csv"
    cohort_out.to_csv(cohort_path, index=False)
    print(f"  cohort.csv           → {cohort_path}  ({len(cohort_out)} rows)")

    # segments.csv
    seg_out = diag_seg_df.copy()
    seg_out["artifact_tag"] = tag
    seg_path = output_dir / "segments.csv"
    seg_out.to_csv(seg_path, index=False)
    print(f"  segments.csv         → {seg_path}  ({len(seg_out)} rows)")

    # forecast_windows.csv
    ret_pids    = set(cohort_s2["participant_id"].astype(str))
    WINDOW_BINS = CONTEXT_BINS + TARGET_BINS
    window_rows = []
    for (pid, seg_id), seg in all_imputed.items():
        if str(pid) not in ret_pids:
            continue
        cgm = seg["cgm_glucose_mean"].values
        n   = len(cgm)
        if n < WINDOW_BINS:
            continue
        for start in range(0, n - WINDOW_BINS + 1, STRIDE_BINS):
            if not np.isnan(cgm[start: start + WINDOW_BINS]).any():
                window_rows.append(dict(
                    participant_id=pid, segment_id=int(seg_id),
                    window_start=int(start), window_end=int(start + WINDOW_BINS),
                ))

    forecast_df = pd.DataFrame(window_rows)
    forecast_df["artifact_tag"] = tag
    fw_path = output_dir / "forecast_windows.csv"
    forecast_df.to_csv(fw_path, index=False)
    print(f"  forecast_windows.csv → {fw_path}  ({len(forecast_df):,} rows)")
    return tag


# ---------------------------------------------------------------------------
# Stage summary printer
# ---------------------------------------------------------------------------

def _print_stratum_table(cohort_now: pd.DataFrame, cohort_full: pd.DataFrame, stage: str) -> None:
    print(f"\n  Stratum breakdown (Stage {stage}):")
    for s in ALL_STRATA:
        n = int((cohort_now["stratum"] == s).sum()) if "stratum" in cohort_now.columns else 0
        print(f"    {s:<22}: {n}")


def print_stage1_summary(cohort_raw, cohort_s1, cohort_trim, cohort_s2, n_imputed_pids):
    print("\n" + "="*55)
    print("STAGE 1 SUMMARY")
    print("="*55)
    stages = [
        ("Raw dataset",                     len(cohort_raw),       ""),
        ("After 1.1 Min duration floor",    len(cohort_s1),        f">={MIN_DURATION_H}h"),
        ("After 1.5 Boundary trim",         len(cohort_trim),      "cleaned edges + re-applied floor"),
        ("After 1.3 Gap splitting",         len(cohort_s2),        "≥1 valid segment"),
        ("After 1.4 Imputation",            n_imputed_pids,        "short gaps filled within segments"),
    ]
    for label, n, crit in stages:
        print(f"  {label:<38}: {n:5d}  {crit}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CGM cohort selection pipeline (Stage 1–3)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="Directory containing the final_multimodal_dataset_*.parquet")
    p.add_argument("--parquet", type=Path, default=None,
                   help="Explicit path to the multimodal parquet (overrides --data-dir auto-discovery)")
    p.add_argument("--static-features", type=Path, default=None,
                   help="Path to participant_static_features.parquet (auto-discovered from --data-dir if omitted)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Directory for cohort.csv, segments.csv, forecast_windows.csv")
    p.add_argument("--benchmark-parquet", type=Path, default=None,
                   help="Optional reference parquet for Stage 3.3 benchmark comparison")
    p.add_argument("--min-duration-h", type=float, default=float(MIN_DURATION_H),
                   help=f"Minimum recording duration in hours (default: {MIN_DURATION_H})")
    p.add_argument("--no-sweep", action="store_true",
                   help="Skip Stage 2 sensitivity sweeps (faster)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Auto-discover inputs ───────────────────────────────────────────────
    parquet_path = args.parquet or _find_latest_parquet(
        args.data_dir, "final_multimodal_dataset_*.parquet"
    )
    if parquet_path is None:
        raise SystemExit(f"No final_multimodal_dataset_*.parquet found in {args.data_dir}. "
                         "Pass --parquet explicitly.")

    static_path = args.static_features or _find_latest_parquet(
        args.data_dir, "participant_static_features.parquet"
    )
    if static_path is None:
        # try csv
        csv_candidate = args.data_dir / "participant_static_features.csv"
        static_path = csv_candidate if csv_candidate.exists() else None

    # ── Load data ──────────────────────────────────────────────────────────
    df = _load_timeseries(parquet_path)

    if static_path:
        sf   = _load_static_features(static_path)
        meta = build_meta(sf)
    else:
        print("WARNING: no participant_static_features found — "
              "stratum inferred from parquet columns if available")
        if "participants_study_group" in df.columns:
            meta = df[["participant_id", "participants_study_group"]].drop_duplicates()
            meta = meta.rename(columns={"participants_study_group": "study_group"})
            meta["stratum"] = meta["study_group"].map(STRATUM_MAP).fillna("unclassified")
        else:
            pids = df["participant_id"].unique()
            meta = pd.DataFrame({"participant_id": pids, "study_group": "unclassified",
                                  "stratum": "unclassified"})

    print(f"\nMeta: {len(meta)} participants | strata: {meta['stratum'].value_counts().to_dict()}")

    # ── Stage 1 ───────────────────────────────────────────────────────────
    cohort_s1, cohort_raw = stage1_duration_floor(df, meta, args.min_duration_h)
    cohort_trim, df_trimmed, trim_log_df = stage1_trim(df, cohort_s1)
    cohort_s2, all_segments, segments_meta = stage1_segment(df_trimmed, cohort_trim)
    all_imputed = stage1_impute(all_segments)
    df_s3 = pd.concat(all_imputed.values()).sort_index()
    n_imputed_pids = df_s3["participant_id"].nunique()

    print_stage1_summary(cohort_raw, cohort_s1, cohort_trim, cohort_s2, n_imputed_pids)

    # ── Stage 2 ───────────────────────────────────────────────────────────
    cohort_s2, seg_stats = stage2_window_count(all_imputed, cohort_s2)
    cohort_reference = cohort_s2.copy()

    sweep_df = pd.DataFrame()
    if not args.no_sweep:
        sens_dur_df  = stage2_duration_sensitivity(cohort_s2)
        sweep_df     = stage2_joint_sweep(cohort_s2)
        sweep_path   = args.output_dir / "sweep_results.csv"
        sweep_df.to_csv(sweep_path, index=False)
        print(f"  Wrote: {sweep_path}")
        dur_sens_path = args.output_dir / "sensitivity_duration.csv"
        sens_dur_df.to_csv(dur_sens_path, index=False)

    pid_diag, diag_seg_df = stage2_diagnostics(all_imputed, cohort_reference)

    # ── Stage 3 ───────────────────────────────────────────────────────────
    stratum_rep  = stage3_stratum_representation(cohort_reference, meta)
    dist_test_df = stage3_distribution_comparison(df_s3, meta, cohort_reference)
    bench        = stage3_benchmark(df_s3, cohort_reference, args.benchmark_parquet)

    stratum_rep.to_csv(args.output_dir / "stage3_stratum_representation.csv", index=False)
    if not dist_test_df.empty:
        dist_test_df.to_csv(args.output_dir / "stage3_distribution_tests.csv", index=False)
    seg_stats.to_csv(args.output_dir / "segment_stats.csv", index=False)

    artifact_tag = stage3_artifact(cohort_reference, pid_diag, diag_seg_df,
                                   all_imputed, args.output_dir)

    # ── Metadata JSON ──────────────────────────────────────────────────────
    meta_out = {
        "artifact_tag": artifact_tag,
        "input_parquet": str(parquet_path),
        "static_features": str(static_path),
        "config": {
            "min_duration_h": args.min_duration_h,
            "gap_thresholds_min": GAP_THRESHOLDS_MIN,
            "trim_window_bins": TRIM_WINDOW_BINS,
            "trim_miss_threshold": TRIM_MISS_THRESHOLD,
            "context_h": CONTEXT_H, "target_h": TARGET_H, "stride_h": STRIDE_H,
        },
        "counts": {
            "raw": len(cohort_raw),
            "after_duration_floor": len(cohort_s1),
            "after_trim": len(cohort_trim),
            "after_segmentation": len(cohort_s2),
            "total_segments": len(diag_seg_df),
            "total_forecast_windows": int(cohort_s2["n_valid_windows"].sum()),
        },
        "benchmark": bench,
    }
    with open(args.output_dir / "cohort_selection_metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2, default=str)

    print(f"\nAll outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
