"""
prepare_ssmcgm_data_enriched.py
================================
Experiment B — dynamic + static features.

Differences from Experiment A (prepare_ssmcgm_data.py):
  - Adds static clinical, demographic, and medication covariates to every row.
  - Dynamic columns are the same exact 29-column schema as Experiment A.
  - Encoding is fit on train participants only (no leakage):
      binary flags     → fill NaN with 0
      numeric baselines→ per-participant median imputation + z-score
      categorical      → one-hot (demo_sex_at_birth)
  - Outputs go to Data/ssmcgm_ready_enriched/ (separate from Exp A)

Why late fusion via TFT static branch:
  pytorch_forecasting's TFT has a Variable Selection Network (VSN) for
  static inputs and a static-enrichment path that conditions the decoder.
  Adding our covariates as extra static_reals in TimeSeriesDataSet gives
  automatic late fusion + feature-importance interpretability at no cost.

Cohort and split: identical to Experiment A (same cohort.csv, horizon=12).
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Import shared dynamic pipeline from Experiment A ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from prepare_ssmcgm_data import (
    load_and_filter,
    rename_columns,
    build_sleep_stage,
    build_timestamps,
    build_time_features,
    build_cgm_features,
    train_test_split,
    PARQUET, COHORT_CSV, HORIZON, FINAL_COLS,
)

# ── Output directory ───────────────────────────────────────────────────────────
ROOT    = Path("/home/myriamcharfeddine/CGM")
OUT_DIR = ROOT / "Data/ssmcgm_ready_enriched"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Static feature definitions ─────────────────────────────────────────────────
STATIC_NUMERIC = [
    "hba1c_percent_baseline",
    "bmi_baseline",
    "waist_to_hip_ratio_baseline",
    "clinical_systolic_bp_mmhg_baseline",
    "clinical_diastolic_bp_mmhg_baseline",
    "clinical_resting_hr_bpm_baseline",
    "serum_glucose_mgdl_baseline",
    "c_peptide_ngml_baseline",
    "serum_insulin_uuml_baseline",
    "hdl_cholesterol_mgdl_baseline",
    "ldl_cholesterol_mgdl_baseline",
    "triglycerides_mgdl_baseline",
]

STATIC_BINARY = [
    "med_metformin",
    "med_insulin",
    "med_glp1_or_gip_glp1",
    "med_sglt2",
    "med_sulfonylurea",
    "med_thiazolidinedione",
    "med_any_diabetes_drug",
]

STATIC_CATEGORICAL = ["demo_sex_at_birth"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def discover_available_cols(df, requested):
    """Return (available, missing) lists; warn on missing."""
    available = [c for c in requested if c in df.columns]
    missing   = [c for c in requested if c not in df.columns]
    if missing:
        print(f"  [WARN] {len(missing)} requested columns not in parquet (skipped): {missing}")
    return available, missing


def encode_static_features(df, avail_numeric, avail_binary, avail_categorical,
                            is_train, metadata=None):
    """
    Encode static features in place.

    is_train=True  → fit imputation / scaling on this df; populate metadata.
    is_train=False → apply existing metadata fitted on train.

    Returns (df, metadata, final_encoded_col_names).
    The original string categorical column is dropped; OHE columns are added.
    """
    if metadata is None:
        metadata = {}
    final_cols = []

    # ── Binary flags: fill NaN → 0.0 ──────────────────────────────────────────
    for col in avail_binary:
        df[col] = df[col].fillna(0.0).astype(float)
        final_cols.append(col)
        if is_train:
            metadata[col] = {"type": "binary"}

    # ── Numeric: median impute (per participant) + z-score ────────────────────
    for col in avail_numeric:
        if is_train:
            pid_first = df.groupby("participant_id")[col].first()
            med  = float(pid_first.median()) if pid_first.notna().any() else 0.0
            filled = pid_first.fillna(med)
            mean = float(filled.mean())
            std  = max(float(filled.std(ddof=1)), 1e-6)
            metadata[col] = {"type": "numeric", "median": med, "mean": mean, "std": std}
        else:
            med  = metadata[col]["median"]
            mean = metadata[col]["mean"]
            std  = metadata[col]["std"]
        df[col] = df[col].fillna(med)
        df[col] = (df[col].astype(float) - mean) / std
        final_cols.append(col)

    # ── Categorical: one-hot (categories fitted on train) ─────────────────────
    for col in avail_categorical:
        if is_train:
            cats = sorted(df[col].dropna().astype(str).unique().tolist())
            metadata[col] = {"type": "categorical", "categories": cats}
        else:
            cats = metadata[col]["categories"]

        df[col] = df[col].fillna("unknown").astype(str)
        for cat in cats:
            slug    = cat.replace(" ", "_").replace("/", "_").lower()
            ohe_col = f"{col}_{slug}"
            df[ohe_col] = (df[col] == cat).astype(float)
            final_cols.append(ohe_col)
        df = df.drop(columns=[col])

    return df, metadata, final_cols


def finalize_enriched(df, raw_static_cols):
    """
    Drop rows with missing CGM, recompute ds (must be consecutive after dropna),
    and keep the 29 dynamic columns + all raw static columns.

    raw_static_cols: columns to carry forward before encoding (may include
    string categoricals like demo_sex_at_birth that get encoded later).
    """
    n_before = len(df)
    df = df.dropna(subset=["cgm_glucose"]).reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped:
        print(f"      Dropped {dropped:,} rows with missing CGM target")

    df = df.sort_values(["participant_id", "ts"]).reset_index(drop=True)
    df["ds"] = df.groupby("participant_id").cumcount().astype("int64")

    assert df["cgm_glucose"].isna().sum() == 0, "CGM still has NaN after dropna — aborting"

    extra = [c for c in raw_static_cols if c in df.columns and c not in FINAL_COLS]
    return df[FINAL_COLS + extra].copy()


def validate_outputs(train, test, final_static_cols):
    """Print diagnostics; abort if any static column has NaN."""
    print("\n=== Validation ===")
    print(f"  Participants  train : {train['participant_id'].nunique():,}")
    print(f"  Participants  test  : {test['participant_id'].nunique():,}")
    print(f"  Rows          train : {len(train):,}")
    print(f"  Rows          test  : {len(test):,}")
    print(f"  Dynamic features    : {len(FINAL_COLS)}")
    print(f"  Static features     : {len(final_static_cols)}")
    print(f"  Static col names    : {final_static_cols}")

    for split_name, sdf in [("train", train), ("test", test)]:
        nan_static = sdf[final_static_cols].isna().sum().sum()
        if nan_static > 0:
            bad = sdf[final_static_cols].isna().sum()
            raise ValueError(f"NaN in {split_name} static columns after encoding:\n{bad[bad>0]}")
        nan_dyn = sdf[FINAL_COLS].isna().sum().sum()
        if nan_dyn > 0:
            print(f"  [WARN] {nan_dyn} NaN in {split_name} dynamic cols (wearable gaps expected)")

    print("  No NaNs in static columns — clean")
    print(f"  Total columns : {len(train.columns)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("[1/6] Loading parquet and filtering to retained cohort...")
    df = load_and_filter(PARQUET, COHORT_CSV)

    print("[2/6] Renaming columns to Shakson's format...")
    df = rename_columns(df)

    print("[3/6] Building sleep_stage...")
    df = build_sleep_stage(df)

    print("[4/6] Building timestamps...")
    df = build_timestamps(df)

    print("[5/6] Computing time features...")
    df = build_time_features(df)

    print("[6/6] Computing CGM lag / rolling features + wearable imputation...")
    df = build_cgm_features(df)

    # ── Discover available static columns ──────────────────────────────────────
    print("\n[Static] Discovering available static columns in parquet...")
    avail_num,  _ = discover_available_cols(df, STATIC_NUMERIC)
    avail_bin,  _ = discover_available_cols(df, STATIC_BINARY)
    avail_cat,  _ = discover_available_cols(df, STATIC_CATEGORICAL)
    all_raw_static = avail_num + avail_bin + avail_cat
    print(f"  Numeric    : {len(avail_num)}/{len(STATIC_NUMERIC)} found")
    print(f"  Binary     : {len(avail_bin)}/{len(STATIC_BINARY)} found")
    print(f"  Categorical: {len(avail_cat)}/{len(STATIC_CATEGORICAL)} found")

    # ── Finalize dynamic pipeline + carry raw static cols ─────────────────────
    print("\n[Finalize] Dropping missing CGM rows, recomputing ds...")
    df = finalize_enriched(df, raw_static_cols=all_raw_static)

    # ── Train / test split (same logic as Exp A) ──────────────────────────────
    train, test = train_test_split(df)
    print(f"\n[Split] Train: {len(train):,} rows | Test: {len(test):,} rows")

    # Missingness before encoding
    miss_before = sum(
        train[c].isna().sum() for c in all_raw_static if c in train.columns
    )
    print(f"\n[Encode] Missingness before encoding: {miss_before:,} NaN cells in static cols")

    # ── Encode static features (fit on train only, apply to both) ─────────────
    print("[Encode] Fitting encoders on train set...")
    train, metadata, final_static_cols = encode_static_features(
        train, avail_num, avail_bin, avail_cat, is_train=True
    )
    print("[Encode] Applying encoders to test set...")
    test, _, _ = encode_static_features(
        test, avail_num, avail_bin, avail_cat, is_train=False, metadata=metadata
    )
    miss_after = sum(
        train[c].isna().sum() for c in final_static_cols if c in train.columns
    )
    print(f"[Encode] Missingness after  encoding: {miss_after:,} NaN cells in static cols")

    # ── Validate ───────────────────────────────────────────────────────────────
    validate_outputs(train, test, final_static_cols)

    # ── ds step check (must be 1 everywhere) ──────────────────────────────────
    step_ok = (
        train.sort_values(["participant_id", "ds"])
             .groupby("participant_id")["ds"]
             .diff()
             .dropna()
             == 1
    ).all()
    print(f"  ds step=1 check : {step_ok}")

    # ── Save ───────────────────────────────────────────────────────────────────
    train_path = OUT_DIR / "train_timeseries_static.feather"
    test_path  = OUT_DIR / "test_timeseries_static.feather"
    feat_path  = OUT_DIR / "static_feature_list.json"
    meta_path  = OUT_DIR / "feature_metadata.json"

    train.reset_index(drop=True).to_feather(train_path)
    test.reset_index(drop=True).to_feather(test_path)

    with open(feat_path, "w") as f:
        json.dump(final_static_cols, f, indent=2)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Done ===")
    print(f"Train  → {train_path}  ({len(train):,} rows)")
    print(f"Test   → {test_path}   ({len(test):,} rows)")
    print(f"Static feature list ({len(final_static_cols)} cols) → {feat_path}")
    print(f"Feature metadata → {meta_path}")


if __name__ == "__main__":
    main()
