"""
prepare_ssmcgm_data_experiment_c.py
=====================================
Experiment C — participant-level split with dynamic + static features.

Differences from Experiment B (prepare_ssmcgm_data_enriched.py):
  - Uses participant-level split from create_experiment_c_split.py output
    (split_participants.csv) instead of Shakson's last-12-rows test split.
  - Outputs THREE feathers: train / val / test by participant.
  - Static encoding is fit on train participants only (no leakage).
  - Outputs go to Data/ssmcgm_ready_exp_C/

Usage:
  python Preprocess/prepare_ssmcgm_data_experiment_c.py
  python Preprocess/prepare_ssmcgm_data_experiment_c.py \
      --split-dir Data/experiment_c_split_adapt6h_seed42 \
      --out-dir   Data/ssmcgm_ready_exp_C_adapt6h
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Import shared dynamic pipeline from Experiment A ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from prepare_ssmcgm_data import (
    load_and_filter,
    rename_columns,
    build_sleep_stage,
    build_timestamps,
    build_time_features,
    build_cgm_features,
    PARQUET,
    COHORT_CSV,
    FINAL_COLS,
)
from prepare_ssmcgm_data_enriched import (
    discover_available_cols,
    encode_static_features,
    finalize_enriched,
    STATIC_NUMERIC,
    STATIC_BINARY,
    STATIC_CATEGORICAL,
)

# ── Default paths ──────────────────────────────────────────────────────────────
ROOT         = Path("/home/myriamcharfeddine/CGM")
DEFAULT_SPLIT = ROOT / "Data/experiment_c_split_adapt48h_seed42"
DEFAULT_OUT   = ROOT / "Data/ssmcgm_ready_exp_C"


def parse_args():
    p = argparse.ArgumentParser(description="Experiment C data prep — participant-level split")
    p.add_argument(
        "--split-dir", type=Path, default=DEFAULT_SPLIT,
        help="Directory produced by create_experiment_c_split.py "
             "(contains split_participants.csv). Default: adapt48h_seed42",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT,
        help="Output directory for feathers and metadata",
    )
    return p.parse_args()


# ── Step 0: load split assignments ────────────────────────────────────────────

def load_split_assignments(split_dir: Path) -> pd.DataFrame:
    """Load participant-level split assignments from Experiment C output."""
    csv = split_dir / "split_participants.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"split_participants.csv not found: {csv}\n"
            "Run create_experiment_c_split.py first."
        )
    split_df = pd.read_csv(csv)
    split_df["participant_id"] = split_df["participant_id"].astype(str)

    counts = split_df["split"].value_counts()
    print("  Split assignments loaded:")
    for s, n in counts.items():
        print(f"    {s:<10} {n:>6,} participants")
    return split_df


# ── Step 7: participant-level partition ───────────────────────────────────────

def split_by_participant(df: pd.DataFrame, split_df: pd.DataFrame):
    """
    Partition the full time-series dataframe by participant split assignment.
    All rows for a participant go to their assigned split (train / val / test).
    Participants not present in split_df are excluded (logged as warning).

    ds is per-participant cumcount so it stays valid within each split — no
    recomputation needed after partitioning.
    """
    pid_to_split = split_df.set_index("participant_id")["split"].to_dict()
    df = df.copy()
    df["_split"] = df["participant_id"].map(pid_to_split).fillna("excluded")

    n_excluded_pids = df[df["_split"] == "excluded"]["participant_id"].nunique()
    if n_excluded_pids > 0:
        print(f"  [WARN] {n_excluded_pids} participants from parquet not in split — excluded")

    train_df = df[df["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    val_df   = df[df["_split"] == "val"  ].drop(columns=["_split"]).reset_index(drop=True)
    test_df  = df[df["_split"] == "test" ].drop(columns=["_split"]).reset_index(drop=True)
    return train_df, val_df, test_df


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_three_way(train, val, test, final_static_cols):
    print("\n=== Validation ===")
    for name, sdf in [("train", train), ("val", val), ("test", test)]:
        n_pids = sdf["participant_id"].nunique()
        n_rows = len(sdf)
        print(f"  {name:<5} : {n_pids:>5,} participants | {n_rows:>10,} rows")

        nan_static = sdf[final_static_cols].isna().sum().sum()
        if nan_static > 0:
            bad = sdf[final_static_cols].isna().sum()
            raise ValueError(f"NaN in {name} static columns after encoding:\n{bad[bad > 0]}")

        ds_ok = (
            sdf.sort_values(["participant_id", "ds"])
               .groupby("participant_id")["ds"]
               .diff()
               .dropna()
               == 1
        ).all()
        status = "OK" if ds_ok else "BROKEN"
        print(f"         ds step=1 : {status}")
        if not ds_ok:
            raise AssertionError(f"ds is not consecutive (step≠1) in {name} split")

    print(f"  Static cols ({len(final_static_cols)}) : no NaN, no Inf")
    print(f"  Dynamic cols ({len(FINAL_COLS)})        : schema matches Exp A")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n========================================================")
    print(" Experiment C Data Prep")
    print(f" Split dir : {args.split_dir}")
    print(f" Output    : {args.out_dir}")
    print("========================================================\n")

    # ── 0. Split assignments ──────────────────────────────────────────────────
    print("[0/7] Loading participant-level split assignments...")
    split_df = load_split_assignments(args.split_dir)

    # ── 1–6. Dynamic pipeline (shared with Exp A / B) ─────────────────────────
    print("[1/7] Loading parquet and filtering to retained cohort...")
    df = load_and_filter(PARQUET, COHORT_CSV)

    print("[2/7] Renaming columns to Shakson's format...")
    df = rename_columns(df)

    print("[3/7] Building sleep_stage...")
    df = build_sleep_stage(df)

    print("[4/7] Building timestamps...")
    df = build_timestamps(df)

    print("[5/7] Computing time features...")
    df = build_time_features(df)

    print("[6/7] Computing CGM lag / rolling features + wearable imputation...")
    df = build_cgm_features(df)

    # ── Discover static columns ────────────────────────────────────────────────
    print("\n[Static] Discovering available static columns in parquet...")
    avail_num, _ = discover_available_cols(df, STATIC_NUMERIC)
    avail_bin, _ = discover_available_cols(df, STATIC_BINARY)
    avail_cat, _ = discover_available_cols(df, STATIC_CATEGORICAL)
    all_raw_static = avail_num + avail_bin + avail_cat
    print(f"  Numeric    : {len(avail_num)}/{len(STATIC_NUMERIC)} found")
    print(f"  Binary     : {len(avail_bin)}/{len(STATIC_BINARY)} found")
    print(f"  Categorical: {len(avail_cat)}/{len(STATIC_CATEGORICAL)} found")

    # ── Finalize dynamic pipeline ──────────────────────────────────────────────
    print("\n[Finalize] Dropping missing CGM rows, recomputing ds...")
    df = finalize_enriched(df, raw_static_cols=all_raw_static)

    # ── 7. Participant-level split ─────────────────────────────────────────────
    print("[7/7] Partitioning by participant split assignment...")
    train_df, val_df, test_df = split_by_participant(df, split_df)
    del df

    print(f"  Train : {train_df['participant_id'].nunique():,} pids | {len(train_df):,} rows")
    print(f"  Val   : {val_df['participant_id'].nunique():,} pids | {len(val_df):,} rows")
    print(f"  Test  : {test_df['participant_id'].nunique():,} pids | {len(test_df):,} rows")

    # ── Encode static features (fit on train only) ─────────────────────────────
    miss_before = sum(train_df[c].isna().sum() for c in all_raw_static if c in train_df.columns)
    print(f"\n[Encode] Missingness before encoding: {miss_before:,} NaN cells in static cols")

    print("[Encode] Fitting encoders on train set (no leakage into val/test)...")
    train_df, metadata, final_static_cols = encode_static_features(
        train_df, avail_num, avail_bin, avail_cat, is_train=True
    )
    print("[Encode] Applying encoders to val set...")
    val_df,  _, _ = encode_static_features(
        val_df,  avail_num, avail_bin, avail_cat, is_train=False, metadata=metadata
    )
    print("[Encode] Applying encoders to test set...")
    test_df, _, _ = encode_static_features(
        test_df, avail_num, avail_bin, avail_cat, is_train=False, metadata=metadata
    )

    miss_after = sum(train_df[c].isna().sum() for c in final_static_cols if c in train_df.columns)
    print(f"[Encode] Missingness after  encoding: {miss_after:,} NaN cells in static cols")

    # ── Validate ───────────────────────────────────────────────────────────────
    validate_three_way(train_df, val_df, test_df, final_static_cols)

    # ── Save ───────────────────────────────────────────────────────────────────
    out = args.out_dir

    train_path = out / "train_timeseries_static.feather"
    val_path   = out / "val_timeseries_static.feather"
    test_path  = out / "test_timeseries_static.feather"
    feat_path  = out / "static_feature_list.json"
    meta_path  = out / "feature_metadata.json"

    train_df.reset_index(drop=True).to_feather(train_path)
    val_df.reset_index(drop=True).to_feather(val_path)
    test_df.reset_index(drop=True).to_feather(test_path)

    with open(feat_path, "w") as f:
        json.dump(final_static_cols, f, indent=2)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Copy split metadata for reference
    meta_src = args.split_dir / "experiment_c_split_metadata.json"
    if meta_src.exists():
        shutil.copy2(meta_src, out / "experiment_c_split_metadata_ref.json")

    print("\n=== Done ===")
    print(f"Train  → {train_path}  ({len(train_df):,} rows)")
    print(f"Val    → {val_path}    ({len(val_df):,} rows)")
    print(f"Test   → {test_path}   ({len(test_df):,} rows)")
    print(f"Static feature list ({len(final_static_cols)} cols) → {feat_path}")
    print(f"Feature metadata                                     → {meta_path}")


if __name__ == "__main__":
    main()
