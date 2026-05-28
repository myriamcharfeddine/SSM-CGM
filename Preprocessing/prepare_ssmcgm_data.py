"""
prepare_ssmcgm_data.py
======================
Transforms the enriched multimodal parquet into the exact 29-column feather
format used by Shakson Isaac's SSM-CGM model (mamba288.py / SSM_CGM.py).

Input:
  Data/enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet
  Data/enriched_multimodal/cohort.csv   (1,591 retained participants)

Output:
  Data/ssmcgm_ready/train_timeseries.feather
  Data/ssmcgm_ready/test_timeseries.feather

Column mapping to Shakson's format:
  cgm_glucose_mean          -> cgm_glucose
  heart_rate_mean           -> heartrate
  respiratory_rate_mean     -> respiration_rate
  oxygen_saturation_mean    -> oxygen_saturation
  stress_level_mean         -> stress_level
  calories_total            -> calories_value
  activity_steps_per_min*5  -> activity_steps
  participants_study_group  -> study_group
  participants_clinical_site-> clinical_site
  participants_age          -> age
  sleep_stage_{awake,...}   -> sleep_stage (dominant category)
  timestamp_local (UTC)     -> ts
  timestamp_local (naive)   -> real_time
  predmeal_flag             =  0.0   (no meal annotations in AI-READI)
  ds                        = per-participant cumcount (consecutive integer)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path("/home/myriamcharfeddine/CGM")
PARQUET    = ROOT / "Data/enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet"
COHORT_CSV = ROOT / "Data/enriched_multimodal/cohort.csv"
OUT_DIR    = ROOT / "Data/ssmcgm_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Shakson's train feather column order (29 cols) — must match exactly
FINAL_COLS = [
    "participant_id", "ts", "real_time",
    "minute_of_day", "tod_cos", "tod_sin",
    "cgm_glucose", "activity_steps", "calories_value",
    "heartrate", "oxygen_saturation", "respiration_rate",
    "sleep_stage", "stress_level",
    "ds", "clinical_site", "study_group", "age",
    "cgm_lag_1", "cgm_lag_3", "cgm_lag_6",
    "cgm_diff_lag_1", "cgm_diff_lag_3", "cgm_diff_lag_6",
    "cgm_lagdiff_1_3", "cgm_lagdiff_3_6",
    "cgm_rolling_mean", "cgm_rolling_std",
    "predmeal_flag",
]

# Horizon = 12 steps (1 hour at 5-min intervals) — reserved for test set
HORIZON = 12


def load_and_filter(parquet_path, cohort_path):
    print("[1/6] Loading parquet and filtering to retained cohort...")
    cohort = pd.read_csv(cohort_path)
    retained = set(cohort["participant_id"].astype(str))

    df = pd.read_parquet(parquet_path)
    df["participant_id"] = df["participant_id"].astype(str)
    df = df[df["participant_id"].isin(retained)].copy()
    print(f"      {df['participant_id'].nunique()} participants, {len(df):,} rows")
    return df


def rename_columns(df):
    print("[2/6] Renaming columns to Shakson's format...")
    df = df.rename(columns={
        "cgm_glucose_mean":          "cgm_glucose",
        "heart_rate_mean":           "heartrate",
        "respiratory_rate_mean":     "respiration_rate",
        "oxygen_saturation_mean":    "oxygen_saturation",
        "stress_level_mean":         "stress_level",
        "calories_total":            "calories_value",
        "participants_study_group":  "study_group",
        "participants_clinical_site":"clinical_site",
        "participants_age":          "age",
    })

    # activity_steps = steps per 5-min bin
    df["activity_steps"] = df["activity_steps_per_min"].fillna(0.0) * 5.0

    # predmeal_flag = 0 (AI-READI has no meal annotations)
    df["predmeal_flag"] = 0.0

    return df


def build_sleep_stage(df):
    print("[3/6] Building sleep_stage categorical from binary columns...")
    sleep_map = {
        "sleep_stage_awake":   "awake",
        "sleep_stage_light":   "light",
        "sleep_stage_deep":    "deep",
        "sleep_stage_rem":     "rem",
        "sleep_stage_unknown": "unknown",
    }
    avail = {k: v for k, v in sleep_map.items() if k in df.columns}
    if avail:
        sdf = df[list(avail.keys())].fillna(0.0)
        sdf.columns = list(avail.values())
        df["sleep_stage"] = sdf.idxmax(axis=1)
        # When all zeros (wearable not worn), label as unknown
        df.loc[sdf.sum(axis=1) == 0, "sleep_stage"] = "unknown"
    else:
        df["sleep_stage"] = "unknown"
    return df


def build_timestamps(df):
    print("[4/6] Building ts (UTC) and real_time (local naive) columns...")
    # timestamp_local is timezone-aware (e.g. America/Los_Angeles)
    ts_aware = pd.to_datetime(df["timestamp_local"])

    # ts: convert to UTC, keep tz-aware (matches Shakson's dtype)
    df["ts"] = ts_aware.dt.tz_convert("UTC")

    # real_time: local time without timezone (matches Shakson's dtype)
    df["real_time"] = ts_aware.dt.tz_localize(None)

    return df


def build_time_features(df):
    print("[5/6] Computing time features and ds index...")
    rt = df["real_time"]  # local naive datetime

    seconds_of_day = rt.dt.hour * 3600 + rt.dt.minute * 60 + rt.dt.second
    df["minute_of_day"] = (seconds_of_day // 60).astype("int64")
    angle = 2 * np.pi * seconds_of_day / 86400
    df["tod_sin"] = np.sin(angle)
    df["tod_cos"] = np.cos(angle)

    # ds: consecutive integer per participant, sorted by timestamp
    df = df.sort_values(["participant_id", "ts"]).reset_index(drop=True)
    df["ds"] = df.groupby("participant_id").cumcount().astype("int64")

    return df


def build_cgm_features(df):
    print("[6/6] Computing CGM lag / rolling features...")
    g = df.groupby("participant_id")["cgm_glucose"]

    df["cgm_lag_1"] = g.shift(1)
    df["cgm_lag_3"] = g.shift(3)
    df["cgm_lag_6"] = g.shift(6)

    df["cgm_diff_lag_1"] = df["cgm_glucose"] - df["cgm_lag_1"]
    df["cgm_diff_lag_3"] = df["cgm_glucose"] - df["cgm_lag_3"]
    df["cgm_diff_lag_6"] = df["cgm_glucose"] - df["cgm_lag_6"]

    df["cgm_lagdiff_1_3"] = df["cgm_lag_1"] - df["cgm_lag_3"]
    df["cgm_lagdiff_3_6"] = df["cgm_lag_3"] - df["cgm_lag_6"]

    df["cgm_rolling_mean"] = g.transform(
        lambda x: x.rolling(12, min_periods=1).mean()
    )
    df["cgm_rolling_std"] = g.transform(
        lambda x: x.rolling(12, min_periods=1).std().fillna(0.0)
    )

    # Fill lag NaNs (first rows of each participant) with current CGM value
    for col in ["cgm_lag_1", "cgm_lag_3", "cgm_lag_6"]:
        df[col] = df[col].fillna(df["cgm_glucose"])
    for col in ["cgm_diff_lag_1", "cgm_diff_lag_3", "cgm_diff_lag_6",
                "cgm_lagdiff_1_3", "cgm_lagdiff_3_6"]:
        df[col] = df[col].fillna(0.0)

    # Impute missing wearable signals:
    #   1st pass — per-participant median (handles sporadic missingness)
    #   2nd pass — global median (handles participants with all-NaN, i.e. device never worn)
    for col in ["heartrate", "respiration_rate", "oxygen_saturation",
                "stress_level", "activity_steps", "calories_value"]:
        if col in df.columns:
            medians = df.groupby("participant_id")[col].transform("median")
            df[col] = df[col].fillna(medians)
            global_median = df[col].median()
            df[col] = df[col].fillna(global_median)

    return df


def train_test_split(df, horizon=HORIZON):
    """
    Test set  = last `horizon` rows per participant (the 1-hour forecast target).
    Train set = everything else.
    Matches Shakson's split: train (n-12), test (12) per participant.
    """
    df["_max_ds"] = df.groupby("participant_id")["ds"].transform("max")
    train = df[df["ds"] <= df["_max_ds"] - horizon].drop(columns=["_max_ds"])
    test  = df[df["ds"] >  df["_max_ds"] - horizon].drop(columns=["_max_ds"])
    return train, test


def finalize(df):
    """Drop rows with missing CGM, recompute consecutive ds, keep Shakson's 29-col order."""
    n_before = len(df)
    df = df.dropna(subset=["cgm_glucose"]).reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped:
        print(f"      Dropped {dropped:,} rows with missing CGM target")
    # Recompute ds per participant so it remains consecutive after the drop
    df = df.sort_values(["participant_id", "ts"]).reset_index(drop=True)
    df["ds"] = df.groupby("participant_id").cumcount().astype("int64")
    return df[FINAL_COLS].copy()


def main():
    df = load_and_filter(PARQUET, COHORT_CSV)
    df = rename_columns(df)
    df = build_sleep_stage(df)
    df = build_timestamps(df)
    df = build_time_features(df)
    df = build_cgm_features(df)

    # Drop rows with missing CGM before splitting
    df = finalize(df)

    train, test = train_test_split(df)

    train_path = OUT_DIR / "train_timeseries.feather"
    test_path  = OUT_DIR / "test_timeseries.feather"

    train.reset_index(drop=True).to_feather(train_path)
    test.reset_index(drop=True).to_feather(test_path)

    print()
    print("=== Done ===")
    print(f"Train : {train['participant_id'].nunique()} participants | {len(train):,} rows -> {train_path}")
    print(f"Test  : {test['participant_id'].nunique()} participants  | {len(test):,} rows  -> {test_path}")
    print(f"Columns ({len(train.columns)}): {list(train.columns)}")

    # Quick sanity check against Shakson's feather
    print()
    print("=== Sanity check ===")
    ref = pd.read_feather("/home/shaksonisaac/Data/train_finaltimeseries_meal_upd.feather")
    missing = [c for c in ref.columns if c not in train.columns]
    extra   = [c for c in train.columns if c not in ref.columns]
    print(f"  Missing cols vs Shakson: {missing or 'none'}")
    print(f"  Extra cols vs Shakson  : {extra   or 'none'}")
    print(f"  study_group values : {sorted(train.study_group.unique())}")
    print(f"  sleep_stage values : {sorted(train.sleep_stage.unique())}")
    print(f"  ds range (train)   : {train.ds.min()} – {train.ds.max()}")
    print(f"  ds step check (=1) : {(train.sort_values(['participant_id','ds']).groupby('participant_id')['ds'].diff().dropna() == 1).all()}")


if __name__ == "__main__":
    main()
