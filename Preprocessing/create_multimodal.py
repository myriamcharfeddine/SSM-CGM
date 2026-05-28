#!/usr/bin/env python3
"""SSMCGM multimodal dataset builder (AIREADI Option B).

Reads exported per-modality Parquet/Feather files (one file per modality) produced by
`export_modalities_to_gcs.py` (default location: `gs://cgmproject2025/ai-ready/raw`).

Outputs:
- A merged, feature-engineered multimodal dataset written locally to:
  `/home/shaksonisaac/CGM/SSMCGM/Data`
- Optionally uploads the finalized dataset to:
  `gs://cgmproject2025/ai-ready/data`

This script intentionally no longer depends on the legacy
`ai-ready/data/*.feather` layout.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import traceback
from collections import Counter

# Keep behavior stable with prior pipeline defaults.
MIN_VALID_DATE = pd.Timestamp("2020-01-01", tz="UTC")
MAX_VALID_DATE = pd.Timestamp("2025-12-31", tz="UTC")

DEFAULT_RAW_ROOT = "gs://cgmproject2025/ai-ready/raw"
DEFAULT_OUTPUT_DIR = "/home/myriamcharfeddine/CGM/Data"
DEFAULT_UPLOAD_ROOT = "gs://cgmproject2025/ai-ready/data"

MANIFEST_GLUCOSE = "gs://cgmproject2025/AIREADI/wearable_blood_glucose/manifest.tsv"
MANIFEST_ACTIVITY = "gs://cgmproject2025/AIREADI/wearable_activity_monitor/manifest.tsv"
DEFAULT_MANIFEST_BUCKET = "cgmproject2025"

# Exporter log (optional diagnostics)
DEFAULT_EXPORT_LOG = "gs://cgmproject2025/ai-ready/raw/logs/load_log.parquet"

# --- IO helpers (local + GCS via gcsfs) ---

def _is_gcs_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith("gs://")


def _read_parquet_any(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read parquet from local disk or GCS."""
    if _is_gcs_path(path):
        import gcsfs

        fs = gcsfs.GCSFileSystem(cache_timeout=3600)
        with fs.open(path, "rb") as f:
            return pd.read_parquet(f, columns=columns)
    return pd.read_parquet(path, columns=columns)


def _write_parquet_any(df: pd.DataFrame, path: str) -> None:
    """Write parquet to local disk or GCS."""
    if _is_gcs_path(path):
        import gcsfs

        fs = gcsfs.GCSFileSystem(cache_timeout=3600)
        with fs.open(path, "wb") as f:
            df.to_parquet(f, index=True)
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


def _as_gs_uri_with_bucket(path: str, default_bucket: str = DEFAULT_MANIFEST_BUCKET) -> str:
    if path is None:
        return path
    s = str(path).strip()
    if not s:
        return s
    if s.startswith("gs://"):
        return s
    if "/" in s and not s.startswith("/"):
        first = s.split("/", 1)[0]
        if first == default_bucket:
            return f"gs://{s}"
        if not s.startswith("AIREADI/") and not s.startswith("ai-ready/") and not s.startswith("raw/"):
            return f"gs://{s}"
    return f"gs://{default_bucket}/{s.lstrip('/')}"


def _read_tsv_any(path: str) -> pd.DataFrame:
    if _is_gcs_path(path):
        import gcsfs

        fs = gcsfs.GCSFileSystem(cache_timeout=3600)
        with fs.open(path, "rb") as f:
            return pd.read_csv(f, sep="\t")
    return pd.read_csv(path, sep="\t")


# --- feature engineering helpers (ported from prior script, simplified) ---

def clean_timestamps(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    clean: Dict[str, pd.DataFrame] = {}
    total_removed = 0

    for name, df in tqdm(list(data_dict.items()), desc="Cleaning timestamps", unit="dataset"):
        if df is None or len(df) == 0:
            clean[name] = df
            continue

        # ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            elif "start_time" in df.columns:
                df = df.set_index("start_time")
            elif "time" in df.columns:
                # calories export uses 'time' (see parse_kcal_payload in load.py)
                df = df.set_index("time")
            else:
                raise ValueError(
                    f"Dataset '{name}' has no DatetimeIndex and no 'timestamp', 'start_time', or 'time' column"
                )

        # Ensure datetime index values
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)

        # Drop rows with unparseable index
        if df.index.hasnans:
            df = df.loc[~df.index.isna()].copy()

        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
            df = df.copy()
            df.index = idx

        invalid = (df.index < MIN_VALID_DATE) | (df.index > MAX_VALID_DATE)
        if "end_time" in df.columns:
            end_t = pd.to_datetime(df["end_time"], errors="coerce")
            if isinstance(end_t, pd.Series) and getattr(end_t.dt, "tz", None) is None:
                end_t = end_t.dt.tz_localize("UTC")
            invalid = invalid | (end_t < MIN_VALID_DATE) | (end_t > MAX_VALID_DATE)

        clean_df = df.loc[~invalid].copy()
        removed = len(df) - len(clean_df)
        total_removed += removed
        clean[name] = clean_df

    tqdm.write(f"Removed {total_removed:,} invalid timestamped records")
    return clean


def find_common_participants(data_dict: Dict[str, pd.DataFrame], min_datasets: int) -> List[str]:
    """Return participant IDs present in at least `min_datasets` modalities.

    Optimized vs the previous implementation by avoiding nested loops over modality sets.
    """
    pid_counts: Dict[str, int] = {}
    for _, df in data_dict.items():
        if df is None or len(df) == 0 or "participant_id" not in df.columns:
            continue
        for pid in pd.Index(df["participant_id"].astype(str).unique()).tolist():
            pid_counts[pid] = pid_counts.get(pid, 0) + 1
    return [pid for pid, c in pid_counts.items() if int(c) >= int(min_datasets)]


def create_time_grid(start_time: pd.Timestamp, end_time: pd.Timestamp, freq: str = "5min") -> pd.DatetimeIndex:
    return pd.date_range(start=start_time, end=end_time, freq=freq)


def calculate_hrv_features(hr_values: np.ndarray) -> Dict[str, float]:
    if len(hr_values) < 2:
        return {"rmssd": np.nan, "pnn50": np.nan}
    valid = hr_values[(hr_values > 30) & (hr_values < 200)]
    if len(valid) < 2:
        return {"rmssd": np.nan, "pnn50": np.nan}
    rr = 60000 / valid
    diff = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff**2)) if len(diff) else np.nan
    pnn50 = (np.sum(np.abs(diff) > 50) / len(diff) * 100) if len(diff) else np.nan
    return {"rmssd": rmssd, "pnn50": pnn50}


# --- interval -> bin explosion helpers (sleep/activity) ---

def _parse_freq_to_minutes(freq: str) -> float:
    td = pd.to_timedelta(freq)
    return float(td.total_seconds() / 60.0)


def _explode_intervals_to_bins(
    df: pd.DataFrame,
    *,
    start_col: str = "start_time",
    end_col: str = "end_time",
    freq: str,
    max_bin_span: int = 5000,
) -> pd.DataFrame:
    """Explode interval rows into per-bin overlap minutes.

    Returns a dataframe with columns: ["timestamp", "row_id", "overlap_min"] plus all
    original df columns (except we keep them as-is).

    max_bin_span is a safety guard against pathological intervals.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["timestamp", "row_id", "overlap_min"])

    d = df.copy()

    # Interval starts are typically the index for your exports (sleep/activity).
    if start_col in d.columns:
        starts = pd.to_datetime(d[start_col], errors="coerce", utc=True)
    else:
        # Ensure starts is a Series aligned to rows (DatetimeIndex has no .loc)
        starts = pd.Series(pd.to_datetime(d.index, errors="coerce", utc=True), index=d.index, name=start_col)
        # preserve the original start explicitly so downstream code can compute durations safely
        d = d.copy()
        d[start_col] = starts

    if not isinstance(starts, pd.Series):
        starts = pd.Series(starts, index=d.index, name=start_col)

    # ends as Series aligned to rows
    if end_col in d.columns:
        ends = pd.to_datetime(d[end_col], errors="coerce", utc=True)
    else:
        ends = pd.Series(pd.NaT, index=d.index, name=end_col)

    if not isinstance(ends, pd.Series):
        ends = pd.Series(ends, index=d.index, name=end_col)

    mask = (~starts.isna()) & (~ends.isna()) & (ends > starts)
    if not bool(mask.any()):
        return pd.DataFrame(columns=["timestamp", "row_id", "overlap_min"])

    d = d.loc[mask].copy()
    starts = starts.loc[mask]
    ends = ends.loc[mask]

    bin_td = pd.to_timedelta(freq)

    # Determine first/last bin touched per interval
    first_bin = starts.dt.floor(freq)
    # end-exclusive: subtract 1ns so exact end at boundary doesn't include next bin
    last_bin = (ends - pd.Timedelta(nanoseconds=1)).dt.floor(freq)

    rows: List[Dict[str, object]] = []
    for i, (s, e, b0, b1) in enumerate(zip(starts.to_list(), ends.to_list(), first_bin.to_list(), last_bin.to_list())):
        if pd.isna(s) or pd.isna(e) or pd.isna(b0) or pd.isna(b1):
            continue
        if b1 < b0:
            continue

        # bins touched by this interval
        bins = pd.date_range(b0, b1, freq=freq, tz="UTC")
        if len(bins) > int(max_bin_span):
            bins = bins[: int(max_bin_span)]

        for b in bins:
            b_end = b + bin_td
            ov_start = max(s, b)
            ov_end = min(e, b_end)
            ov_min = (ov_end - ov_start).total_seconds() / 60.0
            if ov_min <= 0:
                continue
            rows.append({
                "timestamp": b,
                "row_id": int(i),
                "overlap_min": float(ov_min),
            })

    if not rows:
        return pd.DataFrame(columns=["timestamp", "row_id", "overlap_min"])

    exploded = pd.DataFrame(rows)
    # attach original fields via row_id positional join
    exploded = exploded.join(d.reset_index(drop=True), on="row_id", how="left")
    return exploded


def _calorie_features_from_cumulative(
    calorie_df: pd.DataFrame,
    *,
    freq: str,
    max_kcal_per_min: float = 50.0,
) -> pd.DataFrame:
    """Convert Garmin cumulative active-calorie readings into per-bin features.

    Garmin ``kcal_burned`` is a daily cumulative counter, so naive resampling
    (summing raw values) double-counts the same calories and produces impossible
    rates. Correct approach:
      1. split by local day (counter resets daily);
      2. enforce monotonicity within each local day (ignore transient drops);
      3. take differences between consecutive cumulative readings;
      4. spread each calorie increment uniformly over the elapsed interval;
      5. sum corrected increments into the requested time bins.
    """
    out_cols = ["calories_total", "calories_per_min"]
    bin_minutes = _parse_freq_to_minutes(freq)

    if calorie_df is None or len(calorie_df) == 0 or "value" not in calorie_df.columns:
        return pd.DataFrame(columns=out_cols)

    d = pd.DataFrame({
        "timestamp": pd.to_datetime(calorie_df.index, errors="coerce", utc=True),
        "value": pd.to_numeric(calorie_df["value"], errors="coerce"),
    })
    d["timezone"] = calorie_df["timezone"].to_numpy() if "timezone" in calorie_df.columns else None
    d = d.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if len(d) < 2:
        return pd.DataFrame(columns=out_cols)

    d["local_date"] = d["timestamp"].dt.date
    for tz_name, rows in d.groupby(d["timezone"].astype("string"), dropna=True, sort=False).groups.items():
        tz_str = "" if tz_name is None else str(tz_name)
        if tz_str.strip().lower() in {"", "none", "nan", "<na>"}:
            continue
        try:
            d.loc[rows, "local_date"] = pd.DatetimeIndex(d.loc[rows, "timestamp"]).tz_convert(tz_str).date
        except Exception:
            continue

    # If duplicated timestamps exist, keep the highest cumulative value.
    d = (
        d.groupby(["local_date", "timestamp"], as_index=False, sort=True)["value"]
        .max()
        .sort_values(["local_date", "timestamp"])
    )

    d["cum_clean"] = d.groupby("local_date", sort=False)["value"].cummax()
    d["prev_timestamp"] = d.groupby("local_date", sort=False)["timestamp"].shift()
    d["prev_cum"] = d.groupby("local_date", sort=False)["cum_clean"].shift()
    d["duration_min"] = (d["timestamp"] - d["prev_timestamp"]).dt.total_seconds() / 60.0
    d["calories_delta"] = (d["cum_clean"] - d["prev_cum"]).clip(lower=0.0)
    d["calories_rate"] = d["calories_delta"] / d["duration_min"]

    intervals = d[
        (d["duration_min"] > 0)
        & (d["calories_delta"] > 0)
        & (d["calories_rate"] >= 0)
        & (d["calories_rate"] <= float(max_kcal_per_min))
    ].copy()
    if len(intervals) == 0:
        return pd.DataFrame(columns=out_cols)

    intervals = intervals.rename(columns={"prev_timestamp": "start_time", "timestamp": "end_time"})
    ex = _explode_intervals_to_bins(intervals, start_col="start_time", end_col="end_time", freq=freq)
    if ex is None or len(ex) == 0:
        return pd.DataFrame(columns=out_cols)

    ex["calories_in_overlap"] = ex["calories_rate"].astype(float) * ex["overlap_min"].astype(float)
    total = ex.groupby("timestamp")["calories_in_overlap"].sum().astype(float)

    out = pd.DataFrame(index=total.index)
    out["calories_total"] = total
    out["calories_per_min"] = total / float(bin_minutes) if bin_minutes > 0 else 0.0
    return out.reindex(columns=out_cols)


def _sleep_features_from_intervals(sleep_df: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    """Compute sleep per-bin features without per-window loops.

    Outputs the same columns as process_sleep_data() but as a dataframe indexed by bin start:
      sleep_stage_{awake,light,deep,rem,unknown}, sleep_transitions, sleep_efficiency
    """
    bin_minutes = _parse_freq_to_minutes(freq)

    out_cols = [
        "sleep_stage_awake",
        "sleep_stage_light",
        "sleep_stage_deep",
        "sleep_stage_rem",
        "sleep_stage_unknown",
        "sleep_transitions",
        "sleep_efficiency",
    ]

    if sleep_df is None or len(sleep_df) == 0 or "end_time" not in sleep_df.columns:
        return pd.DataFrame(columns=out_cols)

    ex = _explode_intervals_to_bins(sleep_df, end_col="end_time", freq=freq)
    if ex is None or len(ex) == 0:
        # no coverage anywhere => unknown where base exists (handled by join+fill later)
        return pd.DataFrame(columns=out_cols)

    ex["sleep_stage"] = ex.get("sleep_stage", "awake").astype(str).str.lower()
    ex.loc[~ex["sleep_stage"].isin(["awake", "light", "deep", "rem"]), "sleep_stage"] = "awake"

    # stage minutes per bin
    stage_min = (
        ex.pivot_table(index="timestamp", columns="sleep_stage", values="overlap_min", aggfunc="sum", fill_value=0.0)
        .sort_index()
    )

    for st in ["awake", "light", "deep", "rem"]:
        if st not in stage_min.columns:
            stage_min[st] = 0.0

    covered = stage_min[["awake", "light", "deep", "rem"]].sum(axis=1).astype(float)
    unknown = (bin_minutes - covered).clip(lower=0.0)
    total = (covered + unknown).replace(0.0, np.nan)

    out = pd.DataFrame(index=stage_min.index)
    out["sleep_stage_awake"] = stage_min["awake"] / total
    out["sleep_stage_light"] = stage_min["light"] / total
    out["sleep_stage_deep"] = stage_min["deep"] / total
    out["sleep_stage_rem"] = stage_min["rem"] / total
    out["sleep_stage_unknown"] = unknown / total

    # transitions proxy: number of distinct source interval rows contributing to the bin minus 1
    trans = ex.groupby("timestamp")["row_id"].nunique().clip(lower=1) - 1
    out["sleep_transitions"] = trans.astype(int)

    # efficiency: covered_min / window_minutes (matches current semantics in process_sleep_data)
    out["sleep_efficiency"] = (covered / float(bin_minutes)).clip(lower=0.0, upper=1.0)

    return out.fillna({
        "sleep_stage_awake": 0.0,
        "sleep_stage_light": 0.0,
        "sleep_stage_deep": 0.0,
        "sleep_stage_rem": 0.0,
        "sleep_stage_unknown": 1.0,
        "sleep_transitions": 0,
        "sleep_efficiency": 0.0,
    })


def _activity_features_from_intervals(activity_df: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    """Compute activity per-bin features without per-window loops.

    Outputs columns aligned with process_activity_data():
      activity_steps_per_min, activity_stage_{walking,sedentary,generic,running}, activity_transitions, activity_intensity_score

    Semantics:
      - Step totals are allocated uniformly within each original interval (same as current code).
      - Stage minutes are overlap minutes; any uncovered minutes are treated as sedentary.
      - transitions proxy uses number of distinct source rows contributing to the bin minus 1.
    """
    bin_minutes = _parse_freq_to_minutes(freq)

    if activity_df is None or len(activity_df) == 0 or "end_time" not in activity_df.columns:
        return pd.DataFrame(columns=[
            "activity_steps_per_min",
            "activity_stage_walking",
            "activity_stage_sedentary",
            "activity_stage_generic",
            "activity_stage_running",
            "activity_transitions",
            "activity_intensity_score",
        ])

    ex = _explode_intervals_to_bins(activity_df, end_col="end_time", freq=freq)
    if ex is None or len(ex) == 0:
        return pd.DataFrame(columns=[
            "activity_steps_per_min",
            "activity_stage_walking",
            "activity_stage_sedentary",
            "activity_stage_generic",
            "activity_stage_running",
            "activity_transitions",
            "activity_intensity_score",
        ])

    ex["activity_name"] = ex.get("activity_name", "sedentary").astype(str).str.lower()
    ex.loc[~ex["activity_name"].isin(["walking", "sedentary", "generic", "running"]), "activity_name"] = "generic"

    # Minutes per stage per bin
    stage_min = (
        ex.pivot_table(index="timestamp", columns="activity_name", values="overlap_min", aggfunc="sum", fill_value=0.0)
        .sort_index()
    )
    for st in ["walking", "sedentary", "generic", "running"]:
        if st not in stage_min.columns:
            stage_min[st] = 0.0

    covered = stage_min[["walking", "sedentary", "generic", "running"]].sum(axis=1).astype(float)
    # Fill uncovered minutes as sedentary (matches current code behavior)
    uncovered = (bin_minutes - covered).clip(lower=0.0)
    stage_min["sedentary"] = stage_min["sedentary"].astype(float) + uncovered

    total = stage_min[["walking", "sedentary", "generic", "running"]].sum(axis=1).replace(0.0, np.nan)

    out = pd.DataFrame(index=stage_min.index)
    out["activity_stage_walking"] = stage_min["walking"] / total
    out["activity_stage_sedentary"] = stage_min["sedentary"] / total
    out["activity_stage_generic"] = stage_min["generic"] / total
    out["activity_stage_running"] = stage_min["running"] / total

    # Steps: allocate uniformly within each original row
    starts = pd.to_datetime(ex.index if "start_time" not in ex.columns else ex["start_time"], errors="coerce", utc=True)
    # The exploded df has original start as index in exported tables; after join it may be in unnamed column. We fallback to the original activity_df index via row_id.
    # Compute original interval duration (min) per source row_id by looking back into the attached end_time and the original start in the activity_df.
    # Here we reconstruct row starts by using the attached index from activity_df (it was preserved in the join).
    if "index" in ex.columns:
        row_start = pd.to_datetime(ex["index"], errors="coerce", utc=True)
    else:
        # best-effort fallback
        row_start = pd.to_datetime(ex.get("start_time", pd.NaT), errors="coerce", utc=True)

    row_end = pd.to_datetime(ex["end_time"], errors="coerce", utc=True)
    dur_min = ((row_end - row_start).dt.total_seconds() / 60.0).replace(0.0, np.nan)

    steps_total = pd.to_numeric(ex.get("steps", 0.0), errors="coerce").fillna(0.0)
    step_rate = (steps_total / dur_min).fillna(0.0)  # steps per minute
    ex["steps_in_overlap"] = step_rate * ex["overlap_min"].astype(float)

    steps_per_bin = ex.groupby("timestamp")["steps_in_overlap"].sum().astype(float)
    out["activity_steps_per_min"] = (steps_per_bin / float(bin_minutes)).reindex(out.index).fillna(0.0)

    trans = ex.groupby("timestamp")["row_id"].nunique().clip(lower=1) - 1
    out["activity_transitions"] = trans.astype(int)

    intensity_map = {"sedentary": 0.0, "generic": 1.0, "walking": 2.0, "running": 3.0}
    weighted = (
        stage_min[["walking", "sedentary", "generic", "running"]]
        .mul(pd.Series({k: intensity_map[k] for k in ["walking", "sedentary", "generic", "running"]}))
        .sum(axis=1)
        .astype(float)
    )
    out["activity_intensity_score"] = (weighted / total).reindex(out.index)

    return out.fillna({
        "activity_steps_per_min": 0.0,
        "activity_stage_walking": 0.0,
        "activity_stage_sedentary": 1.0,
        "activity_stage_generic": 0.0,
        "activity_stage_running": 0.0,
        "activity_transitions": 0,
        "activity_intensity_score": 0.0,
    })


def process_sleep_data(sleep_df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> Dict[str, float]:
    if sleep_df is None or len(sleep_df) == 0 or "end_time" not in sleep_df.columns:
        return {
            "sleep_stage_awake": 0.0,
            "sleep_stage_light": 0.0,
            "sleep_stage_deep": 0.0,
            "sleep_stage_rem": 0.0,
            "sleep_stage_unknown": 1.0,
            "sleep_transitions": 0,
            "sleep_efficiency": 0.0,
        }

    mask = (sleep_df.index < window_end) & (pd.to_datetime(sleep_df["end_time"], errors="coerce") > window_start)
    ws = sleep_df.loc[mask]
    window_duration_min = (window_end - window_start).total_seconds() / 60

    if len(ws) == 0 or window_duration_min <= 0:
        return {
            "sleep_stage_awake": 0.0,
            "sleep_stage_light": 0.0,
            "sleep_stage_deep": 0.0,
            "sleep_stage_rem": 0.0,
            "sleep_stage_unknown": 1.0,
            "sleep_transitions": 0,
            "sleep_efficiency": 0.0,
        }

    stage_durations = {"awake": 0.0, "light": 0.0, "deep": 0.0, "rem": 0.0}
    for _, row in ws.iterrows():
        row_end = pd.to_datetime(row.get("end_time"), errors="coerce")
        if pd.isna(row_end):
            continue
        overlap_start = max(row.name, window_start)
        overlap_end = min(row_end, window_end)
        duration_min = (overlap_end - overlap_start).total_seconds() / 60

        stage = str(row.get("sleep_stage", "awake")).lower()
        if stage in stage_durations and duration_min > 0:
            stage_durations[stage] += duration_min

    covered_min = float(sum(stage_durations.values()))
    sleep_eff = (covered_min / window_duration_min) if window_duration_min > 0 else 0.0

    # Treat any uncovered minutes as unknown rather than awake.
    unknown_min = float(max(0.0, window_duration_min - covered_min))

    total_time = float(covered_min + unknown_min)
    if total_time <= 0:
        return {
            "sleep_stage_awake": 0.0,
            "sleep_stage_light": 0.0,
            "sleep_stage_deep": 0.0,
            "sleep_stage_rem": 0.0,
            "sleep_stage_unknown": 1.0,
            "sleep_transitions": 0,
            "sleep_efficiency": 0.0,
        }

    stage_props = {f"sleep_stage_{k}": (v / total_time) for k, v in stage_durations.items()}
    stage_props["sleep_stage_unknown"] = unknown_min / total_time

    stage_props.update({
        "sleep_transitions": int(len(ws) - 1 if len(ws) > 1 else 0),
        "sleep_efficiency": float(min(sleep_eff, 1.0)),
    })
    return stage_props


def process_activity_data(activity_df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> Dict[str, float]:
    if activity_df is None or len(activity_df) == 0 or "end_time" not in activity_df.columns:
        return {
            "activity_steps_per_min": 0.0,
            "activity_stage_walking": 0.0,
            "activity_stage_sedentary": 1.0,
            "activity_stage_generic": 0.0,
            "activity_stage_running": 0.0,
            "activity_transitions": 0,
            "activity_intensity_score": 0.0,
        }

    mask = (activity_df.index < window_end) & (pd.to_datetime(activity_df["end_time"], errors="coerce") > window_start)
    wa = activity_df.loc[mask]
    window_duration_min = (window_end - window_start).total_seconds() / 60

    if len(wa) == 0 or window_duration_min <= 0:
        return {
            "activity_steps_per_min": 0.0,
            "activity_stage_walking": 0.0,
            "activity_stage_sedentary": 1.0,
            "activity_stage_generic": 0.0,
            "activity_stage_running": 0.0,
            "activity_transitions": 0,
            "activity_intensity_score": 0.0,
        }

    total_steps = 0.0
    durations = {"walking": 0.0, "sedentary": 0.0, "generic": 0.0, "running": 0.0}

    for _, row in wa.iterrows():
        row_end = pd.to_datetime(row.get("end_time"), errors="coerce")
        if pd.isna(row_end):
            continue

        overlap_start = max(row.name, window_start)
        overlap_end = min(row_end, window_end)
        duration_min = (overlap_end - overlap_start).total_seconds() / 60
        if duration_min <= 0:
            continue

        # steps
        row_start = row.name
        original_duration = (row_end - row_start).total_seconds() / 60
        if original_duration and not pd.isna(row.get("steps")):
            step_rate = float(row.get("steps", 0.0)) / float(original_duration)
            total_steps += step_rate * duration_min

        # type
        a_type = str(row.get("activity_name", "sedentary")).lower()
        if a_type in durations:
            durations[a_type] += duration_min

    steps_per_min = float(total_steps / window_duration_min) if window_duration_min else 0.0

    total_activity_time = float(sum(durations.values()))
    if total_activity_time < window_duration_min:
        durations["sedentary"] += (window_duration_min - total_activity_time)

    total_time = float(sum(durations.values()))
    props = (
        {f"activity_stage_{k}": (v / total_time) for k, v in durations.items()}
        if total_time > 0
        else {
            "activity_stage_walking": 0.0,
            "activity_stage_sedentary": 1.0,
            "activity_stage_generic": 0.0,
            "activity_stage_running": 0.0,
        }
    )

    intensity_map = {"sedentary": 0.0, "generic": 1.0, "walking": 2.0, "running": 3.0}
    weighted = float(sum(intensity_map.get(k, 1.0) * v for k, v in durations.items()))
    intensity = float(weighted / total_time) if total_time else 0.0

    props.update({
        "activity_steps_per_min": steps_per_min,
        "activity_transitions": int(len(wa) - 1 if len(wa) > 1 else 0),
        "activity_intensity_score": intensity,
    })
    return props


def modality_participant_summary(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a summary table of participants present in each modality.

    Includes min/max timestamps (from the index) to help diagnose unexpected
    timestamp parsing/filtering.
    """
    rows: List[Dict[str, object]] = []
    for modality, df in data_dict.items():
        if df is None or len(df) == 0 or "participant_id" not in df.columns:
            rows.append(
                {
                    "modality": modality,
                    "n_rows": 0,
                    "n_participants": 0,
                    "min_time": None,
                    "max_time": None,
                }
            )
            continue

        n_participants = int(df["participant_id"].astype(str).nunique())
        min_time = df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None
        max_time = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None

        rows.append(
            {
                "modality": modality,
                "n_rows": int(len(df)),
                "n_participants": n_participants,
                "min_time": str(min_time) if min_time is not None else None,
                "max_time": str(max_time) if max_time is not None else None,
            }
        )

    out = (
        pd.DataFrame(rows)
        .sort_values(["n_participants", "modality"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return out


def participant_modality_coverage(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return participant x modality coverage (boolean) and counts.

    This is used to explain why `find_common_participants(..., min_datasets=k)` yields
    a smaller set: many participants are missing one or more modality parquet exports.
    """
    modalities = sorted(list(data_dict.keys()))

    pid_to_modalities: Dict[str, set] = {}
    for m in modalities:
        df = data_dict.get(m)
        if df is None or len(df) == 0 or "participant_id" not in df.columns:
            continue
        pids = set(df["participant_id"].astype(str).unique())
        for pid in pids:
            pid_to_modalities.setdefault(pid, set()).add(m)

    all_pids = sorted(pid_to_modalities.keys())
    rows: List[Dict[str, object]] = []
    for pid in all_pids:
        present = pid_to_modalities.get(pid, set())
        row: Dict[str, object] = {"participant_id": pid, "n_modalities": len(present)}
        for m in modalities:
            row[m] = 1 if m in present else 0
        # helpful: missing list
        row["missing_modalities"] = ",".join([m for m in modalities if m not in present])
        rows.append(row)

    return pd.DataFrame(rows)


def modality_coverage_histogram(coverage_df: pd.DataFrame, modalities: List[str]) -> pd.DataFrame:
    """Histogram of how many participants have N modalities present."""
    if coverage_df is None or len(coverage_df) == 0:
        return pd.DataFrame(columns=["n_modalities", "n_participants"])

    hist = (
        coverage_df["n_modalities"]
        .value_counts()
        .sort_index()
        .rename_axis("n_modalities")
        .reset_index(name="n_participants")
    )

    # Ensure all bins from 0..len(modalities)
    full = pd.DataFrame({"n_modalities": list(range(0, len(modalities) + 1))})
    hist = full.merge(hist, on="n_modalities", how="left").fillna({"n_participants": 0}).astype({"n_participants": int})
    return hist


def manifest_participant_summary() -> pd.DataFrame:
    """Compute participant counts per modality directly from manifests."""
    m_g = _read_tsv_any(MANIFEST_GLUCOSE)
    m_a = _read_tsv_any(MANIFEST_ACTIVITY)

    def _pid_series(df: pd.DataFrame) -> pd.Series:
        for c in ["person_id", "participant_id", "participant", "subject", "user_id", "user", "uuid"]:
            if c in df.columns:
                s = df[c].astype(str)
                # tolerate AIREADI- prefix and numeric floats
                s = s.str.replace("AIREADI-", "", regex=False)
                s = s.str.replace(".0", "", regex=False)
                return s
        # fallback: try parse digits from any filepath column
        return pd.Series([], dtype=str)

    pid_g = _pid_series(m_g)
    pid_a = _pid_series(m_a)

    # modality->(manifest_df, filepath_col)
    modality_cols = {
        "cgm": (m_g, "glucose_filepath"),
        "heartrate": (m_a, "heartrate_filepath"),
        "oxygen": (m_a, "oxygen_saturation_filepath"),
        "stress": (m_a, "stress_level_filepath"),
        "sleep": (m_a, "sleep_filepath"),
        "respiration": (m_a, "respiratory_rate_filepath"),
        "activity": (m_a, "physical_activity_filepath"),
        "calories": (m_a, "active_calories_filepath"),
    }

    rows: List[Dict[str, object]] = []
    for modality, (df, col) in modality_cols.items():
        if col not in df.columns:
            rows.append({"modality": modality, "n_manifest_rows": int(len(df)), "n_participants_with_filepath": 0})
            continue

        pid = pid_g if df is m_g else pid_a
        sub = pd.DataFrame({"participant_id": pid, "filepath": df[col]})
        sub = sub.dropna()
        sub = sub[sub["filepath"].astype(str).str.strip().ne("")]

        # normalize a bit (not required for count, but helpful for debugging)
        sub["filepath"] = sub["filepath"].astype(str).map(_as_gs_uri_with_bucket)

        rows.append(
            {
                "modality": modality,
                "n_manifest_rows": int(len(sub)),
                "n_participants_with_filepath": int(sub["participant_id"].astype(str).nunique()),
            }
        )

    out = pd.DataFrame(rows).sort_values(["n_participants_with_filepath", "modality"], ascending=[False, True]).reset_index(drop=True)
    return out


def manifest_participant_sets() -> Dict[str, set]:
    """Return modality -> set(participant_id) from manifests where modality filepath is present."""
    m_g = _read_tsv_any(MANIFEST_GLUCOSE)
    m_a = _read_tsv_any(MANIFEST_ACTIVITY)

    def _pid_series(df: pd.DataFrame) -> pd.Series:
        for c in ["person_id", "participant_id", "participant", "subject", "user_id", "user", "uuid"]:
            if c in df.columns:
                s = df[c].astype(str)
                s = s.str.replace("AIREADI-", "", regex=False)
                s = s.str.replace(".0", "", regex=False)
                return s
        return pd.Series([], dtype=str)

    pid_g = _pid_series(m_g)
    pid_a = _pid_series(m_a)

    modality_cols = {
        "cgm": (m_g, pid_g, "glucose_filepath"),
        "heartrate": (m_a, pid_a, "heartrate_filepath"),
        "oxygen": (m_a, pid_a, "oxygen_saturation_filepath"),
        "stress": (m_a, pid_a, "stress_level_filepath"),
        "sleep": (m_a, pid_a, "sleep_filepath"),
        "respiration": (m_a, pid_a, "respiratory_rate_filepath"),
        "activity": (m_a, pid_a, "physical_activity_filepath"),
        "calories": (m_a, pid_a, "active_calories_filepath"),
    }

    out: Dict[str, set] = {}
    for modality, (df, pid, col) in modality_cols.items():
        if col not in df.columns or len(pid) == 0:
            out[modality] = set()
            continue

        sub = pd.DataFrame({"participant_id": pid, "filepath": df[col]})
        sub = sub.dropna()
        sub = sub[sub["filepath"].astype(str).str.strip().ne("")]
        out[modality] = set(sub["participant_id"].astype(str).unique())

    return out


def parquet_participant_sets(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, set]:
    """Return modality -> set(participant_id) present in loaded parquet dataframes."""
    out: Dict[str, set] = {}
    for modality, df in data_dict.items():
        if df is None or len(df) == 0 or "participant_id" not in df.columns:
            out[modality] = set()
            continue
        out[modality] = set(df["participant_id"].astype(str).unique())
    return out


def manifest_vs_parquet_missing_participants(
    data_dict: Dict[str, pd.DataFrame],
    modalities: List[str],
    write_dir: Optional[str] = None,
    limit: int = 50,
) -> pd.DataFrame:
    """Compare manifest-derived participants to parquet-derived participants.

    Returns a summary dataframe and optionally writes per-modality missing ID lists.
    """
    m_sets = manifest_participant_sets()
    p_sets = parquet_participant_sets(data_dict)

    rows: List[Dict[str, object]] = []
    for m in modalities:
        manifest_pids = set(m_sets.get(m, set()))
        parquet_pids = set(p_sets.get(m, set()))
        missing = sorted(manifest_pids - parquet_pids)

        rows.append(
            {
                "modality": m,
                "n_manifest_participants": int(len(manifest_pids)),
                "n_parquet_participants": int(len(parquet_pids)),
                "n_missing_in_parquet": int(len(missing)),
                "missing_sample": ",".join(missing[: int(limit)]) if missing else "",
            }
        )

        if write_dir is not None:
            Path(write_dir).mkdir(parents=True, exist_ok=True)
            out_path = str(Path(write_dir) / f"missing_participants_{m}.csv")
            pd.DataFrame({"participant_id": missing}).to_csv(out_path, index=False)

    return pd.DataFrame(rows).sort_values(["n_missing_in_parquet", "modality"], ascending=[False, True]).reset_index(drop=True)


# --- performance helpers ---

GroupMap = Dict[str, Dict[str, pd.DataFrame]]  # modality -> participant_id -> df


def _build_group_maps(data_dict: Dict[str, pd.DataFrame]) -> GroupMap:
    """Pre-split each modality dataframe into per-participant dataframes.

    This avoids repeatedly scanning the entire modality dataframe for each participant.
    """
    out: GroupMap = {}
    for modality, df in data_dict.items():
        if df is None or len(df) == 0 or "participant_id" not in df.columns:
            out[modality] = {}
            continue

        d2 = df.copy()
        d2["participant_id"] = d2["participant_id"].astype(str)
        out[modality] = {str(pid): g for pid, g in d2.groupby("participant_id", sort=False)}
    return out


# --- fast, vectorized per-participant windowing for point-sampled modalities ---

def _resample_series(
    df: pd.DataFrame,
    value_col: str,
    freq: str,
    *,
    valid_range: Optional[Tuple[float, float]] = None,
) -> pd.Series:
    """Return a resampled series (mean) for a point-sampled modality."""
    if df is None or len(df) == 0 or value_col not in df.columns or not isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)

    s = pd.to_numeric(df[value_col], errors="coerce")
    # Ensure the series carries the DatetimeIndex required by resample().
    s = pd.Series(s.to_numpy(), index=df.index)
    if valid_range is not None:
        lo, hi = valid_range
        # keep semantics close to prior filters
        s = s.where((s > lo) & (s < hi))
    return s.resample(freq).mean()


def _resample_stats(
    df: pd.DataFrame,
    value_col: str,
    freq: str,
    *,
    valid_range: Optional[Tuple[float, float]] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Fast per-window stats using resample instead of per-window boolean masks."""
    if df is None or len(df) == 0 or value_col not in df.columns or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()

    raw = pd.to_numeric(df[value_col], errors="coerce")
    # Ensure DatetimeIndex on the series; resample() bins by index.
    raw = pd.Series(raw.to_numpy(), index=df.index)

    if valid_range is not None:
        lo, hi = valid_range
        valid = raw.where((raw > lo) & (raw < hi))
    else:
        valid = raw

    agg = valid.resample(freq).agg(["mean", "std", "min", "max", "count"])
    p = prefix or value_col
    agg = agg.rename(
        columns={
            "mean": f"{p}_mean",
            "std": f"{p}_std",
            "min": f"{p}_min",
            "max": f"{p}_max",
            "count": f"{p}_count",
        }
    )

    # Availability = valid_count / raw_count (approx) to match prior semantics.
    raw_count = raw.resample(freq).count()
    agg[f"{p}_device_availability"] = (
        agg[f"{p}_count"].fillna(0) / raw_count.replace(0, np.nan)
    ).fillna(0.0)

    return agg


# --- participant processing ---

def _participant_timezone_from_participant_data(participant_data: Dict[str, pd.DataFrame]) -> Optional[str]:
    """Best-effort participant timezone (IANA) from already-sliced modality frames."""
    for modality in ["cgm", "sleep", "activity", "calories", "heartrate", "respiration", "oxygen", "stress"]:
        df = participant_data.get(modality)
        if df is None or len(df) == 0 or "timezone" not in df.columns:
            continue
        tz = df["timezone"].dropna()
        if len(tz) == 0:
            continue
        tz_mode = tz.astype(str).value_counts().idxmax()
        if tz_mode and str(tz_mode).strip().lower() not in {"none", "nan", "<na>", ""}:
            return str(tz_mode)
    return None


def _process_one_participant(
    pid: str,
    data_dict: Union[Dict[str, pd.DataFrame], GroupMap],
    freq: str,
    *,
    fast_groupby: bool = False,
    return_reason: bool = False,
) -> Union[Optional[pd.DataFrame], Tuple[Optional[pd.DataFrame], str]]:
    """Process a single participant into a per-window feature dataframe.

    If return_reason=True, returns (df_or_none, reason) where reason is one of:
      ok, no_data, no_overlap, grid_too_short
    """
    pid = str(pid)

    participant_data: Dict[str, pd.DataFrame] = {}
    time_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    if fast_groupby:
        # data_dict: modality -> participant_id -> df
        for name, pid_map in data_dict.items():  # type: ignore[union-attr]
            part = pid_map.get(pid)
            if part is None or len(part) == 0:
                continue

            # Ensure monotonic DatetimeIndex for resample/join correctness.
            if isinstance(part.index, pd.DatetimeIndex) and not part.index.is_monotonic_increasing:
                part = part.sort_index()

            participant_data[name] = part
            end_bound = (
                pd.to_datetime(part["end_time"], errors="coerce").max()
                if "end_time" in part.columns
                else part.index.max()
            )
            time_ranges.append((part.index.min(), end_bound))
    else:
        for name, df in data_dict.items():  # type: ignore[union-attr]
            if df is None or len(df) == 0 or "participant_id" not in df.columns:
                continue

            part = df[df["participant_id"].astype(str) == pid].copy()
            if len(part) == 0:
                continue

            if isinstance(part.index, pd.DatetimeIndex) and not part.index.is_monotonic_increasing:
                part = part.sort_index()

            participant_data[name] = part
            end_bound = (
                pd.to_datetime(part["end_time"], errors="coerce").max()
                if "end_time" in part.columns
                else part.index.max()
            )
            time_ranges.append((part.index.min(), end_bound))

    if not time_ranges:
        res = None
        return (res, "no_data") if return_reason else res

    start_time = max(t[0] for t in time_ranges)
    end_time = min(t[1] for t in time_ranges)
    if start_time is pd.NaT or end_time is pd.NaT or start_time >= end_time:
        res = None
        return (res, "no_overlap") if return_reason else res

    # Align grid bounds with bin edges to avoid off-by-one reindex mismatches.
    start_time = pd.Timestamp(start_time).floor(freq)
    end_time = pd.Timestamp(end_time).ceil(freq)

    time_grid = create_time_grid(start_time, end_time, freq=freq)
    if len(time_grid) < 2:
        res = None
        return (res, "grid_too_short") if return_reason else res

    # Base frame: one row per window start.
    base = pd.DataFrame(index=time_grid[:-1])
    base.index.name = "timestamp"
    base["participant_id"] = pid

    # Attach timezone/local timestamp context (if available)
    tz = _participant_timezone_from_participant_data(participant_data)
    if tz is not None:
        base["timezone"] = tz
        try:
            idx = base.index
            if isinstance(idx, pd.DatetimeIndex):
                if idx.tz is None:
                    idx = idx.tz_localize("UTC")
                base["timestamp_local"] = idx.tz_convert(tz)
        except Exception:
            # If tz conversion fails for any reason, keep timezone string only
            pass

    # CGM: mean + count per window
    if "cgm" in participant_data and "glucose" in participant_data["cgm"].columns:
        cgm_df = participant_data["cgm"]
        g = pd.to_numeric(cgm_df["glucose"], errors="coerce")
        g = pd.Series(g.to_numpy(), index=cgm_df.index)
        cgmr = g.resample(freq)
        base["cgm_glucose_mean"] = cgmr.mean().reindex(base.index)
        base["cgm_count"] = cgmr.count().reindex(base.index).fillna(0).astype(int)

        # Debug hint: if we have input rows but got no resampled values, likely index mismatch.
        if len(cgm_df) > 0 and int(base["cgm_count"].sum()) == 0:
            tqdm.write(f"[WARN] pid={pid}: CGM has rows but resample produced zero counts. index={type(cgm_df.index)} tz={getattr(cgm_df.index,'tz',None)}")
    else:
        base["cgm_glucose_mean"] = np.nan
        base["cgm_count"] = 0

    # Calories: convert daily cumulative Garmin readings to per-window increments.
    if "calories" in participant_data and "value" in participant_data["calories"].columns:
        cfeat = _calorie_features_from_cumulative(participant_data["calories"], freq=freq)
        if len(cfeat) > 0:
            cfeat = cfeat.reindex(base.index)
            base["calories_total"] = cfeat["calories_total"].fillna(0.0)
            base["calories_per_min"] = cfeat["calories_per_min"].fillna(0.0)
        else:
            base["calories_total"] = 0.0
            base["calories_per_min"] = 0.0
    else:
        base["calories_total"] = 0.0
        base["calories_per_min"] = 0.0

    # High-frequency point modalities (fast resample)
    hf_specs = {
        "heartrate": ("heart_rate", (30.0, 200.0)),
        "respiration": ("respiratory_rate", (3.0, 35.0)),  # permissive artifact filter for wearable RR
        "oxygen": ("oxygen_saturation", (50.0, 100.0)),
        "stress": ("stress_level", (-1.0, 101.0)),
    }
    for dataset, (col, vr) in hf_specs.items():
        if dataset not in participant_data:
            continue
        stats = _resample_stats(participant_data[dataset], col, freq, valid_range=vr, prefix=col)
        if len(stats) > 0:
            base = base.join(stats, how="left")

    # Sleep + Activity (interval-based, exploded to bins)
    if "sleep" in participant_data:
        sfeat = _sleep_features_from_intervals(participant_data["sleep"], freq=freq)
        if len(sfeat) > 0:
            base = base.join(sfeat, how="left")
        else:
            # no sleep coverage => unknown
            base["sleep_stage_awake"] = base.get("sleep_stage_awake", 0.0)
            base["sleep_stage_light"] = base.get("sleep_stage_light", 0.0)
            base["sleep_stage_deep"] = base.get("sleep_stage_deep", 0.0)
            base["sleep_stage_rem"] = base.get("sleep_stage_rem", 0.0)
            base["sleep_stage_unknown"] = base.get("sleep_stage_unknown", 1.0)
            base["sleep_transitions"] = base.get("sleep_transitions", 0)
            base["sleep_efficiency"] = base.get("sleep_efficiency", 0.0)

    if "activity" in participant_data:
        afeat = _activity_features_from_intervals(participant_data["activity"], freq=freq)
        if len(afeat) > 0:
            base = base.join(afeat, how="left")
        else:
            base["activity_steps_per_min"] = base.get("activity_steps_per_min", 0.0)
            base["activity_stage_walking"] = base.get("activity_stage_walking", 0.0)
            base["activity_stage_sedentary"] = base.get("activity_stage_sedentary", 1.0)
            base["activity_stage_generic"] = base.get("activity_stage_generic", 0.0)
            base["activity_stage_running"] = base.get("activity_stage_running", 0.0)
            base["activity_transitions"] = base.get("activity_transitions", 0)
            base["activity_intensity_score"] = base.get("activity_intensity_score", 0.0)

    res = base
    return (res, "ok") if return_reason else res


def create_merged_dataset(
    data_dict: Dict[str, pd.DataFrame],
    participant_ids: List[str],
    *,
    freq: str = "5min",
    max_participants: Optional[int] = None,
    workers: int = 1,
    fast_groupby: bool = False,
    debug_exceptions: int = 5,
    stop_after_exceptions: int = 0,
    ffill_within_participant: bool = False,
    diagnostics: bool = True,
) -> pd.DataFrame:
    if max_participants:
        participant_ids = participant_ids[:max_participants]

    if not participant_ids:
        return pd.DataFrame()

    tqdm.write(f"Processing {len(participant_ids)} participants")

    grouped: Union[Dict[str, pd.DataFrame], GroupMap] = data_dict
    if fast_groupby:
        tqdm.write("Pre-splitting modality data by participant (fast-groupby)...")
        grouped = _build_group_maps(data_dict)

    all_results: List[pd.DataFrame] = []
    reason_counts: Counter[str] = Counter()
    exception_count = 0
    printed_exceptions = 0

    def _handle_result(result: Tuple[Optional[pd.DataFrame], str]) -> None:
        nonlocal all_results, reason_counts
        pdf, reason = result
        reason_counts[reason] += 1
        if pdf is not None and len(pdf) > 0:
            all_results.append(pdf)

    if workers and int(workers) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            futures = {
                ex.submit(
                    _process_one_participant,
                    str(pid),
                    grouped,
                    freq,
                    fast_groupby=fast_groupby,
                    return_reason=True,
                ): str(pid)
                for pid in participant_ids
            }

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing participants", unit="participant"):
                pid = futures.get(fut, "?")
                try:
                    _handle_result(fut.result())
                except Exception as e:
                    exception_count += 1
                    reason_counts["exception"] += 1
                    if printed_exceptions < int(debug_exceptions):
                        printed_exceptions += 1
                        tqdm.write(f"\n[EXCEPTION] participant_id={pid}: {type(e).__name__}: {e}")
                        tqdm.write(traceback.format_exc())
                    if stop_after_exceptions and exception_count >= int(stop_after_exceptions):
                        tqdm.write(
                            f"Stopping early after {exception_count} exceptions (stop_after_exceptions={stop_after_exceptions})."
                        )
                        break
    else:
        for pid in tqdm(participant_ids, desc="Processing participants", unit="participant"):
            try:
                result = _process_one_participant(
                    str(pid),
                    grouped,
                    freq,
                    fast_groupby=fast_groupby,
                    return_reason=True,
                )
                _handle_result(result)  # type: ignore[arg-type]
            except Exception as e:
                exception_count += 1
                reason_counts["exception"] += 1
                if printed_exceptions < int(debug_exceptions):
                    printed_exceptions += 1
                    tqdm.write(f"\n[EXCEPTION] participant_id={pid}: {type(e).__name__}: {e}")
                    tqdm.write(traceback.format_exc())
                if stop_after_exceptions and exception_count >= int(stop_after_exceptions):
                    tqdm.write(
                        f"Stopping early after {exception_count} exceptions (stop_after_exceptions={stop_after_exceptions})."
                    )
                    break

    tqdm.write("\nParticipant processing outcome counts:")
    try:
        keys = ["ok", "no_data", "no_overlap", "grid_too_short", "exception"]
        rows = [(k, int(reason_counts.get(k, 0))) for k in keys]
        tqdm.write("\n" + pd.DataFrame(rows, columns=["reason", "n_participants"]).to_string(index=False))
    except Exception:
        tqdm.write(str(reason_counts))

    if not all_results:
        return pd.DataFrame()

    final_df = pd.concat(all_results, ignore_index=False)
    final_df = final_df.reset_index().sort_values(["participant_id", "timestamp"]).set_index("timestamp")

    # ------------------------------------------------------------------
    # Time features for circadian analyses (participant local time)
    #
    # We intentionally do *not* rely on a tz-aware `timestamp_local` column in parquet,
    # because parquet/engines frequently round-trip tz-aware datetime columns poorly.
    #
    # Instead, we persist:
    #   - hour_of_day: 0..23
    #   - minute_of_day: 0..1439
    #   - time_step: 1..N per participant (monotonic in time)
    #   - timestamp_local_naive: local wall-clock timestamp (tz-naive) for debugging
    #
    # Computation:
    #   - If `timezone` column exists and has non-null values, compute local time per
    #     timezone group using `final_df.index.tz_convert(tz)`.
    #   - Otherwise fall back to UTC.
    # ------------------------------------------------------------------
    try:
        idx = final_df.index
        if isinstance(idx, pd.DatetimeIndex):
            if idx.tz is None:
                idx = idx.tz_localize("UTC")

            tz_col = "timezone"
            has_tz = tz_col in final_df.columns and final_df[tz_col].notna().any()

            if has_tz:
                tzs = final_df[tz_col].astype("string")
                hour = pd.Series(index=final_df.index, dtype="float64")
                minute = pd.Series(index=final_df.index, dtype="float64")
                ts_local_naive = pd.Series(index=final_df.index, dtype="datetime64[ns]")

                groups = final_df.groupby(tzs, dropna=False, sort=False).groups
                for tz_name, rows in groups.items():
                    tz_str = "" if tz_name is None else str(tz_name)
                    if tz_str.strip().lower() in {"", "none", "nan", "<na>"}:
                        local_idx = idx[rows]
                    else:
                        try:
                            local_idx = idx[rows].tz_convert(tz_str)
                        except Exception:
                            local_idx = idx[rows]

                    hour.loc[rows] = local_idx.hour
                    minute.loc[rows] = local_idx.hour * 60 + local_idx.minute
                    # parquet-friendly local wall-clock timestamp
                    try:
                        ts_local_naive.loc[rows] = local_idx.tz_localize(None)
                    except Exception:
                        ts_local_naive.loc[rows] = pd.to_datetime(local_idx, errors="coerce").tz_localize(None)

                # Fill any stragglers with UTC
                hour = hour.fillna(pd.Series(idx.hour, index=final_df.index))
                minute = minute.fillna(pd.Series(idx.hour * 60 + idx.minute, index=final_df.index))

                final_df["hour_of_day"] = hour.astype(int)
                final_df["minute_of_day"] = minute.astype(int)
                final_df["timestamp_local_naive"] = ts_local_naive
            else:
                final_df["hour_of_day"] = idx.hour.astype(int)
                final_df["minute_of_day"] = (idx.hour * 60 + idx.minute).astype(int)

            # 1..N within each participant, after sorting by time (already sorted above)
            if "participant_id" in final_df.columns:
                final_df["time_step"] = final_df.groupby("participant_id", sort=False).cumcount() + 1
            else:
                final_df["time_step"] = np.arange(1, len(final_df) + 1, dtype=int)
    except Exception as _e:
        # Never fail the pipeline for auxiliary time feature creation.
        pass

    # IMPORTANT:
    # Do NOT forward-fill by default.
    # Group-wise ffill can silently drop/misalign columns (especially with mixed dtypes)
    # and can turn valid engineered numeric columns into all-NaN in the written parquet.
    if diagnostics:
        try:
            nn = final_df.notna().sum().sort_values()
            tqdm.write("\n[diagnostics] Non-null counts (lowest 15) AFTER concat/sort:")
            tqdm.write(nn.head(15).to_string())
        except Exception:
            pass

    if not ffill_within_participant:
        return final_df

    # Optional, explicit forward fill within participant (only affects columns that can be ffilled)
    pid_col = final_df["participant_id"].copy()
    final_ffill = final_df.groupby("participant_id", sort=False).ffill()
    if "participant_id" not in final_ffill.columns:
        final_ffill["participant_id"] = pid_col

    if diagnostics:
        try:
            nn2 = final_ffill.notna().sum().sort_values()
            tqdm.write("\n[diagnostics] Non-null counts (lowest 15) AFTER optional ffill:")
            tqdm.write(nn2.head(15).to_string())
        except Exception:
            pass

    return final_ffill


# --- loading exported modality parquet files ---

MODALITY_TO_FILENAME = {
    # Filenames produced by export_modalities_to_gcs.py (modality.{parquet|feather})
    "cgm": "cgm.parquet",
    "sleep": "sleep.parquet",
    "heartrate": "heartrate.parquet",
    "respiration": "respiration.parquet",
    "oxygen": "oxygen.parquet",
    "stress": "stress.parquet",
    "activity": "step.parquet",
    "calories": "calorie.parquet",
}


def load_exported_modalities(raw_root: str, modalities: Iterable[str]) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}

    # Column pruning: read only what we need for the downstream feature engineering.
    # This reduces IO and memory when parquet contains extra fields.
    needed_cols: Dict[str, List[str]] = {
        "cgm": ["participant_id", "start_time", "end_time", "glucose", "timezone"],
        "sleep": ["participant_id", "start_time", "end_time", "sleep_stage", "timezone"],
        "heartrate": ["participant_id", "timestamp", "heart_rate", "timezone"],
        "respiration": ["participant_id", "timestamp", "respiratory_rate", "timezone"],
        "oxygen": ["participant_id", "timestamp", "oxygen_saturation", "timezone"],
        "stress": ["participant_id", "timestamp", "stress_level", "timezone"],
        "activity": ["participant_id", "start_time", "end_time", "steps", "activity_name", "timezone"],
        "calories": ["participant_id", "time", "value", "activity_name", "timezone"],
    }

    for m in modalities:
        if m not in MODALITY_TO_FILENAME:
            raise ValueError(f"Unknown modality '{m}'. Known: {sorted(MODALITY_TO_FILENAME)}")

        path = f"{raw_root.rstrip('/')}/{MODALITY_TO_FILENAME[m]}"
        try:
            cols = needed_cols.get(m) if getattr(load_exported_modalities, "_use_columns", False) else None
            df = _read_parquet_any(path, columns=cols)
        except Exception as e:
            tqdm.write(f"Failed to load {m} from {path}: {e}")
            df = pd.DataFrame()

        # Ensure a DatetimeIndex for downstream windowing/cleaning.
        if len(df) and not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            elif "start_time" in df.columns:
                df = df.set_index("start_time")
            elif "time" in df.columns:
                df = df.set_index("time")

        # Coerce index to datetime (handles strings)
        if len(df) and isinstance(df.index, pd.Index) and not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
            if df.index.hasnans:
                df = df.loc[~df.index.isna()].copy()

        # Ensure UTC index when possible
        if len(df) and isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")

        data[m] = df
        tqdm.write(f"Loaded {m}: {len(df):,} rows")

    return data


def export_final_dataset(
    merged_df: pd.DataFrame,
    output_dir: str,
    upload: bool,
    upload_root: str,
    prefix: str = "final_multimodal_dataset",
) -> Tuple[str, str]:
    if merged_df is None or len(merged_df) == 0:
        raise ValueError("No merged data to export")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    local_parquet = str(Path(output_dir) / f"{prefix}_{ts}.parquet")
    local_metadata = str(Path(output_dir) / f"{prefix}_{ts}_metadata.json")

    merged_df.to_parquet(local_parquet, index=True)

    metadata = {
        "creation_date": pd.Timestamp.now().isoformat(),
        "shape": list(merged_df.shape),
        "participants": int(merged_df["participant_id"].nunique()) if "participant_id" in merged_df.columns else None,
        "time_range": {
            "start": merged_df.index.min().isoformat() if isinstance(merged_df.index, pd.DatetimeIndex) and len(merged_df) else None,
            "end": merged_df.index.max().isoformat() if isinstance(merged_df.index, pd.DatetimeIndex) and len(merged_df) else None,
        },
        "columns": list(merged_df.columns),
    }

    with open(local_metadata, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    if upload:
        remote_parquet = f"{upload_root.rstrip('/')}/{Path(local_parquet).name}"
        remote_metadata = f"{upload_root.rstrip('/')}/{Path(local_metadata).name}"
        _write_parquet_any(merged_df, remote_parquet)
        # metadata upload
        if _is_gcs_path(remote_metadata):
            import gcsfs

            fs = gcsfs.GCSFileSystem(cache_timeout=3600)
            with fs.open(remote_metadata, "wb") as f:
                f.write(Path(local_metadata).read_bytes())
        else:
            Path(remote_metadata).parent.mkdir(parents=True, exist_ok=True)
            Path(remote_metadata).write_bytes(Path(local_metadata).read_bytes())

        tqdm.write(f"Uploaded: {remote_parquet}")
        tqdm.write(f"Uploaded: {remote_metadata}")

    return local_parquet, local_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create finalized multimodal dataset from per-modality Parquet exports")
    p.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help="Root containing exported modality parquet files (local dir or gs://...)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Local output directory")
    p.add_argument("--upload", action="store_true", help=f"Upload finalized dataset to {DEFAULT_UPLOAD_ROOT}")
    p.add_argument("--upload-root", default=DEFAULT_UPLOAD_ROOT, help="Upload destination root (local dir or gs://...)")
    p.add_argument("--freq", default="5min", help="Time grid frequency for aggregation")
    p.add_argument("--min-modalities", type=int, default=6, help="Minimum number of modalities required per participant")
    p.add_argument("--max-participants", type=int, default=None, help="Optional cap for quick tests")
    p.add_argument("--workers", type=int, default=1, help="Participant-level worker threads for merging (default: 1)")
    p.add_argument(
        "--fast-groupby",
        action="store_true",
        help="Speed up by pre-splitting each modality dataframe into per-participant groups once (reduces repeated slicing)",
    )
    p.add_argument(
        "--use-columns",
        action="store_true",
        help="Read only the columns needed for feature engineering from parquet (faster + less memory)",
    )
    p.add_argument(
        "--modalities",
        default=",".join(MODALITY_TO_FILENAME.keys()),
        help=f"Comma-separated modalities to include. Known: {','.join(MODALITY_TO_FILENAME.keys())}",
    )
    p.add_argument(
        "--write-modality-summary",
        action="store_true",
        help="Write modality participant-count summary CSV to output dir",
    )
    p.add_argument(
        "--manifest-summary",
        action="store_true",
        help="Print participant counts per modality directly from AIREADI manifests (to compare against exported parquet counts)",
    )
    p.add_argument(
        "--manifest-missing",
        action="store_true",
        help="Print participants present in manifests (by filepath presence) but missing from exported parquet files. Writes per-modality missing ID CSVs when --write-modality-summary is set.",
    )
    p.add_argument(
        "--manifest-missing-limit",
        type=int,
        default=50,
        help="How many missing participant IDs to include in the printed sample column (default: 50)",
    )
    p.add_argument(
        "--debug-exceptions",
        type=int,
        default=5,
        help="Print stack traces for the first N participant exceptions (default: 5)",
    )
    p.add_argument(
        "--stop-after-exceptions",
        type=int,
        default=0,
        help="If >0, stop early after this many participant exceptions (default: 0 = never)",
    )
    p.add_argument(
        "--ffill-within-participant",
        action="store_true",
        help="Optional: forward-fill features within each participant after merge (disabled by default; can cause NaN issues if misused)",
    )
    p.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable merge diagnostics (non-null counts) printed after concat/sort.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if getattr(args, "manifest_summary", False):
        tqdm.write("\nManifest participant counts (by filepath presence):")
        ms = manifest_participant_summary()
        try:
            tqdm.write(ms.to_string(index=False))
        except Exception:
            tqdm.write(str(ms))

    modalities = [m.strip() for m in str(args.modalities).split(",") if m.strip()]

    tqdm.write("Loading exported modality datasets...")
    # Hack-free propagation of flag into loader while keeping the signature stable.
    setattr(load_exported_modalities, "_use_columns", bool(getattr(args, "use_columns", False)))
    data_dict = load_exported_modalities(args.raw_root, modalities)

    if not any(len(df) for df in data_dict.values()):
        raise SystemExit("No modality data loaded. Check --raw-root and modality filenames.")

    # Manifest vs parquet mismatch (before timestamp cleaning; cleaning can only reduce rows)
    if getattr(args, "manifest_missing", False):
        tqdm.write("\nManifest vs parquet participant mismatch (manifest has filepath, parquet has no rows):")
        write_dir = args.output_dir if getattr(args, "write_modality_summary", False) else None
        mm = manifest_vs_parquet_missing_participants(
            data_dict,
            modalities=modalities,
            write_dir=write_dir,
            limit=int(getattr(args, "manifest_missing_limit", 50)),
        )
        try:
            tqdm.write(mm.to_string(index=False))
        except Exception:
            tqdm.write(str(mm))
        if write_dir is not None:
            tqdm.write(f"Wrote per-modality missing participant lists under: {write_dir}")

    data_dict = clean_timestamps(data_dict)

    # Diagnostic: participant counts per modality
    summary = modality_participant_summary(data_dict)
    tqdm.write("\nParticipants per modality (after timestamp cleaning):")
    try:
        tqdm.write(summary.to_string(index=False))
    except Exception:
        tqdm.write(str(summary))

    # Diagnostic: participant coverage across modalities
    coverage_df = participant_modality_coverage(data_dict)
    hist_df = modality_coverage_histogram(coverage_df, modalities)
    tqdm.write("\nParticipant modality coverage histogram:")
    try:
        tqdm.write(hist_df.to_string(index=False))
    except Exception:
        tqdm.write(str(hist_df))

    # Save diagnostics if requested
    if args.write_modality_summary:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        out_csv = str(Path(args.output_dir) / "modality_participant_summary.csv")
        summary.to_csv(out_csv, index=False)
        tqdm.write(f"Wrote: {out_csv}")

        cov_csv = str(Path(args.output_dir) / "participant_modality_coverage.csv")
        coverage_df.to_csv(cov_csv, index=False)
        tqdm.write(f"Wrote: {cov_csv}")

        hist_csv = str(Path(args.output_dir) / "participant_modality_coverage_histogram.csv")
        hist_df.to_csv(hist_csv, index=False)
        tqdm.write(f"Wrote: {hist_csv}")

    participant_ids = find_common_participants(data_dict, min_datasets=int(args.min_modalities))
    tqdm.write(f"Participants with >= {args.min_modalities} modalities: {len(participant_ids)}")

    # If someone expects higher counts, show why they are excluded.
    if len(coverage_df) > 0:
        excluded = coverage_df[coverage_df["n_modalities"] < int(args.min_modalities)].copy()
        if len(excluded) > 0:
            # show top missing-modality patterns
            pattern = (
                excluded["missing_modalities"].value_counts().head(15).rename_axis("missing_modalities").reset_index(name="n_participants")
            )
            tqdm.write("\nTop missing-modality patterns among excluded participants:")
            try:
                tqdm.write(pattern.to_string(index=False))
            except Exception:
                tqdm.write(str(pattern))

    if not participant_ids:
        raise SystemExit("No eligible participants found. Try lowering --min-modalities.")

    merged_df = create_merged_dataset(
        data_dict,
        participant_ids,
        freq=args.freq,
        max_participants=args.max_participants,
        workers=int(args.workers),
        fast_groupby=bool(getattr(args, "fast_groupby", False)),
        debug_exceptions=int(getattr(args, "debug_exceptions", 5)),
        stop_after_exceptions=int(getattr(args, "stop_after_exceptions", 0)),
        ffill_within_participant=bool(getattr(args, "ffill_within_participant", False)),
        diagnostics=not bool(getattr(args, "no_diagnostics", False)),
    )
    if merged_df is None or len(merged_df) == 0:
        raise SystemExit("Failed to create merged dataset (empty result).")

    local_parquet, local_metadata = export_final_dataset(
        merged_df,
        output_dir=args.output_dir,
        upload=bool(args.upload),
        upload_root=args.upload_root,
    )

    tqdm.write(f"Wrote: {local_parquet}")
    tqdm.write(f"Wrote: {local_metadata}")


if __name__ == "__main__":
    main()
