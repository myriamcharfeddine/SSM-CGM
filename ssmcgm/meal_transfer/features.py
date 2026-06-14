"""Causal feature engineering for the AI-READI student (Phase E) and the
future-response signals used only for offline labelling (Phase D).

Two clearly separated groups:

* ``add_causal_features`` — uses information available **by time t only**
  (past/current CGM, wearables, static context). These are the *only* columns
  the student is allowed to see at inference time.
* ``add_future_response`` — uses **future** glucose. These are for offline
  pseudo-label construction (Phase D) ONLY and must never be fed to the student.

Every rolling/shift operation is grouped by ``segment_id`` so no feature leaks
across a recording gap or a participant boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STEP_MIN = 5  # AI-READI native cadence


# Columns the student is allowed to consume (past/current only).
CAUSAL_FEATURE_COLUMNS = [
    "cgm_glucose",
    "cgm_slope_15", "cgm_slope_30", "cgm_slope_60",
    "cgm_accel",
    "cgm_roll_mean_30", "cgm_roll_std_30", "cgm_roll_range_60",
    "cgm_pos_excursion_60",
    "hr_recent_mean", "hr_recent_change",
    "steps_15", "steps_30", "steps_60",
    "stress_recent",
    "sleep_rest",
    "tod_sin", "tod_cos",
    "hba1c_percent_baseline", "bmi_baseline",
    "med_insulin", "med_metformin", "med_any_diabetes_drug",
    "study_group_code",
]

# Forbidden as student inputs (leakage guards, asserted at train time).
LEAKAGE_COLUMNS = [
    "cgmacros_teacher_probability", "cgmacros_teacher_flag",
    "future_peak_rise_120", "dg_30", "dg_60", "dg_120", "time_to_peak_min",
    "meal_pseudo_label", "pseudo_label_confidence", "meal_response_size_proxy",
]

_SLEEP_REST = {"light", "deep", "rem"}  # asleep / resting stages


def _grouped_shift(s: pd.Series, by: pd.Series, k: int) -> pd.Series:
    return s.groupby(by).shift(k)


def _grouped_roll(s: pd.Series, by: pd.Series, win: int, fn: str, min_periods: int = 1):
    r = s.groupby(by).rolling(win, min_periods=min_periods)
    r = getattr(r, fn)()
    return r.reset_index(level=0, drop=True)


def add_causal_features(df: pd.DataFrame, segment_col: str = "segment_id") -> pd.DataFrame:
    """Append the causal (past/current-only) feature columns. Returns a copy."""
    out = df.copy()
    by = out[segment_col]
    g = out["cgm_glucose"]

    # CGM slopes (mg/dL per minute) over 15/30/60 min using past values only.
    for mins in (15, 30, 60):
        k = mins // STEP_MIN
        past = _grouped_shift(g, by, k)
        out[f"cgm_slope_{mins}"] = (g - past) / float(mins)

    # Acceleration: change of the 15-min slope over one step.
    out["cgm_accel"] = out["cgm_slope_15"] - _grouped_shift(out["cgm_slope_15"], by, 1)

    # Rolling CGM statistics (trailing windows).
    out["cgm_roll_mean_30"] = _grouped_roll(g, by, 30 // STEP_MIN, "mean")
    out["cgm_roll_std_30"] = _grouped_roll(g, by, 30 // STEP_MIN, "std").fillna(0.0)
    roll_max_60 = _grouped_roll(g, by, 60 // STEP_MIN, "max")
    roll_min_60 = _grouped_roll(g, by, 60 // STEP_MIN, "min")
    out["cgm_roll_range_60"] = roll_max_60 - roll_min_60
    # Recent positive excursion: current minus trailing-60min minimum.
    out["cgm_pos_excursion_60"] = g - roll_min_60

    # Heart rate.
    if "heartrate" in out.columns:
        hr = out["heartrate"]
        out["hr_recent_mean"] = _grouped_roll(hr, by, 30 // STEP_MIN, "mean")
        out["hr_recent_change"] = hr - _grouped_shift(hr, by, 30 // STEP_MIN)
    else:
        out["hr_recent_mean"] = np.nan
        out["hr_recent_change"] = np.nan

    # Steps accumulated over trailing 15/30/60 min.
    if "activity_steps" in out.columns:
        steps = out["activity_steps"].fillna(0.0)
        for mins in (15, 30, 60):
            out[f"steps_{mins}"] = _grouped_roll(steps, by, mins // STEP_MIN, "sum")
    else:
        for mins in (15, 30, 60):
            out[f"steps_{mins}"] = np.nan

    # Stress (recent mean).
    if "stress_level" in out.columns:
        out["stress_recent"] = _grouped_roll(out["stress_level"], by, 30 // STEP_MIN, "mean")
    else:
        out["stress_recent"] = np.nan

    # Sleep / rest indicator.
    if "sleep_stage" in out.columns:
        out["sleep_rest"] = out["sleep_stage"].isin(_SLEEP_REST).astype(float)
    else:
        out["sleep_rest"] = 0.0

    # Time-of-day cyclic features (already present in the feathers; recompute if not).
    if "tod_sin" not in out.columns and "minute_of_day" in out.columns:
        ang = 2 * np.pi * out["minute_of_day"] / 1440.0
        out["tod_sin"] = np.sin(ang)
        out["tod_cos"] = np.cos(ang)

    # Study group as an integer code (stable mapping).
    if "study_group" in out.columns:
        out["study_group_code"] = out["study_group"].astype("category").cat.codes.astype(float)
    else:
        out["study_group_code"] = -1.0

    # Ensure medication/static columns exist.
    for c in ("med_insulin", "med_metformin", "med_any_diabetes_drug",
              "hba1c_percent_baseline", "bmi_baseline"):
        if c not in out.columns:
            out[c] = np.nan

    return out


def add_future_response(
    df: pd.DataFrame,
    horizons_min=(30, 60, 120),
    peak_window_min: int = 120,
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """Append **future-glucose** response signals (offline labelling only).

    * ``dg_30/dg_60/dg_120`` — delta glucose at +30/+60/+120 min.
    * ``future_peak_rise_120`` — max future rise over the next ``peak_window_min``.
    * ``time_to_peak_min`` — minutes from t to the future peak within the window.
    All are NaN near the end of a segment where the horizon is unavailable.

    Vectorised per segment with sliding-window views (no per-row Python loop), so
    it scales to the full cohort.
    """
    out = df.copy()
    by = out[segment_col]
    g = out["cgm_glucose"]

    for mins in horizons_min:
        k = mins // STEP_MIN
        fut = g.groupby(by).shift(-k)
        out[f"dg_{mins}"] = fut - g

    kmax = peak_window_min // STEP_MIN
    rise, ttp = _future_peak_vectorized(g.to_numpy(np.float64),
                                        by.to_numpy(), kmax)
    out["future_peak_rise_120"] = rise
    out["time_to_peak_min"] = ttp * STEP_MIN
    return out


def _future_peak_vectorized(vals: np.ndarray, seg: np.ndarray, kmax: int):
    """Strictly-future (t+1 .. t+kmax) peak rise and steps-to-peak per segment.

    Returns two float arrays aligned to ``vals``; positions with no future point
    in their segment are NaN.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(vals)
    rise = np.full(n, np.nan)
    ttp = np.full(n, np.nan)
    # Contiguous segment boundaries (input is pre-sorted by segment).
    bounds = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, n]
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = b - a
        if m < 2:
            continue
        s = vals[a:b]
        # ext[j] = s[j+1]; pad the tail so every j has a kmax-wide window.
        ext = np.concatenate([s[1:], np.full(kmax, -np.inf)])
        sw = sliding_window_view(ext, kmax)[:m]   # row j -> s[j+1 .. j+kmax]
        fut_max = sw.max(axis=1)
        arg = sw.argmax(axis=1) + 1               # +1 step offset (t+1 based)
        valid = np.isfinite(fut_max)
        r = np.full(m, np.nan)
        tt = np.full(m, np.nan)
        r[valid] = fut_max[valid] - s[valid]
        tt[valid] = arg[valid]
        rise[a:b] = r
        ttp[a:b] = tt
    return rise, ttp
