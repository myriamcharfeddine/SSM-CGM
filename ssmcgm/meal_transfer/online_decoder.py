"""Strictly online causal meal-state decoder.

The decoder is intentionally lightweight and deterministic. At row ``t`` it uses
only the previous filtered state distribution/durations and current causal
features already present in ``causal_student_predictions.parquet``. There is no
full-segment normalization, Viterbi backtracking, future glucose, teacher output,
or event-boundary post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

STEP_MIN = 5
STATE_NAMES = ("none", "possible_onset", "rising", "likely_peak", "recovery")
STATE_TO_CODE = {s: i for i, s in enumerate(STATE_NAMES)}
STATE_PROB_COLS = [f"online_prob_{s}" for s in STATE_NAMES]


@dataclass
class OnlineDecoderConfig:
    """Parameters for the forward-only state estimator."""

    event_threshold: float = 0.50
    event_reset_threshold: float = 0.35
    support_lookback_steps: int = 12
    confidence_temperature: float = 1.0
    min_recovery_steps: int = 2
    transition_matrix: list[list[float]] = field(default_factory=lambda: [
        [0.965, 0.035, 0.000, 0.000, 0.000],
        [0.060, 0.350, 0.570, 0.015, 0.005],
        [0.040, 0.015, 0.675, 0.220, 0.050],
        [0.080, 0.000, 0.040, 0.550, 0.330],
        [0.500, 0.000, 0.060, 0.040, 0.400],
    ])

    @classmethod
    def from_dict(cls, data: dict | None) -> "OnlineDecoderConfig":
        if not data:
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _num(sub: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in sub.columns:
        return np.full(len(sub), default, dtype=np.float64)
    return pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)


def _finite(x: np.ndarray, val: float = 0.0) -> np.ndarray:
    return np.nan_to_num(x, nan=val, posinf=val, neginf=val)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _emission_log_scores(sub: pd.DataFrame) -> np.ndarray:
    """Current-row emission scores for the five states.

    Every term is row-local or trailing-window causal feature. Constants are
    fixed clinical-scale transforms, not segment-level statistics.
    """
    p = np.clip(_finite(_num(sub, "student_meal_probability"), 0.0), 1e-6, 1 - 1e-6)
    s15 = _finite(_num(sub, "cgm_slope_15"), 0.0)
    s30 = _finite(_num(sub, "cgm_slope_30"), 0.0)
    s60 = _finite(_num(sub, "cgm_slope_60"), 0.0)
    accel = _finite(_num(sub, "cgm_accel"), 0.0)
    excursion = _finite(_num(sub, "cgm_pos_excursion_60"), 0.0)
    roll_range = _finite(_num(sub, "cgm_roll_range_60"), 0.0)
    steps30 = _finite(_num(sub, "steps_30"), 0.0)
    sleep = _finite(_num(sub, "sleep_rest"), 0.0)
    hr_change = _finite(_num(sub, "hr_recent_change"), 0.0)
    stress = _finite(_num(sub, "stress_recent"), 0.0)

    log_meal = np.log(p)
    log_none = np.log1p(-p)

    up_fast = np.tanh(s15 / 1.2)
    up_med = np.tanh(s30 / 0.8)
    up_slow = np.tanh(s60 / 0.5)
    accel_pos = np.tanh(accel / 0.6)
    accel_neg = np.tanh(-accel / 0.6)
    flat = 1.0 - np.tanh(np.abs(s15) / 1.0)
    high_recent = np.tanh(excursion / 25.0)
    variability = np.tanh(roll_range / 35.0)

    activity_penalty = 0.35 * np.tanh(steps30 / 900.0) + 0.25 * (sleep > 0).astype(float)
    stress_term = 0.05 * np.tanh((stress - 20.0) / 25.0)
    hr_term = 0.04 * np.tanh(hr_change / 20.0)

    E = np.empty((len(sub), len(STATE_NAMES)), dtype=np.float64)
    E[:, STATE_TO_CODE["none"]] = log_none + 0.70 - 0.20 * high_recent + activity_penalty
    E[:, STATE_TO_CODE["possible_onset"]] = log_meal + 0.45 * accel_pos + 0.25 * up_fast - activity_penalty + hr_term
    E[:, STATE_TO_CODE["rising"]] = log_meal + 0.45 * up_med + 0.25 * up_slow + 0.10 * variability - activity_penalty
    E[:, STATE_TO_CODE["likely_peak"]] = log_meal + 0.45 * high_recent + 0.25 * flat + 0.10 * variability - 0.10 * activity_penalty
    E[:, STATE_TO_CODE["recovery"]] = log_meal + 0.45 * np.tanh(-s30 / 0.8) + 0.20 * accel_neg + 0.15 * high_recent - 0.05 * activity_penalty + stress_term
    return E


def _duration_adjusted_transition(base: np.ndarray, prev_phase: int, phase_duration: int,
                                  event_active: bool, cfg: OnlineDecoderConfig) -> np.ndarray:
    T = base.copy()
    # Duration nudges are based only on the previously emitted MAP state.
    if prev_phase == STATE_TO_CODE["possible_onset"] and phase_duration >= 2:
        T[prev_phase] *= np.array([0.70, 0.45, 1.55, 1.10, 1.00])
    elif prev_phase == STATE_TO_CODE["rising"] and phase_duration >= 6:
        T[prev_phase] *= np.array([0.85, 0.70, 0.65, 1.70, 1.25])
    elif prev_phase == STATE_TO_CODE["likely_peak"] and phase_duration >= 3:
        T[prev_phase] *= np.array([1.10, 0.80, 0.80, 0.65, 1.75])
    elif prev_phase == STATE_TO_CODE["recovery"] and phase_duration >= cfg.min_recovery_steps:
        T[prev_phase] *= np.array([1.65, 0.75, 0.80, 0.80, 0.65])
    if not event_active:
        T[:, STATE_TO_CODE["recovery"]] *= 0.80
    T = T / T.sum(axis=1, keepdims=True)
    return T


def decode_segment_online(sub: pd.DataFrame, cfg: OnlineDecoderConfig | None = None) -> pd.DataFrame:
    """Decode one already-sorted contiguous segment using a causal forward pass."""
    cfg = cfg or OnlineDecoderConfig()
    n = len(sub)
    if n == 0:
        return _empty_output(sub.index)

    E = _emission_log_scores(sub)
    base_T = np.asarray(cfg.transition_matrix, dtype=np.float64)
    base_T = base_T / base_T.sum(axis=1, keepdims=True)

    probs = np.zeros((n, len(STATE_NAMES)), dtype=np.float64)
    phases = np.empty(n, dtype=object)
    phase_prob = np.zeros(n, dtype=np.float64)
    event_active = np.zeros(n, dtype=np.int8)
    time_since = np.full(n, np.nan, dtype=np.float64)
    elapsed_phase = np.zeros(n, dtype=np.float64)
    confidence = np.zeros(n, dtype=np.float64)
    support = np.zeros(n, dtype=np.float64)

    prev = np.zeros(len(STATE_NAMES), dtype=np.float64)
    prev[STATE_TO_CODE["none"]] = 1.0
    prev_phase = STATE_TO_CODE["none"]
    phase_dur = 0
    onset_idx: int | None = None
    active_prev = False
    p_student = np.clip(_finite(_num(sub, "student_meal_probability"), 0.0), 0, 1)

    for i in range(n):
        T = _duration_adjusted_transition(base_T, prev_phase, phase_dur, active_prev, cfg)
        prior = prev @ T
        logp = np.log(np.maximum(prior, 1e-12)) + E[i] / max(cfg.confidence_temperature, 1e-6)
        logp -= np.max(logp)
        cur = np.exp(logp)
        cur /= cur.sum()
        probs[i] = cur

        phase_code = int(np.argmax(cur))
        active_prob = float(cur[1:].sum())
        is_active = bool(active_prob >= cfg.event_threshold and phase_code != STATE_TO_CODE["none"])
        if (not is_active) and active_prev and active_prob >= cfg.event_reset_threshold:
            is_active = True
        if is_active and not active_prev:
            onset_idx = i
        if not is_active:
            onset_idx = None

        if phase_code == prev_phase:
            phase_dur += 1
        else:
            phase_dur = 1
        phases[i] = STATE_NAMES[phase_code]
        phase_prob[i] = float(cur[phase_code])
        event_active[i] = int(is_active)
        elapsed_phase[i] = float(phase_dur * STEP_MIN)
        if onset_idx is not None:
            time_since[i] = float((i - onset_idx) * STEP_MIN)

        lo = max(0, i - cfg.support_lookback_steps + 1)
        support[i] = float(np.nanmax(p_student[lo:i + 1]))
        entropy = -float(np.sum(cur * np.log(np.maximum(cur, 1e-12)))) / np.log(len(STATE_NAMES))
        confidence[i] = float(np.clip((1.0 - entropy) * (0.5 + 0.5 * support[i]), 0, 1))

        prev = cur
        prev_phase = phase_code
        active_prev = is_active

    out = pd.DataFrame({
        "online_meal_probability": probs[:, 1:].sum(axis=1),
        "online_phase": phases,
        "online_phase_code": [STATE_TO_CODE[x] for x in phases],
        "online_phase_probability": phase_prob,
        "online_event_active": event_active.astype(float),
        "online_time_since_onset": time_since,
        "online_elapsed_phase_duration": elapsed_phase,
        "online_confidence": confidence,
        "online_support_score": support,
    }, index=sub.index)
    for j, c in enumerate(STATE_PROB_COLS):
        out[c] = probs[:, j]
    return out


def _empty_output(index: pd.Index) -> pd.DataFrame:
    cols = [
        "online_meal_probability", "online_phase", "online_phase_code",
        "online_phase_probability", "online_event_active", "online_time_since_onset",
        "online_elapsed_phase_duration", "online_confidence", "online_support_score",
        *STATE_PROB_COLS,
    ]
    return pd.DataFrame({c: [] for c in cols}, index=index)


def decode_dataframe_online(df: pd.DataFrame, cfg: OnlineDecoderConfig | None = None,
                            segment_col: str = "segment_id") -> pd.DataFrame:
    """Decode a full table per contiguous segment.

    The returned frame includes the input identifier columns when present plus
    online-state columns aligned exactly to ``df`` rows.
    """
    cfg = cfg or OnlineDecoderConfig()
    pieces = []
    for _seg, sub in df.groupby(segment_col, sort=False):
        pieces.append(decode_segment_online(sub, cfg))
    states = pd.concat(pieces).loc[df.index]
    id_cols = [c for c in ["participant_id", "ts", "ds", "split", "segment_id", "cgm_glucose", "med_insulin", "student_meal_probability"] if c in df.columns]
    return pd.concat([df[id_cols].copy(), states], axis=1)


def assert_prefix_invariance(df: pd.DataFrame, cfg: OnlineDecoderConfig | None = None,
                             segment_col: str = "segment_id", max_segments: int = 12,
                             cuts: Iterable[float] = (0.25, 0.5, 0.75, 1.0)) -> dict:
    """Fail if decoding a full segment disagrees with decoding any prefix.

    This is the core leakage gate for the online decoder.
    """
    cfg = cfg or OnlineDecoderConfig()
    checked = 0
    max_abs_diff = 0.0
    state_cols = [
        "online_meal_probability", "online_phase_probability", "online_event_active",
        "online_time_since_onset", "online_elapsed_phase_duration", "online_confidence",
        "online_support_score", *STATE_PROB_COLS,
    ]
    for _seg, sub in df.groupby(segment_col, sort=False):
        if len(sub) < 24:
            continue
        full = decode_segment_online(sub, cfg)
        for frac in cuts:
            n = max(1, int(round(len(sub) * float(frac))))
            prefix = decode_segment_online(sub.iloc[:n], cfg)
            a = full.iloc[:n]
            if not (a["online_phase"].to_numpy() == prefix["online_phase"].to_numpy()).all():
                raise AssertionError(f"prefix invariance failed for online_phase at segment {_seg}, n={n}")
            for c in state_cols:
                av = pd.to_numeric(a[c], errors="coerce").to_numpy(dtype=np.float64)
                bv = pd.to_numeric(prefix[c], errors="coerce").to_numpy(dtype=np.float64)
                diff = np.nanmax(np.abs(np.nan_to_num(av, nan=-9999.0) - np.nan_to_num(bv, nan=-9999.0)))
                max_abs_diff = max(max_abs_diff, float(diff))
                if diff > 1e-12:
                    raise AssertionError(f"prefix invariance failed for {c} at segment {_seg}, n={n}, maxdiff={diff}")
        checked += 1
        if checked >= max_segments:
            break
    if checked == 0:
        raise AssertionError("prefix invariance test found no eligible segments")
    return {"prefix_invariance_passed": True, "segments_checked": checked, "max_abs_diff": max_abs_diff}


def feature_provenance_rows(include_response_size: bool = True) -> list[dict]:
    rows = [
        ("student_meal_probability", "causal_student_predictions.student_meal_probability", "0 min plus training labels offline", False, False, False, True),
        ("online_meal_probability", "forward filter state probabilities from current/past causal inputs", "causal recursion", False, False, False, True),
        ("online_phase_code", "online_decoder forward MAP phase", "causal recursion", False, False, False, True),
        ("online_phase_probability", "online_decoder current filtered phase probability", "causal recursion", False, False, False, True),
        ("online_event_active", "online_decoder current filtered active probability threshold", "causal recursion", False, False, False, True),
        ("online_time_since_onset", "online_decoder forward onset trigger", "causal recursion since trigger", False, False, False, True),
        ("online_elapsed_phase_duration", "online_decoder forward MAP phase duration", "causal recursion", False, False, False, True),
        ("online_confidence", "current filtered-state entropy and trailing student probability support", "60 min trailing", False, False, False, True),
        ("online_support_score", "trailing max student probability", "60 min trailing", False, False, False, True),
        ("online_prob_none", "online_decoder filtered state probability", "causal recursion", False, False, False, True),
        ("online_prob_possible_onset", "online_decoder filtered state probability", "causal recursion", False, False, False, True),
        ("online_prob_rising", "online_decoder filtered state probability", "causal recursion", False, False, False, True),
        ("online_prob_likely_peak", "online_decoder filtered state probability", "causal recursion", False, False, False, True),
        ("online_prob_recovery", "online_decoder filtered state probability", "causal recursion", False, False, False, True),
    ]
    if include_response_size:
        rows.extend([
            ("predicted_response_size", "online_size_model predicted class from causal online/current features", "60 min trailing plus causal recursion", False, False, False, True),
            ("response_size_prob_none", "online_size_model class probability", "60 min trailing plus causal recursion", False, False, False, True),
            ("response_size_prob_small", "online_size_model class probability", "60 min trailing plus causal recursion", False, False, False, True),
            ("response_size_prob_medium", "online_size_model class probability", "60 min trailing plus causal recursion", False, False, False, True),
            ("response_size_prob_large", "online_size_model class probability", "60 min trailing plus causal recursion", False, False, False, True),
            ("expected_response_size_score", "online_size_model class probabilities small=1, medium=2, large=3", "60 min trailing plus causal recursion", False, False, False, True),
            ("response_size_confidence", "max online_size_model class probability", "60 min trailing plus causal recursion", False, False, False, True),
        ])
    return [
        {
            "feature_name": name,
            "source_columns": source,
            "maximum_lookback": lookback,
            "uses_future_glucose": "yes" if fg else "no",
            "uses_future_wearables": "yes" if fw else "no",
            "uses_teacher_output_at_inference": "yes" if teach else "no",
            "available_online": "yes" if online else "no",
        }
        for name, source, lookback, fg, fw, teach, online in rows
    ]
