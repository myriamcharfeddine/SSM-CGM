"""Phase F — structured meal-state decoder.

A duration-constrained Viterbi (a hidden semi-Markov model expressed through
minimum-dwell sub-state expansion) over five states:

    0 none -> 1 onset -> 2 rising -> 3 peak -> 4 recovery -> 0 none

Each macro-state must persist for at least its configured minimum dwell, which
removes the flicker of point-wise flags (the brief's motivation: an explicit
duration term forbids implausible 5-minute meal events). Emissions combine the
**causal student probability** (meal vs none) with the CGM slope/acceleration
(which separates the within-meal phases). The decoder cleans the *flag and
phase*; it does not invent probability — ``meal_probability`` is the student's
causal probability passed through.

Outputs (per row): ``meal_probability``, ``predmeal_flag_clean``,
``postprandial_phase``, ``time_since_predicted_meal`` (minutes),
``meal_response_size_proxy``, ``meal_confidence``, ``meal_support_score``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DecoderConfig

STEP_MIN = 5
NEG_INF = -1e18

# Allowed macro-state transitions out of the LAST sub-state of each macro-state.
_ALLOWED_NEXT = {
    0: (0, 1),        # none -> none | onset
    1: (1, 2),        # onset -> onset | rising
    2: (2, 3, 4),     # rising -> rising | peak | recovery
    3: (3, 4),        # peak -> peak | recovery
    4: (4, 0),        # recovery -> recovery | none
}


def _emissions(p_student: np.ndarray, slope: np.ndarray, accel: np.ndarray,
               level_z: np.ndarray, cfg: DecoderConfig) -> np.ndarray:
    """Per-step log emission score for each of the 5 macro-states, shape (T,5).

    Meal-vs-`none` is governed purely by the student probability; the per-phase
    shape terms are **centered** across the four meal phases (zero row-mean) and
    scaled, so they only redistribute mass among onset/rising/peak/recovery and
    cannot inflate total meal evidence against `none`.
    """
    eps = 1e-6
    p = np.clip(p_student, eps, 1 - eps)
    meal = np.log(p)
    none = np.log(1 - p) + cfg.none_bias

    # Normalise slope/accel to a robust unit scale (safe for very short segments).
    def _std(x):
        if np.sum(np.isfinite(x)) < 2:
            return 1.0
        v = np.nanstd(x)
        return v if v > eps else 1.0

    s = slope / _std(slope)
    a = accel / _std(accel)
    s = np.nan_to_num(s); a = np.nan_to_num(a); lz = np.nan_to_num(level_z)

    # Raw per-phase preferences (higher = more characteristic of that phase).
    # Peak is LEVEL-dominant (high glucose + flat slope) so it cannot be assigned
    # to low pre-rise plateaus; onset/rising own the low-glucose upswing.
    onset = np.tanh(a)                                  # accelerating (curvature up)
    rising = np.tanh(s) + 0.3 * np.tanh(a)              # sustained positive slope
    peak = np.tanh(lz) + 0.4 * (1.0 - np.tanh(np.abs(s)))  # high & flat
    recovery = np.tanh(-s)                              # negative slope
    shapes = np.stack([onset, rising, peak, recovery], axis=1)   # (T,4)
    shapes = shapes - shapes.mean(axis=1, keepdims=True)         # center -> zero-sum
    shapes *= cfg.phase_shape_scale

    E = np.empty((len(p), 5), dtype=np.float64)
    E[:, 0] = none
    E[:, 1:] = meal[:, None] + shapes
    return E


def _expand_substates(cfg: DecoderConfig):
    """Build sub-state -> macro map and the legal sub-state transition list."""
    dwell = cfg.min_dwell_steps
    sub_macro = []          # macro index for each sub-state
    sub_index = []          # position within the macro chain
    offsets = {}            # macro -> first sub-state id
    for m, d in enumerate(dwell):
        offsets[m] = len(sub_macro)
        for k in range(d):
            sub_macro.append(m)
            sub_index.append(k)
    n_sub = len(sub_macro)

    # transitions[j] = list of reachable sub-states from sub-state j
    transitions = [[] for _ in range(n_sub)]
    for j in range(n_sub):
        m, k = sub_macro[j], sub_index[j]
        last_k = dwell[m] - 1
        if k < last_k:
            transitions[j].append((offsets[m] + k + 1, m))   # forced advance, same macro
        else:
            # at the end of the dwell chain: self-loop or jump to next macro's start
            for mn in _ALLOWED_NEXT[m]:
                if mn == m:
                    transitions[j].append((j, m))             # self-loop (stay)
                else:
                    transitions[j].append((offsets[mn], mn))  # enter next macro
    return np.array(sub_macro), offsets, transitions, n_sub


def _viterbi_segment(E: np.ndarray, cfg: DecoderConfig) -> np.ndarray:
    """Min-dwell Viterbi; returns a length-T macro-state sequence."""
    sub_macro, offsets, transitions, n_sub = _expand_substates(cfg)
    T = E.shape[0]

    # transition score helper
    def trans_score(src_macro, dst_macro):
        if src_macro == dst_macro:
            return cfg.self_transition_bonus
        if src_macro == 0 and dst_macro == 1:
            return -cfg.onset_penalty   # discourage starting an event
        return 0.0

    delta = np.full((T, n_sub), NEG_INF)
    psi = np.full((T, n_sub), -1, dtype=np.int64)

    # Init: must start in `none` chain start (or any none sub-state start).
    start = offsets[0]
    delta[0, start] = E[0, 0]
    # also permit starting already inside none chain end (long prior none)
    delta[0, start + (cfg.min_dwell_steps[0] - 1)] = E[0, 0]

    for t in range(1, T):
        for j in range(n_sub):
            if delta[t - 1, j] <= NEG_INF / 2:
                continue
            base = delta[t - 1, j]
            for (jn, dst_macro) in transitions[j]:
                sc = base + trans_score(sub_macro[j], dst_macro) + E[t, dst_macro]
                if sc > delta[t, jn]:
                    delta[t, jn] = sc
                    psi[t, jn] = j

    # Backtrack from best final sub-state.
    last = int(np.argmax(delta[T - 1]))
    path = np.empty(T, dtype=np.int64)
    j = last
    for t in range(T - 1, -1, -1):
        path[t] = sub_macro[j]
        j = psi[t, j] if psi[t, j] >= 0 else j
    return path


def _event_spans(states: np.ndarray):
    """Yield (start, end_inclusive) for each maximal non-`none` run."""
    spans = []
    in_evt = False
    s = 0
    for i, v in enumerate(states):
        if v != 0 and not in_evt:
            in_evt, s = True, i
        elif v == 0 and in_evt:
            spans.append((s, i - 1)); in_evt = False
    if in_evt:
        spans.append((s, len(states) - 1))
    return spans


def decode_segment(
    cgm: np.ndarray,
    p_student: np.ndarray,
    slope: np.ndarray,
    accel: np.ndarray,
    cfg: DecoderConfig,
    size_small: float,
    size_medium: float,
) -> dict:
    """Decode one contiguous segment; return per-step output arrays."""
    T = len(cgm)
    level_z = (cgm - np.nanmean(cgm)) / (np.nanstd(cgm) + 1e-6)
    E = _emissions(p_student, slope, accel, level_z, cfg)
    states = _viterbi_segment(E, cfg)

    flag_clean = (states != 0).astype(float)
    phase = np.array(cfg.states, dtype=object)[states]

    # time since the onset of the current/most-recent started event (minutes).
    time_since = np.full(T, np.nan)
    spans = _event_spans(states)
    last_onset = None
    span_ptr = 0
    onsets = [s for (s, _e) in spans]
    for t in range(T):
        while span_ptr < len(onsets) and onsets[span_ptr] <= t:
            last_onset = onsets[span_ptr]
            span_ptr += 1
        if last_onset is not None:
            time_since[t] = (t - last_onset) * STEP_MIN

    # Per-event response-size proxy from the realised rise inside the event,
    # plus an event-level confidence and support score.
    size = np.full(T, "none", dtype=object)
    confidence = np.zeros(T)
    support = np.zeros(T)
    for (s, e) in spans:
        seg_cgm = cgm[s:e + 1]
        rise = float(np.nanmax(seg_cgm) - seg_cgm[0]) if len(seg_cgm) else 0.0
        if rise < size_small:
            lab = "small"
        elif rise < size_medium:
            lab = "medium"
        else:
            lab = "large"
        size[s:e + 1] = lab
        conf = float(np.nanmean(p_student[s:e + 1]))
        confidence[s:e + 1] = conf
        # support: peak student prob within the event (independent corroboration).
        support[s:e + 1] = float(np.nanmax(p_student[s:e + 1]))

    return {
        "meal_probability": p_student,
        "predmeal_flag_clean": flag_clean,
        "postprandial_phase": phase,
        "time_since_predicted_meal": time_since,
        "meal_response_size_proxy": size,
        "meal_confidence": confidence,
        "meal_support_score": support,
        "decoded_state": states,
    }


def decode_dataframe(
    df: pd.DataFrame,
    p_student: np.ndarray,
    cfg: DecoderConfig,
    size_small: float,
    size_medium: float,
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """Run the decoder per segment over a full frame. ``df`` must be sorted and
    carry ``segment_id`` plus causal CGM features. Returns a frame of outputs
    aligned to ``df.index``."""
    out_cols = {k: np.empty(len(df), dtype=object) if k in
                ("postprandial_phase", "meal_response_size_proxy")
                else np.full(len(df), np.nan)
                for k in ("meal_probability", "predmeal_flag_clean",
                          "postprandial_phase", "time_since_predicted_meal",
                          "meal_response_size_proxy", "meal_confidence",
                          "meal_support_score", "decoded_state")}

    cgm_all = df["cgm_glucose"].to_numpy(np.float64)
    slope_all = df.get("cgm_slope_15", pd.Series(np.nan, index=df.index)).to_numpy(np.float64)
    accel_all = df.get("cgm_accel", pd.Series(np.nan, index=df.index)).to_numpy(np.float64)
    pos = {ix: i for i, ix in enumerate(df.index)}

    for _, sub in df.groupby(segment_col, sort=False):
        idx = sub.index.to_numpy()
        rows = np.array([pos[i] for i in idx])
        res = decode_segment(
            cgm_all[rows], p_student[rows], slope_all[rows], accel_all[rows],
            cfg, size_small, size_medium,
        )
        for k, v in res.items():
            out_cols[k][rows] = v

    return pd.DataFrame(out_cols, index=df.index)
