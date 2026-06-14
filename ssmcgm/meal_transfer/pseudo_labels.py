"""Phase D — weak AI-READI pseudo-labels.

Three sources of evidence are combined (per the brief / Stage 3):
  E1  CGMacros teacher probability (retrospective, non-causal) — if available.
  E2  retrospective future-glucose response (offline only).
  E3  competing-event exclusions (activity, sleep transition, hypo recovery).

Outputs per row:
  * ``meal_pseudo_label``        in {positive, negative, uncertain}
  * ``pseudo_label_confidence``  in [0, 1]
  * ``meal_response_size_proxy`` in {none, small, medium, large}

All cut points are **quantiles fit on the TRAIN split** (Phase D requirement),
not hardcoded glucose values. ``med_insulin == 0`` is the primary labelling
cohort; for ``med_insulin == 1`` the absence of a glucose rise is NOT taken as a
definite non-meal, confidence is down-weighted, and a ``hidden_insulin_risk``
indicator is set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PseudoLabelConfig
from .features import add_future_response


@dataclass
class PseudoLabelThresholds:
    """Absolute cut points derived from TRAIN-split quantiles."""

    resp_pos: float            # future peak rise marking a strong response
    resp_neg: float            # |response| below this is "flat"
    teacher_pos: float | None  # teacher prob high
    teacher_low: float | None  # teacher prob low
    size_small: float          # response-size tertile cut 1
    size_medium: float         # response-size tertile cut 2
    teacher_available: bool

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def fit_thresholds(train_df: pd.DataFrame, cfg: PseudoLabelConfig) -> PseudoLabelThresholds:
    """Fit quantile cut points on the TRAIN split (non-insulin cohort drives the
    response quantiles, since insulin can mask rises)."""
    prim = train_df[train_df.get("med_insulin", 0).fillna(0) == 0]
    if len(prim) < 1000:
        prim = train_df  # fall back if the primary cohort is tiny (smoke runs)

    resp = prim["future_peak_rise_120"].dropna()
    resp_pos = float(resp.quantile(cfg.pos_response_q)) if len(resp) else 20.0
    # "flat" cut uses the absolute response magnitude.
    resp_neg = float(resp.abs().quantile(cfg.neg_response_q)) if len(resp) else 8.0

    teacher_available = (
        "cgmacros_teacher_probability" in train_df.columns
        and train_df["cgmacros_teacher_probability"].notna().any()
    )
    if teacher_available:
        tp = train_df["cgmacros_teacher_probability"].dropna()
        teacher_pos = float(tp.quantile(cfg.teacher_pos_q))
        teacher_low = float(tp.quantile(cfg.teacher_low_q))
    else:
        teacher_pos = teacher_low = None

    # Response-size tertiles over *positive-response* events on the primary cohort.
    pos_resp = resp[resp > 0]
    if len(pos_resp) >= 10:
        q1, q2 = cfg.size_quantiles
        size_small = float(pos_resp.quantile(q1))
        size_medium = float(pos_resp.quantile(q2))
    else:
        size_small, size_medium = 15.0, 35.0

    return PseudoLabelThresholds(
        resp_pos=resp_pos, resp_neg=resp_neg,
        teacher_pos=teacher_pos, teacher_low=teacher_low,
        size_small=size_small, size_medium=size_medium,
        teacher_available=teacher_available,
    )


def _competing_exclusions(df: pd.DataFrame, cfg: PseudoLabelConfig) -> pd.Series:
    """Boolean: a strong competing event makes a meal interpretation unsafe."""
    excl = pd.Series(False, index=df.index)

    # High / endurance-like activity around t.
    if "steps_30" in df.columns:
        excl |= df["steps_30"].fillna(0) >= cfg.activity_steps_per_5min * 6  # ~30min of steps
    if "hr_recent_mean" in df.columns and df["hr_recent_mean"].notna().any():
        hr = df["hr_recent_mean"]
        hr_z = (hr - hr.mean()) / (hr.std() + 1e-6)
        excl |= hr_z >= cfg.hr_zscore_high

    # Sleep / rest: meals don't start mid-sleep; treat as competing.
    if "sleep_rest" in df.columns:
        excl |= df["sleep_rest"].fillna(0) > 0

    # Recovery out of hypoglycaemia: a rise from <70 mg/dL is counter-regulation,
    # not necessarily a meal.
    if "cgm_roll_mean_30" in df.columns:
        excl |= df["cgm_roll_mean_30"].fillna(df["cgm_glucose"]) < cfg.hypo_recovery_mgdl

    return excl


def build_pseudo_labels(
    df: pd.DataFrame,
    thr: PseudoLabelThresholds,
    cfg: PseudoLabelConfig,
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """Attach pseudo-label columns to ``df`` (must already have causal features
    and a ``segment_id``). Future-response signals (offline-only) are reused if
    already present, else computed here. Returns a copy."""
    if "future_peak_rise_120" in df.columns:
        out = df.copy()
    else:
        out = add_future_response(
            df, horizons_min=cfg.response_horizons_min,
            peak_window_min=cfg.peak_window_min, segment_col=segment_col,
        )

    insulin = out.get("med_insulin", pd.Series(0.0, index=out.index)).fillna(0.0) == 1
    out["hidden_insulin_risk"] = insulin.astype(float)

    resp = out["future_peak_rise_120"]
    has_future = resp.notna()
    excl = _competing_exclusions(out, cfg)

    # Teacher agreement (optional).
    if thr.teacher_available:
        tprob = out["cgmacros_teacher_probability"]
        teacher_high = tprob >= thr.teacher_pos
        teacher_low = tprob <= thr.teacher_low
    else:
        teacher_high = pd.Series(False, index=out.index)
        teacher_low = pd.Series(True, index=out.index)  # neutral: never blocks negatives

    # --- Positive: strong future rise, no competing event. ---
    strong_rise = resp >= thr.resp_pos
    positive = has_future & strong_rise & (~excl)

    # --- Negative: flat response AND (teacher low OR teacher unavailable). ---
    flat = resp.abs() <= thr.resp_neg
    negative = has_future & flat & teacher_low & (~insulin)  # insulin: see below

    # --- Insulin cohort: absence of rise is NOT a definite non-meal. ---
    # Only positives (clear rise despite insulin) are labelled; flats -> uncertain.
    negative = negative & (~insulin)

    label = np.where(positive, "positive",
                     np.where(negative, "negative", "uncertain"))
    out["meal_pseudo_label"] = label

    # --- Confidence (0..1): blend response strength + teacher agreement, scaled
    #     down for insulin participants and zeroed where no future is available. ---
    resp_norm = (resp / (thr.resp_pos + 1e-6)).clip(lower=0, upper=2.0) / 2.0
    if thr.teacher_available:
        tconf = ((out["cgmacros_teacher_probability"] - thr.teacher_low)
                 / (thr.teacher_pos - thr.teacher_low + 1e-6)).clip(0, 1).fillna(0)
        conf_pos = 0.6 * resp_norm + 0.4 * tconf
    else:
        conf_pos = resp_norm
    # Negative confidence: how flat (closer to 0 response => higher confidence).
    conf_neg = (1.0 - (resp.abs() / (thr.resp_neg + 1e-6)).clip(0, 1))

    conf = np.where(out["meal_pseudo_label"] == "positive", conf_pos,
                    np.where(out["meal_pseudo_label"] == "negative", conf_neg, 0.0))
    conf = pd.Series(conf, index=out.index).clip(0, 1)
    conf = conf.where(~insulin, conf * cfg.insulin_confidence_scale)
    conf = conf.where(has_future, 0.0)
    out["pseudo_label_confidence"] = conf.astype(float)

    # --- Response-size proxy (offline; train-split tertiles). ---
    size = np.full(len(out), "none", dtype=object)
    rise = resp.to_numpy()
    pos_mask = (out["meal_pseudo_label"] == "positive").to_numpy()
    small = pos_mask & (rise < thr.size_small)
    medium = pos_mask & (rise >= thr.size_small) & (rise < thr.size_medium)
    large = pos_mask & (rise >= thr.size_medium)
    size[small] = "small"
    size[medium] = "medium"
    size[large] = "large"
    out["meal_response_size_proxy"] = size

    return out


def add_soft_onset_labels(
    df: pd.DataFrame,
    cfg: PseudoLabelConfig,
    *,
    tau_steps: float = 6.0,
    lead_steps: int = 12,
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """Block 4 - rescue the uncertain rows with a soft / ordinal onset-distance
    target instead of discarding them.

    The single most informative region for a glucose forecaster is the pre-rise
    and onset transition, which the hard {positive, negative, uncertain} scheme
    throws away (the uncertain bucket). Here every row gets a continuous target
    ``meal_soft_target`` in [0, 1] equal to its proximity to an upcoming meal
    onset:

      * positive rows                      -> 1.0 (scaled by their confidence),
      * rows up to ``lead_steps`` before an onset -> exp(-steps_to_onset / tau),
      * clearly negative (flat) rows       -> 0.0,
      * everything else                    -> the onset-proximity value.

    A row is a meal *onset* when it starts a contiguous run of positive
    pseudo-labels inside its segment. Insulin users are quarantined from
    label training: ``label_train_eligible`` is 0 for them (Block 4), so the
    student never fits their unobserved-insulin dynamics, and they are reported
    separately as low-confidence meal-like events.
    """
    out = df.copy()
    pos = (out["meal_pseudo_label"] == "positive").to_numpy()
    neg = (out["meal_pseudo_label"] == "negative").to_numpy()
    seg = out[segment_col].to_numpy()
    conf = out.get("pseudo_label_confidence", pd.Series(0.0, index=out.index)).fillna(0.0).to_numpy()

    # onset = first positive of each positive run within a segment
    prev_pos = np.r_[False, pos[:-1]]
    prev_seg = np.r_[seg[0], seg[:-1]]
    onset = pos & (~(prev_pos & (prev_seg == seg)))

    n = len(out)
    soft = np.where(pos, np.clip(0.5 + 0.5 * conf, 0.0, 1.0), 0.0)
    # propagate onset proximity backwards up to lead_steps (pre-rise transition)
    onset_idx = np.flatnonzero(onset)
    for oi in onset_idx:
        s = seg[oi]
        lo = max(0, oi - lead_steps)
        for j in range(oi - 1, lo - 1, -1):
            if seg[j] != s:
                break
            val = float(np.exp(-(oi - j) / tau_steps))
            if val > soft[j]:
                soft[j] = val
    soft[neg] = 0.0
    out["meal_soft_target"] = np.clip(soft, 0.0, 1.0)

    insulin = out.get("med_insulin", pd.Series(0.0, index=out.index)).fillna(0.0).to_numpy() == 1
    has_future = out["future_peak_rise_120"].notna().to_numpy() if "future_peak_rise_120" in out else np.ones(n, bool)
    out["label_train_eligible"] = ((~insulin) & has_future).astype(int)
    out["insulin_quarantine"] = insulin.astype(int)
    return out
