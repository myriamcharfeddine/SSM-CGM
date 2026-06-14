"""Step 2 + Step 3 — lightweight downstream forecast test.

A small residual-correction model is fit on top of the *cached* SSM-CGM-Stream
baseline forecast (``q50``):

    y_hat_meal_{t+h} = q50_{t+h} + r(C_t, M_t, h)

where ``C_t`` is static/baseline context and ``M_t`` is the meal representation at
the anchor (read at ``anchor_ds``). Six setups A-F vary only ``M_t``. Models are
fit on the **validation** cache (239 held-out participants) and evaluated on the
**test** cache (221 disjoint held-out participants), so there is no participant
leakage. Every retained (anchor, horizon) row has an **exact** baseline-target /
meal-state match (enforced after alignment).

Step 3 negative controls (shuffle / time-shift / block-shuffle) corrupt ``M_t`` at
evaluation time; a real gain must not survive them.

Baseline horizon is 60 min (the only cached horizon); 120-min metrics are out of
scope here (see ``baseline_cache_120min_generation_report.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OUTPUT = None  # set by runner
HYPER_MGDL = 180.0
EXACT_TOL = 0.5
Z90_MINUS_Z10 = 2.5631031  # for sigma ~ (q90 - q10) / this

# Meal feature columns per setup (M_t). 'A' adds nothing beyond context.
SETUP_FEATURES = {
    "A_none": [],
    "B_old_predmeal_flag": ["predmeal_flag"],
    "C_teacher_prob": ["cgmacros_teacher_probability"],
    "D_student_prob": ["student_meal_probability"],
    "E_prob_phase_time": ["student_meal_probability", "phase_code",
                          "time_since_predicted_meal"],
    "F_full_state": ["student_meal_probability", "phase_code",
                     "time_since_predicted_meal", "response_size_code",
                     "meal_confidence", "meal_support_score"],
}

# Context features in EVERY setup (baseline shape + static; not meal info).
CONTEXT_FEATURES = ["q50", "spread", "horizon_step", "hba1c_percent_baseline",
                    "bmi_baseline", "med_insulin", "med_any_diabetes_drug",
                    "study_group_code", "site_code"]

# Columns that are the meal representation (targets of the negative controls).
ALL_MEAL_COLS = sorted(set(c for v in SETUP_FEATURES.values() for c in v))

_PHASES = ["none", "onset", "rising", "peak", "recovery"]
_SIZES = ["none", "small", "medium", "large"]


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #
def build_meal_feature_table(meal_dir) -> pd.DataFrame:
    """Per-(participant_id, ds) meal features used as M_t and for meal-window."""
    from pathlib import Path
    meal_dir = Path(meal_dir)
    teach = pd.read_parquet(meal_dir / "teacher_predictions.parquet",
                            columns=["participant_id", "ds", "cgmacros_teacher_probability",
                                     "predmeal_flag"])
    states = pd.read_parquet(meal_dir / "passive_meal_states.parquet",
                             columns=["participant_id", "ds", "segment_id", "cgm_glucose", "meal_probability",
                                      "predmeal_flag_clean", "postprandial_phase",
                                      "time_since_predicted_meal", "decoded_response_size",
                                      "meal_confidence", "meal_support_score"])
    df = teach.merge(states, on=["participant_id", "ds"], how="outer")
    df["participant_id"] = df["participant_id"].astype(str)
    df = df.rename(columns={"meal_probability": "student_meal_probability"})
    df["phase_code"] = df["postprandial_phase"].map({p: i for i, p in enumerate(_PHASES)}).fillna(0)
    df["response_size_code"] = df["decoded_response_size"].map(
        {s: i for i, s in enumerate(_SIZES)}).fillna(0)
    df["time_since_predicted_meal"] = df["time_since_predicted_meal"].fillna(-1.0)
    return df


def attach(aligned: pd.DataFrame, meal: pd.DataFrame, *, cat_maps: dict) -> pd.DataFrame:
    """Attach M_t at anchor_ds and meal-window flag at target_ds; enforce exact
    baseline-target / meal-state match. Returns the modelling frame."""
    df = aligned.copy()
    df["participant_id"] = df["participant_id"].astype(str)
    df["anchor_ds"] = df["anchor_ds"].astype(int)
    df["target_ds"] = df["anchor_ds"] + df["horizon_step"].astype(int)

    # exact match: target == meal-state cgm at target_ds
    cgm = meal.set_index(["participant_id", "ds"])["cgm_glucose"]
    key_t = list(zip(df["participant_id"], df["target_ds"]))
    df["_mcgm"] = cgm.reindex(key_t).to_numpy()
    df = df[np.isfinite(df["_mcgm"]) & (np.abs(df["target"] - df["_mcgm"]) <= EXACT_TOL)].copy()

    # strict alignment gate: anchor and target must live in the same contiguous
    # CGM segment. This drops only boundary rows but prevents segment-crossing joins.
    seg = meal.set_index(["participant_id", "ds"])["segment_id"]
    df["_anchor_segment"] = seg.reindex(list(zip(df["participant_id"], df["anchor_ds"]))).to_numpy()
    df["_target_segment"] = seg.reindex(list(zip(df["participant_id"], df["target_ds"]))).to_numpy()
    same_segment = (pd.notna(df["_anchor_segment"]) & pd.notna(df["_target_segment"])
                    & (df["_anchor_segment"].to_numpy() == df["_target_segment"].to_numpy()))
    df = df[same_segment].copy()

    # meal-window flag at target_ds (is the forecast point inside a decoded event)
    flag = meal.set_index(["participant_id", "ds"])["predmeal_flag_clean"]
    df["in_meal_window"] = (flag.reindex(list(zip(df["participant_id"], df["target_ds"])))
                            .fillna(0).to_numpy() > 0)

    # M_t at anchor_ds
    mfeat = meal.set_index(["participant_id", "ds"])
    keya = list(zip(df["participant_id"], df["anchor_ds"]))
    for c in ALL_MEAL_COLS:
        df[c] = mfeat[c].reindex(keya).to_numpy()

    # context
    df["spread"] = (df["q90"] - df["q10"]).clip(lower=1e-3)
    df["study_group_code"] = df["participants_study_group"].map(cat_maps["study_group"]).fillna(-1)
    df["site_code"] = df["participants_clinical_site"].map(cat_maps["site"]).fillna(-1)
    df["residual"] = df["target"] - df["q50"]
    return df


# --------------------------------------------------------------------------- #
# residual model
# --------------------------------------------------------------------------- #
def _design(df: pd.DataFrame, setup: str) -> np.ndarray:
    cols = CONTEXT_FEATURES + SETUP_FEATURES[setup]
    return df[cols].to_numpy(dtype=np.float64)


def fit_residual(train_df: pd.DataFrame, setup: str, seed: int = 42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = _design(train_df, setup)
    y = train_df["residual"].to_numpy(dtype=np.float64)
    m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_depth=6,
                                      l2_regularization=1.0, random_state=seed)
    m.fit(X, y)
    return m


def apply_correction(df: pd.DataFrame, model, setup: str) -> np.ndarray:
    return df["q50"].to_numpy() + model.predict(_design(df, setup))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _phi(z):
    from math import erf, sqrt
    # vectorised standard normal CDF
    return 0.5 * (1.0 + np.vectorize(lambda x: erf(x / sqrt(2.0)))(z))


def forecast_metrics(df: pd.DataFrame, corrected: np.ndarray) -> dict:
    """All point/peak/hyperglycemia metrics for one corrected forecast."""
    out = {}
    err = np.abs(corrected - df["target"].to_numpy())
    h = df["horizon_step"].to_numpy()
    out["MAE_30min"] = float(err[h == 6].mean())
    out["MAE_60min"] = float(err[h == 12].mean())
    out["MAE_all"] = float(err.mean())
    mw = df["in_meal_window"].to_numpy()
    out["MAE_meal_window"] = float(err[mw].mean()) if mw.any() else np.nan
    out["meal_window_rows"] = int(mw.sum())

    # peak error over the 1h window, per anchor
    tmp = df[["participant_id", "anchor_ds", "horizon_step", "target"]].copy()
    tmp["corr"] = corrected
    grp = tmp.groupby(["participant_id", "anchor_ds"])
    peak_t = grp["target"].max()
    peak_p = grp["corr"].max()
    out["peak_error_1h"] = float(np.abs(peak_t - peak_p).mean())

    # hyperglycemia at 60-min horizon
    m12 = h == 12
    if m12.any():
        sigma = (df["spread"].to_numpy()[m12]) / Z90_MINUS_Z10
        p_hyper = 1.0 - _phi((HYPER_MGDL - corrected[m12]) / np.maximum(sigma, 1e-6))
        label = (df["target"].to_numpy()[m12] >= HYPER_MGDL).astype(int)
        out.update(_hyper_metrics(label, p_hyper))
    return out


def _hyper_metrics(label: np.ndarray, score: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    out = {"hyper_prevalence": float(label.mean())}
    if label.min() == label.max():
        out.update(hyper_AUROC=np.nan, hyper_AUPRC=np.nan, hyper_brier=np.nan, hyper_ECE=np.nan)
        return out
    out["hyper_AUROC"] = float(roc_auc_score(label, score))
    out["hyper_AUPRC"] = float(average_precision_score(label, score))
    out["hyper_brier"] = float(brier_score_loss(label, score))
    # expected calibration error (10 bins)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(score, bins) - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(score[sel].mean() - label[sel].mean())
    out["hyper_ECE"] = float(ece)
    return out


# --------------------------------------------------------------------------- #
# negative controls
# --------------------------------------------------------------------------- #
def corrupt_meal_features(df: pd.DataFrame, mode: str, seed: int = 0) -> pd.DataFrame:
    """Return a copy with the meal columns corrupted per ``mode``."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    cols = ALL_MEAL_COLS
    if mode == "shuffle":
        perm = rng.permutation(len(out))
        out[cols] = out[cols].to_numpy()[perm]
    elif mode == "time_shift":
        # shift meal features by +2 h (24 anchor steps) within each participant
        for _pid, sub in out.groupby("participant_id"):
            ridx = sub.sort_values("anchor_ds").index
            shifted = out.loc[ridx, cols].shift(24)
            out.loc[ridx, cols] = shifted.to_numpy()
        out[cols] = out[cols].bfill().fillna(0)
    elif mode == "block_shuffle":
        # shuffle within (participant, hour-of-day) blocks
        out["_hod"] = ((out["anchor_ds"] * 5) % 1440) // 60
        for _key, sub in out.groupby(["participant_id", "_hod"]):
            ridx = sub.index.to_numpy()
            perm = rng.permutation(len(ridx))
            out.loc[ridx, cols] = out.loc[ridx, cols].to_numpy()[perm]
        out = out.drop(columns="_hod")
    else:
        raise ValueError(mode)
    return out
