"""Evaluation utilities for the strictly online meal-state track."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .online_decoder import STATE_NAMES, STATE_PROB_COLS
from .online_size_model import SIZE_CLASSES, SIZE_PROB_COLS

HYPER_MGDL = 180.0
Z90_MINUS_Z10 = 2.5631031
EXACT_TOL = 0.5

D_FEATURES = ["student_meal_probability"]
G_FEATURES = [
    "student_meal_probability",
    "online_meal_probability", "online_phase_code", "online_phase_probability",
    "online_event_active", "online_time_since_onset", "online_elapsed_phase_duration",
    "online_confidence", "online_support_score", *STATE_PROB_COLS,
]
H_FEATURES = [
    *G_FEATURES,
    *SIZE_PROB_COLS, "expected_response_size_score", "response_size_confidence",
]
CONTEXT_FEATURES = [
    "q50", "spread", "horizon_step", "hba1c_percent_baseline", "bmi_baseline",
    "med_insulin", "med_any_diabetes_drug", "study_group_code", "site_code",
]
SETUP_FEATURES = {
    "D_student_prob": D_FEATURES,
    "G_online_state": G_FEATURES,
    "H_online_full_state": H_FEATURES,
}


@dataclass
class EvalMaskConfig:
    teacher_quantile: float = 0.75
    response_quantile: float = 0.75
    min_time_to_peak_min: int = 10
    max_time_to_peak_min: int = 120
    activity_steps_30_max: float = 1500.0
    event_window_steps: int = 12
    min_event_gap_steps: int = 12
    insulin_confidence_scale: float = 0.6

    @classmethod
    def from_dict(cls, data: dict | None) -> "EvalMaskConfig":
        if not data:
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class AblationConfig:
    max_iter: int = 180
    learning_rate: float = 0.04
    max_depth: int = 4
    max_leaf_nodes: int = 31
    l2_regularization: float = 10.0
    random_state: int = 42
    n_bootstrap: int = 500
    time_shift_steps: int = 24

    @classmethod
    def from_dict(cls, data: dict | None) -> "AblationConfig":
        if not data:
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _phi(z):
    from math import erf, sqrt
    return 0.5 * (1.0 + np.vectorize(lambda x: erf(x / sqrt(2.0)))(z))


def build_independent_eval_mask(student: pd.DataFrame, teacher: pd.DataFrame,
                                pseudo: pd.DataFrame, retrospective: pd.DataFrame | None,
                                cfg: EvalMaskConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Build a fixed high-confidence retrospective mask independent of online outputs."""
    cfg = cfg or EvalMaskConfig()
    cols = ["participant_id", "ts", "ds", "split", "segment_id", "cgm_glucose", "med_insulin",
            "steps_30", "sleep_rest"]
    df = student[[c for c in cols if c in student.columns]].copy()
    df["participant_id"] = df["participant_id"].astype(str)
    key = ["participant_id", "ds"]

    tcols = key + ["cgmacros_teacher_probability"]
    df = df.merge(teacher[tcols], on=key, how="left")
    pcols = key + ["future_peak_rise_120", "time_to_peak_min", "dg_60", "meal_response_size_proxy"]
    df = df.merge(pseudo[pcols], on=key, how="left", suffixes=("", "_pseudo"))
    if retrospective is not None and "predmeal_flag_clean" in retrospective.columns:
        df = df.merge(retrospective[key + ["predmeal_flag_clean"]], on=key, how="left")
    else:
        df["predmeal_flag_clean"] = 0.0

    train_primary = (df["split"].eq("train") & df["med_insulin"].fillna(0).eq(0)
                     & df["cgmacros_teacher_probability"].notna()
                     & df["future_peak_rise_120"].notna())
    ref = df.loc[train_primary]
    teacher_thr = float(ref["cgmacros_teacher_probability"].quantile(cfg.teacher_quantile)) if len(ref) else 0.5
    resp_thr = float(ref["future_peak_rise_120"].quantile(cfg.response_quantile)) if len(ref) else 25.0
    pos_rise = ref.loc[ref["future_peak_rise_120"] > 0, "future_peak_rise_120"]
    if len(pos_rise) >= 10:
        size_small = float(pos_rise.quantile(1 / 3))
        size_medium = float(pos_rise.quantile(2 / 3))
    else:
        size_small, size_medium = 12.0, 30.0

    teacher_norm = ((df["cgmacros_teacher_probability"] - teacher_thr) / max(1.0 - teacher_thr, 1e-6)).clip(0, 1)
    resp_norm = ((df["future_peak_rise_120"] - resp_thr) / max(resp_thr, 1e-6)).clip(0, 1)
    agreement = df["predmeal_flag_clean"].fillna(0).clip(0, 1)
    score = (0.45 * teacher_norm.fillna(0) + 0.45 * resp_norm.fillna(0) + 0.10 * agreement).clip(0, 1)

    plausible_ttp = df["time_to_peak_min"].between(cfg.min_time_to_peak_min, cfg.max_time_to_peak_min)
    coverage = df["future_peak_rise_120"].notna() & df["dg_60"].notna()
    activity_ok = df.get("steps_30", pd.Series(0, index=df.index)).fillna(0) <= cfg.activity_steps_30_max
    sleep_ok = df.get("sleep_rest", pd.Series(0, index=df.index)).fillna(0) <= 0
    candidate = (df["cgmacros_teacher_probability"].ge(teacher_thr)
                 & df["future_peak_rise_120"].ge(resp_thr)
                 & plausible_ttp & coverage & activity_ok & sleep_ok)

    df["eval_meal_event_id"] = -1
    df["eval_in_meal_window"] = 0.0
    df["eval_meal_confidence"] = 0.0
    df["eval_response_size"] = "none"

    event_id = 0
    for (_pid, _seg), sub in df.groupby(["participant_id", "segment_id"], sort=False):
        cand_pos = np.flatnonzero(candidate.loc[sub.index].to_numpy())
        if len(cand_pos) == 0:
            continue
        last_start = -10**9
        for local in cand_pos:
            if local - last_start < cfg.min_event_gap_steps:
                continue
            global_idx = sub.index[local]
            rise = float(df.at[global_idx, "future_peak_rise_120"])
            if rise < size_small:
                size = "small"
            elif rise < size_medium:
                size = "medium"
            else:
                size = "large"
            conf = float(score.loc[global_idx])
            if float(df.at[global_idx, "med_insulin"] or 0) == 1.0:
                conf *= cfg.insulin_confidence_scale
            end_local = min(local + cfg.event_window_steps, len(sub) - 1)
            rows = sub.index[local:end_local + 1]
            empty = df.loc[rows, "eval_meal_event_id"].eq(-1)
            rows = rows[empty.to_numpy()]
            if len(rows) == 0:
                continue
            df.loc[rows, "eval_meal_event_id"] = event_id
            df.loc[rows, "eval_in_meal_window"] = 1.0
            df.loc[rows, "eval_meal_confidence"] = conf
            df.loc[rows, "eval_response_size"] = size
            event_id += 1
            last_start = local

    out_cols = ["participant_id", "ts", "ds", "split", "segment_id", "med_insulin",
                "eval_meal_event_id", "eval_in_meal_window", "eval_meal_confidence", "eval_response_size"]
    summary = {
        "teacher_threshold": teacher_thr,
        "response_threshold": resp_thr,
        "size_small_threshold": size_small,
        "size_medium_threshold": size_medium,
        "n_eval_events": int(event_id),
        "eval_window_rate": float(df["eval_in_meal_window"].mean()),
        "primary_non_insulin_window_rate": float(df.loc[df["med_insulin"].fillna(0).eq(0), "eval_in_meal_window"].mean()),
        "insulin_window_rate": float(df.loc[df["med_insulin"].fillna(0).eq(1), "eval_in_meal_window"].mean()) if (df["med_insulin"].fillna(0).eq(1)).any() else 0.0,
    }
    return df[out_cols], summary


def _map_by_key(source: pd.DataFrame, keys: list[tuple], col: str) -> np.ndarray:
    s = source.set_index(["participant_id", "ds"])[col]
    return s.reindex(keys).to_numpy()


def attach_online_frame(aligned: pd.DataFrame, online_states: pd.DataFrame,
                        size_pred: pd.DataFrame, eval_mask: pd.DataFrame) -> pd.DataFrame:
    """Attach anchor online features and target evaluation mask to forecast rows."""
    df = aligned.copy()
    df["participant_id"] = df["participant_id"].astype(str)
    df["anchor_ds"] = df["anchor_ds"].astype(int)
    df["target_ds"] = df["anchor_ds"] + df["horizon_step"].astype(int)
    key_anchor = list(zip(df["participant_id"], df["anchor_ds"]))
    key_target = list(zip(df["participant_id"], df["target_ds"]))

    states = online_states.copy()
    states["participant_id"] = states["participant_id"].astype(str)
    sizes = size_pred.copy()
    sizes["participant_id"] = sizes["participant_id"].astype(str)
    mask = eval_mask.copy()
    mask["participant_id"] = mask["participant_id"].astype(str)

    # Alignment gate: target CGM and segment must match the online source table.
    df["_target_cgm"] = _map_by_key(states, key_target, "cgm_glucose")
    df = df[np.isfinite(df["_target_cgm"]) & (np.abs(df["target"] - df["_target_cgm"]) <= EXACT_TOL)].copy()
    key_anchor = list(zip(df["participant_id"], df["anchor_ds"]))
    key_target = list(zip(df["participant_id"], df["target_ds"]))
    df["_anchor_segment"] = _map_by_key(states, key_anchor, "segment_id")
    df["_target_segment"] = _map_by_key(states, key_target, "segment_id")
    same_seg = pd.notna(df["_anchor_segment"]) & pd.notna(df["_target_segment"]) & (df["_anchor_segment"].to_numpy() == df["_target_segment"].to_numpy())
    df = df[same_seg].copy()
    key_anchor = list(zip(df["participant_id"], df["anchor_ds"]))
    key_target = list(zip(df["participant_id"], df["target_ds"]))

    anchor_cols = ["student_meal_probability", *G_FEATURES[1:]]
    # student_meal_probability lives in online_states only if copied there; fall back handled by map.
    for c in dict.fromkeys(anchor_cols):
        if c in states.columns:
            df[c] = _map_by_key(states, key_anchor, c)
    for c in [*SIZE_PROB_COLS, "predicted_response_size", "expected_response_size_score", "response_size_confidence"]:
        if c in sizes.columns:
            df[c] = _map_by_key(sizes, key_anchor, c)
    if "student_meal_probability" not in df.columns or df["student_meal_probability"].isna().all():
        raise RuntimeError("student_meal_probability missing from online state table")

    for c in ["eval_meal_event_id", "eval_in_meal_window", "eval_meal_confidence", "eval_response_size"]:
        df[c] = _map_by_key(mask, key_target, c)
    df["eval_in_meal_window"] = pd.to_numeric(df["eval_in_meal_window"], errors="coerce").fillna(0).astype(bool)
    df["spread"] = (df["q90"] - df["q10"]).clip(lower=1e-3)
    df["study_group_code"] = df["participants_study_group"].astype("category").cat.codes.astype(float)
    df["site_code"] = df["participants_clinical_site"].astype("category").cat.codes.astype(float)
    df["hba1c_quartile"] = _quartile(df["hba1c_percent_baseline"])
    df["bmi_quartile"] = _quartile(df["bmi_baseline"])
    df["residual"] = df["target"] - df["q50"]
    return df


def _quartile(s: pd.Series) -> pd.Series:
    try:
        return pd.qcut(s, 4, labels=[1, 2, 3, 4], duplicates="drop").astype(float)
    except Exception:
        return pd.Series(np.nan, index=s.index)


def _design(df: pd.DataFrame, setup: str) -> np.ndarray:
    cols = CONTEXT_FEATURES + SETUP_FEATURES[setup]
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category").cat.codes.astype(float)
    return X.to_numpy(dtype=np.float64)


def fit_residual_model(train: pd.DataFrame, setup: str, cfg: AblationConfig | None = None):
    cfg = cfg or AblationConfig()
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        max_leaf_nodes=cfg.max_leaf_nodes,
        l2_regularization=cfg.l2_regularization,
        random_state=cfg.random_state,
    )
    model.fit(_design(train, setup), train["residual"].to_numpy(dtype=np.float64))
    return model


def apply_residual_model(df: pd.DataFrame, model, setup: str) -> np.ndarray:
    return df["q50"].to_numpy(dtype=np.float64) + model.predict(_design(df, setup))


def hyper_scores(df: pd.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = df["horizon_step"].to_numpy()
    m = h == 12
    sigma = df["spread"].to_numpy(dtype=np.float64)[m] / Z90_MINUS_Z10
    p_hyper = 1.0 - _phi((HYPER_MGDL - pred[m]) / np.maximum(sigma, 1e-6))
    label = (df["target"].to_numpy(dtype=np.float64)[m] >= HYPER_MGDL).astype(int)
    return label, p_hyper


def _hyper_metrics(label: np.ndarray, score: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    out = {"hyper_prevalence": float(label.mean()) if len(label) else np.nan}
    if len(label) == 0 or label.min() == label.max():
        out.update(hyper_AUROC=np.nan, hyper_AUPRC=np.nan, hyper_brier=np.nan, hyper_ECE=np.nan)
        return out
    out["hyper_AUROC"] = float(roc_auc_score(label, score))
    out["hyper_AUPRC"] = float(average_precision_score(label, score))
    out["hyper_brier"] = float(brier_score_loss(label, score))
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(score, bins) - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(score[sel].mean() - label[sel].mean())
    out["hyper_ECE"] = float(ece)
    return out


def forecast_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    target = df["target"].to_numpy(dtype=np.float64)
    err = np.abs(pred - target)
    h = df["horizon_step"].to_numpy()
    out = {
        "MAE_30min": float(err[h == 6].mean()),
        "MAE_60min": float(err[h == 12].mean()),
        "MAE_all": float(err.mean()),
    }
    mw = df["eval_in_meal_window"].to_numpy(dtype=bool)
    out["MAE_eval_meal_window"] = float(err[mw].mean()) if mw.any() else np.nan
    out["eval_meal_window_rows"] = int(mw.sum())
    tmp = df[["participant_id", "anchor_ds", "horizon_step", "target"]].copy()
    tmp["pred"] = pred
    g = tmp.groupby(["participant_id", "anchor_ds"])
    peak_target = g["target"].max()
    peak_pred = g["pred"].max()
    out["peak_error_1h"] = float((peak_target - peak_pred).abs().mean())
    idx_t = tmp.loc[tmp.groupby(["participant_id", "anchor_ds"])["target"].idxmax()][["participant_id", "anchor_ds", "horizon_step"]]
    idx_p = tmp.loc[tmp.groupby(["participant_id", "anchor_ds"])["pred"].idxmax()][["participant_id", "anchor_ds", "horizon_step"]]
    ttp = idx_t.merge(idx_p, on=["participant_id", "anchor_ds"], suffixes=("_target", "_pred"))
    out["time_to_peak_error_min"] = float((ttp["horizon_step_target"] - ttp["horizon_step_pred"]).abs().mean() * 5.0)
    label, score = hyper_scores(df, pred)
    out.update(_hyper_metrics(label, score))
    return out


def metric_by_participant(df: pd.DataFrame, pred: np.ndarray, metric: str) -> pd.Series:
    pid = df["participant_id"].astype(str)
    target = df["target"].to_numpy(dtype=np.float64)
    err = np.abs(pred - target)
    h = df["horizon_step"].to_numpy()
    if metric == "MAE_30min":
        return pd.Series(err[h == 6], index=pid[h == 6]).groupby(level=0).mean()
    if metric == "MAE_60min":
        return pd.Series(err[h == 12], index=pid[h == 12]).groupby(level=0).mean()
    if metric == "MAE_eval_meal_window":
        m = df["eval_in_meal_window"].to_numpy(dtype=bool)
        return pd.Series(err[m], index=pid[m]).groupby(level=0).mean()
    if metric in {"peak_error_1h", "time_to_peak_error_min"}:
        tmp = df[["participant_id", "anchor_ds", "horizon_step", "target"]].copy()
        tmp["pred"] = pred
        g = tmp.groupby(["participant_id", "anchor_ds"])
        if metric == "peak_error_1h":
            per_anchor = (g["target"].max() - g["pred"].max()).abs().reset_index(name=metric)
        else:
            idx_t = tmp.loc[g["target"].idxmax()][["participant_id", "anchor_ds", "horizon_step"]]
            idx_p = tmp.loc[g["pred"].idxmax()][["participant_id", "anchor_ds", "horizon_step"]]
            per_anchor = idx_t.merge(idx_p, on=["participant_id", "anchor_ds"], suffixes=("_target", "_pred"))
            per_anchor[metric] = (per_anchor["horizon_step_target"] - per_anchor["horizon_step_pred"]).abs() * 5.0
        return per_anchor.groupby("participant_id")[metric].mean()
    if metric in {"hyper_brier", "hyper_AUROC", "hyper_AUPRC", "hyper_ECE"}:
        from sklearn.metrics import average_precision_score, roc_auc_score
        m = h == 12
        label, score = hyper_scores(df, pred)
        pidh = pid[m].to_numpy()
        rows = []
        for p in pd.unique(pidh):
            sel = pidh == p
            y = label[sel]
            s = score[sel]
            if metric == "hyper_brier":
                val = float(((s - y) ** 2).mean())
            elif len(y) == 0 or y.min() == y.max():
                val = np.nan
            elif metric == "hyper_AUROC":
                val = float(roc_auc_score(y, s))
            elif metric == "hyper_AUPRC":
                val = float(average_precision_score(y, s))
            else:
                bins = np.linspace(0, 1, 11)
                idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
                val = 0.0
                for b in range(10):
                    ss = idx == b
                    if ss.any():
                        val += ss.mean() * abs(s[ss].mean() - y[ss].mean())
            rows.append((p, val))
        ser = pd.Series({p: v for p, v in rows}, dtype=float)
        return ser.dropna()
    raise ValueError(metric)


def bootstrap_comparison(df: pd.DataFrame, preds: dict[str, np.ndarray], base: str, model: str,
                         metrics: list[str], n_boot: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base_overall_all = forecast_metrics(df, preds[base])
    model_overall_all = forecast_metrics(df, preds[model])
    for metric in metrics:
        a = metric_by_participant(df, preds[base], metric)
        b = metric_by_participant(df, preds[model], metric)
        both = sorted(set(a.index) & set(b.index))
        diff = (a.loc[both] - b.loc[both]).to_numpy(dtype=np.float64)
        if len(diff) == 0:
            lo = hi = mean = np.nan
            excludes = False
        else:
            boots = np.empty(n_boot, dtype=np.float64)
            n = len(diff)
            for i in range(n_boot):
                boots[i] = diff[rng.integers(0, n, size=n)].mean()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            mean = float(diff.mean())
            excludes = bool(lo > 0 or hi < 0)
        base_overall = base_overall_all.get(metric, np.nan)
        model_overall = model_overall_all.get(metric, np.nan)
        rows.append({
            "comparison": f"{base} - {model}",
            "positive_diff_means": f"{model} improves over {base}",
            "metric": metric,
            "n_participants": int(len(diff)),
            "base_overall": base_overall,
            "model_overall": model_overall,
            "overall_diff_positive_improves": base_overall - model_overall,
            "participant_mean_diff_positive_improves": mean,
            "ci95_lo": float(lo) if np.isfinite(lo) else np.nan,
            "ci95_hi": float(hi) if np.isfinite(hi) else np.nan,
            "ci_excludes_zero": excludes,
        })
    return pd.DataFrame(rows)


def corrupt_features(df: pd.DataFrame, setup: str, mode: str, cfg: AblationConfig | None = None) -> pd.DataFrame:
    cfg = cfg or AblationConfig()
    rng = np.random.default_rng(cfg.random_state)
    out = df.copy()
    cols = SETUP_FEATURES[setup]
    if mode == "shuffle":
        perm = rng.permutation(len(out))
        out[cols] = out[cols].to_numpy()[perm]
    elif mode == "time_shift":
        for _pid, sub in out.groupby("participant_id"):
            ridx = sub.sort_values("anchor_ds").index
            out.loc[ridx, cols] = out.loc[ridx, cols].shift(cfg.time_shift_steps).bfill().fillna(0).to_numpy()
    elif mode == "block_shuffle":
        out["_hod"] = ((out["anchor_ds"].astype(int) * 5) % 1440) // 60
        for _key, sub in out.groupby(["participant_id", "_hod"]):
            ridx = sub.index.to_numpy()
            out.loc[ridx, cols] = out.loc[ridx, cols].to_numpy()[rng.permutation(len(ridx))]
        out = out.drop(columns="_hod")
    else:
        raise ValueError(mode)
    return out


def subgroup_metrics(df: pd.DataFrame, preds: dict[str, np.ndarray], setups: list[str]) -> pd.DataFrame:
    groups = ["med_insulin", "participants_study_group", "hba1c_quartile", "bmi_quartile", "participants_clinical_site"]
    rows = []
    for setup in setups:
        pred = preds[setup]
        for col in groups:
            for val, idx in df.groupby(col).groups.items():
                loc = df.index.get_indexer(idx)
                met = forecast_metrics(df.loc[idx], pred[loc])
                rows.append({
                    "setup": setup,
                    "subgroup": col,
                    "value": str(val),
                    "n_rows": int(len(idx)),
                    "n_participants": int(df.loc[idx, "participant_id"].nunique()),
                    "MAE_60min": met["MAE_60min"],
                    "MAE_eval_meal_window": met["MAE_eval_meal_window"],
                    "peak_error_1h": met["peak_error_1h"],
                    "hyper_AUROC": met["hyper_AUROC"],
                    "hyper_brier": met["hyper_brier"],
                })
    return pd.DataFrame(rows)
