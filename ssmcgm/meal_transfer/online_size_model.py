"""Causal expected response-size model for online meal states.

The retrospective response-size proxy is used only as an offline target. Model
inputs are online-state features and past/current causal covariates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SIZE_CLASSES = ("none", "small", "medium", "large")
SIZE_TO_SCORE = {"none": 0.0, "small": 1.0, "medium": 2.0, "large": 3.0}
SIZE_PROB_COLS = [f"response_size_prob_{c}" for c in SIZE_CLASSES]

BASE_FEATURES = [
    "student_meal_probability",
    "online_meal_probability", "online_phase_code", "online_phase_probability",
    "online_event_active", "online_time_since_onset", "online_elapsed_phase_duration",
    "online_confidence", "online_support_score",
    "online_prob_none", "online_prob_possible_onset", "online_prob_rising",
    "online_prob_likely_peak", "online_prob_recovery",
    "cgm_glucose", "cgm_slope_15", "cgm_slope_30", "cgm_slope_60",
    "cgm_accel", "cgm_roll_mean_30", "cgm_roll_std_30", "cgm_roll_range_60",
    "cgm_pos_excursion_60", "hr_recent_mean", "hr_recent_change",
    "steps_15", "steps_30", "steps_60", "stress_recent", "sleep_rest",
    "tod_sin", "tod_cos", "hba1c_percent_baseline", "bmi_baseline",
    "med_insulin", "med_metformin", "med_any_diabetes_drug", "study_group_code",
]


@dataclass
class ResponseSizeConfig:
    max_iter: int = 160
    learning_rate: float = 0.05
    max_depth: int = 4
    max_leaf_nodes: int = 31
    l2_regularization: float = 5.0
    validation_fraction: float = 0.15
    random_state: int = 42
    max_train_rows: int = 800_000
    max_none_rows: int = 250_000

    @classmethod
    def from_dict(cls, data: dict | None) -> "ResponseSizeConfig":
        if not data:
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class ResponseSizeArtifacts:
    model: object | None
    feature_cols: list[str]
    classes_seen: list[str]
    majority_class: str
    train_rows: int
    val_rows: int
    val_accuracy: float | None
    val_macro_f1: float | None


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in BASE_FEATURES if c in df.columns]


def _target(df: pd.DataFrame) -> pd.Series:
    y = df["meal_response_size_proxy"].fillna("none").astype(str)
    return y.where(y.isin(SIZE_CLASSES), "none")


def _sample_training_rows(labelled: pd.DataFrame, cfg: ResponseSizeConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)
    pieces = []
    for cls, sub in labelled.groupby("_target", sort=False):
        if cls == "none":
            n = min(len(sub), cfg.max_none_rows)
        else:
            n = len(sub)
        if n < len(sub):
            take = rng.choice(sub.index.to_numpy(), size=n, replace=False)
            pieces.append(sub.loc[take])
        else:
            pieces.append(sub)
    out = pd.concat(pieces).sample(frac=1.0, random_state=cfg.random_state)
    if len(out) > cfg.max_train_rows:
        out = out.sample(n=cfg.max_train_rows, random_state=cfg.random_state)
    return out


def train_response_size_model(df: pd.DataFrame, cfg: ResponseSizeConfig | None = None) -> ResponseSizeArtifacts:
    """Train a lightweight causal classifier from offline response-size targets."""
    cfg = cfg or ResponseSizeConfig()
    feature_cols = _feature_cols(df)
    labelled = df.copy()
    labelled["_target"] = _target(labelled)
    if "split" in labelled.columns:
        train_pool = labelled[labelled["split"] == "train"].copy()
        if train_pool.empty:
            train_pool = labelled.copy()
    else:
        train_pool = labelled.copy()
    train_pool = _sample_training_rows(train_pool, cfg)

    y = train_pool["_target"].to_numpy()
    majority = pd.Series(y).value_counts().idxmax() if len(y) else "none"
    classes_present = sorted(pd.unique(y), key=lambda x: SIZE_CLASSES.index(x))
    if len(classes_present) < 2 or not feature_cols:
        return ResponseSizeArtifacts(None, feature_cols, classes_present, majority, len(train_pool), 0, None, None)

    rng = np.random.default_rng(cfg.random_state)
    parts = train_pool["participant_id"].astype(str).unique()
    rng.shuffle(parts)
    n_val = max(1, int(round(len(parts) * cfg.validation_fraction))) if len(parts) > 1 else 0
    val_parts = set(parts[:n_val])
    is_val = train_pool["participant_id"].astype(str).isin(val_parts).to_numpy() if val_parts else np.zeros(len(train_pool), dtype=bool)
    if is_val.all():
        is_val[:] = False

    y_codes = np.array([SIZE_CLASSES.index(v) for v in y], dtype=np.int64)
    X = train_pool[feature_cols].to_numpy(dtype=np.float64)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score

    model = HistGradientBoostingClassifier(
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        max_leaf_nodes=cfg.max_leaf_nodes,
        l2_regularization=cfg.l2_regularization,
        early_stopping=False,
        random_state=cfg.random_state,
    )
    model.fit(X[~is_val], y_codes[~is_val])

    val_accuracy = val_macro_f1 = None
    if is_val.any():
        pred = model.predict(X[is_val])
        val_accuracy = float(accuracy_score(y_codes[is_val], pred))
        val_macro_f1 = float(f1_score(y_codes[is_val], pred, average="macro", zero_division=0))

    return ResponseSizeArtifacts(
        model=model,
        feature_cols=feature_cols,
        classes_seen=[SIZE_CLASSES[int(c)] for c in getattr(model, "classes_", [])],
        majority_class=str(majority),
        train_rows=int((~is_val).sum()),
        val_rows=int(is_val.sum()),
        val_accuracy=val_accuracy,
        val_macro_f1=val_macro_f1,
    )


def predict_response_size(df: pd.DataFrame, art: ResponseSizeArtifacts) -> pd.DataFrame:
    """Predict response-size class probabilities for every row."""
    out = pd.DataFrame(index=df.index)
    probs = np.zeros((len(df), len(SIZE_CLASSES)), dtype=np.float64)
    if art.model is None or not art.feature_cols:
        probs[:, SIZE_CLASSES.index(art.majority_class)] = 1.0
    else:
        raw = art.model.predict_proba(df[art.feature_cols].to_numpy(dtype=np.float64))
        for j, cls_code in enumerate(art.model.classes_):
            probs[:, int(cls_code)] = raw[:, j]
    for j, c in enumerate(SIZE_CLASSES):
        out[f"response_size_prob_{c}"] = probs[:, j]
    pred_idx = probs.argmax(axis=1)
    out["predicted_response_size"] = [SIZE_CLASSES[i] for i in pred_idx]
    out["expected_response_size_score"] = (
        probs[:, SIZE_CLASSES.index("small")] * 1.0
        + probs[:, SIZE_CLASSES.index("medium")] * 2.0
        + probs[:, SIZE_CLASSES.index("large")] * 3.0
    )
    out["response_size_confidence"] = probs.max(axis=1)
    id_cols = [c for c in ["participant_id", "ts", "ds", "split", "segment_id"] if c in df.columns]
    return pd.concat([df[id_cols].copy(), out], axis=1)


def feature_provenance_rows() -> list[dict]:
    rows = []
    for c in BASE_FEATURES:
        if c.startswith("online_"):
            src = "online_decoder forward-only output"
            lookback = "causal recursion / trailing 60 min for support"
        elif c in {"steps_15", "steps_30", "steps_60", "cgm_slope_15", "cgm_slope_30", "cgm_slope_60", "cgm_roll_range_60", "cgm_pos_excursion_60"}:
            src = "causal_student_predictions trailing-window feature"
            lookback = c.rsplit("_", 1)[-1] + " min trailing" if c.rsplit("_", 1)[-1].isdigit() else "60 min trailing"
        else:
            src = "causal_student_predictions current/static feature"
            lookback = "current/static"
        rows.append({
            "feature_name": c,
            "source_columns": src,
            "maximum_lookback": lookback,
            "uses_future_glucose": "no",
            "uses_future_wearables": "no",
            "uses_teacher_output_at_inference": "no",
            "available_online": "yes",
        })
    return rows
