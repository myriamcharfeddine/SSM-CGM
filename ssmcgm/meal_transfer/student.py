"""Phase E — fast causal student (CPU gradient boosting).

A ``HistGradientBoostingClassifier`` trained on **past/current-only** features
(``features.CAUSAL_FEATURE_COLUMNS``) with confidence-weighted pseudo-labels.

Hard leakage guards (the brief is emphatic): the student must NOT see future
glucose, future wearables, bidirectional context, or the retrospective teacher
probability. ``assert_no_leakage`` fails loudly if any forbidden column slips
into the feature matrix. The teacher probability is allowed only as a
pseudo-label/distillation *target*, never as an inference-time feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StudentConfig
from .features import CAUSAL_FEATURE_COLUMNS, LEAKAGE_COLUMNS


def assert_no_leakage(feature_cols: list[str]) -> None:
    bad = sorted(set(feature_cols) & set(LEAKAGE_COLUMNS))
    if bad:
        raise AssertionError(
            f"Leakage: forbidden future/teacher columns in student features: {bad}"
        )
    fut = [c for c in feature_cols if c.startswith(("dg_", "future_"))]
    if fut:
        raise AssertionError(f"Leakage: future columns in student features: {fut}")


@dataclass
class StudentArtifacts:
    model: object
    feature_cols: list[str]
    train_rows: int
    pos_rate: float
    val_ap: float | None
    val_auc: float | None


def _xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].to_numpy(dtype=np.float64)  # HistGBM handles NaN natively
    return X


def train_student(
    train_df: pd.DataFrame,
    cfg: StudentConfig,
    feature_cols: list[str] | None = None,
) -> StudentArtifacts:
    """Train on positive/negative pseudo-labels weighted by confidence.

    Uncertain rows are excluded from training (per the brief). Returns the fitted
    model plus held-out validation metrics (participant-disjoint split).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    feature_cols = feature_cols or [c for c in CAUSAL_FEATURE_COLUMNS if c in train_df.columns]
    assert_no_leakage(feature_cols)

    labelled = train_df[train_df["meal_pseudo_label"].isin(["positive", "negative"])].copy()
    labelled["_y"] = (labelled["meal_pseudo_label"] == "positive").astype(int)
    # Confidence weighting (floor so every labelled row contributes a little).
    w = labelled["pseudo_label_confidence"].fillna(0).clip(lower=0).to_numpy()
    w = 0.1 + 0.9 * w

    # Participant-disjoint validation split for honest early stopping / metrics.
    rng = np.random.default_rng(cfg.random_state)
    parts = labelled["participant_id"].unique()
    rng.shuffle(parts)
    n_val = max(1, int(len(parts) * cfg.validation_fraction))
    val_parts = set(parts[:n_val])
    is_val = labelled["participant_id"].isin(val_parts).to_numpy()

    Xtr = _xy(labelled[~is_val], feature_cols)
    ytr = labelled.loc[~is_val, "_y"].to_numpy()
    wtr = w[~is_val]

    model = HistGradientBoostingClassifier(
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        max_leaf_nodes=cfg.max_leaf_nodes,
        l2_regularization=cfg.l2_regularization,
        early_stopping=False,  # we manage our own participant-disjoint val below
        random_state=cfg.random_state,
        # No class_weight rebalancing: confidence sample weights + the natural
        # labelled prevalence keep the probability calibrated, so the decoder is
        # not driven to over-call meal states.
    )
    model.fit(Xtr, ytr, sample_weight=wtr)

    val_ap = val_auc = None
    if is_val.any() and len(np.unique(labelled.loc[is_val, "_y"])) > 1:
        Xva = _xy(labelled[is_val], feature_cols)
        yva = labelled.loc[is_val, "_y"].to_numpy()
        pva = model.predict_proba(Xva)[:, 1]
        val_ap = float(average_precision_score(yva, pva))
        val_auc = float(roc_auc_score(yva, pva))

    return StudentArtifacts(
        model=model,
        feature_cols=feature_cols,
        train_rows=int((~is_val).sum()),
        pos_rate=float(ytr.mean()),
        val_ap=val_ap,
        val_auc=val_auc,
    )


def predict_student(df: pd.DataFrame, art: StudentArtifacts) -> np.ndarray:
    """Causal meal probability for every row (full coverage, including rows that
    were uncertain/unlabelled at training time)."""
    assert_no_leakage(art.feature_cols)
    X = _xy(df, art.feature_cols)
    if getattr(art, "is_regressor", False):
        return np.clip(art.model.predict(X), 0.0, 1.0)
    return art.model.predict_proba(X)[:, 1]


def train_student_soft(
    train_df: pd.DataFrame,
    cfg: StudentConfig,
    feature_cols: list[str] | None = None,
) -> StudentArtifacts:
    """Block 4 causal student trained on the soft onset-distance target.

    Unlike :func:`train_student`, this keeps the rescued uncertain rows: it fits a
    gradient-boosted *regressor* on ``meal_soft_target`` in [0, 1] over all
    label-eligible rows (``label_train_eligible == 1``), which excludes the
    quarantined insulin users. Validation AP / AUC are still reported against the
    hard positive/negative subset so the number is comparable to the old student.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import average_precision_score, roc_auc_score

    feature_cols = feature_cols or [c for c in CAUSAL_FEATURE_COLUMNS if c in train_df.columns]
    assert_no_leakage(feature_cols)

    elig = train_df[train_df.get("label_train_eligible", 1) == 1].copy()
    y = elig["meal_soft_target"].fillna(0.0).to_numpy(dtype=np.float64)

    rng = np.random.default_rng(cfg.random_state)
    parts = elig["participant_id"].unique()
    rng.shuffle(parts)
    n_val = max(1, int(len(parts) * cfg.validation_fraction))
    val_parts = set(parts[:n_val])
    is_val = elig["participant_id"].isin(val_parts).to_numpy()

    model = HistGradientBoostingRegressor(
        max_iter=cfg.max_iter, learning_rate=cfg.learning_rate, max_depth=cfg.max_depth,
        max_leaf_nodes=cfg.max_leaf_nodes, l2_regularization=cfg.l2_regularization,
        random_state=cfg.random_state)
    model.fit(_xy(elig[~is_val], feature_cols), y[~is_val])

    val_ap = val_auc = None
    va = elig[is_val]
    hard = va["meal_pseudo_label"].isin(["positive", "negative"]) if "meal_pseudo_label" in va else None
    if hard is not None and hard.any():
        vh = va[hard]
        yb = (vh["meal_pseudo_label"] == "positive").astype(int).to_numpy()
        if len(np.unique(yb)) > 1:
            pv = np.clip(model.predict(_xy(vh, feature_cols)), 0.0, 1.0)
            val_ap = float(average_precision_score(yb, pv))
            val_auc = float(roc_auc_score(yb, pv))

    art = StudentArtifacts(
        model=model, feature_cols=feature_cols, train_rows=int((~is_val).sum()),
        pos_rate=float((y[~is_val] >= 0.5).mean()), val_ap=val_ap, val_auc=val_auc)
    art.is_regressor = True
    return art
