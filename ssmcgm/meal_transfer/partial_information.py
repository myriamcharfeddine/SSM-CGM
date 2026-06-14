"""Block 2 partial-information test against the CGM slope.

The weak meal labels are glucose-defined ("glucose went up"), so a meal feature
can be largely a re-encoding of the recent CGM slope that the q50 forecaster
already sees. This module measures how much *orthogonal* information a meal
feature carries, conditional on slope, level, and clock.

For each meal feature m:
  1. regress m on [dy15, dy30, dy60, level, hour_sin, hour_cos] on the train
     participants, and form the residual ``m_perp = m - m_hat`` on held-out test
     participants (the part of the feature not explained by CGM trend / clock);
  2. report the partial correlation between ``m_perp`` and the q50 forecast
     residual (target - q50) at the 60-minute horizon, on held-out participants;
  3. report the change in the gate endpoints (meal-window MAE, 1 h peak error)
     from adding ``m_perp`` to a slope-only residual model.

A feature with no partial signal cannot help a forecaster that already uses the
CGM slope, no matter how good its classification metrics are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import endpoints as EP

SLOPE_LEVEL_CLOCK = ["dy15", "dy30", "dy60", "level", "hour_sin", "hour_cos"]


def _first_meal_col(df: pd.DataFrame, meal_cols: list[str]) -> str | None:
    """The representative scalar feature for a setup (its first numeric column)."""
    for c in meal_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _fit_linear(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LinearRegression
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    m = LinearRegression()
    m.fit(X[ok], y[ok])
    return m


def _partial_corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.nanstd(a[ok]) < 1e-9 or np.nanstd(b[ok]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def run_partial_information(train: pd.DataFrame, test: pd.DataFrame, base_slc: list[str],
                            feature_setups: dict[str, list[str]], non_insulin_mask: np.ndarray,
                            cfg) -> pd.DataFrame:
    """Run the orthogonalization test for every feature setup. ``base_slc`` is the
    A' slope/level/clock design; the test restricts to non-insulin rows."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    slc = [c for c in SLOPE_LEVEL_CLOCK if c in train.columns]
    tr = train[train["med_insulin"].fillna(0) == 0].copy()
    te = test[non_insulin_mask].copy()

    # slope/level/clock have NaN at segment starts; impute with TRAIN column means
    # so the linear orthogonalization regression accepts them (HistGBM is fine).
    slc_means = tr[slc].mean(numeric_only=True)
    Xtr_slc = tr[slc].fillna(slc_means).to_numpy(dtype=np.float64)
    Xte_slc = te[slc].fillna(slc_means).to_numpy(dtype=np.float64)
    resid_te = (te["target"] - te["q50"]).to_numpy(dtype=np.float64)
    h12_te = te["horizon_step"].to_numpy() == 12

    # slope-only residual model (within-cohort A') for the delta-endpoint test
    base_model = HistGradientBoostingRegressor(
        max_iter=cfg.max_iter, learning_rate=cfg.learning_rate, max_depth=cfg.max_depth,
        max_leaf_nodes=cfg.max_leaf_nodes, l2_regularization=cfg.l2_regularization,
        random_state=cfg.random_state)
    base_model.fit(tr[base_slc].to_numpy(dtype=np.float64),
                   (tr["target"] - tr["q50"]).to_numpy(dtype=np.float64))
    base_pred = te["q50"].to_numpy(dtype=np.float64) + base_model.predict(te[base_slc].to_numpy(dtype=np.float64))
    base_metrics = EP.forecast_metrics(te, base_pred)

    rows = []
    for setup, meal_cols in feature_setups.items():
        col = _first_meal_col(te, meal_cols)
        if col is None:
            continue
        # orthogonalize the feature against slope/level/clock (fit on train)
        m_on_slc = _fit_linear(Xtr_slc, tr[col].to_numpy(dtype=np.float64))
        m_hat = m_on_slc.predict(Xte_slc)
        m_perp = te[col].to_numpy(dtype=np.float64) - m_hat

        # partial correlation of the orthogonal feature with the q50 residual @60
        pc = _partial_corr(m_perp[h12_te], resid_te[h12_te])

        # add m_perp to the slope-only model and measure the endpoint deltas
        te2 = te.copy()
        te2["_m_perp"] = m_perp
        tr2 = tr.copy()
        tr2["_m_perp"] = tr[col].to_numpy(dtype=np.float64) - m_on_slc.predict(Xtr_slc)
        aug = HistGradientBoostingRegressor(
            max_iter=cfg.max_iter, learning_rate=cfg.learning_rate, max_depth=cfg.max_depth,
            max_leaf_nodes=cfg.max_leaf_nodes, l2_regularization=cfg.l2_regularization,
            random_state=cfg.random_state)
        aug_cols = base_slc + ["_m_perp"]
        aug.fit(tr2[aug_cols].to_numpy(dtype=np.float64),
                (tr2["target"] - tr2["q50"]).to_numpy(dtype=np.float64))
        aug_pred = te2["q50"].to_numpy(dtype=np.float64) + aug.predict(te2[aug_cols].to_numpy(dtype=np.float64))
        aug_metrics = EP.forecast_metrics(te2, aug_pred)

        rows.append({
            "setup": setup,
            "feature": col,
            "partial_corr_with_q50_residual": pc,
            "raw_corr_with_q50_residual": _partial_corr(
                te[col].to_numpy(dtype=np.float64)[h12_te], resid_te[h12_te]),
            "frac_variance_explained_by_slope": float(np.clip(
                m_on_slc.score(Xtr_slc[np.isfinite(tr[col])],
                               tr[col].to_numpy(dtype=np.float64)[np.isfinite(tr[col])]), -1, 1)),
            # delta endpoint = base (slope-only) minus augmented; positive improves
            "delta_meal_window_mae": base_metrics["MAE_eval_meal_window"] - aug_metrics["MAE_eval_meal_window"],
            "delta_peak_error": base_metrics["peak_error_1h"] - aug_metrics["peak_error_1h"],
            "delta_mae60": base_metrics["MAE_60min"] - aug_metrics["MAE_60min"],
        })
    return pd.DataFrame(rows)
