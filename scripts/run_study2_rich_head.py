#!/usr/bin/env python3
"""Study 2 – Rich causal feature set and extended baseline ladder.

Tasks:
  1. Print anchor-count funnel (explains 39 698 vs 2 548 discrepancy).
  2. Compute causal trailing features in the anchor table.
  3. Train separate up/down Ridge and HistGradientBoosting heads.
  4. Evaluate 8 baselines with participant-clustered 95 % CIs.
  5. Report paired CIs for key comparisons.
  6. Save artifacts to outputs/study2_rich_head/.

Run with:
    /home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python \\
        scripts/run_study2_rich_head.py
"""

from __future__ import annotations

import json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
CACHE   = ROOT / "outputs/study2_forecast_cache_5min/study2_forecast_cache.parquet"
TAU_LOC = ROOT / "outputs/study2_forecast_cache_5min/study2_selected_thresholds.json"
OUT     = ROOT / "outputs/study2_rich_head"

TAU_UP   = 0.2
TAU_DOWN = 0.2
EPS      = 1e-3
N_BOOT   = 1000
SEED     = 42
H        = 12
BIN_MIN  = 5

EVENT_GLUCOSE_HI = 130.0
EVENT_GLUCOSE_LO =  85.0


def _sec(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


# ── fixed n_anchors helper ────────────────────────────────────────────────────
def n_unique_anchors(df: pd.DataFrame) -> int:
    """Count distinct (participant_id, segment_id, anchor_ds) tuples.

    anchor_ds is per-stream relative, NOT globally unique.
    df["anchor_ds"].nunique() is wrong and will under-count.
    """
    return int(df.drop_duplicates(["participant_id", "segment_id", "anchor_ds"]).shape[0])


# ── anchor table with causal trailing features ────────────────────────────────

def build_anchor_table_rich(df: pd.DataFrame) -> pd.DataFrame:
    """Anchor table with base fields + causal trailing features.

    Added vs. original build_anchor_table:
      hours_since_start, steps_since_start  (from h=1 cache rows)
      r_t_1 r_t_2 r_t_3     (lagged one-step residuals)
      z_t_1 z_t_2            (lagged z-scores)
      glucose_lag3/6/12      (G at t-3, t-6, t-12)
      slope_15 slope_30 slope_60  (mg/dL/min)
      curvature              (slope_15 - slope_30) / 15
      hour_sin hour_cos      (time-of-day encoding via hours_since_start % 24)
      steps_15 steps_30      (step-count change over 15/30 min)
      NOTE: HR not available in cache; excluded.
    """
    fo = df.copy()
    if "anchor_ds" not in fo.columns:
        fo["anchor_ds"] = fo["anchor_time_idx"]

    # --- base meta columns at each anchor ------------------------------------
    meta_cols = [c for c in [
        "participant_id", "segment_id", "split", "anchor_ds",
        "current_glucose", "med_insulin", "participants_study_group",
        "hba1c_percent_baseline", "participants_clinical_site",
        "med_any_diabetes_drug",
        "hours_since_start", "steps_since_start",
    ] if c in fo.columns]

    anchors = (
        fo.sort_values(["participant_id", "segment_id", "anchor_ds"])
        .drop_duplicates(["participant_id", "segment_id", "anchor_ds"])[meta_cols]
        .copy()
    )

    # --- h=1 forecast stats (one_step residual, interval width) --------------
    h1 = fo[fo["horizon_step"].eq(1)][
        ["participant_id", "segment_id", "anchor_ds", "q10", "q50", "q90", "target"]
    ].copy().rename(columns={
        "anchor_ds": "one_step_issue_anchor_ds",
        "q10": "q10_one_step", "q50": "q50_one_step",
        "q90": "q90_one_step", "target": "one_step_target",
    })
    h1["anchor_ds"] = h1["one_step_issue_anchor_ds"] + 1
    h1 = h1.drop_duplicates(["participant_id", "segment_id", "anchor_ds"])

    out = anchors.merge(
        h1[["participant_id", "segment_id", "anchor_ds",
            "one_step_issue_anchor_ds", "q10_one_step",
            "q50_one_step", "q90_one_step", "one_step_target"]],
        on=["participant_id", "segment_id", "anchor_ds"], how="left",
    )
    out["interval_width"] = (
        pd.to_numeric(out["q90_one_step"], errors="coerce")
        - pd.to_numeric(out["q10_one_step"], errors="coerce")
    )
    out["current_glucose"] = pd.to_numeric(out["current_glucose"], errors="coerce").fillna(
        pd.to_numeric(out["one_step_target"], errors="coerce")
    )
    out["one_step_residual"] = (
        out["current_glucose"] - pd.to_numeric(out["q50_one_step"], errors="coerce")
    )

    # --- abstain flags --------------------------------------------------------
    consecutive    = out["one_step_issue_anchor_ds"].eq(out["anchor_ds"] - 1)
    valid_interval = np.isfinite(out["interval_width"]) & out["interval_width"].gt(0)
    valid_residual = np.isfinite(out["one_step_residual"])
    is_first = (
        out.groupby(["participant_id", "segment_id"], sort=False)["anchor_ds"]
        .transform("min").eq(out["anchor_ds"])
    )
    reasons = pd.Series("none", index=out.index)
    reasons = reasons.where(valid_residual, "missing_residual")
    reasons = reasons.where(valid_interval, "missing_interval")
    reasons = reasons.where(consecutive, "non_consecutive")
    reasons = reasons.where(~is_first, "first_anchor")
    out["abstain"]        = reasons.ne("none")
    out["abstain_reason"] = reasons
    med_ins = pd.to_numeric(
        out.get("med_insulin", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0)
    out["non_insulin"] = med_ins.eq(0)

    # --- causal trailing features -------------------------------------------
    out = out.sort_values(["participant_id", "segment_id", "anchor_ds"])
    grp = out.groupby(["participant_id", "segment_id"], sort=False)

    # Lagged residuals
    out["r_t"]   = out["one_step_residual"]
    out["r_t_1"] = grp["one_step_residual"].shift(1)
    out["r_t_2"] = grp["one_step_residual"].shift(2)
    out["r_t_3"] = grp["one_step_residual"].shift(3)

    # z_t will be computed after trigger code, but z_t_{t-1}, z_t_{t-2} also lagged
    # We need z_t first – compute a provisional version here
    iw_safe   = out["interval_width"].where(out["interval_width"].gt(0), np.nan)
    out["z_t_raw"] = np.where(out["abstain"], np.nan, out["one_step_residual"] / (iw_safe + EPS))
    out["z_t_1"]   = grp["z_t_raw"].shift(1)
    out["z_t_2"]   = grp["z_t_raw"].shift(2)

    # Glucose lags for slopes (5-min bins: 3=15min, 6=30min, 12=60min)
    gluc = out["current_glucose"]
    out["glucose_lag3"]  = grp["current_glucose"].shift(3)
    out["glucose_lag6"]  = grp["current_glucose"].shift(6)
    out["glucose_lag12"] = grp["current_glucose"].shift(12)

    # Slopes in mg/dL per minute
    out["slope_15"] = (gluc - out["glucose_lag3"])  / 15.0
    out["slope_30"] = (gluc - out["glucose_lag6"])  / 30.0
    out["slope_60"] = (gluc - out["glucose_lag12"]) / 60.0

    # Causal curvature: acceleration of 15-min slope vs 30-min slope
    out["curvature"] = (out["slope_15"] - out["slope_30"]) / 15.0

    # Time-of-day via hours_since_start % 24 (proxy; not absolute wall clock)
    if "hours_since_start" in out.columns:
        tod = pd.to_numeric(out["hours_since_start"], errors="coerce") % 24.0
        out["hour_sin"] = np.sin(2 * np.pi * tod / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * tod / 24.0)
    else:
        out["hour_sin"] = np.nan
        out["hour_cos"] = np.nan

    # Step changes (if available)
    if "steps_since_start" in out.columns:
        steps = pd.to_numeric(out["steps_since_start"], errors="coerce")
        out["steps_15"] = steps - grp["steps_since_start"].apply(
            lambda s: pd.to_numeric(s, errors="coerce")
        ).shift(3).values
        out["steps_30"] = steps - grp["steps_since_start"].apply(
            lambda s: pd.to_numeric(s, errors="coerce")
        ).shift(6).values
    else:
        out["steps_15"] = np.nan
        out["steps_30"] = np.nan

    return out.reset_index(drop=True)


def apply_triggers(at: pd.DataFrame) -> pd.DataFrame:
    at = at.copy()
    at["z_t"] = np.where(
        at["abstain"], np.nan,
        at["one_step_residual"] / (at["interval_width"] + EPS),
    )
    at["up_obs"]    = (~at["abstain"]) & at["z_t"].gt(TAU_UP)
    at["down_obs"]  = (~at["abstain"]) & at["z_t"].lt(-TAU_DOWN)
    at["valid_obs"] = ~at["abstain"]

    grp = at.sort_values(["participant_id", "segment_id", "anchor_ds"]).groupby(
        ["participant_id", "segment_id"], sort=False,
    )
    for col in ("valid_obs", "up_obs", "down_obs"):
        at[f"{col}_r3"] = grp[col].transform(
            lambda s: s.astype(int).rolling(3, min_periods=3).sum()
        )
    at["trigger_up"]   = at["valid_obs_r3"].ge(3) & at["up_obs_r3"].ge(3)
    at["trigger_down"] = at["valid_obs_r3"].ge(3) & at["down_obs_r3"].ge(3)
    at["triggered"]    = at["trigger_up"] | at["trigger_down"]
    return at


def add_episode_features(at: pd.DataFrame) -> pd.DataFrame:
    """Add time_since_trigger, trigger_duration_so_far, cumulative_residual."""
    at = at.sort_values(["participant_id", "segment_id", "anchor_ds"])
    grp = at.groupby(["participant_id", "segment_id"], sort=False)

    at["_prev_trig"] = grp["triggered"].shift(1).fillna(False)
    at["_ep_start"]  = at["triggered"] & ~at["_prev_trig"]
    at["_ep_id"]     = grp["_ep_start"].cumsum()

    # time_since_trigger = steps within episode (0 at first triggered anchor)
    ep_grp = at[at["triggered"]].groupby(
        ["participant_id", "segment_id", "_ep_id"], sort=False
    )
    at.loc[at["triggered"], "time_since_trigger"] = (
        ep_grp["triggered"].cumcount()
    )
    at["time_since_trigger"]      = at["time_since_trigger"].fillna(0)
    at["trigger_duration_so_far"] = at["time_since_trigger"]

    # cumulative one_step_residual since trigger onset
    at["_res_in_ep"] = at["one_step_residual"].where(at["triggered"], 0.0)
    at["cumulative_residual_since_trigger"] = (
        at.groupby(["participant_id", "segment_id", "_ep_id"], sort=False)["_res_in_ep"]
        .cumsum()
    ).fillna(0.0)

    at = at.drop(columns=["_prev_trig", "_ep_start", "_ep_id", "_res_in_ep"])
    return at


def event_enriched_proxy(at: pd.DataFrame) -> pd.Series:
    g = pd.to_numeric(at["current_glucose"], errors="coerce")
    return (
        (at["trigger_up"]   & g.ge(EVENT_GLUCOSE_HI)) |
        (at["trigger_down"] & g.le(EVENT_GLUCOSE_LO))
    ).fillna(False)


# ── feature lists ─────────────────────────────────────────────────────────────

BASE_FEATURES     = ["z_t", "one_step_residual", "current_glucose", "trigger_up"]

ZT_SLOPE_CURV     = ["z_t", "slope_15", "curvature"]

CAUSAL_FULL = [
    "r_t", "r_t_1", "r_t_2", "r_t_3",
    "z_t", "z_t_1", "z_t_2",
    "slope_15", "slope_30", "slope_60",
    "curvature",
    "interval_width",
    "current_glucose",
    "time_since_trigger",
    "trigger_duration_so_far",
    "cumulative_residual_since_trigger",
    "hour_sin", "hour_cos",
    "steps_15", "steps_30",
    "trigger_up_float",
]

# ── triggered forecast rows ───────────────────────────────────────────────────

def get_triggered_rows(cache_split: pd.DataFrame, trig_keys: pd.DataFrame,
                       at_full: pd.DataFrame) -> pd.DataFrame:
    mk = ["participant_id", "segment_id", "anchor_ds"]
    fc = cache_split.merge(trig_keys[mk], on=mk, how="inner")
    extra = [c for c in ([
        "z_t", "z_t_1", "z_t_2",
        "r_t", "r_t_1", "r_t_2", "r_t_3",
        "one_step_residual", "interval_width",
        "trigger_up", "trigger_down", "non_insulin", "event_enriched",
        "slope_15", "slope_30", "slope_60", "curvature",
        "time_since_trigger", "trigger_duration_so_far",
        "cumulative_residual_since_trigger",
        "hour_sin", "hour_cos", "steps_15", "steps_30",
        "trigger_up_float",
    ]) if c in at_full.columns]
    fc = fc.merge(at_full[mk + extra], on=mk, how="left")
    return fc


# ── bootstrap CI utilities ───────────────────────────────────────────────────

def _boot_diff(vals_a: dict, vals_b: dict,
               n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    rng  = np.random.default_rng(SEED)
    pids = list(set(vals_a) | set(vals_b))
    nan_ = np.array([float("nan")])
    boot = []
    for _ in range(n_boot):
        samp = rng.choice(pids, size=len(pids), replace=True)
        a = np.concatenate([vals_a.get(p, nan_) for p in samp])
        b = np.concatenate([vals_b.get(p, nan_) for p in samp])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) and len(b):
            boot.append(float(np.mean(a)) - float(np.mean(b)))
    boot  = np.array(boot)
    pt    = (float(np.mean(np.concatenate(list(vals_a.values())))) -
             float(np.mean(np.concatenate(list(vals_b.values())))))
    return (pt, float(np.percentile(boot, alpha / 2 * 100)),
            float(np.percentile(boot, (1 - alpha / 2) * 100)))


def errs_by_pid(err_df: pd.DataFrame) -> dict:
    return {str(p): g["abs_err"].to_numpy()
            for p, g in err_df.groupby("participant_id", sort=False)}


def evaluate_method(fc: pd.DataFrame, pred: pd.Series, split: str,
                    method: str, subset: str = "all") -> dict:
    """Evaluate one method; reports CORRECT n_anchors."""
    mask = fc["split"].eq(split)
    if subset == "event":
        mask &= fc["event_enriched"].fillna(False)
    elif subset == "placebo":
        mask &= ~fc["event_enriched"].fillna(False)
    sub = fc[mask].copy()
    sub["abs_err"] = (sub["target"] - pred[mask]).abs()
    if sub.empty:
        return {"split": split, "method": method, "subset": subset,
                "n_anchors_tuples": 0, "n_participants": 0,
                "mae": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    by_pid = errs_by_pid(sub)
    rng    = np.random.default_rng(SEED)
    pids   = list(by_pid)
    boot   = []
    for _ in range(N_BOOT):
        samp = rng.choice(pids, size=len(pids), replace=True)
        boot.append(float(np.mean(np.concatenate([by_pid[p] for p in samp]))))
    pt = float(np.mean(np.concatenate(list(by_pid.values()))))
    return {
        "split": split, "method": method, "subset": subset,
        "n_anchors_tuples": n_unique_anchors(sub),   # CORRECT count
        "n_participants": int(sub["participant_id"].nunique()),
        "mae": round(pt, 4),
        "ci_lo": round(float(np.percentile(boot, 2.5)), 4),
        "ci_hi": round(float(np.percentile(boot, 97.5)), 4),
    }


# ── model training helpers ───────────────────────────────────────────────────

def _to_X(fc: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = fc[feats].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)   # NaN → 0 (stream start, lags not yet available)
    return X


def train_per_horizon_ridge(train_fc: pd.DataFrame, feats: list[str],
                            direction: str = "both") -> tuple[dict, StandardScaler]:
    """Train one Ridge per horizon with StandardScaler.

    direction: 'up' | 'down' | 'both'
    """
    if direction == "up":
        train_fc = train_fc[train_fc["trigger_up"].fillna(False)]
    elif direction == "down":
        train_fc = train_fc[train_fc["trigger_down"].fillna(False)]

    scaler = StandardScaler()
    X_all  = _to_X(train_fc, feats)
    scaler.fit(X_all)

    models: dict[int, Ridge] = {}
    for h in range(1, H + 1):
        mask = train_fc["horizon_step"].eq(h)
        Xh   = scaler.transform(_to_X(train_fc[mask], feats))
        yh   = (train_fc.loc[mask, "target"] - train_fc.loc[mask, "q50"]).to_numpy(float)
        mdl  = Ridge(alpha=1.0, fit_intercept=True)
        mdl.fit(Xh, yh)
        models[h] = mdl
    return models, scaler


def train_hgbt(train_fc: pd.DataFrame, feats: list[str],
               direction: str = "both") -> HistGradientBoostingRegressor:
    """Single HGBT trained on all horizons simultaneously.

    horizon_step is appended as an additional feature.
    direction: 'up' | 'down' | 'both'
    """
    if direction == "up":
        train_fc = train_fc[train_fc["trigger_up"].fillna(False)]
    elif direction == "down":
        train_fc = train_fc[train_fc["trigger_down"].fillna(False)]

    extended = feats + ["horizon_step"]
    X = np.column_stack([
        _to_X(train_fc, feats),
        train_fc["horizon_step"].to_numpy(float),
    ])
    y = (train_fc["target"] - train_fc["q50"]).to_numpy(float)
    mdl = HistGradientBoostingRegressor(
        max_iter=200, max_leaf_nodes=31, min_samples_leaf=30,
        learning_rate=0.1, random_state=SEED, verbose=0,
    )
    mdl.fit(X, y)
    return mdl


def train_ols_per_horizon(train_fc: pd.DataFrame, feats: list[str]) -> dict[int, np.ndarray]:
    """Ordinary least squares, one coefficient vector per horizon."""
    coefs: dict[int, np.ndarray] = {}
    for h in range(1, H + 1):
        mask = train_fc["horizon_step"].eq(h)
        X = _to_X(train_fc[mask], feats)
        y = (train_fc.loc[mask, "target"] - train_fc.loc[mask, "q50"]).to_numpy(float)
        # OLS via lstsq (with bias column)
        Xb = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        coefs[h] = coef
    return coefs


# ── prediction helpers ───────────────────────────────────────────────────────

def apply_current_shift(fc: pd.DataFrame) -> pd.Series:
    return fc["q50"] + fc["one_step_residual"]


def apply_decaying_shift(fc: pd.DataFrame, decay: float) -> pd.Series:
    return fc["q50"] + fc["one_step_residual"] * (decay ** (fc["horizon_step"] - 1))


def tune_decay(train_fc: pd.DataFrame) -> float:
    best_decay, best_mae = 1.0, np.inf
    for d in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        pred = apply_decaying_shift(train_fc, d)
        mae  = (train_fc["target"] - pred).abs().mean()
        if mae < best_mae:
            best_mae, best_decay = mae, d
    return best_decay


def apply_ols(fc: pd.DataFrame, coefs: dict[int, np.ndarray],
              feats: list[str]) -> pd.Series:
    pred = fc["q50"].copy()
    for h, coef in coefs.items():
        mask = fc["horizon_step"].eq(h)
        X    = _to_X(fc[mask], feats)
        Xb   = np.column_stack([np.ones(mask.sum()), X])
        pred.loc[mask[mask].index] = fc.loc[mask, "q50"] + Xb @ coef
    return pred


def apply_ridge(fc: pd.DataFrame, models: dict[int, Ridge],
                scaler: StandardScaler, feats: list[str]) -> pd.Series:
    pred = fc["q50"].copy()
    for h, mdl in models.items():
        mask = fc["horizon_step"].eq(h)
        if not mask.any():
            continue
        Xs = scaler.transform(_to_X(fc[mask], feats))
        pred.loc[mask[mask].index] = fc.loc[mask, "q50"] + mdl.predict(Xs)
    return pred


def apply_ridge_split_dir(fc: pd.DataFrame,
                          models_up: dict[int, Ridge], scaler_up: StandardScaler,
                          models_dn: dict[int, Ridge], scaler_dn: StandardScaler,
                          feats: list[str]) -> pd.Series:
    """Apply up head where trigger_up, down head where trigger_down."""
    pred = fc["q50"].copy()
    up_fc = fc[fc["trigger_up"].fillna(False)]
    dn_fc = fc[fc["trigger_down"].fillna(False)]
    for h in range(1, H + 1):
        for sub, models, scaler in [(up_fc, models_up, scaler_up),
                                    (dn_fc, models_dn, scaler_dn)]:
            mask = sub["horizon_step"].eq(h)
            if not mask.any() or h not in models:
                continue
            idx = sub[mask].index
            Xs  = scaler.transform(_to_X(sub[mask], feats))
            pred.loc[idx] = sub.loc[idx, "q50"] + models[h].predict(Xs)
    return pred


def apply_hgbt(fc: pd.DataFrame, mdl: HistGradientBoostingRegressor,
               feats: list[str]) -> pd.Series:
    X = np.column_stack([_to_X(fc, feats), fc["horizon_step"].to_numpy(float)])
    return fc["q50"] + pd.Series(mdl.predict(X), index=fc.index)


def apply_hgbt_split_dir(fc: pd.DataFrame,
                         mdl_up: HistGradientBoostingRegressor,
                         mdl_dn: HistGradientBoostingRegressor,
                         feats: list[str]) -> pd.Series:
    pred = fc["q50"].copy()
    for sub, mdl in [(fc[fc["trigger_up"].fillna(False)], mdl_up),
                     (fc[fc["trigger_down"].fillna(False)], mdl_dn)]:
        if sub.empty:
            continue
        X = np.column_stack([_to_X(sub, feats), sub["horizon_step"].to_numpy(float)])
        pred.loc[sub.index] = sub["q50"] + mdl.predict(X)
    return pred


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Load cache ──────────────────────────────────────────────────────────
    _sec("Loading cache")
    df = pd.read_parquet(CACHE)
    print(f"  {len(df):,} rows  |  cols: {len(df.columns)}")
    cache_fo = df.copy()
    if "anchor_ds" not in cache_fo.columns:
        cache_fo["anchor_ds"] = cache_fo["anchor_time_idx"]

    # ── Build anchor table ──────────────────────────────────────────────────
    _sec("Building anchor table with causal trailing features")
    at = apply_triggers(build_anchor_table_rich(df))
    at = add_episode_features(at)
    at["trigger_up_float"] = at["trigger_up"].astype(float)
    at["event_enriched"]   = event_enriched_proxy(at)

    # Steps features: re-compute properly
    at = at.sort_values(["participant_id", "segment_id", "anchor_ds"])
    grp_at = at.groupby(["participant_id", "segment_id"], sort=False)
    if "steps_since_start" in at.columns:
        s = pd.to_numeric(at["steps_since_start"], errors="coerce")
        at["steps_15"] = s - grp_at["steps_since_start"].transform(
            lambda x: pd.to_numeric(x, errors="coerce").shift(3)
        )
        at["steps_30"] = s - grp_at["steps_since_start"].transform(
            lambda x: pd.to_numeric(x, errors="coerce").shift(6)
        )

    # Final feature availability check
    avail = [f for f in CAUSAL_FULL if at[f].notna().any()]
    missing = [f for f in CAUSAL_FULL if not at[f].notna().any()]
    print(f"  Causal features available ({len(avail)}): {avail}")
    if missing:
        print(f"  NOT AVAILABLE (all NaN): {missing}")
    # HR not in cache – expected
    print(f"  HR changes (hr_15, hr_30): NOT AVAILABLE – not in cache parquet")

    trig_ni = at[at["non_insulin"].fillna(True) & at["triggered"]]
    print(f"  Triggered non-insulin anchors: "
          f"train={trig_ni[trig_ni['split']=='train'].shape[0]:,}  "
          f"val={trig_ni[trig_ni['split']=='validation'].shape[0]:,}  "
          f"test={trig_ni[trig_ni['split']=='test'].shape[0]:,}")

    # ── TASK 1: Anchor-count funnel ─────────────────────────────────────────
    _sec("TASK 1: Anchor-count funnel (test split)")
    at_test = at[at["split"].eq("test")]
    print(f"  A. Anchor table test rows (all):               {len(at_test):>10,}")
    at_test_ni = at_test[at_test["non_insulin"].fillna(True)]
    print(f"  B. Non-insulin test anchors:                   {len(at_test_ni):>10,}")
    at_test_ni_trig = at_test_ni[at_test_ni["triggered"]]
    print(f"  C. Non-insulin TRIGGERED test anchors (Block B):{len(at_test_ni_trig):>10,}")
    cache_test = cache_fo[cache_fo["split"].eq("test")]
    print(f"\n  D. Cache_fo test rows (all horizons 1-12):     {len(cache_test):>10,}")
    print(f"     cache_fo anchor_ds.nunique() [WRONG metric]: {cache_test['anchor_ds'].nunique():>10,}")
    print(f"     Correct n_unique (pid,seg,ds) tuples:        {n_unique_anchors(cache_test):>10,}")
    trig_keys = at_test_ni_trig[["participant_id","segment_id","anchor_ds"]].copy()
    merge1 = cache_test.merge(trig_keys, on=["participant_id","segment_id","anchor_ds"], how="inner")
    print(f"\n  E. After inner join (cache x trig_keys):       {len(merge1):>10,}")
    print(f"     anchor_ds.nunique() [WRONG]:                 {merge1['anchor_ds'].nunique():>10,}")
    print(f"     Correct n_unique (pid,seg,ds) tuples:        {n_unique_anchors(merge1):>10,}")
    extra_for_diag = [c for c in ["z_t","one_step_residual","interval_width","trigger_up",
        "trigger_down","non_insulin","event_enriched"] if c in at.columns]
    at_feats = at[["participant_id","segment_id","anchor_ds"] + extra_for_diag].copy()
    merge2 = merge1.merge(at_feats, on=["participant_id","segment_id","anchor_ds"], how="left")
    split_fc_diag = merge2[merge2["non_insulin"].fillna(True)].copy()
    print(f"\n  F. After anchor-feature join + non_insulin filter: {len(split_fc_diag):>10,}")
    print(f"     Correct n_unique anchors evaluated:           {n_unique_anchors(split_fc_diag):>10,}")
    print(f"\n  FINDING: The previous n_anchors=2,548 in the baseline ladder was")
    print(f"  anchor_ds.nunique() = {split_fc_diag['anchor_ds'].nunique()}, which counts")
    print(f"  distinct anchor_ds INTEGER VALUES (not unique tuples).")
    print(f"  anchor_ds resets per participant-segment stream.")
    print(f"  The CORRECT count is {n_unique_anchors(split_fc_diag):,} unique (pid,seg,anchor_ds) tuples.")
    print(f"  The MAE values themselves are unaffected (grouped by participant_id).")

    # ── Build train triggered forecast rows ─────────────────────────────────
    _sec("Building triggered forecast rows")
    at_full = at.copy()
    split_fc_store: dict[str, pd.DataFrame] = {}

    for split in ("train", "validation", "test"):
        tk = at_full[
            at_full["non_insulin"].fillna(True) &
            at_full["triggered"] &
            at_full["split"].eq(split)
        ][["participant_id", "segment_id", "anchor_ds", "event_enriched"]].copy()
        fc = get_triggered_rows(cache_fo[cache_fo["split"].eq(split)], tk, at_full)
        fc = fc[fc["non_insulin"].fillna(True)].copy()
        split_fc_store[split] = fc
        print(f"  {split}: {len(fc):,} rows | "
              f"{n_unique_anchors(fc):,} anchors | "
              f"{fc['participant_id'].nunique()} participants")

    train_fc = split_fc_store["train"]

    # ── TASK 2: Feature engineering audit ──────────────────────────────────
    _sec("TASK 2: Feature audit on train triggered rows")
    # NOTE: steps_since_start is a sequential anchor-index counter (increments
    # by exactly 1 per 5-min anchor), NOT pedometer data.  Differences are
    # always constant (steps_15=3, steps_30=6), so they carry zero variance and
    # are excluded from modelling below.
    usable_feats = [f for f in CAUSAL_FULL if f in at_full.columns and f in train_fc.columns]
    zero_var: list[str] = []
    for f in list(usable_feats):
        col = pd.to_numeric(train_fc[f], errors="coerce")
        if col.std() == 0.0:
            zero_var.append(f)
            usable_feats.remove(f)
    if zero_var:
        print(f"  EXCLUDED (zero variance): {zero_var}")
    for f in usable_feats:
        col = pd.to_numeric(train_fc[f], errors="coerce")
        frac_valid = col.notna().mean()
        print(f"  {f:40s}  valid={frac_valid:.3f}  "
              f"mean={col.mean():.3f}  std={col.std():.3f}")

    # ── TASK 3: Train models ────────────────────────────────────────────────
    _sec("TASK 3: Training all models")

    # Decay baseline
    best_decay = tune_decay(train_fc)
    print(f"  Best decaying-shift decay: {best_decay}")

    # OLS baselines
    print("  Training OLS baselines...")
    coefs_zt   = train_ols_per_horizon(train_fc, ["z_t"])
    coefs_zt_sc = train_ols_per_horizon(train_fc, ZT_SLOPE_CURV)
    coefs_full  = train_ols_per_horizon(train_fc, usable_feats)

    # Ridge combined (all directions)
    print("  Training Ridge (full features, combined)...")
    ridge_full_comb, scaler_full_comb = train_per_horizon_ridge(train_fc, usable_feats, "both")

    # Ridge split by direction
    print("  Training Ridge up head...")
    ridge_up, scaler_up = train_per_horizon_ridge(train_fc, usable_feats, "up")
    print("  Training Ridge down head...")
    ridge_dn, scaler_dn = train_per_horizon_ridge(train_fc, usable_feats, "down")
    n_up_train = int((train_fc["trigger_up"].fillna(False)).groupby(
        train_fc["participant_id"]).any().sum())
    n_dn_train = int((train_fc["trigger_down"].fillna(False)).groupby(
        train_fc["participant_id"]).any().sum())
    print(f"  Train participants with trigger_up: {n_up_train}  trigger_down: {n_dn_train}")

    # HistGradientBoosting combined
    print("  Training HistGradientBoosting (full features, combined)...")
    hgbt_comb = train_hgbt(train_fc, usable_feats, "both")

    # HistGradientBoosting split by direction
    print("  Training HistGradientBoosting up head...")
    hgbt_up = train_hgbt(train_fc, usable_feats, "up")
    print("  Training HistGradientBoosting down head...")
    hgbt_dn = train_hgbt(train_fc, usable_feats, "down")

    print("  All models trained.")

    # ── TASK 4: Baseline ladder ─────────────────────────────────────────────
    _sec("TASK 4: Baseline ladder")

    ladder_rows: list[dict] = []
    err_frames: dict[str, dict[str, pd.DataFrame]] = {}

    for split in ("validation", "test"):
        fc = split_fc_store[split]
        preds: dict[str, pd.Series] = {
            "base":               fc["q50"],
            "current_shift":      apply_current_shift(fc),
            "decaying_shift":     apply_decaying_shift(fc, best_decay),
            "linear_zt":          apply_ols(fc, coefs_zt,    ["z_t"]),
            "linear_zt_sc":       apply_ols(fc, coefs_zt_sc, ZT_SLOPE_CURV),
            "linear_full":        apply_ols(fc, coefs_full,  usable_feats),
            "ridge_full_comb":    apply_ridge(fc, ridge_full_comb, scaler_full_comb, usable_feats),
            "ridge_split_dir":    apply_ridge_split_dir(fc, ridge_up, scaler_up, ridge_dn, scaler_dn, usable_feats),
            "hgbt_full_comb":     apply_hgbt(fc, hgbt_comb,  usable_feats),
            "hgbt_split_dir":     apply_hgbt_split_dir(fc, hgbt_up, hgbt_dn, usable_feats),
        }
        err_frames[split] = {}
        for mname, pred in preds.items():
            edf = fc[["participant_id", "segment_id", "anchor_ds", "event_enriched"]].copy()
            edf["abs_err"] = (fc["target"] - pred).abs().values
            err_frames[split][mname] = edf
            row = evaluate_method(fc, pred, split, mname, "all")
            ladder_rows.append(row)

    ladder_df = pd.DataFrame(ladder_rows)
    ladder_path = OUT / "study2_rich_ladder.csv"
    ladder_df.to_csv(ladder_path, index=False)

    METHOD_LABELS = {
        "base":             "Base (frozen SSM)",
        "current_shift":    "Current shift",
        "decaying_shift":   f"Decaying shift (δ={best_decay})",
        "linear_zt":        "Linear z_t",
        "linear_zt_sc":     "Linear z_t + slope + curv",
        "linear_full":      "Linear (full causal OLS)",
        "ridge_full_comb":  "Ridge (full, combined dir)",
        "ridge_split_dir":  "Ridge (full, split up/down)",
        "hgbt_full_comb":   "HGBT (full, combined dir)",
        "hgbt_split_dir":   "HGBT (full, split up/down)",
    }
    for split in ("validation", "test"):
        _sec(f"Ladder ({split}, all triggered non-insulin)")
        s = ladder_df[ladder_df["split"].eq(split)]
        print(f"  {'Method':35s}  {'n_anchors':>10}  {'n_part':>6}  "
              f"{'MAE':>8}  {'CI_lo':>8}  {'CI_hi':>8}")
        print("  " + "-" * 80)
        for mname, mlabel in METHOD_LABELS.items():
            row = s[s["method"].eq(mname)]
            if row.empty:
                continue
            r = row.iloc[0]
            print(f"  {mlabel:35s}  {int(r['n_anchors_tuples']):>10,}  "
                  f"{int(r['n_participants']):>6}  "
                  f"{r['mae']:>8.4f}  {r['ci_lo']:>8.4f}  {r['ci_hi']:>8.4f}")

    # ── TASK 5: Paired CIs ──────────────────────────────────────────────────
    _sec("TASK 5: Paired difference CIs (test split, participant-clustered bootstrap)")

    # Find best simple baseline on test
    test_s = ladder_df[ladder_df["split"].eq("test")]
    simple_methods = ["base", "current_shift", "decaying_shift", "linear_zt", "linear_zt_sc"]
    best_simple = test_s[test_s["method"].isin(simple_methods)].sort_values("mae").iloc[0]
    best_simple_name = best_simple["method"]
    print(f"\n  Best simple baseline on test: {best_simple_name} "
          f"(MAE={best_simple['mae']:.4f})")

    pairs = [
        (best_simple_name, "ridge_split_dir"),
        (best_simple_name, "hgbt_split_dir"),
        ("ridge_split_dir", "hgbt_split_dir"),
        ("linear_zt",        "ridge_split_dir"),
        ("linear_full",      "ridge_split_dir"),
        ("linear_full",      "hgbt_split_dir"),
    ]
    diff_rows = []
    for (ma, mb) in pairs:
        if ma not in err_frames["test"] or mb not in err_frames["test"]:
            continue
        diff, lo, hi = _boot_diff(errs_by_pid(err_frames["test"][ma]),
                                  errs_by_pid(err_frames["test"][mb]))
        excl = lo > 0
        label = f"{METHOD_LABELS.get(ma, ma)} - {METHOD_LABELS.get(mb, mb)}"
        print(f"  {label}")
        print(f"    diff={diff:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  excl_0={excl}")
        diff_rows.append({
            "split": "test", "method_a": ma, "method_b": mb,
            "comparison": f"{ma}_minus_{mb}",
            "diff": diff, "ci_lo": lo, "ci_hi": hi,
            "ci_excludes_zero": excl,
        })
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(OUT / "study2_rich_diff_cis.csv", index=False)

    # ── Save manifest ────────────────────────────────────────────────────────
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "causal_features_used": usable_feats,
        "features_unavailable": [f for f in CAUSAL_FULL if f not in usable_feats],
        "hr_features": "NOT AVAILABLE – HR not in cache parquet",
        "best_decay": best_decay,
        "models_trained": list(METHOD_LABELS.keys()),
        "n_anchors_note": (
            "n_anchors_tuples counts unique (participant_id, segment_id, anchor_ds) tuples. "
            "The old n_anchors column (anchor_ds.nunique()) was incorrect because anchor_ds "
            "is a per-stream relative integer, not globally unique."
        ),
        "artifacts": {
            "ladder": str(ladder_path),
            "diff_cis": str(OUT / "study2_rich_diff_cis.csv"),
        },
    }
    (OUT / "study2_rich_manifest.json").write_text(json.dumps(manifest, indent=2))

    _sec("Done")
    print(f"  Artifacts in: {OUT}")


if __name__ == "__main__":
    main()
