#!/usr/bin/env python3
"""Study 2 – Finalize the rich event-aware correction head.

Tasks:
  1.  Select the final head using validation metrics only.
  2.  Report val + test MAE with participant-clustered 95 % CIs (all methods).
  3.  Paired CIs: HGBT-comb vs each alternative; final head vs base & decaying.
  4.  Event-enriched proxy vs unmatched control (descriptive, no matching).
  5.  Per-horizon MAE (0-30 min, 30-60 min, terminal 60 min) + shape metrics.
  6.  Feature-group ablation study (Ridge) on 6 subsets.
  7.  Save artifacts to outputs/study2_final_rich_head/.
  8.  Write manifest.
  9.  Print final verdict (cautious).

Run:
    /home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python \\
        scripts/run_study2_final_rich_head.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
CACHE   = ROOT / "outputs/study2_forecast_cache_5min/study2_forecast_cache.parquet"
OUT     = ROOT / "outputs/study2_final_rich_head"

TAU_UP   = 0.2
TAU_DOWN = 0.2
EPS      = 1e-3
N_BOOT   = 1000
SEED     = 42
H        = 12
BIN_MIN  = 5

EVENT_GLUCOSE_HI = 130.0
EVENT_GLUCOSE_LO =  85.0

# horizon groups (5-min bins)
HORIZON_GROUPS = {
    "h01_06_0_30min":    list(range(1, 7)),
    "h07_12_30_60min":   list(range(7, 13)),
    "h12_terminal_60min": [12],
    "h01_12_all":        list(range(1, 13)),
}

# Ablation feature groups
RESIDUALS  = ["r_t", "r_t_1", "r_t_2", "r_t_3"]
SLOPES     = ["slope_15", "slope_30", "slope_60", "curvature"]
EP_PHASE   = ["time_since_trigger", "trigger_duration_so_far",
               "cumulative_residual_since_trigger", "trigger_up_float"]
CONTEXT    = ["interval_width", "current_glucose", "hour_sin", "hour_cos"]
ZT_ONLY    = ["z_t"]
ZT_SC      = ["z_t", "slope_15", "curvature"]

ABLATION_GROUPS = {
    "z_t_only":        ZT_ONLY,
    "z_t_slope_curv":  ZT_SC,
    "residual_only":   ["r_t"],
    "residual_history": RESIDUALS,
    "slope_curvature": SLOPES,
    "trigger_phase":   EP_PHASE,
    "context_only":    CONTEXT,
    "full_causal":     None,   # filled in after usable_feats is known
}


def _sec(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


# ── n_anchors (corrected) ─────────────────────────────────────────────────────
def n_unique_anchors(df: pd.DataFrame) -> int:
    """Correct count: distinct (participant_id, segment_id, anchor_ds) tuples."""
    return int(df.drop_duplicates(["participant_id", "segment_id", "anchor_ds"]).shape[0])


# ── feature engineering ───────────────────────────────────────────────────────

def build_anchor_table_rich(df: pd.DataFrame) -> pd.DataFrame:
    fo = df.copy()
    if "anchor_ds" not in fo.columns:
        fo["anchor_ds"] = fo["anchor_time_idx"]

    meta_cols = [c for c in [
        "participant_id", "segment_id", "split", "anchor_ds",
        "current_glucose", "med_insulin", "participants_study_group",
        "hba1c_percent_baseline", "participants_clinical_site",
        "med_any_diabetes_drug", "hours_since_start", "steps_since_start",
    ] if c in fo.columns]

    anchors = (
        fo.sort_values(["participant_id", "segment_id", "anchor_ds"])
        .drop_duplicates(["participant_id", "segment_id", "anchor_ds"])[meta_cols]
        .copy()
    )

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
    out["abstain"] = reasons.ne("none")
    out["non_insulin"] = pd.to_numeric(
        out.get("med_insulin", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0).eq(0)

    out = out.sort_values(["participant_id", "segment_id", "anchor_ds"])
    grp = out.groupby(["participant_id", "segment_id"], sort=False)

    out["r_t"]   = out["one_step_residual"]
    out["r_t_1"] = grp["one_step_residual"].shift(1)
    out["r_t_2"] = grp["one_step_residual"].shift(2)
    out["r_t_3"] = grp["one_step_residual"].shift(3)

    iw_safe        = out["interval_width"].where(out["interval_width"].gt(0), np.nan)
    out["z_t_raw"] = np.where(out["abstain"], np.nan, out["one_step_residual"] / (iw_safe + EPS))
    out["z_t_1"]   = grp["z_t_raw"].shift(1)
    out["z_t_2"]   = grp["z_t_raw"].shift(2)

    out["glucose_lag3"]  = grp["current_glucose"].shift(3)
    out["glucose_lag6"]  = grp["current_glucose"].shift(6)
    out["glucose_lag12"] = grp["current_glucose"].shift(12)
    gluc = out["current_glucose"]
    out["slope_15"] = (gluc - out["glucose_lag3"])  / 15.0
    out["slope_30"] = (gluc - out["glucose_lag6"])  / 30.0
    out["slope_60"] = (gluc - out["glucose_lag12"]) / 60.0
    out["curvature"] = (out["slope_15"] - out["slope_30"]) / 15.0

    if "hours_since_start" in out.columns:
        tod = pd.to_numeric(out["hours_since_start"], errors="coerce") % 24.0
        out["hour_sin"] = np.sin(2 * np.pi * tod / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * tod / 24.0)
    else:
        out["hour_sin"] = out["hour_cos"] = np.nan

    # steps_since_start is a sequential counter (not pedometer); differences
    # are always constant (steps_15=3, steps_30=6) so excluded.

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
    at = at.sort_values(["participant_id", "segment_id", "anchor_ds"])
    grp = at.groupby(["participant_id", "segment_id"], sort=False)
    at["_prev_trig"] = grp["triggered"].shift(1).fillna(False)
    at["_ep_start"]  = at["triggered"] & ~at["_prev_trig"]
    at["_ep_id"]     = grp["_ep_start"].cumsum()
    at.loc[at["triggered"], "time_since_trigger"] = (
        at[at["triggered"]]
        .groupby(["participant_id", "segment_id", "_ep_id"], sort=False)["triggered"]
        .cumcount()
    )
    at["time_since_trigger"]      = at["time_since_trigger"].fillna(0)
    at["trigger_duration_so_far"] = at["time_since_trigger"]
    at["_res_in_ep"] = at["one_step_residual"].where(at["triggered"], 0.0)
    at["cumulative_residual_since_trigger"] = (
        at.groupby(["participant_id", "segment_id", "_ep_id"], sort=False)["_res_in_ep"]
        .cumsum()
    ).fillna(0.0)
    return at.drop(columns=["_prev_trig", "_ep_start", "_ep_id", "_res_in_ep"])


def event_enriched_proxy(at: pd.DataFrame) -> pd.Series:
    g = pd.to_numeric(at["current_glucose"], errors="coerce")
    return (
        (at["trigger_up"]   & g.ge(EVENT_GLUCOSE_HI)) |
        (at["trigger_down"] & g.le(EVENT_GLUCOSE_LO))
    ).fillna(False)


def get_triggered_rows(cache_split: pd.DataFrame, trig_keys: pd.DataFrame,
                       at_full: pd.DataFrame, usable_feats: list[str]) -> pd.DataFrame:
    mk = ["participant_id", "segment_id", "anchor_ds"]
    fc = cache_split.merge(trig_keys[mk], on=mk, how="inner")
    base_extra = ["z_t", "z_t_1", "z_t_2",
                  "r_t", "r_t_1", "r_t_2", "r_t_3",
                  "one_step_residual", "interval_width",
                  "trigger_up", "trigger_down", "non_insulin", "event_enriched",
                  "current_glucose"]
    extra = [c for c in (base_extra + usable_feats) if c in at_full.columns]
    extra = list(dict.fromkeys(extra))   # dedup, preserve order
    # Drop from fc any columns that will come from at_full to avoid suffix conflicts
    cols_to_drop = [c for c in extra if c in fc.columns and c not in mk]
    if cols_to_drop:
        fc = fc.drop(columns=cols_to_drop)
    fc = fc.merge(at_full[mk + extra], on=mk, how="left")
    return fc


# ── bootstrap helpers ─────────────────────────────────────────────────────────

def _boot_ci(by_pid: dict[str, np.ndarray], rng: np.random.Generator,
             n_boot: int = N_BOOT) -> tuple[float, float, float]:
    pids = list(by_pid)
    boot = []
    for _ in range(n_boot):
        samp = rng.choice(pids, size=len(pids), replace=True)
        boot.append(float(np.mean(np.concatenate([by_pid[p] for p in samp]))))
    pt = float(np.mean(np.concatenate(list(by_pid.values()))))
    return (pt, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def _boot_diff(by_pid_a: dict, by_pid_b: dict, rng: np.random.Generator,
               n_boot: int = N_BOOT) -> tuple[float, float, float]:
    pids  = list(set(by_pid_a) | set(by_pid_b))
    nan_  = np.array([float("nan")])
    boot  = []
    for _ in range(n_boot):
        samp = rng.choice(pids, size=len(pids), replace=True)
        a = np.concatenate([by_pid_a.get(p, nan_) for p in samp])
        b = np.concatenate([by_pid_b.get(p, nan_) for p in samp])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) and len(b):
            boot.append(float(np.mean(a)) - float(np.mean(b)))
    pt = (float(np.mean(np.concatenate(list(by_pid_a.values())))) -
          float(np.mean(np.concatenate(list(by_pid_b.values())))))
    return (pt, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def errs_by_pid(df: pd.DataFrame, pred: pd.Series) -> dict[str, np.ndarray]:
    ae = (df["target"] - pred).abs()
    out: dict[str, np.ndarray] = {}
    for pid, idx in df.groupby("participant_id", sort=False).groups.items():
        out[str(pid)] = ae.loc[idx].to_numpy()
    return out


def errs_by_pid_subset(df: pd.DataFrame, pred: pd.Series,
                       mask: pd.Series) -> dict[str, np.ndarray]:
    sub = df[mask]
    ae  = (sub["target"] - pred[mask]).abs()
    out: dict[str, np.ndarray] = {}
    for pid, idx in sub.groupby("participant_id", sort=False).groups.items():
        out[str(pid)] = ae.loc[idx].to_numpy()
    return out


# ── model helpers ─────────────────────────────────────────────────────────────

def _to_X(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = df[feats].to_numpy(dtype=float)
    return np.where(np.isfinite(X), X, 0.0)


def train_ridge(train_fc: pd.DataFrame, feats: list[str],
                direction: str = "both") -> tuple[dict[int, Ridge], StandardScaler]:
    if direction == "up":
        train_fc = train_fc[train_fc["trigger_up"].fillna(False)]
    elif direction == "down":
        train_fc = train_fc[train_fc["trigger_down"].fillna(False)]
    scaler = StandardScaler()
    scaler.fit(_to_X(train_fc, feats))
    models: dict[int, Ridge] = {}
    for h in range(1, H + 1):
        m = train_fc["horizon_step"].eq(h)
        Xh = scaler.transform(_to_X(train_fc[m], feats))
        yh = (train_fc.loc[m, "target"] - train_fc.loc[m, "q50"]).to_numpy(float)
        models[h] = Ridge(alpha=1.0).fit(Xh, yh)
    return models, scaler


def train_hgbt(train_fc: pd.DataFrame, feats: list[str],
               direction: str = "both") -> HistGradientBoostingRegressor:
    if direction == "up":
        train_fc = train_fc[train_fc["trigger_up"].fillna(False)]
    elif direction == "down":
        train_fc = train_fc[train_fc["trigger_down"].fillna(False)]
    X = np.column_stack([_to_X(train_fc, feats), train_fc["horizon_step"].to_numpy(float)])
    y = (train_fc["target"] - train_fc["q50"]).to_numpy(float)
    return HistGradientBoostingRegressor(
        max_iter=200, max_leaf_nodes=31, min_samples_leaf=30,
        learning_rate=0.1, random_state=SEED, verbose=0,
    ).fit(X, y)


def train_ols(train_fc: pd.DataFrame, feats: list[str]) -> dict[int, np.ndarray]:
    coefs: dict[int, np.ndarray] = {}
    for h in range(1, H + 1):
        m  = train_fc["horizon_step"].eq(h)
        X  = _to_X(train_fc[m], feats)
        Xb = np.column_stack([np.ones(X.shape[0]), X])
        y  = (train_fc.loc[m, "target"] - train_fc.loc[m, "q50"]).to_numpy(float)
        coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        coefs[h] = coef
    return coefs


# ── prediction functions ──────────────────────────────────────────────────────

def pred_base(fc: pd.DataFrame) -> pd.Series:
    return fc["q50"].copy()


def pred_decaying(fc: pd.DataFrame, decay: float) -> pd.Series:
    return fc["q50"] + fc["one_step_residual"] * (decay ** (fc["horizon_step"] - 1))


def pred_ols(fc: pd.DataFrame, coefs: dict[int, np.ndarray], feats: list[str]) -> pd.Series:
    pred = fc["q50"].copy()
    for h, coef in coefs.items():
        m  = fc["horizon_step"].eq(h)
        Xb = np.column_stack([np.ones(m.sum()), _to_X(fc[m], feats)])
        pred.loc[m[m].index] = fc.loc[m, "q50"] + Xb @ coef
    return pred


def pred_ridge(fc: pd.DataFrame, models: dict[int, Ridge],
               scaler: StandardScaler, feats: list[str]) -> pd.Series:
    pred = fc["q50"].copy()
    for h, mdl in models.items():
        m = fc["horizon_step"].eq(h)
        if not m.any():
            continue
        Xs = scaler.transform(_to_X(fc[m], feats))
        pred.loc[m[m].index] = fc.loc[m, "q50"] + mdl.predict(Xs)
    return pred


def pred_ridge_split(fc: pd.DataFrame,
                     mdl_up: tuple, mdl_dn: tuple, feats: list[str]) -> pd.Series:
    pred = fc["q50"].copy()
    for sub, (models, scaler) in [
        (fc[fc["trigger_up"].fillna(False)],   mdl_up),
        (fc[fc["trigger_down"].fillna(False)], mdl_dn),
    ]:
        for h in range(1, H + 1):
            m = sub["horizon_step"].eq(h)
            if not m.any() or h not in models:
                continue
            Xs = scaler.transform(_to_X(sub[m], feats))
            pred.loc[sub[m].index] = sub.loc[m, "q50"] + models[h].predict(Xs)
    return pred


def pred_hgbt(fc: pd.DataFrame, mdl: HistGradientBoostingRegressor,
              feats: list[str]) -> pd.Series:
    X = np.column_stack([_to_X(fc, feats), fc["horizon_step"].to_numpy(float)])
    return fc["q50"] + pd.Series(mdl.predict(X), index=fc.index)


def pred_hgbt_split(fc: pd.DataFrame, mdl_up: HistGradientBoostingRegressor,
                    mdl_dn: HistGradientBoostingRegressor, feats: list[str]) -> pd.Series:
    pred = fc["q50"].copy()
    for sub, mdl in [(fc[fc["trigger_up"].fillna(False)],   mdl_up),
                     (fc[fc["trigger_down"].fillna(False)], mdl_dn)]:
        if sub.empty:
            continue
        X = np.column_stack([_to_X(sub, feats), sub["horizon_step"].to_numpy(float)])
        pred.loc[sub.index] = sub["q50"] + mdl.predict(X)
    return pred


# ── evaluation helpers ────────────────────────────────────────────────────────

def evaluate_mae(fc: pd.DataFrame, pred: pd.Series,
                 rng: np.random.Generator,
                 mask: pd.Series | None = None) -> dict:
    sub = fc[mask] if mask is not None else fc
    p   = pred[sub.index] if mask is not None else pred
    bp  = errs_by_pid(sub, p)
    if not bp:
        return {"mae": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_anchors": 0, "n_participants": 0}
    pt, lo, hi = _boot_ci(bp, rng)
    return {
        "mae": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        "n_anchors": n_unique_anchors(sub),
        "n_participants": int(sub["participant_id"].nunique()),
    }


def evaluate_per_horizon(fc: pd.DataFrame, pred: pd.Series) -> dict[int, float]:
    """Mean absolute error per horizon step."""
    out: dict[int, float] = {}
    for h in range(1, H + 1):
        m = fc["horizon_step"].eq(h)
        if m.any():
            out[h] = float((fc.loc[m, "target"] - pred[m]).abs().mean())
    return out


def evaluate_shape_metrics(fc: pd.DataFrame, pred: pd.Series) -> dict[str, float]:
    """
    Cumulative residual error (mean signed error per anchor, averaged over participants),
    direction accuracy (fraction of horizon-anchor pairs where correction direction
    matches the trigger direction), peak error (mean over participants of per-anchor
    max absolute error).
    """
    err  = fc["target"] - pred

    # --- cumulative residual error ---
    # per anchor: mean signed error across horizons; then participant average
    fc2         = fc.copy()
    fc2["err"]  = err.values
    cum_res_by_anchor = fc2.groupby(
        ["participant_id", "segment_id", "anchor_ds"]
    )["err"].mean()
    cum_res_by_pid = cum_res_by_anchor.groupby(level="participant_id").mean()
    cum_res_err = float(cum_res_by_pid.mean())

    # --- direction accuracy ---
    # expected: trigger_up → correction > 0 (pred > q50)
    correction = (pred - fc["q50"]).values
    is_up      = fc["trigger_up"].fillna(False).values
    exp_sign   = np.where(is_up, 1.0, -1.0)
    correct    = (np.sign(correction) * exp_sign) > 0
    dir_acc    = float(correct.mean())

    # --- peak error (mean per-participant of per-anchor max |error|) ---
    fc2["abs_err"] = err.abs().values
    peak_by_anchor = fc2.groupby(
        ["participant_id", "segment_id", "anchor_ds"]
    )["abs_err"].max()
    peak_by_pid = peak_by_anchor.groupby(level="participant_id").mean()
    peak_err    = float(peak_by_pid.mean())

    return {
        "cumulative_residual_err": round(cum_res_err, 4),
        "direction_accuracy":      round(dir_acc, 4),
        "peak_err_mean":           round(peak_err, 4),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ── Load and build ─────────────────────────────────────────────────────
    _sec("Loading cache and building anchor table")
    df = pd.read_parquet(CACHE)
    print(f"  {len(df):,} rows")
    cache_fo = df.copy()
    if "anchor_ds" not in cache_fo.columns:
        cache_fo["anchor_ds"] = cache_fo["anchor_time_idx"]

    at = apply_triggers(build_anchor_table_rich(df))
    at = add_episode_features(at)
    at["trigger_up_float"] = at["trigger_up"].astype(float)
    at["event_enriched"]   = event_enriched_proxy(at)
    at = at.sort_values(["participant_id", "segment_id", "anchor_ds"])

    # Determine usable features (exclude zero-variance on train)
    ALL_CANDIDATE = [
        "r_t", "r_t_1", "r_t_2", "r_t_3",
        "z_t", "z_t_1", "z_t_2",
        "slope_15", "slope_30", "slope_60", "curvature",
        "interval_width", "current_glucose",
        "time_since_trigger", "trigger_duration_so_far",
        "cumulative_residual_since_trigger",
        "hour_sin", "hour_cos", "trigger_up_float",
    ]
    USABLE = [f for f in ALL_CANDIDATE if f in at.columns]

    # Build triggered forecast rows
    _sec("Building triggered forecast rows")
    splits_fc: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        tk = at[
            at["non_insulin"].fillna(True) & at["triggered"] & at["split"].eq(split)
        ][["participant_id", "segment_id", "anchor_ds", "event_enriched"]].copy()
        fc = get_triggered_rows(cache_fo[cache_fo["split"].eq(split)], tk, at, USABLE)
        fc = fc[fc["non_insulin"].fillna(True)].copy()
        splits_fc[split] = fc
        print(f"  {split}: {len(fc):,} rows | "
              f"{n_unique_anchors(fc):,} anchors | "
              f"{fc['participant_id'].nunique()} participants")

    train_fc = splits_fc["train"]
    val_fc   = splits_fc["validation"]
    test_fc  = splits_fc["test"]

    # Zero-variance filter on USABLE
    zero_var = [f for f in USABLE if train_fc[f].std() == 0.0]
    if zero_var:
        print(f"  Excluding zero-variance features: {zero_var}")
    USABLE = [f for f in USABLE if f not in zero_var]
    ABLATION_GROUPS["full_causal"] = USABLE
    print(f"  Final usable feature set ({len(USABLE)}): {USABLE}")

    # Tune decay on train
    best_decay = 0.7  # confirmed from prior run
    for d in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p   = train_fc["q50"] + train_fc["one_step_residual"] * (d ** (train_fc["horizon_step"] - 1))
        mae = (train_fc["target"] - p).abs().mean()
        if d == 0.5 or mae < best_mae:  # noqa
            best_mae  = mae
            best_decay = d
    print(f"  Best decay = {best_decay}")

    # ── TASK 3: Train all primary models ───────────────────────────────────
    _sec("Training primary models")

    ols_zt_sc = train_ols(train_fc, ZT_SC)
    print("  OLS (z_t + slope + curv)  done")

    ols_full  = train_ols(train_fc, USABLE)
    print("  OLS (full causal)         done")

    ridge_comb, scaler_comb = train_ridge(train_fc, USABLE, "both")
    print("  Ridge combined            done")

    ridge_up,   scaler_up   = train_ridge(train_fc, USABLE, "up")
    ridge_dn,   scaler_dn   = train_ridge(train_fc, USABLE, "down")
    print("  Ridge split up/down       done")

    hgbt_comb = train_hgbt(train_fc, USABLE, "both")
    print("  HGBT combined             done")

    hgbt_up   = train_hgbt(train_fc, USABLE, "up")
    hgbt_dn   = train_hgbt(train_fc, USABLE, "down")
    print("  HGBT split up/down        done")

    # ── Build prediction catalogue (val + test) ─────────────────────────────
    def make_preds(fc: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "base":            pred_base(fc),
            "decaying_shift":  pred_decaying(fc, best_decay),
            "linear_zt_sc":    pred_ols(fc, ols_zt_sc,  ZT_SC),
            "linear_full_ols": pred_ols(fc, ols_full,   USABLE),
            "ridge_full_comb": pred_ridge(fc, ridge_comb, scaler_comb, USABLE),
            "ridge_split_dir": pred_ridge_split(fc, (ridge_up, scaler_up), (ridge_dn, scaler_dn), USABLE),
            "hgbt_full_comb":  pred_hgbt(fc, hgbt_comb, USABLE),
            "hgbt_split_dir":  pred_hgbt_split(fc, hgbt_up, hgbt_dn, USABLE),
        }

    val_preds  = make_preds(val_fc)
    test_preds = make_preds(test_fc)

    METHOD_LABELS = {
        "base":            "Base (frozen SSM)",
        "decaying_shift":  f"Decaying shift (δ={best_decay})",
        "linear_zt_sc":    "Linear z_t + slope + curv (best simple)",
        "linear_full_ols": "Linear full causal OLS",
        "ridge_full_comb": "Ridge full (combined dir)",
        "ridge_split_dir": "Ridge full (split up/down)",
        "hgbt_full_comb":  "HGBT full (combined dir)",
        "hgbt_split_dir":  "HGBT full (split up/down)",
    }

    # ── TASK 1 + 2: Validation-based model selection ────────────────────────
    _sec("TASK 1: Validation-based model selection")

    val_results: list[dict] = []
    val_by_pid:  dict[str, dict] = {}
    for mname, pred in val_preds.items():
        bp  = errs_by_pid(val_fc, pred)
        pt, lo, hi = _boot_ci(bp, np.random.default_rng(SEED))
        val_results.append({
            "method": mname, "label": METHOD_LABELS[mname],
            "val_mae": round(pt, 4), "val_ci_lo": round(lo, 4), "val_ci_hi": round(hi, 4),
        })
        val_by_pid[mname] = bp

    val_results.sort(key=lambda r: r["val_mae"])
    print(f"\n  {'Method':<35}  {'val MAE':>8}  {'95% CI':>22}")
    print("  " + "-" * 70)
    for r in val_results:
        star = " *" if r["method"] == val_results[0]["method"] else ""
        print(f"  {r['label']:<35}  {r['val_mae']:>8.4f}  "
              f"[{r['val_ci_lo']:>+8.4f}, {r['val_ci_hi']:>+8.4f}]{star}")

    # Best on validation
    best_val   = val_results[0]["method"]
    second_val = val_results[1]["method"]

    # Paired CI between best and second-best on validation
    diff_pt, diff_lo, diff_hi = _boot_diff(
        val_by_pid[best_val], val_by_pid[second_val], np.random.default_rng(SEED)
    )
    excludes_zero = diff_lo > 0
    print(f"\n  Validation paired CI: {best_val} - {second_val}")
    print(f"  diff={diff_pt:+.4f}  CI=[{diff_lo:+.4f}, {diff_hi:+.4f}]  "
          f"excludes_zero={excludes_zero}")

    if not excludes_zero:
        # Tie on validation → prefer simpler model
        PREFERENCE_ORDER = [
            "linear_zt_sc", "linear_full_ols",
            "ridge_full_comb", "ridge_split_dir",
            "hgbt_full_comb", "hgbt_split_dir",
        ]
        for preferred in PREFERENCE_ORDER:
            if preferred in {best_val, second_val}:
                final_head = preferred
                break
        print(f"  Tied on validation → prefer simpler: {final_head}")
    else:
        final_head = best_val
        print(f"  Clear winner on validation: {final_head}")

    print(f"\n  ** SELECTED FINAL HEAD (by validation): {final_head} **")
    print(f"     Label: {METHOD_LABELS[final_head]}")

    # ── TASK 2: Full val + test ladder ─────────────────────────────────────
    _sec("TASK 2: Full validation + test MAE ladder")

    ladder_rows: list[dict] = []
    test_by_pid: dict[str, dict] = {}

    for split, fc, preds_d in [("validation", val_fc, val_preds),
                                ("test", test_fc, test_preds)]:
        print(f"\n  {'Method':<35}  {'n_anchors':>10}  "
              f"{'MAE':>8}  {'CI_lo':>8}  {'CI_hi':>8}")
        print(f"  ── {split} ──" + "─" * 55)
        for mname, pred in preds_d.items():
            r = evaluate_mae(fc, pred, np.random.default_rng(SEED))
            r["split"]  = split
            r["method"] = mname
            r["label"]  = METHOD_LABELS[mname]
            ladder_rows.append(r)
            print(f"  {METHOD_LABELS[mname]:<35}  {r['n_anchors']:>10,}  "
                  f"{r['mae']:>8.4f}  {r['ci_lo']:>8.4f}  {r['ci_hi']:>8.4f}")
            if split == "test":
                test_by_pid[mname] = errs_by_pid(fc, pred)

    ladder_df = pd.DataFrame(ladder_rows)
    ladder_df.to_csv(OUT / "study2_final_ladder.csv", index=False)

    # ── TASK 3: Paired CIs (test, participant-clustered) ────────────────────
    _sec("TASK 3: Paired difference CIs (test)")

    COMPARISONS = [
        ("hgbt_full_comb", "ridge_split_dir",  "HGBT-comb vs Ridge-split"),
        ("hgbt_full_comb", "hgbt_split_dir",   "HGBT-comb vs HGBT-split"),
        ("hgbt_full_comb", "linear_full_ols",  "HGBT-comb vs Linear-full-OLS"),
        ("hgbt_full_comb", "linear_zt_sc",     "HGBT-comb vs best-simple"),
        (final_head,       "base",             "Final head vs Base"),
        (final_head,       "decaying_shift",   "Final head vs Decaying shift"),
    ]
    diff_rows: list[dict] = []
    for ma, mb, label in COMPARISONS:
        if ma not in test_by_pid or mb not in test_by_pid:
            print(f"  [{label}] – skipped (missing predictions)")
            continue
        pt, lo, hi = _boot_diff(test_by_pid[ma], test_by_pid[mb],
                                 np.random.default_rng(SEED))
        # CI excludes zero if both bounds have the same sign
        excl = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
        a_beats_b = hi < 0   # method A has significantly lower MAE than method B
        print(f"  {label}")
        print(f"    diff={pt:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  "
              f"excl_0={excl}  a_beats_b={a_beats_b}")
        diff_rows.append({
            "comparison": label, "method_a": ma, "method_b": mb,
            "diff": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "ci_excludes_zero": excl, "a_beats_b": a_beats_b,
        })
    pd.DataFrame(diff_rows).to_csv(OUT / "study2_final_diff_cis.csv", index=False)

    # ── TASK 4: Event-enriched vs unmatched control ─────────────────────────
    _sec("TASK 4: Event-enriched proxy vs unmatched control  [descriptive only]")
    print("  NOTE: This comparison is descriptive, not a matched causal contrast.")
    print("  Event-enriched labels are glucose-level proxies (no confirmed meal")
    print("  timestamps).  Control group is NOT distribution-matched on glucose,")
    print("  slope, z_t, or interval width.")

    final_pred_test = test_preds[final_head]
    base_pred_test  = test_preds["base"]
    ev_mask   = test_fc["event_enriched"].fillna(False)
    ctrl_mask = ~ev_mask

    evc_rows: list[dict] = []
    for group, mask, label in [
        ("event",   ev_mask,   "Event-enriched"),
        ("control", ctrl_mask, "Unmatched control"),
    ]:
        for mname, pred in [("base", base_pred_test), ("final_head", final_pred_test)]:
            r = evaluate_mae(test_fc, pred, np.random.default_rng(SEED), mask=mask)
            r["group"]  = group
            r["group_label"] = label
            r["method"] = mname
            evc_rows.append(r)

    ev_df = pd.DataFrame(evc_rows)
    # compute gain: base - head per group
    for group in ("event", "control"):
        g    = ev_df[ev_df["group"].eq(group)]
        base = g[g["method"].eq("base")]["mae"].values[0]
        head = g[g["method"].eq("final_head")]["mae"].values[0]
        n    = g.iloc[0]["n_anchors"]
        print(f"\n  {group.upper()} group (n={n:,}):")
        print(f"    Base MAE:       {base:.4f}")
        print(f"    Final head MAE: {head:.4f}")
        print(f"    Gain:           {base - head:+.4f} mg/dL")

    # Paired CI for gain: event vs control
    ev_m   = test_fc["event_enriched"].fillna(False)
    ctrl_m = ~ev_m
    diff_ev_gain_pt, diff_ev_gain_lo, diff_ev_gain_hi = _boot_diff(
        errs_by_pid_subset(test_fc, base_pred_test, ev_m),
        errs_by_pid_subset(test_fc, final_pred_test, ev_m),
        np.random.default_rng(SEED),
    )
    diff_ctrl_gain_pt, diff_ctrl_gain_lo, diff_ctrl_gain_hi = _boot_diff(
        errs_by_pid_subset(test_fc, base_pred_test, ctrl_m),
        errs_by_pid_subset(test_fc, final_pred_test, ctrl_m),
        np.random.default_rng(SEED),
    )
    print(f"\n  Gain CI (base - head) [event]:   "
          f"diff={diff_ev_gain_pt:+.4f}  "
          f"CI=[{diff_ev_gain_lo:+.4f}, {diff_ev_gain_hi:+.4f}]  "
          f"excl_0={diff_ev_gain_lo > 0}")
    print(f"  Gain CI (base - head) [control]: "
          f"diff={diff_ctrl_gain_pt:+.4f}  "
          f"CI=[{diff_ctrl_gain_lo:+.4f}, {diff_ctrl_gain_hi:+.4f}]  "
          f"excl_0={diff_ctrl_gain_lo > 0}")

    evc_detail = {
        "event_n_anchors":        int(ev_df[ev_df["group"].eq("event")].iloc[0]["n_anchors"]),
        "control_n_anchors":      int(ev_df[ev_df["group"].eq("control")].iloc[0]["n_anchors"]),
        "event_base_mae":         float(ev_df[(ev_df.group=="event") & (ev_df.method=="base")].mae.values[0]),
        "event_head_mae":         float(ev_df[(ev_df.group=="event") & (ev_df.method=="final_head")].mae.values[0]),
        "control_base_mae":       float(ev_df[(ev_df.group=="control") & (ev_df.method=="base")].mae.values[0]),
        "control_head_mae":       float(ev_df[(ev_df.group=="control") & (ev_df.method=="final_head")].mae.values[0]),
        "event_gain_ci":          [round(diff_ev_gain_lo,4), round(diff_ev_gain_hi,4)],
        "control_gain_ci":        [round(diff_ctrl_gain_lo,4), round(diff_ctrl_gain_hi,4)],
        "descriptive_disclaimer": (
            "Not a matched causal contrast. Event group defined by "
            "glucose-level proxy (trigger_up & glucose>=130 OR trigger_down & glucose<=85). "
            "Control group not distribution-matched."
        ),
    }
    (OUT / "study2_event_vs_control.json").write_text(json.dumps(evc_detail, indent=2))

    # ── TASK 5: Per-horizon and shape metrics ────────────────────────────────
    _sec("TASK 5: Per-horizon MAE and shape metrics (test)")

    horizon_rows: list[dict] = []
    print(f"\n  {'Method':<35}  0-30min  30-60min  Term-60min  CumResErr  DirAcc  PeakErr")
    print("  " + "-" * 90)

    for mname, pred in test_preds.items():
        ph = evaluate_per_horizon(test_fc, pred)
        h_0_30  = round(np.mean([ph[h] for h in range(1, 7)]), 4)
        h_30_60 = round(np.mean([ph[h] for h in range(7, 13)]), 4)
        h_term  = round(ph[12], 4)
        sm      = evaluate_shape_metrics(test_fc, pred)
        print(f"  {METHOD_LABELS[mname]:<35}  "
              f"{h_0_30:>7.4f}  {h_30_60:>8.4f}  {h_term:>10.4f}  "
              f"{sm['cumulative_residual_err']:>9.4f}  "
              f"{sm['direction_accuracy']:>6.4f}  "
              f"{sm['peak_err_mean']:>7.4f}")
        horizon_rows.append({
            "split": "test", "method": mname, "label": METHOD_LABELS[mname],
            "mae_0_30min": h_0_30, "mae_30_60min": h_30_60, "mae_terminal_60min": h_term,
            **{f"mae_h{h:02d}": round(ph[h], 4) for h in range(1, H + 1)},
            **sm,
        })

    pd.DataFrame(horizon_rows).to_csv(OUT / "study2_per_horizon.csv", index=False)

    # ── TASK 6: Feature-group ablation (Ridge, val MAE) ─────────────────────
    _sec("TASK 6: Feature-group ablation (Ridge, validation MAE)")

    ablation_rows: list[dict] = []
    print(f"\n  {'Ablation group':<30}  {'Features':>3}  {'val MAE':>8}  {'test MAE':>9}  "
          f"{'val CI':>22}")
    print("  " + "-" * 80)

    for gname, gfeats in ABLATION_GROUPS.items():
        # Some features may not be present in train_fc
        gf = [f for f in (gfeats or USABLE) if f in train_fc.columns]
        if not gf:
            print(f"  {gname:<30}  --- no features available ---")
            continue
        gf_nz = [f for f in gf if train_fc[f].std() > 0]
        if not gf_nz:
            print(f"  {gname:<30}  --- all features zero-variance ---")
            continue
        mdls, scl = train_ridge(train_fc, gf_nz, "both")
        vp  = pred_ridge(val_fc,  mdls, scl, gf_nz)
        tp  = pred_ridge(test_fc, mdls, scl, gf_nz)
        vr  = evaluate_mae(val_fc,  vp, np.random.default_rng(SEED))
        tr_ = evaluate_mae(test_fc, tp, np.random.default_rng(SEED))
        print(f"  {gname:<30}  {len(gf_nz):>3}  {vr['mae']:>8.4f}  "
              f"{tr_['mae']:>9.4f}  [{vr['ci_lo']:>+8.4f}, {vr['ci_hi']:>+8.4f}]")
        ablation_rows.append({
            "group": gname, "n_features": len(gf_nz), "features": gf_nz,
            "val_mae": vr["mae"], "val_ci_lo": vr["ci_lo"], "val_ci_hi": vr["ci_hi"],
            "test_mae": tr_["mae"], "test_ci_lo": tr_["ci_lo"], "test_ci_hi": tr_["ci_hi"],
        })

    pd.DataFrame(ablation_rows).to_csv(OUT / "study2_ablation.csv", index=False)

    # ── TASK 7 + 8: Save artifacts and manifest ─────────────────────────────
    _sec("Saving artifacts")

    test_row = ladder_df[(ladder_df.split=="test") & (ladder_df.method==final_head)].iloc[0]
    val_row  = ladder_df[(ladder_df.split=="validation") & (ladder_df.method==final_head)].iloc[0]

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        # Study 2 design invariants
        "base_ssm_cgm_frozen": True,
        "cache_used": "5-minute rolling forecast cache (anchor_stride_minutes=5)",
        "cache_path": str(CACHE),
        "tau_up_selected_on_validation": TAU_UP,
        "tau_down_selected_on_validation": TAU_DOWN,
        "final_head_selected_using": "validation MAE only",
        "test_used_for": "final reporting only",
        "no_future_labels_used": True,
        "no_future_covariates_used": True,
        # Head selection
        "final_head": final_head,
        "final_head_label": METHOD_LABELS[final_head],
        "val_mae_top2": [
            {"method": val_results[0]["method"], "val_mae": val_results[0]["val_mae"]},
            {"method": val_results[1]["method"], "val_mae": val_results[1]["val_mae"]},
        ],
        "selection_note": (
            "Best-val vs second-best-val CI did not exclude zero → preferred simpler model."
            if not excludes_zero else "Clear winner on validation."
        ),
        # Final head metrics
        "final_head_val_mae": float(val_row["mae"]),
        "final_head_val_ci":  [float(val_row["ci_lo"]), float(val_row["ci_hi"])],
        "final_head_test_mae": float(test_row["mae"]),
        "final_head_test_ci":  [float(test_row["ci_lo"]), float(test_row["ci_hi"])],
        # Features
        "causal_features_used": USABLE,
        "features_excluded_zero_variance": zero_var,
        "hr_features_available": False,
        "steps_features_note": "steps_since_start is a sequential anchor counter, not pedometer data; excluded",
        # Data counts
        "train_anchors": int(n_unique_anchors(train_fc)),
        "val_anchors":   int(n_unique_anchors(val_fc)),
        "test_anchors":  int(n_unique_anchors(test_fc)),
        # Artifacts
        "artifacts": {
            "ladder_csv":        "study2_final_ladder.csv",
            "diff_cis_csv":      "study2_final_diff_cis.csv",
            "per_horizon_csv":   "study2_per_horizon.csv",
            "ablation_csv":      "study2_ablation.csv",
            "event_vs_control":  "study2_event_vs_control.json",
            "manifest":          "study2_final_manifest.json",
        },
    }
    (OUT / "study2_final_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  All artifacts written to {OUT}/")

    # ── TASK 9: Final verdict ────────────────────────────────────────────────
    _sec("TASK 9: Final verdict")

    test_ladder = ladder_df[ladder_df.split == "test"].set_index("method")
    best_simple_mae = test_ladder.loc["linear_zt_sc", "mae"]
    final_head_mae  = test_ladder.loc[final_head, "mae"]
    base_mae        = test_ladder.loc["base", "mae"]

    # Best vs ridge-split and hgbt-comb
    hgbt_vs_ridge = next(
        (r for r in diff_rows if r["method_a"] == "hgbt_full_comb" and r["method_b"] == "ridge_split_dir"),
        None,
    )
    final_vs_base   = next((r for r in diff_rows if r["comparison"] == "Final head vs Base"), None)
    final_vs_decay  = next((r for r in diff_rows if r["comparison"] == "Final head vs Decaying shift"), None)
    best_vs_ridge   = next(
        (r for r in diff_rows if r["method_a"] == "hgbt_full_comb" and r["method_b"] == "linear_zt_sc"),
        None,
    )

    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Study 2 – Final correction head verdict (test split)           │
  ├─────────────────────────────────────────────────────────────────┤
  │  Base SSM-CGM MAE:                      {base_mae:.4f} mg/dL          │
  │  Best simple baseline (z_t+slope+curv): {best_simple_mae:.4f} mg/dL          │
  │  Selected final head ({final_head:20s}): {final_head_mae:.4f} mg/dL          │
  │  Improvement over best simple:          {best_simple_mae - final_head_mae:+.4f} mg/dL          │
  ├─────────────────────────────────────────────────────────────────┤""")

    # Interpret: diff = MAE(A) - MAE(B); a_beats_b means A has lower MAE than B
    if best_vs_ridge and best_vs_ridge["ci_excludes_zero"] and best_vs_ridge.get("a_beats_b"):
        print(f"  │  Rich head BEATS best simple baseline (CI excludes zero).       │")
    elif best_vs_ridge and best_vs_ridge["ci_excludes_zero"]:
        print(f"  │  Rich head WORSE than best simple (CI excludes zero).           │")
    else:
        print(f"  │  Rich head vs best simple: tied, CI does not exclude zero.      │")

    if hgbt_vs_ridge:
        if not hgbt_vs_ridge["ci_excludes_zero"]:
            print(f"  │  HGBT vs Ridge: STATISTICALLY TIED (CI includes zero).         │")
        elif hgbt_vs_ridge.get("a_beats_b"):
            print(f"  │  HGBT outperforms Ridge (CI excludes zero).                    │")
        else:
            print(f"  │  Ridge outperforms HGBT (CI excludes zero).                    │")

    if final_vs_base and final_vs_base["ci_excludes_zero"] and final_vs_base.get("a_beats_b"):
        print(f"  │  Final head beats frozen base (CI excludes zero).               │")
    elif final_vs_base and not final_vs_base["ci_excludes_zero"]:
        print(f"  │  Final head vs frozen base: CI includes zero.                   │")
    else:
        print(f"  │  Final head vs frozen base: base unexpectedly better.           │")

    print(f"""  ├─────────────────────────────────────────────────────────────────┤
  │  Event-vs-control result: DESCRIPTIVE ONLY.                     │
  │  The control group is NOT matched on glucose level, slope,      │
  │  z_t, or interval width.  Treat event-group gain as suggestive, │
  │  not as causal evidence.                                         │
  ├─────────────────────────────────────────────────────────────────┤
  │  RECOMMENDED MODEL: {final_head:<44s}│
  │  Rationale: lowest or tied-for-lowest validation MAE;           │
  │  all causal features are trailing (no future leakage);          │
  │  base SSM-CGM frozen throughout.                                │
  └─────────────────────────────────────────────────────────────────┘""")


if __name__ == "__main__":
    main()
