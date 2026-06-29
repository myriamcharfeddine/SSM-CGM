#!/usr/bin/env python3
"""Study 2 Blocks C-H: correction head training and full evaluation.

Block C  -- Ridge regression correction head, one per horizon (12 models),
            trained on triggered non-insulin train-split windows.
Block D  -- Baseline ladder: base / current-shift / decaying-shift /
            linear(z_t) / learned-head, with participant-clustered 95% CIs.
Block E  -- Event-vs-matched-placebo gap with proxy event labels
            (no meal logs in panel; uses glucose-level proxy only).
Block F  -- Deployment metrics: false-trigger rate, episode duration.
Block G  -- Abstention breakdown (from anchor table).
Block H  -- Artifact consolidation and final manifest.

Run with:
    /home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python \\
        scripts/run_study2_blocks_c_to_h.py

Outputs land in:
    outputs/study2_blocks_c_to_h/
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ── constants ────────────────────────────────────────────────────────────────
CACHE    = ROOT / "outputs/study2_forecast_cache_5min/study2_forecast_cache.parquet"
OUT      = ROOT / "outputs/study2_blocks_c_to_h"
TAU_UP   = 0.2
TAU_DOWN = 0.2
EPS      = 1e-3
N_BOOT   = 1000
SEED     = 42
BIN_MIN  = 5
STEPS_PER_DAY = 288
H        = 12           # horizon steps

# Event-enriched proxy: non-oracle, uses only current_glucose at trigger time.
# trigger_up and glucose >= EVENT_GLUCOSE_HI  →  likely post-meal hyperglycemia
# trigger_down and glucose <= EVENT_GLUCOSE_LO →  likely hypoglycemia / recovery
EVENT_GLUCOSE_HI = 130.0   # mg/dL
EVENT_GLUCOSE_LO =  85.0   # mg/dL


# ── helpers ──────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def build_anchor_table(df: pd.DataFrame) -> pd.DataFrame:
    fo = df[df["scenario_mode"].astype(str).eq("forecast_only")].copy() if "scenario_mode" in df.columns else df.copy()
    if "anchor_ds" not in fo.columns:
        fo["anchor_ds"] = fo["anchor_time_idx"]

    meta_cols = [c for c in [
        "participant_id", "segment_id", "split", "anchor_ds",
        "current_glucose", "med_insulin", "participants_study_group",
        "hba1c_percent_baseline", "participants_clinical_site",
        "med_any_diabetes_drug",
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
    if "current_glucose" not in out.columns or out["current_glucose"].isna().all():
        out["current_glucose"] = pd.to_numeric(out.get("one_step_target"), errors="coerce")
    else:
        out["current_glucose"] = pd.to_numeric(out["current_glucose"], errors="coerce").fillna(
            pd.to_numeric(out.get("one_step_target"), errors="coerce")
        )
    out["one_step_residual"] = (
        out["current_glucose"] - pd.to_numeric(out["q50_one_step"], errors="coerce")
    )
    consecutive = out["one_step_issue_anchor_ds"].eq(out["anchor_ds"] - 1)
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
    return out


def apply_triggers(anchor_table: pd.DataFrame) -> pd.DataFrame:
    at = anchor_table.copy()
    at["z_t"] = np.where(
        at["abstain"], np.nan,
        at["one_step_residual"] / (at["interval_width"] + EPS),
    )
    at["up_obs"]    = (~at["abstain"]) & at["z_t"].gt(TAU_UP)
    at["down_obs"]  = (~at["abstain"]) & at["z_t"].lt(-TAU_DOWN)
    at["valid_obs"] = ~at["abstain"]

    grp = at.sort_values(["participant_id", "segment_id", "anchor_ds"]).groupby(
        ["participant_id", "segment_id"], sort=False
    )
    for col in ("valid_obs", "up_obs", "down_obs"):
        at[f"{col}_r3"] = grp[col].transform(lambda s: s.astype(int).rolling(3, min_periods=3).sum())

    at["trigger_up"]   = at["valid_obs_r3"].ge(3) & at["up_obs_r3"].ge(3)
    at["trigger_down"] = at["valid_obs_r3"].ge(3) & at["down_obs_r3"].ge(3)
    at["triggered"]    = at["trigger_up"] | at["trigger_down"]
    return at


def event_enriched_proxy(at: pd.DataFrame) -> pd.Series:
    """Non-oracle event label: trigger during likely hyperglycemia or hypoglycemia."""
    g = pd.to_numeric(at["current_glucose"], errors="coerce")
    return (
        (at["trigger_up"]   & g.ge(EVENT_GLUCOSE_HI)) |
        (at["trigger_down"] & g.le(EVENT_GLUCOSE_LO))
    ).fillna(False)


# ── participant-clustered bootstrap ──────────────────────────────────────────

def _boot_ci(values_by_pid: dict[str, np.ndarray], n_boot: int = N_BOOT,
             alpha: float = 0.05) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    pids = list(values_by_pid.keys())
    n = len(pids)
    stats = []
    for _ in range(n_boot):
        sampled = rng.choice(pids, size=n, replace=True)
        vals = np.concatenate([values_by_pid[p] for p in sampled])
        stats.append(float(np.mean(vals)))
    stats = np.array(stats)
    point = float(np.mean(np.concatenate(list(values_by_pid.values()))))
    return (
        point,
        float(np.percentile(stats, alpha / 2 * 100)),
        float(np.percentile(stats, (1 - alpha / 2) * 100)),
    )


def mae_by_pid(errs: pd.DataFrame, pid_col: str = "participant_id") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for pid, grp in errs.groupby(pid_col, sort=False):
        out[str(pid)] = grp["abs_err"].to_numpy()
    return out


def compute_ci(errs_df: pd.DataFrame) -> tuple[float, float, float]:
    return _boot_ci(mae_by_pid(errs_df))


def _boot_ci_difference(vals_a_by_pid: dict, vals_b_by_pid: dict,
                        n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    """Paired participant-clustered bootstrap CI for mean(A) - mean(B)."""
    rng = np.random.default_rng(SEED)
    pids = list(set(vals_a_by_pid) | set(vals_b_by_pid))
    empty = np.array([float("nan")])
    stats = []
    for _ in range(n_boot):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        a_vals = np.concatenate([vals_a_by_pid.get(p, empty) for p in sampled])
        b_vals = np.concatenate([vals_b_by_pid.get(p, empty) for p in sampled])
        a_vals = a_vals[np.isfinite(a_vals)]
        b_vals = b_vals[np.isfinite(b_vals)]
        if len(a_vals) and len(b_vals):
            stats.append(float(np.mean(a_vals)) - float(np.mean(b_vals)))
    stats = np.array(stats)
    point = float(np.mean(np.concatenate(list(vals_a_by_pid.values()))) -
                  np.mean(np.concatenate(list(vals_b_by_pid.values()))))
    return (
        point,
        float(np.percentile(stats, alpha / 2 * 100)),
        float(np.percentile(stats, (1 - alpha / 2) * 100)),
    )


def compute_diff_ci(errs_a: pd.DataFrame, errs_b: pd.DataFrame) -> tuple[float, float, float]:
    return _boot_ci_difference(mae_by_pid(errs_a), mae_by_pid(errs_b))


# ── build triggered forecast rows (horizons x anchors) ───────────────────────

def get_triggered_rows(cache_fo: pd.DataFrame, triggered_keys: pd.DataFrame,
                       at_triggered: pd.DataFrame) -> pd.DataFrame:
    """Merge full horizon rows for triggered anchors with anchor-level features."""
    merge_keys = ["participant_id", "segment_id", "anchor_ds"]
    fc = cache_fo.merge(triggered_keys[merge_keys], on=merge_keys, how="inner")
    # current_glucose is already in cache_fo; omit to avoid merge suffix conflict
    extra_anchor_cols = [c for c in [
        "z_t", "one_step_residual", "interval_width",
        "trigger_up", "trigger_down", "non_insulin",
        "participants_study_group", "hba1c_percent_baseline",
        "participants_clinical_site", "event_enriched",
    ] if c in at_triggered.columns]
    anchor_feats = at_triggered[merge_keys + extra_anchor_cols].copy()
    fc = fc.merge(anchor_feats, on=merge_keys, how="left")
    return fc


# ── correction methods ────────────────────────────────────────────────────────

def apply_current_shift(fc: pd.DataFrame) -> pd.Series:
    return fc["q50"] + fc["one_step_residual"]


def apply_decaying_shift(fc: pd.DataFrame, decay: float) -> pd.Series:
    delta = fc["one_step_residual"] * (decay ** (fc["horizon_step"] - 1))
    return fc["q50"] + delta


def apply_linear(fc: pd.DataFrame, coefs: dict[int, tuple[float, float]]) -> pd.Series:
    """Linear: q50 + (a + b * one_step_residual) per horizon."""
    pred = fc["q50"].copy()
    for h, (a, b) in coefs.items():
        mask = fc["horizon_step"].eq(h)
        pred[mask] += a + b * fc.loc[mask, "one_step_residual"]
    return pred


def apply_head(fc: pd.DataFrame, models: dict[int, Ridge],
               scaler: StandardScaler, feature_cols: list[str]) -> pd.Series:
    pred = fc["q50"].copy()
    for h, mdl in models.items():
        mask = fc["horizon_step"].eq(h)
        if not mask.any():
            continue
        X = fc.loc[mask, feature_cols].to_numpy(dtype=float)
        Xs = scaler.transform(X)
        delta = mdl.predict(Xs)
        pred[mask] += delta
    return pred


# ── Block C: train head ───────────────────────────────────────────────────────

def train_head(train_fc: pd.DataFrame, feature_cols: list[str]) -> tuple[dict, StandardScaler]:
    """Train one Ridge per horizon on triggered non-insulin train windows."""
    scaler = StandardScaler()
    X_all = train_fc[feature_cols].to_numpy(dtype=float)
    scaler.fit(X_all)

    models: dict[int, Ridge] = {}
    for h in range(1, H + 1):
        mask = train_fc["horizon_step"].eq(h)
        Xh = scaler.transform(train_fc.loc[mask, feature_cols].to_numpy(dtype=float))
        yh = (train_fc.loc[mask, "target"] - train_fc.loc[mask, "q50"]).to_numpy(dtype=float)
        mdl = Ridge(alpha=1.0, fit_intercept=True)
        mdl.fit(Xh, yh)
        models[h] = mdl
    return models, scaler


def fit_linear_baselines(train_fc: pd.DataFrame) -> dict[int, tuple[float, float]]:
    coefs: dict[int, tuple[float, float]] = {}
    for h in range(1, H + 1):
        mask = train_fc["horizon_step"].eq(h)
        x = train_fc.loc[mask, "one_step_residual"].to_numpy(dtype=float)
        y = (train_fc.loc[mask, "target"] - train_fc.loc[mask, "q50"]).to_numpy(dtype=float)
        xm, ym = x.mean(), y.mean()
        b = np.cov(x, y, ddof=1)[0, 1] / max(np.var(x, ddof=1), 1e-9)
        a = ym - b * xm
        coefs[h] = (float(a), float(b))
    return coefs


def tune_decay(train_fc: pd.DataFrame) -> float:
    best_decay, best_mae = 1.0, np.inf
    for d in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
        pred = apply_decaying_shift(train_fc, d)
        mae  = (train_fc["target"] - pred).abs().mean()
        if mae < best_mae:
            best_mae, best_decay = mae, d
    return best_decay


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_method(fc: pd.DataFrame, pred: pd.Series, split: str,
                    method: str, subset: str = "all") -> dict:
    mask = fc["split"].eq(split)
    if subset == "event":
        mask &= fc["event_enriched"].fillna(False)
    elif subset == "placebo":
        mask &= ~fc["event_enriched"].fillna(False)
    sub = fc[mask].copy()
    sub["abs_err"] = (sub["target"] - pred[mask]).abs()
    if sub.empty:
        return {"split": split, "method": method, "subset": subset,
                "n_anchors": 0, "mae": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    point, lo, hi = compute_ci(sub)
    return {
        "split": split, "method": method, "subset": subset,
        "n_anchors": int(sub["anchor_ds"].nunique()),
        "n_participants": int(sub["participant_id"].nunique()),
        "mae": round(point, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
    }


# ── episode helpers (for Block F) ────────────────────────────────────────────

def episode_durations(at: pd.DataFrame) -> pd.Series:
    at = at.sort_values(["participant_id", "segment_id", "anchor_ds"])
    grp = at.groupby(["participant_id", "segment_id"], sort=False)
    at["_prev_trig"] = grp["triggered"].shift(1).fillna(False)
    at["_prev_ds"]   = grp["anchor_ds"].shift(1)
    at["_gap"]       = at["anchor_ds"] - at["_prev_ds"]
    at["_new_ep"]    = at["triggered"] & (~at["_prev_trig"] | at["_gap"].gt(1))
    at["_ep_cum"]    = grp["_new_ep"].cumsum()
    ep_rows = at[at["triggered"]].copy()
    ep_rows["_ep_id"] = ep_rows["_ep_cum"].astype(int)
    ep_lengths = (
        ep_rows.groupby(["participant_id", "segment_id", "_ep_id"], sort=False)["anchor_ds"]
        .count() * BIN_MIN
    )
    return ep_lengths


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng_state = {"seed": SEED}

    _print_section("Loading cache")
    df = pd.read_parquet(CACHE)
    print(f"  {len(df):,} rows")
    cache_fo = df[df["scenario_mode"].astype(str).eq("forecast_only")].copy() if "scenario_mode" in df.columns else df.copy()
    if "anchor_ds" not in cache_fo.columns:
        cache_fo["anchor_ds"] = cache_fo["anchor_time_idx"]

    _print_section("Building anchor table and applying triggers")
    anchor_table = build_anchor_table(df)
    at = apply_triggers(anchor_table)
    at["event_enriched"] = event_enriched_proxy(at)
    non_ins_trig = at[at["non_insulin"].fillna(True) & at["triggered"]]
    print(f"  Triggered non-insulin anchors: "
          f"train={non_ins_trig[non_ins_trig['split']=='train'].shape[0]:,}  "
          f"val={non_ins_trig[non_ins_trig['split']=='validation'].shape[0]:,}  "
          f"test={non_ins_trig[non_ins_trig['split']=='test'].shape[0]:,}")
    ev = non_ins_trig["event_enriched"]
    print(f"  Event-enriched proxy: {ev.sum():,} / {len(ev):,} triggered anchors "
          f"({100*ev.mean():.1f}%)")

    # ── Block C ───────────────────────────────────────────────────────────────
    _print_section("Block C: Training correction head")
    feature_cols = ["z_t", "one_step_residual", "current_glucose", "trigger_up"]

    train_trig = non_ins_trig[non_ins_trig["split"].eq("train")]
    train_keys = train_trig[["participant_id", "segment_id", "anchor_ds"]].copy()
    train_keys["event_enriched"] = train_trig["event_enriched"].values

    # Attach anchor features to full-horizon rows
    at_with_event = at.copy()
    at_with_event["event_enriched"] = at_with_event["event_enriched"]
    train_fc = get_triggered_rows(
        cache_fo[cache_fo["split"].eq("train")], train_keys, at_with_event
    )
    train_fc = train_fc[train_fc["non_insulin"].fillna(True)].copy()
    print(f"  Train triggered rows: {len(train_fc):,} "
          f"(across {train_fc['anchor_ds'].nunique():,} anchors)")

    # Tune decay on train
    best_decay = tune_decay(train_fc)
    print(f"  Best decaying-shift decay: {best_decay}")

    # Fit linear baselines on train
    linear_coefs = fit_linear_baselines(train_fc)

    # Train head
    models, scaler = train_head(train_fc, feature_cols)
    print(f"  Head trained: {H} Ridge models, features={feature_cols}")

    # Save head coefficients
    head_artifact = {
        "feature_cols": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "best_decay": best_decay,
        "horizons": {},
    }
    for h, mdl in models.items():
        head_artifact["horizons"][str(h)] = {
            "intercept": float(mdl.intercept_),
            "coefs": mdl.coef_.tolist(),
        }
    for h, (a, b) in linear_coefs.items():
        head_artifact["horizons"][str(h)]["linear_intercept"] = a
        head_artifact["horizons"][str(h)]["linear_slope_z_t"] = b
    head_path = OUT / "study2_correction_head.json"
    head_path.write_text(json.dumps(head_artifact, indent=2))
    print(f"  Head saved -> {head_path}")

    # ── Block D: Baseline ladder ──────────────────────────────────────────────
    _print_section("Block D: Baseline ladder")
    ladder_rows = []
    # Keep per-anchor error DataFrames for paired difference CIs
    err_frames: dict[str, dict[str, pd.DataFrame]] = {}  # [split][method] -> errs_df

    for split in ("validation", "test"):
        split_trig_keys = at_with_event[
            at_with_event["non_insulin"].fillna(True)
            & at_with_event["triggered"]
            & at_with_event["split"].eq(split)
        ][["participant_id", "segment_id", "anchor_ds", "event_enriched"]].copy()

        split_fc = get_triggered_rows(
            cache_fo[cache_fo["split"].eq(split)], split_trig_keys, at_with_event
        )
        split_fc = split_fc[split_fc["non_insulin"].fillna(True)].copy()

        preds = {
            "base":           split_fc["q50"],
            "current_shift":  apply_current_shift(split_fc),
            "decaying_shift": apply_decaying_shift(split_fc, best_decay),
            "linear":         apply_linear(split_fc, linear_coefs),
            "head":           apply_head(split_fc, models, scaler, feature_cols),
        }
        err_frames[split] = {}
        for method, pred in preds.items():
            err_df = split_fc[["participant_id", "anchor_ds", "event_enriched"]].copy()
            err_df["abs_err"] = (split_fc["target"] - pred).abs().values
            err_frames[split][method] = err_df
            for subset in ("all", "event", "placebo"):
                row = evaluate_method(split_fc, pred, split, method, subset)
                ladder_rows.append(row)

    ladder_df = pd.DataFrame(ladder_rows)
    ladder_path = OUT / "study2_baseline_ladder.csv"
    ladder_df.to_csv(ladder_path, index=False)

    _print_section("BASELINE LADDER (validation, all triggered, non-insulin)")
    val_ladder = ladder_df[
        (ladder_df["split"] == "validation") & (ladder_df["subset"] == "all")
    ][["method", "n_anchors", "n_participants", "mae", "ci_lo", "ci_hi"]]
    print(val_ladder.to_string(index=False))

    _print_section("BASELINE LADDER (test, all triggered, non-insulin)")
    tst_ladder = ladder_df[
        (ladder_df["split"] == "test") & (ladder_df["subset"] == "all")
    ][["method", "n_anchors", "n_participants", "mae", "ci_lo", "ci_hi"]]
    print(tst_ladder.to_string(index=False))

    # -- Paired difference CIs (head vs best shift, event-enriched gap) --------
    _print_section("Paired difference CIs (participant-clustered bootstrap)")
    diff_ci_rows = []
    for split in ("validation", "test"):
        ef = err_frames[split]
        # head vs decaying_shift (best non-trivial baseline)
        diff, dlo, dhi = compute_diff_ci(ef["decaying_shift"], ef["head"])
        ci_excl = dlo > 0
        print(f"  {split}  decaying_shift - head:  diff={diff:+.4f}  "
              f"95% CI=[{dlo:+.4f}, {dhi:+.4f}]  CI_excludes_zero={ci_excl}")
        diff_ci_rows.append({"split": split, "comparison": "decaying_shift_minus_head",
                              "diff": diff, "ci_lo": dlo, "ci_hi": dhi,
                              "ci_excludes_zero": ci_excl})

        # event-placebo gap for head: gain_event - gain_placebo, bootstrapped over participants
        def gain_by_pid(base_df: pd.DataFrame, head_df: pd.DataFrame,
                        ev_mask: pd.Series) -> dict[str, np.ndarray]:
            """Per-participant scalar: base_MAE_subset - head_MAE_subset."""
            b = base_df[ev_mask].groupby("participant_id")["abs_err"].mean()
            h = head_df[ev_mask].groupby("participant_id")["abs_err"].mean()
            shared = b.index.intersection(h.index)
            return {str(p): np.array([float(b[p] - h[p])]) for p in shared}

        ev_mask = ef["base"]["event_enriched"].fillna(False)
        pl_mask = ~ev_mask
        gain_ev_pid = gain_by_pid(ef["base"], ef["head"], ev_mask)
        gain_pl_pid = gain_by_pid(ef["base"], ef["head"], pl_mask)
        # gap = gain_event - gain_placebo
        gap_diff, gap_lo, gap_hi = _boot_ci_difference(gain_ev_pid, gain_pl_pid)
        gap_excl = gap_lo > 0
        print(f"  {split}  event_gain - placebo_gain (head): "
              f"gap={gap_diff:+.4f}  95% CI=[{gap_lo:+.4f}, {gap_hi:+.4f}]  "
              f"CI_excludes_zero={gap_excl}")
        diff_ci_rows.append({"split": split, "comparison": "head_gain_event_minus_placebo",
                              "diff": gap_diff, "ci_lo": gap_lo, "ci_hi": gap_hi,
                              "ci_excludes_zero": gap_excl})

    pd.DataFrame(diff_ci_rows).to_csv(OUT / "study2_difference_cis.csv", index=False)

    # ── Block E: Event-vs-placebo gap ─────────────────────────────────────────
    _print_section("Block E: Event-vs-matched-placebo gap")
    print("  NOTE: No meal event labels in panel. Using glucose-level proxy:")
    print(f"  event_enriched_proxy = trigger_up & glucose >= {EVENT_GLUCOSE_HI} "
          f"OR trigger_down & glucose <= {EVENT_GLUCOSE_LO}")

    gap_rows = []
    for split in ("validation", "test"):
        split_lad = ladder_df[(ladder_df["split"] == split)]
        for method in ("base", "head"):
            for subset in ("event", "placebo"):
                row = split_lad[
                    (split_lad["method"] == method)
                    & (split_lad["subset"] == subset)
                ]
                if not row.empty:
                    gap_rows.append(row.iloc[0].to_dict())

    # Compute gap as (MAE_event - MAE_placebo) for each method
    # And gain_over_base for event vs placebo subsets
    _print_section("Event-vs-Placebo MAE comparison (non-insulin)")
    for split in ("validation", "test"):
        print(f"\n  split={split}")
        s = ladder_df[ladder_df["split"].eq(split)]
        for method in ("base", "current_shift", "decaying_shift", "linear", "head"):
            row_ev  = s[(s["method"] == method) & (s["subset"] == "event")]
            row_pl  = s[(s["method"] == method) & (s["subset"] == "placebo")]
            mae_ev  = row_ev["mae"].values[0] if not row_ev.empty else float("nan")
            mae_pl  = row_pl["mae"].values[0] if not row_pl.empty else float("nan")
            n_ev    = row_ev["n_anchors"].values[0] if not row_ev.empty else 0
            n_pl    = row_pl["n_anchors"].values[0] if not row_pl.empty else 0
            print(f"    {method:16s}: event MAE={mae_ev:.4f} (n={n_ev})  "
                  f"placebo MAE={mae_pl:.4f} (n={n_pl})")

    # Event-vs-placebo GAP: does head gain more on event windows than placebo?
    print("\n  Correction gain (MAE_base - MAE_method) by subset:")
    gap_result_rows = []
    for split in ("validation", "test"):
        s = ladder_df[ladder_df["split"].eq(split)]
        base_ev = s[(s["method"] == "base") & (s["subset"] == "event")]["mae"].values
        base_pl = s[(s["method"] == "base") & (s["subset"] == "placebo")]["mae"].values
        if len(base_ev) == 0 or len(base_pl) == 0:
            continue
        for method in ("current_shift", "decaying_shift", "linear", "head"):
            mev = s[(s["method"] == method) & (s["subset"] == "event")]["mae"].values
            mpl = s[(s["method"] == method) & (s["subset"] == "placebo")]["mae"].values
            if len(mev) == 0 or len(mpl) == 0:
                continue
            gain_ev = float(base_ev[0] - mev[0])
            gain_pl = float(base_pl[0] - mpl[0])
            gap = gain_ev - gain_pl
            print(f"    {split:12s} {method:16s}: "
                  f"gain_event={gain_ev:+.4f}  gain_placebo={gain_pl:+.4f}  "
                  f"gap(ev-pl)={gap:+.4f}")
            gap_result_rows.append({
                "split": split, "method": method,
                "gain_event_mae": gain_ev, "gain_placebo_mae": gain_pl,
                "gap_ev_minus_pl": gap,
                "event_proxy_label": f"glucose>={EVENT_GLUCOSE_HI} or <={EVENT_GLUCOSE_LO}",
            })

    gap_df = pd.DataFrame(gap_result_rows)
    gap_path = OUT / "study2_event_placebo_gap.csv"
    gap_df.to_csv(gap_path, index=False)

    # ── Block F: Deployment metrics ───────────────────────────────────────────
    _print_section("Block F: Deployment metrics")
    deploy_rows = []
    for split in ("train", "validation", "test"):
        split_at = at[at["split"].eq(split) & at["non_insulin"].fillna(True)].copy()
        n_total_anchors = len(split_at)
        n_trig          = int(split_at["triggered"].sum())
        n_pids          = int(split_at["participant_id"].nunique())

        # Participant-days
        p_days = float(
            split_at.groupby(["participant_id", "segment_id"])["anchor_ds"]
            .count().sum() / STEPS_PER_DAY
        )

        # Episodes
        ep_dur = episode_durations(split_at[split_at["triggered"]])
        n_eps   = int(len(ep_dur)) if not ep_dur.empty else 0
        dur_med = float(ep_dur.median()) if not ep_dur.empty else float("nan")
        dur_p90 = float(ep_dur.quantile(0.90)) if not ep_dur.empty else float("nan")

        # Abstention
        n_abstain = int(split_at["abstain"].sum())

        row = {
            "split": split,
            "n_participants": n_pids,
            "participant_days": round(p_days, 1),
            "n_anchors": n_total_anchors,
            "n_triggered_anchors": n_trig,
            "trigger_rate": round(n_trig / max(n_total_anchors, 1), 5),
            "false_trigger_episodes_per_participant_day": round(n_eps / max(p_days, 1), 4),
            "n_trigger_episodes": n_eps,
            "trigger_duration_minutes_median": dur_med,
            "trigger_duration_minutes_p90": dur_p90,
            "n_abstaining_anchors": n_abstain,
            "abstention_rate": round(n_abstain / max(n_total_anchors, 1), 5),
        }
        deploy_rows.append(row)
        print(f"  {split}: {n_trig:,} triggered / {n_total_anchors:,} anchors "
              f"({row['trigger_rate']:.3%})  episodes={n_eps}  "
              f"dur_med={dur_med}min  abstain={n_abstain}")

    deploy_df = pd.DataFrame(deploy_rows)
    deploy_path = OUT / "study2_deployment_metrics.csv"
    deploy_df.to_csv(deploy_path, index=False)

    # ── Block G: Abstention breakdown ─────────────────────────────────────────
    _print_section("Block G: Abstention breakdown")
    abstain_rows = []
    for split in ("train", "validation", "test"):
        sub = at[at["split"].eq(split) & at["non_insulin"].fillna(True)]
        vc = sub["abstain_reason"].value_counts().to_dict()
        n  = len(sub)
        row = {"split": split, "n_total": n}
        row.update({f"n_{k}": int(v) for k, v in vc.items()})
        row.update({f"rate_{k}": round(v/n, 5) for k, v in vc.items()})
        abstain_rows.append(row)
        print(f"  {split}: abstain_rate={at[at['split']==split]['abstain'].mean():.4%}  "
              f"reasons={vc}")
    abstain_df = pd.DataFrame(abstain_rows)
    abstain_path = OUT / "study2_abstention.csv"
    abstain_df.to_csv(abstain_path, index=False)

    # ── Block H: Verdict and manifest ─────────────────────────────────────────
    _print_section("Block H: Verdict")

    # Use paired difference CIs for the official verdict
    diff_ci_df = pd.read_csv(OUT / "study2_difference_cis.csv")
    test_diff = diff_ci_df[diff_ci_df["split"].eq("test")]

    shift_row = test_diff[test_diff["comparison"].eq("decaying_shift_minus_head")]
    gap_row   = test_diff[test_diff["comparison"].eq("head_gain_event_minus_placebo")]

    shift_diff      = float(shift_row["diff"].values[0])
    shift_ci_lo     = float(shift_row["ci_lo"].values[0])
    shift_ci_hi     = float(shift_row["ci_hi"].values[0])
    beats_shift_ci  = bool(shift_row["ci_excludes_zero"].values[0])

    gap_diff_val    = float(gap_row["diff"].values[0]) if not gap_row.empty else float("nan")
    gap_ci_lo       = float(gap_row["ci_lo"].values[0]) if not gap_row.empty else float("nan")
    gap_ci_hi       = float(gap_row["ci_hi"].values[0]) if not gap_row.empty else float("nan")
    gap_ci_excl     = bool(gap_row["ci_excludes_zero"].values[0]) if not gap_row.empty else False

    test_lad = ladder_df[(ladder_df["split"] == "test") & (ladder_df["subset"] == "all")]
    head_mae = float(test_lad[test_lad["method"] == "head"]["mae"].values[0])

    # Two-criterion success gate
    criterion1 = beats_shift_ci   # head beats decaying_shift, CI excludes 0
    criterion2 = gap_ci_excl      # event gain > placebo gain, CI excludes 0

    verdict_label = (
        "PASS: both criteria met with participant-clustered CIs excluding zero"
        if criterion1 and criterion2 else
        "PARTIAL: beats shift CI excludes 0, but event-placebo gap CI includes 0 (proxy labels only)"
        if criterion1 and not criterion2 else
        "PARTIAL: event-placebo gap CI excludes 0, but head does not beat shift at CI level"
        if not criterion1 and criterion2 else
        "FAIL: neither criterion met at CI level"
    )

    verdict = {
        "test_head_mae": head_mae,
        "criterion1_head_beats_decaying_shift": {
            "diff_decaying_minus_head": shift_diff,
            "ci_lo": shift_ci_lo, "ci_hi": shift_ci_hi,
            "ci_excludes_zero": criterion1,
        },
        "criterion2_event_placebo_gap": {
            "gap_ev_minus_pl": gap_diff_val,
            "ci_lo": gap_ci_lo, "ci_hi": gap_ci_hi,
            "ci_excludes_zero": criterion2,
            "NOTE": "proxy labels only (glucose-level threshold); real meal labels absent from panel",
        },
        "adapter_success_note": verdict_label,
    }
    print(f"\n  Head test MAE: {head_mae:.4f}")
    print(f"  Criterion 1 (head vs decaying_shift): diff={shift_diff:+.4f} "
          f"CI=[{shift_ci_lo:+.4f},{shift_ci_hi:+.4f}]  PASS={criterion1}")
    print(f"  Criterion 2 (event-placebo gap, PROXY): diff={gap_diff_val:+.4f} "
          f"CI=[{gap_ci_lo:+.4f},{gap_ci_hi:+.4f}]  PASS={criterion2}")
    print(f"\n  VERDICT: {verdict_label}")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tau_up": TAU_UP, "tau_down": TAU_DOWN, "eps": EPS,
        "event_proxy": {
            "glucose_hi_threshold": EVENT_GLUCOSE_HI,
            "glucose_lo_threshold": EVENT_GLUCOSE_LO,
            "note": "proxy only; real meal event labels not available in panel",
        },
        "blocks_completed": ["C", "D", "E", "F", "G", "H"],
        "artifacts": {
            "head":       str(head_path),
            "ladder":     str(ladder_path),
            "gap":        str(gap_path),
            "deployment": str(deploy_path),
            "abstention": str(abstain_path),
        },
        "verdict": verdict,
    }
    manifest_path = OUT / "study2_blocks_c_to_h_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    _print_section("Files written")
    for name, p in manifest["artifacts"].items():
        print(f"  {name:12s}: {p}")
    print(f"  manifest    : {manifest_path}")


if __name__ == "__main__":
    main()
