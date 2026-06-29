#!/usr/bin/env python3
"""Pre-Block I verification checks for Study 2.

Check 1 — Extended paired difference CIs:
    base-head, base-linear, linear-head,
    decaying_shift-linear, decaying_shift-head
    (participant-clustered bootstrap, N=1000)

Check 2 — Placebo audit + matched-placebo implementation:
    Audit balance on: current_glucose, |z_t|, interval_width, glucose_slope
    Note time-of-day availability.
    Implement 1:1 caliper nearest-neighbor matching within participant.
    Rename generic/unmatched set to "generic_placebo".
    Compute matched-placebo gap CI; compare to generic.

Check 3 — S2-F4 episode selection (test split):
    Select 3-4 diverse triggered episodes (event+placebo, trigger_up/down).
    Save participant_id, segment_id, anchor_ds range, trigger direction,
    event_enriched status, glucose trace, base vs head prediction.

Run with:
    /home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python \\
        scripts/run_study2_preblock_i_checks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
CACHE_PATH = ROOT / "outputs/study2_forecast_cache_5min/study2_forecast_cache.parquet"
HEAD_PATH  = ROOT / "outputs/study2_blocks_c_to_h/study2_correction_head.json"
OUT_C      = ROOT / "outputs/study2_blocks_c_to_h"
OUT_PRE    = ROOT / "outputs/study2_preblock_i"

# ── constants ─────────────────────────────────────────────────────────────────
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
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ── rebuild anchor table + triggers (verbatim from blocks_c_to_h) ─────────────

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
    consecutive  = out["one_step_issue_anchor_ds"].eq(out["anchor_ds"] - 1)
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
    g = pd.to_numeric(at["current_glucose"], errors="coerce")
    return (
        (at["trigger_up"]   & g.ge(EVENT_GLUCOSE_HI)) |
        (at["trigger_down"] & g.le(EVENT_GLUCOSE_LO))
    ).fillna(False)


def add_glucose_slope(at: pd.DataFrame) -> pd.DataFrame:
    """5-minute glucose slope: current_glucose_t - current_glucose_{t-1}."""
    at = at.sort_values(["participant_id", "segment_id", "anchor_ds"])
    at["glucose_slope"] = at.groupby(
        ["participant_id", "segment_id"], sort=False
    )["current_glucose"].diff()
    return at


# ── load head from JSON (no retraining) ──────────────────────────────────────

def load_head(path: Path, feature_cols: list[str]) -> tuple[dict[int, Ridge], StandardScaler, float, dict]:
    data = json.loads(path.read_text())
    assert data["feature_cols"] == feature_cols, \
        f"Feature mismatch: expected {feature_cols}, got {data['feature_cols']}"
    scaler = StandardScaler()
    scaler.mean_           = np.array(data["scaler_mean"])
    scaler.scale_          = np.array(data["scaler_scale"])
    scaler.var_            = scaler.scale_ ** 2
    scaler.n_features_in_  = len(feature_cols)
    scaler.n_samples_seen_ = 99999  # dummy required by sklearn internals
    models: dict[int, Ridge] = {}
    linear_coefs: dict[int, tuple[float, float]] = {}
    for h_str, h_data in data["horizons"].items():
        h = int(h_str)
        mdl = Ridge(alpha=1.0, fit_intercept=True)
        mdl.coef_           = np.array(h_data["coefs"])
        mdl.intercept_      = float(h_data["intercept"])
        mdl.n_features_in_  = len(feature_cols)
        models[h] = mdl
        if "linear_intercept" in h_data:
            linear_coefs[h] = (h_data["linear_intercept"], h_data["linear_slope_z_t"])
    return models, scaler, float(data["best_decay"]), linear_coefs


# ── correction methods ────────────────────────────────────────────────────────

def apply_current_shift(fc: pd.DataFrame) -> pd.Series:
    return fc["q50"] + fc["one_step_residual"]


def apply_decaying_shift(fc: pd.DataFrame, decay: float) -> pd.Series:
    return fc["q50"] + fc["one_step_residual"] * (decay ** (fc["horizon_step"] - 1))


def apply_linear(fc: pd.DataFrame, coefs: dict[int, tuple[float, float]]) -> pd.Series:
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
        X  = fc.loc[mask, feature_cols].to_numpy(dtype=float)
        Xs = scaler.transform(X)
        pred[mask] += mdl.predict(Xs)
    return pred


# ── triggered forecast rows ───────────────────────────────────────────────────

def get_triggered_rows(cache_fo: pd.DataFrame, trig_keys: pd.DataFrame,
                       at_full: pd.DataFrame) -> pd.DataFrame:
    mk = ["participant_id", "segment_id", "anchor_ds"]
    fc = cache_fo.merge(trig_keys[mk], on=mk, how="inner")
    extra = [c for c in [
        "z_t", "one_step_residual", "interval_width",
        "trigger_up", "trigger_down", "non_insulin", "event_enriched",
    ] if c in at_full.columns]
    fc = fc.merge(at_full[mk + extra], on=mk, how="left")
    return fc


# ── bootstrap utilities ───────────────────────────────────────────────────────

def _boot_diff_ci(vals_a_by_pid: dict, vals_b_by_pid: dict,
                  n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    rng  = np.random.default_rng(SEED)
    pids = list(set(vals_a_by_pid) | set(vals_b_by_pid))
    nan_ = np.array([float("nan")])
    boot = []
    for _ in range(n_boot):
        samp = rng.choice(pids, size=len(pids), replace=True)
        a = np.concatenate([vals_a_by_pid.get(p, nan_) for p in samp])
        b = np.concatenate([vals_b_by_pid.get(p, nan_) for p in samp])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) and len(b):
            boot.append(float(np.mean(a)) - float(np.mean(b)))
    boot = np.array(boot)
    pt   = (float(np.mean(np.concatenate(list(vals_a_by_pid.values())))) -
            float(np.mean(np.concatenate(list(vals_b_by_pid.values())))))
    return (pt,
            float(np.percentile(boot, alpha / 2 * 100)),
            float(np.percentile(boot, (1 - alpha / 2) * 100)))


def errs_by_pid(err_df: pd.DataFrame) -> dict:
    out = {}
    for pid, grp in err_df.groupby("participant_id", sort=False):
        out[str(pid)] = grp["abs_err"].to_numpy()
    return out


def compute_diff(ef_a: pd.DataFrame, ef_b: pd.DataFrame) -> tuple[float, float, float]:
    return _boot_diff_ci(errs_by_pid(ef_a), errs_by_pid(ef_b))


# ── Check 2 helpers ───────────────────────────────────────────────────────────

def audit_balance(event_at: pd.DataFrame, placebo_at: pd.DataFrame,
                  variables: list[str]) -> pd.DataFrame:
    """Compare distributional balance on key variables (KS test + means)."""
    rows = []
    for var in variables:
        ev  = event_at[var].dropna().to_numpy()
        pl  = placebo_at[var].dropna().to_numpy()
        ks, p = (float("nan"), float("nan"))
        if len(ev) > 0 and len(pl) > 0:
            ks, p = sp_stats.ks_2samp(ev, pl)
        rows.append({
            "variable":    var,
            "event_mean":  float(np.mean(ev)) if len(ev) else float("nan"),
            "event_std":   float(np.std(ev))  if len(ev) else float("nan"),
            "event_median":float(np.median(ev)) if len(ev) else float("nan"),
            "placebo_mean":float(np.mean(pl)) if len(pl) else float("nan"),
            "placebo_std": float(np.std(pl))  if len(pl) else float("nan"),
            "placebo_median":float(np.median(pl)) if len(pl) else float("nan"),
            "std_mean_diff": (abs(np.mean(ev) - np.mean(pl)) / ((np.std(ev) + np.std(pl)) / 2 + 1e-9)
                              if len(ev) > 0 and len(pl) > 0 else float("nan")),
            "ks_stat":     float(ks),
            "ks_p":        float(p),
        })
    return pd.DataFrame(rows)


def nearest_neighbour_match_placebo(
    event_at: pd.DataFrame,
    placebo_at: pd.DataFrame,
    match_cols: list[str],
    caliper: float = 0.25,
) -> pd.DataFrame:
    """1:1 caliper nearest-neighbour matching within participant.

    For each event-enriched trigger, finds the closest generic-placebo trigger
    from the same participant in normalized Euclidean space.
    Rows without a placebo match within `caliper` are dropped.

    Returns the matched placebo DataFrame (same length as matched event set).
    """
    # Normalize match_cols globally (z-score)
    combined = pd.concat([event_at[match_cols], placebo_at[match_cols]], ignore_index=True)
    mu  = combined.mean()
    sig = combined.std().replace(0, 1)

    def _norm(df: pd.DataFrame) -> np.ndarray:
        return ((df[match_cols] - mu) / sig).fillna(0).to_numpy(dtype=float)

    matched_pl_rows = []
    total_event   = 0
    total_matched = 0

    for pid in event_at["participant_id"].unique():
        ev_pid = event_at[event_at["participant_id"].eq(pid)].copy()
        pl_pid = placebo_at[placebo_at["participant_id"].eq(pid)].copy()
        if pl_pid.empty or ev_pid.empty:
            total_event += len(ev_pid)
            continue

        Xev = _norm(ev_pid)
        Xpl = _norm(pl_pid)

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(Xpl)
        dists, idxs = nn.kneighbors(Xev)
        dists = dists.ravel()
        idxs  = idxs.ravel()

        # Greedy 1:1 matching: sort by distance, assign each placebo once
        used_pl = set()
        order   = np.argsort(dists)
        for i in order:
            if dists[i] > caliper:
                continue
            pl_row_idx = pl_pid.index[idxs[i]]
            if pl_row_idx in used_pl:
                continue
            used_pl.add(pl_row_idx)
            matched_pl_rows.append(pl_pid.loc[pl_row_idx])

        total_event   += len(ev_pid)
        total_matched += len(used_pl)

    print(f"  Matched: {total_matched}/{total_event} event anchors found a placebo match "
          f"({100*total_matched/max(total_event,1):.1f}%) within caliper={caliper}")
    if not matched_pl_rows:
        return pd.DataFrame(columns=placebo_at.columns)
    return pd.DataFrame(matched_pl_rows)


# ── Check 3: episode selection ────────────────────────────────────────────────

def select_representative_episodes(
    cache_fo: pd.DataFrame,
    at: pd.DataFrame,
    split: str = "test",
    n_episodes: int = 4,
) -> list[dict]:
    """Pick diverse triggered episodes from `split` for S2-F4.

    Targets:
      ≥1 trigger_up + event_enriched (hyperglycemia context)
      ≥1 trigger_up + NOT event_enriched (generic placebo)
      ≥1 trigger_down if any exist (hypoglycemia)
    For each selected episode return metadata + glucose/prediction traces.
    """
    test_at = at[at["split"].eq(split) & at["non_insulin"].fillna(True)].copy()

    # Find episode starts: first triggered anchor in each run
    test_at = test_at.sort_values(["participant_id", "segment_id", "anchor_ds"])
    grp = test_at.groupby(["participant_id", "segment_id"], sort=False)
    test_at["_prev_trig"] = grp["triggered"].shift(1).fillna(False)
    ep_starts = test_at[test_at["triggered"] & ~test_at["_prev_trig"]].copy()

    cats = {
        "trigger_up_event":   ep_starts[ep_starts["trigger_up"] & ep_starts["event_enriched"]],
        "trigger_up_placebo": ep_starts[ep_starts["trigger_up"] & ~ep_starts["event_enriched"]],
        "trigger_down":       ep_starts[ep_starts["trigger_down"]],
    }

    rng = np.random.default_rng(SEED)
    selected = []
    for cat, pool in cats.items():
        if pool.empty:
            continue
        row = pool.sample(1, random_state=rng.integers(0, 9999)).iloc[0]
        selected.append((cat, row))
        if len(selected) >= n_episodes:
            break

    # Fill remaining slots with trigger_up_event
    while len(selected) < n_episodes and not cats["trigger_up_event"].empty:
        remaining = cats["trigger_up_event"][
            ~cats["trigger_up_event"]["anchor_ds"].isin([s[1]["anchor_ds"] for s in selected])
        ]
        if remaining.empty:
            break
        row = remaining.sample(1, random_state=rng.integers(0, 9999)).iloc[0]
        selected.append(("trigger_up_event_extra", row))

    # For each episode, extract a window of context + forecasts
    context_before_steps = 6   # 30 minutes before trigger
    results = []
    for cat, row in selected:
        pid, seg, anchor = row["participant_id"], row["segment_id"], row["anchor_ds"]
        # Anchor-level context: -context to +1 anchors
        context_at = test_at[
            test_at["participant_id"].eq(pid) &
            test_at["segment_id"].eq(seg) &
            test_at["anchor_ds"].between(anchor - context_before_steps, anchor + 1)
        ][["anchor_ds", "current_glucose", "z_t", "trigger_up", "trigger_down",
           "event_enriched", "one_step_residual", "interval_width"]].to_dict("records")

        # Forecast rows at the trigger anchor (all 12 horizons)
        fc_rows = cache_fo[
            cache_fo["participant_id"].eq(pid) &
            cache_fo["segment_id"].eq(seg) &
            cache_fo["anchor_ds"].eq(anchor)
        ][["horizon_step", "q10", "q50", "q90", "target"]].sort_values("horizon_step")

        results.append({
            "category":         cat,
            "split":            split,
            "participant_id":   str(pid),
            "segment_id":       str(seg),
            "trigger_anchor_ds": int(anchor),
            "trigger_up":       bool(row["trigger_up"]),
            "trigger_down":     bool(row["trigger_down"]),
            "event_enriched":   bool(row["event_enriched"]),
            "current_glucose_at_trigger": float(row["current_glucose"]) if np.isfinite(row["current_glucose"]) else None,
            "z_t":              float(row["z_t"])  if np.isfinite(row.get("z_t", float("nan"))) else None,
            "interval_width":   float(row["interval_width"]) if np.isfinite(row.get("interval_width", float("nan"))) else None,
            "context_anchors":  context_at,
            "forecast_at_trigger": fc_rows.to_dict("records"),
        })
        print(f"  [{cat}] pid={pid} seg={seg} anchor={anchor} "
              f"glucose={row['current_glucose']:.0f} event={row['event_enriched']}")

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_PRE.mkdir(parents=True, exist_ok=True)
    feature_cols = ["z_t", "one_step_residual", "current_glucose", "trigger_up"]

    # ── Load cache ──────────────────────────────────────────────────────────
    _sec("Loading cache")
    df = pd.read_parquet(CACHE_PATH)
    print(f"  {len(df):,} rows")
    cache_fo = df[df["scenario_mode"].astype(str).eq("forecast_only")].copy() \
               if "scenario_mode" in df.columns else df.copy()
    if "anchor_ds" not in cache_fo.columns:
        cache_fo["anchor_ds"] = cache_fo["anchor_time_idx"]

    # ── Build anchor table ──────────────────────────────────────────────────
    _sec("Building anchor table + triggers")
    at = apply_triggers(build_anchor_table(df))
    at = add_glucose_slope(at)
    at["event_enriched"] = event_enriched_proxy(at)
    print(f"  Total anchors: {len(at):,}")
    trig = at[at["non_insulin"].fillna(True) & at["triggered"]]
    print(f"  Triggered non-insulin: {len(trig):,}")

    # ── Load head ───────────────────────────────────────────────────────────
    _sec("Loading head from JSON (no retraining)")
    models, scaler, best_decay, linear_coefs = load_head(HEAD_PATH, feature_cols)
    print(f"  Loaded {len(models)} Ridge models  decay={best_decay}")

    # ── Build val/test triggered forecast rows ──────────────────────────────
    _sec("Building triggered forecast rows (val + test)")
    at_full = at.copy()
    ef: dict[str, dict[str, pd.DataFrame]] = {}
    split_fc_store: dict[str, pd.DataFrame] = {}

    for split in ("validation", "test"):
        trig_keys = at_full[
            at_full["non_insulin"].fillna(True) &
            at_full["triggered"] &
            at_full["split"].eq(split)
        ][["participant_id", "segment_id", "anchor_ds", "event_enriched"]].copy()

        fc = get_triggered_rows(
            cache_fo[cache_fo["split"].eq(split)], trig_keys, at_full
        )
        fc = fc[fc["non_insulin"].fillna(True)].copy()
        split_fc_store[split] = fc
        print(f"  {split}: {len(fc):,} rows  "
              f"{fc['participant_id'].nunique()} participants")

        preds = {
            "base":           fc["q50"],
            "current_shift":  apply_current_shift(fc),
            "decaying_shift": apply_decaying_shift(fc, best_decay),
            "linear":         apply_linear(fc, linear_coefs),
            "head":           apply_head(fc, models, scaler, feature_cols),
        }
        ef[split] = {}
        for mname, pred in preds.items():
            err_df = fc[["participant_id", "anchor_ds", "event_enriched"]].copy()
            err_df["abs_err"] = (fc["target"] - pred).abs().values
            ef[split][mname] = err_df

    # ────────────────────────────────────────────────────────────────────────
    # CHECK 1: Extended paired difference CIs
    # ────────────────────────────────────────────────────────────────────────
    _sec("CHECK 1: Extended Paired Difference CIs")
    pairs = [
        ("base",           "head",           "base_minus_head"),
        ("base",           "linear",         "base_minus_linear"),
        ("linear",         "head",           "linear_minus_head"),
        ("decaying_shift", "linear",         "decaying_shift_minus_linear"),
        ("decaying_shift", "head",           "decaying_shift_minus_head"),
    ]
    diff_rows = []
    for split in ("validation", "test"):
        for (ma, mb, label) in pairs:
            diff, lo, hi = compute_diff(ef[split][ma], ef[split][mb])
            excl = lo > 0
            print(f"  {split:10s}  {label:30s}  diff={diff:+.4f}  "
                  f"CI=[{lo:+.4f}, {hi:+.4f}]  excl_0={excl}")
            diff_rows.append({
                "split": split, "comparison": label,
                "diff": diff, "ci_lo": lo, "ci_hi": hi,
                "ci_excludes_zero": excl,
            })
    diff_df = pd.DataFrame(diff_rows)
    diff_path = OUT_PRE / "study2_extended_diff_cis.csv"
    diff_df.to_csv(diff_path, index=False)
    print(f"\n  Saved -> {diff_path}")

    # Key finding: linear-head
    lh_test = diff_df[(diff_df["split"] == "test") &
                      (diff_df["comparison"] == "linear_minus_head")]
    lh_lo = float(lh_test["ci_lo"].values[0])
    if lh_lo <= 0:
        print("\n  FINDING (linear-head, test): CI INCLUDES ZERO — "
              "head does not clearly improve beyond linear surprise correction.")
        print("  Report must state this explicitly.")
    else:
        print("\n  FINDING (linear-head, test): CI excludes zero — "
              "head meaningfully improves beyond linear.")

    # ────────────────────────────────────────────────────────────────────────
    # CHECK 2: Placebo audit + matched placebo
    # ────────────────────────────────────────────────────────────────────────
    _sec("CHECK 2: Placebo audit")

    # Work with the anchor-level trigger table (not full horizon rows)
    # to audit the balance of event vs generic_placebo on matching features.
    audit_splits = {}
    for split in ("validation", "test"):
        split_at = at_full[
            at_full["split"].eq(split) &
            at_full["non_insulin"].fillna(True) &
            at_full["triggered"]
        ].copy()
        event_at   = split_at[split_at["event_enriched"]].copy()
        placebo_at = split_at[~split_at["event_enriched"]].copy()

        audit_vars = [
            v for v in ["current_glucose", "z_t", "interval_width", "glucose_slope"]
            if v in split_at.columns
        ]
        if "glucose_slope" not in split_at.columns:
            print(f"  [{split}] WARNING: glucose_slope not available; skipping slope audit")

        print(f"\n  [{split}] event={len(event_at):,}  generic_placebo={len(placebo_at):,}")
        balance = audit_balance(event_at, placebo_at, audit_vars)
        print(balance[["variable","event_mean","placebo_mean","std_mean_diff","ks_p"]].to_string(index=False))

        # Time-of-day note
        if "anchor_ds" in split_at.columns and "hours_since_start" not in split_at.columns:
            print(f"  [{split}] TIME-OF-DAY: No wall-clock timestamp in anchor table. "
                  "anchor_ds is relative to stream start, not absolute clock. "
                  "Time-of-day matching is NOT possible without rejoining the original panel timestamps.")

        # Matched placebo: use match_cols available
        match_cols = [v for v in ["current_glucose", "z_t", "interval_width", "glucose_slope"]
                      if v in split_at.columns and v in audit_vars]
        print(f"\n  [{split}] Implementing caliper nearest-neighbour matched placebo")
        print(f"  [{split}] Matching on: {match_cols}")
        matched_pl = nearest_neighbour_match_placebo(
            event_at.reset_index(drop=True),
            placebo_at.reset_index(drop=True),
            match_cols=match_cols,
            caliper=0.25,
        )

        if not matched_pl.empty:
            matched_balance = audit_balance(event_at, matched_pl, audit_vars)
            print(f"\n  [{split}] Balance AFTER matching:")
            print(matched_balance[["variable","event_mean","placebo_mean","std_mean_diff","ks_p"]].to_string(index=False))

        audit_splits[split] = {
            "balance_before": balance,
            "matched_pl":     matched_pl,
            "event_at":       event_at,
            "generic_pl_at":  placebo_at,
        }

    # Compute matched-placebo gap CI for test
    _sec("CHECK 2b: Matched-placebo gap CI (test)")
    test_audi = audit_splits.get("test", {})
    test_ev_at   = test_audi.get("event_at",   pd.DataFrame())
    test_pl_at   = test_audi.get("matched_pl", pd.DataFrame())

    if not test_ev_at.empty and not test_pl_at.empty:
        # Re-derive the test forecast rows filtered to matched keys
        test_fc = split_fc_store["test"]
        ev_keys  = test_ev_at[["participant_id", "segment_id", "anchor_ds"]].drop_duplicates()
        pl_keys  = test_pl_at[["participant_id", "segment_id", "anchor_ds"]].drop_duplicates()

        ev_fc  = test_fc.merge(ev_keys,  on=["participant_id","segment_id","anchor_ds"], how="inner")
        pl_fc  = test_fc.merge(pl_keys,  on=["participant_id","segment_id","anchor_ds"], how="inner")

        head_ev_pred = apply_head(ev_fc,  models, scaler, feature_cols)
        head_pl_pred = apply_head(pl_fc,  models, scaler, feature_cols)
        base_ev_pred = ev_fc["q50"]
        base_pl_pred = pl_fc["q50"]

        def gain_pid_map(base_df, head_df, base_pred, head_pred):
            base_err = (base_df["target"] - base_pred).abs()
            head_err = (base_df["target"] - head_pred).abs()
            diff_by_pid = {}
            merged = base_df[["participant_id"]].copy()
            merged["base_err"] = base_err.values
            merged["head_err"] = head_err.values
            for pid, grp in merged.groupby("participant_id", sort=False):
                diff_by_pid[str(pid)] = (grp["base_err"] - grp["head_err"]).to_numpy()
            return diff_by_pid

        gain_ev_pid  = gain_pid_map(ev_fc,  None, base_ev_pred, head_ev_pred)
        gain_pl_pid  = gain_pid_map(pl_fc,  None, base_pl_pred, head_pl_pred)

        gap_matched, gap_lo, gap_hi = _boot_diff_ci(gain_ev_pid, gain_pl_pid)
        gap_excl = gap_lo > 0
        print(f"  test  matched_placebo  event_gain - placebo_gain:")
        print(f"    gap={gap_matched:+.4f}  95% CI=[{gap_lo:+.4f}, {gap_hi:+.4f}]  "
              f"CI_excludes_zero={gap_excl}")

        # Compare to generic gap from diff_cis
        old_diff = pd.read_csv(OUT_C / "study2_difference_cis.csv")
        old_gap  = old_diff[(old_diff["split"]=="test") &
                            (old_diff["comparison"]=="head_gain_event_minus_placebo")]
        if not old_gap.empty:
            print(f"\n  COMPARISON:")
            print(f"    Generic placebo gap:  {float(old_gap['diff']):.4f} "
                  f"CI=[{float(old_gap['ci_lo']):.4f}, {float(old_gap['ci_hi']):.4f}]")
            print(f"    Matched placebo gap:  {gap_matched:.4f} "
                  f"CI=[{gap_lo:.4f}, {gap_hi:.4f}]")
    else:
        gap_matched, gap_lo, gap_hi, gap_excl = float("nan"), float("nan"), float("nan"), False
        print("  WARNING: could not compute matched gap (empty event or matched placebo)")

    # Save audit report
    audit_report = {
        "placebo_type_generic": "all triggered non-insulin anchors where event_enriched=False",
        "placebo_type_matched": "1:1 caliper nearest-neighbour within participant, "
                                "caliper=0.25 (normalised Euclidean)",
        "match_variables": match_cols,
        "time_of_day_matching": "NOT IMPLEMENTED — anchor_ds is stream-relative integer, "
                                "no wall-clock timestamp available without rejoining original panel",
        "terminology_decision": (
            "The original 'placebo' in study2_event_placebo_gap.csv is GENERIC (unmatched). "
            "It is renamed to 'generic_placebo' in all report figures. "
            "The matched version uses the caliper set above."
        ),
        "test_generic_gap_ci": {"source": "study2_difference_cis.csv"},
        "test_matched_gap": {
            "gap": gap_matched, "ci_lo": gap_lo, "ci_hi": gap_hi,
            "ci_excludes_zero": gap_excl,
        },
    }
    (OUT_PRE / "study2_placebo_audit.json").write_text(
        json.dumps(audit_report, indent=2, default=str)
    )
    print(f"\n  Placebo audit saved -> {OUT_PRE}/study2_placebo_audit.json")

    # Save balance tables
    for split, audi in audit_splits.items():
        audi["balance_before"].to_csv(
            OUT_PRE / f"study2_placebo_balance_{split}_before.csv", index=False
        )

    # ────────────────────────────────────────────────────────────────────────
    # CHECK 3: S2-F4 episode selection (test split)
    # ────────────────────────────────────────────────────────────────────────
    _sec("CHECK 3: S2-F4 Episode Selection (test split)")
    episodes = select_representative_episodes(
        cache_fo=cache_fo,
        at=at,
        split="test",
        n_episodes=4,
    )
    ep_path = OUT_PRE / "study2_s2f4_episodes.json"
    ep_path.write_text(json.dumps(episodes, indent=2, default=str))
    print(f"\n  {len(episodes)} episodes saved -> {ep_path}")

    # Print summary table
    print("\n  Summary:")
    print(f"  {'Category':30s}  {'PID':12s}  {'AnchorDS':8s}  "
          f"{'Glucose':8s}  {'Event':5s}  {'TrigUp':6s}")
    for ep in episodes:
        print(f"  {ep['category']:30s}  {ep['participant_id']:12s}  "
              f"{ep['trigger_anchor_ds']:8d}  "
              f"{str(ep['current_glucose_at_trigger'] or 'N/A'):8s}  "
              f"{str(ep['event_enriched']):5s}  {str(ep['trigger_up']):6s}")

    # ── Final summary ────────────────────────────────────────────────────────
    _sec("SUMMARY: Pre-Block I check findings")
    lh_test = diff_df[(diff_df["split"] == "test") &
                      (diff_df["comparison"] == "linear_minus_head")]
    lh_diff = float(lh_test["diff"].values[0])
    lh_lo   = float(lh_test["ci_lo"].values[0])
    lh_hi   = float(lh_test["ci_hi"].values[0])
    print(f"\n  1. linear-head (test): diff={lh_diff:+.4f} CI=[{lh_lo:+.4f},{lh_hi:+.4f}]")
    print(f"     → {'CI excludes 0: head beats linear' if lh_lo > 0 else 'CI includes 0: head does NOT clearly beat linear — report must caveat this'}")
    print(f"\n  2. Placebo terminology:")
    print(f"     → Rename to 'generic_placebo' throughout (unmatched by design)")
    print(f"     → Matched-placebo gap (test): {gap_matched:+.4f} CI=[{gap_lo:+.4f},{gap_hi:+.4f}]")
    print(f"       {'CI excludes 0' if gap_excl else 'CI includes 0'}")
    print(f"     → Time-of-day matching NOT possible (no wall-clock timestamp in cache)")
    print(f"\n  3. S2-F4: {len(episodes)} test-split episodes selected and saved")
    print(f"\n  All artifacts in: {OUT_PRE}")


if __name__ == "__main__":
    main()
