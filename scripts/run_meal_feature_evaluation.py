#!/usr/bin/env python
"""Unified meal-feature evaluation harness (Blocks 1-3, 5).

Re-anchors the meal-feature evaluation on post-prandial endpoints (peak error,
time-to-peak, meal-window MAE, hyperglycemia AUPRC / recall at precision 0.80)
and decides selection with a coded gate (``ssmcgm.meal_transfer.selection``)
instead of prose. Whole-horizon MAE60 is kept for reporting only.

Residual correction ``yhat = q50 + r(base, meal_feature, h)`` is fit on the
validation cache and evaluated on the participant-disjoint test cache, exactly as
before. The change is the feature design, the endpoints, the baseline, and the
decision rule:

  --base context  : Block 1 baseline is the no-meal context residual model (A).
  --base slc      : Block 2 baseline is A', slope + level + clock only.
  --with-teacher  : add the causal-teacher ceiling C' and the flagged leaky C.
  --orthogonalize : run the Block 2 partial-information test against CGM slope.
  --teacher-mode  : causal | horizon_disjoint | legacy_bidirectional (default causal).

The gate is evaluated on the non-insulin primary cohort (Block 4 quarantine).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ssmcgm.meal_transfer import online_evaluation as OE        # noqa: E402
from ssmcgm.meal_transfer import endpoints as EP                # noqa: E402
from ssmcgm.meal_transfer import selection as SEL               # noqa: E402

MEAL_DIR = REPO / "outputs" / "no_log_scenarios" / "meal_transfer"
ONLINE_DIR = MEAL_DIR / "online_causal"
OUT_DIR = MEAL_DIR / "feature_evaluation"

# Feature designs ---------------------------------------------------------- #
CONTEXT_BASE = ["q50", "spread", "horizon_step", "hba1c_percent_baseline",
                "bmi_baseline", "med_insulin", "med_any_diabetes_drug",
                "study_group_code", "site_code"]
# A' = slope + level + clock only (plus the structural horizon index).
SLC_BASE = ["horizon_step", "dy15", "dy30", "dy60", "level", "hour_sin", "hour_cos"]

MEAL_SETUPS = {
    "B_old_predmeal_flag": ["predmeal_flag"],
    "Cprime_causal_teacher": ["cgmacros_teacher_probability_causal"],   # reachable ceiling
    "C_legacy_bidir_teacher": ["cgmacros_teacher_probability"],          # LEAKY diagnostic
    "D_student_prob": OE.D_FEATURES,
    "G_online_state": OE.G_FEATURES,
    "H_online_full_state": OE.H_FEATURES,
}
# Never eligible for selection: both are derived from the bidirectional teacher
# (the legacy probability and the thresholded predmeal_flag), so their value at an
# anchor encodes future glucose inside the forecast horizon.
LEAKY_SETUPS = {"C_legacy_bidir_teacher", "B_old_predmeal_flag"}
TEACHER_SETUPS = {"B_old_predmeal_flag", "Cprime_causal_teacher", "C_legacy_bidir_teacher"}
CONTROL_MODES = ["shuffle", "time_shift", "block_shuffle"]
ENDPOINT_SHOW = (list(EP.GATE_ENDPOINTS) + EP.REPORT_ENDPOINTS + ["MAE_all"])


# ------------------------------------------------------------------------- #
def _design(df: pd.DataFrame, base: list[str], meal: list[str]) -> np.ndarray:
    X = df[base + meal].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category").cat.codes.astype(float)
    return X.to_numpy(dtype=np.float64)


def _fit(train: pd.DataFrame, base: list[str], meal: list[str], cfg: OE.AblationConfig):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(
        max_iter=cfg.max_iter, learning_rate=cfg.learning_rate, max_depth=cfg.max_depth,
        max_leaf_nodes=cfg.max_leaf_nodes, l2_regularization=cfg.l2_regularization,
        random_state=cfg.random_state)
    m.fit(_design(train, base, meal), train["residual"].to_numpy(dtype=np.float64))
    return m


def _apply(df: pd.DataFrame, model, base: list[str], meal: list[str]) -> np.ndarray:
    return df["q50"].to_numpy(dtype=np.float64) + model.predict(_design(df, base, meal))


def _corrupt(df: pd.DataFrame, meal: list[str], mode: str, seed: int, shift: int) -> pd.DataFrame:
    """Vectorised negative-control corruption of the meal feature columns.

    ``shuffle`` permutes globally, ``time_shift`` lags within participant,
    ``block_shuffle`` permutes within (participant, hour-of-day) blocks. All are
    O(n) / one sort, so they scale to the full cohort.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    if not meal:
        return out
    vals = out[meal].to_numpy()
    if mode == "shuffle":
        out[meal] = vals[rng.permutation(len(out))]
    elif mode == "time_shift":
        # lag by `shift` within participant, in anchor order
        pid = out["participant_id"].astype(str).to_numpy()
        a = out["anchor_ds"].to_numpy()
        order = np.lexsort((a, pid))                  # group by pid, then time
        sorted_pid = pid[order]
        new = np.full_like(vals, np.nan, dtype=np.float64)
        src = order[:-shift] if shift < len(order) else np.array([], dtype=int)
        dst = order[shift:] if shift < len(order) else np.array([], dtype=int)
        # only carry within the same participant
        same = sorted_pid[shift:] == sorted_pid[:-shift] if shift < len(order) else np.array([], dtype=bool)
        new[dst[same]] = vals[src[same]]
        nan_rows = ~np.isfinite(new).all(axis=1)
        new[nan_rows] = 0.0
        out[meal] = new
    elif mode == "block_shuffle":
        hod = ((out["anchor_ds"].astype(int).to_numpy() * 5) % 1440) // 60
        pid = out["participant_id"].astype(str).to_numpy()
        g = pd.factorize(pd.Series(list(zip(pid, hod))))[0]
        r = rng.random(len(out))
        shuffled_order = np.lexsort((r, g))           # within-group random order
        natural_order = np.lexsort((np.arange(len(out)), g))
        new = vals.copy()
        new[natural_order] = vals[shuffled_order]
        out[meal] = new
    else:
        raise ValueError(mode)
    return out


def _build_extra_anchor(with_teacher: bool, teacher_mode: str, keep: set | None) -> pd.DataFrame:
    """Per-(participant_id, ds) anchor features beyond the online state table:
    slope/level/clock (always) and teacher probabilities (optional)."""
    student = pd.read_parquet(
        MEAL_DIR / "causal_student_predictions.parquet",
        columns=["participant_id", "ds", "cgm_slope_15", "cgm_slope_30", "cgm_slope_60",
                 "cgm_glucose", "tod_sin", "tod_cos", "student_meal_probability"])
    student["participant_id"] = student["participant_id"].astype(str)
    if keep is not None:
        student = student[student["participant_id"].isin(keep)].copy()
    student = student.rename(columns={
        "cgm_slope_15": "dy15", "cgm_slope_30": "dy30", "cgm_slope_60": "dy60",
        "cgm_glucose": "level", "tod_sin": "hour_sin", "tod_cos": "hour_cos"})
    student["participant_id"] = student["participant_id"].astype(str)
    extra = student
    if with_teacher:
        tcols = ["participant_id", "ds", "cgmacros_teacher_probability", "predmeal_flag"]
        teach = pd.read_parquet(MEAL_DIR / "teacher_predictions.parquet", columns=tcols)
        teach["participant_id"] = teach["participant_id"].astype(str)
        if keep is not None:
            teach = teach[teach["participant_id"].isin(keep)].copy()
        extra = extra.merge(teach, on=["participant_id", "ds"], how="left")
        causal_path = MEAL_DIR / "teacher_predictions_causal.parquet"
        if causal_path.exists():
            tc = pd.read_parquet(causal_path)
            tc["participant_id"] = tc["participant_id"].astype(str)
            if keep is not None:
                tc = tc[tc["participant_id"].isin(keep)].copy()
            col = {"causal": "cgmacros_teacher_probability_causal",
                   "horizon_disjoint": "cgmacros_teacher_probability_horizon_disjoint",
                   "legacy_bidirectional": None}[teacher_mode]
            tcausal_cols = ["participant_id", "ds"] + [c for c in tc.columns
                                                       if c.startswith("cgmacros_teacher_probability")]
            extra = extra.merge(tc[tcausal_cols], on=["participant_id", "ds"], how="left")
            if col is not None and col in extra.columns:
                extra["cgmacros_teacher_probability_causal"] = extra[col]
        else:
            # No causal teacher computed yet: fall back to legacy so the column
            # exists; the provenance assertion will flag it as leaky.
            extra["cgmacros_teacher_probability_causal"] = extra["cgmacros_teacher_probability"]
    return extra


def _attach_extra(df: pd.DataFrame, extra: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    s = extra.set_index(["participant_id", "ds"])
    key = list(zip(df["participant_id"].astype(str), df["anchor_ds"].astype(int)))
    for c in cols:
        if c in s.columns:
            df[c] = s[c].reindex(key).to_numpy()
    return df


# ------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["context", "slc"], default="context")
    ap.add_argument("--with-teacher", action="store_true")
    ap.add_argument("--teacher-mode", choices=["causal", "horizon_disjoint", "legacy_bidirectional"],
                    default="causal")
    ap.add_argument("--orthogonalize", action="store_true")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--max-participants", type=int, default=0, help="0 = all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = OE.AblationConfig(random_state=args.seed)
    base = SLC_BASE if args.base == "slc" else CONTEXT_BASE
    baseline_setup = "Aprime" if args.base == "slc" else "A_context"

    print(f"[load] online states + eval mask  (base={args.base}, teacher={args.with_teacher}, "
          f"teacher_mode={args.teacher_mode})", flush=True)
    states = pd.read_parquet(ONLINE_DIR / "online_meal_states.parquet")
    sizes = pd.read_parquet(ONLINE_DIR / "online_response_size_predictions.parquet")
    mask = pd.read_parquet(ONLINE_DIR / "independent_meal_eval_mask.parquet")

    val_a = pd.read_parquet(MEAL_DIR / "_aligned_validation.parquet")
    tst_a = pd.read_parquet(MEAL_DIR / "_aligned_test.parquet")
    if args.max_participants:
        rng = np.random.default_rng(args.seed)
        # validation and test caches are participant-disjoint, so sample each.
        def _subsample(df):
            ids = np.array(sorted(set(df["participant_id"].astype(str))))
            keep = set(rng.choice(ids, size=min(args.max_participants, len(ids)), replace=False))
            return df[df["participant_id"].astype(str).isin(keep)].copy()
        val_a = _subsample(val_a)
        tst_a = _subsample(tst_a)

    # Restrict the heavy source tables to participants in the caches (~460 of
    # 1591), which makes the repeated MultiIndex reindex in attach fast.
    keep_union = (set(val_a["participant_id"].astype(str)) |
                  set(tst_a["participant_id"].astype(str)))
    for nm, frame in (("states", states), ("sizes", sizes), ("mask", mask)):
        frame["participant_id"] = frame["participant_id"].astype(str)
    states = states[states["participant_id"].isin(keep_union)].copy()
    sizes = sizes[sizes["participant_id"].isin(keep_union)].copy()
    mask = mask[mask["participant_id"].isin(keep_union)].copy()

    print("[attach] online features to aligned forecast caches", flush=True)
    train = OE.attach_online_frame(val_a, states, sizes, mask)
    test = OE.attach_online_frame(tst_a, states, sizes, mask)

    extra = _build_extra_anchor(args.with_teacher, args.teacher_mode, keep_union)
    # student_meal_probability is overwritten from the (Block 4) rebuilt causal
    # student so feature D reflects the insulin-quarantine + uncertain-rescue model.
    extra_cols = ["dy15", "dy30", "dy60", "level", "hour_sin", "hour_cos",
                  "student_meal_probability"]
    if args.with_teacher:
        extra_cols += ["cgmacros_teacher_probability", "cgmacros_teacher_probability_causal",
                       "predmeal_flag"]
    train = _attach_extra(train, extra, extra_cols)
    test = _attach_extra(test, extra, extra_cols)

    # active setups
    active = ["D_student_prob", "G_online_state", "H_online_full_state"]
    if args.with_teacher:
        active = ["B_old_predmeal_flag", "Cprime_causal_teacher",
                  "C_legacy_bidir_teacher"] + active
    print(f"[fit] baseline {baseline_setup} + {len(active)} feature setups", flush=True)

    cache = EP.AnchorCache(test)
    # baseline (no meal feature)
    bm = _fit(train, base, [], cfg)
    base_pred = _apply(test, bm, base, [])
    cache.add_forecast(baseline_setup, base_pred)
    cache.add_forecast("BASELINE_q50_uncorrected", test["q50"].to_numpy(dtype=np.float64))

    models, preds = {}, {baseline_setup: base_pred,
                        "BASELINE_q50_uncorrected": test["q50"].to_numpy(dtype=np.float64)}
    control_preds: dict = {}
    for setup in active:
        meal = MEAL_SETUPS[setup]
        m = _fit(train, base, meal, cfg)
        models[setup] = m
        preds[setup] = _apply(test, m, base, meal)
        cache.add_forecast(setup, preds[setup])
        for mode in CONTROL_MODES:
            corrupted = _corrupt(test, meal, mode, args.seed + 7, cfg.time_shift_steps)
            cp = _apply(corrupted, m, base, meal)
            control_preds[(setup, mode)] = cp
            cache.add_forecast(f"{setup}__{mode}", cp)

    # endpoint table (overall + non-insulin) ------------------------------- #
    non_insulin = test["med_insulin"].fillna(0).to_numpy() == 0
    rows = []
    for name, pred in [("BASELINE_q50_uncorrected", preds["BASELINE_q50_uncorrected"]),
                       (baseline_setup, base_pred)] + [(s, preds[s]) for s in active]:
        rows.append({"setup": name, "cohort": "all", **EP.forecast_metrics(test, pred)})
        rows.append({"setup": name, "cohort": "non_insulin",
                     **EP.forecast_metrics(test[non_insulin], pred[non_insulin])})
    endpoint_tbl = pd.DataFrame(rows)
    tag = (args.tag + "_") if args.tag else ""
    endpoint_tbl.to_csv(OUT_DIR / f"{tag}meal_feature_endpoints.csv", index=False)

    # negative controls (non-insulin): endpoint values for real vs each control --
    ctrl_rows = []
    for setup in active:
        ctrl_rows.append({"setup": setup, "control": "real",
                          **EP.forecast_metrics(test[non_insulin], preds[setup][non_insulin])})
        for mode in CONTROL_MODES:
            cp = control_preds[(setup, mode)]
            ctrl_rows.append({"setup": setup, "control": mode,
                              **EP.forecast_metrics(test[non_insulin], cp[non_insulin])})
    pd.DataFrame(ctrl_rows).to_csv(OUT_DIR / f"{tag}meal_feature_negative_controls.csv", index=False)

    # subgroups: explicit insulin vs non-insulin split, plus study group --------
    sg_rows = []
    insulin = test["med_insulin"].fillna(0).to_numpy()
    for name, pred in [("BASELINE_q50_uncorrected", preds["BASELINE_q50_uncorrected"]),
                       (baseline_setup, base_pred)] + [(s, preds[s]) for s in active]:
        for grp, m in [("med_insulin=0", insulin == 0), ("med_insulin=1", insulin == 1)]:
            if m.sum() == 0:
                continue
            sg_rows.append({"setup": name, "subgroup": "med_insulin",
                            "value": grp, "n_participants": int(test.loc[m, "participant_id"].nunique()),
                            **EP.forecast_metrics(test[m], pred[m])})
        for val, idx in test.groupby("participants_study_group").groups.items():
            loc = test.index.get_indexer(idx)
            sg_rows.append({"setup": name, "subgroup": "participants_study_group",
                            "value": str(val), "n_participants": int(test.loc[idx, "participant_id"].nunique()),
                            **EP.forecast_metrics(test.loc[idx], pred[loc])})
    pd.DataFrame(sg_rows).to_csv(OUT_DIR / f"{tag}meal_feature_subgroups.csv", index=False)

    # harness cost A'/A vs untouched q50 (report) -------------------------- #
    qm = EP.forecast_metrics(test[non_insulin], preds["BASELINE_q50_uncorrected"][non_insulin])
    am = EP.forecast_metrics(test[non_insulin], base_pred[non_insulin])
    harness_cost = {f"harness_cost_{k}": am[k] - qm[k] for k in
                    ["MAE_60min", "MAE_eval_meal_window", "peak_error_1h"]}

    # selection gate (non-insulin) ---------------------------------------- #
    print(f"[gate] coded selection on non-insulin cohort (n_boot={args.n_boot})", flush=True)
    parts_ni = np.array(sorted(set(test.loc[non_insulin, "participant_id"].astype(str))))
    evidence, summary = SEL.evaluate_selection(
        cache, baseline_setup, active, parts_ni,
        leaky_setups=LEAKY_SETUPS, n_boot=args.n_boot, seed=args.seed)
    evidence.to_csv(OUT_DIR / f"{tag}meal_selection_decision.csv", index=False)
    summary.to_csv(OUT_DIR / f"{tag}meal_selected_summary.csv", index=False)

    # provenance (Block 3) ------------------------------------------------ #
    provenance = _provenance_table(args, extra)
    provenance.to_csv(OUT_DIR / f"{tag}teacher_provenance.csv", index=False)
    _assert_provenance(provenance, active)

    # orthogonalization / partial information (Block 2) ------------------- #
    if args.orthogonalize:
        from ssmcgm.meal_transfer import partial_information as PI
        pinfo = PI.run_partial_information(train, test, SLC_BASE,
                                           feature_setups={s: MEAL_SETUPS[s] for s in active},
                                           non_insulin_mask=non_insulin, cfg=cfg)
        pinfo.to_csv(OUT_DIR / f"{tag}partial_information.csv", index=False)
    else:
        pinfo = None

    _print_summary(endpoint_tbl, summary, harness_cost, baseline_setup, active, pinfo)
    manifest = {"base": args.base, "baseline_setup": baseline_setup,
                "with_teacher": args.with_teacher, "teacher_mode": args.teacher_mode,
                "n_boot": args.n_boot, "harness_cost": harness_cost,
                "active_setups": active, "leaky_setups": sorted(LEAKY_SETUPS)}
    (OUT_DIR / f"{tag}meal_feature_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))
    return 0


def _provenance_table(args, extra: pd.DataFrame) -> pd.DataFrame:
    """Record, per meal feature, whether it can use a CGM sample later than the
    anchor. The teacher legacy read averages windows that extend ~6 h past the
    anchor (leaky); the causal read uses only samples <= anchor."""
    L = 72
    recs = [
        {"feature": "student_meal_probability", "max_input_offset_steps_vs_anchor": 0,
         "uses_future_glucose": False, "leakage_free": True,
         "note": "strictly causal HistGBM on past/current features"},
        {"feature": "online_state_features", "max_input_offset_steps_vs_anchor": 0,
         "uses_future_glucose": False, "leakage_free": True,
         "note": "forward-only online decoder / size model"},
        {"feature": "predmeal_flag", "max_input_offset_steps_vs_anchor": L - 1,
         "uses_future_glucose": True, "leakage_free": False,
         "note": "thresholded bidirectional teacher flag; value at anchor encodes future CGM"},
    ]
    if args.with_teacher:
        recs.append({"feature": "cgmacros_teacher_probability (legacy_bidirectional)",
                     "max_input_offset_steps_vs_anchor": L - 1,
                     "uses_future_glucose": True, "leakage_free": False,
                     "note": "bidirectional 6h overlap-mean reads CGM up to ~+6h past anchor"})
        causal_path = MEAL_DIR / "teacher_predictions_causal.parquet"
        recs.append({"feature": "cgmacros_teacher_probability_causal (teacher_mode=%s)" % args.teacher_mode,
                     "max_input_offset_steps_vs_anchor": 0 if causal_path.exists() else (L - 1),
                     "uses_future_glucose": not causal_path.exists(),
                     "leakage_free": causal_path.exists(),
                     "note": "trailing-window read ending at anchor" if causal_path.exists()
                             else "causal teacher not built yet, fell back to legacy"})
    return pd.DataFrame(recs)


def _assert_provenance(provenance: pd.DataFrame, active: list[str]) -> None:
    """Hard gate: a teacher probability used as a *selectable* model input must
    never use a CGM sample later than its anchor. The leaky legacy teacher is
    only permitted as an explicitly flagged diagnostic setup."""
    if "Cprime_causal_teacher" in active:
        row = provenance[provenance["feature"].str.startswith("cgmacros_teacher_probability_causal")]
        if not row.empty and not bool(row.iloc[0]["leakage_free"]):
            raise AssertionError(
                "PROVENANCE FAILURE: Cprime uses a teacher probability computed from CGM "
                "later than the anchor. Build the causal teacher first "
                "(scripts/build_causal_teacher.py) or set --teacher-mode causal.")


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{float(x):.{nd}f}"


def _print_summary(endpoint_tbl, summary, harness_cost, baseline_setup, active, pinfo):
    ni = endpoint_tbl[endpoint_tbl.cohort == "non_insulin"].set_index("setup")
    cols = list(EP.GATE_ENDPOINTS) + ["MAE_60min"]
    print("\n=== ENDPOINTS (non-insulin cohort) ===")
    print("setup | " + " | ".join(cols))
    order = ["BASELINE_q50_uncorrected", baseline_setup] + active
    for s in order:
        if s in ni.index:
            print(f"{s} | " + " | ".join(_fmt(ni.loc[s, c]) for c in cols))
    print(f"\nharness cost ({baseline_setup} - q50, non-insulin): " +
          ", ".join(f"{k.replace('harness_cost_','')}={_fmt(v)}" for k, v in harness_cost.items()))
    print("\n=== SELECTION DECISION (non-insulin gate) ===")
    for _, r in summary.iterrows():
        print(f"{r.setup}: selected={bool(r.selected)} "
              f"(passing {int(r.n_passing_endpoints)}/{int(r.min_required)} endpoints, "
              f"leakage_free={bool(r.leakage_free)})")
    if pinfo is not None and len(pinfo):
        print("\n=== PARTIAL INFORMATION vs CGM slope (held-out) ===")
        for _, r in pinfo.iterrows():
            print(f"{r['setup']}: partial_corr={_fmt(r['partial_corr_with_q50_residual'])}, "
                  f"d_meal_window_MAE={_fmt(r['delta_meal_window_mae'])}, "
                  f"d_peak={_fmt(r['delta_peak_error'])}")


if __name__ == "__main__":
    raise SystemExit(main())
