"""Marginal single-feature regression, h_t = f(one clinical feature), all 6
features x all 5 timepoints (30 cells). Locked spec.

Reuses, unmodified: H_orth reconstruction (orthogonalize/standardize,
including the hour-0 constant-column drop), the closed-form OLS/PRESS
machinery (hat_diagonal/fit_and_score), and the row-permutation procedure
(run_permutations) already validated for the 6-feature block regression
(scripts/t2d_confound_h_on_clinical_regression.py) and its hour-0 robustness
checks (scripts/t2d_confound_h_on_clinical_hour0_robustness.py). Only the
design matrix changes: [ones, single standardized clinical column] (N x 2)
instead of [ones, all 6] (N x 7).

Cohort: N=61, the block regression's primary run (severity-valid AND all-6-
clinical-complete), including the 3 flagged participants -- explicitly NOT
the N=58 leverage-excluded variant from the hour-0 robustness check.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from t2d_confound_h_on_clinical_regression import (  # noqa: E402
    fit_and_score,
    hat_diagonal,
)
from t2d_confound_h_on_clinical_hour0_robustness import (  # noqa: E402
    drop_constant_columns,
    run_permutations,
)
from t2d_confound_permutation_test import (  # noqa: E402
    GLYCEMIC_FEATURES,
    HOURS,
    OUTPUT_DIR,
    TARGET_COLUMNS,
    compute_severity,
    load_all_snapshots,
    load_clinical_targets,
    load_cohort,
    orthogonalize,
    standardize,
)

OUTPUT_TABLE_PATH = OUTPUT_DIR / "t2d_confound_h_on_clinical_marginal_results.csv"

DESIGN_RANK = 2  # p = intercept + 1 feature
LEVERAGE_FLAG_MULTIPLIER = 3.0
N_PERMUTATIONS_FULL = 1000
TIME_BUDGET_SECONDS = 10 * 60
TIMING_FEATURE = "tg_hdl_ratio"
TIMING_HOUR = 6
TIMING_N_PERMUTATIONS = 50
BONFERRONI_TESTS = len(HOURS) * len(TARGET_COLUMNS)  # 30
BONFERRONI_ALPHA = 0.05 / BONFERRONI_TESTS
RANDOM_SEED = 42


def build_single_design(col_std: np.ndarray) -> np.ndarray:
    n = col_std.shape[0]
    return np.hstack([np.ones((n, 1)), col_std.reshape(-1, 1)])


def main() -> None:
    print("=== Cohort / snapshots / severity / clinical targets (reused, unchanged) ===")
    cohort = load_cohort()
    cohort_ids = cohort["participant_id"]
    snapshots = load_all_snapshots(cohort_ids)
    severity = compute_severity(cohort_ids)
    severity_valid = ~severity[GLYCEMIC_FEATURES].isna().any(axis=1).to_numpy()
    severity_raw_full = severity[GLYCEMIC_FEATURES].to_numpy(dtype=float)
    clinical = load_clinical_targets(cohort_ids)
    clinical_valid = ~clinical[TARGET_COLUMNS].isna().any(axis=1).to_numpy()
    valid = severity_valid & clinical_valid
    n_used = int(valid.sum())
    print(f"Cohort N = {n_used} (severity-valid AND all-6-clinical-complete, 3 flagged "
          f"participants included, matching the block regression's primary N=61 run)\n")

    clinical_raw = clinical.loc[valid, TARGET_COLUMNS].to_numpy(dtype=float)
    clinical_std = standardize(clinical_raw)  # same 6-column standardization as the block regression

    severity_raw = severity_raw_full[valid]
    severity_std = standardize(severity_raw)
    y_by_hour = {}
    for hour in HOURS:
        h_raw = snapshots[hour][valid]
        h_orth = orthogonalize(h_raw, severity_std)
        h_orth = drop_constant_columns(h_orth, f"hour {hour}")
        y_by_hour[hour] = standardize(h_orth)
        assert not np.any(np.isnan(y_by_hour[hour])), f"hour {hour}: NaN survived column drop"

    avg_leverage = DESIGN_RANK / n_used
    flag_threshold = LEVERAGE_FLAG_MULTIPLIER * avg_leverage
    print(f"p={DESIGN_RANK}, average leverage (p/N) = {avg_leverage:.4f}, "
          f"flag threshold (3p/N) = {flag_threshold:.4f}\n")

    print("=== Fitting all 30 cells (in-sample + closed-form LOO, no permutation yet) ===")
    cells = {}  # (feature, hour) -> dict
    for feature_idx, feature in enumerate(TARGET_COLUMNS):
        col_std = clinical_std[:, feature_idx]
        design = build_single_design(col_std)
        xtx_pinv = np.linalg.pinv(design.T @ design)
        h_diag = hat_diagonal(design, xtx_pinv)
        max_leverage = float(h_diag.max())
        n_flagged = int((h_diag > flag_threshold).sum())
        if n_flagged:
            flagged_ids = cohort_ids[valid].reset_index(drop=True)[h_diag > flag_threshold]
            for pid, lev in zip(flagged_ids, h_diag[h_diag > flag_threshold]):
                print(f"  FLAGGED (console only): feature={feature}, participant_id={pid}, h_ii={lev:.4f}")
        for hour in HOURS:
            y = y_by_hour[hour]
            r2_insample, r2_loo, ss_tot = fit_and_score(design, y, xtx_pinv, h_diag)
            cells[(feature, hour)] = {
                "design": design, "xtx_pinv": xtx_pinv, "h_diag": h_diag, "ss_tot": ss_tot,
                "r2_insample": r2_insample, "r2_loo": r2_loo, "max_leverage": max_leverage,
            }
            print(f"  {feature:<26} hour {hour:>2}: R2_insample={r2_insample:+.4f}  R2_loo={r2_loo:+.4f}  "
                  f"max_leverage={max_leverage:.4f}")
    print()

    print(f"=== TIMING CHECK: {TIMING_FEATURE} @ hour {TIMING_HOUR}, {TIMING_N_PERMUTATIONS} permutations ===")
    cell = cells[(TIMING_FEATURE, TIMING_HOUR)]
    p_timing, elapsed_timing = run_permutations(
        cell["design"], y_by_hour[TIMING_HOUR], cell["h_diag"], cell["ss_tot"], cell["r2_loo"],
        TIMING_N_PERMUTATIONS, seed=RANDOM_SEED,
    )
    t_per_permutation = elapsed_timing / TIMING_N_PERMUTATIONS
    est_total = BONFERRONI_TESTS * N_PERMUTATIONS_FULL * t_per_permutation
    print(f"  extrapolated total at {N_PERMUTATIONS_FULL} permutations x {BONFERRONI_TESTS} cells: "
          f"{est_total:.1f}s ({est_total / 60:.2f} min)")

    if est_total > TIME_BUDGET_SECONDS:
        print(
            f"\nSTOP: extrapolated total ({est_total / 60:.2f} min) exceeds the "
            f"{TIME_BUDGET_SECONDS / 60:.0f} min budget used for this machinery's earlier runs. "
            f"Stopping for confirmation before committing to the full grid, per the locked spec's "
            f"'before committing' instruction."
        )
        return
    print(f"  Under the {TIME_BUDGET_SECONDS / 60:.0f} min budget (same policy already used for this "
          f"machinery); proceeding directly to {N_PERMUTATIONS_FULL} permutations per cell.\n")

    print(f"=== Permutation grid: {BONFERRONI_TESTS} cells x {N_PERMUTATIONS_FULL} permutations ===")
    rows = []
    for feature in TARGET_COLUMNS:
        for hour in HOURS:
            cell = cells[(feature, hour)]
            seed = RANDOM_SEED * 1000 + hour * 10 + TARGET_COLUMNS.index(feature)
            p_value, elapsed = run_permutations(
                cell["design"], y_by_hour[hour], cell["h_diag"], cell["ss_tot"], cell["r2_loo"],
                N_PERMUTATIONS_FULL, seed=seed,
            )
            pass_bonferroni = bool(p_value < BONFERRONI_ALPHA)
            print(f"  {feature:<26} hour {hour:>2}: R2_loo={cell['r2_loo']:+.4f}  p={p_value:.4f}  "
                  f"pass={pass_bonferroni}  [{elapsed:.2f}s]")
            rows.append({
                "feature": feature,
                "hour": hour,
                "R2_insample": cell["r2_insample"],
                "R2_loo": cell["r2_loo"],
                "p_value": p_value,
                "bonferroni_pass": pass_bonferroni,
                "max_leverage": cell["max_leverage"],
            })

    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_TABLE_PATH, index=False)
    print(f"\nWrote {OUTPUT_TABLE_PATH}")

    print("\n=== Final table ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(results.to_string(index=False))

    print(f"\nBonferroni threshold (30 cells): 0.05/30 = {BONFERRONI_ALPHA:.5f}")
    print(f"Cells passing: {int(results['bonferroni_pass'].sum())}/30")

    print("\n=== tg_hdl_ratio marginal vs. joint block regression ===")
    tg_marginal = results[results["feature"] == "tg_hdl_ratio"].sort_values("hour")
    print(tg_marginal[["hour", "R2_insample", "R2_loo", "p_value", "bonferroni_pass"]].to_string(index=False))


if __name__ == "__main__":
    main()
