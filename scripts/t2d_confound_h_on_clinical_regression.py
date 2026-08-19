"""Multivariate regression of severity-residualized h_t on the clinical
profile, T2D oral non-insulin, all five stream snapshots.

Reverse direction of scripts/t2d_confound_permutation_test.py (clinical ~
h_t): here h_t (35,072-dim, standardized per column) is the response and
the six clinical factors (+intercept) are the design, so N=61 >> p=7 and
this is a well-conditioned ordinary multivariate least-squares fit -- no
ridge, no PLS, no dimensionality reduction of the response, closed-form
leave-one-out via the hat-matrix diagonal (PRESS), not a per-participant
refit loop. Locked spec; do not deviate based on intermediate results.

API/provenance note, same as the earlier CCA script: H_orth was never
persisted to disk by t2d_confound_permutation_test.py (computed in-memory
per grid cell, discarded). Regenerated here via the identical, imported
orthogonalize() call on the same cached snapshots and the same severity
control -- deterministic linear algebra, reproduces the original matrices
exactly.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

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

OUTPUT_TABLE_PATH = OUTPUT_DIR / "t2d_confound_h_on_clinical_results.csv"

N_CLINICAL_COLUMNS = len(TARGET_COLUMNS)  # 6
DESIGN_RANK = N_CLINICAL_COLUMNS + 1  # p = 7, intercept + 6 clinical factors
LEVERAGE_FLAG_MULTIPLIER = 3.0  # flag h_ii > 3p/N
LEVERAGE_STOP_THRESHOLD = 0.95  # stop before fitting if any h_ii exceeds this
MIN_N_FOR_POWER = 40

# Divergence sanity check for the in-sample vs LOO(PRESS) R^2 gap. This is a
# smaller threshold than the earlier CCA/regression checks (0.10) because
# this design is well-conditioned (p=7, N~61), not the N<<P regime those
# checks guarded against -- the expected honest gap here is modest, not
# dramatic. The purpose is the same: catch a broken leverage computation
# (e.g. h_ii all zero would make R2_loo == R2_insample exactly).
DIVERGENCE_GAP_MIN = 0.01

N_PERMUTATIONS_FULL = 1000
TIME_BUDGET_SECONDS = 10 * 60
TIMING_HOUR = 48
TIMING_N_PERMUTATIONS = 50
BONFERRONI_TESTS = len(HOURS)  # 5
BONFERRONI_ALPHA = 0.05 / BONFERRONI_TESTS
RANDOM_SEED = 42


def build_design(clinical_std: np.ndarray) -> np.ndarray:
    n = clinical_std.shape[0]
    return np.hstack([np.ones((n, 1)), clinical_std])


def hat_diagonal(design: np.ndarray, xtx_pinv: np.ndarray) -> np.ndarray:
    # h_ii = x_i @ pinv(X'X) @ x_i, computed row-wise (equivalent to but
    # cheaper than materializing the full N x N hat matrix and taking its
    # diagonal, though at this N either is trivial).
    return np.einsum("ij,jk,ik->i", design, xtx_pinv, design)


def fit_and_score(design: np.ndarray, y: np.ndarray, xtx_pinv: np.ndarray, h_diag: np.ndarray) -> tuple[float, float, float]:
    b = xtx_pinv @ design.T @ y
    y_hat = design @ b
    residual = y - y_hat
    ss_tot = float(np.sum(y ** 2))
    ss_res = float(np.sum(residual ** 2))
    r2_insample = 1.0 - ss_res / ss_tot

    loo_residual = residual / (1.0 - h_diag)[:, None]
    press = float(np.sum(loo_residual ** 2))
    r2_loo = 1.0 - press / ss_tot
    return r2_insample, r2_loo, ss_tot


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
    print(f"Cohort N = {n_used} (severity-valid AND all-6-clinical-complete)")
    underpowered = n_used < MIN_N_FOR_POWER
    if underpowered:
        print(f"  UNDERPOWERED: N={n_used} < {MIN_N_FOR_POWER}")

    clinical_valid_frame = clinical.loc[valid].reset_index(drop=True)
    clinical_raw = clinical_valid_frame[TARGET_COLUMNS].to_numpy(dtype=float)
    clinical_std = standardize(clinical_raw)
    design = build_design(clinical_std)
    print(f"Clinical_design shape: {design.shape} (p={DESIGN_RANK}, intercept + {N_CLINICAL_COLUMNS} factors)\n")

    print("=== LEVERAGE DIAGNOSTIC (run first, X-only, timepoint-invariant) ===")
    xtx = design.T @ design
    xtx_pinv = np.linalg.pinv(xtx)
    h_diag = hat_diagonal(design, xtx_pinv)
    avg_leverage = DESIGN_RANK / n_used
    flag_threshold = LEVERAGE_FLAG_MULTIPLIER * avg_leverage
    print(f"  p/N average leverage = {avg_leverage:.4f}, flag threshold (3p/N) = {flag_threshold:.4f}")
    cca_cohort_ids = cohort_ids[valid].reset_index(drop=True)
    n_flagged = 0
    for i in range(n_used):
        if h_diag[i] > flag_threshold:
            n_flagged += 1
            print(f"    FLAGGED (console only): participant_id={cca_cohort_ids.iloc[i]}, h_ii={h_diag[i]:.4f}")
    if n_flagged == 0:
        print("  No participant exceeds the 3p/N flag threshold.")
    max_leverage = float(h_diag.max())
    print(f"  Max leverage: {max_leverage:.4f}")
    if max_leverage > LEVERAGE_STOP_THRESHOLD:
        print(
            f"\nSTOP: max leverage {max_leverage:.4f} exceeds {LEVERAGE_STOP_THRESHOLD}. That "
            "participant's leave-one-out residual would be numerically unstable regardless of any "
            "real relationship. Stopping before fitting, per the locked spec."
        )
        return
    print("  Max leverage below the 0.95 stop threshold; proceeding to fit.\n")

    print("=== H_orth reconstruction ===")
    print(
        "  Note: H_orth was not persisted to disk by the original regression script; regenerated "
        "here via the identical, imported orthogonalize() call on the same cached snapshots and "
        "severity control -- deterministic, reproduces the original matrices exactly.\n"
    )
    severity_raw = severity_raw_full[valid]
    severity_std = standardize(severity_raw)
    h_orth_std_by_hour = {}
    n_response_columns_by_hour = {}
    for hour in HOURS:
        h_raw = snapshots[hour][valid]
        h_orth = orthogonalize(h_raw, severity_std)
        col_std = h_orth.std(axis=0)
        zero_var_mask = col_std == 0
        n_zero_var = int(zero_var_mask.sum())
        if n_zero_var:
            # Per user decision (2026-08-13): drop constant columns from the fit at
            # this timepoint rather than dividing 0/0 -> NaN, and rather than
            # zero-mapping them in place (which would silently keep a fake column).
            print(
                f"  hour {hour:>2}: {n_zero_var}/{h_orth.shape[1]} response columns are exactly "
                f"constant (zero variance) across this cohort at this timepoint -- dropped from "
                f"the fit at this timepoint only, not zero-mapped, not left in to divide by zero."
            )
            h_orth = h_orth[:, ~zero_var_mask]
        h_orth_std_by_hour[hour] = standardize(h_orth)  # per-column center+scale, per spec
        n_response_columns_by_hour[hour] = h_orth.shape[1]
        assert not np.any(np.isnan(h_orth_std_by_hour[hour])), f"hour {hour}: NaN survived column drop"
        print(f"  hour {hour:>2}: H_orth_standardized shape {h_orth_std_by_hour[hour].shape}")
    print()

    print("=== MODEL, per timepoint: in-sample vs closed-form LOO (PRESS) R^2 ===")
    r2_insample_by_hour = {}
    r2_loo_by_hour = {}
    ss_tot_by_hour = {}
    for hour in HOURS:
        y = h_orth_std_by_hour[hour]
        r2_insample, r2_loo, ss_tot = fit_and_score(design, y, xtx_pinv, h_diag)
        r2_insample_by_hour[hour] = r2_insample
        r2_loo_by_hour[hour] = r2_loo
        ss_tot_by_hour[hour] = ss_tot
        gap = r2_insample - r2_loo
        print(
            f"  hour {hour:>2}: R2_insample={r2_insample:+.4f} (expected to be inflated, not the "
            f"result to use)  R2_loo={r2_loo:+.4f}  gap={gap:+.4f}"
        )

    min_gap = min(r2_insample_by_hour[h] - r2_loo_by_hour[h] for h in HOURS)
    print(f"\n  Minimum in-sample - LOO gap across timepoints: {min_gap:+.4f} "
          f"(divergence threshold: {DIVERGENCE_GAP_MIN})")
    if min_gap < DIVERGENCE_GAP_MIN:
        print(
            f"\nSTOP: at least one timepoint's in-sample and LOO R^2 do not diverge meaningfully "
            f"(gap {min_gap:+.4f} < {DIVERGENCE_GAP_MIN}). Per the locked spec, this suggests "
            f"something is wrong with the leverage computation. Stopping before the permutation "
            f"step rather than proceeding."
        )
        return
    print("  Divergence check passed at all 5 timepoints; proceeding to permutations.\n")

    print(f"=== PERMUTATION TEST: timing check, hour={TIMING_HOUR}, {TIMING_N_PERMUTATIONS} permutations ===")
    y_timing = h_orth_std_by_hour[TIMING_HOUR]
    ss_tot_timing = ss_tot_by_hour[TIMING_HOUR]
    rng = np.random.default_rng(RANDOM_SEED)
    t0 = time.perf_counter()
    for _ in range(TIMING_N_PERMUTATIONS):
        perm_idx = rng.permutation(n_used)
        design_perm = design[perm_idx]
        h_perm = h_diag[perm_idx]
        b_perm = xtx_pinv @ design_perm.T @ y_timing
        y_hat_perm = design_perm @ b_perm
        residual_perm = y_timing - y_hat_perm
        loo_residual_perm = residual_perm / (1.0 - h_perm)[:, None]
        press_perm = float(np.sum(loo_residual_perm ** 2))
        _ = 1.0 - press_perm / ss_tot_timing
    t_timing_block = time.perf_counter() - t0
    t_per_permutation = t_timing_block / TIMING_N_PERMUTATIONS
    print(f"  {TIMING_N_PERMUTATIONS} permutations took {t_timing_block:.3f}s "
          f"({t_per_permutation * 1000:.2f} ms/permutation)")

    est_total = BONFERRONI_TESTS * N_PERMUTATIONS_FULL * t_per_permutation
    print(f"  extrapolated total at {N_PERMUTATIONS_FULL} permutations x {BONFERRONI_TESTS} timepoints: "
          f"{est_total:.1f}s ({est_total / 60:.2f} min)")

    if est_total > TIME_BUDGET_SECONDS:
        print(
            f"\nSTOP: extrapolated total ({est_total / 60:.2f} min) exceeds the "
            f"{TIME_BUDGET_SECONDS / 60:.0f} min budget. Per the locked spec, stopping for "
            f"confirmation before choosing a lower permutation count rather than picking one "
            f"autonomously."
        )
        return
    print(f"  Under the {TIME_BUDGET_SECONDS / 60:.0f} min budget; proceeding directly to "
          f"{N_PERMUTATIONS_FULL} permutations per timepoint, as pre-authorized.\n")

    print(f"=== Permutation grid: {BONFERRONI_TESTS} timepoints x {N_PERMUTATIONS_FULL} permutations ===")
    rows = []
    for hour in HOURS:
        y = h_orth_std_by_hour[hour]
        ss_tot = ss_tot_by_hour[hour]
        observed = r2_loo_by_hour[hour]
        seed = RANDOM_SEED * 1000 + hour
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        permuted = np.empty(N_PERMUTATIONS_FULL, dtype=float)
        for p in range(N_PERMUTATIONS_FULL):
            perm_idx = rng.permutation(n_used)
            design_perm = design[perm_idx]
            h_perm = h_diag[perm_idx]
            b_perm = xtx_pinv @ design_perm.T @ y
            y_hat_perm = design_perm @ b_perm
            residual_perm = y - y_hat_perm
            loo_residual_perm = residual_perm / (1.0 - h_perm)[:, None]
            press_perm = float(np.sum(loo_residual_perm ** 2))
            permuted[p] = 1.0 - press_perm / ss_tot
        elapsed = time.perf_counter() - t0
        count_ge = int(np.sum(permuted >= observed))
        p_value = (count_ge + 1) / (N_PERMUTATIONS_FULL + 1)
        pass_001 = bool(p_value < BONFERRONI_ALPHA)
        print(
            f"  hour {hour:>2}: R2_loo_observed={observed:+.4f}  p={p_value:.4f}  "
            f"pass@0.01={pass_001}  [{elapsed:.2f}s]"
        )
        rows.append({
            "hour": hour,
            "R2_insample": r2_insample_by_hour[hour],
            "R2_loo": r2_loo_by_hour[hour],
            "n_permutations": N_PERMUTATIONS_FULL,
            "p_value": p_value,
            "pass_at_0.01": pass_001,
            "max_leverage": max_leverage,
            "n_used": n_used,
            "n_response_columns": n_response_columns_by_hour[hour],  # 35072, except hour 0 (constant columns dropped)
        })

    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_TABLE_PATH, index=False)
    print(f"\nWrote {OUTPUT_TABLE_PATH}")

    print("\n=== Final table ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(results.to_string(index=False))

    print(f"\nBonferroni threshold (5 timepoints): 0.05/5 = {BONFERRONI_ALPHA:.5f}")
    print(f"Timepoints passing at 0.01: {int(results['pass_at_0.01'].sum())}/5")


if __name__ == "__main__":
    main()
