"""Two robustness checks on the hour-0 result from
scripts/t2d_confound_h_on_clinical_regression.py. Hour 0 only; hours
6/12/24/48 are untouched. Reuses the exact closed-form LOO regression and
permutation procedure already implemented and validated there (imported,
not reimplemented).

CHECK 1: same N=61 hour-0 fit, n_permutations 1000 -> 5000 (resolution only).
CHECK 2: rebuild the hour-0 pipeline end to end (severity standardization,
clinical standardization, projection, design matrix, leverage, fit,
permutation) on N=58, excluding the three highest-leverage participants
(1355, 4159, 4345) identified in the N=61 leverage diagnostic. Standardization
statistics are recomputed on the N=58 subset itself, not reused from N=61,
since "rerun the full hour-0 pipeline on N=58" means a genuine rebuild, not a
post-hoc row deletion from N=61-fitted matrices.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from t2d_confound_h_on_clinical_regression import (  # noqa: E402
    LEVERAGE_STOP_THRESHOLD,
    RANDOM_SEED,
    build_design,
    fit_and_score,
    hat_diagonal,
)
from t2d_confound_permutation_test import (  # noqa: E402
    GLYCEMIC_FEATURES,
    OUTPUT_DIR,
    TARGET_COLUMNS,
    compute_severity,
    load_all_snapshots,
    load_clinical_targets,
    load_cohort,
    orthogonalize,
    standardize,
)

OUTPUT_TABLE_PATH = OUTPUT_DIR / "t2d_confound_h_on_clinical_hour0_robustness.csv"

HOUR = 0
N_PERMUTATIONS_CHECK1 = 5000
N_PERMUTATIONS_CHECK2 = 1000
BONFERRONI_ALPHA = 0.01  # same threshold as the original per-timepoint table, for direct comparability
EXCLUDED_PARTICIPANT_IDS = ["1355", "4159", "4345"]  # the three highest-leverage participants at N=61


def drop_constant_columns(h_orth: np.ndarray, label: str) -> np.ndarray:
    col_std = h_orth.std(axis=0)
    zero_var_mask = col_std == 0
    n_zero = int(zero_var_mask.sum())
    if n_zero:
        print(f"  {label}: {n_zero}/{h_orth.shape[1]} response columns constant, dropped")
        h_orth = h_orth[:, ~zero_var_mask]
    return h_orth


def run_permutations(design: np.ndarray, y: np.ndarray, h_diag: np.ndarray, ss_tot: float,
                      observed: float, n_permutations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = design.shape[0]
    xtx_pinv = np.linalg.pinv(design.T @ design)
    permuted = np.empty(n_permutations, dtype=float)
    t0 = time.perf_counter()
    for p in range(n_permutations):
        perm_idx = rng.permutation(n)
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
    p_value = (count_ge + 1) / (n_permutations + 1)
    print(f"  {n_permutations} permutations took {elapsed:.2f}s; "
          f"count(permuted >= observed)={count_ge}/{n_permutations}; p_value={p_value:.6f}")
    return p_value, elapsed


def build_hour0_pipeline(cohort_ids: pd.Series, snapshots: dict, severity_raw_full: np.ndarray,
                          severity_valid_full: np.ndarray, clinical_full: pd.DataFrame,
                          clinical_valid_full: np.ndarray, exclude_ids: set[str] | None = None):
    valid = severity_valid_full & clinical_valid_full
    if exclude_ids:
        keep = ~cohort_ids.isin(exclude_ids)
        valid = valid & keep.to_numpy()
    n_used = int(valid.sum())

    clinical_valid_frame = clinical_full.loc[valid].reset_index(drop=True)
    clinical_raw = clinical_valid_frame[TARGET_COLUMNS].to_numpy(dtype=float)
    clinical_std = standardize(clinical_raw)
    design = build_design(clinical_std)

    xtx = design.T @ design
    xtx_pinv = np.linalg.pinv(xtx)
    h_diag = hat_diagonal(design, xtx_pinv)
    p = design.shape[1]
    avg_leverage = p / n_used
    max_leverage = float(h_diag.max())

    severity_raw = severity_raw_full[valid]
    severity_std = standardize(severity_raw)
    h_raw = snapshots[HOUR][valid]
    h_orth = orthogonalize(h_raw, severity_std)
    h_orth = drop_constant_columns(h_orth, f"N={n_used}")
    y = standardize(h_orth)
    assert not np.any(np.isnan(y)), "NaN survived column drop"

    r2_insample, r2_loo, ss_tot = fit_and_score(design, y, xtx_pinv, h_diag)
    return {
        "n_used": n_used, "avg_leverage": avg_leverage, "max_leverage": max_leverage,
        "design": design, "y": y, "h_diag": h_diag, "ss_tot": ss_tot,
        "r2_insample": r2_insample, "r2_loo": r2_loo,
    }


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

    rows = []

    # -----------------------------------------------------------------
    # CHECK 1: N=61, same fit, higher permutation resolution
    # -----------------------------------------------------------------
    print("\n=== CHECK 1: permutation resolution, N=61, hour 0, n_permutations=5000 ===")
    n61 = build_hour0_pipeline(cohort_ids, snapshots, severity_raw_full, severity_valid, clinical, clinical_valid)
    print(f"  N={n61['n_used']}  R2_insample={n61['r2_insample']:+.4f}  R2_loo={n61['r2_loo']:+.4f}")
    p_check1, elapsed_check1 = run_permutations(
        n61["design"], n61["y"], n61["h_diag"], n61["ss_tot"], n61["r2_loo"],
        N_PERMUTATIONS_CHECK1, seed=RANDOM_SEED * 1000 + HOUR,
    )
    rows.append({
        "check": "N=61 original (n_permutations=5000)",
        "n_used": n61["n_used"],
        "R2_insample": n61["r2_insample"],
        "R2_loo": n61["r2_loo"],
        "n_permutations": N_PERMUTATIONS_CHECK1,
        "p_value": p_check1,
        "pass_at_0.01": bool(p_check1 < BONFERRONI_ALPHA),
        "max_leverage": n61["max_leverage"],
    })

    # -----------------------------------------------------------------
    # CHECK 2: N=58, full pipeline rebuild excluding the 3 highest-leverage participants
    # -----------------------------------------------------------------
    print("\n=== CHECK 2: leverage robustness, N=58, hour 0, full pipeline rebuilt ===")
    n58 = build_hour0_pipeline(
        cohort_ids, snapshots, severity_raw_full, severity_valid, clinical, clinical_valid,
        exclude_ids=set(EXCLUDED_PARTICIPANT_IDS),
    )
    print(f"  N={n58['n_used']} (excluded {EXCLUDED_PARTICIPANT_IDS})")
    print(f"  Reduced-design leverage: avg (p/N) = {n58['avg_leverage']:.4f}, max = {n58['max_leverage']:.4f}")
    if n58["max_leverage"] > LEVERAGE_STOP_THRESHOLD:
        print(f"\nSTOP: N=58 max leverage {n58['max_leverage']:.4f} exceeds {LEVERAGE_STOP_THRESHOLD}.")
        return
    print(f"  R2_insample={n58['r2_insample']:+.4f}  R2_loo={n58['r2_loo']:+.4f}")
    p_check2, elapsed_check2 = run_permutations(
        n58["design"], n58["y"], n58["h_diag"], n58["ss_tot"], n58["r2_loo"],
        N_PERMUTATIONS_CHECK2, seed=RANDOM_SEED * 1000 + HOUR,
    )
    rows.append({
        "check": "N=58 leverage-excluded (n_permutations=1000)",
        "n_used": n58["n_used"],
        "R2_insample": n58["r2_insample"],
        "R2_loo": n58["r2_loo"],
        "n_permutations": N_PERMUTATIONS_CHECK2,
        "p_value": p_check2,
        "pass_at_0.01": bool(p_check2 < BONFERRONI_ALPHA),
        "max_leverage": n58["max_leverage"],
    })

    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_TABLE_PATH, index=False)
    print(f"\nWrote {OUTPUT_TABLE_PATH}")
    print("\n=== Final table ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
