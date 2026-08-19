"""Single-component canonical correlation, clinical features vs severity-
residualized h_t, T2D oral non-insulin, all five stream snapshots.

Exploratory and secondary to the regression result already obtained (0/30
Bonferroni-significant in scripts/t2d_confound_permutation_test.py). Locked
spec: do not switch to multi-component or full kernel CCA, do not drop
timepoints, do not report an in-sample correlation as the result (that
exact failure mode -- RidgeCV in-sample R^2 standing in for the effect size
-- already happened once in this pipeline; see
scripts/t2d_confound_loo_diagnostic.py).

API note: this sklearn version (1.7.2) deprecated PLSCanonical's fitted
x_scores_/y_scores_ attributes. In-sample scores are obtained instead via
model.transform(X_train, Y_train) on the training data, which is mechanically
identical to what x_scores_/y_scores_ held in older sklearn -- not a change
to the method.

Reuses (imported, not reconstructed): cohort loading, snapshot loading,
severity control, clinical-target loading, and the orthogonalize()/
standardize() functions from scripts/t2d_confound_permutation_test.py. Note
that script never persisted its H_orth matrices to disk (they were computed
in-memory per grid cell and discarded); this script regenerates them via the
identical, deterministic orthogonalize() call on the same cached h0/h_t
snapshots and the same severity control, which reproduces them exactly --
this is reported explicitly rather than silently assumed.

Usage:
    python scripts/t2d_confound_cca_analysis.py                    # diagnostic + in-sample/OOS check only
    python scripts/t2d_confound_cca_analysis.py --run-permutations # full pipeline including permutation grid
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.cross_decomposition import PLSCanonical

from t2d_confound_permutation_test import (  # noqa: E402
    GLYCEMIC_FEATURES,
    HOURS,
    OUTPUT_DIR,
    TARGET_COLUMNS,
    build_subset,
    compute_severity,
    load_all_snapshots,
    load_clinical_targets,
    load_cohort,
    orthogonalize,
    standardize,
)

CCA_OUTPUT_PATH = OUTPUT_DIR / "t2d_confound_cca_results.csv"

OUTLIER_Z_THRESHOLD = 3.0
DIVERGENCE_GAP_MIN = 0.10  # in-sample corr expected near-ceiling; OOS must sit at least this much lower

N_PERMUTATIONS_CANDIDATES = [1000, 500, 200]
TIME_BUDGET_SECONDS = 25 * 60
TIMING_HOUR = 48
TIMING_N_PERMUTATIONS = 50
BONFERRONI_TESTS_CCA = len(HOURS)  # 5
BONFERRONI_ALPHA_CCA = 0.05 / BONFERRONI_TESTS_CCA
REGRESSION_ALPHA_FOR_COMPARISON = 0.05 / 30  # the regression run's threshold, comparison only
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Pre-fit diagnostic
# ---------------------------------------------------------------------------
def pre_fit_diagnostic(clinical_raw: pd.DataFrame, cohort_ids: pd.Series) -> np.ndarray:
    print("=== PRE-FIT DIAGNOSTIC: standardized clinical column stats ===")
    raw = clinical_raw[TARGET_COLUMNS].to_numpy(dtype=float)
    std = standardize(raw)
    flagged_any = False
    for j, col in enumerate(TARGET_COLUMNS):
        col_std = std[:, j]
        col_raw = raw[:, j]
        print(
            f"  {col:<26} mean={col_std.mean():+.4f} std={col_std.std(ddof=0):.4f} "
            f"min={col_std.min():+.4f} max={col_std.max():+.4f} skew={skew(col_std):+.4f}"
        )
        max_abs_idx = int(np.argmax(np.abs(col_std)))
        if np.abs(col_std[max_abs_idx]) > OUTLIER_Z_THRESHOLD:
            flagged_any = True
            pid = cohort_ids.iloc[max_abs_idx]
            print(
                f"    FLAGGED: |z|={np.abs(col_std[max_abs_idx]):.3f} > {OUTLIER_Z_THRESHOLD} "
                f"at participant_id={pid}, raw value={col_raw[max_abs_idx]:.4f}"
            )
    if not flagged_any:
        print("  No column exceeds |z|=3 in this cohort.")
    print()
    return std


# ---------------------------------------------------------------------------
# H_orth reconstruction (reused function, not a new projection formula)
# ---------------------------------------------------------------------------
def build_h_orth_per_hour(
    snapshots: dict[int, np.ndarray], severity_raw_full: np.ndarray, severity_valid: np.ndarray
) -> dict[int, np.ndarray]:
    print(
        "=== H_orth reconstruction ===\n"
        "  Note: scripts/t2d_confound_permutation_test.py did not persist H_orth matrices to "
        "disk (computed in-memory per grid cell, discarded after scoring). Regenerating here via "
        "the identical, imported orthogonalize() call on the same cached snapshots and the same "
        "severity control -- deterministic linear algebra, no fitting or randomness involved, so "
        "this reproduces the original matrices exactly rather than approximating them."
    )
    h_orth_by_hour = {}
    severity_raw = severity_raw_full[severity_valid]
    severity_std = standardize(severity_raw)
    for hour in HOURS:
        h_raw = snapshots[hour][severity_valid]
        h_orth_by_hour[hour] = orthogonalize(h_raw, severity_std)
        print(f"  hour {hour:>2}: H_orth shape {h_orth_by_hour[hour].shape}")
    print()
    return h_orth_by_hour


# ---------------------------------------------------------------------------
# CCA primitives
# ---------------------------------------------------------------------------
def fit_insample(clinical_std: np.ndarray, h_orth: np.ndarray) -> tuple[float, np.ndarray]:
    model = PLSCanonical(n_components=1)
    model.fit(clinical_std, h_orth)
    x_scores, y_scores = model.transform(clinical_std, h_orth)
    corr = float(np.corrcoef(x_scores.ravel(), y_scores.ravel())[0, 1])
    return corr, model


def fit_loo(clinical_std: np.ndarray, h_orth: np.ndarray) -> float:
    n = clinical_std.shape[0]
    x_oos = np.empty(n, dtype=float)
    y_oos = np.empty(n, dtype=float)
    for i in range(n):
        train_idx = np.delete(np.arange(n), i)
        model = PLSCanonical(n_components=1)
        model.fit(clinical_std[train_idx], h_orth[train_idx])
        x_s, y_s = model.transform(clinical_std[i : i + 1], h_orth[i : i + 1])
        x_oos[i] = x_s[0, 0]
        y_oos[i] = y_s[0, 0]
    return float(np.corrcoef(x_oos, y_oos)[0, 1])


def loadings(clinical_raw: pd.DataFrame, model: PLSCanonical, clinical_std: np.ndarray) -> dict[str, float]:
    x_scores = model.transform(clinical_std)  # Y=None -> returns x_scores array directly, not a tuple
    variate = x_scores.ravel()
    out = {}
    for col in TARGET_COLUMNS:
        raw = clinical_raw[col].to_numpy(dtype=float)
        out[col] = float(np.corrcoef(raw, variate)[0, 1])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-permutations", action="store_true")
    parser.add_argument(
        "--n-permutations", type=int, default=None,
        help="Explicit permutation count, applied uniformly across all 5 timepoints. Overrides the "
             "adaptive 1000/500/200 budget selection. Must be explicitly authorized by the user, not "
             "chosen autonomously.",
    )
    args = parser.parse_args()

    print("=== Cohort / snapshots / severity / clinical targets (reused, unchanged) ===")
    cohort = load_cohort()
    cohort_ids = cohort["participant_id"]
    snapshots = load_all_snapshots(cohort_ids)
    severity = compute_severity(cohort_ids)
    severity_valid = ~severity[GLYCEMIC_FEATURES].isna().any(axis=1).to_numpy()
    severity_raw_full = severity[GLYCEMIC_FEATURES].to_numpy(dtype=float)
    clinical = load_clinical_targets(cohort_ids)
    clinical_valid = ~clinical[TARGET_COLUMNS].isna().any(axis=1).to_numpy()
    cca_valid = severity_valid & clinical_valid
    n_cca = int(cca_valid.sum())
    print(f"CCA cohort N = {n_cca} (severity-valid AND all-6-clinical-complete)\n")

    clinical_cca = clinical.loc[cca_valid].reset_index(drop=True)
    cca_cohort_ids = cohort_ids[cca_valid].reset_index(drop=True)

    clinical_std = pre_fit_diagnostic(clinical_cca, cca_cohort_ids)

    h_orth_by_hour = build_h_orth_per_hour(snapshots, severity_raw_full, cca_valid)

    print("=== TEST STATISTIC: in-sample vs leave-one-out out-of-sample, all 5 timepoints ===")
    insample_corrs = {}
    oos_corrs = {}
    insample_models = {}
    for hour in HOURS:
        h_orth = h_orth_by_hour[hour]
        t0 = time.perf_counter()
        insample_corr, model = fit_insample(clinical_std, h_orth)
        t_insample = time.perf_counter() - t0
        t0 = time.perf_counter()
        oos_corr = fit_loo(clinical_std, h_orth)
        t_oos = time.perf_counter() - t0
        gap = abs(insample_corr) - abs(oos_corr)
        insample_corrs[hour] = insample_corr
        oos_corrs[hour] = oos_corr
        insample_models[hour] = model
        print(
            f"  hour {hour:>2}: in-sample={insample_corr:+.4f} (expected to be inflated, not the "
            f"result to use) [{t_insample:.2f}s]  out-of-sample(LOO)={oos_corr:+.4f} [{t_oos:.2f}s]  "
            f"gap={gap:+.4f}"
        )

    min_gap = min(abs(insample_corrs[h]) - abs(oos_corrs[h]) for h in HOURS)
    print(f"\n  Minimum |in-sample| - |OOS| gap across timepoints: {min_gap:+.4f} "
          f"(divergence threshold: {DIVERGENCE_GAP_MIN})")
    if min_gap < DIVERGENCE_GAP_MIN:
        print(
            f"\nSTOP: at least one timepoint's in-sample and out-of-sample correlations do not "
            f"diverge meaningfully (gap {min_gap:+.4f} < {DIVERGENCE_GAP_MIN}). Per the locked "
            f"spec's step-3 stop condition, this suggests the held-out transform may not actually "
            f"be using held-out weights. Stopping before the permutation grid rather than proceeding."
        )
        return
    print("  Divergence check passed at all 5 timepoints; proceeding.\n")

    print("=== LOADINGS: correlation of each raw clinical feature with the full-sample clinical variate ===")
    loading_rows = {}
    for hour in HOURS:
        row = loadings(clinical_cca, insample_models[hour], clinical_std)
        loading_rows[hour] = row
        ranked = sorted(row.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_feature, top_value = ranked[0]
        severity_flavored = top_feature in ("hba1c_percent_baseline", "participants_age")
        physiologically_specific = top_feature in ("c_peptide_ngml_baseline", "tg_hdl_ratio")
        flavor = (
            "severity-flavored (hba1c/age)" if severity_flavored else
            "physiologically specific (c-peptide/tg-hdl)" if physiologically_specific else
            "neither hba1c/age nor c-peptide/tg-hdl"
        )
        print(f"  hour {hour:>2}: " + ", ".join(f"{k}={v:+.3f}" for k, v in row.items()))
        print(f"    dominant: {top_feature} ({top_value:+.3f}) -- {flavor}")
        if top_feature == "tg_hdl_ratio":
            print(
                "    tg_hdl_ratio dominates this loading. It was flagged in the pre-fit diagnostic "
                "or is known from the regression run's LOO recomputation (delta_r2_loo ~ -3.0 at "
                "every timepoint when used as a sole regression target) to be unstable out-of-sample. "
                "This combination means the loading here is likely driven by the same instability, "
                "not a real signal."
            )
    print()

    if not args.run_permutations:
        print(
            "In-sample/OOS check complete, divergence confirmed. Re-run with --run-permutations to "
            "run the timing check and (if the budget allows) the full permutation grid."
        )
        return

    # -----------------------------------------------------------------
    # Permutation test, adaptive count
    # -----------------------------------------------------------------
    print(f"=== TIMING CHECK: hour={TIMING_HOUR}, {TIMING_N_PERMUTATIONS} permutations "
          f"(each a full {n_cca}-fold LOO refit) ===")
    rng = np.random.default_rng(RANDOM_SEED)
    h_orth_timing = h_orth_by_hour[TIMING_HOUR]
    t0 = time.perf_counter()
    for _ in range(TIMING_N_PERMUTATIONS):
        perm_idx = rng.permutation(n_cca)
        fit_loo(clinical_std[perm_idx], h_orth_timing)
    t_timing_block = time.perf_counter() - t0
    t_per_permutation = t_timing_block / TIMING_N_PERMUTATIONS
    print(f"  {TIMING_N_PERMUTATIONS} permutations took {t_timing_block:.1f}s "
          f"({t_per_permutation:.3f}s/permutation)")

    if args.n_permutations is not None:
        chosen_n_permutations = args.n_permutations
        est_total = BONFERRONI_TESTS_CCA * chosen_n_permutations * t_per_permutation
        print(
            f"  n_permutations={chosen_n_permutations} explicitly authorized by the user (overrides "
            f"the adaptive 1000/500/200 budget selection). Estimated total: {est_total:.1f}s "
            f"({est_total / 60:.1f} min)."
        )
    else:
        chosen_n_permutations = None
        for candidate in N_PERMUTATIONS_CANDIDATES:
            est_total = BONFERRONI_TESTS_CCA * candidate * t_per_permutation
            print(f"  extrapolated total at {candidate} permutations x {BONFERRONI_TESTS_CCA} timepoints: "
                  f"{est_total:.1f}s ({est_total / 60:.1f} min)")
            if est_total <= TIME_BUDGET_SECONDS:
                chosen_n_permutations = candidate
                break

        if chosen_n_permutations is None:
            smallest = N_PERMUTATIONS_CANDIDATES[-1]
            est_smallest = BONFERRONI_TESTS_CCA * smallest * t_per_permutation
            print(
                f"\nSTOP: even the smallest pre-authorized permutation count ({smallest}) is "
                f"estimated at {est_smallest / 60:.1f} min, over the {TIME_BUDGET_SECONDS / 60:.0f} min "
                f"budget. The locked spec only pre-authorizes scaling down through 1000 -> 500 -> 200; "
                f"it does not authorize a count below 200. Stopping rather than picking an unauthorized "
                f"count. Reporting for a decision: current 5-timepoint x 200-permutation estimate is "
                f"{est_smallest:.0f}s ({est_smallest / 60:.1f} min)."
            )
            return

    print(f"\n  Using n_permutations={chosen_n_permutations} for all 5 timepoints "
          f"(estimated {BONFERRONI_TESTS_CCA * chosen_n_permutations * t_per_permutation / 60:.1f} min total).\n")

    print(f"=== Permutation grid: {BONFERRONI_TESTS_CCA} timepoints x {chosen_n_permutations} permutations ===")
    rows = []
    for hour in HOURS:
        h_orth = h_orth_by_hour[hour]
        observed = oos_corrs[hour]
        seed = RANDOM_SEED * 1000 + hour
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        permuted = np.empty(chosen_n_permutations, dtype=float)
        for p in range(chosen_n_permutations):
            perm_idx = rng.permutation(n_cca)
            permuted[p] = fit_loo(clinical_std[perm_idx], h_orth)
        elapsed = time.perf_counter() - t0
        count_ge = int(np.sum(permuted >= observed))
        p_value = (count_ge + 1) / (chosen_n_permutations + 1)
        pass_001 = bool(p_value < BONFERRONI_ALPHA_CCA)
        pass_regression_threshold = bool(p_value < REGRESSION_ALPHA_FOR_COMPARISON)
        print(
            f"  hour {hour:>2}: observed(OOS)={observed:+.4f}  p={p_value:.4f}  "
            f"pass@0.01={pass_001}  pass@0.00167={pass_regression_threshold}  [{elapsed:.1f}s]"
        )
        row = {
            "hour": hour,
            "insample_corr": insample_corrs[hour],
            "oos_corr": oos_corrs[hour],
            "n_permutations": chosen_n_permutations,
            "p_value": p_value,
            "pass_at_0.01": pass_001,
            "pass_at_0.00167": pass_regression_threshold,
        }
        row.update({f"loading_{k}": v for k, v in loading_rows[hour].items()})
        rows.append(row)

    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(CCA_OUTPUT_PATH, index=False)
    print(f"\nWrote {CCA_OUTPUT_PATH}")

    print("\n=== Final table ===")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(results.to_string(index=False))

    print(f"\nBonferroni threshold (5 timepoints): 0.05/5 = {BONFERRONI_ALPHA_CCA:.5f}")
    print(f"Regression-comparison threshold (not a formal claim here): {REGRESSION_ALPHA_FOR_COMPARISON:.5f}")
    print(f"Timepoints passing at 0.01: {int(results['pass_at_0.01'].sum())}/5")
    print(f"Timepoints passing at 0.00167: {int(results['pass_at_0.00167'].sum())}/5")


if __name__ == "__main__":
    main()
