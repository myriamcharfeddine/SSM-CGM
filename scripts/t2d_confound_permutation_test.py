"""Confound-controlled permutation test for clinical-feature signal in h_t.

Locked spec: tests, for the T2D oral non-insulin subtype only, whether raw
(no PCA, no truncation) h_t carries signal for six baseline clinical factors
beyond what is already explained by 48h glycemic severity
[mean_cgm, time_in_range, time_above_180], at five stream snapshots
(0/6/12/24/48h). Confound control is a linear projection of severity out of
h_t; the resulting delta R^2 (ridge full model minus OLS severity-only
baseline) is tested against a Freedman-Lane permutation null (1000 draws per
timepoint x target, Bonferroni-corrected across the 5x6=30 tests) and given a
separate participant-bootstrap percentile CI (1000 draws) as an effect-size
interval. See task instructions for the full locked spec; do not alter the
target list, covariate set, ridge form, or permutation procedure based on
intermediate results.

Reuses compute_glycemic_features from scripts/global_realignment_phase1.py
and load_participant_table from scripts/within_subtype_phase1.py verbatim
(imported, not reimplemented).

Usage:
    python scripts/t2d_confound_permutation_test.py            # timing check only
    python scripts/t2d_confound_permutation_test.py --run-full # full 30-run grid
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from global_realignment_phase1 import (  # noqa: E402
    GLYCEMIC_FEATURES,
    GLYCEMIC_WINDOW_HOURS,
    HYPER_THRESHOLD_MGDL,
    TIR_HIGH_MGDL,
    TIR_LOW_MGDL,
    compute_glycemic_features,
)
from within_subtype_phase1 import load_participant_table  # noqa: E402

from ssmcgm.analysis.within_subtype_config import SOURCE_FACTOR_COLUMNS  # noqa: E402

# ---------------------------------------------------------------------------
# Named constants (locked spec)
# ---------------------------------------------------------------------------
MATCHED_PANEL_PATH = (
    REPO_ROOT
    / "outputs/static_phenotype_trajectory_stratified_v2/global_realignment"
    / "matched_panel_participant_ids.csv"
)
TARGET_SUBTYPE = "T2D oral non-insulin"
EXPECTED_N_APPROX = 61
N_TOLERANCE = 5  # "materially different" guard band around the expected N

H0_PATH = REPO_ROOT / "outputs/static_phenotype_trajectory/step2/h0_matrix.parquet"
SNAPSHOT_DIR = (
    REPO_ROOT
    / "outputs/static_phenotype_trajectory_stratified_v2/phase4_time_resolved_extension/snapshots"
)
SNAPSHOT_PATHS = {
    0: H0_PATH,
    6: SNAPSHOT_DIR / "h_t_full_hour06.parquet",
    12: SNAPSHOT_DIR / "h_t_full_hour12.parquet",
    24: SNAPSHOT_DIR / "h_t_full_hour24.parquet",
    48: SNAPSHOT_DIR / "h_t_full_hour48.parquet",
}
HOURS = [0, 6, 12, 24, 48]
EXPECTED_H_DIM = 35_072
PARTICIPANT_ID_COLUMN = "participant_id"

TARGET_COLUMNS = [
    "participants_age",
    "bmi_baseline",
    "hba1c_percent_baseline",
    "c_peptide_ngml_baseline",
    "tg_hdl_ratio",
    "waist_to_hip_ratio_baseline",
]
TG_COLUMN, HDL_COLUMN = SOURCE_FACTOR_COLUMNS["tg_hdl_ratio"]

MIN_N_FOR_POWER = 40  # below this after listwise deletion, flag as underpowered

RIDGE_ALPHAS = np.logspace(-2, 8, 50)  # broad grid; RidgeCV selects via built-in LOO

N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
BONFERRONI_TESTS = len(HOURS) * len(TARGET_COLUMNS)  # 30
BONFERRONI_ALPHA = 0.05 / BONFERRONI_TESTS
RANDOM_SEED = 42

OUTPUT_DIR = (
    REPO_ROOT / "outputs/static_phenotype_trajectory_stratified_v2/confound_permutation_test"
)
OUTPUT_TABLE_PATH = OUTPUT_DIR / "t2d_confound_permutation_results.csv"

TIMING_TARGET = "c_peptide_ngml_baseline"
TIMING_HOUR = 48
TIMING_N_PERMUTATIONS = 50
TIMING_N_BOOTSTRAP = 50


# ---------------------------------------------------------------------------
# Cohort and hidden-state loading
# ---------------------------------------------------------------------------
def load_cohort() -> pd.DataFrame:
    panel = pd.read_csv(MATCHED_PANEL_PATH)
    panel[PARTICIPANT_ID_COLUMN] = panel[PARTICIPANT_ID_COLUMN].astype(str)
    cohort = (
        panel[panel["diagnostic_subtype"] == TARGET_SUBTYPE]
        .sort_values(PARTICIPANT_ID_COLUMN)
        .reset_index(drop=True)
    )
    n = len(cohort)
    print(f"Cohort N = {n} (diagnostic_subtype == {TARGET_SUBTYPE!r}, from {MATCHED_PANEL_PATH})")
    if abs(n - EXPECTED_N_APPROX) > N_TOLERANCE:
        raise RuntimeError(
            f"Cohort N={n} is materially different from the expected ~{EXPECTED_N_APPROX} "
            f"(tolerance +/-{N_TOLERANCE}). Stopping per the locked spec's stop condition "
            "instead of proceeding on an unexpected cohort."
        )
    return cohort


def load_snapshot_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    ids = df[PARTICIPANT_ID_COLUMN].astype(str).to_numpy()
    dim_cols = sorted((c for c in df.columns if c.isdigit()), key=int)
    if len(dim_cols) != EXPECTED_H_DIM:
        raise RuntimeError(
            f"{path.name}: expected {EXPECTED_H_DIM} hidden-state columns, found {len(dim_cols)}. "
            "Stopping rather than proceeding on an unexpected dimensionality."
        )
    matrix = df[dim_cols].to_numpy(dtype=np.float64)
    return ids, matrix


def align_to_cohort(ids: np.ndarray, matrix: np.ndarray, cohort_ids: pd.Series) -> np.ndarray:
    id_to_row = {pid: i for i, pid in enumerate(ids)}
    positions = np.array([id_to_row[pid] for pid in cohort_ids], dtype=int)
    return matrix[positions]


def load_all_snapshots(cohort_ids: pd.Series) -> dict[int, np.ndarray]:
    snapshots = {}
    for hour in HOURS:
        path = SNAPSHOT_PATHS[hour]
        ids, matrix = load_snapshot_matrix(path)
        aligned = align_to_cohort(ids, matrix, cohort_ids)
        print(f"  hour {hour:>2}: {path.name} -> aligned shape {aligned.shape} (dim check passed)")
        snapshots[hour] = aligned
    return snapshots


# ---------------------------------------------------------------------------
# Severity control and clinical targets
# ---------------------------------------------------------------------------
def compute_severity(cohort_ids: pd.Series) -> pd.DataFrame:
    glycemic = compute_glycemic_features(cohort_ids)  # reused verbatim, unmodified
    n_missing = int(glycemic[GLYCEMIC_FEATURES].isna().any(axis=1).sum())
    if n_missing:
        print(
            f"Warning: {n_missing} T2D cohort participants have missing glycemic features "
            f"(no CGM in the {GLYCEMIC_WINDOW_HOURS}h window); excluded from the severity control."
        )
    print(
        f"Severity features (TIR_LOW={TIR_LOW_MGDL}, TIR_HIGH={TIR_HIGH_MGDL}, "
        f"HYPER_THRESHOLD={HYPER_THRESHOLD_MGDL}, WINDOW={GLYCEMIC_WINDOW_HOURS}h) computed for "
        f"{len(glycemic) - n_missing}/{len(glycemic)} cohort participants."
    )
    return glycemic


def load_clinical_targets(cohort_ids: pd.Series) -> pd.DataFrame:
    columns = [
        PARTICIPANT_ID_COLUMN,
        "participants_age",
        "bmi_baseline",
        "hba1c_percent_baseline",
        "c_peptide_ngml_baseline",
        TG_COLUMN,
        HDL_COLUMN,
        "waist_to_hip_ratio_baseline",
    ]
    table, conflicts = load_participant_table(columns)
    if any(conflicts.values()):
        raise RuntimeError(
            f"load_participant_table found conflicting duplicate values, stopping: {conflicts}"
        )
    table["tg_hdl_ratio"] = table[TG_COLUMN] / table[HDL_COLUMN]
    cohort_frame = pd.DataFrame({PARTICIPANT_ID_COLUMN: cohort_ids})
    merged = cohort_frame.merge(
        table[[PARTICIPANT_ID_COLUMN] + TARGET_COLUMNS], on=PARTICIPANT_ID_COLUMN, how="left"
    )
    return merged


def report_missingness(clinical: pd.DataFrame, n_cohort: int) -> dict[str, int]:
    print(f"\nClinical-target missingness in the {TARGET_SUBTYPE} cohort (N={n_cohort}):")
    n_available = {}
    for target in TARGET_COLUMNS:
        n_missing = int(clinical[target].isna().sum())
        n_avail = n_cohort - n_missing
        n_available[target] = n_avail
        flag = " -- UNDERPOWERED after listwise deletion" if n_avail < MIN_N_FOR_POWER else ""
        print(f"  {target}: missing {n_missing}/{n_cohort}, N after listwise deletion = {n_avail}{flag}")
    return n_available


# ---------------------------------------------------------------------------
# Pipeline primitives
# ---------------------------------------------------------------------------
def standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)
    return (matrix - mean) / std


def orthogonalize(h: np.ndarray, severity_std: np.ndarray) -> np.ndarray:
    gram = severity_std.T @ severity_std
    hat = severity_std @ np.linalg.pinv(gram) @ severity_std.T
    return h - hat @ h


def fit_full_model(h_orth: np.ndarray, target: np.ndarray) -> float:
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=None)  # cv=None -> efficient built-in LOO
    model.fit(h_orth, target)
    return float(model.score(h_orth, target))


def fit_baseline_model(severity_raw: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    model = LinearRegression()
    model.fit(severity_raw, target)
    fitted = model.predict(severity_raw)
    resid = target - fitted
    r2 = float(model.score(severity_raw, target))
    return r2, fitted, resid


def build_subset(
    h_raw: np.ndarray,
    severity_raw: np.ndarray,
    target: np.ndarray,
    severity_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = severity_valid & ~np.isnan(target)
    return h_raw[valid], severity_raw[valid], target[valid], valid


def run_permutation_test(
    h_orth: np.ndarray, severity_raw: np.ndarray, target: np.ndarray, n_permutations: int, seed: int
) -> tuple[float, float, float, float, np.ndarray]:
    """Returns (delta_r2_observed, full_r2_observed, baseline_r2_observed, p_value, delta_r2_permuted)."""
    full_r2_observed = fit_full_model(h_orth, target)
    baseline_r2_observed, target_hat, resid = fit_baseline_model(severity_raw, target)
    delta_r2_observed = full_r2_observed - baseline_r2_observed

    rng = np.random.default_rng(seed)
    delta_r2_permuted = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        resid_star = rng.permutation(resid)
        target_star = target_hat + resid_star
        full_r2_perm = fit_full_model(h_orth, target_star)
        baseline_r2_perm, _, _ = fit_baseline_model(severity_raw, target_star)
        delta_r2_permuted[i] = full_r2_perm - baseline_r2_perm

    count_ge = int(np.sum(delta_r2_permuted >= delta_r2_observed))
    p_value = (count_ge + 1) / (n_permutations + 1)
    return delta_r2_observed, full_r2_observed, baseline_r2_observed, p_value, delta_r2_permuted


def run_bootstrap_ci(
    h_raw: np.ndarray, severity_raw: np.ndarray, target: np.ndarray, n_bootstrap: int, seed: int
) -> tuple[float, float, np.ndarray]:
    n = h_raw.shape[0]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        h_boot = h_raw[idx]
        severity_raw_boot = severity_raw[idx]
        target_boot = target[idx]
        severity_std_boot = standardize(severity_raw_boot)
        h_orth_boot = orthogonalize(h_boot, severity_std_boot)
        full_r2_boot = fit_full_model(h_orth_boot, target_boot)
        baseline_r2_boot, _, _ = fit_baseline_model(severity_raw_boot, target_boot)
        deltas[i] = full_r2_boot - baseline_r2_boot
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return float(ci_low), float(ci_high), deltas


# ---------------------------------------------------------------------------
# Timing check
# ---------------------------------------------------------------------------
def timing_check(
    snapshots: dict[int, np.ndarray],
    severity: pd.DataFrame,
    clinical: pd.DataFrame,
    severity_valid: np.ndarray,
) -> None:
    print(
        f"\n=== TIMING CHECK: hour={TIMING_HOUR}, target={TIMING_TARGET}, "
        f"{TIMING_N_PERMUTATIONS} permutations, {TIMING_N_BOOTSTRAP} bootstrap draws ==="
    )
    h_raw_full = snapshots[TIMING_HOUR]
    severity_raw_full = severity[GLYCEMIC_FEATURES].to_numpy(dtype=float)
    target_full = clinical[TIMING_TARGET].to_numpy(dtype=float)

    h_raw, severity_raw, target, _ = build_subset(h_raw_full, severity_raw_full, target_full, severity_valid)
    n_used = h_raw.shape[0]
    print(f"  subset N = {n_used}")

    severity_std = standardize(severity_raw)

    t0 = time.perf_counter()
    h_orth = orthogonalize(h_raw, severity_std)
    t_projection = time.perf_counter() - t0

    t0 = time.perf_counter()
    delta_observed, full_r2, baseline_r2, p_value, _ = run_permutation_test(
        h_orth, severity_raw, target, TIMING_N_PERMUTATIONS, seed=RANDOM_SEED
    )
    t_permutation_block = time.perf_counter() - t0
    t_per_permutation = t_permutation_block / TIMING_N_PERMUTATIONS

    t0 = time.perf_counter()
    ci_low, ci_high, _ = run_bootstrap_ci(h_raw, severity_raw, target, TIMING_N_BOOTSTRAP, seed=RANDOM_SEED)
    t_bootstrap_block = time.perf_counter() - t0
    t_per_bootstrap = t_bootstrap_block / TIMING_N_BOOTSTRAP

    print(f"  projection time (one-off per timepoint): {t_projection:.4f}s")
    print(f"  observed delta_R2 = {delta_observed:.4f} (full R2={full_r2:.4f}, baseline R2={baseline_r2:.4f})")
    print(f"  {TIMING_N_PERMUTATIONS} permutations took {t_permutation_block:.2f}s "
          f"({t_per_permutation * 1000:.1f} ms/permutation)")
    print(f"  {TIMING_N_BOOTSTRAP} bootstrap draws took {t_bootstrap_block:.2f}s "
          f"({t_per_bootstrap * 1000:.1f} ms/draw)")
    print(f"  provisional p-value from only {TIMING_N_PERMUTATIONS} permutations "
          f"(not the final estimate): {p_value:.4f}")

    est_permutation_total = BONFERRONI_TESTS * N_PERMUTATIONS * t_per_permutation
    est_bootstrap_total = BONFERRONI_TESTS * N_BOOTSTRAP * t_per_bootstrap
    est_projection_total = BONFERRONI_TESTS * t_projection
    est_total = est_permutation_total + est_bootstrap_total + est_projection_total

    print(f"\n  Extrapolated full grid ({BONFERRONI_TESTS} runs x {N_PERMUTATIONS} permutations "
          f"+ {N_BOOTSTRAP} bootstrap draws):")
    print(f"    permutation time:  {est_permutation_total:.1f}s ({est_permutation_total / 60:.1f} min)")
    print(f"    bootstrap time:    {est_bootstrap_total:.1f}s ({est_bootstrap_total / 60:.1f} min)")
    print(f"    projection time:   {est_projection_total:.1f}s ({est_projection_total / 60:.1f} min)")
    print(f"    ESTIMATED TOTAL:   {est_total:.1f}s ({est_total / 60:.1f} min)")


# ---------------------------------------------------------------------------
# Full grid
# ---------------------------------------------------------------------------
def run_full_grid(
    snapshots: dict[int, np.ndarray],
    severity: pd.DataFrame,
    clinical: pd.DataFrame,
    severity_valid: np.ndarray,
    n_available: dict[str, int],
) -> pd.DataFrame:
    severity_raw_full = severity[GLYCEMIC_FEATURES].to_numpy(dtype=float)
    rows = []
    combo_index = 0
    for hour in HOURS:
        h_raw_full = snapshots[hour]
        for target_name in TARGET_COLUMNS:
            combo_index += 1
            target_full = clinical[target_name].to_numpy(dtype=float)
            h_raw, severity_raw, target, valid = build_subset(
                h_raw_full, severity_raw_full, target_full, severity_valid
            )
            n_used = h_raw.shape[0]
            underpowered = n_used < MIN_N_FOR_POWER

            severity_std = standardize(severity_raw)
            h_orth = orthogonalize(h_raw, severity_std)

            seed = RANDOM_SEED * 1000 + hour * 10 + TARGET_COLUMNS.index(target_name)

            t0 = time.perf_counter()
            delta_observed, full_r2, baseline_r2, p_value, _ = run_permutation_test(
                h_orth, severity_raw, target, N_PERMUTATIONS, seed=seed
            )
            ci_low, ci_high, _ = run_bootstrap_ci(h_raw, severity_raw, target, N_BOOTSTRAP, seed=seed + 1)
            elapsed = time.perf_counter() - t0

            bonferroni_pass = bool(p_value < BONFERRONI_ALPHA)
            print(
                f"[{combo_index}/{BONFERRONI_TESTS}] hour={hour:>2} target={target_name:<26} "
                f"N={n_used:<3} delta_R2={delta_observed:+.4f} CI=[{ci_low:+.4f},{ci_high:+.4f}] "
                f"p={p_value:.4f} bonferroni_pass={bonferroni_pass} "
                f"{'(UNDERPOWERED)' if underpowered else ''} [{elapsed:.1f}s]"
            )

            rows.append({
                "target": target_name,
                "hour": hour,
                "delta_r2": delta_observed,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "bonferroni_pass": bonferroni_pass,
                "full_r2": full_r2,
                "baseline_r2": baseline_r2,
                "n_used": n_used,
                "n_cohort": len(clinical),
                "n_missing_target": len(clinical) - n_available[target_name],
                "underpowered": underpowered,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-full", action="store_true",
        help="Run the full 30-run grid (1000 permutations + 1000 bootstrap draws each) "
             "after the timing check. Without this flag, only the timing check runs.",
    )
    args = parser.parse_args()

    print("=== Cohort ===")
    cohort = load_cohort()
    cohort_ids = cohort[PARTICIPANT_ID_COLUMN]

    print("\n=== Hidden-state snapshots ===")
    snapshots = load_all_snapshots(cohort_ids)

    print("\n=== Severity control (compute_glycemic_features, reused verbatim) ===")
    severity = compute_severity(cohort_ids)
    severity_valid = ~severity[GLYCEMIC_FEATURES].isna().any(axis=1).to_numpy()

    print("\n=== Clinical targets (load_participant_table, reused verbatim) ===")
    clinical = load_clinical_targets(cohort_ids)
    n_available = report_missingness(clinical, len(cohort))

    timing_check(snapshots, severity, clinical, severity_valid)

    if not args.run_full:
        print(
            "\nTiming check complete. Re-run with --run-full to execute the entire "
            f"{BONFERRONI_TESTS}-run grid ({N_PERMUTATIONS} permutations + {N_BOOTSTRAP} "
            "bootstrap draws per run)."
        )
        return

    print(f"\n=== Full grid: {BONFERRONI_TESTS} runs ===")
    results = run_full_grid(snapshots, severity, clinical, severity_valid, n_available)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_TABLE_PATH, index=False)
    print(f"\nWrote {OUTPUT_TABLE_PATH}")

    print("\n=== Pivoted view: delta_R2 [hour x target] ===")
    pivot = results.pivot(index="hour", columns="target", values="delta_r2").reindex(
        columns=TARGET_COLUMNS
    )
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(pivot.to_string())

    print(f"\nBonferroni-corrected threshold: 0.05 / {BONFERRONI_TESTS} = {BONFERRONI_ALPHA:.5f}")
    print(f"Tests surviving correction: {int(results['bonferroni_pass'].sum())}/{len(results)}")

    print(
        "\nInterpretive note: hba1c_percent_baseline is expected to show near-zero delta R^2 by "
        "construction, since it measures overlapping information to the severity control (48h "
        "glycemic exposure) at a different time resolution (roughly a 3-month average). A null "
        "there should be read as expected, not as evidence against the hypothesis. This is an "
        "interpretive note only; it does not change the analysis or exclude the target."
    )


if __name__ == "__main__":
    main()
