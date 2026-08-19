"""Diagnostic: recompute observed delta R^2 with genuine leave-one-out scoring.

scripts/t2d_confound_permutation_test.py's fit_full_model/fit_baseline_model
fit on all N participants and score on that same N (in-sample R^2). With
N=61 << P=35,072 this lets RidgeCV interpolate almost any target vector
regardless of real signal. This script recomputes ONLY the 30 observed
delta_r2 values with genuine leave-one-out out-of-sample predictions
(fit on N-1, predict the held-out point, repeat for all N, then R^2 on the
pooled held-out predictions) for both the ridge full model and the OLS
baseline. It reuses the same cohort, snapshots, severity control, clinical
targets, and per-cell (h_orth, severity_raw, target) subsets as the original
script (imported, not reconstructed). It does NOT touch the existing
permutation p-values / bootstrap CIs in t2d_confound_permutation_results.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

from t2d_confound_permutation_test import (  # noqa: E402
    GLYCEMIC_FEATURES,
    HOURS,
    OUTPUT_DIR,
    OUTPUT_TABLE_PATH,
    RIDGE_ALPHAS,
    TARGET_COLUMNS,
    build_subset,
    compute_severity,
    load_all_snapshots,
    load_clinical_targets,
    load_cohort,
    orthogonalize,
    standardize,
)

LOO_OUTPUT_PATH = OUTPUT_DIR / "t2d_confound_loo_diagnostic.csv"


def loo_r2(fit_predict, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    n = len(y)
    preds = np.empty(n, dtype=float)
    for i in range(n):
        train_idx = np.delete(np.arange(n), i)
        preds[i] = fit_predict(x[train_idx], y[train_idx], x[i : i + 1])
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot, preds


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> float:
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=None)
    model.fit(x_train, y_train)
    return float(model.predict(x_test)[0])


def ols_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> float:
    model = LinearRegression()
    model.fit(x_train, y_train)
    return float(model.predict(x_test)[0])


def main() -> None:
    print("=== Cohort / snapshots / severity / clinical targets (reused, unchanged) ===")
    cohort = load_cohort()
    cohort_ids = cohort["participant_id"]
    snapshots = load_all_snapshots(cohort_ids)
    severity = compute_severity(cohort_ids)
    severity_valid = ~severity[GLYCEMIC_FEATURES].isna().any(axis=1).to_numpy()
    clinical = load_clinical_targets(cohort_ids)
    severity_raw_full = severity[GLYCEMIC_FEATURES].to_numpy(dtype=float)

    existing = pd.read_csv(OUTPUT_TABLE_PATH)

    print(f"\n=== Genuine leave-one-out recomputation: {len(HOURS) * len(TARGET_COLUMNS)} cells ===")
    rows = []
    combo_index = 0
    t_start = time.perf_counter()
    for hour in HOURS:
        h_raw_full = snapshots[hour]
        for target_name in TARGET_COLUMNS:
            combo_index += 1
            target_full = clinical[target_name].to_numpy(dtype=float)
            h_raw, severity_raw, target, _ = build_subset(
                h_raw_full, severity_raw_full, target_full, severity_valid
            )
            n_used = h_raw.shape[0]

            # Same projection as the original run: fit on the full per-cell subset,
            # not re-derived per LOO fold. Only the regression scoring changes here.
            severity_std = standardize(severity_raw)
            h_orth = orthogonalize(h_raw, severity_std)

            t0 = time.perf_counter()
            full_r2_loo, _ = loo_r2(ridge_fit_predict, h_orth, target)
            baseline_r2_loo, _ = loo_r2(ols_fit_predict, severity_raw, target)
            elapsed = time.perf_counter() - t0
            delta_r2_loo = full_r2_loo - baseline_r2_loo

            existing_row = existing[(existing["hour"] == hour) & (existing["target"] == target_name)].iloc[0]

            print(
                f"[{combo_index}/30] hour={hour:>2} target={target_name:<26} N={n_used:<3} "
                f"delta_r2_insample={existing_row['delta_r2']:+.4f} -> delta_r2_loo={delta_r2_loo:+.4f} "
                f"(full_r2_loo={full_r2_loo:+.4f}, baseline_r2_loo={baseline_r2_loo:+.4f}) "
                f"p={existing_row['p_value']:.4f} [{elapsed:.1f}s]"
            )

            rows.append({
                "target": target_name,
                "hour": hour,
                "delta_r2_insample": existing_row["delta_r2"],
                "delta_r2_loo": delta_r2_loo,
                "full_r2_insample": existing_row["full_r2"],
                "full_r2_loo": full_r2_loo,
                "baseline_r2_insample": existing_row["baseline_r2"],
                "baseline_r2_loo": baseline_r2_loo,
                "p_value": existing_row["p_value"],
                "bonferroni_pass": existing_row["bonferroni_pass"],
                "ci_low_insample": existing_row["ci_low"],
                "ci_high_insample": existing_row["ci_high"],
                "n_used": n_used,
            })
    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal LOO recomputation time: {total_elapsed:.1f}s")

    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(LOO_OUTPUT_PATH, index=False)
    print(f"Wrote {LOO_OUTPUT_PATH}")

    print("\n=== Pivoted view: delta_r2_loo [hour x target] ===")
    pivot_loo = results.pivot(index="hour", columns="target", values="delta_r2_loo").reindex(
        columns=TARGET_COLUMNS
    )
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(pivot_loo.to_string())

    print("\n=== Pivoted view: delta_r2_insample [hour x target] (for comparison) ===")
    pivot_insample = results.pivot(index="hour", columns="target", values="delta_r2_insample").reindex(
        columns=TARGET_COLUMNS
    )
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(pivot_insample.to_string())


if __name__ == "__main__":
    main()
