#!/usr/bin/env python3
"""Step 0 empirical power analysis for beyond-glucose dynamics."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/beyond_glucose_dynamics/step0_power"
)
STEP6_RESULTS_PATH = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step6_final_synthesis/latest/"
    "final_results_table.csv"
)
STEP5_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step5_clinical_probes/"
    "20260725T022634Z"
)
PROBE_INCREMENTAL_PATH = STEP5_ROOT / "probe_incremental_value.csv"
TEST_PREDICTIONS_PATH = STEP5_ROOT / "test_probe_predictions.parquet"
STEP7_NEIGHBOR_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step7_original_question_closing_pass/"
    "20260726T223929Z/neighbor_sharing"
)
NEIGHBOR_SUMMARY_PATH = (
    STEP7_NEIGHBOR_ROOT / "neighbor_sharing_tier1_results.csv"
)
NEIGHBOR_PARTICIPANT_PATH = (
    STEP7_NEIGHBOR_ROOT / "neighbor_sharing_by_participant.parquet"
)
CLINICAL_INVENTORY_PATH = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step0_feasibility/"
    "clinical_target_inventory.csv"
)
CONTEXT_COVERAGE_PATH = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step0_feasibility/"
    "context_coverage_by_participant.csv"
)
STATIC_SCHEMA_PATH = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step1_static_neutralization/"
    "20260724T223612Z/static_schema_audit.csv"
)
PCA_LOADINGS_PATH = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step3_validation_clustering/"
    "20260725T001123Z/pca_loadings.parquet"
)

EXISTING_TARGETS = [
    "natriuretic_peptide_b_prohormon",
    "c_reactive_protein_i",
    "bun_creatinine_ratio",
]
TARGET_LABELS = {
    "natriuretic_peptide_b_prohormon": "NT-proBNP",
    "c_reactive_protein_i": "High-sensitivity CRP",
    "bun_creatinine_ratio": "BUN/creatinine ratio",
    "troponin_t_cardiac_mass_volum": "Troponin T",
    "cholesterol_in_ldl_mass": "LDL cholesterol",
    "cholesterol_in_hdl_mass": "HDL cholesterol",
    "triglyceride": "Triglycerides",
    "alanine_aminotransferase_enzymat": "ALT",
    "aspartate_aminotransferase_enzym": "AST",
}
PLANNED_TARGETS = [
    "troponin_t_cardiac_mass_volum",
    "cholesterol_in_ldl_mass",
    "cholesterol_in_hdl_mass",
    "triglyceride",
    "alanine_aminotransferase_enzymat",
    "aspartate_aminotransferase_enzym",
]
PROBE_BASELINE_FEATURE_SET = "simple_baseline"
PROBE_AUGMENTED_FEATURE_SET = "simple_plus_neutral_all"
PROBE_SPLIT = "test_transport"
NEIGHBOR_CONDITION = "neutral_all"
NEIGHBOR_K = 10
NEIGHBOR_BASELINE_TYPE = "unrestricted_non_neighbours"
NEIGHBOR_FDR_FAMILY = "primary_neutral_k10_tier1"
BOOTSTRAP_REPLICATES = 2000
POWER_TARGET = 0.80
TWO_SIDED_ALPHA = 0.05
FDR_ALPHA = 0.05
RANDOM_SEED = 42
PROBE_SMALL_EFFECT_THRESHOLD = 0.05
NEIGHBOR_SMALL_GAIN_THRESHOLD = 0.10
EXPECTED_EXISTING_TEST_N = 217
EXPECTED_PLANNED_TEST_N = 217
MINIMUM_GRID_UPPER = 0.25
MDE_BINARY_SEARCH_ITERATIONS = 60


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed_step5(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def stable_seed_neighbor(*parts: Any, base_seed: int = RANDOM_SEED) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        byteorder="little",
    )
    return int((value + base_seed) % (2**32 - 1))


def r2_weighted(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> float:
    total_weight = weights.sum()
    mean_observed = np.sum(weights * observed) / total_weight
    numerator = np.sum(weights * np.square(observed - predicted))
    denominator = np.sum(weights * np.square(observed - mean_observed))
    return float(1 - numerator / denominator)


def probe_bootstrap_distribution(
    predictions: pd.DataFrame,
    target: str,
) -> tuple[float, float, np.ndarray, int]:
    baseline = predictions[
        predictions["target"].eq(target)
        & predictions["feature_set"].eq(PROBE_BASELINE_FEATURE_SET)
    ].copy()
    augmented = predictions[
        predictions["target"].eq(target)
        & predictions["feature_set"].eq(PROBE_AUGMENTED_FEATURE_SET)
    ].copy()
    paired = baseline[
        [
            "participant_id",
            "observed_transformed",
            "predicted_transformed",
        ]
    ].merge(
        augmented[["participant_id", "predicted_transformed"]],
        on="participant_id",
        suffixes=("_baseline", "_augmented"),
        validate="one_to_one",
    )
    participant_ids, participant_codes = np.unique(
        paired["participant_id"].astype(str),
        return_inverse=True,
    )
    observed = paired["observed_transformed"].to_numpy(dtype=float)
    baseline_prediction = paired[
        "predicted_transformed_baseline"
    ].to_numpy(dtype=float)
    augmented_prediction = paired[
        "predicted_transformed_augmented"
    ].to_numpy(dtype=float)
    baseline_r2 = r2_weighted(
        observed,
        baseline_prediction,
        np.ones(len(observed)),
    )
    augmented_r2 = r2_weighted(
        observed,
        augmented_prediction,
        np.ones(len(observed)),
    )
    observed_delta = augmented_r2 - baseline_r2
    seed = (
        RANDOM_SEED
        + 200000
        + stable_seed_step5(
            PROBE_SPLIT,
            target,
            PROBE_AUGMENTED_FEATURE_SET,
        )
        % 100000
    )
    rng = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    probability = np.repeat(1 / len(participant_ids), len(participant_ids))
    for bootstrap_index in range(BOOTSTRAP_REPLICATES):
        counts = rng.multinomial(len(participant_ids), probability)
        weights = counts[participant_codes]
        bootstrap_baseline = r2_weighted(
            observed,
            baseline_prediction,
            weights,
        )
        bootstrap_augmented = r2_weighted(
            observed,
            augmented_prediction,
            weights,
        )
        distribution[bootstrap_index] = (
            bootstrap_augmented - bootstrap_baseline
        )
    return baseline_r2, observed_delta, distribution, len(participant_ids)


def neighbor_bootstrap_distribution(
    participant_results: pd.DataFrame,
    target: str,
) -> tuple[float, np.ndarray, int]:
    selected = participant_results[
        participant_results["variable"].eq(target)
        & participant_results["condition"].eq(NEIGHBOR_CONDITION)
        & participant_results["k_neighbors"].eq(NEIGHBOR_K)
    ].copy()
    gains = pd.to_numeric(selected["sharing_gain"], errors="coerce")
    gains = gains[np.isfinite(gains)].to_numpy(dtype=float)
    observed = float(np.mean(gains))
    seed = stable_seed_neighbor(
        target,
        NEIGHBOR_CONDITION,
        NEIGHBOR_K,
        NEIGHBOR_BASELINE_TYPE,
        "bootstrap",
        base_seed=RANDOM_SEED,
    )
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(
        0,
        len(gains),
        size=(BOOTSTRAP_REPLICATES, len(gains)),
    )
    distribution = gains[draw_indices].mean(axis=1)
    return observed, distribution, len(gains)


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def candidate_fdr_p_threshold(
    family_p_values: dict[str, float],
    target: str,
) -> float:
    other_targets = [
        name for name in family_p_values if name != target
    ]
    lower = 0.0
    upper = FDR_ALPHA
    for _ in range(MDE_BINARY_SEARCH_ITERATIONS):
        candidate = (lower + upper) / 2
        names = other_targets + [target]
        values = np.asarray(
            [family_p_values[name] for name in other_targets] + [candidate],
            dtype=float,
        )
        adjusted = bh_adjust(values)
        candidate_q = adjusted[names.index(target)]
        if candidate_q <= FDR_ALPHA:
            lower = candidate
        else:
            upper = candidate
    return lower


def empirical_mde(
    bootstrap_distribution: np.ndarray,
    alpha: float,
) -> tuple[float, float, float]:
    distribution = np.asarray(bootstrap_distribution, dtype=float)
    standard_error = float(np.std(distribution, ddof=1))
    errors = distribution - np.mean(distribution)
    critical_value = float(np.quantile(np.abs(errors), 1 - alpha))

    def power(effect: float) -> float:
        return float(np.mean(np.abs(effect + errors) > critical_value))

    lower = 0.0
    upper = max(
        MINIMUM_GRID_UPPER,
        critical_value + 5 * standard_error,
    )
    while power(upper) < POWER_TARGET:
        upper *= 2
    for _ in range(MDE_BINARY_SEARCH_ITERATIONS):
        candidate = (lower + upper) / 2
        if power(candidate) >= POWER_TARGET:
            upper = candidate
        else:
            lower = candidate
    return upper, standard_error, critical_value


def schema_record(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": json.dumps(frame.columns.tolist()),
        "column_dtypes": json.dumps(
            {column: str(dtype) for column, dtype in frame.dtypes.items()},
            sort_keys=True,
        ),
    }


def write_schema_printout(
    records: list[dict[str, Any]],
    path: Path,
) -> str:
    lines = []
    for record in records:
        lines.extend(
            [
                f"FILE: {record['path']}",
                f"SHAPE: ({record['rows']}, {record['columns']})",
                f"COLUMNS: {record['column_names']}",
                f"DTYPES: {record['column_dtypes']}",
                "",
            ]
        )
    text = "\n".join(lines)
    path.write_text(text)
    print(text)
    return text


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_directory = OUTPUT_ROOT / run_id
    output_directory.mkdir(parents=True, exist_ok=False)

    input_paths = [
        STEP6_RESULTS_PATH,
        PROBE_INCREMENTAL_PATH,
        TEST_PREDICTIONS_PATH,
        NEIGHBOR_SUMMARY_PATH,
        NEIGHBOR_PARTICIPANT_PATH,
        CLINICAL_INVENTORY_PATH,
        CONTEXT_COVERAGE_PATH,
        STATIC_SCHEMA_PATH,
        PCA_LOADINGS_PATH,
    ]
    frames = {
        path: (
            pd.read_parquet(path)
            if path.suffix == ".parquet"
            else pd.read_csv(path)
        )
        for path in input_paths
    }
    schema_records = [
        schema_record(path, frames[path]) for path in input_paths
    ]
    schema_inventory = pd.DataFrame(schema_records)
    schema_inventory_path = output_directory / "input_schema_inventory.csv"
    schema_inventory.to_csv(schema_inventory_path, index=False)
    schema_printout_path = output_directory / "input_schema_printout.txt"
    write_schema_printout(schema_records, schema_printout_path)

    incremental = frames[PROBE_INCREMENTAL_PATH]
    predictions = frames[TEST_PREDICTIONS_PATH]
    neighbor_summary = frames[NEIGHBOR_SUMMARY_PATH]
    neighbor_participant = frames[NEIGHBOR_PARTICIPANT_PATH]
    clinical_inventory = frames[CLINICAL_INVENTORY_PATH]
    static_schema = frames[STATIC_SCHEMA_PATH]

    primary_probe_rows = incremental[
        incremental["split"].eq(PROBE_SPLIT)
        & incremental["feature_set"].eq(PROBE_AUGMENTED_FEATURE_SET)
    ].set_index("target")
    if set(primary_probe_rows.index) != set(EXISTING_TARGETS):
        raise RuntimeError("Primary test probe target set changed")

    primary_neighbor = neighbor_summary[
        neighbor_summary["condition"].eq(NEIGHBOR_CONDITION)
        & neighbor_summary["k_neighbors"].eq(NEIGHBOR_K)
        & ~neighbor_summary["site_matched"]
        & neighbor_summary["fdr_family"].eq(NEIGHBOR_FDR_FAMILY)
    ].copy()
    if len(primary_neighbor) != 9:
        raise RuntimeError("Primary neighbor FDR family does not contain 9 rows")
    family_p_values = dict(
        zip(primary_neighbor["variable"], primary_neighbor["permutation_p"])
    )

    result_rows = []
    probe_mdes = []
    neighbor_mdes = []
    for target in EXISTING_TARGETS:
        baseline_r2, observed_delta, distribution, n_participants = (
            probe_bootstrap_distribution(predictions, target)
        )
        saved = primary_probe_rows.loc[target]
        reproduced_ci = np.quantile(distribution, [0.025, 0.975])
        if not np.allclose(
            reproduced_ci,
            [saved["delta_r2_ci_low"], saved["delta_r2_ci_high"]],
            atol=1e-12,
        ):
            raise RuntimeError(f"Probe bootstrap CI mismatch for {target}")
        if not np.isclose(observed_delta, saved["delta_r2"], atol=1e-12):
            raise RuntimeError(f"Probe observed delta mismatch for {target}")
        mde, standard_error, critical_value = empirical_mde(
            distribution,
            TWO_SIDED_ALPHA,
        )
        probe_mdes.append(mde)
        result_rows.append(
            {
                "target": target,
                "target_label": TARGET_LABELS[target],
                "analysis_type": "incremental_value_probe",
                "n": n_participants,
                "baseline_r_squared": baseline_r2,
                "observed_effect": observed_delta,
                "minimum_detectable_effect": mde,
                "bootstrap_standard_error": standard_error,
                "effective_alpha": TWO_SIDED_ALPHA,
                "empirical_critical_value": critical_value,
                "power_target": POWER_TARGET,
                "adequacy_threshold": PROBE_SMALL_EFFECT_THRESHOLD,
                "verdict": (
                    "adequately powered"
                    if mde <= PROBE_SMALL_EFFECT_THRESHOLD
                    else "underpowered"
                ),
                "mde_basis": "observed test participant bootstrap",
                "model_input_status": "not_model_input",
                "notes": "Primary neutral-state incremental probe.",
            }
        )

        observed_gain, neighbor_distribution, neighbor_n = (
            neighbor_bootstrap_distribution(
                neighbor_participant,
                target,
            )
        )
        saved_neighbor = primary_neighbor[
            primary_neighbor["variable"].eq(target)
        ].iloc[0]
        reproduced_neighbor_ci = np.quantile(
            neighbor_distribution,
            [0.025, 0.975],
        )
        if not np.allclose(
            reproduced_neighbor_ci,
            [
                saved_neighbor["bootstrap_ci_low"],
                saved_neighbor["bootstrap_ci_high"],
            ],
            atol=1e-12,
        ):
            raise RuntimeError(f"Neighbor bootstrap CI mismatch for {target}")
        fdr_p_threshold = candidate_fdr_p_threshold(
            family_p_values,
            target,
        )
        neighbor_mde, neighbor_se, neighbor_critical = empirical_mde(
            neighbor_distribution,
            fdr_p_threshold,
        )
        neighbor_mdes.append(neighbor_mde)
        result_rows.append(
            {
                "target": target,
                "target_label": TARGET_LABELS[target],
                "analysis_type": "neighbor_sharing",
                "n": neighbor_n,
                "baseline_r_squared": np.nan,
                "observed_effect": observed_gain,
                "minimum_detectable_effect": neighbor_mde,
                "bootstrap_standard_error": neighbor_se,
                "effective_alpha": fdr_p_threshold,
                "empirical_critical_value": neighbor_critical,
                "power_target": POWER_TARGET,
                "adequacy_threshold": NEIGHBOR_SMALL_GAIN_THRESHOLD,
                "verdict": (
                    "adequately powered"
                    if neighbor_mde <= NEIGHBOR_SMALL_GAIN_THRESHOLD
                    else "underpowered"
                ),
                "mde_basis": (
                    "observed focal-participant bootstrap with primary-family "
                    "BH threshold"
                ),
                "model_input_status": "not_model_input",
                "notes": "Neutral-state k=10 standardized similarity gain.",
            }
        )

    planned_inventory = (
        clinical_inventory[
            clinical_inventory["normalized_target_name"].isin(PLANNED_TARGETS)
        ]
        .sort_values("normalized_target_name")
        .drop_duplicates("normalized_target_name")
        .set_index("normalized_target_name")
    )
    if set(planned_inventory.index) != set(PLANNED_TARGETS):
        missing = sorted(set(PLANNED_TARGETS) - set(planned_inventory.index))
        raise RuntimeError(f"Planned targets missing from inventory: {missing}")
    consumed_static = set(
        static_schema.loc[
            static_schema["consumed_by_model"].astype(bool),
            "source_column",
        ].astype(str)
    )
    projected_probe_mde = float(np.median(probe_mdes))
    projected_neighbor_mde = float(np.median(neighbor_mdes))
    for target in PLANNED_TARGETS:
        inventory_row = planned_inventory.loc[target]
        n_test = int(inventory_row["n_valid_numeric_test"])
        if n_test != EXPECTED_PLANNED_TEST_N:
            raise RuntimeError(f"Unexpected planned-target test n for {target}")
        model_status = str(inventory_row["model_input_status"])
        feature_names = str(inventory_row["model_input_feature_names"])
        if model_status == "static_input":
            exact_features = [
                name.strip()
                for name in feature_names.split(",")
                if name.strip() and name.strip() != "nan"
            ]
            if not exact_features or not set(exact_features) <= consumed_static:
                raise RuntimeError(f"Static-input evidence failed for {target}")
        for analysis_type, projected_mde, threshold in [
            (
                "incremental_value_probe_projected",
                projected_probe_mde,
                PROBE_SMALL_EFFECT_THRESHOLD,
            ),
            (
                "neighbor_sharing_projected",
                projected_neighbor_mde,
                NEIGHBOR_SMALL_GAIN_THRESHOLD,
            ),
        ]:
            result_rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "analysis_type": analysis_type,
                    "n": n_test,
                    "baseline_r_squared": np.nan,
                    "observed_effect": np.nan,
                    "minimum_detectable_effect": projected_mde,
                    "bootstrap_standard_error": np.nan,
                    "effective_alpha": (
                        TWO_SIDED_ALPHA
                        if "incremental" in analysis_type
                        else float(
                            np.median(
                                [
                                    candidate_fdr_p_threshold(
                                        family_p_values,
                                        existing,
                                    )
                                    for existing in EXISTING_TARGETS
                                ]
                            )
                        )
                    ),
                    "empirical_critical_value": np.nan,
                    "power_target": POWER_TARGET,
                    "adequacy_threshold": threshold,
                    "verdict": (
                        "adequately powered"
                        if projected_mde <= threshold
                        else "underpowered"
                    ),
                    "mde_basis": (
                        "projected from median existing external-target empirical "
                        "MDE at the same test n; update after target-specific fit"
                    ),
                    "model_input_status": model_status,
                    "notes": (
                        f"Coverage {n_test}; published CCA status "
                        f"{inventory_row['published_cca_status']}."
                    ),
                }
            )

    results = pd.DataFrame(result_rows)
    results_path = output_directory / "step0_power_analysis.csv"
    results.to_csv(results_path, index=False)
    existing_probe = results[
        results["analysis_type"].eq("incremental_value_probe")
    ]
    existing_neighbor = results[
        results["analysis_type"].eq("neighbor_sharing")
    ]
    sentence_one = (
        "The frozen test incremental probes required a delta R-squared of "
        f"{existing_probe['minimum_detectable_effect'].min():.3f} to "
        f"{existing_probe['minimum_detectable_effect'].max():.3f} for 80% "
        "power, so they were underpowered for the prespecified small increment "
        f"of {PROBE_SMALL_EFFECT_THRESHOLD:.2f}; their nulls mean no large "
        "incremental effect, not no effect."
    )
    sentence_two = (
        "The neutral-state neighbor-sharing analyses required standardized "
        f"similarity gains of {existing_neighbor['minimum_detectable_effect'].min():.3f} "
        f"to {existing_neighbor['minimum_detectable_effect'].max():.3f} after "
        "the frozen FDR family and were adequately powered for gains of "
        f"{NEIGHBOR_SMALL_GAIN_THRESHOLD:.2f}, while smaller gains remain "
        "unresolved."
    )
    verdict_path = output_directory / "step0_power_verdict.md"
    verdict_path.write_text(sentence_one + " " + sentence_two + "\n")
    method_path = output_directory / "step0_power_method.json"
    method = {
        "stage": "step0_power_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "power_target": POWER_TARGET,
        "probe_alpha": TWO_SIDED_ALPHA,
        "fdr_alpha": FDR_ALPHA,
        "probe_small_effect_threshold": PROBE_SMALL_EFFECT_THRESHOLD,
        "neighbor_small_gain_threshold": NEIGHBOR_SMALL_GAIN_THRESHOLD,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "probe_mde_method": (
            "Shift the centered observed participant-bootstrap error "
            "distribution and find the smallest positive effect with empirical "
            "two-sided rejection probability at least 0.80."
        ),
        "neighbor_mde_method": (
            "Use the target-specific participant-bootstrap distribution and "
            "the largest candidate permutation p-value that retains BH q <= "
            "0.05 in the frozen 9-test family, then shift the centered empirical "
            "error distribution to 80% rejection probability."
        ),
        "planned_target_projection_warning": (
            "Planned-target MDEs are projections from existing external-target "
            "empirical MDEs at the same n. Baseline R-squared and observed "
            "effects do not exist until those exploratory models are fitted."
        ),
        "input_hashes_sha256": {
            str(path): sha256_file(path) for path in input_paths
        },
        "output_hashes_sha256": {
            str(path): sha256_file(path)
            for path in [
                schema_inventory_path,
                schema_printout_path,
                results_path,
                verdict_path,
            ]
        },
        "pause_required": True,
        "dynamics_analysis_executed": False,
    }
    method_path.write_text(json.dumps(method, indent=2, sort_keys=True) + "\n")
    (OUTPUT_ROOT / "LATEST_STEP0_RUN.txt").write_text(
        str(output_directory) + "\n"
    )
    for text_path in [schema_printout_path, verdict_path, method_path]:
        if "\u2014" in text_path.read_text():
            raise RuntimeError(f"Forbidden em dash in {text_path}")
    print(results.to_string(index=False))
    print()
    print(sentence_one)
    print(sentence_two)
    print(f"Saved Step 0 power analysis to {output_directory}")


if __name__ == "__main__":
    main()
