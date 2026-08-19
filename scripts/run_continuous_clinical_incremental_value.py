#!/usr/bin/env python3
"""Exploratory T2D incremental clinical value beyond glycemia."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import torch
from sklearn.model_selection import GridSearchCV, KFold


PROJECT_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import run_beyond_glucose_dynamics_power as existing_power
import run_hidden_state_clinical_probes as existing_incremental
import run_continuous_clinical_encoding_test as encoding
import run_continuous_reorganization as reorganization


OUTPUT_ROOT = PROJECT_ROOT / "outputs/continuous_clinical"
MODEL_ROOT = OUTPUT_ROOT / "incremental_value_frozen_models"
TARGETS_PATH = OUTPUT_ROOT / "clinical_targets.parquet"
VALIDATION_GLYCEMIC_PATH = encoding.VALIDATION_GLYCEMIC_PATH
TEST_GLYCEMIC_PATH = (
    encoding.STEP4_ROOT / "test_glycemic_nuisance_features.parquet"
)
NEIGHBOR_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs/continuous_reorg/neighborhood_homogeneity_results.csv"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/"
    "best_model_checkpoint.pt"
)
RESULTS_PATH = OUTPUT_ROOT / "incremental_value_results.csv"
MDE_PATH = OUTPUT_ROOT / "incremental_value_mde.csv"
PROMPT2_RESULTS_PATH = OUTPUT_ROOT / "ridge_probe_results.csv"
PREDICTIONS_PATH = (
    OUTPUT_ROOT / "incremental_value_test_predictions.parquet"
)
FIGURE_PATH = OUTPUT_ROOT / "fig9_incremental_value_forest.png"
MANIFEST_PATH = OUTPUT_ROOT / "incremental_value_manifest.json"
PC2_NOTE_PATH = OUTPUT_ROOT / "incremental_value_pc2_note.md"

RANDOM_SEED = 42
INNER_FOLDS = 5
BOOTSTRAP_REPLICATES = 2000
POWER_TARGET = 0.80
TWO_SIDED_ALPHA = 0.05
SMALL_DELTA_R2_FLOOR = 0.05
N_JOBS = -1
EXPECTED_VALIDATION_N = 91
EXPECTED_TEST_N = 83
PRIMARY_K = 10
SENSITIVITY_K = [5, 20]
GLYCEMIC_COLUMNS = ["mean_glucose", "glucose_cv", "tir_70_180"]
STATE_COLUMNS = [f"state_{index:03d}" for index in range(128)]
BASE_TARGETS = ["bmi", "log_tg_hdl", "log_c_peptide", "clinical_pc1"]
PC2_TARGET = "clinical_pc2"
TARGETS = [*BASE_TARGETS, PC2_TARGET]
SPACES = ["h0", "full_ht", "neutral_ht"]
FEATURE_SETS = ["glycemia_only", "state_only", "glycemia_plus_state"]
TARGET_LABELS = {
    "bmi": "BMI",
    "log_tg_hdl": "Log TG/HDL",
    "log_c_peptide": "Log C-peptide",
    "clinical_pc1": "Clinical PC1: insulin-resistance axis",
    "clinical_pc2": (
        "Clinical PC2: BMI/TG-HDL dissociation axis "
        "(independent of C-peptide)"
    ),
}
DIRECT_INPUT_COLUMNS = {
    "bmi": ["bmi_baseline"],
    "log_tg_hdl": [
        "triglycerides_mgdl_baseline",
        "hdl_cholesterol_mgdl_baseline",
    ],
    "log_c_peptide": ["c_peptide_ngml_baseline"],
    "clinical_pc1": [
        "bmi_baseline",
        "triglycerides_mgdl_baseline",
        "hdl_cholesterol_mgdl_baseline",
        "c_peptide_ngml_baseline",
    ],
    "clinical_pc2": [
        "bmi_baseline",
        "triglycerides_mgdl_baseline",
        "hdl_cholesterol_mgdl_baseline",
    ],
}
SPACE_LABELS = {
    "h0": "h0",
    "full_ht": "Full ht",
    "neutral_ht": "Neutral ht",
}
SPACE_COLORS = {
    "h0": "#BA2828",
    "full_ht": "#003366",
    "neutral_ht": "#5BBABA",
}
SPACE_MARKERS = {"h0": "o", "full_ht": "s", "neutral_ht": "D"}
STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
ANALYSIS_LABEL = "T2D-only exploratory incremental clinical value"
AGE_CAVEAT = (
    "participants_age is participant age at study visit, NOT age at "
    "diabetes diagnosis. Age is not a predictor or target here."
)
NONFASTING_CAVEAT = (
    "C-peptide and triglycerides were not confirmed fasting measurements. "
    "Interpret log C-peptide and log TG/HDL accordingly."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def verify_direct_input_overlap() -> dict:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    static_reals = checkpoint["metadata"]["feature_spec"]["static_reals"]
    static_set = set(static_reals)
    missing = {
        target: [column for column in columns if column not in static_set]
        for target, columns in DIRECT_INPUT_COLUMNS.items()
    }
    if any(missing.values()):
        raise RuntimeError(f"Direct static input audit failed: {missing}")
    return {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "all_targets_are_direct_inputs_or_derivatives": True,
        "target_to_direct_static_inputs": DIRECT_INPUT_COLUMNS,
        "interpretation_scope": (
            "recoverability of input-derived phenotype information beyond "
            "glycemic summaries, not prediction of unseen external targets"
        ),
        "h0_caveat": (
            "h0 is full-profile static-conditioned initialization. Positive "
            "h0 increments can reflect retention of direct static inputs and "
            "must not be described as information acquired during streaming."
        ),
    }


def load_inputs() -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame], dict]:
    targets = pd.read_parquet(TARGETS_PATH)
    targets["participant_id"] = targets["participant_id"].astype(str)
    glycemic = {}
    for split, path in [
        ("validation", VALIDATION_GLYCEMIC_PATH),
        ("test", TEST_GLYCEMIC_PATH),
    ]:
        frame = pd.read_parquet(
            path,
            columns=["participant_id", *GLYCEMIC_COLUMNS],
        )
        frame["participant_id"] = frame["participant_id"].astype(str)
        if frame["participant_id"].duplicated().any():
            raise RuntimeError(f"Duplicate {split} glycemic participants")
        if not np.isfinite(frame[GLYCEMIC_COLUMNS].to_numpy(float)).all():
            raise RuntimeError(f"Nonfinite {split} glycemic summaries")
        glycemic[split] = frame
    representations, representation_audit = (
        reorganization.load_all_representations()
    )
    expected = {"validation": EXPECTED_VALIDATION_N, "test": EXPECTED_TEST_N}
    analysis_frames = {}
    for split, expected_n in expected.items():
        target = (
            targets[targets["split"].eq(split)]
            [["participant_id", *TARGETS]]
            .merge(
                glycemic[split],
                on="participant_id",
                validate="one_to_one",
            )
            .sort_values("participant_id")
            .reset_index(drop=True)
        )
        if len(target) != expected_n:
            raise RuntimeError(
                f"{split} T2D count changed: {len(target)} != {expected_n}"
            )
        if not np.isfinite(
            target[[*TARGETS, *GLYCEMIC_COLUMNS]].to_numpy(float)
        ).all():
            raise RuntimeError(f"Nonfinite {split} target or glycemic values")
        analysis_frames[split] = target
        expected_ids = set(target["participant_id"])
        for space in SPACES:
            state = representations[split][space]["state"]
            if set(state["participant_id"]) != set(
                glycemic[split]["participant_id"]
            ):
                raise RuntimeError(
                    f"{split} {space} full-cohort representation mismatch"
                )
            selected = state[state["participant_id"].isin(expected_ids)]
            if len(selected) != expected_n:
                raise RuntimeError(
                    f"{split} {space} T2D representation coverage changed"
                )
    return targets, representations, analysis_frames, representation_audit


def feature_definition(feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set == "glycemia_only":
        return GLYCEMIC_COLUMNS, []
    if feature_set == "state_only":
        return [], STATE_COLUMNS
    if feature_set == "glycemia_plus_state":
        return GLYCEMIC_COLUMNS, STATE_COLUMNS
    raise KeyError(feature_set)


def fit_frozen_pipeline(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    space: str,
    feature_set: str,
) -> tuple[np.ndarray, float, Path]:
    numeric, hidden = feature_definition(feature_set)
    pipeline, grid = existing_incremental.estimator(
        numeric,
        [],
        hidden,
        False,
    )
    search = GridSearchCV(
        pipeline,
        grid,
        cv=KFold(
            INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED + 19000,
        ),
        scoring="neg_mean_squared_error",
        n_jobs=N_JOBS,
        refit=True,
        error_score="raise",
    )
    search.fit(validation, validation[target].to_numpy(float))
    prediction = search.predict(test)
    model_path = (
        MODEL_ROOT
        / f"{target}__{space}__{feature_set}__validation_frozen.joblib"
    )
    joblib.dump(search.best_estimator_, model_path)
    return (
        prediction,
        float(search.best_params_["model__alpha"]),
        model_path,
    )


def r2_interval(
    target: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    point = encoding.performance_metrics(target, observed, predicted)["r2"]
    distribution = encoding.bootstrap_performance(
        target,
        observed,
        predicted,
        seed,
    )["r2"]
    return (
        float(point),
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    )


def paired_delta_distribution(
    target: str,
    space: str,
    participant_ids: np.ndarray,
    observed: np.ndarray,
    baseline_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
) -> tuple[float, float, np.ndarray, int]:
    bootstrap_target = f"{target}__{space}"
    rows = []
    for feature_set, prediction in [
        (existing_power.PROBE_BASELINE_FEATURE_SET, baseline_prediction),
        (existing_power.PROBE_AUGMENTED_FEATURE_SET, augmented_prediction),
    ]:
        for participant_id, value, estimate in zip(
            participant_ids,
            observed,
            prediction,
        ):
            rows.append(
                {
                    "participant_id": participant_id,
                    "target": bootstrap_target,
                    "feature_set": feature_set,
                    "observed_transformed": value,
                    "predicted_transformed": estimate,
                }
            )
    return existing_power.probe_bootstrap_distribution(
        pd.DataFrame(rows),
        bootstrap_target,
    )


def null_interpretation(
    delta: float,
    ci_low: float,
    ci_high: float,
    mde: float,
) -> str:
    if ci_low > 0:
        return "positive_incremental_value"
    if ci_high < 0:
        return "no_effect_combined_model_worse"
    if mde <= SMALL_DELTA_R2_FLOOR:
        return "no_effect_at_small_delta_r2_floor"
    return "no_large_effect_detected_underpowered_below_mde"


def run_incremental_analysis(
    representations: dict,
    analysis_frames: dict[str, pd.DataFrame],
    targets_to_run: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    result_rows = []
    mde_rows = []
    prediction_rows = []
    model_records = []
    glycemia_cache = {}
    selected_targets = TARGETS if targets_to_run is None else targets_to_run
    for target in selected_targets:
        validation_base = analysis_frames["validation"].copy()
        test_base = analysis_frames["test"].copy()
        glycemia_prediction, glycemia_alpha, glycemia_model_path = (
            fit_frozen_pipeline(
                validation_base,
                test_base,
                target,
                "shared",
                "glycemia_only",
            )
        )
        glycemia_cache[target] = {
            "prediction": glycemia_prediction,
            "alpha": glycemia_alpha,
            "model_path": glycemia_model_path,
        }
        observed = test_base[target].to_numpy(float)
        participant_ids = test_base["participant_id"].to_numpy()
        glycemia_r2 = r2_interval(
            target,
            observed,
            glycemia_prediction,
            stable_seed("r2", target, "glycemia_only", RANDOM_SEED),
        )
        for participant_id, value, estimate in zip(
            participant_ids,
            observed,
            glycemia_prediction,
        ):
            prediction_rows.append(
                {
                    "participant_id": participant_id,
                    "split": "test",
                    "target": target,
                    "space": "shared",
                    "feature_set": "glycemia_only",
                    "observed_analysis_scale": value,
                    "predicted_analysis_scale": estimate,
                }
            )
        model_records.append(
            {
                "target": target,
                "space": "shared",
                "feature_set": "glycemia_only",
                "best_alpha": glycemia_alpha,
                "model_path": str(glycemia_model_path),
            }
        )
        for space in SPACES:
            validation = validation_base.merge(
                representations["validation"][space]["state"],
                on="participant_id",
                validate="one_to_one",
            )
            test = test_base.merge(
                representations["test"][space]["state"],
                on="participant_id",
                validate="one_to_one",
            )
            if not np.array_equal(
                test["participant_id"].to_numpy(),
                participant_ids,
            ):
                raise RuntimeError(f"Test participant order changed for {space}")
            state_prediction, state_alpha, state_model_path = (
                fit_frozen_pipeline(
                    validation,
                    test,
                    target,
                    space,
                    "state_only",
                )
            )
            combined_prediction, combined_alpha, combined_model_path = (
                fit_frozen_pipeline(
                    validation,
                    test,
                    target,
                    space,
                    "glycemia_plus_state",
                )
            )
            state_r2 = r2_interval(
                target,
                observed,
                state_prediction,
                stable_seed("r2", target, space, "state_only"),
            )
            combined_r2 = r2_interval(
                target,
                observed,
                combined_prediction,
                stable_seed("r2", target, space, "combined"),
            )
            (
                reproduced_baseline_r2,
                delta_r2,
                delta_distribution,
                n_participants,
            ) = paired_delta_distribution(
                target,
                space,
                participant_ids,
                observed,
                glycemia_prediction,
                combined_prediction,
            )
            if not np.isclose(
                reproduced_baseline_r2,
                glycemia_r2[0],
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"Baseline R2 mismatch for {target} {space}"
                )
            delta_ci = np.quantile(delta_distribution, [0.025, 0.975])
            mde, standard_error, critical_value = (
                existing_power.empirical_mde(
                    delta_distribution,
                    TWO_SIDED_ALPHA,
                )
            )
            interpretation = null_interpretation(
                delta_r2,
                float(delta_ci[0]),
                float(delta_ci[1]),
                mde,
            )
            result_rows.append(
                {
                    "analysis_label": ANALYSIS_LABEL,
                    "tier": "exploratory_not_frozen_tier1",
                    "target_is_direct_static_input_or_derivative": True,
                    "direct_static_input_columns": json.dumps(
                        DIRECT_INPUT_COLUMNS[target]
                    ),
                    "interpretation_scope": (
                        "input_recoverability_not_unseen_external_prediction"
                    ),
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "space": space,
                    "space_label": SPACE_LABELS[space],
                    "validation_n": len(validation),
                    "test_n": len(test),
                    "glycemic_predictors": json.dumps(GLYCEMIC_COLUMNS),
                    "hidden_state_dimensions": len(STATE_COLUMNS),
                    "model_family": "Ridge",
                    "alpha_grid": json.dumps(
                        existing_incremental.ALPHAS.tolist()
                    ),
                    "glycemia_only_best_alpha": glycemia_alpha,
                    "state_only_best_alpha": state_alpha,
                    "combined_best_alpha": combined_alpha,
                    "glycemia_only_test_r2": glycemia_r2[0],
                    "glycemia_only_r2_ci_low": glycemia_r2[1],
                    "glycemia_only_r2_ci_high": glycemia_r2[2],
                    "state_only_test_r2": state_r2[0],
                    "state_only_r2_ci_low": state_r2[1],
                    "state_only_r2_ci_high": state_r2[2],
                    "combined_test_r2": combined_r2[0],
                    "combined_r2_ci_low": combined_r2[1],
                    "combined_r2_ci_high": combined_r2[2],
                    "test_delta_r2": delta_r2,
                    "delta_r2_ci_low": float(delta_ci[0]),
                    "delta_r2_ci_high": float(delta_ci[1]),
                    "minimum_detectable_delta_r2_80": mde,
                    "null_interpretation": interpretation,
                    "bootstrap_unit": "participant",
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "validation_tuned_test_frozen": True,
                    "fdr_family": "none_exploratory",
                    "neighbor_results_recomputed": False,
                }
            )
            mde_rows.append(
                {
                    "analysis_label": ANALYSIS_LABEL,
                    "tier": "exploratory_not_frozen_tier1",
                    "target_is_direct_static_input_or_derivative": True,
                    "direct_static_input_columns": json.dumps(
                        DIRECT_INPUT_COLUMNS[target]
                    ),
                    "interpretation_scope": (
                        "input_recoverability_not_unseen_external_prediction"
                    ),
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "space": space,
                    "space_label": SPACE_LABELS[space],
                    "test_n": n_participants,
                    "observed_test_delta_r2": delta_r2,
                    "delta_r2_ci_low": float(delta_ci[0]),
                    "delta_r2_ci_high": float(delta_ci[1]),
                    "minimum_detectable_delta_r2_80": mde,
                    "bootstrap_standard_error": standard_error,
                    "bootstrap_critical_value": critical_value,
                    "power_target": POWER_TARGET,
                    "two_sided_alpha": TWO_SIDED_ALPHA,
                    "small_delta_r2_floor": SMALL_DELTA_R2_FLOOR,
                    "null_interpretation": interpretation,
                    "method": (
                        "Step 0 empirical centered participant-bootstrap "
                        "error MDE binary search"
                    ),
                }
            )
            for feature_set, prediction in [
                ("state_only", state_prediction),
                ("glycemia_plus_state", combined_prediction),
            ]:
                for participant_id, value, estimate in zip(
                    participant_ids,
                    observed,
                    prediction,
                ):
                    prediction_rows.append(
                        {
                            "participant_id": participant_id,
                            "split": "test",
                            "target": target,
                            "space": space,
                            "feature_set": feature_set,
                            "observed_analysis_scale": value,
                            "predicted_analysis_scale": estimate,
                        }
                    )
            model_records.extend(
                [
                    {
                        "target": target,
                        "space": space,
                        "feature_set": "state_only",
                        "best_alpha": state_alpha,
                        "model_path": str(state_model_path),
                    },
                    {
                        "target": target,
                        "space": space,
                        "feature_set": "glycemia_plus_state",
                        "best_alpha": combined_alpha,
                        "model_path": str(combined_model_path),
                    },
                ]
            )
    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(mde_rows),
        pd.DataFrame(prediction_rows),
        {"models": model_records, "glycemia_cache": glycemia_cache},
    )


def verify_neighbor_results() -> dict:
    frame = pd.read_csv(NEIGHBOR_RESULTS_PATH)
    required_targets = {
        "clinical_pc1",
        "clinical_pc2",
        "bmi",
        "log_tg_hdl",
        "log_c_peptide",
    }
    selected = frame[
        frame["target"].isin(required_targets)
        & frame["k_neighbors"].isin([PRIMARY_K, *SENSITIVITY_K])
    ]
    spaces = selected[
        selected["record_type"].eq("space_summary")
    ]
    transitions = selected[
        selected["record_type"].eq("h0_to_ht_transition")
    ]
    neutral_k10 = transitions[
        transitions["k_neighbors"].eq(PRIMARY_K)
        & transitions["ht_space"].eq("neutral_ht")
    ].copy()
    if (
        set(selected["target"]) != required_targets
        or set(selected["k_neighbors"]) != {5, 10, 20}
        or len(spaces) != 45
        or len(transitions) != 30
        or len(neutral_k10) != 5
    ):
        raise RuntimeError("Prompt 3 neighbor-sharing coverage changed")
    return {
        "source_path": str(NEIGHBOR_RESULTS_PATH),
        "source_sha256": sha256_file(NEIGHBOR_RESULTS_PATH),
        "recomputed": False,
        "targets_confirmed": sorted(required_targets),
        "k_confirmed": [5, 10, 20],
        "space_summary_rows": len(spaces),
        "transition_rows": len(transitions),
        "neutral_k10_change_min": float(
            neutral_k10["h0_to_ht_gap_change"].min()
        ),
        "neutral_k10_change_max": float(
            neutral_k10["h0_to_ht_gap_change"].max()
        ),
        "neutral_k10_all_cis_below_zero": bool(
            (neutral_k10["gap_change_ci_high"] < 0).all()
        ),
        "neutral_k10_rows": neutral_k10[
            [
                "target",
                "h0_to_ht_gap_change",
                "gap_change_ci_low",
                "gap_change_ci_high",
            ]
        ].to_dict(orient="records"),
    }


def result_synthesis(results: pd.DataFrame, neighbor_audit: dict) -> dict:
    positive = results[
        results["delta_r2_ci_low"] > 0
    ][["target", "space", "test_delta_r2", "delta_r2_ci_low", "delta_r2_ci_high"]]
    positive_by_space = {
        space: sorted(
            positive.loc[positive["space"].eq(space), "target"]
            .unique()
            .tolist()
        )
        for space in SPACES
    }
    streaming_positive_targets = sorted(
        set(positive_by_space["full_ht"])
        | set(positive_by_space["neutral_ht"])
    )
    comparison = (
        "Prompt 3 showed uniform neutralization-related homogeneity loss "
        "across all five targets, without target specificity. "
    )
    if streaming_positive_targets:
        named = ", ".join(TARGET_LABELS[target] for target in streaming_positive_targets)
        comparison += (
            f"In contrast, incremental value identifies {named} as a "
            "streaming-space exception with a confidence interval above "
            "zero. The analyses therefore differ in target specificity."
        )
        completed_conclusion = (
            "At least one streaming-space target shows confirmed "
            "incremental value beyond glycemia: " + named + "."
        )
    else:
        comparison += (
            "No full ht or neutral ht increment has a confidence interval "
            "above zero. No target-specific discrepancy is claimed. Both "
            "analyses point to loss of clinical organization after streaming "
            "or neutralization, although they test different endpoints."
        )
        completed_conclusion = (
            "No streaming-space target among the completed five-target set "
            "shows confirmed incremental value beyond glycemia."
        )
    prompt2 = pd.read_csv(PROMPT2_RESULTS_PATH)
    prompt2_pc2 = prompt2[
        prompt2["target"].eq(PC2_TARGET)
        & prompt2["space"].eq("neutral_ht")
    ]
    if len(prompt2_pc2) != 1:
        raise RuntimeError("Prompt 2 Clinical PC2 neutral result is missing")
    direct = prompt2_pc2.iloc[0]
    current = results[
        results["target"].eq(PC2_TARGET)
        & results["space"].eq("neutral_ht")
    ]
    prompt2_contrast = None
    if len(current) == 1:
        incremental = current.iloc[0]
        prompt2_contrast = (
            f"Prompt 2 showed positive direct neutral-state encoding for "
            f"Clinical PC2: R2 {direct.test_r2:.4f}, 95% CI "
            f"[{direct.test_r2_ci_low:.4f}, {direct.test_r2_ci_high:.4f}]. "
            f"After conditioning on glycemic summaries, neutral ht delta R2 "
            f"was {incremental.test_delta_r2:.4f}, 95% CI "
            f"[{incremental.delta_r2_ci_low:.4f}, "
            f"{incremental.delta_r2_ci_high:.4f}], with MDE "
            f"{incremental.minimum_detectable_delta_r2_80:.4f}. Direct "
            "encoding therefore did not translate into confirmed incremental "
            "value beyond glycemia."
        )
    return {
        "positive_incremental_rows": positive.to_dict(orient="records"),
        "targets_with_positive_incremental_value": sorted(
            positive["target"].unique().tolist()
        ),
        "positive_targets_by_space": positive_by_space,
        "streaming_targets_with_positive_incremental_value": (
            streaming_positive_targets
        ),
        "completed_set_conclusion": completed_conclusion,
        "prompt2_pc2_contrast": prompt2_contrast,
        "prompt3_comparison": comparison,
        "prompt3_neutral_k10_change_range": [
            neighbor_audit["neutral_k10_change_min"],
            neighbor_audit["neutral_k10_change_max"],
        ],
    }


def make_figure(results: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(13.5, 9.5))
    target_order = (
        results[results["space"].eq("neutral_ht")]
        .sort_values("test_delta_r2", ascending=False)["target"]
        .tolist()
    )
    if len(target_order) != results["target"].nunique():
        raise RuntimeError("Figure target ordering is incomplete")
    y_base = np.arange(len(target_order))[::-1]
    offsets = {"h0": 0.22, "full_ht": 0.0, "neutral_ht": -0.22}
    for row_index, target in enumerate(target_order):
        y = y_base[row_index]
        target_rows = results[results["target"].eq(target)]
        target_mde = float(
            target_rows["minimum_detectable_delta_r2_80"].max()
        )
        axis.add_patch(
            Rectangle(
                (-target_mde, y - 0.42),
                2 * target_mde,
                0.84,
                color="#D9D9D9",
                alpha=0.34,
                linewidth=0,
                zorder=0,
            )
        )
    for space in SPACES:
        selected = (
            results[results["space"].eq(space)]
            .set_index("target")
            .reindex(target_order)
        )
        y = y_base + offsets[space]
        values = selected["test_delta_r2"].to_numpy(float)
        low = selected["delta_r2_ci_low"].to_numpy(float)
        high = selected["delta_r2_ci_high"].to_numpy(float)
        axis.errorbar(
            values,
            y,
            xerr=np.vstack([values - low, high - values]),
            fmt=SPACE_MARKERS[space],
            markersize=8,
            linewidth=1.8,
            capsize=4,
            color=SPACE_COLORS[space],
            label=SPACE_LABELS[space],
            zorder=3,
        )
    axis.axvline(0, color="black", linewidth=1.3)
    axis.set_yticks(y_base)
    axis.set_yticklabels([TARGET_LABELS[target] for target in target_order])
    axis.set_xlabel(
        "Test delta R2: glycemia plus state minus glycemia only"
    )
    axis.set_ylabel("")
    axis.set_title(
        "Exploratory T2D incremental clinical value beyond glycemia",
        fontweight="bold",
        pad=16,
    )
    axis.text(
        0.0,
        1.01,
        (
            "Participant-bootstrap 95% CIs. Gray bands show the largest "
            "80% MDE across spaces for each target. All targets are direct "
            "static model inputs or derivatives."
        ),
        transform=axis.transAxes,
        fontsize=11,
        ha="left",
        va="bottom",
    )
    axis.legend(
        title="Hidden-state space",
        frameon=True,
        loc="best",
    )
    sns.despine(ax=axis)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)



def pc2_space_statement(row: pd.Series) -> str:
    point = float(row["test_delta_r2"])
    low = float(row["delta_r2_ci_low"])
    high = float(row["delta_r2_ci_high"])
    mde = float(row["minimum_detectable_delta_r2_80"])
    if low > 0:
        mde_phrase = (
            "The point estimate exceeds its 80% MDE."
            if point >= mde
            else "The CI is positive, although the point estimate is below its MDE."
        )
        return (
            f"genuine positive: delta R2 {point:.4f}, 95% CI "
            f"[{low:.4f}, {high:.4f}], MDE {mde:.4f}. {mde_phrase}"
        )
    if mde <= SMALL_DELTA_R2_FLOOR:
        return (
            f"confirmed null at the small-effect floor: delta R2 {point:.4f}, "
            f"95% CI [{low:.4f}, {high:.4f}], MDE {mde:.4f}. The estimate "
            "is tightly pinned near zero."
        )
    return (
        f"no large effect detected, underpowered below MDE: delta R2 "
        f"{point:.4f}, 95% CI [{low:.4f}, {high:.4f}], MDE {mde:.4f}. "
        "The estimate remains inside its detectable-effect band."
    )


def write_pc2_note(pc2_results: pd.DataFrame, synthesis: dict) -> None:
    selected = pc2_results.set_index("space").reindex(SPACES)
    lines = [
        "# Clinical PC2 incremental value beyond glycemia",
        "",
        "This is an exploratory T2D-only analysis with 91 validation and 83 test participants.",
        "Models were tuned on validation and frozen before test evaluation. Confidence intervals use 2,000 participant-bootstrap replicates.",
        "",
        "## Space-specific results",
        "",
    ]
    for space in SPACES:
        lines.append(
            f"- {SPACE_LABELS[space]}: {pc2_space_statement(selected.loc[space])}"
        )
    lines.extend(
        [
            "",
            "## Completed five-target conclusion",
            "",
            synthesis["completed_set_conclusion"],
            "",
            "## Prompt 2 contrast",
            "",
            synthesis["prompt2_pc2_contrast"],
            "",
            "## Interpretation guardrail",
            "",
            "Clinical PC2 is derived from BMI and log TG/HDL. Its direct static model inputs are bmi_baseline, triglycerides_mgdl_baseline, and hdl_cholesterol_mgdl_baseline. The h0 result measures input recoverability, not unseen physiological discovery.",
            "",
            "## Prompt 3 comparison",
            "",
            synthesis["prompt3_comparison"],
            "",
        ]
    )
    PC2_NOTE_PATH.write_text("\n".join(lines))


def append_pc2_only() -> None:
    started = time.time()
    existing_results_bytes = RESULTS_PATH.read_bytes()
    existing_mde_bytes = MDE_PATH.read_bytes()
    if not existing_results_bytes.endswith(b"\n") or not existing_mde_bytes.endswith(b"\n"):
        raise RuntimeError("Existing CSV does not end with a newline")
    existing_results = pd.read_csv(RESULTS_PATH)
    existing_mde = pd.read_csv(MDE_PATH)
    existing_predictions = pd.read_parquet(PREDICTIONS_PATH)
    existing_manifest = json.loads(MANIFEST_PATH.read_text())
    for name, frame in [
        ("results", existing_results),
        ("mde", existing_mde),
    ]:
        if set(frame["target"]) != set(BASE_TARGETS):
            raise RuntimeError(f"Existing {name} target set changed")
        if frame.groupby("target").size().to_dict() != {
            target: len(SPACES) for target in BASE_TARGETS
        }:
            raise RuntimeError(f"Existing {name} rows are not the frozen four-target set")
        if frame["target"].eq(PC2_TARGET).any():
            raise RuntimeError(f"Clinical PC2 is already present in {name}")
    if existing_manifest.get("status") != "QC_COMPLETE":
        raise RuntimeError("Existing incremental-value manifest is not QC_COMPLETE")

    direct_input_audit = verify_direct_input_overlap()
    targets, representations, analysis_frames, representation_audit = load_inputs()
    neighbor_audit = verify_neighbor_results()
    new_results, new_mde, new_predictions, fitting_audit = run_incremental_analysis(
        representations,
        analysis_frames,
        targets_to_run=[PC2_TARGET],
    )
    if len(new_results) != len(SPACES) or len(new_mde) != len(SPACES):
        raise RuntimeError("Clinical PC2 did not produce three result rows")
    if len(new_predictions) != EXPECTED_TEST_N * (1 + 2 * len(SPACES)):
        raise RuntimeError("Clinical PC2 prediction count changed")
    if set(new_results["space"]) != set(SPACES):
        raise RuntimeError("Clinical PC2 space coverage is incomplete")
    expected_direct_inputs = json.dumps(DIRECT_INPUT_COLUMNS[PC2_TARGET])
    if not new_results["direct_static_input_columns"].eq(expected_direct_inputs).all():
        raise RuntimeError("Clinical PC2 direct static input annotation mismatch")
    if not new_results["interpretation_scope"].eq(
        "input_recoverability_not_unseen_external_prediction"
    ).all():
        raise RuntimeError("Clinical PC2 interpretation scope mismatch")
    if list(new_results.columns) != list(existing_results.columns):
        raise RuntimeError("Clinical PC2 results schema differs from existing CSV")
    if list(new_mde.columns) != list(existing_mde.columns):
        raise RuntimeError("Clinical PC2 MDE schema differs from existing CSV")

    results_append_bytes = new_results.to_csv(index=False, header=False).encode()
    mde_append_bytes = new_mde.to_csv(index=False, header=False).encode()
    with RESULTS_PATH.open("ab") as handle:
        handle.write(results_append_bytes)
    with MDE_PATH.open("ab") as handle:
        handle.write(mde_append_bytes)
    if not RESULTS_PATH.read_bytes().startswith(existing_results_bytes):
        raise RuntimeError("Existing result bytes were not preserved as CSV prefix")
    if not MDE_PATH.read_bytes().startswith(existing_mde_bytes):
        raise RuntimeError("Existing MDE bytes were not preserved as CSV prefix")

    complete_results = pd.read_csv(RESULTS_PATH)
    complete_mde = pd.read_csv(MDE_PATH)
    pd.testing.assert_frame_equal(
        existing_results,
        complete_results[complete_results["target"].isin(BASE_TARGETS)].reset_index(drop=True),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        existing_mde,
        complete_mde[complete_mde["target"].isin(BASE_TARGETS)].reset_index(drop=True),
        check_exact=True,
    )
    if complete_results.groupby("target").size().to_dict() != {
        target: len(SPACES) for target in TARGETS
    }:
        raise RuntimeError("Completed result table is not five targets by three spaces")
    if complete_mde.groupby("target").size().to_dict() != {
        target: len(SPACES) for target in TARGETS
    }:
        raise RuntimeError("Completed MDE table is not five targets by three spaces")

    complete_predictions = pd.concat(
        [existing_predictions, new_predictions],
        ignore_index=True,
    )
    if complete_predictions.duplicated(
        ["participant_id", "target", "space", "feature_set"]
    ).any():
        raise RuntimeError("Duplicate test prediction keys after PC2 append")
    complete_predictions.to_parquet(
        PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )
    make_figure(complete_results)
    synthesis = result_synthesis(complete_results, neighbor_audit)
    write_pc2_note(new_results, synthesis)

    manifest = existing_manifest
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["targets"] = TARGETS
    manifest["direct_input_overlap"] = direct_input_audit
    manifest["representation_audit"] = representation_audit
    manifest["synthesis"] = synthesis
    manifest["pc2_extension"] = {
        "target": PC2_TARGET,
        "execution_mode": "append_pc2_only",
        "existing_four_targets_recomputed": False,
        "existing_results_bytes_preserved_as_prefix": True,
        "existing_mde_bytes_preserved_as_prefix": True,
        "pre_append_results_sha256": hashlib.sha256(existing_results_bytes).hexdigest(),
        "pre_append_mde_sha256": hashlib.sha256(existing_mde_bytes).hexdigest(),
        "validation_n": EXPECTED_VALIDATION_N,
        "test_n": EXPECTED_TEST_N,
        "new_result_rows": len(new_results),
        "new_mde_rows": len(new_mde),
        "new_prediction_rows": len(new_predictions),
        "results": new_results[
            [
                "space",
                "test_delta_r2",
                "delta_r2_ci_low",
                "delta_r2_ci_high",
                "minimum_detectable_delta_r2_80",
                "null_interpretation",
            ]
        ].to_dict(orient="records"),
    }
    old_models = manifest["fitting_audit"]["models"]
    old_models.extend(fitting_audit["models"])
    manifest["fitting_audit"]["model_count"] = len(old_models)
    manifest["qc"].update(
        {
            "result_rows": len(complete_results),
            "mde_rows": len(complete_mde),
            "prediction_rows": len(complete_predictions),
            "targets": sorted(TARGETS),
            "pc2_rows_appended": len(new_results),
            "existing_four_targets_recomputed": False,
            "existing_results_bytes_preserved_as_prefix": True,
            "existing_mde_bytes_preserved_as_prefix": True,
            "direct_input_overlap_verified": True,
        }
    )
    manifest["output_paths"]["pc2_note"] = str(PC2_NOTE_PATH)
    manifest["output_hashes"] = {
        key: sha256_file(Path(path))
        for key, path in manifest["output_paths"].items()
    }
    manifest["runtime_seconds_pc2_extension"] = time.time() - started
    manifest["status"] = "QC_COMPLETE"
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "pc2_results": manifest["pc2_extension"]["results"],
                "completed_set_conclusion": synthesis["completed_set_conclusion"],
                "existing_four_targets_recomputed": False,
                "note": str(PC2_NOTE_PATH),
            },
            indent=2,
            default=json_default,
        )
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--append-pc2-only",
        action="store_true",
        help="Fit and append Clinical PC2 without recomputing existing targets.",
    )
    arguments = parser.parse_args()
    if arguments.append_pc2_only:
        append_pc2_only()
        return
    started = time.time()
    np.random.seed(RANDOM_SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "clinical_targets": TARGETS_PATH,
        "validation_glycemic_summaries": VALIDATION_GLYCEMIC_PATH,
        "test_glycemic_summaries": TEST_GLYCEMIC_PATH,
        "validation_representations": encoding.VALIDATION_REPRESENTATIONS_PATH,
        "test_representations": encoding.TEST_REPRESENTATIONS_PATH,
        "prompt3_neighbor_results": NEIGHBOR_RESULTS_PATH,
        "canonical_checkpoint": CHECKPOINT_PATH,
        "reused_incremental_script": (
            SCRIPTS_ROOT / "run_hidden_state_clinical_probes.py"
        ),
        "reused_power_script": (
            SCRIPTS_ROOT / "run_beyond_glucose_dynamics_power.py"
        ),
    }
    missing = [
        str(path) for path in input_paths.values() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    target_schema = pq.read_schema(TARGETS_PATH)
    age_metadata = {
        key.decode(): value.decode()
        for key, value in (
            target_schema.field("participants_age").metadata or {}
        ).items()
    }
    direct_input_audit = verify_direct_input_overlap()
    targets, representations, analysis_frames, representation_audit = (
        load_inputs()
    )
    neighbor_audit = verify_neighbor_results()
    results, mde, predictions, fitting_audit = run_incremental_analysis(
        representations,
        analysis_frames,
    )
    synthesis = result_synthesis(results, neighbor_audit)
    results.to_csv(RESULTS_PATH, index=False)
    mde.to_csv(MDE_PATH, index=False)
    predictions.to_parquet(
        PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )
    make_figure(results)

    required_outputs = {
        "incremental_value_results": RESULTS_PATH,
        "incremental_value_mde": MDE_PATH,
        "figure9_incremental_value_forest": FIGURE_PATH,
        "test_predictions": PREDICTIONS_PATH,
    }
    qc = {
        "validation_t2d_n": len(analysis_frames["validation"]),
        "test_t2d_n": len(analysis_frames["test"]),
        "result_rows": len(results),
        "mde_rows": len(mde),
        "prediction_rows": len(predictions),
        "targets": sorted(results["target"].unique().tolist()),
        "spaces": sorted(results["space"].unique().tolist()),
        "all_validation_tuned_test_frozen": bool(
            results["validation_tuned_test_frozen"].all()
        ),
        "all_bootstrap_n_2000": bool(
            results["bootstrap_replicates"].eq(
                BOOTSTRAP_REPLICATES
            ).all()
        ),
        "all_mdes_positive": bool(
            (mde["minimum_detectable_delta_r2_80"] > 0).all()
        ),
        "exploratory_not_fdr": bool(
            results["fdr_family"].eq("none_exploratory").all()
        ),
        "neighbor_results_recomputed": False,
        "neighbor_targets_and_sensitivities_confirmed": True,
        "direct_input_overlap_verified": (
            direct_input_audit[
                "all_targets_are_direct_inputs_or_derivatives"
            ]
        ),
        "all_required_outputs_present": all(
            path.exists() and path.stat().st_size > 0
            for path in required_outputs.values()
        ),
        "age_field_metadata_verified": age_metadata,
    }
    if not all(
        [
            qc["validation_t2d_n"] == EXPECTED_VALIDATION_N,
            qc["test_t2d_n"] == EXPECTED_TEST_N,
            qc["result_rows"] == len(TARGETS) * len(SPACES),
            qc["mde_rows"] == len(TARGETS) * len(SPACES),
            qc["prediction_rows"]
            == (
                len(TARGETS)
                * EXPECTED_TEST_N
                * (1 + 2 * len(SPACES))
            ),
            qc["all_validation_tuned_test_frozen"],
            qc["all_bootstrap_n_2000"],
            qc["all_mdes_positive"],
            qc["exploratory_not_fdr"],
            qc["direct_input_overlap_verified"],
            qc["all_required_outputs_present"],
            age_metadata.get("is_age_at_diabetes_diagnosis") == "false",
        ]
    ):
        raise RuntimeError(f"Incremental-value QC failed: {qc}")

    manifest = {
        "analysis": "Incremental clinical value beyond glycemia",
        "analysis_label": ANALYSIS_LABEL,
        "status": "QC_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tier": "exploratory_not_frozen_tier1",
        "fdr_family": "none",
        "cohort": {
            "population": "T2D-only complete cases",
            "validation_n": EXPECTED_VALIDATION_N,
            "test_n": EXPECTED_TEST_N,
        },
        "targets": TARGETS,
        "spaces": SPACES,
        "predictor_sets": {
            "glycemia_only": GLYCEMIC_COLUMNS,
            "state_only": "128-dimensional hidden-state representation",
            "glycemia_plus_state": (
                "three glycemic summaries plus 128-dimensional state"
            ),
        },
        "method": {
            "model": "Ridge",
            "preprocessing_and_estimator_reused_from": str(
                input_paths["reused_incremental_script"]
            ),
            "alpha_grid": existing_incremental.ALPHAS.tolist(),
            "regularization_tuning": (
                "5-fold validation-only GridSearchCV; final pipeline fitted "
                "on all T2D validation participants and frozen for test"
            ),
            "test_delta_r2": (
                "R2(glycemia plus state) minus R2(glycemia only)"
            ),
            "bootstrap": {
                "reused_function": (
                    "probe_bootstrap_distribution from "
                    "run_beyond_glucose_dynamics_power.py"
                ),
                "unit": "participant",
                "replicates": BOOTSTRAP_REPLICATES,
                "interval": "percentile 95%",
            },
            "mde": {
                "reused_function": (
                    "empirical_mde from "
                    "run_beyond_glucose_dynamics_power.py"
                ),
                "power": POWER_TARGET,
                "two_sided_alpha": TWO_SIDED_ALPHA,
                "small_delta_r2_floor": SMALL_DELTA_R2_FLOOR,
                "basis": "observed T2D test participant bootstrap",
            },
            "test_tuning": False,
        },
        "direct_input_overlap": direct_input_audit,
        "interpretation_guardrail": (
            "Positive h0 increments quantify retention of direct static "
            "model inputs beyond glycemic summaries. They are not evidence "
            "that streaming discovered unseen clinical information."
        ),
        "step_b_neighbor_sharing": neighbor_audit,
        "synthesis": synthesis,
        "age_caveat": AGE_CAVEAT,
        "nonfasting_caveat": NONFASTING_CAVEAT,
        "representation_audit": representation_audit,
        "fitting_audit": {
            "model_count": len(fitting_audit["models"]),
            "models": fitting_audit["models"],
        },
        "qc": qc,
        "input_paths": {
            key: str(path) for key, path in input_paths.items()
        },
        "input_hashes": {
            key: sha256_file(path) for key, path in input_paths.items()
        },
        "output_paths": {
            key: str(path) for key, path in required_outputs.items()
        },
        "output_hashes": {
            key: sha256_file(path)
            for key, path in required_outputs.items()
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "runtime_seconds": time.time() - started,
    }
    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output_directory": str(OUTPUT_ROOT),
                "status": manifest["status"],
                "results": results[
                    [
                        "target",
                        "space",
                        "test_delta_r2",
                        "delta_r2_ci_low",
                        "delta_r2_ci_high",
                        "minimum_detectable_delta_r2_80",
                        "null_interpretation",
                    ]
                ].to_dict(orient="records"),
                "synthesis": synthesis,
                "neighbor_recomputed": False,
                "manifest": str(MANIFEST_PATH),
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
