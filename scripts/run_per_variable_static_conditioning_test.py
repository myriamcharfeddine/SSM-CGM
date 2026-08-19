#!/usr/bin/env python3
"""Apply the frozen per-variable static interventions once to test."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

import run_per_variable_static_conditioning_validation as validation

from ssmcgm.data.aireadi import (
    build_stream_feature_spec,
    infer_or_validate_schema,
    make_aireadi_stream_splits,
    make_participant_streams,
    prepare_aireadi_panel,
)


REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
RUN_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/per_variable_static_conditioning_audit/"
    "20260727T214535Z"
)
FROZEN_ROOT = RUN_ROOT / "frozen_plan"
FROZEN_PLAN_PATH = FROZEN_ROOT / "frozen_test_application_plan.json"
FROZEN_HASH_PATH = FROZEN_ROOT / "frozen_test_application_plan.sha256"
FREEZE_MANIFEST_PATH = FROZEN_ROOT / "freeze_manifest.json"
OUTPUT_ROOT = RUN_ROOT / "test_interventions"
STEP4_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step4_test_confirmation/"
    "20260725T010440Z"
)
CANONICAL_TEST_REPRESENTATIONS_PATH = (
    STEP4_ROOT / "test_participant_representations.parquet"
)
STEP4_MANIFEST_PATH = STEP4_ROOT / "step4_manifest.json"
VALIDATION_PRESENTATION_PATH = (
    RUN_ROOT
    / "validation_interventions/"
    "validation_per_variable_static_effects_presentation.csv"
)

VARIABLE_ORDER = validation.VARIABLE_ORDER
MEDICATION_VARIABLES = validation.MEDICATION_VARIABLES
GLOBAL_VARIABLE_LABEL = validation.GLOBAL_VARIABLE_LABEL
ONE_VARIABLE_CONDITION = validation.ONE_VARIABLE_CONDITION
GLOBAL_CONDITION = validation.GLOBAL_CONDITION
HIDDEN_COLUMNS = validation.HIDDEN_COLUMNS
HIDDEN_SIZE = validation.HIDDEN_SIZE
EXPECTED_TEST_N = 221
TEST_LABEL = "test"
MIN_STREAM_STEPS = validation.MIN_STREAM_STEPS
K_NN = validation.K_NN
RANDOM_SEED = validation.RANDOM_SEED
BOOTSTRAP_REPLICATES = validation.BOOTSTRAP_REPLICATES
QC_PARTICIPANT_COUNT = validation.QC_PARTICIPANT_COUNT
QC_REPRESENTATION_ABS_TOLERANCE = 1e-4
PARQUET_COMPRESSION = validation.PARQUET_COMPRESSION
SEX_VARIABLE = "sex"
SEX_COLUMN = "demo_sex_at_birth"
SEX_REFERENCE_CATEGORY = "F"
MEDICATION_MIN_GROUP_N = 30
LOG_PROGRESS_INTERVAL = 10


def verify_frozen_plan() -> tuple[dict, str]:
    plan_hash = validation.sha256_file(FROZEN_PLAN_PATH)
    expected_hash = FROZEN_HASH_PATH.read_text().split()[0]
    freeze_manifest = json.loads(FREEZE_MANIFEST_PATH.read_text())
    if plan_hash != expected_hash:
        raise RuntimeError("Frozen plan hash does not match the hash file")
    if plan_hash != freeze_manifest["plan_sha256"]:
        raise RuntimeError("Frozen plan hash does not match the freeze manifest")
    if not freeze_manifest["test_application_authorized"]:
        raise RuntimeError("Frozen plan does not authorize test application")
    plan = json.loads(FROZEN_PLAN_PATH.read_text())
    if plan["variables_in_order"] != VARIABLE_ORDER:
        raise RuntimeError("Frozen variable order differs from implementation")
    if plan["metrics"]["k_nn"] != K_NN:
        raise RuntimeError("Frozen K_NN differs from implementation")
    if plan["test_participant_count"] != EXPECTED_TEST_N:
        raise RuntimeError("Frozen test participant count differs from implementation")
    return plan, plan_hash


def canonical_test_full_and_global(
    test_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    step4_manifest = json.loads(STEP4_MANIFEST_PATH.read_text())
    if step4_manifest["participant_count"] != EXPECTED_TEST_N:
        raise RuntimeError("Canonical Step 4 participant count is not 221")
    if sorted(step4_manifest["participant_ids"]) != test_ids:
        raise RuntimeError("Canonical Step 4 participant IDs differ from split")
    if step4_manifest["burn_in_minutes"] != 0:
        raise RuntimeError("Canonical Step 4 burn-in is not zero")
    frame = pd.read_parquet(CANONICAL_TEST_REPRESENTATIONS_PATH)
    frame["participant_id"] = frame["participant_id"].astype(str)
    if set(frame["split"]) != {TEST_LABEL}:
        raise RuntimeError("Canonical test representations contain another split")
    full = (
        frame[frame["representation_type"].eq("full_all")]
        .set_index("participant_id")
        .reindex(test_ids)
    )
    global_neutral = (
        frame[frame["representation_type"].eq("neutral_all")]
        .set_index("participant_id")
        .reindex(test_ids)
    )
    if len(full) != EXPECTED_TEST_N or full[HIDDEN_COLUMNS].isna().any().any():
        raise RuntimeError("Canonical full test representations are incomplete")
    if global_neutral[HIDDEN_COLUMNS].isna().any().any():
        raise RuntimeError("Canonical global-neutral test representations are incomplete")
    return (
        full[HIDDEN_COLUMNS].to_numpy(dtype=np.float32),
        global_neutral[HIDDEN_COLUMNS].to_numpy(dtype=np.float32),
    )


def load_test_panel(test_ids, feature_spec, preprocessor):
    with validation.CONFIG_PATH.open() as handle:
        config = yaml.safe_load(handle)
    saved_schema = json.loads(validation.SCHEMA_PATH.read_text())
    panel = pd.read_parquet(
        validation.MULTIMODAL_PATH,
        filters=[("participant_id", "in", test_ids)],
    )
    panel["participant_id"] = panel["participant_id"].astype(str)
    static = pd.read_parquet(
        validation.STATIC_PATH,
        filters=[("participant_id", "in", test_ids)],
    )
    static["participant_id"] = static["participant_id"].astype(str)
    static = static.drop_duplicates("participant_id")
    added_columns = [
        column
        for column in static.columns
        if column == "participant_id" or column not in panel.columns
    ]
    panel = panel.merge(
        static[added_columns],
        on="participant_id",
        how="left",
        validate="many_to_one",
    )
    if set(panel["participant_id"]) != set(test_ids):
        raise RuntimeError("Test panel participant set is incomplete or contaminated")
    schema = infer_or_validate_schema(panel, saved_schema["schema"])
    prepared = prepare_aireadi_panel(
        panel,
        schema,
        bin_minutes=validation.BIN_MINUTES,
        clean_min_segment_hours=config["dataset"]["clean_min_segment_hours"],
    )
    current_spec = build_stream_feature_spec(
        prepared,
        schema,
        horizon_steps=validation.HORIZON_STEPS,
        bin_minutes=validation.BIN_MINUTES,
    )
    if asdict(current_spec) != asdict(feature_spec):
        raise RuntimeError("Prepared test feature contract differs from checkpoint")
    split = make_aireadi_stream_splits(
        prepared,
        existing_split_path=validation.SPLIT_PATH,
        seed=RANDOM_SEED,
    )
    return prepared, schema, split, static


def sex_sensitivity(
    per_participant: pd.DataFrame,
    static: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    sex = per_participant[
        per_participant["variable"].eq(SEX_VARIABLE)
    ].merge(
        static[["participant_id", SEX_COLUMN]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    nonreference = sex[
        ~sex[SEX_COLUMN].astype(str).eq(SEX_REFERENCE_CATEGORY)
    ].copy()
    rows = []
    definitions = [
        ("all_test_participants", "median", sex, np.median, RANDOM_SEED),
        ("all_test_participants", "mean", sex, np.mean, RANDOM_SEED + 100),
        (
            "participants_differing_from_reference",
            "median",
            nonreference,
            np.median,
            RANDOM_SEED + 200,
        ),
        (
            "participants_differing_from_reference",
            "mean",
            nonreference,
            np.mean,
            RANDOM_SEED + 300,
        ),
    ]
    for population, statistic_name, frame, statistic, seed in definitions:
        state = frame["state_l2"].to_numpy(dtype=float)
        overlap = frame["nn10_overlap"].to_numpy(dtype=float)
        state_low, state_high = bootstrap_ci(state, statistic, seed)
        overlap_low, overlap_high = bootstrap_ci(overlap, statistic, seed + 1)
        rows.append(
            {
                "variable": SEX_VARIABLE,
                "population": population,
                "statistic": statistic_name,
                "n_participants": len(frame),
                "reference_category": SEX_REFERENCE_CATEGORY,
                "observed_categories": ",".join(
                    sorted(frame[SEX_COLUMN].astype(str).unique())
                ),
                "state_l2": statistic(state),
                "state_l2_ci_low": state_low,
                "state_l2_ci_high": state_high,
                "nn10_overlap": statistic(overlap),
                "nn10_ci_low": overlap_low,
                "nn10_ci_high": overlap_high,
            }
        )
    sensitivity = pd.DataFrame(rows)
    conditional = sensitivity[
        sensitivity["population"].eq("participants_differing_from_reference")
        & sensitivity["statistic"].eq("median")
    ].iloc[0]
    return sensitivity, conditional


def bootstrap_ci(values, statistic, seed):
    clean = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(clean),
        size=(BOOTSTRAP_REPLICATES, len(clean)),
    )
    estimates = np.asarray(
        [statistic(clean[index]) for index in indices],
        dtype=float,
    )
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def medication_prevalence(static: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in MEDICATION_VARIABLES:
        column = validation.PRIMARY_COLUMNS[variable]
        values = pd.to_numeric(static[column], errors="coerce")
        exposed = int(values.eq(1).sum())
        unexposed = int(values.eq(0).sum())
        underpowered = exposed < MEDICATION_MIN_GROUP_N or unexposed < MEDICATION_MIN_GROUP_N
        rows.append(
            {
                "variable": variable,
                "test_n": len(static),
                "exposed_n": exposed,
                "unexposed_n": unexposed,
                "missing_n": int(values.isna().sum()),
                "underpowered_by_frozen_threshold": underpowered,
                "minimum_group_n_threshold": MEDICATION_MIN_GROUP_N,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    frozen_plan, frozen_plan_hash = verify_frozen_plan()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    partition_root = OUTPUT_ROOT / "participant_representations"
    completion_root = OUTPUT_ROOT / "_completion"
    partition_root.mkdir()
    completion_root.mkdir()
    validation.configure_logging(OUTPUT_ROOT / "step_c_test_run.log")
    start_time = time.time()
    device = validation.resolve_device("cuda")

    application_start = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_plan_path": str(FROZEN_PLAN_PATH),
        "frozen_plan_sha256": frozen_plan_hash,
        "split": TEST_LABEL,
        "status": "test_application_started_after_plan_freeze",
    }
    validation.write_json(
        OUTPUT_ROOT / "test_application_start.json",
        application_start,
    )

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.use_deterministic_algorithms(True)

    split_manifest = pd.read_csv(validation.SPLIT_PATH, dtype=str)
    test_ids = sorted(
        split_manifest.loc[
            split_manifest["split"].eq(TEST_LABEL),
            "participant_id",
        ].astype(str)
    )
    if len(test_ids) != EXPECTED_TEST_N:
        raise RuntimeError("Test split does not contain exactly 221 participants")
    full_representations, global_representations = canonical_test_full_and_global(
        test_ids
    )

    reference = json.loads(validation.REFERENCE_PATH.read_text())
    if validation.sha256_file(validation.REFERENCE_PATH) != frozen_plan[
        "reference_profile_hash"
    ]:
        raise RuntimeError("Reference profile changed after plan freeze")
    reference_cont = np.asarray(
        reference["transformed_static_cont"],
        dtype=np.float32,
    )
    reference_cat = np.asarray(
        reference["transformed_static_cat"],
        dtype=np.int64,
    )
    model, feature_spec, preprocessor = validation.load_model_and_contract(device)
    prepared, schema, split, test_static = load_test_panel(
        test_ids,
        feature_spec,
        preprocessor,
    )
    validation.LOG.info(
        "Frozen test application starting participants=%d variables=%d device=%s",
        len(test_ids),
        len(VARIABLE_ORDER),
        device,
    )

    participant_frames = []
    qc_max_abs = 0.0
    for participant_number, participant_id in enumerate(test_ids, start=1):
        participant_panel = prepared[
            prepared["participant_id"].astype(str).eq(participant_id)
        ].copy()
        streams = make_participant_streams(
            participant_panel,
            split,
            schema,
            feature_spec=feature_spec,
            preprocessor=preprocessor,
            splits=[TEST_LABEL],
            min_steps=MIN_STREAM_STEPS,
        )
        if not streams:
            raise RuntimeError(f"No test stream for {participant_id}")
        variable_states = {variable: [] for variable in VARIABLE_ORDER}
        factual_states = []
        for raw_stream in streams:
            stream = raw_stream.to(device)
            for variable in VARIABLE_ORDER:
                static_cont, static_cat = validation.intervention_tensors(
                    stream,
                    variable,
                    feature_spec,
                    reference_cont,
                    reference_cat,
                    device,
                )
                states, _ = validation.scan_anchor_states(
                    model,
                    stream,
                    static_cont,
                    static_cat,
                )
                variable_states[variable].append(states)
            if participant_number <= QC_PARTICIPANT_COUNT:
                factual, _ = validation.scan_anchor_states(
                    model,
                    stream,
                    stream.static_cont,
                    stream.static_cat,
                )
                factual_states.append(factual)

        participant_rows = []
        for variable in VARIABLE_ORDER:
            combined = np.concatenate(variable_states[variable], axis=0)
            representation = np.median(combined, axis=0)
            participant_rows.append(
                {
                    "participant_id": participant_id,
                    "split": TEST_LABEL,
                    "variable": variable,
                    "condition": ONE_VARIABLE_CONDITION,
                    "encoding_columns": validation.PRIMARY_COLUMNS[variable],
                    "shared_any_diabetes_drug_disposition": (
                        validation.SHARED_MEDICATION_DISPOSITION
                        if variable in MEDICATION_VARIABLES
                        else "not_applicable"
                    ),
                    "aggregation": "dimensionwise_median",
                    "burn_in_minutes": 0,
                    "representation_frequency_minutes": (
                        validation.REPRESENTATION_FREQUENCY_MINUTES
                    ),
                    "n_anchors": len(combined),
                    "n_segments": len(streams),
                    **dict(zip(HIDDEN_COLUMNS, representation)),
                }
            )
        participant_frame = pd.DataFrame(participant_rows)
        participant_frames.append(participant_frame)

        factual_difference = 0.0
        if factual_states:
            factual_representation = np.median(
                np.concatenate(factual_states, axis=0),
                axis=0,
            )
            factual_difference = float(
                np.max(
                    np.abs(
                        factual_representation
                        - full_representations[participant_number - 1]
                    )
                )
            )
            qc_max_abs = max(qc_max_abs, factual_difference)
            if factual_difference > QC_REPRESENTATION_ABS_TOLERANCE:
                raise RuntimeError(
                    f"Factual test replay mismatch for {participant_id}: "
                    f"{factual_difference}"
                )
        partition_path = (
            partition_root / f"participant_id={participant_id}" / "data.parquet"
        )
        validation.write_parquet(participant_frame, partition_path)
        validation.write_json(
            completion_root / f"{participant_id}.json",
            {
                "participant_id": participant_id,
                "n_segments": len(streams),
                "n_anchors": int(participant_frame["n_anchors"].iloc[0]),
                "partition_path": str(partition_path),
                "partition_sha256": validation.sha256_file(partition_path),
                "factual_replay_qc_max_abs": factual_difference,
            },
        )
        if (
            participant_number % LOG_PROGRESS_INTERVAL == 0
            or participant_number == len(test_ids)
        ):
            elapsed = time.time() - start_time
            remaining = (
                len(test_ids) - participant_number
            ) * elapsed / participant_number
            validation.LOG.info(
                "complete %d/%d elapsed=%.1f min eta=%.1f min",
                participant_number,
                len(test_ids),
                elapsed / 60,
                remaining / 60,
            )

    intervention_frame = pd.concat(participant_frames, ignore_index=True)
    intervention_path = OUTPUT_ROOT / "test_intervention_representations.parquet"
    validation.write_parquet(intervention_frame, intervention_path)
    indexed = intervention_frame.set_index(["variable", "participant_id"])
    representation_arrays = {
        variable: (
            indexed.loc[variable]
            .reindex(test_ids)[HIDDEN_COLUMNS]
            .to_numpy(dtype=np.float32)
        )
        for variable in VARIABLE_ORDER
    }
    representation_arrays[GLOBAL_VARIABLE_LABEL] = global_representations

    kept_dimensions = np.load(validation.PCA_KEEP_PATH)
    pca_feature_order = json.loads(validation.PCA_FEATURE_ORDER_PATH.read_text())
    component_count = int(pca_feature_order["primary_components"])
    if component_count != frozen_plan["pca"]["component_count"]:
        raise RuntimeError("Frozen PCA component count changed")
    scaler = joblib.load(validation.PCA_SCALER_PATH)
    pca = joblib.load(validation.PCA_MODEL_PATH)
    full_scores = validation.transform_frozen_pca(
        full_representations,
        kept_dimensions,
        scaler,
        pca,
        component_count,
    )
    factual_neighbors = validation.nearest_neighbor_indices(full_scores)

    per_participant_rows = []
    summary_rows = []
    analysis_order = VARIABLE_ORDER + [GLOBAL_VARIABLE_LABEL]
    for order, variable in enumerate(analysis_order):
        intervened = representation_arrays[variable]
        state_l2 = np.linalg.norm(full_representations - intervened, axis=1)
        scores = validation.transform_frozen_pca(
            intervened,
            kept_dimensions,
            scaler,
            pca,
            component_count,
        )
        nn_overlap = validation.neighbor_overlap(factual_neighbors, scores)
        l2_low, l2_high = validation.participant_bootstrap_ci(
            state_l2,
            RANDOM_SEED + order * 100,
        )
        nn_low, nn_high = validation.participant_bootstrap_ci(
            nn_overlap,
            RANDOM_SEED + order * 100 + 1,
        )
        condition = (
            GLOBAL_CONDITION
            if variable == GLOBAL_VARIABLE_LABEL
            else ONE_VARIABLE_CONDITION
        )
        summary_rows.append(
            {
                "variable": variable,
                "condition": condition,
                "n_participants": len(test_ids),
                "median_state_l2": float(np.median(state_l2)),
                "l2_ci_low": l2_low,
                "l2_ci_high": l2_high,
                "median_nn10_overlap": float(np.median(nn_overlap)),
                "nn10_ci_low": nn_low,
                "nn10_ci_high": nn_high,
                "prevalence_note": (
                    "global full-card reference"
                    if variable == GLOBAL_VARIABLE_LABEL
                    else "frozen validation definition"
                ),
            }
        )
        for index, participant_id in enumerate(test_ids):
            per_participant_rows.append(
                {
                    "participant_id": participant_id,
                    "split": TEST_LABEL,
                    "variable": variable,
                    "condition": condition,
                    "state_l2": state_l2[index],
                    "nn10_overlap": nn_overlap[index],
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_ROOT / "test_per_variable_static_effects.csv"
    summary.to_csv(summary_path, index=False)
    per_participant = pd.DataFrame(per_participant_rows)
    per_participant_path = (
        OUTPUT_ROOT / "test_per_variable_static_effects_by_participant.csv"
    )
    per_participant.to_csv(per_participant_path, index=False)

    test_static = test_static.copy()
    test_static["participant_id"] = test_static["participant_id"].astype(str)
    sex_table, conditional_sex = sex_sensitivity(per_participant, test_static)
    sex_path = OUTPUT_ROOT / "test_sex_effect_sensitivity.csv"
    sex_table.to_csv(sex_path, index=False)
    presentation = summary.copy()
    sex_index = presentation.index[presentation["variable"].eq(SEX_VARIABLE)][0]
    presentation.loc[sex_index, "n_participants"] = int(
        conditional_sex["n_participants"]
    )
    presentation.loc[sex_index, "median_state_l2"] = conditional_sex["state_l2"]
    presentation.loc[sex_index, "l2_ci_low"] = conditional_sex[
        "state_l2_ci_low"
    ]
    presentation.loc[sex_index, "l2_ci_high"] = conditional_sex[
        "state_l2_ci_high"
    ]
    presentation.loc[sex_index, "median_nn10_overlap"] = conditional_sex[
        "nn10_overlap"
    ]
    presentation.loc[sex_index, "nn10_ci_low"] = conditional_sex[
        "nn10_ci_low"
    ]
    presentation.loc[sex_index, "nn10_ci_high"] = conditional_sex[
        "nn10_ci_high"
    ]
    presentation.loc[sex_index, "prevalence_note"] = (
        f"conditional on differing from reference F; "
        f"n={int(conditional_sex['n_participants'])}"
    )
    presentation_path = (
        OUTPUT_ROOT / "test_per_variable_static_effects_presentation.csv"
    )
    presentation.to_csv(presentation_path, index=False)

    med_prevalence = medication_prevalence(test_static)
    med_prevalence_path = OUTPUT_ROOT / "medication_prevalence_test.csv"
    med_prevalence.to_csv(med_prevalence_path, index=False)
    validation_presentation = pd.read_csv(VALIDATION_PRESENTATION_PATH)
    transport = validation_presentation.merge(
        presentation,
        on="variable",
        suffixes=("_validation", "_test"),
        validate="one_to_one",
    )
    transport["state_l2_test_minus_validation"] = (
        transport["median_state_l2_test"]
        - transport["median_state_l2_validation"]
    )
    transport["nn10_overlap_test_minus_validation"] = (
        transport["median_nn10_overlap_test"]
        - transport["median_nn10_overlap_validation"]
    )
    transport_path = OUTPUT_ROOT / "validation_test_effect_transport.csv"
    transport.to_csv(transport_path, index=False)

    output_paths = [
        intervention_path,
        summary_path,
        per_participant_path,
        sex_path,
        presentation_path,
        med_prevalence_path,
        transport_path,
        OUTPUT_ROOT / "test_application_start.json",
    ]
    manifest = {
        "stage": "step_c_frozen_test_application",
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_plan_path": str(FROZEN_PLAN_PATH),
        "frozen_plan_sha256": frozen_plan_hash,
        "split": TEST_LABEL,
        "participant_count": len(test_ids),
        "participant_ids": test_ids,
        "variables": VARIABLE_ORDER,
        "intervention_forward_passes_per_participant": len(VARIABLE_ORDER),
        "reference_profile_rederived_on_test": False,
        "encoding_groups_rederived_on_test": False,
        "pca_refit_on_test": False,
        "aggregation_changed_on_test": False,
        "medication_shared_any_drug_disposition": (
            validation.SHARED_MEDICATION_DISPOSITION
        ),
        "sex_presentation_rule": frozen_plan["sex_reporting"],
        "pca_component_count": component_count,
        "k_nn": K_NN,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "factual_replay_qc_participants": QC_PARTICIPANT_COUNT,
        "factual_replay_qc_max_abs": qc_max_abs,
        "factual_replay_qc_tolerance": QC_REPRESENTATION_ABS_TOLERANCE,
        "elapsed_seconds": time.time() - start_time,
        "source_hashes_sha256": {
            str(path): validation.sha256_file(path)
            for path in [
                FROZEN_PLAN_PATH,
                CANONICAL_TEST_REPRESENTATIONS_PATH,
                STEP4_MANIFEST_PATH,
                validation.CHECKPOINT_PATH,
                validation.CONFIG_PATH,
                validation.SCHEMA_PATH,
                validation.MULTIMODAL_PATH,
                validation.STATIC_PATH,
                validation.SPLIT_PATH,
                validation.REFERENCE_PATH,
                validation.PCA_SCALER_PATH,
                validation.PCA_MODEL_PATH,
                validation.PCA_KEEP_PATH,
                validation.PCA_FEATURE_ORDER_PATH,
            ]
        },
        "output_hashes_sha256": {
            str(path): validation.sha256_file(path) for path in output_paths
        },
    }
    manifest_path = OUTPUT_ROOT / "step_c_test_manifest.json"
    validation.write_json(manifest_path, manifest)
    validation.LOG.info("Step C frozen test application complete")
    print(presentation.to_string(index=False))
    print(f"Saved test presentation results to {presentation_path}")


if __name__ == "__main__":
    main()
