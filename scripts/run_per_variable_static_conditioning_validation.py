#!/usr/bin/env python3
"""Run gated validation-only one-variable static interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import pairwise_distances

from ssmcgm.data.aireadi import (
    AireadiFeatureSpec,
    AireadiPreprocessor,
    build_stream_feature_spec,
    infer_or_validate_schema,
    make_aireadi_stream_splits,
    make_participant_streams,
    prepare_aireadi_panel,
)
from ssmcgm.models.aireadi_stream import (
    AireadiStreamModel,
    AireadiStreamModelConfig,
)


STEP1_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step1_static_neutralization/"
    "20260724T223612Z"
)
STEP2_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step2_validation_export/"
    "20260724T231513Z"
)
STEP3_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step3_validation_clustering/"
    "20260725T001123Z"
)
AUDIT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/per_variable_static_conditioning_audit"
)
DEFAULT_AUDIT_RUN_ID = "20260727T214535Z"

CHECKPOINT_PATH = (
    REPO_ROOT
    / "outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/"
    "best_model_checkpoint.pt"
)
CONFIG_PATH = REPO_ROOT / "configs/aireadi_stream_full.yaml"
SCHEMA_PATH = (
    REPO_ROOT
    / "outputs/aireadi_stream_mamba_stateful_5epoch/schema_mapping.json"
)
MULTIMODAL_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "final_multimodal_dataset_20260515_184339.parquet"
)
STATIC_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "participant_static_features.parquet"
)
SPLIT_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/"
    "experiment_c_split_adapt6h_seed42/split_participants.csv"
)
REFERENCE_PATH = STEP1_ROOT / "static_reference_profile.json"
STEP1_MANIFEST_PATH = STEP1_ROOT / "step1_manifest.json"
STEP2_MANIFEST_PATH = STEP2_ROOT / "step2_manifest.json"
CANONICAL_REPRESENTATIONS_PATH = STEP2_ROOT / "participant_representations.parquet"
PCA_LOADINGS_PATH = STEP3_ROOT / "pca_loadings.parquet"
PCA_SCALER_PATH = (
    STEP3_ROOT
    / "frozen_validation_pipeline/full_all/full_all_scaler.joblib"
)
PCA_MODEL_PATH = (
    STEP3_ROOT
    / "frozen_validation_pipeline/full_all/full_all_pca.joblib"
)
PCA_KEEP_PATH = (
    STEP3_ROOT
    / "frozen_validation_pipeline/full_all/kept_dimensions.npy"
)
PCA_FEATURE_ORDER_PATH = (
    STEP3_ROOT
    / "frozen_validation_pipeline/full_all/feature_order.json"
)

VARIABLE_ORDER = [
    "age",
    "bmi",
    "hba1c",
    "study_group",
    "sex",
    "insulin",
    "metformin",
    "glp1",
]
PRIMARY_COLUMNS = {
    "age": "participants_age",
    "bmi": "bmi_baseline",
    "hba1c": "hba1c_percent_baseline",
    "study_group": "participants_study_group",
    "sex": "demo_sex_at_birth",
    "insulin": "med_insulin",
    "metformin": "med_metformin",
    "glp1": "med_glp1_or_gip_glp1",
}
MEDICATION_VARIABLES = ["insulin", "metformin", "glp1"]
MEDICATION_PREVALENCE_NOTES = {
    "insulin": "low confidence: exposed n=18 is below 30",
    "metformin": "exposed n=61; unexposed n=178",
    "glp1": "exposed n=40; unexposed n=199",
}
ANY_MEDICATION_COLUMN = "med_any_diabetes_drug"
SHARED_MEDICATION_DISPOSITION = "leave_factual"
GLOBAL_VARIABLE_LABEL = "all static"
ONE_VARIABLE_CONDITION = "one_variable_neutralized"
GLOBAL_CONDITION = "global_static_neutral"
VALIDATION_LABEL = "validation"
RAW_VALIDATION_LABEL = "val"
EXPECTED_VALIDATION_N = 239
EXPECTED_TRAIN_N = 1131
EXPECTED_TEST_N = 221
EXPECTED_STATIC_INPUT_N = 44
HIDDEN_SIZE = 128
HORIZON_STEPS = 12
BIN_MINUTES = 5
REPRESENTATION_FREQUENCY_MINUTES = 15
MIN_STREAM_STEPS = 14
K_NN = 10
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_ALPHA = 0.05
RANDOM_SEED = 42
QC_PARTICIPANT_COUNT = 5
QC_REPRESENTATION_ABS_TOLERANCE = 1e-5
LOG_PROGRESS_INTERVAL = 10
PCA_SPACE = "full_all"
PARQUET_COMPRESSION = "zstd"
HIDDEN_COLUMNS = [f"r_{index:03d}" for index in range(HIDDEN_SIZE)]

LOG = logging.getLogger("per_variable_validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-run-id", default=DEFAULT_AUDIT_RUN_ID)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    LOG.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s",
        "%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(path)
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    LOG.handlers[:] = [file_handler, stream_handler]


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    os.replace(temporary, path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(
        temporary,
        index=False,
        compression=PARQUET_COMPRESSION,
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def participant_bootstrap_ci(
    values: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        low=0,
        high=len(clean),
        size=(BOOTSTRAP_REPLICATES, len(clean)),
    )
    estimates = np.median(clean[indices], axis=1)
    lower = BOOTSTRAP_ALPHA / 2
    upper = 1 - lower
    return tuple(np.quantile(estimates, [lower, upper]))


def canonical_full_and_global(
    validation_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(CANONICAL_REPRESENTATIONS_PATH)
    frame["participant_id"] = frame["participant_id"].astype(str)
    selected = frame[
        frame["balanced_anchor_variant"].eq("all_anchors")
        & frame["burn_in_minutes"].eq(0)
        & frame["representation_type"].isin(["full_all", "neutral_all"])
    ].copy()
    full = (
        selected[selected["representation_type"].eq("full_all")]
        .set_index("participant_id")
        .reindex(validation_ids)
    )
    global_neutral = (
        selected[selected["representation_type"].eq("neutral_all")]
        .set_index("participant_id")
        .reindex(validation_ids)
    )
    if full[HIDDEN_COLUMNS].isna().any().any():
        raise RuntimeError("Canonical full-profile representations are incomplete")
    if global_neutral[HIDDEN_COLUMNS].isna().any().any():
        raise RuntimeError("Canonical global-neutral representations are incomplete")
    return (
        full[HIDDEN_COLUMNS].to_numpy(dtype=np.float32),
        global_neutral[HIDDEN_COLUMNS].to_numpy(dtype=np.float32),
        full["n_anchors"].to_numpy(dtype=int),
    )


def load_model_and_contract(device: str):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    metadata = checkpoint["metadata"]
    feature_spec = AireadiFeatureSpec(**metadata["feature_spec"])
    preprocessor = AireadiPreprocessor.from_jsonable(metadata["preprocessor"])
    model_config = AireadiStreamModelConfig(**metadata["model_config"])
    if (
        len(feature_spec.static_reals) + len(feature_spec.static_categoricals)
        != EXPECTED_STATIC_INPUT_N
    ):
        raise RuntimeError("Checkpoint does not consume exactly 44 static inputs")
    model = AireadiStreamModel(
        feature_spec,
        preprocessor,
        model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    del checkpoint
    return model, feature_spec, preprocessor


def load_validation_panel(
    validation_ids: list[str],
    feature_spec: AireadiFeatureSpec,
    preprocessor: AireadiPreprocessor,
):
    with CONFIG_PATH.open() as handle:
        config = yaml.safe_load(handle)
    saved_schema = json.loads(SCHEMA_PATH.read_text())
    panel = pd.read_parquet(
        MULTIMODAL_PATH,
        filters=[("participant_id", "in", validation_ids)],
    )
    panel["participant_id"] = panel["participant_id"].astype(str)
    static = pd.read_parquet(
        STATIC_PATH,
        filters=[("participant_id", "in", validation_ids)],
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
    if set(panel["participant_id"]) != set(validation_ids):
        raise RuntimeError("Validation panel participant set is incomplete or contaminated")
    schema = infer_or_validate_schema(panel, saved_schema["schema"])
    prepared = prepare_aireadi_panel(
        panel,
        schema,
        bin_minutes=BIN_MINUTES,
        clean_min_segment_hours=config["dataset"]["clean_min_segment_hours"],
    )
    current_spec = build_stream_feature_spec(
        prepared,
        schema,
        horizon_steps=HORIZON_STEPS,
        bin_minutes=BIN_MINUTES,
    )
    if asdict(current_spec) != asdict(feature_spec):
        raise RuntimeError("Prepared validation feature contract differs from checkpoint")
    split = make_aireadi_stream_splits(
        prepared,
        existing_split_path=SPLIT_PATH,
        seed=RANDOM_SEED,
    )
    return prepared, schema, split


def intervention_tensors(
    stream,
    variable: str,
    feature_spec: AireadiFeatureSpec,
    reference_cont: np.ndarray,
    reference_cat: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    factual_cont = stream.static_cont
    factual_cat = stream.static_cat
    intervened_cont = factual_cont.clone()
    intervened_cat = factual_cat.clone()
    column = PRIMARY_COLUMNS[variable]
    if column in feature_spec.static_reals:
        index = feature_spec.static_reals.index(column)
        intervened_cont[index] = torch.as_tensor(
            reference_cont[index],
            dtype=intervened_cont.dtype,
            device=device,
        )
        other = torch.arange(len(intervened_cont), device=device).ne(index)
        if not torch.equal(intervened_cont[other], factual_cont[other]):
            raise RuntimeError(f"{variable} changed a non-target continuous channel")
        if not torch.equal(intervened_cat, factual_cat):
            raise RuntimeError(f"{variable} changed a categorical channel")
    elif column in feature_spec.static_categoricals:
        index = feature_spec.static_categoricals.index(column)
        intervened_cat[index] = torch.as_tensor(
            reference_cat[index],
            dtype=intervened_cat.dtype,
            device=device,
        )
        other = torch.arange(len(intervened_cat), device=device).ne(index)
        if not torch.equal(intervened_cat[other], factual_cat[other]):
            raise RuntimeError(f"{variable} changed a non-target categorical channel")
        if not torch.equal(intervened_cont, factual_cont):
            raise RuntimeError(f"{variable} changed a continuous channel")
    else:
        raise RuntimeError(f"{variable} is absent from the checkpoint static inputs")
    if variable in MEDICATION_VARIABLES:
        any_index = feature_spec.static_reals.index(ANY_MEDICATION_COLUMN)
        if intervened_cont[any_index].item() != factual_cont[any_index].item():
            raise RuntimeError("The confirmed factual any-drug channel was changed")
    return intervened_cont, intervened_cat


@torch.no_grad()
def scan_anchor_states(
    model: AireadiStreamModel,
    stream,
    static_cont: torch.Tensor,
    static_cat: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    context = model.encode_static(static_cat, static_cont)
    initial_state = model.init_stream(context)
    _, output = model.scan_chunk(stream.dynamic, context, initial_state)
    states = output[0].float().cpu().numpy()
    if states.shape != (stream.n_steps, HIDDEN_SIZE):
        raise RuntimeError(f"Unexpected hidden-state shape {states.shape}")
    observed = stream.observed.cpu().numpy().astype(bool)
    positions = np.arange(stream.n_steps, dtype=int)
    minutes_since_reset = (positions + 1) * BIN_MINUTES
    anchor_positions = np.flatnonzero(
        observed
        & np.equal(
            np.mod(minutes_since_reset, REPRESENTATION_FREQUENCY_MINUTES),
            0,
        )
    )
    selected = states[anchor_positions]
    if not np.isfinite(selected).all():
        raise RuntimeError("Nonfinite intervened hidden state")
    return selected, anchor_positions


def transform_frozen_pca(
    representations: np.ndarray,
    kept_dimensions: np.ndarray,
    scaler,
    pca,
    component_count: int,
) -> np.ndarray:
    kept = representations[:, kept_dimensions]
    return pca.transform(scaler.transform(kept))[:, :component_count]


def nearest_neighbor_indices(scores: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(scores, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1)[:, :K_NN]


def neighbor_overlap(
    factual_neighbors: np.ndarray,
    intervened_scores: np.ndarray,
) -> np.ndarray:
    intervened_neighbors = nearest_neighbor_indices(intervened_scores)
    return np.asarray(
        [
            len(set(factual_neighbors[index]) & set(intervened_neighbors[index]))
            / K_NN
            for index in range(len(factual_neighbors))
        ],
        dtype=float,
    )


def main() -> None:
    args = parse_args()
    output_root = AUDIT_OUTPUT_ROOT / args.audit_run_id
    schema_audit_root = output_root / "schema_audit"
    output_dir = output_root / "validation_interventions"
    output_dir.mkdir(parents=True, exist_ok=False)
    partition_root = output_dir / "participant_representations"
    completion_root = output_dir / "_completion"
    partition_root.mkdir()
    completion_root.mkdir()
    configure_logging(output_dir / "step_b_run.log")

    start_time = time.time()
    device = resolve_device(args.device)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.use_deterministic_algorithms(True)

    step_a_manifest = json.loads(
        (schema_audit_root / "step_a_manifest.json").read_text()
    )
    if step_a_manifest["forward_pass_executed"]:
        raise RuntimeError("Step A manifest unexpectedly reports a forward pass")
    mapping = pd.read_csv(
        schema_audit_root / "variable_encoding_group_audit.csv"
    )
    if set(mapping["variable"]) != set(VARIABLE_ORDER):
        raise RuntimeError("Step A variable mapping differs from requested variables")
    confirmation = {
        "confirmed_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation": "option_1",
        "medication_intervention_columns": {
            variable: [PRIMARY_COLUMNS[variable]]
            for variable in MEDICATION_VARIABLES
        },
        "shared_any_diabetes_drug_disposition": SHARED_MEDICATION_DISPOSITION,
        "test_access_authorized": False,
    }
    write_json(output_dir / "step_a_confirmation.json", confirmation)

    split_manifest = pd.read_csv(SPLIT_PATH, dtype=str)
    split_counts = split_manifest["split"].value_counts().to_dict()
    expected_counts = {
        "train": EXPECTED_TRAIN_N,
        RAW_VALIDATION_LABEL: EXPECTED_VALIDATION_N,
        "test": EXPECTED_TEST_N,
    }
    if split_counts != expected_counts:
        raise RuntimeError(f"Unexpected split counts: {split_counts}")
    validation_ids = sorted(
        split_manifest.loc[
            split_manifest["split"].eq(RAW_VALIDATION_LABEL),
            "participant_id",
        ].astype(str)
    )
    if len(validation_ids) != EXPECTED_VALIDATION_N:
        raise RuntimeError("Validation cohort is not exactly 239 participants")

    step2_manifest = json.loads(STEP2_MANIFEST_PATH.read_text())
    if sorted(step2_manifest["validation_participant_ids"]) != validation_ids:
        raise RuntimeError("Step 2 validation cohort differs from split manifest")

    reference = json.loads(REFERENCE_PATH.read_text())
    reference_cont = np.asarray(
        reference["transformed_static_cont"],
        dtype=np.float32,
    )
    reference_cat = np.asarray(
        reference["transformed_static_cat"],
        dtype=np.int64,
    )
    full_representations, global_representations, canonical_anchor_counts = (
        canonical_full_and_global(validation_ids)
    )

    model, feature_spec, preprocessor = load_model_and_contract(device)
    prepared, schema, split = load_validation_panel(
        validation_ids,
        feature_spec,
        preprocessor,
    )
    if prepared["participant_id"].astype(str).nunique() != EXPECTED_VALIDATION_N:
        raise RuntimeError("Prepared panel does not contain 239 validation participants")

    LOG.info(
        "Step B starting validation participants=%d variables=%d device=%s",
        len(validation_ids),
        len(VARIABLE_ORDER),
        device,
    )
    rows: list[pd.DataFrame] = []
    qc_max_abs = 0.0
    for participant_number, participant_id in enumerate(validation_ids, start=1):
        participant_panel = prepared[
            prepared["participant_id"].astype(str).eq(participant_id)
        ].copy()
        streams = make_participant_streams(
            participant_panel,
            split,
            schema,
            feature_spec=feature_spec,
            preprocessor=preprocessor,
            splits=[VALIDATION_LABEL],
            min_steps=MIN_STREAM_STEPS,
        )
        if not streams:
            raise RuntimeError(f"No validation stream for {participant_id}")
        variable_states: dict[str, list[np.ndarray]] = {
            variable: [] for variable in VARIABLE_ORDER
        }
        factual_states: list[np.ndarray] = []
        segment_anchor_count = 0
        for raw_stream in streams:
            stream = raw_stream.to(device)
            for variable in VARIABLE_ORDER:
                static_cont, static_cat = intervention_tensors(
                    stream,
                    variable,
                    feature_spec,
                    reference_cont,
                    reference_cat,
                    device,
                )
                states, positions = scan_anchor_states(
                    model,
                    stream,
                    static_cont,
                    static_cat,
                )
                variable_states[variable].append(states)
                if variable == VARIABLE_ORDER[0]:
                    segment_anchor_count += len(positions)
            if participant_number <= QC_PARTICIPANT_COUNT:
                factual, _ = scan_anchor_states(
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
                    "split": VALIDATION_LABEL,
                    "variable": variable,
                    "condition": ONE_VARIABLE_CONDITION,
                    "encoding_columns": PRIMARY_COLUMNS[variable],
                    "shared_any_diabetes_drug_disposition": (
                        SHARED_MEDICATION_DISPOSITION
                        if variable in MEDICATION_VARIABLES
                        else "not_applicable"
                    ),
                    "aggregation": "dimensionwise_median",
                    "burn_in_minutes": 0,
                    "representation_frequency_minutes": (
                        REPRESENTATION_FREQUENCY_MINUTES
                    ),
                    "n_anchors": len(combined),
                    "n_segments": len(streams),
                    **dict(zip(HIDDEN_COLUMNS, representation)),
                }
            )
        participant_frame = pd.DataFrame(participant_rows)
        if segment_anchor_count != canonical_anchor_counts[participant_number - 1]:
            raise RuntimeError(
                f"Anchor count differs from canonical Step 2 for {participant_id}: "
                f"{segment_anchor_count} versus "
                f"{canonical_anchor_counts[participant_number - 1]}"
            )
        if factual_states:
            factual_representation = np.median(
                np.concatenate(factual_states, axis=0),
                axis=0,
            )
            difference = float(
                np.max(
                    np.abs(
                        factual_representation
                        - full_representations[participant_number - 1]
                    )
                )
            )
            qc_max_abs = max(qc_max_abs, difference)
            if difference > QC_REPRESENTATION_ABS_TOLERANCE:
                raise RuntimeError(
                    f"Factual representation replay mismatch for {participant_id}: "
                    f"{difference}"
                )
        partition_path = (
            partition_root / f"participant_id={participant_id}" / "data.parquet"
        )
        write_parquet(participant_frame, partition_path)
        write_json(
            completion_root / f"{participant_id}.json",
            {
                "participant_id": participant_id,
                "n_segments": len(streams),
                "n_anchors": segment_anchor_count,
                "partition_path": str(partition_path),
                "partition_sha256": sha256_file(partition_path),
            },
        )
        rows.append(participant_frame)
        if (
            participant_number % LOG_PROGRESS_INTERVAL == 0
            or participant_number == len(validation_ids)
        ):
            elapsed = time.time() - start_time
            remaining = (
                len(validation_ids) - participant_number
            ) * elapsed / participant_number
            LOG.info(
                "complete %d/%d elapsed=%.1f min eta=%.1f min",
                participant_number,
                len(validation_ids),
                elapsed / 60,
                remaining / 60,
            )

    intervention_frame = pd.concat(rows, ignore_index=True)
    combined_representation_path = (
        output_dir / "validation_intervention_representations.parquet"
    )
    write_parquet(intervention_frame, combined_representation_path)

    representation_arrays: dict[str, np.ndarray] = {}
    indexed = intervention_frame.set_index(["variable", "participant_id"])
    for variable in VARIABLE_ORDER:
        representation_arrays[variable] = (
            indexed.loc[variable]
            .reindex(validation_ids)[HIDDEN_COLUMNS]
            .to_numpy(dtype=np.float32)
        )
    representation_arrays[GLOBAL_VARIABLE_LABEL] = global_representations

    kept_dimensions = np.load(PCA_KEEP_PATH)
    pca_feature_order = json.loads(PCA_FEATURE_ORDER_PATH.read_text())
    component_count = int(pca_feature_order["primary_components"])
    scaler = joblib.load(PCA_SCALER_PATH)
    pca = joblib.load(PCA_MODEL_PATH)
    full_scores = transform_frozen_pca(
        full_representations,
        kept_dimensions,
        scaler,
        pca,
        component_count,
    )
    factual_neighbors = nearest_neighbor_indices(full_scores)

    per_participant_rows = []
    summary_rows = []
    analysis_order = VARIABLE_ORDER + [GLOBAL_VARIABLE_LABEL]
    for order, variable in enumerate(analysis_order):
        intervened = representation_arrays[variable]
        state_l2 = np.linalg.norm(full_representations - intervened, axis=1)
        intervened_scores = transform_frozen_pca(
            intervened,
            kept_dimensions,
            scaler,
            pca,
            component_count,
        )
        nn_overlap = neighbor_overlap(factual_neighbors, intervened_scores)
        l2_ci_low, l2_ci_high = participant_bootstrap_ci(
            state_l2,
            RANDOM_SEED + order * 100,
        )
        nn_ci_low, nn_ci_high = participant_bootstrap_ci(
            nn_overlap,
            RANDOM_SEED + order * 100 + 1,
        )
        condition = (
            GLOBAL_CONDITION
            if variable == GLOBAL_VARIABLE_LABEL
            else ONE_VARIABLE_CONDITION
        )
        prevalence_note = (
            "global full-card reference"
            if variable == GLOBAL_VARIABLE_LABEL
            else MEDICATION_PREVALENCE_NOTES.get(variable, "not applicable")
        )
        summary_rows.append(
            {
                "variable": variable,
                "condition": condition,
                "n_participants": len(validation_ids),
                "median_state_l2": float(np.median(state_l2)),
                "l2_ci_low": l2_ci_low,
                "l2_ci_high": l2_ci_high,
                "median_nn10_overlap": float(np.median(nn_overlap)),
                "nn10_ci_low": nn_ci_low,
                "nn10_ci_high": nn_ci_high,
                "prevalence_note": prevalence_note,
            }
        )
        for index, participant_id in enumerate(validation_ids):
            per_participant_rows.append(
                {
                    "participant_id": participant_id,
                    "split": VALIDATION_LABEL,
                    "variable": variable,
                    "condition": condition,
                    "state_l2": state_l2[index],
                    "nn10_overlap": nn_overlap[index],
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "validation_per_variable_static_effects.csv"
    summary.to_csv(summary_path, index=False)
    per_participant = pd.DataFrame(per_participant_rows)
    per_participant_path = (
        output_dir / "validation_per_variable_static_effects_by_participant.csv"
    )
    per_participant.to_csv(per_participant_path, index=False)

    source_paths = {
        "checkpoint": CHECKPOINT_PATH,
        "config": CONFIG_PATH,
        "schema": SCHEMA_PATH,
        "multimodal": MULTIMODAL_PATH,
        "static": STATIC_PATH,
        "split": SPLIT_PATH,
        "step1_reference": REFERENCE_PATH,
        "step1_manifest": STEP1_MANIFEST_PATH,
        "step2_manifest": STEP2_MANIFEST_PATH,
        "canonical_representations": CANONICAL_REPRESENTATIONS_PATH,
        "pca_loadings": PCA_LOADINGS_PATH,
        "pca_scaler": PCA_SCALER_PATH,
        "pca_model": PCA_MODEL_PATH,
        "pca_kept_dimensions": PCA_KEEP_PATH,
        "pca_feature_order": PCA_FEATURE_ORDER_PATH,
        "step_a_manifest": schema_audit_root / "step_a_manifest.json",
        "step_a_mapping": (
            schema_audit_root / "variable_encoding_group_audit.csv"
        ),
    }
    manifest = {
        "stage": "step_b_validation_one_variable_interventions",
        "status": "complete_paused_before_test_freeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_run_id": args.audit_run_id,
        "split_loaded": VALIDATION_LABEL,
        "validation_participant_count": len(validation_ids),
        "test_participant_forward_pass_count": 0,
        "variables": VARIABLE_ORDER,
        "global_reference_label": GLOBAL_VARIABLE_LABEL,
        "medication_option": "option_1",
        "shared_any_diabetes_drug_disposition": SHARED_MEDICATION_DISPOSITION,
        "aggregation": "dimensionwise median over valid CGM 15-minute anchors",
        "burn_in_minutes": 0,
        "state_metric": "participant representation L2",
        "neighbor_metric": "participant NN10 overlap",
        "k_nn": K_NN,
        "pca_space": PCA_SPACE,
        "pca_primary_component_count": component_count,
        "bootstrap_unit": "participant",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_interval": "percentile 95%",
        "random_seed": RANDOM_SEED,
        "factual_replay_qc_participants": QC_PARTICIPANT_COUNT,
        "factual_replay_qc_max_abs": qc_max_abs,
        "factual_replay_qc_tolerance": QC_REPRESENTATION_ABS_TOLERANCE,
        "source_hashes_sha256": {
            key: sha256_file(path) for key, path in source_paths.items()
        },
        "output_hashes_sha256": {
            str(path): sha256_file(path)
            for path in [
                summary_path,
                per_participant_path,
                combined_representation_path,
                output_dir / "step_a_confirmation.json",
            ]
        },
        "elapsed_seconds": time.time() - start_time,
    }
    manifest_path = output_dir / "step_b_manifest.json"
    write_json(manifest_path, manifest)
    (AUDIT_OUTPUT_ROOT / "LATEST_STEP_B_RUN.txt").write_text(
        str(output_dir) + "\n"
    )
    LOG.info("Step B complete and paused before test")
    print(summary.to_string(index=False))
    print(f"Saved validation results to {summary_path}")


if __name__ == "__main__":
    main()
