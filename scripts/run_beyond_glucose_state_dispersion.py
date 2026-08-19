#!/usr/bin/env python3
"""Step 1 exploratory hidden-state dispersion analysis.

This script only reads frozen per-anchor hidden-state exports. It does not
replay the forecasting model, regenerate hidden states, alter frozen analyses,
or reopen a confirmatory family.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.decomposition import randomized_svd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_hidden_state_clinical_probes import (
    SNUM,
    SCAT,
    build_baselines,
    estimator,
    forward,
    inverse,
)
from ssmcgm.analysis.neighbor_clinical_sharing import (
    pairwise_euclidean,
    participant_metrics_and_weights,
)


STEP2_DIR = ROOT / (
    "outputs/hidden_state_phenotype/step2_validation_export/"
    "20260724T231513Z"
)
STEP3_DIR = ROOT / (
    "outputs/hidden_state_phenotype/step3_validation_clustering/"
    "20260725T001123Z"
)
STEP3B_DIR = ROOT / (
    "outputs/hidden_state_phenotype/step3b_exploratory_k2_freeze/"
    "20260725T005617Z"
)
STEP4_DIR = ROOT / (
    "outputs/hidden_state_phenotype/step4_test_confirmation/"
    "20260725T010440Z"
)
STEP5_DIR = ROOT / (
    "outputs/hidden_state_phenotype/step5_clinical_probes/"
    "20260725T022634Z"
)
PANEL_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "final_multimodal_dataset_20260515_184339.parquet"
)
STATIC_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "participant_static_features.parquet"
)
OUTPUT_ROOT = ROOT / (
    "outputs/hidden_state_phenotype/beyond_glucose_dynamics/"
    "step1_dispersion"
)

SPLIT_DIRECTORIES = {
    "validation": STEP2_DIR / "validation_hidden_states",
    "test": STEP4_DIR / "test_hidden_states",
}
EXPECTED_PARTICIPANTS = {"validation": 239, "test": 221}
CONDITION_DIRECTORIES = {
    "full_all": "full_profile",
    "neutral_all": "static_neutral",
}
CONDITION_LABELS = {
    "full_all": "Full profile",
    "neutral_all": "Static neutral",
}
STATE_COLUMNS = [f"h_{index:03d}" for index in range(128)]
MEDIAN_COLUMNS = [f"state_{index:03d}" for index in range(128)]
IQR_COLUMNS = [f"iqr_h_{index:03d}" for index in range(128)]
DISPERSION_RAW_FEATURES = [
    "iqr_l2",
    "covariance_trace",
    "cov_eigenvalue_1",
    "cov_eigenvalue_2",
    "cov_eigenvalue_3",
    "cov_eigenvalue_4",
    "cov_eigenvalue_5",
    "leading_eigenvalue_fraction",
]
DISPERSION_FEATURES = [f"log1p_{name}" for name in DISPERSION_RAW_FEATURES]
TOP_EIGENVALUES = 5
RANDOMIZED_SVD_ITERATIONS = 5
RANDOMIZED_SVD_OVERSAMPLES = 10
ANCHOR_INTERVAL_MINUTES = 15
MINIMUM_ELIGIBLE_ANCHORS = 48
K_NEIGHBORS = 10
RANDOM_BASELINE_REPEATS = 2000
BOOTSTRAP_REPLICATES = 2000
POWER_TARGET = 0.80
ALPHA = 0.05
SMALL_NEIGHBOR_GAIN = 0.10
SMALL_DELTA_R2 = 0.05
OUTER_REPETITIONS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 5
RANDOM_SEED = 42
NO_EM_DASH = "\u2014"

TARGETS = [
    "c_reactive_protein_i",
    "natriuretic_peptide_b_prohormon",
    "bun_creatinine_ratio",
]
TARGET_LABELS = {
    "c_reactive_protein_i": "High-sensitivity CRP",
    "natriuretic_peptide_b_prohormon": "NT-proBNP",
    "bun_creatinine_ratio": "BUN/creatinine ratio",
}
NEIGHBOR_VARIABLES = [
    ("mean_glucose", "Mean glucose", "continuous"),
    ("glucose_cv", "Glucose CV", "continuous"),
    ("tir_70_180", "Time in range", "continuous"),
    ("glucose_sd", "Glucose SD", "continuous"),
    ("hba1c", "HbA1c", "continuous"),
    ("study_group", "Study group", "categorical"),
    (
        "natriuretic_peptide_b_prohormon",
        "NT-proBNP",
        "continuous",
    ),
    ("c_reactive_protein_i", "High-sensitivity CRP", "continuous"),
    ("bun_creatinine_ratio", "BUN/creatinine ratio", "continuous"),
]
STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
LOG = logging.getLogger("beyond_glucose_dispersion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES
    )
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
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


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any, base_seed: int = RANDOM_SEED) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="little"
    )
    return int((value + base_seed) % (2**32 - 1))


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def setup_logging(path: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    handlers = [logging.FileHandler(path), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(formatter)
    LOG.handlers[:] = handlers
    LOG.setLevel(logging.INFO)


def parquet_schema_record(path: Path, role: str) -> tuple[dict[str, Any], str]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    record = {
        "role": role,
        "path": str(path.resolve()),
        "format": "parquet",
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": len(schema.names),
        "schema_sha256": hashlib.sha256(str(schema).encode()).hexdigest(),
        "file_sha256": sha256_file(path),
    }
    text = (
        f"ROLE: {role}\nPATH: {path.resolve()}\n"
        f"ROWS: {parquet.metadata.num_rows}\nSCHEMA:\n{schema}\n"
    )
    return record, text


def csv_schema_record(path: Path, role: str) -> tuple[dict[str, Any], str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    record = {
        "role": role,
        "path": str(path.resolve()),
        "format": "csv",
        "rows": rows,
        "row_groups": np.nan,
        "columns": len(header),
        "schema_sha256": hashlib.sha256(
            "|".join(header).encode()
        ).hexdigest(),
        "file_sha256": sha256_file(path),
    }
    text = (
        f"ROLE: {role}\nPATH: {path.resolve()}\nROWS: {rows}\n"
        f"COLUMNS:\n{header}\n"
    )
    return record, text


def save_input_schemas(output_directory: Path) -> pd.DataFrame:
    paths: list[tuple[Path, str, str]] = []
    for split, root in SPLIT_DIRECTORIES.items():
        for condition, directory_name in CONDITION_DIRECTORIES.items():
            files = sorted(
                (root / f"condition={directory_name}").glob(
                    "participant_id=*/data.parquet"
                )
            )
            if not files:
                raise FileNotFoundError(
                    f"No per-anchor partitions for {split} {condition}"
                )
            paths.append(
                (
                    files[0],
                    f"{split}_{condition}_per_anchor_sample",
                    "parquet",
                )
            )
    paths.extend(
        [
            (
                STEP2_DIR / "participant_representations.parquet",
                "validation_median_representations",
                "parquet",
            ),
            (
                STEP4_DIR / "test_participant_representations.parquet",
                "test_median_representations",
                "parquet",
            ),
            (
                STEP3_DIR / "validation_glycemic_nuisance_features.parquet",
                "validation_clinical_features",
                "parquet",
            ),
            (
                STEP4_DIR / "test_glycemic_nuisance_features.parquet",
                "test_clinical_features",
                "parquet",
            ),
            (
                STEP3_DIR / "validation_external_targets.parquet",
                "validation_external_targets",
                "parquet",
            ),
            (
                STEP4_DIR / "test_external_targets.parquet",
                "test_external_targets",
                "parquet",
            ),
            (
                STEP5_DIR / "validation_probe_predictions.parquet",
                "validation_frozen_probe_predictions",
                "parquet",
            ),
            (
                STEP5_DIR / "test_probe_predictions.parquet",
                "test_frozen_probe_predictions",
                "parquet",
            ),
            (
                STEP2_DIR / "validation_export_status_by_participant.csv",
                "validation_export_status",
                "csv",
            ),
            (
                STEP4_DIR / "test_export_status_by_participant.csv",
                "test_export_status",
                "csv",
            ),
        ]
    )
    records: list[dict[str, Any]] = []
    text_blocks: list[str] = []
    for path, role, format_name in paths:
        if format_name == "parquet":
            record, text = parquet_schema_record(path, role)
        else:
            record, text = csv_schema_record(path, role)
        records.append(record)
        text_blocks.append(text)
        print(text)
    inventory = pd.DataFrame(records)
    write_csv(output_directory / "input_schema_inventory.csv", inventory)
    (output_directory / "input_schema_printout.txt").write_text(
        "\n".join(text_blocks)
    )
    return inventory


def state_partition_paths(split: str, condition: str) -> list[Path]:
    directory_name = CONDITION_DIRECTORIES[condition]
    return sorted(
        (
            SPLIT_DIRECTORIES[split] / f"condition={directory_name}"
        ).glob("participant_id=*/data.parquet")
    )


def audit_anchor_availability(output_directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    identifier_sets: dict[tuple[str, str], set[str]] = {}
    required_columns = {
        "participant_id",
        "split",
        "condition",
        "minutes_since_reset",
        "is_h0_row",
        "is_post_update_state",
        *STATE_COLUMNS,
    }
    for split in ("validation", "test"):
        for condition in ("full_all", "neutral_all"):
            files = state_partition_paths(split, condition)
            identifiers = {
                path.parent.name.replace("participant_id=", "") for path in files
            }
            identifier_sets[(split, condition)] = identifiers
            total_rows = 0
            eligible_rows = 0
            participants_below_minimum = 0
            schema_mismatch = 0
            for path in files:
                parquet = pq.ParquetFile(path)
                total_rows += parquet.metadata.num_rows
                if not required_columns.issubset(parquet.schema_arrow.names):
                    schema_mismatch += 1
                    continue
                flags = pd.read_parquet(
                    path,
                    columns=[
                        "minutes_since_reset",
                        "is_h0_row",
                        "is_post_update_state",
                    ],
                )
                eligible = (
                    (~flags["is_h0_row"])
                    & flags["is_post_update_state"]
                    & (
                        flags["minutes_since_reset"]
                        % ANCHOR_INTERVAL_MINUTES
                        == 0
                    )
                )
                count = int(eligible.sum())
                eligible_rows += count
                participants_below_minimum += int(
                    count < MINIMUM_ELIGIBLE_ANCHORS
                )
            rows.append(
                {
                    "split": split,
                    "condition": condition,
                    "partition_count": len(files),
                    "unique_participants": len(identifiers),
                    "expected_participants": EXPECTED_PARTICIPANTS[split],
                    "total_saved_state_rows": total_rows,
                    "eligible_15min_post_update_rows": eligible_rows,
                    "participants_below_minimum": participants_below_minimum,
                    "schema_mismatch_partitions": schema_mismatch,
                    "availability_status": (
                        "available"
                        if len(files) == EXPECTED_PARTICIPANTS[split]
                        and participants_below_minimum == 0
                        and schema_mismatch == 0
                        else "blocked"
                    ),
                }
            )
    for split in ("validation", "test"):
        if (
            identifier_sets[(split, "full_all")]
            != identifier_sets[(split, "neutral_all")]
        ):
            raise RuntimeError(f"Condition participant mismatch for {split}")
    audit = pd.DataFrame(rows)
    write_csv(output_directory / "per_anchor_availability_audit.csv", audit)
    if set(audit["availability_status"]) != {"available"}:
        raise RuntimeError("Per-anchor availability gate failed")
    return audit


def freeze_plan(output_directory: Path, bootstrap_replicates: int) -> Path:
    plan = {
        "stage": "step1_state_dispersion",
        "analysis_role": (
            "exploratory additive dynamics analysis; no confirmatory claims"
        ),
        "state_source": "existing frozen per-anchor exports only",
        "model_replay": False,
        "state_regeneration": False,
        "participants": EXPECTED_PARTICIPANTS,
        "conditions": ["full_all", "neutral_all"],
        "anchor_eligibility": {
            "post_update_only": True,
            "exclude_h0": True,
            "minutes_since_reset_modulo": ANCHOR_INTERVAL_MINUTES,
            "minimum_anchors": MINIMUM_ELIGIBLE_ANCHORS,
            "burn_in_minutes": 0,
        },
        "dispersion_definition": {
            "per_dimension_iqr": True,
            "iqr_summary": "L2 norm of the 128-dimensional IQR vector",
            "covariance_trace": True,
            "top_covariance_eigenvalues": TOP_EIGENVALUES,
            "top_eigenvalue_method": (
                "deterministic randomized SVD of participant-centered anchors"
            ),
            "randomized_svd_iterations": RANDOMIZED_SVD_ITERATIONS,
            "randomized_svd_oversamples": RANDOMIZED_SVD_OVERSAMPLES,
            "compact_features": DISPERSION_RAW_FEATURES,
            "analysis_transform": "elementwise log1p then validation scaling",
        },
        "neighbor_sharing": {
            "variables": [item[0] for item in NEIGHBOR_VARIABLES],
            "k_neighbors": K_NEIGHBORS,
            "distance": "Euclidean in validation-scaled compact dispersion",
            "random_baseline_repeats": RANDOM_BASELINE_REPEATS,
            "bootstrap_unit": "focal participant",
            "bootstrap_replicates": bootstrap_replicates,
            "p_values": False,
            "fdr_family": None,
        },
        "incremental_value": {
            "targets": TARGETS,
            "baseline": "unchanged Step 5 simple baseline",
            "comparisons": [
                "median_only versus baseline",
                "dispersion_only versus baseline",
                "median_plus_dispersion versus baseline",
                "median_plus_dispersion versus median_only",
            ],
            "model": "Ridge regression with unchanged Step 5 preprocessing",
            "outer_repetitions": OUTER_REPETITIONS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "folds": "reuse frozen Step 5 outer participant assignments",
            "test": "validation-selected pipeline transported without test tuning",
            "bootstrap_unit": "participant retaining repeated OOF rows",
            "bootstrap_replicates": bootstrap_replicates,
            "p_values": False,
            "fdr_family": None,
        },
        "minimum_detectable_effect": {
            "method": (
                "shift centered participant-bootstrap errors and find the "
                "smallest positive effect with at least 80 percent two-sided "
                "rejection"
            ),
            "power": POWER_TARGET,
            "alpha": ALPHA,
            "neighbor_adequacy_threshold": SMALL_NEIGHBOR_GAIN,
            "incremental_adequacy_threshold": SMALL_DELTA_R2,
        },
        "random_seed": RANDOM_SEED,
        "pause_after_step": True,
        "prohibitions": [
            "no forecasting replay",
            "no hidden-state regeneration",
            "no target selection",
            "no test tuning",
            "no confirmatory language",
            "no modification of frozen Tier 1 results",
        ],
    }
    path = output_directory / "step1_analysis_plan_frozen.json"
    write_json(path, plan)
    return path


def eligible_state_array(path: Path) -> tuple[str, np.ndarray, int]:
    columns = [
        "participant_id",
        "minutes_since_reset",
        "is_h0_row",
        "is_post_update_state",
        *STATE_COLUMNS,
    ]
    frame = pd.read_parquet(path, columns=columns)
    participant_values = frame["participant_id"].astype(str).unique()
    if len(participant_values) != 1:
        raise RuntimeError(f"Partition does not contain exactly one participant: {path}")
    eligible = (
        (~frame["is_h0_row"])
        & frame["is_post_update_state"]
        & (
            frame["minutes_since_reset"] % ANCHOR_INTERVAL_MINUTES
            == 0
        )
    )
    values = frame.loc[eligible, STATE_COLUMNS].to_numpy(
        dtype=np.float64, copy=True
    )
    if len(values) < MINIMUM_ELIGIBLE_ANCHORS:
        raise RuntimeError(
            f"Insufficient anchors for {participant_values[0]}: {len(values)}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(f"Nonfinite state values in {path}")
    return participant_values[0], values, len(frame)


def summarize_state_array(
    participant_id: str,
    values: np.ndarray,
    split: str,
    condition: str,
    saved_rows: int,
) -> dict[str, Any]:
    quartiles = np.percentile(values, [25.0, 75.0], axis=0)
    iqr = quartiles[1] - quartiles[0]
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, _ = randomized_svd(
        centered,
        n_components=TOP_EIGENVALUES,
        n_iter=RANDOMIZED_SVD_ITERATIONS,
        n_oversamples=RANDOMIZED_SVD_OVERSAMPLES,
        random_state=stable_seed(participant_id, split, condition, "svd"),
        flip_sign=True,
    )
    eigenvalues = singular_values**2 / (len(values) - 1)
    covariance_trace = float(
        np.sum(centered * centered) / (len(values) - 1)
    )
    row: dict[str, Any] = {
        "participant_id": participant_id,
        "split": split,
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "n_saved_rows": saved_rows,
        "n_eligible_anchors": len(values),
        "iqr_l2": float(np.linalg.norm(iqr)),
        "covariance_trace": covariance_trace,
        "leading_eigenvalue_fraction": (
            float(eigenvalues[0] / covariance_trace)
            if covariance_trace > 0
            else np.nan
        ),
        "top5_eigenvalue_fraction": (
            float(eigenvalues.sum() / covariance_trace)
            if covariance_trace > 0
            else np.nan
        ),
    }
    for index, value in enumerate(iqr):
        row[IQR_COLUMNS[index]] = float(value)
    for index, value in enumerate(eigenvalues, start=1):
        row[f"cov_eigenvalue_{index}"] = float(value)
        row[f"cov_eigenvalue_fraction_{index}"] = (
            float(value / covariance_trace)
            if covariance_trace > 0
            else np.nan
        )
    for raw_name, transformed_name in zip(
        DISPERSION_RAW_FEATURES, DISPERSION_FEATURES
    ):
        row[transformed_name] = float(np.log1p(row[raw_name]))
    return row


def compute_dispersion_summaries(output_directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for condition in ("full_all", "neutral_all"):
            paths = state_partition_paths(split, condition)
            for index, path in enumerate(paths, start=1):
                participant_id, values, saved_rows = eligible_state_array(path)
                rows.append(
                    summarize_state_array(
                        participant_id,
                        values,
                        split,
                        condition,
                        saved_rows,
                    )
                )
                if index % 25 == 0 or index == len(paths):
                    LOG.info(
                        "dispersion %s %s %d/%d",
                        split,
                        condition,
                        index,
                        len(paths),
                    )
    frame = pd.DataFrame(rows)
    expected_rows = 2 * sum(EXPECTED_PARTICIPANTS.values())
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} dispersion rows, found {len(frame)}"
        )
    if frame[DISPERSION_FEATURES].isna().any().any():
        raise RuntimeError("Nonfinite compact dispersion feature")
    write_parquet(output_directory / "participant_state_dispersion.parquet", frame)
    coverage = (
        frame.groupby(["split", "condition"], as_index=False)
        .agg(
            participants=("participant_id", "nunique"),
            median_eligible_anchors=("n_eligible_anchors", "median"),
            minimum_eligible_anchors=("n_eligible_anchors", "min"),
            maximum_eligible_anchors=("n_eligible_anchors", "max"),
            median_iqr_l2=("iqr_l2", "median"),
            median_covariance_trace=("covariance_trace", "median"),
            median_leading_eigenvalue_fraction=(
                "leading_eigenvalue_fraction",
                "median",
            ),
        )
    )
    write_csv(output_directory / "dispersion_summary_by_split_condition.csv", coverage)
    return frame


def load_median_representations() -> dict[str, dict[str, pd.DataFrame]]:
    validation = pd.read_parquet(STEP2_DIR / "participant_representations.parquet")
    test = pd.read_parquet(STEP4_DIR / "test_participant_representations.parquet")
    validation["participant_id"] = validation["participant_id"].astype(str)
    test["participant_id"] = test["participant_id"].astype(str)
    output: dict[str, dict[str, pd.DataFrame]] = {
        "validation": {},
        "test": {},
    }
    for condition in ("full_all", "neutral_all"):
        selected = validation[
            (validation["representation_type"] == condition)
            & (validation["balanced_anchor_variant"] == "all_anchors")
            & (validation["burn_in_minutes"] == 0)
        ][["participant_id", *[f"r_{index:03d}" for index in range(128)]]]
        selected = selected.copy()
        selected.columns = ["participant_id", *MEDIAN_COLUMNS]
        if len(selected) != EXPECTED_PARTICIPANTS["validation"]:
            raise RuntimeError(f"Validation median cohort mismatch: {condition}")
        output["validation"][condition] = selected
        selected = test[test["representation_type"] == condition][
            ["participant_id", *[f"r_{index:03d}" for index in range(128)]]
        ].copy()
        selected.columns = ["participant_id", *MEDIAN_COLUMNS]
        if len(selected) != EXPECTED_PARTICIPANTS["test"]:
            raise RuntimeError(f"Test median cohort mismatch: {condition}")
        output["test"][condition] = selected
    return output


def bootstrap_mean_samples(
    values: np.ndarray, repeats: int, seed: int
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    output = np.empty(repeats, dtype=float)
    batch_size = 200
    for start in range(0, repeats, batch_size):
        stop = min(start + batch_size, repeats)
        indices = rng.integers(
            0, len(values), size=(stop - start, len(values))
        )
        output[start:stop] = values[indices].mean(axis=1)
    return output


def empirical_mde(
    bootstrap_estimates: np.ndarray,
    observed_effect: float,
    power_target: float = POWER_TARGET,
    alpha: float = ALPHA,
) -> float:
    errors = np.asarray(bootstrap_estimates, dtype=float) - observed_effect
    lower, upper = np.quantile(errors, [alpha / 2, 1 - alpha / 2])
    standard_error = float(np.std(errors, ddof=1))
    upper_search = max(
        abs(lower),
        abs(upper),
        standard_error,
        np.finfo(float).eps,
    ) * 10.0
    grid = np.linspace(0.0, upper_search, 20001)
    rejection = np.mean(
        (errors[None, :] + grid[:, None] < lower)
        | (errors[None, :] + grid[:, None] > upper),
        axis=1,
    )
    eligible = np.flatnonzero(rejection >= power_target)
    return float(grid[eligible[0]]) if len(eligible) else np.nan


def prepare_clinical_frames() -> dict[str, pd.DataFrame]:
    validation = pd.read_parquet(
        STEP3_DIR / "validation_glycemic_nuisance_features.parquet"
    )
    test = pd.read_parquet(
        STEP4_DIR / "test_glycemic_nuisance_features.parquet"
    )
    validation["participant_id"] = validation["participant_id"].astype(str)
    test["participant_id"] = test["participant_id"].astype(str)
    for split, base, target_path in [
        (
            "validation",
            validation,
            STEP3_DIR / "validation_external_targets.parquet",
        ),
        ("test", test, STEP4_DIR / "test_external_targets.parquet"),
    ]:
        external = pd.read_parquet(target_path)
        external["participant_id"] = external["participant_id"].astype(str)
        external = external[external["eligible_for_analysis"]]
        wide = external.pivot(
            index="participant_id",
            columns="target_name",
            values="analysis_value",
        ).reset_index()
        base.drop(columns=TARGETS, errors="ignore", inplace=True)
        merged = base.merge(wide, on="participant_id", how="left")
        if merged["participant_id"].nunique() != EXPECTED_PARTICIPANTS[split]:
            raise RuntimeError(f"Clinical cohort mismatch for {split}")
        if split == "validation":
            validation = merged
        else:
            test = merged
    return {"validation": validation, "test": test}


def run_neighbor_sharing(
    output_directory: Path,
    dispersion: pd.DataFrame,
    clinical_frames: dict[str, pd.DataFrame],
    bootstrap_replicates: int,
) -> pd.DataFrame:
    result_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    scalers: dict[str, StandardScaler] = {}
    for condition in ("full_all", "neutral_all"):
        validation_dispersion = dispersion[
            (dispersion["split"] == "validation")
            & (dispersion["condition"] == condition)
        ].sort_values("participant_id")
        scaler = StandardScaler().fit(
            validation_dispersion[DISPERSION_FEATURES]
        )
        scalers[condition] = scaler
        joblib.dump(
            scaler,
            output_directory / f"validation_dispersion_scaler_{condition}.joblib",
        )
    for split in ("validation", "test"):
        clinical = clinical_frames[split].copy()
        clinical["participant_id"] = clinical["participant_id"].astype(str)
        for condition in ("full_all", "neutral_all"):
            selected = dispersion[
                (dispersion["split"] == split)
                & (dispersion["condition"] == condition)
            ].sort_values("participant_id")
            participant_ids = selected["participant_id"].tolist()
            clinical_selected = (
                clinical.set_index("participant_id")
                .reindex(participant_ids)
                .reset_index()
            )
            if clinical_selected["clinical_site"].isna().any():
                raise RuntimeError(f"Missing clinical row for {split} {condition}")
            scores = scalers[condition].transform(
                selected[DISPERSION_FEATURES]
            )
            distances = pairwise_euclidean(scores)
            np.fill_diagonal(distances, np.inf)
            neighbors = np.argsort(distances, axis=1)[:, :K_NEIGHBORS]
            sites = clinical_selected["clinical_site"].astype(str).to_numpy()
            for variable, label, variable_type in NEIGHBOR_VARIABLES:
                values = clinical_selected[variable].to_numpy()
                focal_rows, _, _, _ = participant_metrics_and_weights(
                    values=values,
                    variable_type=variable_type,
                    sites=sites,
                    neighbors=neighbors,
                    condition=f"{split}_{condition}_dispersion",
                    variable=variable,
                    k_neighbors=K_NEIGHBORS,
                    random_repeats=RANDOM_BASELINE_REPEATS,
                    seed=RANDOM_SEED,
                )
                focal = pd.DataFrame(focal_rows)
                focal["participant_id"] = [
                    participant_ids[index] for index in focal["focal_index"]
                ]
                focal["split"] = split
                focal["condition"] = condition
                focal["variable"] = variable
                participant_rows.extend(
                    focal.drop(columns=["focal_index"]).to_dict("records")
                )
                values_for_bootstrap = focal["sharing_gain"].to_numpy(float)
                estimate = float(np.nanmean(values_for_bootstrap))
                bootstrap = bootstrap_mean_samples(
                    values_for_bootstrap,
                    bootstrap_replicates,
                    stable_seed(split, condition, variable, "neighbor_bootstrap"),
                )
                ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
                standard_error = float(np.std(bootstrap, ddof=1))
                minimum_detectable = empirical_mde(bootstrap, estimate)
                result_rows.append(
                    {
                        "split": split,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "variable": variable,
                        "variable_label": label,
                        "variable_type": variable_type,
                        "n_participants": len(focal),
                        "k_neighbors": K_NEIGHBORS,
                        "standardized_similarity_gain": estimate,
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "bootstrap_standard_error": standard_error,
                        "minimum_detectable_effect_80": minimum_detectable,
                        "adequacy_threshold": SMALL_NEIGHBOR_GAIN,
                        "power_verdict": (
                            "adequately powered"
                            if minimum_detectable <= SMALL_NEIGHBOR_GAIN
                            else "underpowered"
                        ),
                        "bootstrap_replicates": bootstrap_replicates,
                        "analysis_role": "exploratory",
                        "p_value": np.nan,
                        "fdr_q": np.nan,
                    }
                )
    participant_frame = pd.DataFrame(participant_rows)
    result_frame = pd.DataFrame(result_rows)
    write_parquet(
        output_directory / "dispersion_neighbor_sharing_by_participant.parquet",
        participant_frame,
    )
    write_csv(
        output_directory / "dispersion_neighbor_sharing_results.csv",
        result_frame,
    )
    return result_frame


def dispersion_feature_frame(
    dispersion: pd.DataFrame, split: str, condition: str
) -> pd.DataFrame:
    selected = dispersion[
        (dispersion["split"] == split)
        & (dispersion["condition"] == condition)
    ][["participant_id", *DISPERSION_FEATURES]].copy()
    rename = {
        feature: f"dispersion_{condition}_{feature}"
        for feature in DISPERSION_FEATURES
    }
    return selected.rename(columns=rename)


def weighted_r2(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return np.nan
    mean = np.sum(weights * y) / weight_sum
    denominator = np.sum(weights * (y - mean) ** 2)
    if denominator <= 0:
        return np.nan
    numerator = np.sum(weights * (y - prediction) ** 2)
    return float(1.0 - numerator / denominator)


def aligned_delta_bootstrap(
    baseline: pd.DataFrame,
    augmented: pd.DataFrame,
    repeats: int,
    seed: int,
) -> tuple[float, float, float, float, float, np.ndarray]:
    repeated = not baseline["outer_repetition"].isna().all()
    keys = (
        ["participant_id", "outer_repetition", "outer_fold"]
        if repeated
        else ["participant_id"]
    )
    merged = baseline[
        [*keys, "observed_transformed", "predicted_transformed"]
    ].merge(
        augmented[[*keys, "predicted_transformed"]],
        on=keys,
        suffixes=("_baseline", "_augmented"),
        validate="one_to_one",
    )
    y = merged["observed_transformed"].to_numpy(float)
    baseline_prediction = merged["predicted_transformed_baseline"].to_numpy(
        float
    )
    augmented_prediction = merged["predicted_transformed_augmented"].to_numpy(
        float
    )
    baseline_r2 = r2_score(y, baseline_prediction)
    augmented_r2 = r2_score(y, augmented_prediction)
    observed_delta = float(augmented_r2 - baseline_r2)
    identifiers, codes = np.unique(
        merged["participant_id"].astype(str), return_inverse=True
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats, dtype=float)
    probability = np.repeat(1.0 / len(identifiers), len(identifiers))
    for index in range(repeats):
        counts = rng.multinomial(len(identifiers), probability)
        weights = counts[codes].astype(float)
        bootstrap[index] = weighted_r2(
            y, augmented_prediction, weights
        ) - weighted_r2(y, baseline_prediction, weights)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    standard_error = float(np.std(bootstrap, ddof=1))
    return (
        float(baseline_r2),
        float(augmented_r2),
        observed_delta,
        float(ci_low),
        float(ci_high),
        standard_error,
        bootstrap,
    )


def run_incremental_probes(
    output_directory: Path,
    dispersion: pd.DataFrame,
    median_representations: dict[str, dict[str, pd.DataFrame]],
    bootstrap_replicates: int,
    n_jobs: int,
) -> pd.DataFrame:
    step2_manifest = json.loads((STEP2_DIR / "step2_manifest.json").read_text())
    step4_manifest = json.loads((STEP4_DIR / "step4_manifest.json").read_text())
    validation_ids = sorted(map(str, step2_manifest["validation_participant_ids"]))
    test_ids = sorted(map(str, step4_manifest["participant_ids"]))
    validation_base, test_base, _, _ = build_baselines(
        STEP3_DIR,
        STEP4_DIR,
        PANEL_PATH,
        STATIC_PATH,
        validation_ids,
        test_ids,
        [],
        [],
        STEP3B_DIR,
    )
    dispersion_frames: dict[str, dict[str, pd.DataFrame]] = {
        split: {
            condition: dispersion_feature_frame(dispersion, split, condition)
            for condition in ("full_all", "neutral_all")
        }
        for split in ("validation", "test")
    }
    target_validation = pd.read_parquet(
        STEP3_DIR / "validation_external_targets.parquet"
    )
    target_test = pd.read_parquet(STEP4_DIR / "test_external_targets.parquet")
    target_validation["participant_id"] = target_validation[
        "participant_id"
    ].astype(str)
    target_test["participant_id"] = target_test["participant_id"].astype(str)
    frozen_validation_predictions = pd.read_parquet(
        STEP5_DIR / "validation_probe_predictions.parquet"
    )
    frozen_test_predictions = pd.read_parquet(
        STEP5_DIR / "test_probe_predictions.parquet"
    )
    frozen_validation_predictions["participant_id"] = (
        frozen_validation_predictions["participant_id"].astype(str)
    )
    frozen_test_predictions["participant_id"] = frozen_test_predictions[
        "participant_id"
    ].astype(str)
    feature_sets: dict[str, tuple[str, str]] = {}
    for condition in ("full_all", "neutral_all"):
        feature_sets[f"dispersion_{condition}"] = (
            condition,
            "dispersion_only",
        )
        feature_sets[f"median_plus_dispersion_{condition}"] = (
            condition,
            "median_plus_dispersion",
        )
    predictions: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    model_directory = output_directory / "frozen_dispersion_probe_models"
    model_directory.mkdir()
    for target in TARGETS:
        validation_target = target_validation[
            (target_validation["target_name"] == target)
            & target_validation["eligible_for_analysis"]
        ][["participant_id", "analysis_value"]]
        test_target = target_test[
            (target_test["target_name"] == target)
            & target_test["eligible_for_analysis"]
        ][["participant_id", "analysis_value"]]
        validation_frame = validation_base.merge(
            validation_target, on="participant_id", validate="one_to_one"
        )
        test_frame = test_base.merge(
            test_target, on="participant_id", validate="one_to_one"
        )
        validation_frame["observed_transformed"] = forward(
            target, validation_frame["analysis_value"].to_numpy(float)
        )
        test_frame["observed_transformed"] = forward(
            target, test_frame["analysis_value"].to_numpy(float)
        )
        fold_source = frozen_validation_predictions[
            (frozen_validation_predictions["target"] == target)
            & (
                frozen_validation_predictions["feature_set"]
                == "simple_baseline"
            )
        ][["participant_id", "outer_repetition", "outer_fold"]]
        if len(fold_source) != len(validation_frame) * OUTER_REPETITIONS:
            raise RuntimeError(f"Frozen fold count mismatch for {target}")
        for feature_set, (condition, mode) in feature_sets.items():
            validation_data = validation_frame.merge(
                dispersion_frames["validation"][condition],
                on="participant_id",
                validate="one_to_one",
            )
            test_data = test_frame.merge(
                dispersion_frames["test"][condition],
                on="participant_id",
                validate="one_to_one",
            )
            hidden_features = [
                f"dispersion_{condition}_{feature}"
                for feature in DISPERSION_FEATURES
            ]
            if mode == "median_plus_dispersion":
                validation_data = validation_data.merge(
                    median_representations["validation"][condition],
                    on="participant_id",
                    validate="one_to_one",
                )
                test_data = test_data.merge(
                    median_representations["test"][condition],
                    on="participant_id",
                    validate="one_to_one",
                )
                hidden_features = [*MEDIAN_COLUMNS, *hidden_features]
            for repetition in range(OUTER_REPETITIONS):
                repetition_folds = fold_source[
                    fold_source["outer_repetition"] == repetition
                ]
                mapping = dict(
                    zip(
                        repetition_folds["participant_id"],
                        repetition_folds["outer_fold"],
                    )
                )
                for fold in range(OUTER_FOLDS):
                    test_mask = (
                        validation_data["participant_id"].map(mapping) == fold
                    ).to_numpy()
                    train_indices = np.flatnonzero(~test_mask)
                    test_indices = np.flatnonzero(test_mask)
                    pipeline, grid = estimator(
                        SNUM, SCAT, hidden_features, elastic=False
                    )
                    inner = KFold(
                        INNER_FOLDS,
                        shuffle=True,
                        random_state=RANDOM_SEED
                        + 1000 * repetition
                        + fold,
                    )
                    search = GridSearchCV(
                        pipeline,
                        grid,
                        cv=inner,
                        scoring="neg_mean_squared_error",
                        n_jobs=n_jobs,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(
                        validation_data.iloc[train_indices],
                        validation_data.iloc[train_indices][
                            "observed_transformed"
                        ],
                    )
                    predicted = search.predict(
                        validation_data.iloc[test_indices]
                    )
                    selected_alpha = search.best_params_["model__alpha"]
                    hyperparameters.append(
                        {
                            "target": target,
                            "feature_set": feature_set,
                            "stage": "validation_outer",
                            "outer_repetition": repetition,
                            "outer_fold": fold,
                            "selected_alpha": selected_alpha,
                            "inner_best_neg_mse": search.best_score_,
                        }
                    )
                    for row_index, predicted_value in zip(
                        test_indices, predicted
                    ):
                        row = validation_data.iloc[row_index]
                        predictions.append(
                            {
                                "participant_id": row["participant_id"],
                                "target": target,
                                "outer_repetition": repetition,
                                "outer_fold": fold,
                                "feature_set": feature_set,
                                "observed_raw": row["analysis_value"],
                                "observed_transformed": row[
                                    "observed_transformed"
                                ],
                                "predicted_transformed": predicted_value,
                                "predicted_raw_when_invertible": inverse(
                                    target, [predicted_value]
                                )[0],
                                "selected_alpha": selected_alpha,
                                "model_status": "nested_validation_held_out",
                            }
                        )
            pipeline, grid = estimator(
                SNUM, SCAT, hidden_features, elastic=False
            )
            inner = KFold(
                INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED + 9000
            )
            search = GridSearchCV(
                pipeline,
                grid,
                cv=inner,
                scoring="neg_mean_squared_error",
                n_jobs=n_jobs,
                refit=True,
                error_score="raise",
            )
            search.fit(
                validation_data, validation_data["observed_transformed"]
            )
            model = search.best_estimator_
            predicted = model.predict(test_data)
            selected_alpha = search.best_params_["model__alpha"]
            joblib.dump(
                model,
                model_directory / f"{target}__{feature_set}.joblib",
            )
            hyperparameters.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "stage": "final_validation_fit",
                    "outer_repetition": np.nan,
                    "outer_fold": np.nan,
                    "selected_alpha": selected_alpha,
                    "inner_best_neg_mse": search.best_score_,
                }
            )
            for row_index, predicted_value in enumerate(predicted):
                row = test_data.iloc[row_index]
                predictions.append(
                    {
                        "participant_id": row["participant_id"],
                        "target": target,
                        "outer_repetition": np.nan,
                        "outer_fold": np.nan,
                        "feature_set": feature_set,
                        "observed_raw": row["analysis_value"],
                        "observed_transformed": row["observed_transformed"],
                        "predicted_transformed": predicted_value,
                        "predicted_raw_when_invertible": inverse(
                            target, [predicted_value]
                        )[0],
                        "selected_alpha": selected_alpha,
                        "model_status": (
                            "frozen_validation_pipeline_test_transport"
                        ),
                    }
                )
            LOG.info("probe target=%s feature_set=%s complete", target, feature_set)
    new_predictions = pd.DataFrame(predictions)
    new_validation = new_predictions[
        new_predictions["model_status"] == "nested_validation_held_out"
    ]
    new_test = new_predictions[
        new_predictions["model_status"]
        == "frozen_validation_pipeline_test_transport"
    ]
    keep_frozen = ["simple_baseline", "simple_plus_full_all", "simple_plus_neutral_all"]
    frozen_validation = frozen_validation_predictions[
        frozen_validation_predictions["feature_set"].isin(keep_frozen)
    ].copy()
    frozen_test = frozen_test_predictions[
        frozen_test_predictions["feature_set"].isin(keep_frozen)
    ].copy()
    validation_predictions = pd.concat(
        [frozen_validation, new_validation], ignore_index=True
    )
    test_predictions = pd.concat([frozen_test, new_test], ignore_index=True)
    write_parquet(
        output_directory / "validation_dispersion_probe_predictions.parquet",
        validation_predictions,
    )
    write_parquet(
        output_directory / "test_dispersion_probe_predictions.parquet",
        test_predictions,
    )
    write_csv(
        output_directory / "dispersion_probe_hyperparameters.csv",
        pd.DataFrame(hyperparameters),
    )
    comparison_rows: list[dict[str, Any]] = []
    split_predictions = {
        "validation_nested_cv": validation_predictions,
        "test_transport": test_predictions,
    }
    for split_label, frame in split_predictions.items():
        for target in TARGETS:
            target_frame = frame[frame["target"] == target]
            simple_baseline = target_frame[
                target_frame["feature_set"] == "simple_baseline"
            ]
            for condition in ("full_all", "neutral_all"):
                median_name = f"simple_plus_{condition}"
                dispersion_name = f"dispersion_{condition}"
                combined_name = f"median_plus_dispersion_{condition}"
                comparisons = [
                    (
                        "median_only_vs_baseline",
                        simple_baseline,
                        target_frame[
                            target_frame["feature_set"] == median_name
                        ],
                    ),
                    (
                        "dispersion_only_vs_baseline",
                        simple_baseline,
                        target_frame[
                            target_frame["feature_set"] == dispersion_name
                        ],
                    ),
                    (
                        "median_plus_dispersion_vs_baseline",
                        simple_baseline,
                        target_frame[
                            target_frame["feature_set"] == combined_name
                        ],
                    ),
                    (
                        "dispersion_added_beyond_median",
                        target_frame[
                            target_frame["feature_set"] == median_name
                        ],
                        target_frame[
                            target_frame["feature_set"] == combined_name
                        ],
                    ),
                ]
                for comparison, baseline, augmented in comparisons:
                    (
                        baseline_r2,
                        augmented_r2,
                        delta_r2,
                        ci_low,
                        ci_high,
                        standard_error,
                        bootstrap,
                    ) = aligned_delta_bootstrap(
                        baseline,
                        augmented,
                        bootstrap_replicates,
                        stable_seed(
                            split_label,
                            target,
                            condition,
                            comparison,
                            "probe_bootstrap",
                        ),
                    )
                    minimum_detectable = empirical_mde(bootstrap, delta_r2)
                    comparison_rows.append(
                        {
                            "split": split_label,
                            "target": target,
                            "target_label": TARGET_LABELS[target],
                            "condition": condition,
                            "condition_label": CONDITION_LABELS[condition],
                            "comparison": comparison,
                            "n_participants": baseline[
                                "participant_id"
                            ].nunique(),
                            "baseline_r2": baseline_r2,
                            "augmented_r2": augmented_r2,
                            "delta_r2": delta_r2,
                            "bootstrap_ci_low": ci_low,
                            "bootstrap_ci_high": ci_high,
                            "bootstrap_standard_error": standard_error,
                            "minimum_detectable_effect_80": minimum_detectable,
                            "adequacy_threshold": SMALL_DELTA_R2,
                            "power_verdict": (
                                "adequately powered"
                                if minimum_detectable <= SMALL_DELTA_R2
                                else "underpowered"
                            ),
                            "bootstrap_replicates": bootstrap_replicates,
                            "analysis_role": "exploratory",
                            "p_value": np.nan,
                            "fdr_q": np.nan,
                        }
                    )
    results = pd.DataFrame(comparison_rows)
    write_csv(
        output_directory / "dispersion_incremental_value_results.csv", results
    )
    return results


def make_figures(
    output_directory: Path,
    dispersion: pd.DataFrame,
    neighbor_results: pd.DataFrame,
    incremental_results: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric, label in [
        (axes[0], "iqr_l2", "IQR vector L2 norm"),
        (axes[1], "covariance_trace", "Within-participant covariance trace"),
    ]:
        sns.boxplot(
            data=dispersion,
            x="split",
            y=metric,
            hue="condition_label",
            palette=STRATUM_COLORS[:2],
            showfliers=False,
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel(label)
        axis.set_title(label)
    figure.suptitle("Hidden-state dispersion from frozen per-anchor states")
    figure.tight_layout()
    figure.savefig(
        output_directory / "fig_state_dispersion_summary.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    selected_neighbor = neighbor_results[
        neighbor_results["split"] == "test"
    ].copy()
    variable_order = [item[1] for item in NEIGHBOR_VARIABLES]
    figure, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for axis, condition, color in zip(
        axes,
        ("full_all", "neutral_all"),
        STRATUM_COLORS[:2],
    ):
        selected = (
            selected_neighbor[
                selected_neighbor["condition"] == condition
            ]
            .set_index("variable_label")
            .reindex(variable_order)
            .reset_index()
        )
        y_positions = np.arange(len(selected))
        axis.errorbar(
            selected["standardized_similarity_gain"],
            y_positions,
            xerr=[
                selected["standardized_similarity_gain"]
                - selected["bootstrap_ci_low"],
                selected["bootstrap_ci_high"]
                - selected["standardized_similarity_gain"],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
        )
        axis.axvline(0.0, color=STRATUM_COLORS[4], linewidth=1)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(variable_order)
        axis.set_xlabel("Standardized similarity gain")
        axis.set_title(CONDITION_LABELS[condition])
    figure.suptitle("Test clinical sharing in state-dispersion space")
    figure.tight_layout()
    figure.savefig(
        output_directory / "fig_dispersion_neighbor_sharing_test.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    selected_incremental = incremental_results[
        (
            incremental_results["comparison"]
            == "dispersion_added_beyond_median"
        )
    ].copy()
    selected_incremental["row_label"] = (
        selected_incremental["target_label"]
        + " | "
        + selected_incremental["condition_label"]
    )
    row_order = [
        f"{TARGET_LABELS[target]} | {CONDITION_LABELS[condition]}"
        for target in TARGETS
        for condition in ("full_all", "neutral_all")
    ]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for axis, split_label in zip(
        axes, ("validation_nested_cv", "test_transport")
    ):
        selected = (
            selected_incremental[
                selected_incremental["split"] == split_label
            ]
            .set_index("row_label")
            .reindex(row_order)
            .reset_index()
        )
        y_positions = np.arange(len(selected))
        colors = [
            STRATUM_COLORS[0]
            if condition == "Full profile"
            else STRATUM_COLORS[1]
            for condition in selected["condition_label"]
        ]
        for index, row in selected.iterrows():
            axis.errorbar(
                row["delta_r2"],
                y_positions[index],
                xerr=[
                    [row["delta_r2"] - row["bootstrap_ci_low"]],
                    [row["bootstrap_ci_high"] - row["delta_r2"]],
                ],
                fmt="o",
                color=colors[index],
                ecolor=colors[index],
                capsize=3,
            )
        axis.axvline(0.0, color=STRATUM_COLORS[4], linewidth=1)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(row_order)
        axis.set_xlabel("Delta R squared beyond participant median")
        axis.set_title(
            "Validation nested CV"
            if split_label == "validation_nested_cv"
            else "Test transport"
        )
    figure.suptitle("Incremental value of hidden-state dispersion")
    figure.tight_layout()
    figure.savefig(
        output_directory / "fig_dispersion_incremental_value.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_report(
    output_directory: Path,
    availability: pd.DataFrame,
    dispersion: pd.DataFrame,
    neighbor_results: pd.DataFrame,
    incremental_results: pd.DataFrame,
) -> None:
    dispersion_summary = (
        dispersion.groupby(["split", "condition_label"], as_index=False)
        .agg(
            n=("participant_id", "nunique"),
            median_anchors=("n_eligible_anchors", "median"),
            median_iqr_l2=("iqr_l2", "median"),
            median_covariance_trace=("covariance_trace", "median"),
            median_leading_fraction=("leading_eigenvalue_fraction", "median"),
        )
    )
    neighbor_test = neighbor_results[
        neighbor_results["split"] == "test"
    ][
        [
            "condition_label",
            "variable_label",
            "n_participants",
            "standardized_similarity_gain",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "minimum_detectable_effect_80",
            "power_verdict",
        ]
    ]
    incremental_focus = incremental_results[
        incremental_results["comparison"].isin(
            [
                "dispersion_only_vs_baseline",
                "dispersion_added_beyond_median",
            ]
        )
    ][
        [
            "split",
            "target_label",
            "condition_label",
            "comparison",
            "n_participants",
            "delta_r2",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "minimum_detectable_effect_80",
            "power_verdict",
        ]
    ]
    report = [
        "# Step 1 state-dispersion analysis",
        "",
        "## Scope",
        "",
        (
            "This exploratory analysis used only the existing frozen per-anchor "
            "hidden-state exports. It did not replay the model, regenerate "
            "states, tune on test, or change any frozen confirmatory result."
        ),
        "",
        "## Per-anchor availability gate",
        "",
        markdown_table(availability),
        "",
        "## Dispersion summaries",
        "",
        markdown_table(dispersion_summary),
        "",
        "## Test neighbor-sharing results",
        "",
        markdown_table(neighbor_test),
        "",
        "## Incremental-value results",
        "",
        markdown_table(incremental_focus),
        "",
        "## Power interpretation",
        "",
        (
            "Every neighbor-sharing and incremental-value result carries an "
            "empirical 80 percent minimum detectable effect calculated from "
            "its participant-bootstrap distribution. A confidence interval "
            "that includes zero is interpreted against that detectable-effect "
            "floor, so an underpowered null means no detectable effect above "
            "the reported floor rather than evidence of no effect."
        ),
        "",
        "## Required pause",
        "",
        (
            "Step 1 is complete. Excursion-response, exercise, and added-target "
            "analyses remain unexecuted pending confirmation."
        ),
        "",
    ]
    (output_directory / "step1_dispersion_report.md").write_text(
        "\n".join(report)
    )


def scan_no_em_dash(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        if path.suffix.lower() not in {
            ".py",
            ".md",
            ".json",
            ".csv",
            ".txt",
        }:
            continue
        if NO_EM_DASH in path.read_text(errors="ignore"):
            failures.append(str(path))
    return failures


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < BOOTSTRAP_REPLICATES:
        raise ValueError("At least 2,000 bootstrap replicates are required")
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output_directory = OUTPUT_ROOT / run_id
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    setup_logging(output_directory / "step1_run.log")
    started = time.time()

    schema_inventory = save_input_schemas(output_directory)
    availability = audit_anchor_availability(output_directory)
    plan_path = freeze_plan(output_directory, args.bootstrap_replicates)
    plan_hash = sha256_file(plan_path)
    plan_frozen_at = datetime.now(timezone.utc).isoformat()
    LOG.info("analysis plan frozen before target values hash=%s", plan_hash)

    dispersion = compute_dispersion_summaries(output_directory)
    median_representations = load_median_representations()
    target_values_loaded_at = datetime.now(timezone.utc).isoformat()
    clinical_frames = prepare_clinical_frames()
    neighbor_results = run_neighbor_sharing(
        output_directory,
        dispersion,
        clinical_frames,
        args.bootstrap_replicates,
    )
    incremental_results = run_incremental_probes(
        output_directory,
        dispersion,
        median_representations,
        args.bootstrap_replicates,
        args.n_jobs,
    )
    make_figures(
        output_directory, dispersion, neighbor_results, incremental_results
    )
    make_report(
        output_directory,
        availability,
        dispersion,
        neighbor_results,
        incremental_results,
    )

    required_outputs = [
        "input_schema_inventory.csv",
        "input_schema_printout.txt",
        "per_anchor_availability_audit.csv",
        "step1_analysis_plan_frozen.json",
        "participant_state_dispersion.parquet",
        "dispersion_summary_by_split_condition.csv",
        "dispersion_neighbor_sharing_by_participant.parquet",
        "dispersion_neighbor_sharing_results.csv",
        "validation_dispersion_probe_predictions.parquet",
        "test_dispersion_probe_predictions.parquet",
        "dispersion_incremental_value_results.csv",
        "fig_state_dispersion_summary.png",
        "fig_dispersion_neighbor_sharing_test.png",
        "fig_dispersion_incremental_value.png",
        "step1_dispersion_report.md",
    ]
    missing = [
        name for name in required_outputs if not (output_directory / name).exists()
    ]
    text_paths = [
        Path(__file__),
        *[
            path
            for path in output_directory.iterdir()
            if path.is_file()
        ],
    ]
    em_dash_failures = scan_no_em_dash(text_paths)
    qc = {
        "status": "QC_COMPLETE"
        if not missing and not em_dash_failures
        else "QC_FAILED",
        "analysis_plan_frozen_before_target_load": (
            plan_frozen_at < target_values_loaded_at
        ),
        "analysis_plan_hash": plan_hash,
        "availability_gate": set(availability["availability_status"])
        == {"available"},
        "validation_participants": EXPECTED_PARTICIPANTS["validation"],
        "test_participants": EXPECTED_PARTICIPANTS["test"],
        "dispersion_rows": len(dispersion),
        "neighbor_result_rows": len(neighbor_results),
        "incremental_result_rows": len(incremental_results),
        "neighbor_mde_complete": bool(
            neighbor_results["minimum_detectable_effect_80"].notna().all()
        ),
        "incremental_mde_complete": bool(
            incremental_results["minimum_detectable_effect_80"].notna().all()
        ),
        "missing_outputs": missing,
        "em_dash_failures": em_dash_failures,
        "model_replay_executed": False,
        "state_regeneration_executed": False,
        "step2_excursion_executed": False,
        "pause_required": True,
        "schema_inventory_rows": len(schema_inventory),
        "blockers": [],
    }
    if not all(
        [
            qc["analysis_plan_frozen_before_target_load"],
            qc["availability_gate"],
            qc["neighbor_mde_complete"],
            qc["incremental_mde_complete"],
            not missing,
            not em_dash_failures,
        ]
    ):
        qc["status"] = "QC_FAILED"
        qc["blockers"].append("One or more Step 1 integrity checks failed")
    write_json(output_directory / "step1_independent_qc.json", qc)
    if qc["status"] != "QC_COMPLETE":
        raise RuntimeError(json.dumps(qc, default=json_default))

    input_hashes = {
        row.role: row.file_sha256
        for row in schema_inventory.itertuples(index=False)
    }
    output_hashes = {
        path.name: sha256_file(path)
        for path in output_directory.iterdir()
        if path.is_file() and path.name != "step1_manifest.json"
    }
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "QC_COMPLETE",
        "analysis_plan_hash": plan_hash,
        "analysis_plan_frozen_at": plan_frozen_at,
        "target_values_loaded_at": target_values_loaded_at,
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "participant_counts": EXPECTED_PARTICIPANTS,
        "bootstrap_replicates": args.bootstrap_replicates,
        "power_target": POWER_TARGET,
        "alpha": ALPHA,
        "runtime_seconds": time.time() - started,
        "pause_required": True,
        "next_authorized_step": (
            "none until user confirms Step 1; then Step 2 excursion response"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "blockers": [],
    }
    write_json(output_directory / "step1_manifest.json", manifest)
    latest = OUTPUT_ROOT / "latest"
    temporary_latest = OUTPUT_ROOT / ".latest.tmp"
    if temporary_latest.exists() or temporary_latest.is_symlink():
        temporary_latest.unlink()
    temporary_latest.symlink_to(run_id)
    os.replace(temporary_latest, latest)
    (OUTPUT_ROOT / "LATEST_STEP1_RUN.txt").write_text(
        str(output_directory.resolve()) + "\n"
    )
    LOG.info("QC COMPLETE output=%s", output_directory)
    print(
        json.dumps(
            {
                "output_directory": str(output_directory),
                "availability": availability.to_dict("records"),
                "dispersion_summary": pd.read_csv(
                    output_directory
                    / "dispersion_summary_by_split_condition.csv"
                ).to_dict("records"),
                "neighbor_test": neighbor_results[
                    neighbor_results["split"] == "test"
                ].to_dict("records"),
                "incremental_focus": incremental_results[
                    incremental_results["comparison"]
                    == "dispersion_added_beyond_median"
                ].to_dict("records"),
                "pause_required": True,
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
