#!/usr/bin/env python3
"""Step 7 closing pass for the hidden-state participant-similarity study.

The stage interface enforces the two required execution gates. The schema
stage never loads the forecasting checkpoint, fits a scaler or PCA, replays a
participant, or modifies any Step 0 through Step 6 artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "outputs/hidden_state_phenotype/step7_original_question_closing_pass"
)
DEFAULT_STATIC_TABLE = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "participant_static_features.parquet"
)
DEFAULT_STEP_DIRS = {
    "step0": REPOSITORY_ROOT
    / "outputs/hidden_state_phenotype/step0_feasibility",
    "step1": REPOSITORY_ROOT
    / "outputs/hidden_state_phenotype/step1_static_neutralization/"
    "20260724T223612Z",
    "step2": REPOSITORY_ROOT
    / "outputs/hidden_state_phenotype/step2_validation_export/"
    "20260724T231513Z",
    "step3": REPOSITORY_ROOT
    / "outputs/hidden_state_phenotype/step3_validation_clustering/"
    "20260725T001123Z",
    "step3b": REPOSITORY_ROOT
    / "outputs/hidden_state_phenotype/step3b_exploratory_k2_freeze/"
    "20260725T005617Z",
    "step4": REPOSITORY_ROOT
    / "outputs/hidden_state_phenotype/step4_test_confirmation/"
    "20260725T010440Z",
    "step5": REPOSITORY_ROOT
    / "outputs/hidden_state_phenotype/step5_clinical_probes/"
    "20260725T022634Z",
    "step6": REPOSITORY_ROOT
    / "outputs/hidden_state_phenotype/step6_final_synthesis/"
    "20260725T172415Z",
}
DEFAULT_FIGURE_REVISION_SCRIPT = (
    REPOSITORY_ROOT / "scripts/hidden_state_final_figure_revisions.py"
)

PRIMARY_K_NEIGHBORS = 10
SENSITIVITY_K_NEIGHBORS = (5, 20)
DISTANCE_METRIC = "euclidean"
N_BOOTSTRAP = 2000
N_RANDOM_BASELINE_REPEATS = 2000
N_PERMUTATIONS = 2000
RANDOM_SEED = 42
REPRESENTATION_DIMENSIONS = tuple(f"r_{index:03d}" for index in range(128))
PRIMARY_REPRESENTATIONS = ("full_all", "neutral_all")
EXTERNAL_TARGET_ALIASES = {
    "NT-proBNP": "natriuretic_peptide_b_prohormon",
    "High-sensitivity CRP": "c_reactive_protein_i",
    "BUN/creatinine ratio": "bun_creatinine_ratio",
}
STUDY_GROUP_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes_lifestyle_controlled": "Prediabetes",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled":
        "Oral medication",
    "insulin_dependent": "Insulin",
}
GLYCEMIC_COLUMNS = {
    "mean glucose": "mean_glucose",
    "glucose SD": "glucose_sd",
    "glucose CV": "glucose_cv",
    "TIR 70 to 180": "tir_70_180",
    "TAR above 180": "tar_above_180",
    "TBR below 70": "tbr_below_70",
    "mean absolute glucose slope": "mean_absolute_glucose_slope",
    "glucose range": "glucose_range",
    "valid CGM hours": "available_cgm_hours",
}
NO_EM_DASH = "\u2014"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for step_name, default_path in DEFAULT_STEP_DIRS.items():
        parser.add_argument(
            f"--{step_name}-dir",
            type=Path,
            default=default_path,
        )
    parser.add_argument("--static-table", type=Path, default=DEFAULT_STATIC_TABLE)
    parser.add_argument(
        "--figure-revision-script",
        type=Path,
        default=DEFAULT_FIGURE_REVISION_SCRIPT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--stage",
        choices=("schema", "neighbors", "figures", "hba1c", "text", "all"),
        required=True,
    )
    parser.add_argument(
        "--primary-k", type=int, default=PRIMARY_K_NEIGHBORS
    )
    parser.add_argument(
        "--sensitivity-k",
        type=int,
        nargs=2,
        default=SENSITIVITY_K_NEIGHBORS,
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=N_BOOTSTRAP
    )
    parser.add_argument(
        "--random-baseline-repeats",
        type=int,
        default=N_RANDOM_BASELINE_REPEATS,
    )
    parser.add_argument(
        "--permutation-replicates", type=int, default=N_PERMUTATIONS
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path, datetime, pd.Timestamp)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_value) + "\n"
    )
    os.replace(temporary_path, path)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported tabular file: {path}")


def natural_participant_key(frame: pd.DataFrame) -> list[str]:
    columns = set(frame.columns)
    candidates = [
        ["participant_id", "representation_type"],
        ["participant_id", "representation_space"],
        ["participant_id", "target_name"],
        ["participant_id", "target", "feature_set", "outer_repetition"],
        ["participant_id", "target", "feature_set"],
        ["participant_id"],
    ]
    for candidate in candidates:
        if set(candidate).issubset(columns):
            return candidate
    return []


def schema_inventory_row(name: str, path: Path) -> dict[str, Any]:
    base = {
        "artifact_name": name,
        "path": str(path.resolve(strict=False)),
        "existence": path.exists(),
        "type": "missing",
        "row_count": np.nan,
        "column_count": np.nan,
        "columns": "[]",
        "dtypes": "{}",
        "participant_count": np.nan,
        "split_values": "[]",
        "duplicate_participant_keys": "{}",
        "missing_participant_ids": np.nan,
        "file_sha256": "",
        "notes": "",
    }
    if not path.exists():
        base["notes"] = "Requested artifact is absent."
        return base
    if path.is_dir():
        base["type"] = "directory"
        base["notes"] = (
            f"{sum(1 for item in path.rglob('*') if item.is_file())} files"
        )
        return base
    base["type"] = path.suffix.lstrip(".") or "file"
    base["file_sha256"] = sha256_file(path)
    if path.suffix not in {".csv", ".parquet"}:
        base["notes"] = "Non-tabular artifact; schema fields are not applicable."
        return base
    frame = read_table(path)
    base["row_count"] = len(frame)
    base["column_count"] = len(frame.columns)
    base["columns"] = json.dumps(list(frame.columns))
    base["dtypes"] = json.dumps(
        {column: str(dtype) for column, dtype in frame.dtypes.items()},
        sort_keys=True,
    )
    if "participant_id" in frame:
        participant_ids = frame["participant_id"]
        base["participant_count"] = participant_ids.dropna().astype(str).nunique()
        base["missing_participant_ids"] = int(participant_ids.isna().sum())
        key = natural_participant_key(frame)
        base["duplicate_participant_keys"] = json.dumps(
            {
                "key": key,
                "duplicate_rows": int(frame.duplicated(key, keep=False).sum())
                if key
                else 0,
                "repeated_participant_rows": int(
                    participant_ids.duplicated(keep=False).sum()
                ),
            },
            sort_keys=True,
        )
    if "split" in frame:
        base["split_values"] = json.dumps(
            sorted(frame["split"].dropna().astype(str).unique().tolist())
        )
    return base


def required_inputs(args: argparse.Namespace) -> dict[str, Path]:
    step0 = args.step0_dir.resolve()
    step1 = args.step1_dir.resolve()
    step2 = args.step2_dir.resolve()
    step3 = args.step3_dir.resolve()
    step4 = args.step4_dir.resolve()
    step5 = args.step5_dir.resolve()
    inputs = {
        "step4_test_representations":
            step4 / "test_participant_representations.parquet",
        "step4_test_representation_metadata":
            step4 / "test_representation_metadata.csv",
        "step4_test_glycemic_features":
            step4 / "test_glycemic_nuisance_features.parquet",
        "step4_test_external_targets":
            step4 / "test_external_targets.parquet",
        "step4_test_continuous_geometry":
            step4 / "test_continuous_geometry_associations.csv",
        "step4_test_context_geometry":
            step4 / "test_context_geometry_comparison.csv",
        "step4_requested_test_pca_scores":
            step4 / "test_pca_scores.parquet",
        "step3_validation_pca_scores":
            step3 / "pca_participant_scores.parquet",
        "step3_pca_loadings": step3 / "pca_loadings.parquet",
        "step3_residual_representations":
            step3 / "glucose_residualized_representations.parquet",
        "step3_validation_external_targets":
            step3 / "validation_external_targets.parquet",
        "step3_validation_glycemic_features":
            step3 / "validation_glycemic_nuisance_features.parquet",
        "step3_frozen_pipeline": step3 / "frozen_validation_pipeline",
        "step2_validation_representations":
            step2 / "participant_representations.parquet",
        "step2_validation_representation_metadata":
            step2 / "participant_representation_metadata.csv",
        "step5_incremental_value": step5 / "probe_incremental_value.csv",
        "step5_permutation_tests":
            step5 / "probe_incremental_permutation_tests.csv",
        "step5_validation_predictions":
            step5 / "validation_probe_predictions.parquet",
        "step5_test_predictions": step5 / "test_probe_predictions.parquet",
        "step5_feature_sets": step5 / "probe_feature_sets.json",
        "step5_frozen_plan": step5 / "step5_analysis_plan_frozen.json",
        "step1_static_schema_audit": step1 / "static_schema_audit.csv",
        "step0_clinical_target_inventory":
            step0 / "clinical_target_inventory.csv",
        "step0_context_coverage":
            step0 / "context_coverage_by_participant.csv",
        "participant_static_features": args.static_table.resolve(),
        "step6_canonical_synthesis": args.step6_dir.resolve(),
        "figure_revision_script": args.figure_revision_script.resolve(),
    }
    return inputs


def frozen_pipeline_audit(step3: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    pipeline_root = step3 / "frozen_validation_pipeline"
    for condition in PRIMARY_REPRESENTATIONS:
        condition_dir = pipeline_root / condition
        paths = {
            "feature_order": condition_dir / "feature_order.json",
            "kept_dimensions": condition_dir / "kept_dimensions.npy",
            "validation_scaler": condition_dir / f"{condition}_scaler.joblib",
            "validation_pca": condition_dir / f"{condition}_pca.joblib",
        }
        metadata: dict[str, Any] = {}
        for artifact_type, artifact_path in paths.items():
            rows.append(
                {
                    "condition": condition,
                    "artifact_type": artifact_type,
                    "path": str(artifact_path),
                    "exists": artifact_path.exists(),
                    "file_sha256": sha256_file(artifact_path)
                    if artifact_path.exists()
                    else "",
                    "notes": "",
                }
            )
        complete = all(path.exists() for path in paths.values())
        if complete:
            feature_order = json.loads(paths["feature_order"].read_text())
            kept = np.load(paths["kept_dimensions"])
            scaler = joblib.load(paths["validation_scaler"])
            pca = joblib.load(paths["validation_pca"])
            source_dimensions = feature_order.get("source_dimensions", [])
            removed = feature_order.get("removed_indices", [])
            retained = int(feature_order.get("primary_components", 0))
            metadata = {
                "source_dimension_count": len(source_dimensions),
                "source_dimensions_match_required_order":
                    source_dimensions == list(REPRESENTATION_DIMENSIONS),
                "kept_dimension_count": len(kept),
                "kept_indices_match_feature_order": np.array_equal(
                    kept, np.asarray(feature_order.get("kept_indices", []))
                ),
                "removed_indices": removed,
                "scaler_feature_count": int(scaler.n_features_in_),
                "pca_input_feature_count": int(pca.n_features_in_),
                "pca_fitted_component_count": int(pca.n_components_),
                "primary_retained_component_count": retained,
                "explained_variance_ratios": pca.explained_variance_ratio_.tolist(),
            }
            complete = bool(
                metadata["source_dimensions_match_required_order"]
                and metadata["kept_indices_match_feature_order"]
                and metadata["scaler_feature_count"] == len(kept)
                and metadata["pca_input_feature_count"] == len(kept)
                and retained > 0
                and retained <= metadata["pca_fitted_component_count"]
            )
        metadata["complete_frozen_projection_pipeline"] = complete
        metadata["artifact_paths"] = {
            name: str(path) for name, path in paths.items()
        }
        metadata["artifact_hashes"] = {
            name: sha256_file(path) if path.exists() else ""
            for name, path in paths.items()
        }
        summaries[condition] = metadata
    return pd.DataFrame(rows), summaries


def representation_audit(
    representations_path: Path,
    pipeline_summary: dict[str, Any],
) -> dict[str, Any]:
    frame = pd.read_parquet(representations_path)
    frame["participant_id"] = frame["participant_id"].astype(str)
    selected: dict[str, Any] = {}
    participant_sets: dict[str, set[str]] = {}
    for condition in PRIMARY_REPRESENTATIONS:
        part = frame.loc[frame["representation_type"] == condition].copy()
        participant_sets[condition] = set(part["participant_id"])
        values = part.loc[:, REPRESENTATION_DIMENSIONS].to_numpy(float)
        selected[condition] = {
            "source_path": str(representations_path),
            "selection": f"representation_type == '{condition}'",
            "row_count": len(part),
            "participant_count": part["participant_id"].nunique(),
            "duplicate_participant_rows": int(
                part.duplicated(["participant_id", "representation_type"]).sum()
            ),
            "missing_values": int(np.isnan(values).sum()),
            "nonfinite_values": int((~np.isfinite(values)).sum()),
            "raw_dimension_count": values.shape[1],
            "primary_pca_component_count": pipeline_summary[condition].get(
                "primary_retained_component_count"
            ),
            "split_values": sorted(part["split"].astype(str).unique().tolist()),
        }
    selected["exact_participant_overlap"] = bool(
        participant_sets["full_all"] == participant_sets["neutral_all"]
    )
    selected["overlap_count"] = len(
        participant_sets["full_all"] & participant_sets["neutral_all"]
    )
    return selected


def make_required_column_audit(
    inputs: dict[str, Path],
    static_schema: pd.DataFrame,
) -> pd.DataFrame:
    test_features = pd.read_parquet(inputs["step4_test_glycemic_features"])
    validation_features = pd.read_parquet(
        inputs["step3_validation_glycemic_features"]
    )
    representations = pd.read_parquet(inputs["step4_test_representations"])
    static_columns = pd.read_parquet(
        inputs["participant_static_features"]
    ).columns.tolist()
    external = pd.read_parquet(inputs["step4_test_external_targets"])
    external_levels = set(external["target_name"].dropna().astype(str))
    rows: list[dict[str, Any]] = []

    def add(
        requirement: str,
        resolved_column: str,
        source: Path,
        status: str,
        alias: str = "",
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "requirement": requirement,
                "resolved_column_or_level": resolved_column,
                "source_file": str(source),
                "status": status,
                "alias_mapping": alias,
                "notes": notes,
            }
        )

    add(
        "participant identifier",
        "participant_id",
        inputs["step4_test_representations"],
        "available" if "participant_id" in representations else "missing",
    )
    add(
        "test split",
        "split",
        inputs["step4_test_representations"],
        "available"
        if set(representations["split"].astype(str)) == {"test"}
        else "invalid",
    )
    for condition in ("full_all", "neutral_all", "neutral_glucose_residual"):
        add(
            f"representation {condition}",
            condition,
            inputs["step4_test_representations"],
            "available"
            if condition in set(representations["representation_type"])
            else "missing",
            "representation_type value",
        )
    add(
        "hidden representation dimensions",
        "r_000 through r_127",
        inputs["step4_test_representations"],
        "available"
        if set(REPRESENTATION_DIMENSIONS).issubset(representations.columns)
        else "missing",
    )
    for label, column in GLYCEMIC_COLUMNS.items():
        test_available = column in test_features
        validation_available = column in validation_features
        status = (
            "available"
            if test_available
            else "validation_only" if validation_available else "missing"
        )
        notes = ""
        if column == "glucose_range" and not test_available:
            notes = (
                "Step 5 computes the test value from valid CGM panel rows using "
                "numpy.ptp; it is not stored in the Step 4 feature export."
            )
        add(
            label,
            column,
            inputs["step4_test_glycemic_features"]
            if test_available
            else inputs["step3_validation_glycemic_features"],
            status,
            notes=notes,
        )
    for requirement, source_column, test_alias in [
        ("HbA1c", "hba1c_percent_baseline", "hba1c"),
        ("study group", "participants_study_group", "study_group"),
        ("clinical site", "participants_clinical_site", "clinical_site"),
    ]:
        available = source_column in static_columns and test_alias in test_features
        add(
            requirement,
            source_column,
            inputs["participant_static_features"],
            "available" if available else "missing",
            f"{source_column} -> {test_alias}",
        )
    for label, source_name in EXTERNAL_TARGET_ALIASES.items():
        add(
            label,
            source_name,
            inputs["step4_test_external_targets"],
            "available" if source_name in external_levels else "missing",
            f"{source_name} -> {label}",
        )
    model_inputs = set(
        static_schema.loc[
            static_schema["consumed_by_model"].astype(bool), "source_column"
        ].astype(str)
    )
    add(
        "HbA1c confirmed forecasting-model input",
        "hba1c_percent_baseline",
        inputs["step1_static_schema_audit"],
        "confirmed"
        if "hba1c_percent_baseline" in model_inputs
        else "not_confirmed",
    )
    for label, source_name in EXTERNAL_TARGET_ALIASES.items():
        add(
            f"{label} confirmed external, not model input",
            source_name,
            inputs["step1_static_schema_audit"],
            "confirmed" if source_name not in model_inputs else "invalid",
        )
    return pd.DataFrame(rows)


def git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "dirty_tree": bool(status.strip()),
        "status_lines": status.splitlines(),
    }


def create_run_directory(args: argparse.Namespace) -> Path:
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output_root = args.output_root.resolve()
    output_directory = output_root / run_id
    if output_directory.exists():
        raise FileExistsError(
            f"Step 7 run directory already exists: {output_directory}"
        )
    for relative_directory in (
        "schema_audit",
        "neighbor_sharing",
        "hba1c_positive_control",
        "revised_figures",
        "revised_text",
        "manifests",
    ):
        (output_directory / relative_directory).mkdir(
            parents=True, exist_ok=False
        )
    return output_directory


def scan_em_dash(output_directory: Path) -> list[str]:
    affected: list[str] = []
    for path in output_directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".csv", ".json", ".log", ".md", ".py", ".tex"
        }:
            try:
                if NO_EM_DASH in path.read_text():
                    affected.append(str(path))
            except UnicodeDecodeError:
                continue
    return affected


def run_schema_stage(args: argparse.Namespace) -> dict[str, Any]:
    output_directory = create_run_directory(args)
    schema_directory = output_directory / "schema_audit"
    log_path = output_directory / "step7_run.log"
    log_lines = [
        f"{datetime.now(timezone.utc).isoformat()} STEP 7 schema stage started",
        "No forecasting checkpoint was loaded.",
        "No participant replay, model inference, scaler fit, or PCA fit occurred.",
    ]
    inputs = required_inputs(args)
    inventory_rows = [
        schema_inventory_row(name, path) for name, path in inputs.items()
    ]
    schema_inventory = pd.DataFrame(inventory_rows)
    schema_inventory.to_csv(
        schema_directory / "input_file_schema_inventory.csv", index=False
    )

    pipeline_audit, pipeline_summary = frozen_pipeline_audit(
        args.step3_dir.resolve()
    )
    pipeline_audit.to_csv(
        schema_directory / "frozen_pipeline_artifact_audit.csv", index=False
    )
    artifact_rows = []
    for name, path in inputs.items():
        artifact_rows.append(
            {
                "artifact_name": name,
                "path": str(path.resolve(strict=False)),
                "exists": path.exists(),
                "type": "directory" if path.is_dir() else path.suffix.lstrip("."),
                "size_bytes": path.stat().st_size
                if path.exists() and path.is_file()
                else np.nan,
                "file_sha256": sha256_file(path)
                if path.exists() and path.is_file()
                else "",
                "required_for_neighbor_gate": name
                not in {"step4_requested_test_pca_scores"},
                "notes": (
                    "Absent score export is nonblocking because complete frozen "
                    "full and neutral projection pipelines are available."
                    if name == "step4_requested_test_pca_scores"
                    else ""
                ),
            }
        )
    artifact_inventory = pd.concat(
        [pd.DataFrame(artifact_rows), pipeline_audit.rename(
            columns={"condition": "artifact_name"}
        ).assign(
            size_bytes=np.nan,
            required_for_neighbor_gate=True,
        )[
            [
                "artifact_name", "path", "exists", "size_bytes",
                "file_sha256", "required_for_neighbor_gate", "notes"
            ]
        ].assign(type="frozen_pipeline_artifact")],
        ignore_index=True,
    )
    artifact_inventory.to_csv(
        schema_directory / "input_artifact_inventory.csv", index=False
    )

    static_schema = pd.read_csv(inputs["step1_static_schema_audit"])
    required_columns = make_required_column_audit(inputs, static_schema)
    required_columns.to_csv(
        schema_directory / "required_column_audit.csv", index=False
    )
    representation_summary = representation_audit(
        inputs["step4_test_representations"], pipeline_summary
    )
    test_features = pd.read_parquet(inputs["step4_test_glycemic_features"])
    test_features["participant_id"] = test_features["participant_id"].astype(str)
    static = pd.read_parquet(
        inputs["participant_static_features"],
        columns=[
            "participant_id",
            "hba1c_percent_baseline",
            "hba1c_percent_baseline_date",
            "hba1c_percent_n_records",
            "hba1c_percent_days_to_cgm_start",
            "participants_study_group",
            "participants_clinical_site",
        ],
    )
    static["participant_id"] = static["participant_id"].astype(str)
    test_ids = set(
        test_features["participant_id"].dropna().astype(str).unique()
    )
    static_test = static.loc[static["participant_id"].isin(test_ids)].copy()
    hba1c_coverage = {
        "exact_column": "hba1c_percent_baseline",
        "unit": "percent",
        "test_participants": len(test_ids),
        "test_nonmissing": int(
            static_test["hba1c_percent_baseline"].notna().sum()
        ),
        "test_coverage": float(
            static_test["hba1c_percent_baseline"].notna().mean()
        ),
        "duplicate_participant_rows": int(
            static_test.duplicated("participant_id").sum()
        ),
        "consumed_by_forecasting_model": bool(
            (
                (static_schema["source_column"] == "hba1c_percent_baseline")
                & static_schema["consumed_by_model"].astype(bool)
            ).any()
        ),
    }
    study_group_levels = sorted(
        test_features["study_group"].dropna().astype(str).unique().tolist()
    )
    unknown_study_groups = sorted(
        set(study_group_levels) - set(STUDY_GROUP_LABELS)
    )
    clinical_aliases = {
        "HbA1c": "hba1c_percent_baseline",
        "study_group": "participants_study_group",
        "clinical_site": "participants_clinical_site",
        **EXTERNAL_TARGET_ALIASES,
    }
    missing_files = schema_inventory.loc[
        ~schema_inventory["existence"], "path"
    ].tolist()
    required_failures = required_columns.loc[
        required_columns["status"].isin(
            ["missing", "invalid", "not_confirmed"]
        )
    ]
    frozen_complete = all(
        summary.get("complete_frozen_projection_pipeline", False)
        for summary in pipeline_summary.values()
    )
    representation_complete = (
        representation_summary["exact_participant_overlap"]
        and representation_summary["overlap_count"] == 221
        and all(
            representation_summary[condition]["participant_count"] == 221
            and representation_summary[condition]["duplicate_participant_rows"] == 0
            and representation_summary[condition]["nonfinite_values"] == 0
            and representation_summary[condition]["split_values"] == ["test"]
            for condition in PRIMARY_REPRESENTATIONS
        )
    )
    blockers = []
    if len(required_failures):
        blockers.append(
            "One or more required aliases or model-input confirmations failed."
        )
    if not frozen_complete:
        blockers.append("The complete frozen full and neutral pipelines failed.")
    if not representation_complete:
        blockers.append("The primary test representation audit failed.")
    if unknown_study_groups:
        blockers.append(
            "Unmapped study-group levels: " + ", ".join(unknown_study_groups)
        )
    go_status = "GO" if not blockers else "NO-GO"
    warnings = []
    requested_test_scores = inputs["step4_requested_test_pca_scores"]
    if not requested_test_scores.exists():
        warnings.append(
            "test_pca_scores.parquet is absent. Frozen scores can be transformed "
            "from verified Step 4 representations with the complete serialized "
            "validation pipelines, without fitting."
        )
    if "glucose_range" not in test_features:
        warnings.append(
            "Test glucose_range is not stored in the Step 4 feature file. Step 5 "
            "documents deterministic computation from valid CGM rows."
        )
    report_lines = [
        "# Step 7 schema audit report",
        "",
        "## Exact resolved Step 6 run",
        "",
        str(args.step6_dir.resolve()),
        "",
        "## Artifact summary",
        "",
        f"Files or directories found: {int(schema_inventory['existence'].sum())}",
        f"Requested artifacts missing: {len(missing_files)}",
        "",
        "Missing paths:",
        "",
    ]
    report_lines.extend(
        [f"- {path}" for path in missing_files] or ["- None"]
    )
    report_lines.extend(
        [
            "",
            "## Test representation rows selected",
            "",
            f"- Full profile: {representation_summary['full_all']['row_count']} "
            "rows, selection representation_type == 'full_all'.",
            f"- Static neutral: "
            f"{representation_summary['neutral_all']['row_count']} rows, "
            "selection representation_type == 'neutral_all'.",
            f"- Exact participant overlap: "
            f"{representation_summary['exact_participant_overlap']}.",
            f"- Overlap count: {representation_summary['overlap_count']}.",
            "",
            "## Frozen preprocessing",
            "",
            f"- Full profile retained PCA components: "
            f"{pipeline_summary['full_all'].get('primary_retained_component_count')}.",
            f"- Static neutral retained PCA components: "
            f"{pipeline_summary['neutral_all'].get('primary_retained_component_count')}.",
            f"- Complete matching frozen projection pipelines: {frozen_complete}.",
            "",
            "## Clinical aliases",
            "",
            f"- HbA1c: hba1c_percent_baseline, test coverage "
            f"{hba1c_coverage['test_nonmissing']}/{hba1c_coverage['test_participants']} "
            f"({hba1c_coverage['test_coverage']:.3%}).",
            "- Study group: participants_study_group -> study_group.",
            "- Study-group levels: " + ", ".join(study_group_levels) + ".",
            "- Clinical site: participants_clinical_site -> clinical_site.",
            "- NT-proBNP: natriuretic_peptide_b_prohormon.",
            "- High-sensitivity CRP: c_reactive_protein_i.",
            "- BUN/creatinine ratio: bun_creatinine_ratio.",
            "",
            "HbA1c was confirmed as a forecasting-model static input. The three "
            "external biomarkers were confirmed absent from the model static "
            "input schema.",
            "",
            "## Warnings",
            "",
        ]
    )
    report_lines.extend([f"- {warning}" for warning in warnings] or ["- None"])
    report_lines.extend(["", "## Blockers", ""])
    report_lines.extend([f"- {blocker}" for blocker in blockers] or ["- None"])
    report_lines.extend(
        [
            "",
            "## Gate 1 decision",
            "",
            f"**{go_status} for the test neighbour-sharing analysis.**",
            "",
            "No analysis beyond schema and artifact verification was run.",
        ]
    )
    report_path = schema_directory / "schema_audit_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    schema_manifest = {
        "run_id": output_directory.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": "schema",
        "exact_step6_run": str(args.step6_dir.resolve()),
        "git": git_metadata(),
        "input_paths": {name: str(path) for name, path in inputs.items()},
        "input_hashes": {
            name: sha256_file(path)
            for name, path in inputs.items()
            if path.exists() and path.is_file()
        },
        "missing_paths": missing_files,
        "representation_audit": representation_summary,
        "frozen_pipeline_audit": pipeline_summary,
        "hba1c_audit": hba1c_coverage,
        "study_group_column": "participants_study_group",
        "study_group_levels": study_group_levels,
        "study_group_label_mapping": STUDY_GROUP_LABELS,
        "clinical_variable_aliases": clinical_aliases,
        "neighbor_settings": {
            "primary_k": args.primary_k,
            "sensitivity_k": args.sensitivity_k,
            "distance_metric": DISTANCE_METRIC,
            "bootstrap_replicates": args.bootstrap_replicates,
            "random_baseline_repeats": args.random_baseline_repeats,
            "permutation_replicates": args.permutation_replicates,
            "seed": args.seed,
        },
        "warnings": warnings,
        "blockers": blockers,
        "gate1_status": go_status,
        "latest_pointer_created": False,
    }
    write_json(
        output_directory / "manifests/schema_stage_manifest.json",
        schema_manifest,
    )
    write_json(output_directory / "step7_manifest.json", schema_manifest)
    (output_directory / "step7_report.md").write_text(
        "# Step 7 closing pass\n\n"
        "Gate 1 schema and artifact verification is complete. See "
        "schema_audit/schema_audit_report.md. Later stages have not run.\n"
    )
    (output_directory / "step7_changelog.md").write_text(
        "# Step 7 changelog\n\n"
        "## Added\n\n"
        "- Schema and frozen-artifact verification for the closing pass.\n\n"
        "## Changed\n\n"
        "- Nothing in Steps 0 through 6.\n\n"
        "## Unchanged\n\n"
        "- Canonical split, frozen PCA, burn-in, participant aggregation, "
        "static-neutralization intervention, and all source artifacts.\n"
    )
    log_lines.extend(
        [
            f"{datetime.now(timezone.utc).isoformat()} Gate 1 status: {go_status}",
            f"Output directory: {output_directory}",
            "Stopped before neighbour-sharing analysis as required.",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n")
    affected = scan_em_dash(output_directory)
    if affected:
        raise RuntimeError(
            "Generated text contains forbidden Unicode U+2014: "
            + ", ".join(affected)
        )
    return {
        "output_directory": str(output_directory),
        "schema_audit_directory": str(schema_directory),
        "files_found": int(schema_inventory["existence"].sum()),
        "files_missing": missing_files,
        "representation_audit": representation_summary,
        "frozen_pipeline_audit": pipeline_summary,
        "hba1c_audit": hba1c_coverage,
        "study_group_levels": study_group_levels,
        "external_biomarker_mappings": EXTERNAL_TARGET_ALIASES,
        "warnings": warnings,
        "blockers": blockers,
        "gate1_status": go_status,
    }


def main() -> None:
    args = parse_args()
    if args.stage == "schema":
        result = run_schema_stage(args)
    elif args.stage == "neighbors":
        if not args.run_id:
            raise RuntimeError("The neighbors stage requires the confirmed Gate 1 run ID.")
        import sys
        sys.path.insert(0, str(REPOSITORY_ROOT))
        from ssmcgm.analysis.neighbor_clinical_sharing import run_neighbor_stage
        result = run_neighbor_stage(
            run_directory=args.output_root.resolve() / args.run_id,
            step3_directory=args.step3_dir.resolve(),
            step4_directory=args.step4_dir.resolve(),
            bootstrap_replicates=args.bootstrap_replicates,
            random_baseline_repeats=args.random_baseline_repeats,
            permutation_replicates=args.permutation_replicates,
            primary_k=args.primary_k,
            sensitivity_k=tuple(args.sensitivity_k),
            seed=args.seed,
        )
    elif args.stage == "figures":
        if not args.run_id:
            raise RuntimeError("The figures stage requires the confirmed Step 7 run ID.")
        import sys
        sys.path.insert(0, str(REPOSITORY_ROOT))
        from ssmcgm.analysis.final_closing_figures import run_figure_stage
        result = run_figure_stage(
            run_directory=args.output_root.resolve() / args.run_id,
            step3_directory=args.step3_dir.resolve(),
            step4_directory=args.step4_dir.resolve(),
            step5_directory=args.step5_dir.resolve(),
            step6_directory=args.step6_dir.resolve(),
        )
    elif args.stage == "hba1c":
        if not args.run_id:
            raise RuntimeError("The HbA1c stage requires the confirmed Step 7 run ID.")
        import sys
        sys.path.insert(0, str(REPOSITORY_ROOT))
        from ssmcgm.analysis.hba1c_positive_control import run_hba1c_stage
        result = run_hba1c_stage(
            run_directory=args.output_root.resolve() / args.run_id,
            step2_directory=args.step2_dir.resolve(),
            step3_directory=args.step3_dir.resolve(),
            step4_directory=args.step4_dir.resolve(),
            step5_directory=args.step5_dir.resolve(),
            static_table=args.static_table.resolve(),
            bootstrap_replicates=args.bootstrap_replicates,
            permutation_replicates=args.permutation_replicates,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
    elif args.stage == "text":
        if not args.run_id:
            raise RuntimeError("The text stage requires the confirmed Step 7 run ID.")
        import sys
        sys.path.insert(0, str(REPOSITORY_ROOT))
        from ssmcgm.analysis.final_text_revision import run_text_stage
        result = run_text_stage(
            run_directory=args.output_root.resolve() / args.run_id,
            repository_root=REPOSITORY_ROOT,
            step_directories={
                "step0": args.step0_dir.resolve(),
                "step1": args.step1_dir.resolve(),
                "step2": args.step2_dir.resolve(),
                "step3": args.step3_dir.resolve(),
                "step3b": args.step3b_dir.resolve(),
                "step4": args.step4_dir.resolve(),
                "step5": args.step5_dir.resolve(),
                "step6": args.step6_dir.resolve(),
            },
        )
    elif args.stage == "all":
        if not args.run_id:
            raise RuntimeError("The all stage requires the confirmed Step 7 run ID.")
        import sys
        sys.path.insert(0, str(REPOSITORY_ROOT))
        run_directory = args.output_root.resolve() / args.run_id
        manifest_path = run_directory / "step7_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("gate1_status") != "GO":
            raise RuntimeError("Gate 1 has not been completed with GO status.")
        if manifest.get("neighbor_stage", {}).get("gate2_status") != "GO":
            raise RuntimeError("Gate 2 has not been completed with GO status.")
        completed_stages = {}
        if manifest.get("figure_stage", {}).get("status") != "QC_COMPLETE":
            from ssmcgm.analysis.final_closing_figures import run_figure_stage
            completed_stages["figures"] = run_figure_stage(
                run_directory=run_directory,
                step3_directory=args.step3_dir.resolve(),
                step4_directory=args.step4_dir.resolve(),
                step5_directory=args.step5_dir.resolve(),
                step6_directory=args.step6_dir.resolve(),
            )
            manifest = json.loads(manifest_path.read_text())
        else:
            completed_stages["figures"] = "already_complete"
        if manifest.get("hba1c_stage", {}).get("status") != "QC_COMPLETE":
            from ssmcgm.analysis.hba1c_positive_control import run_hba1c_stage
            completed_stages["hba1c"] = run_hba1c_stage(
                run_directory=run_directory,
                step2_directory=args.step2_dir.resolve(),
                step3_directory=args.step3_dir.resolve(),
                step4_directory=args.step4_dir.resolve(),
                step5_directory=args.step5_dir.resolve(),
                static_table=args.static_table.resolve(),
                bootstrap_replicates=args.bootstrap_replicates,
                permutation_replicates=args.permutation_replicates,
                seed=args.seed,
                n_jobs=args.n_jobs,
            )
            manifest = json.loads(manifest_path.read_text())
        else:
            completed_stages["hba1c"] = "already_complete"
        if manifest.get("text_stage", {}).get("status") != "QC_COMPLETE":
            from ssmcgm.analysis.final_text_revision import run_text_stage
            completed_stages["text"] = run_text_stage(
                run_directory=run_directory,
                repository_root=REPOSITORY_ROOT,
                step_directories={
                    "step0": args.step0_dir.resolve(),
                    "step1": args.step1_dir.resolve(),
                    "step2": args.step2_dir.resolve(),
                    "step3": args.step3_dir.resolve(),
                    "step3b": args.step3b_dir.resolve(),
                    "step4": args.step4_dir.resolve(),
                    "step5": args.step5_dir.resolve(),
                    "step6": args.step6_dir.resolve(),
                },
            )
        else:
            completed_stages["text"] = "already_complete"
        result = {
            "run_directory": str(run_directory),
            "status": "QC_COMPLETE",
            "stages": completed_stages,
        }
    else:
        raise RuntimeError(
            "The requested remaining stage has not been connected yet."
        )
    print(json.dumps(result, indent=2, default=json_value))


if __name__ == "__main__":
    main()
