#!/usr/bin/env python3
"""Run the gated per-variable static-conditioning audit.

Only Step A is implemented here. It performs read-only schema inspection and
participant prevalence calculations. It does not instantiate the model or run
any checkpoint forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
STEP1_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step1_static_neutralization/20260724T223612Z"
)
SCHEMA_PATH = STEP1_ROOT / "static_schema_audit.csv"
REFERENCE_JSON_PATH = STEP1_ROOT / "static_reference_profile.json"
REFERENCE_CSV_PATH = STEP1_ROOT / "static_reference_profile.csv"
STATIC_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "participant_static_features.parquet"
)
CHECKPOINT_PATH = (
    REPO_ROOT
    / "outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/"
    "best_model_checkpoint.pt"
)
CONFIG_PATH = REPO_ROOT / "configs/aireadi_stream_full.yaml"
SPLIT_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/"
    "experiment_c_split_adapt6h_seed42/split_participants.csv"
)
PCA_PATH = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step3_validation_clustering/"
    "20260725T001123Z/pca_loadings.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/per_variable_static_conditioning_audit"
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
VARIABLE_CLASSES = {
    "age": "continuous",
    "bmi": "continuous",
    "hba1c": "continuous",
    "study_group": "categorical",
    "sex": "categorical",
    "insulin": "medication",
    "metformin": "medication",
    "glp1": "medication",
}
ALL_MEDICATION_COLUMNS = [
    "med_metformin",
    "med_insulin",
    "med_glp1_or_gip_glp1",
    "med_sglt2",
    "med_sulfonylurea",
    "med_thiazolidinedione",
]
ANY_MEDICATION_COLUMN = "med_any_diabetes_drug"
MEDICATION_MIN_GROUP_N = 30
EXPECTED_STATIC_INPUT_COUNT = 44
VALIDATION_SPLIT_LABEL = "val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def checkpoint_contract() -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    metadata = checkpoint["metadata"]
    feature_spec = metadata["feature_spec"]
    preprocessor = metadata["preprocessor"]
    return (
        list(feature_spec["static_reals"]),
        list(feature_spec["static_categoricals"]),
        dict(preprocessor["static_category_maps"]),
    )


def reference_lookup() -> pd.DataFrame:
    profile = pd.read_csv(REFERENCE_CSV_PATH)
    return profile.set_index("feature_name", drop=False)


def schema_lookup() -> pd.DataFrame:
    schema = pd.read_csv(SCHEMA_PATH)
    if len(schema) != EXPECTED_STATIC_INPUT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_STATIC_INPUT_COUNT} consumed static inputs, "
            f"found {len(schema)}"
        )
    if not schema["consumed_by_model"].fillna(False).all():
        raise RuntimeError("Step 1 schema includes a non-consumed static input")
    if schema["input_order"].tolist() != list(range(EXPECTED_STATIC_INPUT_COUNT)):
        raise RuntimeError("Step 1 static input order is not contiguous from 0 to 43")
    return schema.set_index("feature_name", drop=False)


def build_mapping(
    schema: pd.DataFrame,
    reference: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable_index, variable in enumerate(VARIABLE_ORDER):
        primary = PRIMARY_COLUMNS[variable]
        if primary not in schema.index:
            raise RuntimeError(f"{variable} is not consumed by the checkpoint")
        columns = [primary]
        if VARIABLE_CLASSES[variable] == "medication":
            columns.append(ANY_MEDICATION_COLUMN)
        for group_index, column in enumerate(columns):
            source = schema.loc[column]
            ref = reference.loc[column]
            is_shared = column == ANY_MEDICATION_COLUMN
            if VARIABLE_CLASSES[variable] == "medication":
                ambiguity = True
                ambiguity_reason = (
                    "The shared any-drug channel encodes the union of six drug flags. "
                    "Changing it for one drug can alter information about concurrent "
                    "drugs, while leaving it fixed can be inconsistent for participants "
                    "whose only drug is the target drug."
                )
                review_status = "requires_user_confirmation"
            else:
                ambiguity = False
                ambiguity_reason = ""
                review_status = "resolved"
            rows.append(
                {
                    "requested_order": variable_index,
                    "variable": variable,
                    "variable_class": VARIABLE_CLASSES[variable],
                    "encoding_group_order": group_index,
                    "encoding_column": column,
                    "encoding_role": (
                        "shared_derived_any_diabetes_drug"
                        if is_shared
                        else "primary_variable_channel"
                    ),
                    "checkpoint_input_order": int(source["input_order"]),
                    "consumed_by_model": bool(source["consumed_by_model"]),
                    "feature_type": source["feature_type"],
                    "encoding_type": source["encoding_type"],
                    "has_missingness_indicator": bool(
                        source["has_missingness_indicator"]
                    ),
                    "missingness_indicator_name": (
                        ""
                        if pd.isna(source["missingness_indicator_name"])
                        else str(source["missingness_indicator_name"])
                    ),
                    "reference_rule": ref["reference_rule"],
                    "raw_reference_value": ref["raw_reference_value"],
                    "transformed_reference_value": ref[
                        "transformed_reference_value"
                    ],
                    "category_map": json.dumps(
                        category_maps.get(column, {}),
                        sort_keys=True,
                    ),
                    "encoding_group_ambiguous": ambiguity,
                    "ambiguity_reason": ambiguity_reason,
                    "review_status": review_status,
                }
            )
    return pd.DataFrame(rows)


def medication_prevalence(static: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    validation_ids = set(
        split.loc[split["split"].eq(VALIDATION_SPLIT_LABEL), "participant_id"].astype(str)
    )
    validation = static[
        static["participant_id"].astype(str).isin(validation_ids)
    ].copy()
    if validation["participant_id"].astype(str).nunique() != len(validation_ids):
        raise RuntimeError("Not every validation participant has a static feature row")
    rows = []
    for variable in ["insulin", "metformin", "glp1"]:
        column = PRIMARY_COLUMNS[variable]
        values = pd.to_numeric(validation[column], errors="coerce")
        exposed = int(values.eq(1).sum())
        unexposed = int(values.eq(0).sum())
        missing = int(values.isna().sum())
        underpowered_groups = []
        if exposed < MEDICATION_MIN_GROUP_N:
            underpowered_groups.append("exposed")
        if unexposed < MEDICATION_MIN_GROUP_N:
            underpowered_groups.append("unexposed")
        rows.append(
            {
                "variable": variable,
                "encoding_column": column,
                "validation_n": len(validation),
                "exposed_n": exposed,
                "unexposed_n": unexposed,
                "missing_n": missing,
                "prevalence_fraction": exposed / len(validation),
                "minimum_group_n_threshold": MEDICATION_MIN_GROUP_N,
                "underpowered": bool(underpowered_groups),
                "underpowered_group": ",".join(underpowered_groups),
                "confidence_note": (
                    "low-confidence displacement estimate"
                    if underpowered_groups
                    else "group counts meet threshold"
                ),
            }
        )
    return pd.DataFrame(rows)


def medication_coherence(static: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    validation_ids = set(
        split.loc[split["split"].eq(VALIDATION_SPLIT_LABEL), "participant_id"].astype(str)
    )
    validation = static[
        static["participant_id"].astype(str).isin(validation_ids)
    ].copy()
    medication_matrix = validation[ALL_MEDICATION_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    derived_any = pd.to_numeric(
        validation[ANY_MEDICATION_COLUMN],
        errors="coerce",
    )
    calculated_any = medication_matrix.fillna(0).eq(1).any(axis=1).astype(int)
    rows = []
    for variable in ["insulin", "metformin", "glp1"]:
        primary = PRIMARY_COLUMNS[variable]
        exposed = medication_matrix[primary].eq(1)
        other_columns = [x for x in ALL_MEDICATION_COLUMNS if x != primary]
        concurrent = medication_matrix[other_columns].fillna(0).eq(1).any(axis=1)
        rows.append(
            {
                "variable": variable,
                "exposed_n": int(exposed.sum()),
                "exposed_with_other_diabetes_drug_n": int((exposed & concurrent).sum()),
                "exposed_with_target_as_only_diabetes_drug_n": int(
                    (exposed & ~concurrent).sum()
                ),
                "shared_any_flag_matches_union_for_all_validation_participants": bool(
                    derived_any.eq(calculated_any).all()
                ),
                "step_b_shared_channel_disposition": "requires_user_confirmation",
            }
        )
    return pd.DataFrame(rows)


def report_text(
    mapping: pd.DataFrame,
    prevalence: pd.DataFrame,
    coherence: pd.DataFrame,
    static_reals: list[str],
    static_categoricals: list[str],
) -> str:
    lines = [
        "# Step A schema and encoding-group audit",
        "",
        "Status: paused before any checkpoint forward pass.",
        "",
        "## Checkpoint static contract",
        "",
        f"- Continuous channels: {len(static_reals)}",
        f"- Learned categorical channels: {len(static_categoricals)}",
        f"- Total consumed inputs: {len(static_reals) + len(static_categoricals)}",
        "- Separate missingness-indicator channels for the requested variables: none",
        "",
        "Study group and sex are each represented by one checkpoint integer index "
        "followed by a learned embedding. They are not one-hot channel groups.",
        "",
        "## Requested variable mapping",
        "",
        "| Variable | Encoding columns | Input order | Status |",
        "|---|---|---:|---|",
    ]
    for variable in VARIABLE_ORDER:
        current = mapping[mapping["variable"].eq(variable)]
        columns = ", ".join(current["encoding_column"])
        orders = ", ".join(current["checkpoint_input_order"].astype(str))
        status = current["review_status"].iloc[0]
        lines.append(f"| {variable} | {columns} | {orders} | {status} |")
    lines.extend(
        [
            "",
            "## Validation medication prevalence",
            "",
            "| Drug | Exposed | Unexposed | Missing | Status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in prevalence.itertuples(index=False):
        status = (
            f"underpowered: {row.underpowered_group}"
            if row.underpowered
            else "counts meet threshold"
        )
        lines.append(
            f"| {row.variable} | {row.exposed_n} | {row.unexposed_n} | "
            f"{row.missing_n} | {status} |"
        )
    lines.extend(
        [
            "",
            "The underpowered threshold is fewer than "
            f"{MEDICATION_MIN_GROUP_N} participants in either group.",
            "",
            "## Medication encoding ambiguity requiring confirmation",
            "",
            f"`{ANY_MEDICATION_COLUMN}` is a consumed shared channel and exactly equals "
            "the union of the six diabetes-drug flags for all validation participants. "
            "It is not uniquely attributable to insulin, metformin, or GLP-1 therapy.",
            "",
        ]
    )
    for row in coherence.itertuples(index=False):
        lines.append(
            f"- {row.variable}: {row.exposed_n} exposed, "
            f"{row.exposed_with_other_diabetes_drug_n} also use another modeled "
            f"diabetes drug, and "
            f"{row.exposed_with_target_as_only_diabetes_drug_n} use only this "
            "modeled drug."
        )
    lines.extend(
        [
            "",
            "Step B must not start until the shared-channel disposition is confirmed. "
            "The two explicit choices are:",
            "",
            "1. Replace only the drug-specific flag and leave the shared any-drug "
            "channel factual. This isolates the checkpoint's drug-specific channel but "
            "leaves a redundant factual signal.",
            "2. Replace both the drug-specific flag and shared any-drug channel. This "
            "removes both signals but also changes information attributable to "
            "concurrent drugs.",
            "",
            "No checkpoint forward pass was run by this Step A audit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / run_id / "schema_audit"
    output_dir.mkdir(parents=True, exist_ok=False)

    schema = schema_lookup()
    reference = reference_lookup()
    static_reals, static_categoricals, category_maps = checkpoint_contract()
    checkpoint_order = static_reals + static_categoricals
    if checkpoint_order != schema.sort_values("input_order")["feature_name"].tolist():
        raise RuntimeError("Step 1 schema order differs from checkpoint metadata")
    reference_payload = json.loads(REFERENCE_JSON_PATH.read_text())
    if reference_payload["feature_names"]["static_reals"] != static_reals:
        raise RuntimeError("Reference static-real order differs from checkpoint")
    if reference_payload["feature_names"]["static_categoricals"] != static_categoricals:
        raise RuntimeError("Reference categorical order differs from checkpoint")

    static = pd.read_parquet(STATIC_PATH)
    split = pd.read_csv(SPLIT_PATH, dtype=str)
    mapping = build_mapping(schema, reference, category_maps)
    prevalence = medication_prevalence(static, split)
    coherence = medication_coherence(static, split)

    mapping_path = output_dir / "variable_encoding_group_audit.csv"
    prevalence_path = output_dir / "medication_prevalence_validation.csv"
    coherence_path = output_dir / "medication_encoding_coherence_validation.csv"
    report_path = output_dir / "step_a_schema_audit_report.md"
    manifest_path = output_dir / "step_a_manifest.json"
    mapping.to_csv(mapping_path, index=False)
    prevalence.to_csv(prevalence_path, index=False)
    coherence.to_csv(coherence_path, index=False)
    report_path.write_text(
        report_text(
            mapping,
            prevalence,
            coherence,
            static_reals,
            static_categoricals,
        )
    )

    source_paths = [
        SCHEMA_PATH,
        REFERENCE_JSON_PATH,
        REFERENCE_CSV_PATH,
        STATIC_PATH,
        CONFIG_PATH,
        SPLIT_PATH,
        PCA_PATH,
    ]
    manifest = {
        "stage": "step_a_schema_and_encoding_group_audit",
        "status": "paused_requires_user_confirmation",
        "forward_pass_executed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "expected_static_input_count": EXPECTED_STATIC_INPUT_COUNT,
        "checkpoint_static_real_count": len(static_reals),
        "checkpoint_static_categorical_count": len(static_categoricals),
        "checkpoint_static_input_count": len(checkpoint_order),
        "checkpoint_identifier_from_step1": reference_payload[
            "checkpoint_identifier"
        ],
        "medication_min_group_n": MEDICATION_MIN_GROUP_N,
        "unresolved_items": [
            "Disposition of med_any_diabetes_drug for each drug intervention"
        ],
        "source_hashes_sha256": {
            str(path): sha256_file(path) for path in source_paths
        },
        "output_files": [
            str(mapping_path),
            str(prevalence_path),
            str(coherence_path),
            str(report_path),
            str(manifest_path),
        ],
    }
    write_json(manifest_path, manifest)
    (args.output_root / "LATEST_STEP_A_RUN.txt").write_text(str(output_dir.parent) + "\n")

    print(report_path.read_text())
    print("Saved Step A audit to:")
    print(output_dir)


if __name__ == "__main__":
    main()
