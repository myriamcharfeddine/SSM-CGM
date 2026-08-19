#!/usr/bin/env python3
"""Build the T2D clinical marker inventory without clustering or raw GCS reads."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
ENRICHED_ROOT = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs/clinical_marker_inventory"

FINAL_MULTIMODAL_PATH = (
    ENRICHED_ROOT / "final_multimodal_dataset_20260515_184339.parquet"
)
FINAL_METADATA_PATH = (
    ENRICHED_ROOT / "final_multimodal_dataset_20260515_184339_metadata.json"
)
STATIC_PATH = ENRICHED_ROOT / "participant_static_features.parquet"
CLINICAL_METADATA_PATH = ENRICHED_ROOT / "clinical_enrichment_metadata.json"
SPLIT_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/"
    "experiment_c_split_adapt6h_seed42/split_participants.csv"
)
CLINICAL_TARGETS_PATH = (
    PROJECT_ROOT / "outputs/continuous_clinical/clinical_targets.parquet"
)
COVERAGE_AUDIT_PATH = PROJECT_ROOT / "subtype_partition/t2d_coverage_audit.csv"
STATIC_SCHEMA_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs/hidden_state_phenotype/step1_static_neutralization/"
    "20260724T223612Z/static_schema_audit.csv"
)
PRIOR_RAW_INVENTORY_PATH = (
    PROJECT_ROOT
    / "outputs/hidden_state_phenotype/step0_feasibility/"
    "clinical_target_inventory.csv"
)
PRIOR_STEP0_MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs/hidden_state_phenotype/step0_feasibility/step0_manifest.json"
)

ENRICHED_OUTPUT_PATH = (
    OUTPUT_ROOT / "enriched_dataset_clinical_columns.csv"
)
RAW_OUTPUT_PATH = OUTPUT_ROOT / "raw_gcs_candidate_variables.csv"
CONSOLIDATED_OUTPUT_PATH = OUTPUT_ROOT / "consolidated_tier_inventory.csv"
MANIFEST_PATH = OUTPUT_ROOT / "inventory_manifest.json"

PARTICIPANT_COLUMN = "participant_id"
SPLIT_COLUMN = "split"
EXPECTED_COUNTS = {"validation": 91, "test": 83}
FULL_SPLIT_COUNTS = {"validation": 239, "test": 221}
LOW_COVERAGE_TEST_THRESHOLD = 60
TIER_1_LABEL = "TIER_1_MODEL_INPUT"
TIER_2_LABEL = "TIER_2_NOT_MODEL_INPUT"
NO_EM_DASH = "\u2014"

T2D_STUDY_GROUPS = {
    "insulin_dependent",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
}
COMPLETE_CASE_MARKERS = [
    "bmi_baseline",
    "c_peptide_ngml_baseline",
    "triglycerides_mgdl_baseline",
    "hdl_cholesterol_mgdl_baseline",
    "participants_age",
]
NONFASTING_CAVEAT = (
    "C-peptide and triglycerides were not confirmed fasting measurements."
)
AGE_CAVEAT = (
    "participants_age is age at study visit, not age at diabetes diagnosis."
)

RAW_GCS_OBJECTS = [
    {
        "path": "gs://cgmproject2025/AIREADI/clinical_data/condition_occurrence.csv",
        "size_bytes": 1855347,
        "updated_utc": "2026-03-08T19:27:50Z",
        "format": "CSV",
    },
    {
        "path": "gs://cgmproject2025/AIREADI/clinical_data/dqd_omop.json",
        "size_bytes": 2276116,
        "updated_utc": "2026-03-08T19:27:50Z",
        "format": "JSON",
    },
    {
        "path": "gs://cgmproject2025/AIREADI/clinical_data/measurement.csv",
        "size_bytes": 36643391,
        "updated_utc": "2026-03-08T19:27:53Z",
        "format": "CSV",
    },
    {
        "path": "gs://cgmproject2025/AIREADI/clinical_data/observation.csv",
        "size_bytes": 113309618,
        "updated_utc": "2026-03-08T19:28:02Z",
        "format": "CSV",
    },
    {
        "path": "gs://cgmproject2025/AIREADI/clinical_data/person.csv",
        "size_bytes": 137116,
        "updated_utc": "2026-03-08T19:27:50Z",
        "format": "CSV",
    },
    {
        "path": "gs://cgmproject2025/AIREADI/clinical_data/procedure_occurrence.csv",
        "size_bytes": 7307425,
        "updated_utc": "2026-03-08T19:27:51Z",
        "format": "CSV",
    },
    {
        "path": "gs://cgmproject2025/AIREADI/clinical_data/visit_occurrence.csv",
        "size_bytes": 643135,
        "updated_utc": "2026-03-08T19:27:50Z",
        "format": "CSV",
    },
    {
        "path": (
            "gs://cgmproject2025/AIREADI/protected/"
            "AIREADI_DataDicitonary_Protected_Dataset.xlsx"
        ),
        "size_bytes": 14244,
        "updated_utc": "2026-05-07T18:12:23Z",
        "format": "XLSX",
    },
    {
        "path": (
            "gs://cgmproject2025/AIREADI/protected/"
            "AIREADI_Demographics_Protected_Dataset.xlsx"
        ),
        "size_bytes": 260883,
        "updated_utc": "2026-05-07T18:12:23Z",
        "format": "XLSX",
    },
    {
        "path": (
            "gs://cgmproject2025/AIREADI/protected/"
            "AIREADI_Medications_Protected_Dataset.xlsx"
        ),
        "size_bytes": 1148179,
        "updated_utc": "2026-05-07T18:12:23Z",
        "format": "XLSX",
    },
]

RAW_TABLE_PATHS = {
    "measurement": "gs://cgmproject2025/AIREADI/clinical_data/measurement.csv",
    "observation": "gs://cgmproject2025/AIREADI/clinical_data/observation.csv",
    "condition_occurrence": (
        "gs://cgmproject2025/AIREADI/clinical_data/condition_occurrence.csv"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_t2d_cohort(static: pd.DataFrame) -> pd.DataFrame:
    split = pd.read_csv(SPLIT_PATH, dtype={PARTICIPANT_COLUMN: str})
    split[SPLIT_COLUMN] = split[SPLIT_COLUMN].replace(
        {"val": "validation"}
    )
    split = split[split[SPLIT_COLUMN].isin(EXPECTED_COUNTS)].copy()
    static_ids = static.copy()
    static_ids[PARTICIPANT_COLUMN] = static_ids[PARTICIPANT_COLUMN].astype(str)
    cohort = split[[PARTICIPANT_COLUMN, SPLIT_COLUMN]].merge(
        static_ids[
            [PARTICIPANT_COLUMN, "participants_study_group", *COMPLETE_CASE_MARKERS]
        ],
        on=PARTICIPANT_COLUMN,
        how="left",
        validate="one_to_one",
    )
    cohort = cohort[
        cohort["participants_study_group"].isin(T2D_STUDY_GROUPS)
    ].copy()
    complete = cohort[COMPLETE_CASE_MARKERS].notna().all(axis=1)
    complete &= cohort["hdl_cholesterol_mgdl_baseline"] > 0
    cohort = cohort[complete][[PARTICIPANT_COLUMN, SPLIT_COLUMN]].copy()
    cohort = cohort.sort_values([SPLIT_COLUMN, PARTICIPANT_COLUMN]).reset_index(
        drop=True
    )
    counts = cohort[SPLIT_COLUMN].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Reconstructed T2D counts changed: {counts}")

    targets = pd.read_parquet(
        CLINICAL_TARGETS_PATH,
        columns=[PARTICIPANT_COLUMN, SPLIT_COLUMN],
    )
    targets[PARTICIPANT_COLUMN] = targets[PARTICIPANT_COLUMN].astype(str)
    target_pairs = set(map(tuple, targets[[PARTICIPANT_COLUMN, SPLIT_COLUMN]].values))
    cohort_pairs = set(map(tuple, cohort[[PARTICIPANT_COLUMN, SPLIT_COLUMN]].values))
    if target_pairs != cohort_pairs:
        raise RuntimeError("Reconstructed T2D IDs do not match clinical targets")

    coverage = pd.read_csv(COVERAGE_AUDIT_PATH)
    audit_counts = (
        coverage[coverage["metric"] == "complete_case_four_marker"]
        .set_index("split")["count"]
        .astype(int)
        .to_dict()
    )
    if audit_counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Coverage audit counts changed: {audit_counts}")
    return cohort


def infer_unit_and_notes(variable: str, tier_label: str) -> tuple[str, str]:
    unit = "not applicable or not recorded"
    notes: list[str] = []
    if variable == "participants_age":
        unit = "years"
        notes.append(AGE_CAVEAT)
    elif "bmi" in variable and "days_to" not in variable:
        unit = "kg/m2" if "value_range" not in variable else "kg/m2 range"
    elif "c_peptide_ngml" in variable:
        unit = "ng/mL"
        notes.append(NONFASTING_CAVEAT)
    elif "triglycerides_mgdl" in variable:
        unit = "mg/dL"
        notes.append(NONFASTING_CAVEAT)
    elif any(
        token in variable
        for token in [
            "hdl_cholesterol_mgdl",
            "ldl_cholesterol_mgdl",
            "serum_glucose_mgdl",
        ]
    ):
        unit = "mg/dL"
    elif "bp_mmhg" in variable:
        unit = "mmHg"
    elif "hr_bpm" in variable:
        unit = "beats/min"
    elif "hba1c_percent" in variable:
        unit = "%"
    elif "waist_to_hip_ratio" in variable:
        unit = "ratio"
    elif variable.endswith("_date") or variable == "participants_study_visit_date":
        unit = "date"
    elif variable.endswith("_n_records"):
        unit = "record count"
    elif variable.endswith("_days_to_cgm_start"):
        unit = "days"
    elif variable.startswith("demo_") or variable.startswith("med_"):
        unit = "categorical or binary"
    elif variable in {"participants_clinical_site", "participants_study_group"}:
        unit = "categorical"

    if variable.endswith("_date"):
        notes.append("Provenance date, not a physiological marker value.")
    if variable.endswith("_n_records"):
        notes.append("Source-record count metadata.")
    if variable.endswith("_value_range"):
        notes.append("Within-participant source-value range metadata.")
    if variable.endswith("_days_to_cgm_start"):
        notes.append("Timing metadata relative to CGM start.")
    if tier_label == TIER_1_LABEL:
        notes.append("Consumed by the frozen checkpoint per static_schema_audit.csv.")
    else:
        notes.append("Present in enriched static data but not consumed by the checkpoint.")
    return unit, " ".join(notes)


def build_enriched_inventory(
    static: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    final_schema = pq.read_schema(FINAL_MULTIMODAL_PATH)
    static_schema = pq.read_schema(STATIC_PATH)
    final_columns = set(final_schema.names)
    static_columns = [
        column for column in static_schema.names if column != PARTICIPANT_COLUMN
    ]
    metadata = json.loads(CLINICAL_METADATA_PATH.read_text(encoding="utf-8"))
    added_columns = set(metadata["added_columns"])
    if set(static_columns) != added_columns:
        missing_from_metadata = sorted(set(static_columns) - added_columns)
        missing_from_static = sorted(added_columns - set(static_columns))
        raise RuntimeError(
            "Static and clinical metadata differ: "
            f"static_only={missing_from_metadata}, metadata_only={missing_from_static}"
        )
    missing_from_final = sorted(set(static_columns) - final_columns)
    if missing_from_final:
        raise RuntimeError(
            f"Static clinical columns missing from final multimodal: {missing_from_final}"
        )

    audit = pd.read_csv(STATIC_SCHEMA_AUDIT_PATH)
    consumed = audit[audit["consumed_by_model"] == True].copy()  # noqa: E712
    consumed_columns = set(consumed["source_column"])
    if len(consumed_columns) != 44:
        raise RuntimeError(
            f"Expected 44 consumed static columns, observed {len(consumed_columns)}"
        )
    audit_by_column = consumed.set_index("source_column")

    static_for_cohort = cohort.merge(
        static,
        on=PARTICIPANT_COLUMN,
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for variable in static_columns:
        tier = 1 if variable in consumed_columns else 2
        tier_label = TIER_1_LABEL if tier == 1 else TIER_2_LABEL
        validation_values = static_for_cohort.loc[
            static_for_cohort[SPLIT_COLUMN] == "validation", variable
        ]
        test_values = static_for_cohort.loc[
            static_for_cohort[SPLIT_COLUMN] == "test", variable
        ]
        validation_n = int(validation_values.notna().sum())
        test_n = int(test_values.notna().sum())
        complete_n = validation_n + test_n
        coverage_flag = (
            "LOW_COVERAGE_CAUTION"
            if test_n < LOW_COVERAGE_TEST_THRESHOLD
            else "ADEQUATE_TEST_COVERAGE"
        )
        unit, notes = infer_unit_and_notes(variable, tier_label)
        if coverage_flag == "LOW_COVERAGE_CAUTION":
            notes = f"LOW_COVERAGE_CAUTION. {notes}"
        audit_row = audit_by_column.loc[variable] if variable in consumed_columns else None
        field = static_schema.field(variable)
        rows.append(
            {
                "variable_name": variable,
                "source": "enriched",
                "source_files": (
                    f"{STATIC_PATH}|{FINAL_MULTIMODAL_PATH}"
                ),
                "parquet_dtype": str(field.type),
                "tier": tier,
                "tier_label": tier_label,
                "consumed_by_model": tier == 1,
                "model_input_order": (
                    int(audit_row["input_order"])
                    if audit_row is not None
                    else np.nan
                ),
                "model_feature_type": (
                    str(audit_row["feature_type"])
                    if audit_row is not None
                    else "not consumed"
                ),
                "t2d_validation_n": validation_n,
                "t2d_test_n": test_n,
                "complete_case_n": complete_n,
                "validation_coverage_pct": 100.0 * validation_n / 91,
                "test_coverage_pct": 100.0 * test_n / 83,
                "coverage_status": "ACTUAL_T2D_COUNTS",
                "coverage_flag": coverage_flag,
                "unit_or_representation": unit,
                "notes": notes,
                "model_input_evidence": (
                    str(STATIC_SCHEMA_AUDIT_PATH)
                    if tier == 1
                    else "No source-column match in the 44-input audit."
                ),
            }
        )
    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["tier", "complete_case_n", "variable_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return result


def estimate_t2d_coverage(row: pd.Series) -> tuple[float, float, str]:
    if row["source_table"] == "condition_occurrence":
        return (
            float(EXPECTED_COUNTS["validation"]),
            float(EXPECTED_COUNTS["test"]),
            "DEFINED_FOR_ALL_IF_CONDITION_ABSENCE_IS_ZERO",
        )
    full_validation = float(row["n_valid_numeric_validation"])
    full_test = float(row["n_valid_numeric_test"])
    if not np.isfinite(full_validation) or not np.isfinite(full_test):
        return np.nan, np.nan, "NOT_DETERMINABLE_WITHOUT_RAW_CONTENT_REVIEW"
    if (
        full_validation == FULL_SPLIT_COUNTS["validation"]
        and full_test == FULL_SPLIT_COUNTS["test"]
    ):
        return (
            float(EXPECTED_COUNTS["validation"]),
            float(EXPECTED_COUNTS["test"]),
            "EXACT_FROM_COMPLETE_FULL_SPLIT_COVERAGE",
        )
    validation_estimate = round(
        EXPECTED_COUNTS["validation"]
        * full_validation
        / FULL_SPLIT_COUNTS["validation"]
    )
    test_estimate = round(
        EXPECTED_COUNTS["test"] * full_test / FULL_SPLIT_COUNTS["test"]
    )
    return (
        float(validation_estimate),
        float(test_estimate),
        "PROPORTIONAL_ESTIMATE_FROM_PRIOR_FULL_COHORT_COVERAGE",
    )


def raw_notes(row: pd.Series, coverage_status: str, test_n: float) -> str:
    notes = [
        "TIER_2_NOT_MODEL_INPUT by definition; raw GCS data were not model inputs.",
        "No GCS object content was read or downloaded in this inventory step.",
    ]
    if coverage_status == "PROPORTIONAL_ESTIMATE_FROM_PRIOR_FULL_COHORT_COVERAGE":
        notes.append(
            "T2D counts are estimates scaled from the prior Step 0 full-cohort coverage; actual T2D counts require approved raw-table processing."
        )
    elif coverage_status == "EXACT_FROM_COMPLETE_FULL_SPLIT_COVERAGE":
        notes.append(
            "Prior Step 0 coverage included every validation and test participant, so all 91/83 T2D participants are necessarily covered."
        )
    elif coverage_status == "DEFINED_FOR_ALL_IF_CONDITION_ABSENCE_IS_ZERO":
        notes.append(
            "Condition flag coverage assumes absence of a condition-occurrence row is a valid zero; this requires semantic confirmation before analysis."
        )
    else:
        notes.append("Coverage requires approved raw content review.")
    if np.isfinite(test_n) and test_n < LOW_COVERAGE_TEST_THRESHOLD:
        notes.insert(0, "LOW_COVERAGE_CAUTION.")
    if bool(row.get("unsafe_to_analyze", False)):
        notes.append("Prior unit audit marked this candidate unsafe to analyze as-is.")
    if bool(row.get("manual_review_required", False)):
        notes.append("Manual review is required.")
    if row.get("clinical_domain") in {"lipid", "metabolic_glycemic"}:
        notes.append(
            "Fasting status is not established from object metadata and must be checked before use."
        )
    notes.append("Requires confirmation before any raw pull, merge, or processing.")
    return " ".join(notes)


def build_raw_inventory() -> pd.DataFrame:
    prior = pd.read_csv(PRIOR_RAW_INVENTORY_PATH, low_memory=False)
    raw_only = prior[prior["present_in_enriched_dataset"] == False].copy()  # noqa: E712
    if len(raw_only) != 127:
        raise RuntimeError(f"Expected 127 raw-only named candidates, observed {len(raw_only)}")
    rows: list[dict[str, object]] = []
    for _, row in raw_only.iterrows():
        validation_n, test_n, coverage_status = estimate_t2d_coverage(row)
        complete_n = (
            validation_n + test_n
            if np.isfinite(validation_n) and np.isfinite(test_n)
            else np.nan
        )
        coverage_flag = (
            "LOW_COVERAGE_CAUTION"
            if np.isfinite(test_n) and test_n < LOW_COVERAGE_TEST_THRESHOLD
            else (
                "COVERAGE_UNKNOWN_CAUTION"
                if not np.isfinite(test_n)
                else "ESTIMATED_ADEQUATE_TEST_COVERAGE"
            )
        )
        source_table = str(row["source_table"])
        rows.append(
            {
                "variable_name": str(row["normalized_target_name"]),
                "source": "raw_gcs",
                "source_table": source_table,
                "source_field_or_concept": row["source_field_or_concept"],
                "measurement_concept_id": row["measurement_concept_id"],
                "measurement_source_value": row["measurement_source_value"],
                "file_location": RAW_TABLE_PATHS[source_table],
                "apparent_format": "CSV, OMOP-style long table",
                "clinical_domain": row["clinical_domain"],
                "target_type": row["target_type"],
                "value_representation": row["value_representation"],
                "tier": 2,
                "tier_label": TIER_2_LABEL,
                "t2d_validation_n": validation_n,
                "t2d_test_n": test_n,
                "complete_case_n": complete_n,
                "coverage_status": coverage_status,
                "coverage_flag": coverage_flag,
                "prior_full_validation_n": row["n_valid_numeric_validation"],
                "prior_full_test_n": row["n_valid_numeric_test"],
                "prior_full_validation_coverage_pct": row[
                    "coverage_validation_pct"
                ],
                "prior_full_test_coverage_pct": row["coverage_test_pct"],
                "dominant_unit": row["dominant_unit"],
                "unit_status": row["unit_status"],
                "unsafe_to_analyze_as_is": row["unsafe_to_analyze"],
                "requires_raw_bucket_pull": True,
                "current_step_raw_content_accessed": False,
                "notes": raw_notes(row, coverage_status, test_n),
                "coverage_evidence": str(PRIOR_RAW_INVENTORY_PATH),
            }
        )

    placeholders = [
        (
            "procedure_occurrence_concepts_unenumerated",
            "procedure_occurrence",
            "gs://cgmproject2025/AIREADI/clinical_data/procedure_occurrence.csv",
            "CSV, OMOP-style long table",
            "procedures",
        ),
        (
            "visit_occurrence_detail_unenumerated",
            "visit_occurrence",
            "gs://cgmproject2025/AIREADI/clinical_data/visit_occurrence.csv",
            "CSV, OMOP-style long table",
            "encounters",
        ),
        (
            "person_demographic_detail_unenumerated",
            "person",
            "gs://cgmproject2025/AIREADI/clinical_data/person.csv",
            "CSV, OMOP-style table",
            "demographic",
        ),
        (
            "protected_medication_fields_unenumerated",
            "protected_medications",
            "gs://cgmproject2025/AIREADI/protected/AIREADI_Medications_Protected_Dataset.xlsx",
            "XLSX",
            "medication",
        ),
        (
            "protected_demographic_fields_unenumerated",
            "protected_demographics",
            "gs://cgmproject2025/AIREADI/protected/AIREADI_Demographics_Protected_Dataset.xlsx",
            "XLSX",
            "demographic",
        ),
    ]
    for variable, source_table, location, apparent_format, domain in placeholders:
        rows.append(
            {
                "variable_name": variable,
                "source": "raw_gcs",
                "source_table": source_table,
                "source_field_or_concept": "not enumerated",
                "measurement_concept_id": np.nan,
                "measurement_source_value": np.nan,
                "file_location": location,
                "apparent_format": apparent_format,
                "clinical_domain": domain,
                "target_type": "candidate variable set",
                "value_representation": "unknown without content review",
                "tier": 2,
                "tier_label": TIER_2_LABEL,
                "t2d_validation_n": np.nan,
                "t2d_test_n": np.nan,
                "complete_case_n": np.nan,
                "coverage_status": "NOT_DETERMINABLE_WITHOUT_RAW_CONTENT_REVIEW",
                "coverage_flag": "COVERAGE_UNKNOWN_CAUTION",
                "prior_full_validation_n": np.nan,
                "prior_full_test_n": np.nan,
                "prior_full_validation_coverage_pct": np.nan,
                "prior_full_test_coverage_pct": np.nan,
                "dominant_unit": "unknown",
                "unit_status": "not reviewed",
                "unsafe_to_analyze_as_is": True,
                "requires_raw_bucket_pull": True,
                "current_step_raw_content_accessed": False,
                "notes": (
                    "COVERAGE_UNKNOWN_CAUTION. Object name and metadata only; variable fields and coverage were not enumerated because content access requires confirmation. TIER_2_NOT_MODEL_INPUT by definition."
                ),
                "coverage_evidence": "GCS object listing metadata only",
            }
        )
    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["complete_case_n", "variable_name"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    return result


def build_consolidated(
    enriched: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    enriched_rows = enriched[
        [
            "variable_name",
            "source",
            "tier",
            "tier_label",
            "t2d_validation_n",
            "t2d_test_n",
            "complete_case_n",
            "coverage_status",
            "coverage_flag",
            "unit_or_representation",
            "notes",
            "source_files",
            "model_input_evidence",
        ]
    ].rename(
        columns={
            "source_files": "source_location",
            "unit_or_representation": "unit_or_format",
        }
    )
    enriched_rows["clinical_domain"] = "enriched clinical/static"
    raw_rows = raw[
        [
            "variable_name",
            "source",
            "tier",
            "tier_label",
            "t2d_validation_n",
            "t2d_test_n",
            "complete_case_n",
            "coverage_status",
            "coverage_flag",
            "dominant_unit",
            "notes",
            "file_location",
            "clinical_domain",
        ]
    ].rename(
        columns={
            "file_location": "source_location",
            "dominant_unit": "unit_or_format",
        }
    )
    raw_rows["model_input_evidence"] = (
        "Raw GCS-only candidate; not available to the frozen model."
    )
    combined = pd.concat([enriched_rows, raw_rows], ignore_index=True)
    ordered_columns = [
        "variable_name",
        "source",
        "tier",
        "tier_label",
        "t2d_validation_n",
        "t2d_test_n",
        "complete_case_n",
        "coverage_status",
        "coverage_flag",
        "clinical_domain",
        "unit_or_format",
        "source_location",
        "model_input_evidence",
        "notes",
    ]
    combined = combined[ordered_columns].sort_values(
        ["tier", "complete_case_n", "variable_name"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return combined


def build_manifest(
    enriched: pd.DataFrame,
    raw: pd.DataFrame,
    consolidated: pd.DataFrame,
) -> dict[str, object]:
    named_raw = raw[~raw["variable_name"].str.endswith("_unenumerated")]
    return {
        "analysis_name": "t2d_clinical_marker_inventory_tier_split",
        "status": "AWAITING_USER_REVIEW_BEFORE_RAW_PULL_OR_CLUSTERING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "inventory_and_sourcing_only": True,
            "clustering_performed": False,
            "raw_gcs_content_read_or_downloaded_current_step": False,
            "raw_gcs_operations": (
                "Object and prefix listing plus size/timestamp metadata only."
            ),
            "enriched_dataset_read_only": True,
        },
        "cohort": {
            "population": "T2D complete-case participants",
            "validation_n": 91,
            "test_n": 83,
            "pooled_n": 174,
            "id_source": str(CLINICAL_TARGETS_PATH),
            "id_reconstruction_validated": True,
            "coverage_audit": str(COVERAGE_AUDIT_PATH),
        },
        "tier_definition": {
            "1": TIER_1_LABEL,
            "2": TIER_2_LABEL,
            "tier_1_rule": (
                "Exact source-column match among the 44 consumed checkpoint static inputs in static_schema_audit.csv."
            ),
            "tier_2_rule": (
                "Enriched clinical/static column not in the 44-input audit, or raw GCS-only candidate."
            ),
        },
        "coverage_definition": {
            "enriched": (
                "Actual participant-level non-missing count among the frozen 91 validation and 83 test T2D IDs."
            ),
            "raw_named_candidate": (
                "Exact 91/83 only when prior full-split coverage was complete; otherwise proportional estimate from prior Step 0 aggregate coverage. Condition flags assume absence-as-zero."
            ),
            "raw_unenumerated_object": (
                "Unknown until user confirms content review or pull."
            ),
            "complete_case_n": "t2d_validation_n plus t2d_test_n",
            "low_coverage_rule": (
                f"LOW_COVERAGE_CAUTION when t2d_test_n < {LOW_COVERAGE_TEST_THRESHOLD}."
            ),
        },
        "inventory_counts": {
            "enriched_clinical_static_columns_excluding_participant_id": len(
                enriched
            ),
            "enriched_tier_1_model_inputs": int((enriched["tier"] == 1).sum()),
            "enriched_tier_2_not_model_inputs": int(
                (enriched["tier"] == 2).sum()
            ),
            "raw_named_candidates_absent_from_enriched": len(named_raw),
            "raw_unenumerated_candidate_sets": int(
                raw["variable_name"].str.endswith("_unenumerated").sum()
            ),
            "consolidated_rows": len(consolidated),
            "low_coverage_cautions": int(
                (consolidated["coverage_flag"] == "LOW_COVERAGE_CAUTION").sum()
            ),
            "coverage_unknown_cautions": int(
                (consolidated["coverage_flag"] == "COVERAGE_UNKNOWN_CAUTION").sum()
            ),
        },
        "raw_discovery": {
            "bucket_root": "gs://cgmproject2025/AIREADI/",
            "prefixes_listed": [
                "gs://cgmproject2025/AIREADI/",
                "gs://cgmproject2025/AIREADI/clinical_data/",
                "gs://cgmproject2025/AIREADI/protected/",
            ],
            "objects": RAW_GCS_OBJECTS,
            "current_step_content_access": "none",
            "named_candidate_evidence": str(PRIOR_RAW_INVENTORY_PATH),
            "named_candidate_evidence_note": (
                "Reused an existing aggregate Step 0 inventory; cached raw tables were not opened or reprocessed in this step."
            ),
        },
        "caveats": {
            "nonfasting": NONFASTING_CAVEAT,
            "age": AGE_CAVEAT,
            "raw_coverage": (
                "Most raw T2D counts are estimates, not participant-level counts. Do not use them for modeling decisions without approved raw validation."
            ),
            "condition_semantics": (
                "Condition coverage assumes absent occurrence rows represent negative flags and requires confirmation."
            ),
            "raw_units": (
                "Candidates marked unsafe or with missing/mixed units require harmonization before use."
            ),
        },
        "inputs": [
            {"path": str(FINAL_MULTIMODAL_PATH), "sha256": sha256_file(FINAL_MULTIMODAL_PATH)},
            {"path": str(STATIC_PATH), "sha256": sha256_file(STATIC_PATH)},
            {"path": str(FINAL_METADATA_PATH), "sha256": sha256_file(FINAL_METADATA_PATH)},
            {"path": str(CLINICAL_METADATA_PATH), "sha256": sha256_file(CLINICAL_METADATA_PATH)},
            {"path": str(SPLIT_PATH), "sha256": sha256_file(SPLIT_PATH)},
            {"path": str(CLINICAL_TARGETS_PATH), "sha256": sha256_file(CLINICAL_TARGETS_PATH)},
            {"path": str(COVERAGE_AUDIT_PATH), "sha256": sha256_file(COVERAGE_AUDIT_PATH)},
            {"path": str(STATIC_SCHEMA_AUDIT_PATH), "sha256": sha256_file(STATIC_SCHEMA_AUDIT_PATH)},
            {"path": str(PRIOR_RAW_INVENTORY_PATH), "sha256": sha256_file(PRIOR_RAW_INVENTORY_PATH)},
            {"path": str(PRIOR_STEP0_MANIFEST_PATH), "sha256": sha256_file(PRIOR_STEP0_MANIFEST_PATH)},
        ],
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        "outputs": [
            {"path": str(ENRICHED_OUTPUT_PATH), "sha256": sha256_file(ENRICHED_OUTPUT_PATH)},
            {"path": str(RAW_OUTPUT_PATH), "sha256": sha256_file(RAW_OUTPUT_PATH)},
            {"path": str(CONSOLIDATED_OUTPUT_PATH), "sha256": sha256_file(CONSOLIDATED_OUTPUT_PATH)},
            {"path": str(MANIFEST_PATH), "sha256": None, "note": "Self-hash intentionally omitted."},
        ],
    }


def assert_no_em_dash(paths: list[Path]) -> None:
    for path in paths:
        if NO_EM_DASH in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"Em dash found in output: {path}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    static = pd.read_parquet(STATIC_PATH)
    static[PARTICIPANT_COLUMN] = static[PARTICIPANT_COLUMN].astype(str)
    if static[PARTICIPANT_COLUMN].duplicated().any():
        raise RuntimeError("Duplicate participant IDs in static feature table")
    cohort = load_t2d_cohort(static)
    enriched = build_enriched_inventory(static, cohort)
    raw = build_raw_inventory()
    consolidated = build_consolidated(enriched, raw)

    atomic_csv(enriched, ENRICHED_OUTPUT_PATH)
    atomic_csv(raw, RAW_OUTPUT_PATH)
    atomic_csv(consolidated, CONSOLIDATED_OUTPUT_PATH)
    manifest = build_manifest(enriched, raw, consolidated)
    atomic_json(manifest, MANIFEST_PATH)
    assert_no_em_dash(
        [ENRICHED_OUTPUT_PATH, RAW_OUTPUT_PATH, CONSOLIDATED_OUTPUT_PATH, MANIFEST_PATH]
    )

    print(f"status={manifest['status']}")
    print(f"enriched_columns={len(enriched)}")
    print(f"tier1={int((enriched['tier'] == 1).sum())}")
    print(f"tier2_enriched={int((enriched['tier'] == 2).sum())}")
    print(f"raw_candidates={len(raw)}")
    print(f"consolidated_rows={len(consolidated)}")
    print(
        "low_coverage_cautions="
        f"{int((consolidated['coverage_flag'] == 'LOW_COVERAGE_CAUTION').sum())}"
    )
    print(f"consolidated={CONSOLIDATED_OUTPUT_PATH}")
    print(f"manifest={MANIFEST_PATH}")


if __name__ == "__main__":
    main()
