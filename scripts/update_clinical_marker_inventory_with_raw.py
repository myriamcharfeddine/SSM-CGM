#!/usr/bin/env python3
"""Extend the clinical marker inventory using authorized local copies of raw AI-READI data.

This script performs sourcing and aggregate coverage auditing only. It does not modify
the enriched dataset, create modeling inputs, or run clustering.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
RAW_ROOT = Path("/home/myriamcharfeddine/CGM/Data/clinical_data")
PROTECTED_ROOT = RAW_ROOT / "protected"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/clinical_marker_inventory"
ENRICHED_TABLE = OUTPUT_ROOT / "enriched_dataset_clinical_columns.csv"
PRIOR_TARGET_INVENTORY = (
    PROJECT_ROOT
    / "outputs/hidden_state_phenotype/step0_feasibility/clinical_target_inventory.csv"
)
CLINICAL_TARGETS = PROJECT_ROOT / "outputs/continuous_clinical/clinical_targets.parquet"
STATIC_AUDIT = (
    PROJECT_ROOT
    / "outputs/hidden_state_phenotype/step1_static_neutralization/20260724T223612Z/static_schema_audit.csv"
)

TIER_LABEL = "TIER_2_NOT_MODEL_INPUT"
LOW_COVERAGE_TEST_N = 60
EXPECTED_VALIDATION_N = 91
EXPECTED_TEST_N = 83

RAW_OBJECTS = {
    "condition_occurrence.csv": (
        "gs://cgmproject2025/AIREADI/clinical_data/condition_occurrence.csv",
        1_855_347,
        "liPxdMfRAkBDAthzg253CA==",
    ),
    "dqd_omop.json": (
        "gs://cgmproject2025/AIREADI/clinical_data/dqd_omop.json",
        2_276_116,
        "0wWQtHNd9oGmprrnYYNwHQ==",
    ),
    "measurement.csv": (
        "gs://cgmproject2025/AIREADI/clinical_data/measurement.csv",
        36_643_391,
        "sW9vhjYre6rYo7V8JPhuwQ==",
    ),
    "observation.csv": (
        "gs://cgmproject2025/AIREADI/clinical_data/observation.csv",
        113_309_618,
        "9m7/CLYL3Rc/Ez715ToTtA==",
    ),
    "person.csv": (
        "gs://cgmproject2025/AIREADI/clinical_data/person.csv",
        137_116,
        "+TDrzIUTervmjI1wqWzU2g==",
    ),
    "procedure_occurrence.csv": (
        "gs://cgmproject2025/AIREADI/clinical_data/procedure_occurrence.csv",
        7_307_425,
        "RohVwt0ZN/jTlOZxPcaIcQ==",
    ),
    "visit_occurrence.csv": (
        "gs://cgmproject2025/AIREADI/clinical_data/visit_occurrence.csv",
        643_135,
        "KMHWP5Ut1aKRKXIcXdBkKg==",
    ),
    "protected/AIREADI_DataDicitonary_Protected_Dataset.xlsx": (
        "gs://cgmproject2025/AIREADI/protected/AIREADI_DataDicitonary_Protected_Dataset.xlsx",
        14_244,
        "j1/jJmDFr35bhB9yY2bm/w==",
    ),
    "protected/AIREADI_Demographics_Protected_Dataset.xlsx": (
        "gs://cgmproject2025/AIREADI/protected/AIREADI_Demographics_Protected_Dataset.xlsx",
        260_883,
        "9bp5g+4vEKz68uDl4DkwNg==",
    ),
    "protected/AIREADI_Medications_Protected_Dataset.xlsx": (
        "gs://cgmproject2025/AIREADI/protected/AIREADI_Medications_Protected_Dataset.xlsx",
        1_148_179,
        "vbhfuOR3YaBado67GNKVvw==",
    ),
}

LONG_TABLES = {
    "measurement": {
        "path": RAW_ROOT / "measurement.csv",
        "source": "measurement_source_value",
        "concept": "measurement_concept_id",
        "value": ["value_as_number", "value_as_concept_id", "value_source_value"],
        "unit": "unit_source_value",
        "date": "measurement_date",
    },
    "observation": {
        "path": RAW_ROOT / "observation.csv",
        "source": "observation_source_value",
        "concept": "observation_concept_id",
        "value": [
            "value_as_number",
            "value_as_string",
            "value_as_concept_id",
            "value_source_value",
        ],
        "unit": "unit_source_value",
        "date": "observation_date",
    },
    "condition": {
        "path": RAW_ROOT / "condition_occurrence.csv",
        "source": "condition_source_value",
        "concept": "condition_concept_id",
        "value": [],
        "unit": None,
        "date": "condition_start_date",
    },
    "procedure": {
        "path": RAW_ROOT / "procedure_occurrence.csv",
        "source": "procedure_source_value",
        "concept": "procedure_concept_id",
        "value": [],
        "unit": None,
        "date": "procedure_date",
    },
    "visit": {
        "path": RAW_ROOT / "visit_occurrence.csv",
        "source": "visit_source_value",
        "concept": "visit_concept_id",
        "value": [],
        "unit": None,
        "date": "visit_start_date",
    },
}

TECHNICAL_OBSERVATION_PATTERNS = (
    "studyid",
    "cmpdat",
    "visdat",
    "date",
    "datetime",
    "brthyy",
    "cage",
)

DEMOGRAPHIC_REFLECTED_IN_ENRICHED = {
    "studyid",
    "rp_childpot",
    "scrsex",
    "raceot",
    "race2",
    "ethnicot",
    "racetrib",
    "ancestry",
    "cl_maristat",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_hex(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def base64_md5_to_hex(value: str) -> str:
    return base64.b64decode(value).hex()


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def nonmissing(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        text = series.astype("string").str.strip()
        return text.notna() & ~text.str.lower().isin({"", "nan", "none", "<na>"})
    return series.notna()


def slug(value: object) -> str:
    text = str(value).split(",", 1)[0].strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:100] or "unnamed_variable"


def split_counts(ids: pd.Series, cohort: pd.DataFrame) -> tuple[int, int, int]:
    present = set(normalize_id(ids).dropna())
    validation = int(cohort.loc[cohort["split"] == "validation", "participant_id"].isin(present).sum())
    test = int(cohort.loc[cohort["split"] == "test", "participant_id"].isin(present).sum())
    return validation, test, validation + test


def coverage_flag(test_n: int) -> str:
    return "LOW_COVERAGE_CAUTION" if test_n < LOW_COVERAGE_TEST_N else "ADEQUATE_TEST_COVERAGE"


def unique_join(series: pd.Series, limit: int = 12) -> str:
    values = sorted({str(v).strip() for v in series.dropna() if str(v).strip()})
    if len(values) > limit:
        return "|".join(values[:limit]) + f"|...({len(values)} total)"
    return "|".join(values)


def all_long_variables(cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    loaded: dict[str, pd.DataFrame] = {}
    for table, spec in LONG_TABLES.items():
        frame = pd.read_csv(spec["path"], low_memory=False)
        frame["person_id"] = normalize_id(frame["person_id"])
        loaded[table] = frame
        for (source_value, concept_id), group in frame.groupby(
            [spec["source"], spec["concept"]], dropna=False, sort=True
        ):
            t2d = group[group["person_id"].isin(set(cohort["participant_id"]))].copy()
            if spec["value"]:
                valid = pd.Series(False, index=t2d.index)
                for value_column in spec["value"]:
                    valid = valid | nonmissing(t2d[value_column])
                complete_ids = t2d.loc[valid, "person_id"]
                complete_definition = "participant has at least one nonmissing value"
            else:
                complete_ids = t2d["person_id"]
                complete_definition = "participant has at least one occurrence row"
            val_n, test_n, complete_n = split_counts(complete_ids, cohort)
            unit = unique_join(group[spec["unit"]]) if spec["unit"] else "not_applicable"
            raw_label = str(source_value)
            rows.append(
                {
                    "variable_name": slug(raw_label),
                    "source_table": table,
                    "source_field_or_concept": f"{spec['source']}|{concept_id}",
                    "concept_id": concept_id,
                    "source_value": raw_label,
                    "file_location": str(spec["path"]),
                    "gcs_location": RAW_OBJECTS[spec["path"].name][0],
                    "apparent_format": "CSV, OMOP-style long table",
                    "t2d_validation_n": val_n,
                    "t2d_test_n": test_n,
                    "complete_case_n": complete_n,
                    "coverage_status": "DIRECT_ACTUAL_T2D_COUNTS",
                    "coverage_flag": coverage_flag(test_n),
                    "complete_case_definition": complete_definition,
                    "unit_or_format": unit or "missing_or_not_recorded",
                    "n_rows_all_participants": int(len(group)),
                    "n_unique_participants_all": int(group["person_id"].nunique()),
                    "n_rows_t2d": int(len(t2d)),
                    "date_min": str(group[spec["date"]].dropna().min()) if group[spec["date"]].notna().any() else "",
                    "date_max": str(group[spec["date"]].dropna().max()) if group[spec["date"]].notna().any() else "",
                    "raw_content_accessed": True,
                }
            )
    result = pd.DataFrame(rows)
    result["variable_name"] = make_unique_names(result, ["source_table", "source_value", "concept_id"])
    return result, loaded


def make_unique_names(frame: pd.DataFrame, identity: list[str]) -> pd.Series:
    output = frame["variable_name"].copy()
    duplicates = output.duplicated(keep=False)
    for index in frame.index[duplicates]:
        suffix = slug("_".join(str(frame.at[index, column]) for column in identity))[-60:]
        output.at[index] = f"{output.at[index]}__{suffix}"
    return output


def wide_field_inventory(
    frame: pd.DataFrame,
    id_column: str,
    source_table: str,
    path: Path,
    gcs_location: str,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    work = frame.copy()
    work[id_column] = normalize_id(work[id_column])
    t2d = work[work[id_column].isin(set(cohort["participant_id"]))]
    rows = []
    for column in work.columns:
        if column == id_column:
            continue
        valid_t2d = t2d.loc[nonmissing(t2d[column]), id_column]
        val_n, test_n, complete_n = split_counts(valid_t2d, cohort)
        rows.append(
            {
                "variable_name": slug(column),
                "source_table": source_table,
                "source_field_or_concept": column,
                "concept_id": "",
                "source_value": column,
                "file_location": str(path),
                "gcs_location": gcs_location,
                "apparent_format": "XLSX, protected wide table" if path.suffix == ".xlsx" else "CSV, OMOP person table",
                "t2d_validation_n": val_n,
                "t2d_test_n": test_n,
                "complete_case_n": complete_n,
                "coverage_status": "DIRECT_ACTUAL_T2D_COUNTS",
                "coverage_flag": coverage_flag(test_n),
                "complete_case_definition": "participant has at least one nonmissing field value",
                "unit_or_format": str(work[column].dtype),
                "n_rows_all_participants": int(nonmissing(work[column]).sum()),
                "n_unique_participants_all": int(work.loc[nonmissing(work[column]), id_column].nunique()),
                "n_rows_t2d": int(nonmissing(t2d[column]).sum()),
                "date_min": "",
                "date_max": "",
                "raw_content_accessed": True,
            }
        )
    return pd.DataFrame(rows)


def build_all_raw_variables(cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    long_variables, loaded = all_long_variables(cohort)
    person_path = RAW_ROOT / "person.csv"
    person = pd.read_csv(person_path, low_memory=False)
    person_variables = wide_field_inventory(
        person,
        "person_id",
        "person",
        person_path,
        RAW_OBJECTS["person.csv"][0],
        cohort,
    )
    demo_path = PROTECTED_ROOT / "AIREADI_Demographics_Protected_Dataset.xlsx"
    demo = pd.read_excel(demo_path)
    demo_variables = wide_field_inventory(
        demo,
        "studyid",
        "protected_demographics",
        demo_path,
        RAW_OBJECTS[f"protected/{demo_path.name}"][0],
        cohort,
    )
    medication_path = PROTECTED_ROOT / "AIREADI_Medications_Protected_Dataset.xlsx"
    medication = pd.read_excel(medication_path)
    medication_variables = wide_field_inventory(
        medication,
        "studyid",
        "protected_medications",
        medication_path,
        RAW_OBJECTS[f"protected/{medication_path.name}"][0],
        cohort,
    )
    all_variables = pd.concat(
        [long_variables, person_variables, demo_variables, medication_variables],
        ignore_index=True,
    )
    all_variables.insert(1, "source", "raw_gcs")
    all_variables["tier"] = 2
    all_variables["tier_label"] = TIER_LABEL
    all_variables["requires_raw_bucket_pull"] = False
    all_variables["coverage_evidence"] = "Direct authorized scan of current GCS-matched local source file"
    all_variables = all_variables.sort_values(
        ["source_table", "complete_case_n", "variable_name"], ascending=[True, False, True]
    ).reset_index(drop=True)
    return all_variables, loaded


def technical_observation(source_value: str) -> bool:
    lowered = source_value.lower()
    code = lowered.split(",", 1)[0].strip()
    return any(pattern in code or pattern in lowered for pattern in TECHNICAL_OBSERVATION_PATTERNS)


def candidate_priority(domain: str, test_n: int, unsafe: bool, source_table: str) -> str:
    if test_n < LOW_COVERAGE_TEST_N or unsafe:
        return "LOW"
    if source_table == "measurement" and domain in {
        "cardiovascular",
        "cognitive",
        "hematologic",
        "hepatic",
        "inflammatory",
        "lipid",
        "metabolic_glycemic",
        "neuropathy",
        "renal",
    }:
        return "HIGH"
    return "MEDIUM"


def build_candidates(
    all_raw: pd.DataFrame,
    prior: pd.DataFrame,
    cohort: pd.DataFrame,
    loaded: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    candidates: list[dict[str, object]] = []
    lookup = {
        (str(row.source_table), str(row.source_value)): row
        for row in all_raw.itertuples(index=False)
    }
    concept_lookup: dict[tuple[str, str], list[object]] = {}
    for row in all_raw.itertuples(index=False):
        concept_key = (str(row.source_table), str(row.concept_id))
        concept_lookup.setdefault(concept_key, []).append(row)
    prior_raw_only = prior.loc[~prior["present_in_enriched_dataset"].fillna(False).astype(bool)].copy()
    prior_observation_values = set(prior.loc[prior["source_table"] == "observation", "measurement_source_value"].astype(str))
    condition_codes = {
        str(value).split(",", 1)[0].strip()
        for value in all_raw.loc[all_raw["source_table"] == "condition", "source_value"]
    }
    for old in prior_raw_only.itertuples(index=False):
        old_table = {"condition_occurrence": "condition"}.get(str(old.source_table), str(old.source_table))
        key = (old_table, str(old.measurement_source_value))
        direct = lookup.get(key)
        if direct is None:
            concept_key = (old_table, str(old.measurement_concept_id))
            matches = concept_lookup.get(concept_key, [])
            if len(matches) == 1:
                direct = matches[0]
            else:
                prefix_matches = [
                    row
                    for row in matches
                    if str(row.source_value).startswith(str(old.measurement_source_value))
                    or str(old.measurement_source_value).startswith(str(row.source_value))
                ]
                if len(prefix_matches) == 1:
                    direct = prefix_matches[0]
        if direct is None:
            raise RuntimeError(f"Could not directly match prior raw target: {key}")
        source_column = LONG_TABLES[old_table]["source"]
        source_rows = loaded[old_table].loc[
            loaded[old_table][source_column].astype(str) == str(direct.source_value)
        ]
        if str(old.value_representation) == "value_as_number":
            complete_ids = source_rows.loc[nonmissing(source_rows["value_as_number"]), "person_id"]
            complete_definition = "participant has at least one nonmissing value_as_number"
        else:
            complete_ids = source_rows["person_id"]
            complete_definition = "participant has at least one occurrence row"
        prior_val_n, prior_test_n, prior_complete_n = split_counts(complete_ids, cohort)
        unsafe = bool(old.unsafe_to_analyze)
        row = direct._asdict()
        row["t2d_validation_n"] = prior_val_n
        row["t2d_test_n"] = prior_test_n
        row["complete_case_n"] = prior_complete_n
        row["complete_case_definition"] = complete_definition
        row.update(
            {
                "variable_name": old.normalized_target_name,
                "clinical_domain": old.clinical_domain,
                "target_type": old.target_type,
                "value_representation": old.value_representation,
                "already_reflected_in_enriched": False,
                "model_input_evidence": old.model_input_evidence_source,
                "unsafe_to_analyze_as_is": unsafe,
                "unit_status": old.unit_status,
                "usefulness_priority": candidate_priority(
                    str(old.clinical_domain), prior_test_n, unsafe, old_table
                ),
                "candidate_origin": "Prior Step 0 target, now directly verified in fixed T2D cohort",
                "notes": "Raw-only Tier 2 candidate. Direct T2D coverage replaces the earlier full-cohort proportional estimate.",
            }
        )
        candidates.append(row)

    observation_rows = all_raw[all_raw["source_table"] == "observation"]
    for direct in observation_rows.itertuples(index=False):
        source_value = str(direct.source_value)
        code = source_value.split(",", 1)[0].strip()
        if source_value in prior_observation_values:
            continue
        if technical_observation(source_value):
            continue
        if code in condition_codes or code.startswith("mhoccur_") or code.startswith("mhterm_"):
            continue
        row = direct._asdict()
        row.update(
            {
                "clinical_domain": "questionnaire_or_self_report",
                "target_type": "mixed_questionnaire_field",
                "value_representation": "OMOP observation value fields",
                "already_reflected_in_enriched": False,
                "model_input_evidence": str(STATIC_AUDIT),
                "unsafe_to_analyze_as_is": False,
                "unit_status": "review_value_coding_before_analysis",
                "usefulness_priority": "LOW" if int(direct.t2d_test_n) < LOW_COVERAGE_TEST_N else "MEDIUM",
                "candidate_origin": "Newly enumerated after authorized raw observation scan",
                "notes": "Tier 2 questionnaire or self-report candidate. Coding and construct validity require review before analysis.",
            }
        )
        candidates.append(row)

    demographic_keep = {
        "genderid",
        "genderidot",
        "mhcat_dmtaot",
        "mhoccur_cnsot",
        "mhoccur_cnrot",
        "pxhic6",
        "pxhic9",
        "dvenvlocn",
    }
    for direct in all_raw[all_raw["source_table"] == "protected_demographics"].itertuples(index=False):
        if str(direct.source_value) not in demographic_keep:
            continue
        row = direct._asdict()
        row.update(
            {
                "clinical_domain": "demographic_or_social_context",
                "target_type": "categorical_or_free_text",
                "value_representation": "protected wide field",
                "already_reflected_in_enriched": False,
                "model_input_evidence": str(STATIC_AUDIT),
                "unsafe_to_analyze_as_is": str(direct.source_value).endswith("ot"),
                "unit_status": "not_applicable",
                "usefulness_priority": "MEDIUM" if str(direct.source_value) in {"genderid", "pxhic6", "pxhic9"} else "LOW",
                "candidate_origin": "Newly enumerated after authorized protected-demographics scan",
                "notes": "Tier 2 protected field. Use only aggregate or appropriately encoded values; free text requires governance review.",
            }
        )
        candidates.append(row)

    medication_exclude = {"redcap_repeat_instrument", "redcap_repeat_instance"}
    for direct in all_raw[all_raw["source_table"] == "protected_medications"].itertuples(index=False):
        if str(direct.source_value) in medication_exclude:
            continue
        row = direct._asdict()
        row.update(
            {
                "clinical_domain": "medication_detail",
                "target_type": "participant_level_medication_record_field",
                "value_representation": "protected repeated medication record",
                "already_reflected_in_enriched": False,
                "model_input_evidence": str(STATIC_AUDIT),
                "unsafe_to_analyze_as_is": False,
                "unit_status": "field_specific",
                "usefulness_priority": "HIGH" if str(direct.source_value) in {"rxnorm_code", "rxnorm_term"} else "MEDIUM",
                "candidate_origin": "Newly enumerated after authorized protected-medication scan",
                "notes": "Tier 2 medication detail. The frozen model used only seven derived diabetes-drug flags, not drug identity, route, dose, or frequency.",
            }
        )
        candidates.append(row)

    result = pd.DataFrame(candidates)
    result["tier"] = 2
    result["tier_label"] = TIER_LABEL
    result["coverage_flag"] = result["t2d_test_n"].astype(int).map(coverage_flag)
    result["requires_raw_bucket_pull"] = False
    result["raw_content_accessed"] = True
    result["coverage_status"] = "DIRECT_ACTUAL_T2D_COUNTS"
    if result["variable_name"].duplicated().any():
        duplicate_indices = result.index[result["variable_name"].duplicated(keep=False)]
        for index in duplicate_indices:
            result.at[index, "variable_name"] = (
                f"{result.at[index, 'variable_name']}__{slug(result.at[index, 'source_table'])}"
            )
    result = result.sort_values(
        ["usefulness_priority", "complete_case_n", "variable_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    preferred = [
        "variable_name",
        "source",
        "tier",
        "tier_label",
        "source_table",
        "source_field_or_concept",
        "concept_id",
        "source_value",
        "clinical_domain",
        "target_type",
        "value_representation",
        "t2d_validation_n",
        "t2d_test_n",
        "complete_case_n",
        "coverage_status",
        "coverage_flag",
        "complete_case_definition",
        "unit_or_format",
        "unit_status",
        "unsafe_to_analyze_as_is",
        "usefulness_priority",
        "already_reflected_in_enriched",
        "file_location",
        "gcs_location",
        "apparent_format",
        "requires_raw_bucket_pull",
        "raw_content_accessed",
        "candidate_origin",
        "notes",
        "coverage_evidence",
        "model_input_evidence",
        "n_rows_all_participants",
        "n_unique_participants_all",
        "n_rows_t2d",
        "date_min",
        "date_max",
    ]
    return result[preferred]


def build_consolidated(enriched: pd.DataFrame, raw_candidates: pd.DataFrame) -> pd.DataFrame:
    enriched_rows = pd.DataFrame(
        {
            "variable_name": enriched["variable_name"],
            "source": "enriched",
            "tier": enriched["tier"].astype(int),
            "tier_label": enriched["tier_label"],
            "t2d_validation_n": enriched["t2d_validation_n"].astype(int),
            "t2d_test_n": enriched["t2d_test_n"].astype(int),
            "complete_case_n": enriched["complete_case_n"].astype(int),
            "coverage_status": enriched["coverage_status"],
            "coverage_flag": enriched["coverage_flag"],
            "clinical_domain": "enriched clinical/static",
            "unit_or_format": enriched["unit_or_representation"],
            "source_location": enriched["source_files"],
            "usefulness_priority": "NOT_APPLICABLE",
            "model_input_evidence": enriched["model_input_evidence"],
            "notes": enriched["notes"],
        }
    )
    raw_rows = pd.DataFrame(
        {
            "variable_name": raw_candidates["variable_name"],
            "source": "raw_gcs",
            "tier": 2,
            "tier_label": TIER_LABEL,
            "t2d_validation_n": raw_candidates["t2d_validation_n"].astype(int),
            "t2d_test_n": raw_candidates["t2d_test_n"].astype(int),
            "complete_case_n": raw_candidates["complete_case_n"].astype(int),
            "coverage_status": "DIRECT_ACTUAL_T2D_COUNTS",
            "coverage_flag": raw_candidates["coverage_flag"],
            "clinical_domain": raw_candidates["clinical_domain"],
            "unit_or_format": raw_candidates["unit_or_format"],
            "source_location": raw_candidates["gcs_location"],
            "usefulness_priority": raw_candidates["usefulness_priority"],
            "model_input_evidence": raw_candidates["model_input_evidence"],
            "notes": raw_candidates["notes"],
        }
    )
    result = pd.concat([enriched_rows, raw_rows], ignore_index=True)
    result = result.sort_values(
        ["tier", "complete_case_n", "variable_name"], ascending=[True, False, True]
    ).reset_index(drop=True)
    return result


def build_source_manifest() -> pd.DataFrame:
    rows = []
    for relative, (gcs_uri, expected_size, cloud_md5_b64) in RAW_OBJECTS.items():
        local = RAW_ROOT / relative
        local_size = local.stat().st_size
        local_md5 = md5_hex(local)
        cloud_md5_hex = base64_md5_to_hex(cloud_md5_b64)
        rows.append(
            {
                "gcs_uri": gcs_uri,
                "local_path": str(local),
                "size_bytes": local_size,
                "expected_cloud_size_bytes": expected_size,
                "cloud_md5_base64": cloud_md5_b64,
                "cloud_md5_hex": cloud_md5_hex,
                "local_md5_hex": local_md5,
                "size_verified": local_size == expected_size,
                "md5_verified": local_md5 == cloud_md5_hex,
                "local_sha256": sha256(local),
            }
        )
    result = pd.DataFrame(rows)
    if not result["size_verified"].all() or not result["md5_verified"].all():
        raise RuntimeError("At least one local raw file does not match the current GCS object metadata")
    return result


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(CLINICAL_TARGETS, columns=["participant_id", "split"])
    cohort["participant_id"] = normalize_id(cohort["participant_id"])
    counts = cohort["split"].value_counts().to_dict()
    if counts != {"validation": EXPECTED_VALIDATION_N, "test": EXPECTED_TEST_N}:
        raise RuntimeError(f"Unexpected fixed T2D cohort counts: {counts}")
    if cohort["participant_id"].duplicated().any():
        raise RuntimeError("T2D cohort has duplicate participant identifiers")

    source_manifest = build_source_manifest()
    dqd = json.loads((RAW_ROOT / "dqd_omop.json").read_text())
    dqd_overview = {
        key: (value[0] if isinstance(value, list) and len(value) == 1 else value)
        for key, value in dqd["Overview"].items()
    }
    dqd_failed_by_table: dict[str, int] = {}
    for check in dqd["CheckResults"]:
        if int(check.get("failed", 0)) == 1:
            table = str(check.get("cdmTableName", "UNKNOWN"))
            dqd_failed_by_table[table] = dqd_failed_by_table.get(table, 0) + 1
    all_raw, loaded = build_all_raw_variables(cohort)
    prior = pd.read_csv(PRIOR_TARGET_INVENTORY, low_memory=False)
    candidates = build_candidates(all_raw, prior, cohort, loaded)
    enriched = pd.read_csv(ENRICHED_TABLE)
    consolidated = build_consolidated(enriched, candidates)

    output_paths = {
        "enriched": ENRICHED_TABLE,
        "raw_all": OUTPUT_ROOT / "raw_gcs_all_variables.csv",
        "raw_candidates": OUTPUT_ROOT / "raw_gcs_candidate_variables.csv",
        "consolidated": OUTPUT_ROOT / "consolidated_tier_inventory.csv",
        "source_manifest": OUTPUT_ROOT / "raw_gcs_source_manifest.csv",
    }
    atomic_csv(all_raw, output_paths["raw_all"])
    atomic_csv(candidates, output_paths["raw_candidates"])
    atomic_csv(consolidated, output_paths["consolidated"])
    atomic_csv(source_manifest, output_paths["source_manifest"])

    source_id_overlap = {}
    for table, frame in loaded.items():
        ids = set(normalize_id(frame["person_id"]).dropna())
        source_id_overlap[table] = int(cohort["participant_id"].isin(ids).sum())
    medication = pd.read_excel(PROTECTED_ROOT / "AIREADI_Medications_Protected_Dataset.xlsx")
    demographics = pd.read_excel(PROTECTED_ROOT / "AIREADI_Demographics_Protected_Dataset.xlsx")
    source_id_overlap["protected_medications"] = int(
        cohort["participant_id"].isin(set(normalize_id(medication["studyid"]).dropna())).sum()
    )
    source_id_overlap["protected_demographics"] = int(
        cohort["participant_id"].isin(set(normalize_id(demographics["studyid"]).dropna())).sum()
    )

    manifest_path = OUTPUT_ROOT / "inventory_manifest.json"
    manifest = {
        "analysis_name": "Clinical marker inventory and Tier 1/Tier 2 split, authorized raw extension",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AWAITING_USER_REVIEW_BEFORE_CLUSTERING",
        "scope": {
            "clustering_performed": False,
            "enriched_dataset_modified": False,
            "raw_gcs_content_access_authorized": True,
            "participant_level_values_exported": False,
            "aggregate_coverage_only": True,
        },
        "cohort": {
            "source": str(CLINICAL_TARGETS),
            "t2d_validation_n": EXPECTED_VALIDATION_N,
            "t2d_test_n": EXPECTED_TEST_N,
            "t2d_total_n": EXPECTED_VALIDATION_N + EXPECTED_TEST_N,
            "source_identifier_overlap": source_id_overlap,
        },
        "tier_definition": {
            "tier_1": "Exact enriched static source columns consumed by the frozen checkpoint.",
            "tier_2": "Enriched columns not consumed or raw-only variables not fed to the checkpoint.",
        },
        "coverage_definition": {
            "raw_long_value": "Participant has at least one row with a nonmissing value field.",
            "raw_occurrence": "Participant has at least one occurrence row. Absence is not automatically a confirmed negative.",
            "raw_wide": "Participant has at least one nonmissing field value.",
            "complete_case_n": "Validation plus test participant coverage.",
            "low_coverage_threshold": LOW_COVERAGE_TEST_N,
            "all_raw_coverage_is_direct": True,
        },
        "inventory_counts": {
            "enriched_columns": int(len(enriched)),
            "tier_1_model_inputs": int((enriched["tier"].astype(int) == 1).sum()),
            "enriched_tier_2": int((enriched["tier"].astype(int) == 2).sum()),
            "raw_all_source_variables": int(len(all_raw)),
            "raw_tier_2_candidates": int(len(candidates)),
            "consolidated_rows": int(len(consolidated)),
            "raw_candidate_priority": candidates["usefulness_priority"].value_counts().to_dict(),
            "raw_candidate_low_coverage": int((candidates["coverage_flag"] == "LOW_COVERAGE_CAUTION").sum()),
        },
        "raw_source_verification": {
            "all_local_sizes_match_gcs": bool(source_manifest["size_verified"].all()),
            "all_local_md5_match_gcs": bool(source_manifest["md5_verified"].all()),
            "dqd_omop_downloaded_in_this_extension": True,
            "other_files_already_local_and_current": True,
            "source_manifest": str(output_paths["source_manifest"]),
        },
        "data_quality_report": {
            "path": str(RAW_ROOT / "dqd_omop.json"),
            "overview": dqd_overview,
            "failed_checks_by_table": dict(sorted(dqd_failed_by_table.items())),
            "interpretation": "The DQD report is source-level, not specific to the fixed T2D subset. Unit and foreign-key failures require marker-level review.",
        },
        "candidate_selection": {
            "included": [
                "All prior Step 0 raw-only clinical targets with direct fixed-T2D coverage",
                "Nontechnical questionnaire and self-report observations not duplicated as condition concepts",
                "Raw-only protected demographic and social-context fields",
                "Medication identity, route, dose, unit, frequency, and RxNorm fields",
            ],
            "excluded_from_candidate_table_but_retained_in_raw_gcs_all_variables": [
                "Identifiers and form or visit dates",
                "Raw fields already reflected in enriched static columns",
                "Procedure and visit metadata without phenotype outcomes",
                "Person fields used to derive age, sex, race, ethnicity, or site",
                "Duplicate observation condition flags when condition_occurrence provides the concept",
            ],
        },
        "caveats": [
            "C-peptide and triglycerides are not guaranteed fasting.",
            "Condition and procedure row absence cannot be interpreted as a confirmed negative without semantic validation.",
            "Questionnaire coding must be reviewed before constructing analytic targets.",
            "Medication detail is repeated-record data and requires participant-level aggregation before analysis.",
            "Free-text protected fields require governance review and should not be exported at participant level.",
            "Raw variables remain external Tier 2 targets and must not be added to forecasting-model inputs.",
        ],
        "inputs": [
            {"path": str(CLINICAL_TARGETS), "sha256": sha256(CLINICAL_TARGETS)},
            {"path": str(PRIOR_TARGET_INVENTORY), "sha256": sha256(PRIOR_TARGET_INVENTORY)},
            {"path": str(STATIC_AUDIT), "sha256": sha256(STATIC_AUDIT)},
            {"path": str(ENRICHED_TABLE), "sha256": sha256(ENRICHED_TABLE)},
        ],
        "outputs": [],
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "python": os.sys.version.split()[0],
            "pandas": pd.__version__,
        },
    }
    for key, path in output_paths.items():
        manifest["outputs"].append({"name": key, "path": str(path), "sha256": sha256(path)})
    manifest["outputs"].append(
        {"name": "manifest", "path": str(manifest_path), "sha256": None, "note": "Self-hash intentionally omitted."}
    )
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "raw_all_source_variables": len(all_raw),
                "raw_tier_2_candidates": len(candidates),
                "consolidated_rows": len(consolidated),
                "candidate_priorities": candidates["usefulness_priority"].value_counts().to_dict(),
                "raw_candidate_low_coverage": int((candidates["coverage_flag"] == "LOW_COVERAGE_CAUTION").sum()),
                "source_id_overlap": source_id_overlap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
