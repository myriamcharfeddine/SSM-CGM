#!/usr/bin/env python3
"""Freeze the confirmed per-variable test application plan."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
RUN_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/per_variable_static_conditioning_audit/"
    "20260727T214535Z"
)
SCHEMA_ROOT = RUN_ROOT / "schema_audit"
VALIDATION_ROOT = RUN_ROOT / "validation_interventions"
FROZEN_ROOT = RUN_ROOT / "frozen_plan"
REFERENCE_PATH = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step1_static_neutralization/"
    "20260724T223612Z/static_reference_profile.json"
)
PCA_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step3_validation_clustering/"
    "20260725T001123Z/frozen_validation_pipeline/full_all"
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
ENCODING_GROUPS = {
    "age": ["participants_age"],
    "bmi": ["bmi_baseline"],
    "hba1c": ["hba1c_percent_baseline"],
    "study_group": ["participants_study_group"],
    "sex": ["demo_sex_at_birth"],
    "insulin": ["med_insulin"],
    "metformin": ["med_metformin"],
    "glp1": ["med_glp1_or_gip_glp1"],
}
SHARED_ANY_DRUG_DISPOSITION = "leave_factual"
SEX_REFERENCE_CATEGORY = "F"
SEX_PRESENTATION_ESTIMAND = "median_among_participants_differing_from_reference"
K_NN = 10
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_ALPHA = 0.05
RANDOM_SEED = 42
MEDICATION_MIN_GROUP_N = 30
REPRESENTATION_FREQUENCY_MINUTES = 15
BURN_IN_MINUTES = 0
PCA_COMPONENT_COUNT = 23
EXPECTED_VALIDATION_N = 239
EXPECTED_TEST_N = 221


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    FROZEN_ROOT.mkdir(parents=True, exist_ok=False)
    validation_summary_path = (
        VALIDATION_ROOT / "validation_per_variable_static_effects.csv"
    )
    presentation_summary_path = (
        VALIDATION_ROOT
        / "validation_per_variable_static_effects_presentation.csv"
    )
    sex_sensitivity_path = (
        VALIDATION_ROOT / "validation_sex_effect_sensitivity.csv"
    )
    step_b_manifest_path = VALIDATION_ROOT / "step_b_manifest.json"
    sex_amendment_path = (
        VALIDATION_ROOT / "step_b_sex_presentation_amendment.json"
    )
    mapping_path = SCHEMA_ROOT / "variable_encoding_group_audit.csv"
    prevalence_path = SCHEMA_ROOT / "medication_prevalence_validation.csv"

    validation_summary = pd.read_csv(validation_summary_path)
    presentation_summary = pd.read_csv(presentation_summary_path)
    medication_prevalence = pd.read_csv(prevalence_path)
    if validation_summary["n_participants"].nunique() != 1:
        raise RuntimeError("Validation summary participant counts are inconsistent")
    if int(validation_summary["n_participants"].iloc[0]) != EXPECTED_VALIDATION_N:
        raise RuntimeError("Validation summary does not contain 239 participants")
    sex_row = presentation_summary[
        presentation_summary["variable"].eq("sex")
    ].iloc[0]
    if int(sex_row["n_participants"]) != 100:
        raise RuntimeError("Confirmed validation sex presentation does not use n=100")

    reference = json.loads(REFERENCE_PATH.read_text())
    plan = {
        "plan_name": "per_variable_static_conditioning_test_application",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "user_confirmed_amended_step_b",
        "test_participant_count": EXPECTED_TEST_N,
        "variables_in_order": VARIABLE_ORDER,
        "encoding_groups": ENCODING_GROUPS,
        "reference_profile_path": str(REFERENCE_PATH),
        "reference_profile_hash": sha256_file(REFERENCE_PATH),
        "reference_profile_internal_hash": reference["profile_hash"],
        "medication_intervention": {
            "direction": "replace drug-specific channel with Step 1 reference",
            "shared_any_diabetes_drug_disposition": SHARED_ANY_DRUG_DISPOSITION,
            "minimum_group_n": MEDICATION_MIN_GROUP_N,
            "validation_prevalence": medication_prevalence.to_dict("records"),
            "insulin_low_confidence": True,
        },
        "representation": {
            "aggregation": "dimensionwise_median",
            "eligible_rows": "valid CGM post-update states",
            "sampling_frequency_minutes": REPRESENTATION_FREQUENCY_MINUTES,
            "burn_in_minutes": BURN_IN_MINUTES,
            "hidden_dimensions": 128,
        },
        "metrics": {
            "state_displacement": (
                "L2 between factual and intervened participant representations"
            ),
            "neighbor_change": "NN10 overlap in frozen full_all PCA space",
            "k_nn": K_NN,
        },
        "pca": {
            "space": "full_all",
            "component_count": PCA_COMPONENT_COUNT,
            "scaler_path": str(PCA_ROOT / "full_all_scaler.joblib"),
            "scaler_hash": sha256_file(PCA_ROOT / "full_all_scaler.joblib"),
            "model_path": str(PCA_ROOT / "full_all_pca.joblib"),
            "model_hash": sha256_file(PCA_ROOT / "full_all_pca.joblib"),
            "kept_dimensions_path": str(PCA_ROOT / "kept_dimensions.npy"),
            "kept_dimensions_hash": sha256_file(PCA_ROOT / "kept_dimensions.npy"),
            "feature_order_path": str(PCA_ROOT / "feature_order.json"),
            "feature_order_hash": sha256_file(PCA_ROOT / "feature_order.json"),
        },
        "uncertainty": {
            "unit": "participant",
            "method": "percentile bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "alpha": BOOTSTRAP_ALPHA,
            "seed": RANDOM_SEED,
        },
        "sex_reporting": {
            "reference_category": SEX_REFERENCE_CATEGORY,
            "audit_estimand": "all-participant median retained",
            "presentation_estimand": SEX_PRESENTATION_ESTIMAND,
            "also_report_full_cohort_mean": True,
            "figure_label_rule": (
                "Sex, M vs F reference, with non-reference test n shown"
            ),
        },
        "global_reference": {
            "label": "all static",
            "source": "canonical Step 4 full_all and neutral_all representations",
        },
        "figure": {
            "panel_a": "median state L2 with participant-bootstrap CI",
            "panel_b": "median 1 minus NN10 overlap with participant-bootstrap CI",
            "sort": "descending state displacement with all static at top",
            "variable_color": "#BA2828",
            "all_static_color": "#888888",
            "style": "seaborn whitegrid",
        },
        "prohibitions": [
            "no refitting",
            "no test-derived reference values",
            "no test-derived encoding groups",
            "no test-derived PCA",
            "no dose-response analysis",
            "no forecast-difference analysis",
            "no recoverability regression",
        ],
        "confirmed_validation_hashes": {
            str(path): sha256_file(path)
            for path in [
                validation_summary_path,
                presentation_summary_path,
                sex_sensitivity_path,
                step_b_manifest_path,
                sex_amendment_path,
                mapping_path,
                prevalence_path,
            ]
        },
    }
    plan_path = FROZEN_ROOT / "frozen_test_application_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    plan_hash = sha256_file(plan_path)
    hash_path = FROZEN_ROOT / "frozen_test_application_plan.sha256"
    hash_path.write_text(f"{plan_hash}  {plan_path.name}\n")
    lock = {
        "status": "frozen_before_per_variable_test_replay",
        "frozen_utc": plan["frozen_utc"],
        "plan_path": str(plan_path),
        "plan_sha256": plan_hash,
        "test_application_authorized": True,
        "test_application_started": False,
    }
    lock_path = FROZEN_ROOT / "freeze_manifest.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"Frozen plan: {plan_path}")
    print(f"SHA256: {plan_hash}")


if __name__ == "__main__":
    main()
