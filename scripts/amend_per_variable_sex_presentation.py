#!/usr/bin/env python3
"""Add the validation sex-effect presentation amendment without model replay."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
RUN_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/per_variable_static_conditioning_audit/"
    "20260727T214535Z"
)
VALIDATION_ROOT = RUN_ROOT / "validation_interventions"
PARTICIPANT_EFFECT_PATH = (
    VALIDATION_ROOT / "validation_per_variable_static_effects_by_participant.csv"
)
ORIGINAL_SUMMARY_PATH = (
    VALIDATION_ROOT / "validation_per_variable_static_effects.csv"
)
STATIC_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "participant_static_features.parquet"
)
SPLIT_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/"
    "experiment_c_split_adapt6h_seed42/split_participants.csv"
)
REFERENCE_PROFILE_PATH = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/step1_static_neutralization/"
    "20260724T223612Z/static_reference_profile.csv"
)

SEX_VARIABLE = "sex"
SEX_COLUMN = "demo_sex_at_birth"
VALIDATION_SPLIT = "val"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_ALPHA = 0.05
RANDOM_SEED = 42
EXPECTED_VALIDATION_N = 239
EXPECTED_NONREFERENCE_N = 100
PRESENTATION_ESTIMAND = "nonreference_participants_median"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    seed: int,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
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
    lower = BOOTSTRAP_ALPHA / 2
    return tuple(np.quantile(estimates, [lower, 1 - lower]))


def summary_row(
    frame: pd.DataFrame,
    population: str,
    statistic_name: str,
    statistic: Callable[[np.ndarray], float],
    seed: int,
    reference_category: str,
) -> dict:
    state = frame["state_l2"].to_numpy(dtype=float)
    overlap = frame["nn10_overlap"].to_numpy(dtype=float)
    state_low, state_high = bootstrap_ci(state, statistic, seed)
    overlap_low, overlap_high = bootstrap_ci(overlap, statistic, seed + 1)
    return {
        "variable": SEX_VARIABLE,
        "population": population,
        "statistic": statistic_name,
        "n_participants": len(frame),
        "reference_category": reference_category,
        "observed_categories": ",".join(
            sorted(frame[SEX_COLUMN].dropna().astype(str).unique())
        ),
        "state_l2": statistic(state),
        "state_l2_ci_low": state_low,
        "state_l2_ci_high": state_high,
        "nn10_overlap": statistic(overlap),
        "nn10_ci_low": overlap_low,
        "nn10_ci_high": overlap_high,
    }


def main() -> None:
    effects = pd.read_csv(
        PARTICIPANT_EFFECT_PATH,
        dtype={"participant_id": str},
    )
    sex_effects = effects[effects["variable"].eq(SEX_VARIABLE)].copy()
    if len(sex_effects) != EXPECTED_VALIDATION_N:
        raise RuntimeError("Sex effect table does not contain 239 validation participants")

    split = pd.read_csv(SPLIT_PATH, dtype=str)
    validation_ids = set(
        split.loc[
            split["split"].eq(VALIDATION_SPLIT),
            "participant_id",
        ].astype(str)
    )
    static = pd.read_parquet(
        STATIC_PATH,
        columns=["participant_id", SEX_COLUMN],
    )
    static["participant_id"] = static["participant_id"].astype(str)
    static = static[static["participant_id"].isin(validation_ids)].drop_duplicates(
        "participant_id"
    )
    sex_effects = sex_effects.merge(
        static,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    if sex_effects[SEX_COLUMN].isna().any():
        raise RuntimeError("A validation participant is missing sex-at-birth")

    reference_profile = pd.read_csv(REFERENCE_PROFILE_PATH)
    reference_category = str(
        reference_profile.loc[
            reference_profile["feature_name"].eq(SEX_COLUMN),
            "raw_reference_value",
        ].iloc[0]
    )
    nonreference = sex_effects[
        ~sex_effects[SEX_COLUMN].astype(str).eq(reference_category)
    ].copy()
    if len(nonreference) != EXPECTED_NONREFERENCE_N:
        raise RuntimeError(
            f"Expected {EXPECTED_NONREFERENCE_N} non-reference participants, "
            f"found {len(nonreference)}"
        )

    sensitivity_rows = [
        summary_row(
            sex_effects,
            "all_validation_participants",
            "median",
            np.median,
            RANDOM_SEED,
            reference_category,
        ),
        summary_row(
            sex_effects,
            "all_validation_participants",
            "mean",
            np.mean,
            RANDOM_SEED + 100,
            reference_category,
        ),
        summary_row(
            nonreference,
            "participants_differing_from_reference",
            "median",
            np.median,
            RANDOM_SEED + 200,
            reference_category,
        ),
        summary_row(
            nonreference,
            "participants_differing_from_reference",
            "mean",
            np.mean,
            RANDOM_SEED + 300,
            reference_category,
        ),
    ]
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_path = VALIDATION_ROOT / "validation_sex_effect_sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False)

    original = pd.read_csv(ORIGINAL_SUMMARY_PATH)
    conditional = sensitivity[
        sensitivity["population"].eq("participants_differing_from_reference")
        & sensitivity["statistic"].eq("median")
    ].iloc[0]
    presentation = original.copy()
    sex_index = presentation.index[presentation["variable"].eq(SEX_VARIABLE)]
    if len(sex_index) != 1:
        raise RuntimeError("Original summary does not have exactly one sex row")
    row_index = sex_index[0]
    presentation.loc[row_index, "n_participants"] = int(
        conditional["n_participants"]
    )
    presentation.loc[row_index, "median_state_l2"] = conditional["state_l2"]
    presentation.loc[row_index, "l2_ci_low"] = conditional["state_l2_ci_low"]
    presentation.loc[row_index, "l2_ci_high"] = conditional["state_l2_ci_high"]
    presentation.loc[row_index, "median_nn10_overlap"] = conditional[
        "nn10_overlap"
    ]
    presentation.loc[row_index, "nn10_ci_low"] = conditional["nn10_ci_low"]
    presentation.loc[row_index, "nn10_ci_high"] = conditional["nn10_ci_high"]
    presentation.loc[row_index, "prevalence_note"] = (
        f"conditional on differing from reference {reference_category}; "
        f"n={int(conditional['n_participants'])}"
    )
    presentation_path = (
        VALIDATION_ROOT
        / "validation_per_variable_static_effects_presentation.csv"
    )
    presentation.to_csv(presentation_path, index=False)

    amendment = {
        "stage": "step_b_sex_presentation_amendment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_forward_pass_executed": False,
        "test_data_accessed": False,
        "reason": (
            "The all-participant median is zero because most participants equal "
            "the categorical reference and therefore have zero displacement by "
            "construction."
        ),
        "reference_category": reference_category,
        "all_validation_n": len(sex_effects),
        "reference_matching_n": int(
            sex_effects[SEX_COLUMN].astype(str).eq(reference_category).sum()
        ),
        "nonreference_n": len(nonreference),
        "presentation_estimand": PRESENTATION_ESTIMAND,
        "presentation_label": (
            f"Sex, participants differing from reference "
            f"{reference_category}, n={len(nonreference)}"
        ),
        "audit_result_retained": str(ORIGINAL_SUMMARY_PATH),
        "source_hashes_sha256": {
            str(path): sha256_file(path)
            for path in [
                PARTICIPANT_EFFECT_PATH,
                ORIGINAL_SUMMARY_PATH,
                STATIC_PATH,
                SPLIT_PATH,
                REFERENCE_PROFILE_PATH,
            ]
        },
        "output_hashes_sha256": {
            str(path): sha256_file(path)
            for path in [sensitivity_path, presentation_path]
        },
    }
    amendment_path = VALIDATION_ROOT / "step_b_sex_presentation_amendment.json"
    amendment_path.write_text(
        json.dumps(amendment, indent=2, sort_keys=True) + "\n"
    )

    print(sensitivity.to_string(index=False))
    print()
    print(presentation.to_string(index=False))
    print(f"Saved presentation-ready results to {presentation_path}")


if __name__ == "__main__":
    main()
