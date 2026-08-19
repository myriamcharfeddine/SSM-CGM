#!/usr/bin/env python3
"""Continuous h0-to-ht reorganization analyses."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from scipy.stats import spearmanr


PROJECT_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))
import run_continuous_clinical_encoding_test as encoding
from ssmcgm.analysis import neighbor_clinical_sharing as neighbor_method


OUTPUT_ROOT = PROJECT_ROOT / "outputs/continuous_reorg"
CLINICAL_ROOT = PROJECT_ROOT / "outputs/continuous_clinical"
TARGETS_PATH = CLINICAL_ROOT / "clinical_targets.parquet"
RIDGE_RESULTS_PATH = CLINICAL_ROOT / "ridge_probe_results.csv"
PREDICTIONS_PATH = CLINICAL_ROOT / "ridge_probe_test_predictions.parquet"
ENCODING_MANIFEST_PATH = CLINICAL_ROOT / "encoding_test_manifest.json"
STEP2_ROOT = encoding.STEP2_ROOT
STEP3_ROOT = encoding.STEP3_ROOT
STEP4_ROOT = encoding.STEP4_ROOT
VALIDATION_FEATURES_PATH = encoding.VALIDATION_GLYCEMIC_PATH
TEST_FEATURES_PATH = (
    STEP4_ROOT / "test_glycemic_nuisance_features.parquet"
)
PCA_LOADINGS_PATH = STEP3_ROOT / "pca_loadings.parquet"
FINAL_MULTIMODAL_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "final_multimodal_dataset_20260515_184339.parquet"
)

DELTA_RESULTS_PATH = OUTPUT_ROOT / "delta_error_results.csv"
DELTA_PARTICIPANT_PATH = OUTPUT_ROOT / "delta_error_by_participant.parquet"
DELTA_EXTREMES_PATH = OUTPUT_ROOT / "delta_error_extreme_participants.csv"
NEIGHBOR_RESULTS_PATH = (
    OUTPUT_ROOT / "neighborhood_homogeneity_results.csv"
)
NEIGHBOR_PARTICIPANT_PATH = (
    OUTPUT_ROOT / "neighborhood_homogeneity_by_participant.parquet"
)
RANKSHIFT_PATH = OUTPUT_ROOT / "allcohort_rankshift.csv"
MOVERS_PATH = OUTPUT_ROOT / "mild_to_severe_movers.csv"
MOVER_SUMMARY_PATH = (
    OUTPUT_ROOT / "mild_to_severe_mover_glycemic_summary.csv"
)
FIGURE_DELTA_PATH = OUTPUT_ROOT / "fig5_delta_error_forest.png"
FIGURE_NEIGHBOR_PATH = (
    OUTPUT_ROOT / "fig6_neighborhood_homogeneity_bars.png"
)
FIGURE_RANK_PATH = OUTPUT_ROOT / "fig7_allcohort_rankshift_diagonal.png"
FIGURE_MOVERS_PATH = (
    OUTPUT_ROOT / "fig8_mild_to_severe_movers_glycemic.png"
)
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"

RANDOM_SEED = 42
BOOTSTRAP_REPLICATES = 2000
RANDOM_BASELINE_REPEATS = 2000
PRIMARY_K = 10
SENSITIVITY_K = (5, 20)
K_VALUES = (PRIMARY_K, *SENSITIVITY_K)
EXTREME_PARTICIPANTS_PER_DIRECTION = 5
MILD_QUARTILE_THRESHOLD = 0.25
LARGE_SHIFT_QUANTILE = 0.75
STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
SPACES = ["h0", "full_ht", "neutral_ht"]
HT_SPACES = ["full_ht", "neutral_ht"]
SPACE_LABELS = {
    "h0": "h0",
    "full_ht": "Full ht",
    "neutral_ht": "Neutral ht",
}
TARGETS = encoding.TARGETS
TARGET_LABELS = encoding.TARGET_LABELS
NATIVE_UNITS = encoding.NATIVE_UNITS
T2D_GROUPS = {
    "insulin_dependent",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
}
DIAGNOSIS_COLORS = {
    "T2D": STRATUM_COLORS[0],
    "Prediabetes": STRATUM_COLORS[1],
    "Non-diabetic": STRATUM_COLORS[2],
}
ANALYSIS_LABELS = {
    "step_a": "T2D-only primary prediction-error reorganization",
    "step_b": "T2D-only secondary neighborhood clinical homogeneity",
    "step_c": "all-cohort glycemic rank-shift",
}
AGE_CAVEAT = (
    "participants_age is participant age at study visit, NOT age at diabetes "
    "diagnosis. Age is not a target in this analysis."
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


def native_values(target: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if target in {"log_tg_hdl", "log_c_peptide"}:
        return np.exp(values)
    return values


def bootstrap_summary(
    values: np.ndarray,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    indexes = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_REPLICATES, len(values)),
    )
    samples = values[indexes]
    mean_distribution = samples.mean(axis=1)
    median_distribution = np.median(samples, axis=1)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "mean_ci_low": float(np.quantile(mean_distribution, 0.025)),
        "mean_ci_high": float(np.quantile(mean_distribution, 0.975)),
        "median": float(np.median(values)),
        "median_ci_low": float(
            np.quantile(median_distribution, 0.025)
        ),
        "median_ci_high": float(
            np.quantile(median_distribution, 0.975)
        ),
    }


def load_all_representations() -> tuple[dict, dict]:
    h0_validation, h0_validation_audit = encoding.load_h0(
        encoding.VALIDATION_H0_ROOT,
        "validation",
    )
    h0_test, h0_test_audit = encoding.load_h0(
        encoding.TEST_H0_ROOT,
        "test",
    )
    ht_validation = encoding.load_ht(
        encoding.VALIDATION_REPRESENTATIONS_PATH,
        "validation",
    )
    ht_test = encoding.load_ht(
        encoding.TEST_REPRESENTATIONS_PATH,
        "test",
    )
    raw = {
        "validation": {"h0": h0_validation, **ht_validation},
        "test": {"h0": h0_test, **ht_test},
    }
    frozen = encoding.load_frozen_pca()
    output = {"validation": {}, "test": {}}
    for split_name in ("validation", "test"):
        for space in SPACES:
            value_columns = encoding.H_COLS if space == "h0" else encoding.R_COLS
            selected = raw[split_name][space].sort_values(
                "participant_id"
            ).reset_index(drop=True)
            pca_name = encoding.SPACE_TO_PCA[space]
            scores = encoding.project_scores(
                selected,
                value_columns,
                frozen,
                pca_name,
            )
            state = selected[["participant_id", *value_columns]].copy()
            state.columns = ["participant_id", *encoding.STATE_COLS]
            output[split_name][space] = {
                "state": state,
                "scores": scores,
            }
    audit = {
        "validation_h0": h0_validation_audit,
        "test_h0": h0_test_audit,
        "model_forward_pass_run": False,
        "pca_fit_run": False,
    }
    return output, audit


def delta_error_analysis(
    targets: pd.DataFrame,
    predictions: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if set(predictions["space"]) != set(SPACES):
        raise RuntimeError("Prompt 2 prediction spaces changed")
    participant_rows = []
    result_rows = []
    for target in TARGETS:
        target_values = targets[targets["split"] == "test"][
            ["participant_id", target]
        ].copy()
        validation_sd = float(
            targets.loc[
                targets["split"] == "validation", target
            ].std(ddof=1)
        )
        target_predictions = predictions[
            predictions["target"] == target
        ].copy()
        wide = target_predictions.pivot(
            index="participant_id",
            columns="space",
            values="predicted_analysis_scale",
        ).reset_index()
        data = target_values.merge(
            wide,
            on="participant_id",
            validate="one_to_one",
        )
        observed = data[target].to_numpy(float)
        observed_native = native_values(target, observed)
        h0_prediction = data["h0"].to_numpy(float)
        h0_native = native_values(target, h0_prediction)
        h0_error = np.abs(observed - h0_prediction)
        h0_error_native = np.abs(observed_native - h0_native)
        for ht_space in HT_SPACES:
            ht_prediction = data[ht_space].to_numpy(float)
            ht_native = native_values(target, ht_prediction)
            ht_error = np.abs(observed - ht_prediction)
            ht_error_native = np.abs(observed_native - ht_native)
            delta_analysis = ht_error - h0_error
            delta_native = ht_error_native - h0_error_native
            delta_standardized = delta_analysis / validation_sd
            for index, participant_id in enumerate(data["participant_id"]):
                participant_rows.append(
                    {
                        "participant_id": participant_id,
                        "analysis_label": ANALYSIS_LABELS["step_a"],
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "ht_space": ht_space,
                        "ht_space_label": SPACE_LABELS[ht_space],
                        "true_target_analysis": observed[index],
                        "true_target_native": observed_native[index],
                        "prediction_h0_analysis": h0_prediction[index],
                        "prediction_ht_analysis": ht_prediction[index],
                        "absolute_error_h0_analysis": h0_error[index],
                        "absolute_error_ht_analysis": ht_error[index],
                        "delta_error_analysis": delta_analysis[index],
                        "absolute_error_h0_native": h0_error_native[index],
                        "absolute_error_ht_native": ht_error_native[index],
                        "delta_error_native": delta_native[index],
                        "delta_error_standardized": delta_standardized[index],
                        "native_unit": NATIVE_UNITS[target],
                        "interpretation": (
                            "improved"
                            if delta_analysis[index] < 0
                            else "worsened"
                            if delta_analysis[index] > 0
                            else "unchanged"
                        ),
                    }
                )
            for scale, values, unit in (
                ("analysis", delta_analysis, "target analysis scale"),
                ("native", delta_native, NATIVE_UNITS[target]),
                ("validation_sd_standardized", delta_standardized, "SD units"),
            ):
                summary = bootstrap_summary(
                    values,
                    stable_seed(
                        "delta_error",
                        target,
                        ht_space,
                        scale,
                        RANDOM_SEED,
                    ),
                )
                result_rows.append(
                    {
                        "analysis_label": ANALYSIS_LABELS["step_a"],
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "ht_space": ht_space,
                        "ht_space_label": SPACE_LABELS[ht_space],
                        "error_scale": scale,
                        "unit": unit,
                        "validation_target_sd": validation_sd,
                        **summary,
                        "direction_rule": (
                            "negative means ht improved the frozen estimate; "
                            "positive means ht worsened it"
                        ),
                        "bootstrap_unit": "participant",
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "predictions_source": str(PREDICTIONS_PATH),
                        "models_reloaded": False,
                        "inference_rerun": False,
                    }
                )
    participant_frame = pd.DataFrame(participant_rows)
    profiles = test_features[
        [
            "participant_id",
            "study_group",
            "mean_glucose",
            "glucose_cv",
            "tir_70_180",
            "hba1c",
        ]
    ].copy()
    extremes = []
    for (target, ht_space), group in participant_frame.groupby(
        ["target", "ht_space"]
    ):
        ordered = group.sort_values("delta_error_standardized")
        selected = pd.concat(
            [
                ordered.head(EXTREME_PARTICIPANTS_PER_DIRECTION).assign(
                    extreme_direction="largest improvement"
                ),
                ordered.tail(EXTREME_PARTICIPANTS_PER_DIRECTION)
                .sort_values("delta_error_standardized", ascending=False)
                .assign(extreme_direction="largest worsening"),
            ],
            ignore_index=True,
        )
        selected["extreme_rank_within_direction"] = (
            selected.groupby("extreme_direction").cumcount() + 1
        )
        extremes.append(selected.merge(
            profiles,
            on="participant_id",
            how="left",
            validate="many_to_one",
        ))
    return (
        pd.DataFrame(result_rows),
        participant_frame,
        pd.concat(extremes, ignore_index=True),
    )


def neighborhood_homogeneity(
    targets: pd.DataFrame,
    representations: dict,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_test = targets[targets["split"] == "test"].copy()
    participant_ids = sorted(target_test["participant_id"].tolist())
    clinical = (
        target_test[["participant_id", *TARGETS]]
        .merge(
            test_features[
                ["participant_id", "clinical_site", "study_group"]
            ],
            on="participant_id",
            validate="one_to_one",
        )
        .set_index("participant_id")
        .reindex(participant_ids)
        .reset_index()
    )
    sites = clinical["clinical_site"].astype(str).to_numpy()
    participant_rows = []
    summary_rows = []
    for space in SPACES:
        scores = (
            representations["test"][space]["scores"]
            .set_index("participant_id")
            .reindex(participant_ids)
        )
        pc_columns = [
            column for column in scores.columns if column.startswith("hs_pc")
        ]
        distances = neighbor_method.pairwise_euclidean(
            scores[pc_columns].to_numpy(float)
        )
        np.fill_diagonal(distances, np.inf)
        ordering = np.argsort(distances, axis=1)
        for k_neighbors in K_VALUES:
            graph = ordering[:, :k_neighbors]
            for target in TARGETS:
                (
                    focal_rows,
                    _,
                    _,
                    _,
                ) = neighbor_method.participant_metrics_and_weights(
                    values=clinical[target].to_numpy(),
                    variable_type="continuous",
                    sites=sites,
                    neighbors=graph,
                    condition=space,
                    variable=target,
                    k_neighbors=k_neighbors,
                    random_repeats=RANDOM_BASELINE_REPEATS,
                    seed=RANDOM_SEED,
                )
                focal = pd.DataFrame(focal_rows)
                focal["participant_id"] = [
                    participant_ids[index]
                    for index in focal["focal_index"]
                ]
                focal["analysis_label"] = ANALYSIS_LABELS["step_b"]
                focal["space"] = space
                focal["space_label"] = SPACE_LABELS[space]
                focal["k_neighbors"] = k_neighbors
                focal["target"] = target
                focal["target_label"] = TARGET_LABELS[target]
                focal["raw_similarity_gap"] = (
                    focal["random_raw_metric"]
                    - focal["neighbor_raw_metric"]
                )
                participant_rows.extend(
                    focal.drop(columns=["focal_index"]).to_dict("records")
                )
                neighbor_summary = bootstrap_summary(
                    focal["neighbor_raw_metric"].to_numpy(float),
                    stable_seed("neighbor_raw", space, k_neighbors, target),
                )
                random_summary = bootstrap_summary(
                    focal["random_raw_metric"].to_numpy(float),
                    stable_seed("random_raw", space, k_neighbors, target),
                )
                raw_gap_summary = bootstrap_summary(
                    focal["raw_similarity_gap"].to_numpy(float),
                    stable_seed("raw_gap", space, k_neighbors, target),
                )
                standardized_summary = bootstrap_summary(
                    focal["sharing_gain"].to_numpy(float),
                    stable_seed(
                        "standardized_gap",
                        space,
                        k_neighbors,
                        target,
                    ),
                )
                summary_rows.append(
                    {
                        "record_type": "space_summary",
                        "analysis_label": ANALYSIS_LABELS["step_b"],
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "k_neighbors": k_neighbors,
                        "space": space,
                        "space_label": SPACE_LABELS[space],
                        "ht_space": "",
                        "n_participants": len(focal),
                        "neighbor_raw_mean_difference": neighbor_summary["mean"],
                        "neighbor_raw_ci_low": neighbor_summary["mean_ci_low"],
                        "neighbor_raw_ci_high": neighbor_summary["mean_ci_high"],
                        "random_raw_mean_difference": random_summary["mean"],
                        "random_raw_ci_low": random_summary["mean_ci_low"],
                        "random_raw_ci_high": random_summary["mean_ci_high"],
                        "raw_similarity_gap": raw_gap_summary["mean"],
                        "raw_gap_ci_low": raw_gap_summary["mean_ci_low"],
                        "raw_gap_ci_high": raw_gap_summary["mean_ci_high"],
                        "standardized_similarity_gap": standardized_summary["mean"],
                        "standardized_gap_ci_low": standardized_summary["mean_ci_low"],
                        "standardized_gap_ci_high": standardized_summary["mean_ci_high"],
                        "h0_to_ht_gap_change": np.nan,
                        "gap_change_ci_low": np.nan,
                        "gap_change_ci_high": np.nan,
                        "reorganization_direction": "",
                        "random_baseline": "unrestricted non-neighbors",
                        "random_baseline_repeats": RANDOM_BASELINE_REPEATS,
                        "bootstrap_unit": "focal participant",
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    }
                )
    participant_frame = pd.DataFrame(participant_rows)
    for k_neighbors in K_VALUES:
        for target in TARGETS:
            h0 = participant_frame[
                (participant_frame["space"] == "h0")
                & (participant_frame["k_neighbors"] == k_neighbors)
                & (participant_frame["target"] == target)
            ][["participant_id", "sharing_gain", "raw_similarity_gap"]]
            for ht_space in HT_SPACES:
                ht = participant_frame[
                    (participant_frame["space"] == ht_space)
                    & (participant_frame["k_neighbors"] == k_neighbors)
                    & (participant_frame["target"] == target)
                ][["participant_id", "sharing_gain", "raw_similarity_gap"]]
                paired = h0.merge(
                    ht,
                    on="participant_id",
                    suffixes=("_h0", "_ht"),
                    validate="one_to_one",
                )
                standardized_change = (
                    paired["sharing_gain_ht"]
                    - paired["sharing_gain_h0"]
                ).to_numpy(float)
                raw_change = (
                    paired["raw_similarity_gap_ht"]
                    - paired["raw_similarity_gap_h0"]
                ).to_numpy(float)
                standardized_summary = bootstrap_summary(
                    standardized_change,
                    stable_seed(
                        "neighbor_transition",
                        target,
                        k_neighbors,
                        ht_space,
                    ),
                )
                raw_summary = bootstrap_summary(
                    raw_change,
                    stable_seed(
                        "neighbor_transition_raw",
                        target,
                        k_neighbors,
                        ht_space,
                    ),
                )
                if standardized_summary["mean_ci_low"] > 0:
                    direction = "widens, more homogeneous under streaming"
                elif standardized_summary["mean_ci_high"] < 0:
                    direction = "narrows, less homogeneous under streaming"
                else:
                    direction = "no clear widening or narrowing"
                summary_rows.append(
                    {
                        "record_type": "h0_to_ht_transition",
                        "analysis_label": ANALYSIS_LABELS["step_b"],
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "k_neighbors": k_neighbors,
                        "space": ht_space,
                        "space_label": SPACE_LABELS[ht_space],
                        "ht_space": ht_space,
                        "n_participants": len(paired),
                        "neighbor_raw_mean_difference": np.nan,
                        "neighbor_raw_ci_low": np.nan,
                        "neighbor_raw_ci_high": np.nan,
                        "random_raw_mean_difference": np.nan,
                        "random_raw_ci_low": np.nan,
                        "random_raw_ci_high": np.nan,
                        "raw_similarity_gap": raw_summary["mean"],
                        "raw_gap_ci_low": raw_summary["mean_ci_low"],
                        "raw_gap_ci_high": raw_summary["mean_ci_high"],
                        "standardized_similarity_gap": np.nan,
                        "standardized_gap_ci_low": np.nan,
                        "standardized_gap_ci_high": np.nan,
                        "h0_to_ht_gap_change": standardized_summary["mean"],
                        "gap_change_ci_low": standardized_summary["mean_ci_low"],
                        "gap_change_ci_high": standardized_summary["mean_ci_high"],
                        "reorganization_direction": direction,
                        "random_baseline": "unrestricted non-neighbors",
                        "random_baseline_repeats": RANDOM_BASELINE_REPEATS,
                        "bootstrap_unit": "paired focal participant",
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    }
                )
    return pd.DataFrame(summary_rows), participant_frame


def diagnosis_status(study_group: pd.Series) -> pd.Series:
    return np.where(
        study_group.isin(T2D_GROUPS),
        "T2D",
        np.where(
            study_group.eq("pre_diabetes_lifestyle_controlled"),
            "Prediabetes",
            "Non-diabetic",
        ),
    )


def local_neighbor_mean(
    scores: pd.DataFrame,
    participant_ids: list[str],
    mean_glucose: np.ndarray,
) -> np.ndarray:
    selected = scores.set_index("participant_id").reindex(participant_ids)
    pc_columns = [
        column for column in selected.columns if column.startswith("hs_pc")
    ]
    distances = neighbor_method.pairwise_euclidean(
        selected[pc_columns].to_numpy(float)
    )
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :PRIMARY_K]
    return mean_glucose[neighbors].mean(axis=1)


def compute_cgm_metrics(participant_ids: list[str]) -> pd.DataFrame:
    string_ids = [str(participant_id) for participant_id in participant_ids]
    frame = pd.read_parquet(
        FINAL_MULTIMODAL_PATH,
        columns=["participant_id", "cgm_glucose_mean", "cgm_count"],
        filters=[("participant_id", "in", string_ids)],
    )
    frame["participant_id"] = frame["participant_id"].astype(str)
    rows = []
    for participant_id, group in frame.groupby("participant_id"):
        valid = (
            pd.to_numeric(group["cgm_count"], errors="coerce").fillna(0) > 0
        ) & pd.to_numeric(
            group["cgm_glucose_mean"], errors="coerce"
        ).notna()
        glucose = pd.to_numeric(
            group.loc[valid, "cgm_glucose_mean"],
            errors="coerce",
        ).to_numpy(float)
        if not len(glucose):
            raise RuntimeError(
                f"No valid CGM rows for participant {participant_id}"
            )
        rows.append(
            {
                "participant_id": participant_id,
                "valid_cgm_rows": len(glucose),
                "tir_70_180_percent": 100 * np.mean(
                    (glucose >= 70) & (glucose <= 180)
                ),
                "time_above_140_percent": 100 * np.mean(glucose > 140),
                "time_above_200_percent": 100 * np.mean(glucose > 200),
            }
        )
    result = pd.DataFrame(rows)
    if set(result["participant_id"]) != set(participant_ids):
        raise RuntimeError("CGM metric participant coverage mismatch")
    return result


def rank_shift_analysis(
    representations: dict,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    validation_orientation = {}
    for space in ("h0", "full_ht"):
        validation_space = (
            representations["validation"][space]["scores"]
            [["participant_id", "hs_pc1"]]
            .merge(
                validation_features[["participant_id", "mean_glucose"]],
                on="participant_id",
                validate="one_to_one",
            )
        )
        validation_rho = float(
            spearmanr(
                validation_space["hs_pc1"],
                validation_space["mean_glucose"],
            ).statistic
        )
        validation_orientation[space] = {
            "rho": validation_rho,
            "multiplier": 1.0 if validation_rho >= 0 else -1.0,
        }
    participant_ids = sorted(test_features["participant_id"].tolist())
    clinical = (
        test_features[
            [
                "participant_id",
                "study_group",
                "mean_glucose",
                "glucose_cv",
                "tir_70_180",
                "hba1c",
            ]
        ]
        .set_index("participant_id")
        .reindex(participant_ids)
        .reset_index()
    )
    clinical["diagnosis_status"] = diagnosis_status(
        clinical["study_group"]
    )
    h0_scores = (
        representations["test"]["h0"]["scores"]
        .set_index("participant_id")
        .reindex(participant_ids)
        .reset_index()
    )
    ht_scores = (
        representations["test"]["full_ht"]["scores"]
        .set_index("participant_id")
        .reindex(participant_ids)
        .reset_index()
    )
    rank = clinical.copy()
    rank["hidden_pc1_h0"] = (
        validation_orientation["h0"]["multiplier"]
        * h0_scores["hs_pc1"]
    )
    rank["hidden_pc1_ht"] = (
        validation_orientation["full_ht"]["multiplier"]
        * ht_scores["hs_pc1"]
    )
    rank["pc1_rank_h0"] = rank["hidden_pc1_h0"].rank(
        method="average",
        pct=True,
    )
    rank["pc1_rank_ht"] = rank["hidden_pc1_ht"].rank(
        method="average",
        pct=True,
    )
    rank["pc1_shift"] = rank["pc1_rank_ht"] - rank["pc1_rank_h0"]
    mean_glucose = rank["mean_glucose"].to_numpy(float)
    rank["local_mean_glucose_h0"] = local_neighbor_mean(
        representations["test"]["h0"]["scores"],
        participant_ids,
        mean_glucose,
    )
    rank["local_mean_glucose_ht"] = local_neighbor_mean(
        representations["test"]["full_ht"]["scores"],
        participant_ids,
        mean_glucose,
    )
    rank["mean_glucose_rank_h0"] = rank[
        "local_mean_glucose_h0"
    ].rank(method="average", pct=True)
    rank["mean_glucose_rank_ht"] = rank[
        "local_mean_glucose_ht"
    ].rank(method="average", pct=True)
    rank["mean_glucose_shift"] = (
        rank["mean_glucose_rank_ht"] - rank["mean_glucose_rank_h0"]
    )
    thresholds = {}
    for method in ("mean_glucose", "pc1"):
        h0_column = f"{method}_rank_h0"
        shift_column = f"{method}_shift"
        mild = rank[h0_column] <= MILD_QUARTILE_THRESHOLD
        cutoff = max(
            0.0,
            float(
                rank.loc[mild, shift_column].quantile(
                    LARGE_SHIFT_QUANTILE
                )
            ),
        )
        flag_column = f"is_mild_to_severe_mover_{method}"
        rank[flag_column] = (
            mild
            & (rank[shift_column] >= cutoff)
            & (rank[shift_column] > 0)
        )
        thresholds[method] = {
            "mild_rank_threshold": MILD_QUARTILE_THRESHOLD,
            "large_positive_shift_cutoff": cutoff,
            "shift_quantile_within_mild_group": LARGE_SHIFT_QUANTILE,
            "mover_count": int(rank[flag_column].sum()),
        }
    rank["is_consensus_mover"] = (
        rank["is_mild_to_severe_mover_mean_glucose"]
        & rank["is_mild_to_severe_mover_pc1"]
    )
    rank["split"] = "test"
    rank["analysis_label"] = ANALYSIS_LABELS["step_c"]
    cgm = compute_cgm_metrics(participant_ids)
    rank = rank.merge(
        cgm,
        on="participant_id",
        validate="one_to_one",
    )
    movers = rank[
        rank["is_mild_to_severe_mover_mean_glucose"]
    ].copy()
    movers["not_diagnosed_diabetes_or_prediabetes"] = (
        movers["diagnosis_status"] == "Non-diabetic"
    )
    if movers.empty:
        raise RuntimeError("Primary rank-shift mover subgroup is empty")
    summary_rows = []
    metric_columns = {
        "TIR 70 to 180": "tir_70_180_percent",
        "Time above 140": "time_above_140_percent",
        "Time above 200": "time_above_200_percent",
    }
    for diagnosis in [
        *DIAGNOSIS_COLORS.keys(),
        "All movers",
    ]:
        selected = (
            movers
            if diagnosis == "All movers"
            else movers[movers["diagnosis_status"] == diagnosis]
        )
        for metric_label, metric_column in metric_columns.items():
            if selected.empty:
                continue
            summary = bootstrap_summary(
                selected[metric_column].to_numpy(float),
                stable_seed("mover_metric", diagnosis, metric_column),
            )
            summary_rows.append(
                {
                    "analysis_label": ANALYSIS_LABELS["step_c"],
                    "subgroup_definition": (
                        "mildest h0 quartile and top-quartile positive "
                        "neighbor-smoothed mean-glucose rank shift within "
                        "that mild group"
                    ),
                    "diagnosis_status": diagnosis,
                    "metric": metric_column,
                    "metric_label": metric_label,
                    "unit": "percent of valid CGM rows",
                    **summary,
                    "bootstrap_unit": "participant",
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    headline_values = movers[
        "not_diagnosed_diabetes_or_prediabetes"
    ].astype(float).to_numpy()
    headline = bootstrap_summary(
        headline_values,
        stable_seed("headline_fraction"),
    )
    headline.update(
        {
            "fraction": headline["mean"],
            "percent": 100 * headline["mean"],
            "percent_ci_low": 100 * headline["mean_ci_low"],
            "percent_ci_high": 100 * headline["mean_ci_high"],
            "numerator": int(headline_values.sum()),
            "denominator": len(headline_values),
        }
    )
    metadata = {
        "pc1_definition": (
            "Frozen full-cohort hidden-state PC1, not Clinical PC1"
        ),
        "pc1_orientation_source": (
            "space-specific validation Spearman correlation with mean "
            "glucose; signs frozen before test ranking"
        ),
        "validation_pc1_mean_glucose_rho_h0": (
            validation_orientation["h0"]["rho"]
        ),
        "validation_pc1_mean_glucose_rho_full_ht": (
            validation_orientation["full_ht"]["rho"]
        ),
        "pc1_orientation_multiplier_h0": (
            validation_orientation["h0"]["multiplier"]
        ),
        "pc1_orientation_multiplier_full_ht": (
            validation_orientation["full_ht"]["multiplier"]
        ),
        "mean_glucose_rank_definition": (
            "rank of k=10 hidden-neighbor-smoothed participant mean glucose "
            "within each representation space"
        ),
        "thresholds": thresholds,
        "headline_not_diagnosed_diabetes_or_prediabetes": headline,
    }
    return rank, movers, pd.DataFrame(summary_rows), metadata


def figure_delta_error(results: pd.DataFrame) -> None:
    plot = results[
        results["error_scale"] == "validation_sd_standardized"
    ].copy()
    y_base = np.arange(len(TARGETS))[::-1]
    offsets = {"full_ht": 0.12, "neutral_ht": -0.12}
    figure, axis = plt.subplots(figsize=(11.5, 6.8))
    for space_index, ht_space in enumerate(HT_SPACES):
        selected = plot.set_index(["target", "ht_space"])
        for target_index, target in enumerate(TARGETS):
            row = selected.loc[(target, ht_space)]
            y = y_base[target_index] + offsets[ht_space]
            axis.errorbar(
                row["mean"],
                y,
                xerr=[
                    [row["mean"] - row["mean_ci_low"]],
                    [row["mean_ci_high"] - row["mean"]],
                ],
                fmt="o",
                color=STRATUM_COLORS[space_index],
                capsize=3,
                linewidth=1.5,
                label=(
                    SPACE_LABELS[ht_space]
                    if target_index == 0
                    else None
                ),
            )
    axis.axvline(0, color="#222222", linewidth=1)
    axis.set_yticks(y_base)
    axis.set_yticklabels([TARGET_LABELS[target] for target in TARGETS])
    axis.set_xlabel(
        "Mean change in absolute prediction error (validation SD units)\n"
        "Negative means ht improved the frozen estimate"
    )
    axis.set_title(
        "T2D-only primary: frozen-probe prediction-error reorganization"
    )
    axis.legend(frameon=True)
    axis.grid(axis="y", visible=False)
    figure.tight_layout()
    figure.savefig(FIGURE_DELTA_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def figure_neighbor(results: pd.DataFrame) -> None:
    plot = results[
        (results["record_type"] == "space_summary")
        & (results["k_neighbors"] == PRIMARY_K)
    ].copy()
    figure, axes = plt.subplots(1, len(TARGETS), figsize=(22, 5.8))
    for axis, target in zip(axes, TARGETS):
        selected = (
            plot[plot["target"] == target]
            .set_index("space")
            .reindex(SPACES)
        )
        x = np.arange(len(SPACES))
        y = selected["standardized_similarity_gap"].to_numpy(float)
        lower = y - selected["standardized_gap_ci_low"].to_numpy(float)
        upper = selected["standardized_gap_ci_high"].to_numpy(float) - y
        axis.bar(
            x,
            y,
            color=STRATUM_COLORS[: len(SPACES)],
            alpha=0.88,
            yerr=np.vstack([lower, upper]),
            capsize=3,
        )
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [SPACE_LABELS[space] for space in SPACES],
            rotation=18,
        )
        axis.set_title(TARGET_LABELS[target])
        axis.set_ylabel("Standardized neighbor similarity gap")
    figure.suptitle(
        "T2D-only secondary: clinical homogeneity among k=10 hidden-state neighbors",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.015,
        "Positive values mean neighbors are more clinically similar than random non-neighbors.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.94))
    figure.savefig(FIGURE_NEIGHBOR_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def figure_rankshift(rank: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharex=True, sharey=True)
    panels = [
        (
            "mean_glucose_rank_h0",
            "mean_glucose_rank_ht",
            "is_mild_to_severe_mover_mean_glucose",
            "Neighbor-smoothed mean-glucose rank",
        ),
        (
            "pc1_rank_h0",
            "pc1_rank_ht",
            "is_mild_to_severe_mover_pc1",
            "Frozen hidden-state PC1 rank",
        ),
    ]
    for axis, (x_column, y_column, flag_column, title) in zip(axes, panels):
        for diagnosis, color in DIAGNOSIS_COLORS.items():
            selected = rank[rank["diagnosis_status"] == diagnosis]
            axis.scatter(
                selected[x_column],
                selected[y_column],
                color=color,
                s=38,
                alpha=0.76,
                label=f"{diagnosis} (n={len(selected)})",
            )
        movers = rank[rank[flag_column]]
        axis.scatter(
            movers[x_column],
            movers[y_column],
            facecolor="none",
            edgecolor="#111111",
            linewidth=1.2,
            s=88,
            label=f"Mild-to-severe movers (n={len(movers)})",
        )
        axis.plot([0, 1], [0, 1], color="#555555", linestyle="--")
        axis.axvline(MILD_QUARTILE_THRESHOLD, color="#AAAAAA", linestyle=":")
        axis.set_xlim(0, 1.02)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("h0 percentile rank")
        axis.set_ylabel("Full ht percentile rank")
        axis.set_title(title)
    axes[0].legend(frameon=True, fontsize=8)
    figure.suptitle(
        "All-cohort glycemic rank-shift, kept separate from fixed clinical targets",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(FIGURE_RANK_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def figure_movers(summary: pd.DataFrame) -> None:
    diagnosis_order = [
        diagnosis
        for diagnosis in DIAGNOSIS_COLORS
        if diagnosis in set(summary["diagnosis_status"])
    ]
    metric_order = [
        "tir_70_180_percent",
        "time_above_140_percent",
        "time_above_200_percent",
    ]
    metric_titles = {
        "tir_70_180_percent": "Time in range 70 to 180",
        "time_above_140_percent": "Time above 140 mg/dL",
        "time_above_200_percent": "Time above 200 mg/dL",
    }
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 5.8), sharey=True)
    for axis, metric in zip(axes, metric_order):
        selected = (
            summary[
                (summary["metric"] == metric)
                & summary["diagnosis_status"].isin(diagnosis_order)
            ]
            .set_index("diagnosis_status")
            .reindex(diagnosis_order)
        )
        x = np.arange(len(diagnosis_order))
        y = selected["mean"].to_numpy(float)
        axis.bar(
            x,
            y,
            color=[DIAGNOSIS_COLORS[item] for item in diagnosis_order],
            alpha=0.88,
            yerr=np.vstack(
                [
                    y - selected["mean_ci_low"].to_numpy(float),
                    selected["mean_ci_high"].to_numpy(float) - y,
                ]
            ),
            capsize=3,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(diagnosis_order, rotation=15)
        axis.set_ylim(0, 105)
        axis.set_ylabel("Percent of valid CGM rows")
        axis.set_title(metric_titles[metric])
        for index, n_value in enumerate(selected["n"]):
            axis.text(index, 2, f"n={int(n_value)}", ha="center", fontsize=8)
    figure.suptitle(
        "All-cohort glycemic rank-shift: mild-h0 to severe-ht movers",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(FIGURE_MOVERS_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    np.random.seed(RANDOM_SEED)
    input_paths = {
        "clinical_targets": TARGETS_PATH,
        "ridge_probe_results": RIDGE_RESULTS_PATH,
        "frozen_test_predictions": PREDICTIONS_PATH,
        "encoding_manifest": ENCODING_MANIFEST_PATH,
        "validation_representations": encoding.VALIDATION_REPRESENTATIONS_PATH,
        "test_representations": encoding.TEST_REPRESENTATIONS_PATH,
        "pca_loadings": PCA_LOADINGS_PATH,
        "validation_features": VALIDATION_FEATURES_PATH,
        "test_features": TEST_FEATURES_PATH,
        "final_multimodal": FINAL_MULTIMODAL_PATH,
        "existing_neighbor_method": (
            PROJECT_ROOT / "ssmcgm/analysis/neighbor_clinical_sharing.py"
        ),
    }
    missing = [
        str(path) for path in input_paths.values() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing reorganization inputs: {missing}")

    targets = pd.read_parquet(TARGETS_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    ridge_results = pd.read_csv(RIDGE_RESULTS_PATH)
    validation_features = pd.read_parquet(VALIDATION_FEATURES_PATH)
    test_features = pd.read_parquet(TEST_FEATURES_PATH)
    for frame in (
        targets,
        predictions,
        validation_features,
        test_features,
    ):
        frame["participant_id"] = frame["participant_id"].astype(str)
    if targets.groupby("split").size().to_dict() != {
        "validation": 91,
        "test": 83,
    }:
        raise RuntimeError("T2D target cohort changed")
    if len(predictions) != 83 * len(TARGETS) * len(SPACES):
        raise RuntimeError("Frozen Prompt 2 prediction count changed")
    if not ridge_results["validation_tuned_test_frozen"].all():
        raise RuntimeError("Prompt 2 frozen model gate failed")

    representations, representation_audit = load_all_representations()
    delta_results, delta_participant, delta_extremes = (
        delta_error_analysis(
            targets,
            predictions,
            test_features,
        )
    )
    delta_results.to_csv(DELTA_RESULTS_PATH, index=False)
    delta_participant.to_parquet(
        DELTA_PARTICIPANT_PATH,
        index=False,
        compression="zstd",
    )
    delta_extremes.to_csv(DELTA_EXTREMES_PATH, index=False)

    neighbor_results, neighbor_participant = neighborhood_homogeneity(
        targets,
        representations,
        test_features,
    )
    neighbor_results.to_csv(NEIGHBOR_RESULTS_PATH, index=False)
    neighbor_participant.to_parquet(
        NEIGHBOR_PARTICIPANT_PATH,
        index=False,
        compression="zstd",
    )

    rank, movers, mover_summary, rank_metadata = rank_shift_analysis(
        representations,
        validation_features,
        test_features,
    )
    rank.to_csv(RANKSHIFT_PATH, index=False)
    movers.to_csv(MOVERS_PATH, index=False)
    mover_summary.to_csv(MOVER_SUMMARY_PATH, index=False)

    figure_delta_error(delta_results)
    figure_neighbor(neighbor_results)
    figure_rankshift(rank)
    figure_movers(mover_summary)

    required_output_paths = {
        "delta_error_results": DELTA_RESULTS_PATH,
        "delta_error_by_participant": DELTA_PARTICIPANT_PATH,
        "delta_error_extremes": DELTA_EXTREMES_PATH,
        "neighborhood_homogeneity_results": NEIGHBOR_RESULTS_PATH,
        "neighborhood_homogeneity_by_participant": NEIGHBOR_PARTICIPANT_PATH,
        "allcohort_rankshift": RANKSHIFT_PATH,
        "mild_to_severe_movers": MOVERS_PATH,
        "mover_glycemic_summary": MOVER_SUMMARY_PATH,
        "figure_delta_error": FIGURE_DELTA_PATH,
        "figure_neighborhood": FIGURE_NEIGHBOR_PATH,
        "figure_rankshift": FIGURE_RANK_PATH,
        "figure_movers": FIGURE_MOVERS_PATH,
    }
    target_schema = pq.read_schema(TARGETS_PATH)
    age_metadata = {
        key.decode(): value.decode()
        for key, value in (
            target_schema.field("participants_age").metadata or {}
        ).items()
    }
    qc = {
        "t2d_validation_targets": int(
            (targets["split"] == "validation").sum()
        ),
        "t2d_test_targets": int((targets["split"] == "test").sum()),
        "frozen_prediction_rows": len(predictions),
        "delta_summary_rows": len(delta_results),
        "delta_participant_rows": len(delta_participant),
        "delta_extreme_rows": len(delta_extremes),
        "neighbor_summary_rows": len(neighbor_results),
        "neighbor_participant_rows": len(neighbor_participant),
        "allcohort_test_participants": len(rank),
        "primary_mover_count": len(movers),
        "all_required_outputs_present": all(
            path.exists() for path in required_output_paths.values()
        ),
        "no_model_reload_for_step_a": True,
        "no_inference_rerun_for_step_a": True,
        "no_forecasting_replay": True,
        "neighbor_method_reused": True,
        "step_c_has_no_fixed_external_clinical_target": True,
        "pc1_is_hidden_state_pc1_not_clinical_pc1": True,
        "age_field_metadata_verified": age_metadata,
    }
    if not all(
        [
            qc["t2d_validation_targets"] == 91,
            qc["t2d_test_targets"] == 83,
            qc["frozen_prediction_rows"] == 1245,
            qc["delta_summary_rows"] == len(TARGETS) * 2 * 3,
            qc["delta_participant_rows"] == 83 * len(TARGETS) * 2,
            qc["delta_extreme_rows"]
            == len(TARGETS) * 2 * 2 * EXTREME_PARTICIPANTS_PER_DIRECTION,
            qc["allcohort_test_participants"] == 221,
            qc["primary_mover_count"] > 0,
            qc["all_required_outputs_present"],
            age_metadata.get("is_age_at_diabetes_diagnosis") == "false",
        ]
    ):
        raise RuntimeError(f"Continuous reorganization QC failed: {qc}")

    manifest = {
        "analysis": "Continuous reorganization from h0 to ht",
        "status": "QC_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(OUTPUT_ROOT),
        "analysis_separation": {
            "step_a": ANALYSIS_LABELS["step_a"],
            "step_b": ANALYSIS_LABELS["step_b"],
            "step_c": ANALYSIS_LABELS["step_c"],
            "step_c_separation_rule": (
                "Step C uses all 221 test participants and no fixed external "
                "clinical target. It is not pooled with T2D Steps A or B."
            ),
        },
        "step_a": {
            "prediction_source": str(PREDICTIONS_PATH),
            "models_reloaded": False,
            "inference_rerun": False,
            "primary_error_definition": (
                "absolute frozen-probe error in each target analysis scale"
            ),
            "delta_definition": "absolute error ht minus absolute error h0",
            "direction": (
                "negative is improvement; positive is worsening"
            ),
            "figure_scale": (
                "analysis-scale error divided by validation target SD, "
                "avoiding mixed target units on one axis"
            ),
            "native_error_rules": {
                "bmi": "kg/m2",
                "log_tg_hdl": "exponentiated to TG/HDL ratio",
                "log_c_peptide": "exponentiated to C-peptide ng/mL",
                "clinical_pc1": "clinical PC score",
                "clinical_pc2": "clinical PC score",
            },
            "extreme_participants_per_direction": (
                EXTREME_PARTICIPANTS_PER_DIRECTION
            ),
        },
        "step_b": {
            "methodology_source": str(
                input_paths["existing_neighbor_method"]
            ),
            "reused_functions": [
                "pairwise_euclidean",
                "participant_metrics_and_weights",
                "cyclic_sample_indices",
                "stable_seed",
            ],
            "cohort": "83 T2D test complete cases",
            "spaces": SPACES,
            "primary_k": PRIMARY_K,
            "sensitivity_k": list(SENSITIVITY_K),
            "random_baseline": (
                "2,000 deterministic without-replacement random "
                "non-neighbor sets per focal participant"
            ),
            "gap_definition": (
                "random mean absolute difference minus neighbor mean "
                "absolute difference"
            ),
            "positive_gap": "neighbors are more clinically homogeneous",
            "transition": (
                "ht standardized gap minus h0 standardized gap"
            ),
        },
        "step_c": {
            "cohort": "all 221 test participants",
            "fixed_external_clinical_target_used": False,
            "primary_rank": (
                "k=10 neighbor-smoothed mean-glucose percentile rank "
                "within h0 and full ht spaces"
            ),
            "secondary_rank": rank_metadata["pc1_definition"],
            "pc1_clarification": (
                "PC1 is the hidden-state's own frozen full-cohort PC1, "
                "not the T2D clinical insulin-resistance composite"
            ),
            "pc1_orientation": {
                key: value
                for key, value in rank_metadata.items()
                if key.startswith("pc1_")
                or key.startswith("validation_pc1")
            },
            "mover_thresholds": rank_metadata["thresholds"],
            "primary_mover_subgroup": (
                "mildest h0 quartile and top-quartile positive shift "
                "within that mild group, using neighbor-smoothed "
                "mean-glucose rank"
            ),
            "cgm_validity_rule": (
                "cgm_count > 0 and cgm_glucose_mean nonmissing"
            ),
            "glycemic_metrics": [
                "TIR 70 to 180 percent",
                "time above 140 mg/dL percent",
                "time above 200 mg/dL percent",
            ],
            "headline": rank_metadata[
                "headline_not_diagnosed_diabetes_or_prediabetes"
            ],
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "interval": "participant percentile 95%",
            "seed": RANDOM_SEED,
        },
        "age_caveat": AGE_CAVEAT,
        "representation_audit": representation_audit,
        "qc": qc,
        "input_paths": {
            key: str(path) for key, path in input_paths.items()
        },
        "input_hashes": {
            key: sha256_file(path) for key, path in input_paths.items()
        },
        "output_paths": {
            key: str(path) for key, path in required_output_paths.items()
        },
        "output_hashes": {
            key: sha256_file(path)
            for key, path in required_output_paths.items()
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
    delta_primary = delta_results[
        delta_results["error_scale"] == "validation_sd_standardized"
    ][
        [
            "target",
            "ht_space",
            "mean",
            "mean_ci_low",
            "mean_ci_high",
        ]
    ]
    neighbor_transitions = neighbor_results[
        (neighbor_results["record_type"] == "h0_to_ht_transition")
        & (neighbor_results["k_neighbors"] == PRIMARY_K)
    ][
        [
            "target",
            "ht_space",
            "h0_to_ht_gap_change",
            "gap_change_ci_low",
            "gap_change_ci_high",
            "reorganization_direction",
        ]
    ]
    print(
        json.dumps(
            {
                "output_directory": str(OUTPUT_ROOT),
                "status": manifest["status"],
                "delta_error_standardized": delta_primary.to_dict(
                    orient="records"
                ),
                "neighbor_k10_transitions": neighbor_transitions.to_dict(
                    orient="records"
                ),
                "rankshift": {
                    "primary_movers": len(movers),
                    "headline": manifest["step_c"]["headline"],
                    "thresholds": rank_metadata["thresholds"],
                },
                "manifest": str(MANIFEST_PATH),
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
