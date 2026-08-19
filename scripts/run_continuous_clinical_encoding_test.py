#!/usr/bin/env python3
"""Test continuous T2D clinical target encoding in frozen hidden states."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
import run_beyond_glucose_dynamics_power as existing_power
import run_hidden_state_clinical_probes as existing_probe


OUTPUT_ROOT = PROJECT_ROOT / "outputs/continuous_clinical"
MODEL_ROOT = OUTPUT_ROOT / "encoding_frozen_models"
TARGETS_PATH = OUTPUT_ROOT / "clinical_targets.parquet"
TARGETS_MANIFEST_PATH = OUTPUT_ROOT / "clinical_targets_manifest.json"
STEP2_ROOT = (
    PROJECT_ROOT
    / "outputs/hidden_state_phenotype/step2_validation_export/20260724T231513Z"
)
STEP3_ROOT = (
    PROJECT_ROOT
    / "outputs/hidden_state_phenotype/step3_validation_clustering/20260725T001123Z"
)
STEP4_ROOT = (
    PROJECT_ROOT
    / "outputs/hidden_state_phenotype/step4_test_confirmation/20260725T010440Z"
)
VALIDATION_REPRESENTATIONS_PATH = STEP2_ROOT / "participant_representations.parquet"
TEST_REPRESENTATIONS_PATH = STEP4_ROOT / "test_participant_representations.parquet"
VALIDATION_H0_ROOT = STEP2_ROOT / "validation_hidden_states/condition=full_profile"
TEST_H0_ROOT = STEP4_ROOT / "test_hidden_states/condition=full_profile"
VALIDATION_GLYCEMIC_PATH = STEP3_ROOT / "validation_glycemic_nuisance_features.parquet"
STEP3_PCA_LOADINGS_PATH = STEP3_ROOT / "pca_loadings.parquet"

AXIS_RESULTS_PATH = OUTPUT_ROOT / "axis_scan_results.csv"
RIDGE_RESULTS_PATH = OUTPUT_ROOT / "ridge_probe_results.csv"
PAIRWISE_RESULTS_PATH = OUTPUT_ROOT / "ridge_probe_pairwise_comparisons.csv"
MDE_PATH = OUTPUT_ROOT / "ridge_probe_mde.csv"
VALIDATION_PREDICTIONS_PATH = OUTPUT_ROOT / "ridge_probe_validation_oof_predictions.parquet"
TEST_PREDICTIONS_PATH = OUTPUT_ROOT / "ridge_probe_test_predictions.parquet"
FIGURE_AXIS_PATH = OUTPUT_ROOT / "fig3_axis_scan_heatmap.png"
FIGURE_RIDGE_PATH = OUTPUT_ROOT / "fig4_ridge_probe_r2_forest.png"
MANIFEST_PATH = OUTPUT_ROOT / "encoding_test_manifest.json"

RANDOM_SEED = 42
BOOTSTRAP_REPLICATES = 2000
OUTER_FOLDS = 5
OUTER_REPETITIONS = 5
INNER_FOLDS = 5
AXIS_PC_COUNT = 5
FDR_ALPHA = 0.05
MDE_ALPHA = 0.05
MDE_POWER = 0.80
SMALL_R2_FLOOR = 0.05
N_JOBS = -1
STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
SPACES = ["h0", "full_ht", "neutral_ht"]
SPACE_LABELS = {
    "h0": "h0",
    "full_ht": "Full ht",
    "neutral_ht": "Neutral ht",
}
TARGETS = [
    "clinical_pc1",
    "clinical_pc2",
    "bmi",
    "log_tg_hdl",
    "log_c_peptide",
]
TARGET_LABELS = {
    "clinical_pc1": "Clinical PC1: insulin-resistance axis",
    "clinical_pc2": (
        "Clinical PC2: BMI/TG-HDL dissociation axis "
        "(independent of C-peptide)"
    ),
    "bmi": "BMI",
    "log_tg_hdl": "Log TG/HDL",
    "log_c_peptide": "Log C-peptide",
}
NATIVE_UNITS = {
    "clinical_pc1": "clinical PC score",
    "clinical_pc2": "clinical PC score",
    "bmi": "kg/m2",
    "log_tg_hdl": "TG/HDL ratio",
    "log_c_peptide": "ng/mL",
}
H_COLS = [f"h_{index:03d}" for index in range(128)]
R_COLS = [f"r_{index:03d}" for index in range(128)]
STATE_COLS = [f"state_{index:03d}" for index in range(128)]
PCA_COMPONENT_COUNTS = {"full_all": 23, "neutral_all": 8}
SPACE_TO_PCA = {"h0": "full_all", "full_ht": "full_all", "neutral_ht": "neutral_all"}
AGE_CAVEAT = (
    "participants_age is participant age at study visit, NOT age at diabetes "
    "diagnosis; age is not used as a target in this encoding analysis."
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


def load_h0(root: Path, split_name: str) -> tuple[pd.DataFrame, dict]:
    paths = sorted(root.glob("participant_id=*/data.parquet"))
    if not paths:
        raise FileNotFoundError(f"No h0 participant files under {root}")
    rows = []
    counts = []
    repeats_equal = True
    one_per_segment = True
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=["participant_id", "segment_id", "is_h0_row", *H_COLS],
        )
        selected = frame[frame["is_h0_row"].astype(bool)]
        if selected.empty:
            raise RuntimeError(f"No h0 row in {path}")
        counts.append(len(selected))
        one_per_segment &= selected["segment_id"].nunique() == len(selected)
        values = selected[H_COLS].to_numpy(float)
        repeats_equal &= bool(
            np.allclose(values, values[[0]], atol=0.0, rtol=0.0)
        )
        rows.append(
            {
                "participant_id": str(selected["participant_id"].iloc[0]),
                **dict(zip(H_COLS, values[0])),
            }
        )
    result = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    if result["participant_id"].duplicated().any():
        raise RuntimeError(f"Duplicate h0 participants in {split_name}")
    if not np.isfinite(result[H_COLS].to_numpy(float)).all():
        raise RuntimeError(f"Nonfinite h0 values in {split_name}")
    audit = {
        "split": split_name,
        "source_root": str(root),
        "participant_count": len(result),
        "total_h0_rows": int(sum(counts)),
        "h0_rows_per_participant_min": int(min(counts)),
        "h0_rows_per_participant_max": int(max(counts)),
        "one_h0_per_segment": bool(one_per_segment),
        "within_participant_repeats_exactly_equal": bool(repeats_equal),
        "model_forward_pass_run": False,
    }
    if not one_per_segment or not repeats_equal:
        raise RuntimeError(f"h0 QC failed for {split_name}")
    return result, audit


def load_ht(path: Path, split_name: str) -> dict[str, pd.DataFrame]:
    frame = pd.read_parquet(path)
    frame["participant_id"] = frame["participant_id"].astype(str)
    if split_name == "validation":
        frame = frame[
            (frame["balanced_anchor_variant"] == "all_anchors")
            & (frame["context"] == "all")
            & (frame["burn_in_minutes"] == 0)
            & frame["representation_eligible"].astype(bool)
        ]
    else:
        frame = frame[frame["aggregation"] == "all_anchors"]
    result = {}
    for source_name, space_name in (
        ("full_all", "full_ht"),
        ("neutral_all", "neutral_ht"),
    ):
        selected = frame[
            frame["representation_type"] == source_name
        ][["participant_id", *R_COLS]].copy()
        if selected["participant_id"].duplicated().any():
            raise RuntimeError(f"Duplicate {space_name} rows in {split_name}")
        if not np.isfinite(selected[R_COLS].to_numpy(float)).all():
            raise RuntimeError(f"Nonfinite {space_name} values in {split_name}")
        result[space_name] = selected.sort_values("participant_id").reset_index(
            drop=True
        )
    return result


def load_frozen_pca() -> dict[str, dict]:
    result = {}
    for pca_name in ("full_all", "neutral_all"):
        root = STEP3_ROOT / "frozen_validation_pipeline" / pca_name
        result[pca_name] = {
            "scaler": joblib.load(root / f"{pca_name}_scaler.joblib"),
            "pca": joblib.load(root / f"{pca_name}_pca.joblib"),
            "keep": np.load(root / "kept_dimensions.npy"),
            "n_components": PCA_COMPONENT_COUNTS[pca_name],
        }
    return result


def project_scores(
    frame: pd.DataFrame,
    value_columns: list[str],
    frozen: dict,
    pca_name: str,
) -> pd.DataFrame:
    pipeline = frozen[pca_name]
    values = frame[value_columns].to_numpy(float)
    scores = pipeline["pca"].transform(
        pipeline["scaler"].transform(values[:, pipeline["keep"]])
    )[:, : pipeline["n_components"]]
    result = frame[["participant_id"]].copy()
    for index in range(scores.shape[1]):
        result[f"hs_pc{index + 1}"] = scores[:, index]
    return result


def build_representations(
    targets: pd.DataFrame,
) -> tuple[dict[str, dict[str, dict[str, pd.DataFrame]]], dict]:
    h0_validation, h0_validation_audit = load_h0(
        VALIDATION_H0_ROOT, "validation"
    )
    h0_test, h0_test_audit = load_h0(TEST_H0_ROOT, "test")
    ht_validation = load_ht(VALIDATION_REPRESENTATIONS_PATH, "validation")
    ht_test = load_ht(TEST_REPRESENTATIONS_PATH, "test")
    raw = {
        "validation": {"h0": h0_validation, **ht_validation},
        "test": {"h0": h0_test, **ht_test},
    }
    frozen = load_frozen_pca()
    output = {"validation": {}, "test": {}}
    for split_name in ("validation", "test"):
        required_ids = set(
            targets.loc[targets["split"] == split_name, "participant_id"]
        )
        for space in SPACES:
            value_columns = H_COLS if space == "h0" else R_COLS
            selected = raw[split_name][space]
            if not required_ids.issubset(set(selected["participant_id"])):
                raise RuntimeError(
                    f"Missing {space} participants in {split_name}"
                )
            selected = (
                selected[selected["participant_id"].isin(required_ids)]
                .sort_values("participant_id")
                .reset_index(drop=True)
            )
            state = selected[["participant_id", *value_columns]].copy()
            state.columns = ["participant_id", *STATE_COLS]
            scores = project_scores(
                selected,
                value_columns,
                frozen,
                SPACE_TO_PCA[space],
            )
            output[split_name][space] = {"state": state, "scores": scores}
    return output, {
        "validation": h0_validation_audit,
        "test": h0_test_audit,
        "pca_transport": {
            "h0": "frozen full_all validation PCA",
            "full_ht": "frozen full_all validation PCA",
            "neutral_ht": "frozen neutral_all validation PCA",
        },
    }


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(x), size=len(x))
        distribution[index] = spearmanr(x[draw], y[draw]).statistic
    return (
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    )


def axis_scan(
    targets: pd.DataFrame,
    representations: dict,
) -> pd.DataFrame:
    rows = []
    for split_name in ("validation", "test"):
        target_split = targets[targets["split"] == split_name]
        for space in SPACES:
            data = target_split.merge(
                representations[split_name][space]["scores"],
                on="participant_id",
                validate="one_to_one",
            )
            pc_columns = [
                f"hs_pc{index}" for index in range(1, AXIS_PC_COUNT + 1)
            ]
            for target in TARGETS:
                y = data[target].to_numpy(float)
                for pc_index, pc_column in enumerate(pc_columns, start=1):
                    x = data[pc_column].to_numpy(float)
                    test = spearmanr(x, y)
                    ci_low, ci_high = bootstrap_spearman(
                        x,
                        y,
                        stable_seed(
                            "axis",
                            split_name,
                            space,
                            target,
                            pc_index,
                            RANDOM_SEED,
                        ),
                    )
                    rows.append(
                        {
                            "split": split_name,
                            "space": space,
                            "space_label": SPACE_LABELS[space],
                            "target": target,
                            "target_label": TARGET_LABELS[target],
                            "hidden_state_pc": pc_index,
                            "n_participants": len(data),
                            "spearman_rho": float(test.statistic),
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "p_value": float(test.pvalue),
                            "bootstrap_unit": "participant",
                            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        }
                    )
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan
    for _, indexes in result.groupby(
        ["split", "space", "target"]
    ).groups.items():
        result.loc[indexes, "fdr_q_value"] = multipletests(
            result.loc[indexes, "p_value"],
            method="fdr_bh",
        )[1]
    result["significant_fdr"] = result["fdr_q_value"] < FDR_ALPHA
    result["fdr_family"] = (
        "PC1-PC5 within split, target, and space; spaces are not pooled"
    )
    return result


def target_to_native(
    target: str,
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if target in {"log_tg_hdl", "log_c_peptide"}:
        return np.exp(values)
    return values


def performance_metrics(
    target: str,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    observed_native = target_to_native(target, observed)
    predicted_native = target_to_native(target, predicted)
    return {
        "r2": float(r2_score(observed, predicted)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "mae_native": float(
            mean_absolute_error(observed_native, predicted_native)
        ),
    }


def bootstrap_performance(
    target: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result = {
        "r2": np.empty(BOOTSTRAP_REPLICATES),
        "spearman": np.empty(BOOTSTRAP_REPLICATES),
        "mae_native": np.empty(BOOTSTRAP_REPLICATES),
    }
    for index in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(observed), size=len(observed))
        metrics = performance_metrics(
            target,
            observed[draw],
            predicted[draw],
        )
        for metric in result:
            result[metric][index] = metrics[metric]
    return result


def bootstrap_pairwise(
    target: str,
    observed: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    point_a = performance_metrics(target, observed, prediction_a)
    point_b = performance_metrics(target, observed, prediction_b)
    point = {
        "delta_r2": point_a["r2"] - point_b["r2"],
        "delta_spearman": point_a["spearman"] - point_b["spearman"],
        "delta_mae_native": (
            point_a["mae_native"] - point_b["mae_native"]
        ),
    }
    rng = np.random.default_rng(seed)
    distribution = {
        key: np.empty(BOOTSTRAP_REPLICATES) for key in point
    }
    for index in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(observed), size=len(observed))
        metric_a = performance_metrics(
            target, observed[draw], prediction_a[draw]
        )
        metric_b = performance_metrics(
            target, observed[draw], prediction_b[draw]
        )
        distribution["delta_r2"][index] = (
            metric_a["r2"] - metric_b["r2"]
        )
        distribution["delta_spearman"][index] = (
            metric_a["spearman"] - metric_b["spearman"]
        )
        distribution["delta_mae_native"][index] = (
            metric_a["mae_native"] - metric_b["mae_native"]
        )
    intervals = {
        key: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for key, values in distribution.items()
    }
    return point, intervals


def fit_ridge_probes(
    targets: pd.DataFrame,
    representations: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clinical_site = pd.read_parquet(VALIDATION_GLYCEMIC_PATH)[
        ["participant_id", "clinical_site"]
    ].copy()
    clinical_site["participant_id"] = clinical_site["participant_id"].astype(
        str
    )
    validation_rows = []
    test_rows = []
    result_rows = []
    mde_rows = []
    prediction_lookup = {}
    for target_index, target in enumerate(TARGETS):
        validation_target = (
            targets[targets["split"] == "validation"]
            [["participant_id", target]]
            .merge(clinical_site, on="participant_id", validate="one_to_one")
            .sort_values("participant_id")
            .reset_index(drop=True)
        )
        test_target = (
            targets[targets["split"] == "test"]
            [["participant_id", target]]
            .sort_values("participant_id")
            .reset_index(drop=True)
        )
        for space_index, space in enumerate(SPACES):
            validation = validation_target.merge(
                representations["validation"][space]["state"],
                on="participant_id",
                validate="one_to_one",
            )
            test = test_target.merge(
                representations["test"][space]["state"],
                on="participant_id",
                validate="one_to_one",
            )
            y_validation = validation[target].to_numpy(float)
            y_test = test[target].to_numpy(float)
            oof_predictions = np.empty(
                (OUTER_REPETITIONS, len(validation)), dtype=float
            )
            outer_alpha_rows = []
            fold_strategies = []
            for repetition in range(OUTER_REPETITIONS):
                outer_splits, strategy = existing_probe.folds(
                    y_validation,
                    validation["clinical_site"],
                    OUTER_FOLDS,
                    RANDOM_SEED + repetition,
                )
                fold_strategies.append(strategy)
                for fold_index, (train_index, holdout_index) in enumerate(
                    outer_splits
                ):
                    pipeline, grid = existing_probe.estimator(
                        [],
                        [],
                        STATE_COLS,
                        False,
                    )
                    search = GridSearchCV(
                        pipeline,
                        grid,
                        cv=KFold(
                            INNER_FOLDS,
                            shuffle=True,
                            random_state=(
                                RANDOM_SEED
                                + 1000 * repetition
                                + fold_index
                            ),
                        ),
                        scoring="neg_mean_squared_error",
                        n_jobs=N_JOBS,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(
                        validation.iloc[train_index],
                        y_validation[train_index],
                    )
                    oof_predictions[repetition, holdout_index] = (
                        search.predict(validation.iloc[holdout_index])
                    )
                    outer_alpha_rows.append(
                        {
                            "target": target,
                            "space": space,
                            "outer_repetition": repetition,
                            "outer_fold": fold_index,
                            "best_alpha": float(
                                search.best_params_["model__alpha"]
                            ),
                        }
                    )
            for repetition in range(OUTER_REPETITIONS):
                for participant_index, participant_id in enumerate(
                    validation["participant_id"]
                ):
                    validation_rows.append(
                        {
                            "participant_id": participant_id,
                            "split": "validation_nested_oof",
                            "target": target,
                            "space": space,
                            "outer_repetition": repetition,
                            "observed_analysis_scale": y_validation[
                                participant_index
                            ],
                            "predicted_analysis_scale": oof_predictions[
                                repetition, participant_index
                            ],
                        }
                    )
            final_pipeline, final_grid = existing_probe.estimator(
                [],
                [],
                STATE_COLS,
                False,
            )
            final_search = GridSearchCV(
                final_pipeline,
                final_grid,
                cv=KFold(
                    INNER_FOLDS,
                    shuffle=True,
                    random_state=RANDOM_SEED + 19000,
                ),
                scoring="neg_mean_squared_error",
                n_jobs=N_JOBS,
                refit=True,
                error_score="raise",
            )
            final_search.fit(validation, y_validation)
            test_prediction = final_search.predict(test)
            model_path = (
                MODEL_ROOT / f"{target}__{space}__validation_frozen.joblib"
            )
            joblib.dump(final_search.best_estimator_, model_path)
            point = performance_metrics(target, y_test, test_prediction)
            bootstrap = bootstrap_performance(
                target,
                y_test,
                test_prediction,
                stable_seed(
                    "ridge",
                    target,
                    space,
                    RANDOM_SEED,
                ),
            )
            intervals = {
                metric: (
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                )
                for metric, values in bootstrap.items()
            }
            mde, standard_error, critical_value = (
                existing_power.empirical_mde(
                    bootstrap["r2"],
                    MDE_ALPHA,
                )
            )
            if intervals["r2"][0] > 0:
                interpretation = "positive_test_encoding"
            elif intervals["r2"][1] < 0:
                interpretation = "no_effect"
            elif mde <= SMALL_R2_FLOOR:
                interpretation = "no_effect_at_small_r2_floor"
            else:
                interpretation = (
                    "no_large_effect_detected_underpowered_below_mde"
                )
            nested_point = performance_metrics(
                target,
                np.tile(y_validation, OUTER_REPETITIONS),
                oof_predictions.reshape(-1),
            )
            result_rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "space": space,
                    "space_label": SPACE_LABELS[space],
                    "validation_n": len(validation),
                    "test_n": len(test),
                    "predictor_count": len(STATE_COLS),
                    "model_family": "Ridge",
                    "alpha_grid": json.dumps(
                        existing_probe.ALPHAS.tolist()
                    ),
                    "best_alpha_final_validation": float(
                        final_search.best_params_["model__alpha"]
                    ),
                    "outer_selected_alpha_median": float(
                        pd.DataFrame(outer_alpha_rows)["best_alpha"].median()
                    ),
                    "validation_nested_r2": nested_point["r2"],
                    "validation_nested_spearman": nested_point["spearman"],
                    "validation_nested_mae_native": nested_point[
                        "mae_native"
                    ],
                    "test_r2": point["r2"],
                    "test_r2_ci_low": intervals["r2"][0],
                    "test_r2_ci_high": intervals["r2"][1],
                    "test_spearman": point["spearman"],
                    "test_spearman_ci_low": intervals["spearman"][0],
                    "test_spearman_ci_high": intervals["spearman"][1],
                    "test_mae_native": point["mae_native"],
                    "test_mae_native_ci_low": intervals["mae_native"][0],
                    "test_mae_native_ci_high": intervals["mae_native"][1],
                    "native_unit": NATIVE_UNITS[target],
                    "mde_r2_80": mde,
                    "null_interpretation": interpretation,
                    "outer_fold_strategies": json.dumps(
                        sorted(set(fold_strategies))
                    ),
                    "model_path": str(model_path),
                    "validation_tuned_test_frozen": True,
                    "bootstrap_unit": "participant",
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
            mde_rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "space": space,
                    "space_label": SPACE_LABELS[space],
                    "test_n": len(test),
                    "observed_test_r2": point["r2"],
                    "r2_ci_low": intervals["r2"][0],
                    "r2_ci_high": intervals["r2"][1],
                    "minimum_detectable_r2_80": mde,
                    "bootstrap_standard_error": standard_error,
                    "bootstrap_critical_value": critical_value,
                    "power_target": MDE_POWER,
                    "two_sided_alpha": MDE_ALPHA,
                    "small_r2_floor": SMALL_R2_FLOOR,
                    "null_interpretation": interpretation,
                    "method": (
                        "Step 0 empirical bootstrap-error MDE binary search"
                    ),
                }
            )
            prediction_lookup[(target, space)] = {
                "participant_id": test["participant_id"].to_numpy(),
                "observed": y_test,
                "predicted": test_prediction,
            }
            observed_native = target_to_native(target, y_test)
            predicted_native = target_to_native(target, test_prediction)
            for index, participant_id in enumerate(test["participant_id"]):
                test_rows.append(
                    {
                        "participant_id": participant_id,
                        "split": "test",
                        "target": target,
                        "space": space,
                        "observed_analysis_scale": y_test[index],
                        "predicted_analysis_scale": test_prediction[index],
                        "observed_native": observed_native[index],
                        "predicted_native": predicted_native[index],
                        "native_unit": NATIVE_UNITS[target],
                        "model_status": (
                            "validation_tuned_frozen_test_transport"
                        ),
                    }
                )
    pair_rows = []
    pairs = [("h0", "full_ht"), ("h0", "neutral_ht"), ("full_ht", "neutral_ht")]
    for target in TARGETS:
        for space_a, space_b in pairs:
            left = prediction_lookup[(target, space_a)]
            right = prediction_lookup[(target, space_b)]
            if not np.array_equal(
                left["participant_id"], right["participant_id"]
            ):
                raise RuntimeError(
                    f"Pairwise participant mismatch for {target}"
                )
            point, intervals = bootstrap_pairwise(
                target,
                left["observed"],
                left["predicted"],
                right["predicted"],
                stable_seed(
                    "pairwise",
                    target,
                    space_a,
                    space_b,
                    RANDOM_SEED,
                ),
            )
            pair_rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "space_a": space_a,
                    "space_b": space_b,
                    "comparison": f"{space_a}_minus_{space_b}",
                    "test_n": len(left["observed"]),
                    "delta_r2": point["delta_r2"],
                    "delta_r2_ci_low": intervals["delta_r2"][0],
                    "delta_r2_ci_high": intervals["delta_r2"][1],
                    "delta_spearman": point["delta_spearman"],
                    "delta_spearman_ci_low": intervals[
                        "delta_spearman"
                    ][0],
                    "delta_spearman_ci_high": intervals[
                        "delta_spearman"
                    ][1],
                    "delta_mae_native": point["delta_mae_native"],
                    "delta_mae_native_ci_low": intervals[
                        "delta_mae_native"
                    ][0],
                    "delta_mae_native_ci_high": intervals[
                        "delta_mae_native"
                    ][1],
                    "native_unit": NATIVE_UNITS[target],
                    "bootstrap_unit": "paired participant",
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(mde_rows),
        pd.DataFrame(validation_rows),
        pd.DataFrame(test_rows),
    )


def plot_axis_heatmap(results: pd.DataFrame) -> None:
    selected = results[results["split"] == "test"].copy()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.8, 6.5),
        sharex=True,
        sharey=True,
    )
    target_order = TARGETS
    for axis, space in zip(axes, SPACES):
        block = selected[selected["space"] == space]
        matrix = block.pivot(
            index="target",
            columns="hidden_state_pc",
            values="spearman_rho",
        ).reindex(target_order)
        significance = block.pivot(
            index="target",
            columns="hidden_state_pc",
            values="significant_fdr",
        ).reindex(target_order)
        annotations = matrix.copy().astype(object)
        for row in matrix.index:
            for column in matrix.columns:
                marker = "*" if significance.loc[row, column] else ""
                annotations.loc[row, column] = (
                    f"{matrix.loc[row, column]:.2f}{marker}"
                )
        sns.heatmap(
            matrix,
            ax=axis,
            vmin=-0.65,
            vmax=0.65,
            center=0,
            cmap="vlag",
            annot=annotations,
            fmt="",
            linewidths=0.5,
            linecolor="white",
            cbar=space == SPACES[-1],
            cbar_kws={"label": "Test Spearman rho"},
        )
        axis.set_title(SPACE_LABELS[space])
        axis.set_xlabel("Frozen hidden-state PC")
        axis.set_ylabel("")
        axis.set_xticklabels(
            [f"PC{int(label.get_text())}" for label in axis.get_xticklabels()]
        )
        axis.set_yticklabels(
            [TARGET_LABELS[target] for target in target_order],
            rotation=0,
        )
    fig.suptitle(
        "T2D-only continuous clinical target encoding: axis-level test scan",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.015,
        "Asterisks indicate Benjamini-Hochberg q < 0.05 within each target and space.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(FIGURE_AXIS_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ridge_forest(
    results: pd.DataFrame,
    mde: pd.DataFrame,
) -> None:
    target_order = TARGETS
    y_positions = np.arange(len(target_order))[::-1]
    offsets = {"h0": 0.20, "full_ht": 0.0, "neutral_ht": -0.20}
    fig, axis = plt.subplots(figsize=(11.5, 7.2))
    all_bounds = []
    for row_index, target in enumerate(target_order):
        y = y_positions[row_index]
        target_mde = float(
            mde[mde["target"] == target]["minimum_detectable_r2_80"].max()
        )
        axis.add_patch(
            Rectangle(
                (-target_mde, y - 0.38),
                2 * target_mde,
                0.76,
                facecolor="#CCCCCC",
                alpha=0.22,
                edgecolor="none",
                zorder=0,
            )
        )
        for space_index, space in enumerate(SPACES):
            row = results[
                (results["target"] == target)
                & (results["space"] == space)
            ].iloc[0]
            point_y = y + offsets[space]
            axis.errorbar(
                row["test_r2"],
                point_y,
                xerr=[
                    [row["test_r2"] - row["test_r2_ci_low"]],
                    [row["test_r2_ci_high"] - row["test_r2"]],
                ],
                fmt="o",
                color=STRATUM_COLORS[space_index],
                markersize=6.5,
                capsize=3,
                linewidth=1.4,
                label=SPACE_LABELS[space] if row_index == 0 else None,
                zorder=3,
            )
            all_bounds.extend(
                [
                    row["test_r2_ci_low"],
                    row["test_r2_ci_high"],
                    -target_mde,
                    target_mde,
                ]
            )
    axis.axvline(0, color="#222222", linewidth=1)
    axis.set_yticks(y_positions)
    axis.set_yticklabels([TARGET_LABELS[target] for target in target_order])
    axis.set_xlabel("Frozen test R² with participant-bootstrap 95% CI")
    axis.set_title("T2D-only multivariate ridge encoding probes")
    axis.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=STRATUM_COLORS[index],
                linestyle="none",
                label=SPACE_LABELS[space],
            )
            for index, space in enumerate(SPACES)
        ]
        + [Patch(facecolor="#CCCCCC", alpha=0.35, label="80% MDE region")],
        loc="best",
        frameon=True,
    )
    finite_bounds = np.asarray(all_bounds, dtype=float)
    finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
    if len(finite_bounds):
        lower = min(float(finite_bounds.min()), -0.05)
        upper = max(float(finite_bounds.max()), 0.05)
        padding = 0.08 * (upper - lower)
        axis.set_xlim(lower - padding, upper + padding)
    axis.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURE_RIDGE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    np.random.seed(RANDOM_SEED)

    required_inputs = {
        "clinical_targets": TARGETS_PATH,
        "clinical_targets_manifest": TARGETS_MANIFEST_PATH,
        "validation_representations": VALIDATION_REPRESENTATIONS_PATH,
        "test_representations": TEST_REPRESENTATIONS_PATH,
        "step3_pca_loadings": STEP3_PCA_LOADINGS_PATH,
        "full_pca_scaler": STEP3_ROOT / "frozen_validation_pipeline/full_all/full_all_scaler.joblib",
        "full_pca_model": STEP3_ROOT / "frozen_validation_pipeline/full_all/full_all_pca.joblib",
        "full_pca_kept_dimensions": STEP3_ROOT / "frozen_validation_pipeline/full_all/kept_dimensions.npy",
        "neutral_pca_scaler": STEP3_ROOT / "frozen_validation_pipeline/neutral_all/neutral_all_scaler.joblib",
        "neutral_pca_model": STEP3_ROOT / "frozen_validation_pipeline/neutral_all/neutral_all_pca.joblib",
        "neutral_pca_kept_dimensions": STEP3_ROOT / "frozen_validation_pipeline/neutral_all/kept_dimensions.npy",
        "validation_glycemic_features": VALIDATION_GLYCEMIC_PATH,
        "existing_ridge_probe_script": (
            SCRIPTS_ROOT / "run_hidden_state_clinical_probes.py"
        ),
        "existing_mde_script": (
            SCRIPTS_ROOT / "run_beyond_glucose_dynamics_power.py"
        ),
    }
    missing = [
        str(path) for path in required_inputs.values() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Prompt 2 inputs: {missing}")

    clinical_manifest = json.loads(TARGETS_MANIFEST_PATH.read_text())
    if not clinical_manifest["validation_fit_test_frozen"]:
        raise RuntimeError("Clinical target validation/test freeze gate failed")
    if (
        clinical_manifest["pca"]["component_names"]["PC2"]
        != "BMI/TG-HDL dissociation axis (independent of C-peptide)"
    ):
        raise RuntimeError("PC2 name correction is not frozen")

    targets = pd.read_parquet(TARGETS_PATH)
    targets["participant_id"] = targets["participant_id"].astype(str)
    counts = targets.groupby("split").size().to_dict()
    if counts != {"validation": 91, "test": 83}:
        raise RuntimeError(f"Unexpected clinical target counts: {counts}")
    if not np.isfinite(targets[TARGETS].to_numpy(float)).all():
        raise RuntimeError("Nonfinite clinical targets")

    representations, representation_audit = build_representations(targets)
    axis_results = axis_scan(targets, representations)
    axis_results.to_csv(AXIS_RESULTS_PATH, index=False)
    plot_axis_heatmap(axis_results)

    (
        ridge_results,
        pairwise_results,
        mde_results,
        validation_predictions,
        test_predictions,
    ) = fit_ridge_probes(targets, representations)
    ridge_results.to_csv(RIDGE_RESULTS_PATH, index=False)
    pairwise_results.to_csv(PAIRWISE_RESULTS_PATH, index=False)
    mde_results.to_csv(MDE_PATH, index=False)
    validation_predictions.to_parquet(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )
    test_predictions.to_parquet(
        TEST_PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )
    plot_ridge_forest(ridge_results, mde_results)

    expected_axis_rows = 2 * len(SPACES) * len(TARGETS) * AXIS_PC_COUNT
    expected_ridge_rows = len(SPACES) * len(TARGETS)
    expected_pair_rows = 3 * len(TARGETS)
    qc = {
        "axis_rows_expected": expected_axis_rows,
        "axis_rows_observed": len(axis_results),
        "ridge_rows_expected": expected_ridge_rows,
        "ridge_rows_observed": len(ridge_results),
        "pairwise_rows_expected": expected_pair_rows,
        "pairwise_rows_observed": len(pairwise_results),
        "mde_rows_expected": expected_ridge_rows,
        "mde_rows_observed": len(mde_results),
        "validation_prediction_rows": len(validation_predictions),
        "test_prediction_rows": len(test_predictions),
        "frozen_model_count": len(list(MODEL_ROOT.glob("*.joblib"))),
        "finite_axis_estimates": bool(
            np.isfinite(
                axis_results[
                    ["spearman_rho", "ci_low", "ci_high", "fdr_q_value"]
                ].to_numpy(float)
            ).all()
        ),
        "finite_ridge_estimates": bool(
            np.isfinite(
                ridge_results[
                    [
                        "test_r2",
                        "test_r2_ci_low",
                        "test_r2_ci_high",
                        "test_spearman",
                        "test_mae_native",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "validation_tuned_test_frozen": bool(
            ridge_results["validation_tuned_test_frozen"].all()
        ),
        "no_model_forward_pass": True,
        "no_hidden_state_regeneration": True,
    }
    if not all(
        [
            qc["axis_rows_observed"] == qc["axis_rows_expected"],
            qc["ridge_rows_observed"] == qc["ridge_rows_expected"],
            qc["pairwise_rows_observed"] == qc["pairwise_rows_expected"],
            qc["mde_rows_observed"] == qc["mde_rows_expected"],
            qc["frozen_model_count"] == expected_ridge_rows,
            qc["finite_axis_estimates"],
            qc["finite_ridge_estimates"],
            qc["validation_tuned_test_frozen"],
        ]
    ):
        raise RuntimeError(f"Prompt 2 QC failed: {qc}")

    output_paths = {
        "axis_scan_results": AXIS_RESULTS_PATH,
        "ridge_probe_results": RIDGE_RESULTS_PATH,
        "ridge_probe_pairwise_comparisons": PAIRWISE_RESULTS_PATH,
        "ridge_probe_mde": MDE_PATH,
        "validation_oof_predictions": VALIDATION_PREDICTIONS_PATH,
        "test_predictions": TEST_PREDICTIONS_PATH,
        "figure_axis_scan": FIGURE_AXIS_PATH,
        "figure_ridge_forest": FIGURE_RIDGE_PATH,
    }
    significant_counts = (
        axis_results.groupby(["split", "space"])["significant_fdr"]
        .sum()
        .astype(int)
        .to_dict()
    )
    manifest = {
        "analysis": (
            "Continuous clinical encoding test: axis-level and multivariate, "
            "diagnosis-defined T2D only"
        ),
        "status": "QC_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_counts": counts,
        "targets": TARGETS,
        "target_labels": TARGET_LABELS,
        "spaces": SPACES,
        "axis_scan": {
            "hidden_state_pcs": AXIS_PC_COUNT,
            "splits_reported": ["validation", "test"],
            "figure_split": "test",
            "correlation": "Spearman",
            "bootstrap": {
                "unit": "participant",
                "replicates": BOOTSTRAP_REPLICATES,
                "interval": "percentile 95%",
            },
            "fdr": (
                "Benjamini-Hochberg across PC1-PC5 within each split, "
                "target, and space; spaces are not pooled"
            ),
            "fdr_alpha": FDR_ALPHA,
            "significant_cell_counts": {
                "|".join(key): value
                for key, value in significant_counts.items()
            },
        },
        "ridge_probe": {
            "methodology_source": str(
                SCRIPTS_ROOT / "run_hidden_state_clinical_probes.py"
            ),
            "reused_functions": [
                "estimator",
                "preprocessor",
                "folds",
                "ALPHAS",
            ],
            "predictors": "full 128-dimensional representation",
            "model_family": "Ridge",
            "outer_cv": {
                "folds": OUTER_FOLDS,
                "repetitions": OUTER_REPETITIONS,
                "stratification": (
                    "target quintile x clinical site when supported, then "
                    "target quintile, then shuffled KFold"
                ),
            },
            "inner_cv_folds": INNER_FOLDS,
            "alpha_grid": existing_probe.ALPHAS.tolist(),
            "final_fit": (
                "all T2D validation participants with validation CV tuning"
            ),
            "test_rule": "single frozen application without test refit",
            "bootstrap": {
                "unit": "participant",
                "replicates": BOOTSTRAP_REPLICATES,
                "interval": "percentile 95%",
            },
            "pairwise_comparisons": (
                "paired participant bootstrap for all three space pairs"
            ),
            "native_mae_rules": {
                "bmi": "BMI kg/m2",
                "log_tg_hdl": "exp transform to TG/HDL ratio",
                "log_c_peptide": "exp transform to C-peptide ng/mL",
                "clinical_pc1": "clinical PC score units",
                "clinical_pc2": "clinical PC score units",
            },
        },
        "mde": {
            "methodology_source": str(
                SCRIPTS_ROOT / "run_beyond_glucose_dynamics_power.py"
            ),
            "reused_function": "empirical_mde",
            "test_n": 83,
            "power": MDE_POWER,
            "two_sided_alpha": MDE_ALPHA,
            "effect_metric": "test R2",
            "small_r2_floor": SMALL_R2_FLOOR,
            "null_labels": [
                "no_effect",
                "no_effect_at_small_r2_floor",
                "no_large_effect_detected_underpowered_below_mde",
            ],
        },
        "representation_audit": representation_audit,
        "age_caveat": AGE_CAVEAT,
        "clinical_component_names": clinical_manifest["pca"][
            "component_names"
        ],
        "qc": qc,
        "input_paths": {
            name: str(path) for name, path in required_inputs.items()
        },
        "input_hashes": {
            name: sha256_file(path) for name, path in required_inputs.items()
        },
        "output_paths": {
            name: str(path) for name, path in output_paths.items()
        },
        "output_hashes": {
            name: sha256_file(path) for name, path in output_paths.items()
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
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
    print(
        json.dumps(
            {
                "output_directory": str(OUTPUT_ROOT),
                "status": manifest["status"],
                "cohort_counts": counts,
                "axis_significant_counts": manifest["axis_scan"][
                    "significant_cell_counts"
                ],
                "ridge_test_results": ridge_results[
                    [
                        "target",
                        "space",
                        "test_r2",
                        "test_r2_ci_low",
                        "test_r2_ci_high",
                        "test_spearman",
                        "test_mae_native",
                        "mde_r2_80",
                        "null_interpretation",
                    ]
                ].to_dict(orient="records"),
                "manifest": str(MANIFEST_PATH),
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
