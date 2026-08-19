"""Targeted HbA1c positive-control probe using frozen participant exports."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUTER_FOLDS = 5
INNER_FOLDS = 5
OUTER_REPETITIONS = 5
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 2000
RANDOM_SEED = 42
RIDGE_ALPHA_GRID = np.power(10.0, np.arange(-4.0, 4.0001, 0.5))
HIDDEN_COLUMNS = tuple(f"state_{index:03d}" for index in range(128))
RAW_HIDDEN_COLUMNS = tuple(f"r_{index:03d}" for index in range(128))
GLYCEMIC_NUMERIC = (
    "mean_glucose",
    "glucose_sd",
    "glucose_cv",
    "tir_70_180",
    "tar_above_180",
    "tbr_below_70",
    "mean_absolute_glucose_slope",
    "glucose_range",
    "available_cgm_hours",
)
ADJUSTED_NUMERIC = (*GLYCEMIC_NUMERIC, "age")
ADJUSTED_CATEGORICAL = ("sex", "clinical_site", "study_group")
TARGET_COLUMN = "hba1c_percent_baseline"
TARGET_LABEL = "HbA1c"
TARGET_UNIT = "percent"
PANEL_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "final_multimodal_dataset_20260515_184339.parquet"
)
HOUSE_NAVY = "#003366"
HOUSE_CRIMSON = "#BA2828"
HOUSE_TEAL = "#5BBABA"
HOUSE_KEY_RED = "#FF0000"
HOUSE_GRAY = "#888888"
NO_EM_DASH = "\u2014"


def stable_seed(*parts: Any, base_seed: int = RANDOM_SEED) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="little"
    )
    return int((value + base_seed) % (2**32 - 1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_converter(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, default=json_converter
        )
        + "\n"
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "r2": float(r2_score(observed, predicted)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "pearson": float(pearsonr(observed, predicted).statistic),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "signed_bias": float(np.mean(predicted - observed)),
    }


def preprocessing_pipeline(
    numeric: tuple[str, ...],
    categorical: tuple[str, ...],
    hidden: tuple[str, ...],
) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median", add_indicator=True
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric),
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                list(categorical),
            )
        )
    if hidden:
        transformers.append(
            ("hidden_state", StandardScaler(), list(hidden))
        )
    return ColumnTransformer(
        transformers, remainder="drop", sparse_threshold=0
    )


def estimator(
    numeric: tuple[str, ...],
    categorical: tuple[str, ...],
    hidden: tuple[str, ...],
) -> tuple[Pipeline, dict[str, list[float]]]:
    pipeline = Pipeline(
        [
            (
                "preprocess",
                preprocessing_pipeline(numeric, categorical, hidden),
            ),
            ("model", Ridge()),
        ]
    )
    return pipeline, {"model__alpha": RIDGE_ALPHA_GRID.tolist()}


def outer_folds(
    target: np.ndarray,
    clinical_site: pd.Series,
    fold_count: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    quintile = pd.qcut(
        pd.Series(target),
        q=5,
        labels=False,
        duplicates="drop",
    ).astype(str)
    combined = (
        clinical_site.fillna("<missing>").astype(str).reset_index(drop=True)
        + "|"
        + quintile
    )
    if combined.value_counts().min() >= fold_count:
        labels = combined
        strategy = "target_quintile_x_site"
    elif quintile.value_counts().min() >= fold_count:
        labels = quintile
        strategy = "target_quintile"
    else:
        labels = None
        strategy = "deterministic_shuffled_kfold"
    splitter = (
        KFold(fold_count, shuffle=True, random_state=seed)
        if labels is None
        else StratifiedKFold(fold_count, shuffle=True, random_state=seed)
    )
    positions = np.arange(len(target))
    splits = (
        list(splitter.split(positions))
        if labels is None
        else list(splitter.split(positions, labels))
    )
    return splits, strategy


def panel_glucose_range(
    panel_path: Path,
    participant_ids: list[str],
) -> pd.DataFrame:
    panel = pd.read_parquet(
        panel_path,
        columns=["participant_id", "cgm_glucose_mean", "cgm_count"],
        filters=[("participant_id", "in", participant_ids)],
    )
    panel["participant_id"] = panel["participant_id"].astype(str)
    rows = []
    for participant_id, group in panel.groupby(
        "participant_id", sort=True
    ):
        glucose = pd.to_numeric(
            group["cgm_glucose_mean"], errors="coerce"
        )
        valid = group["cgm_count"].fillna(0).gt(0) & glucose.notna()
        values = glucose.loc[valid].to_numpy(float)
        rows.append(
            {
                "participant_id": participant_id,
                "glucose_range": float(np.ptp(values))
                if len(values)
                else np.nan,
                "valid_cgm_rows_for_range": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def load_participant_ids(
    step2_directory: Path,
    step4_directory: Path,
) -> tuple[list[str], list[str]]:
    step2_manifest = json.loads(
        (step2_directory / "step2_manifest.json").read_text()
    )
    step4_manifest = json.loads(
        (step4_directory / "step4_manifest.json").read_text()
    )
    validation_ids = sorted(
        map(str, step2_manifest["validation_participant_ids"])
    )
    test_ids = sorted(map(str, step4_manifest["participant_ids"]))
    if len(validation_ids) != 239 or len(test_ids) != 221:
        raise RuntimeError("Canonical validation or test count changed")
    if set(validation_ids) & set(test_ids):
        raise RuntimeError("Validation and test participants overlap")
    return validation_ids, test_ids


def representation_frames(
    step2_directory: Path,
    step3_directory: Path,
    step4_directory: Path,
    validation_ids: list[str],
    test_ids: list[str],
) -> dict[str, dict[str, pd.DataFrame]]:
    validation_export = pd.read_parquet(
        step2_directory / "participant_representations.parquet"
    )
    test_export = pd.read_parquet(
        step4_directory / "test_participant_representations.parquet"
    )
    validation_export["participant_id"] = validation_export[
        "participant_id"
    ].astype(str)
    test_export["participant_id"] = test_export["participant_id"].astype(str)
    output: dict[str, dict[str, pd.DataFrame]] = {
        "validation": {},
        "test": {},
    }
    for condition in ("full_all", "neutral_all"):
        validation = validation_export.loc[
            validation_export["representation_type"].eq(condition)
            & validation_export["balanced_anchor_variant"].eq("all_anchors"),
            ["participant_id", *RAW_HIDDEN_COLUMNS],
        ].copy()
        test = test_export.loc[
            test_export["representation_type"].eq(condition),
            ["participant_id", *RAW_HIDDEN_COLUMNS],
        ].copy()
        if set(validation["participant_id"]) != set(validation_ids):
            raise RuntimeError(
                f"Validation representation mismatch for {condition}"
            )
        if set(test["participant_id"]) != set(test_ids):
            raise RuntimeError(
                f"Test representation mismatch for {condition}"
            )
        if validation.duplicated("participant_id").any():
            raise RuntimeError(
                f"Duplicate validation representation for {condition}"
            )
        if test.duplicated("participant_id").any():
            raise RuntimeError(
                f"Duplicate test representation for {condition}"
            )
        validation.columns = ["participant_id", *HIDDEN_COLUMNS]
        test.columns = ["participant_id", *HIDDEN_COLUMNS]
        output["validation"][condition] = validation
        output["test"][condition] = test
    residual_validation = pd.read_parquet(
        step3_directory / "glucose_residualized_representations.parquet"
    )
    residual_validation["participant_id"] = residual_validation[
        "participant_id"
    ].astype(str)
    residual_columns = [
        column
        for column in residual_validation.columns
        if column.startswith("h_") or column.startswith("r_")
    ][-128:]
    residual_test = test_export.loc[
        test_export["representation_type"].eq(
            "neutral_glucose_residual"
        ),
        ["participant_id", *RAW_HIDDEN_COLUMNS],
    ].copy()
    if (
        set(residual_validation["participant_id"]) != set(validation_ids)
        or len(residual_columns) != 128
    ):
        raise RuntimeError("Validation residual representation mismatch")
    if set(residual_test["participant_id"]) != set(test_ids):
        raise RuntimeError("Test residual representation mismatch")
    residual_validation = residual_validation[
        ["participant_id", *residual_columns]
    ].copy()
    residual_validation.columns = ["participant_id", *HIDDEN_COLUMNS]
    residual_test.columns = ["participant_id", *HIDDEN_COLUMNS]
    output["validation"][
        "neutral_glucose_residual"
    ] = residual_validation
    output["test"]["neutral_glucose_residual"] = residual_test
    for split_frames in output.values():
        for name, frame in split_frames.items():
            hidden = frame.loc[:, HIDDEN_COLUMNS].to_numpy(float)
            if not np.isfinite(hidden).all():
                raise RuntimeError(f"Nonfinite hidden values in {name}")
    return output


def build_baseline_frames(
    step3_directory: Path,
    step4_directory: Path,
    static_table: Path,
    validation_ids: list[str],
    test_ids: list[str],
    panel_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation = pd.read_parquet(
        step3_directory / "validation_glycemic_nuisance_features.parquet"
    )
    test = pd.read_parquet(
        step4_directory / "test_glycemic_nuisance_features.parquet"
    )
    validation["participant_id"] = validation["participant_id"].astype(str)
    test["participant_id"] = test["participant_id"].astype(str)
    static_columns = [
        "participant_id",
        TARGET_COLUMN,
        f"{TARGET_COLUMN}_date",
        "hba1c_percent_n_records",
        "hba1c_percent_value_range",
        "hba1c_percent_days_to_cgm_start",
        "participants_age",
        "demo_sex_at_birth",
        "participants_clinical_site",
        "participants_study_group",
    ]
    static = pd.read_parquet(static_table, columns=static_columns)
    static["participant_id"] = static["participant_id"].astype(str)
    if static.duplicated("participant_id").any():
        raise RuntimeError("Duplicate participant rows in static table")
    static_aliases = static.rename(
        columns={
            "participants_age": "age",
            "demo_sex_at_birth": "sex",
            "participants_clinical_site": "clinical_site_static",
            "participants_study_group": "study_group_static",
        }
    )
    merge_columns = [
        "participant_id",
        TARGET_COLUMN,
        f"{TARGET_COLUMN}_date",
        "hba1c_percent_n_records",
        "hba1c_percent_value_range",
        "hba1c_percent_days_to_cgm_start",
        "age",
        "sex",
        "clinical_site_static",
        "study_group_static",
    ]
    validation = validation.merge(
        static_aliases[merge_columns],
        on="participant_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_static"),
    )
    test = test.merge(
        static_aliases[merge_columns],
        on="participant_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_static"),
    )
    for frame in (validation, test):
        if "age_static" in frame:
            frame["age"] = frame["age"].fillna(frame["age_static"])
            frame.drop(columns=["age_static"], inplace=True)
        if "sex_static" in frame:
            frame["sex"] = frame["sex"].fillna(frame["sex_static"])
            frame.drop(columns=["sex_static"], inplace=True)
        frame["clinical_site"] = frame["clinical_site"].fillna(
            frame["clinical_site_static"]
        )
        frame["study_group"] = frame["study_group"].fillna(
            frame["study_group_static"]
        )
        frame.drop(
            columns=["clinical_site_static", "study_group_static"],
            inplace=True,
        )
    test_range = panel_glucose_range(panel_path, test_ids)
    test = test.drop(columns=["glucose_range"], errors="ignore").merge(
        test_range,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    if set(validation["participant_id"]) != set(validation_ids):
        raise RuntimeError("Validation baseline cohort mismatch")
    if set(test["participant_id"]) != set(test_ids):
        raise RuntimeError("Test baseline cohort mismatch")
    validation = validation.set_index("participant_id").reindex(
        validation_ids
    ).reset_index()
    test = test.set_index("participant_id").reindex(test_ids).reset_index()
    return validation, test, static


def feature_specs() -> dict[str, dict[str, Any]]:
    return {
        "glycemic_baseline": {
            "numeric": GLYCEMIC_NUMERIC,
            "categorical": (),
            "hidden": (),
            "representation": None,
        },
        "full_state_only": {
            "numeric": (),
            "categorical": (),
            "hidden": HIDDEN_COLUMNS,
            "representation": "full_all",
        },
        "neutral_state_only": {
            "numeric": (),
            "categorical": (),
            "hidden": HIDDEN_COLUMNS,
            "representation": "neutral_all",
        },
        "glycemic_plus_full_state": {
            "numeric": GLYCEMIC_NUMERIC,
            "categorical": (),
            "hidden": HIDDEN_COLUMNS,
            "representation": "full_all",
        },
        "glycemic_plus_neutral_state": {
            "numeric": GLYCEMIC_NUMERIC,
            "categorical": (),
            "hidden": HIDDEN_COLUMNS,
            "representation": "neutral_all",
        },
        "glycemic_plus_residual_neutral_state": {
            "numeric": GLYCEMIC_NUMERIC,
            "categorical": (),
            "hidden": HIDDEN_COLUMNS,
            "representation": "neutral_glucose_residual",
        },
        "adjusted_baseline": {
            "numeric": ADJUSTED_NUMERIC,
            "categorical": ADJUSTED_CATEGORICAL,
            "hidden": (),
            "representation": None,
        },
        "adjusted_plus_full_state": {
            "numeric": ADJUSTED_NUMERIC,
            "categorical": ADJUSTED_CATEGORICAL,
            "hidden": HIDDEN_COLUMNS,
            "representation": "full_all",
        },
        "adjusted_plus_neutral_state": {
            "numeric": ADJUSTED_NUMERIC,
            "categorical": ADJUSTED_CATEGORICAL,
            "hidden": HIDDEN_COLUMNS,
            "representation": "neutral_all",
        },
    }


def merge_representation(
    baseline: pd.DataFrame,
    representation: pd.DataFrame | None,
) -> pd.DataFrame:
    if representation is None:
        return baseline.copy()
    return baseline.merge(
        representation, on="participant_id", validate="one_to_one"
    )


def weighted_bootstrap_delta(
    baseline: pd.DataFrame,
    augmented: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    keys = ["participant_id", "outer_repetition"]
    merged = baseline[
        [*keys, "observed", "predicted"]
    ].merge(
        augmented[[*keys, "predicted"]],
        on=keys,
        suffixes=("_baseline", "_augmented"),
        validate="one_to_one",
    )
    participant_ids = sorted(merged["participant_id"].astype(str).unique())
    repetitions = sorted(merged["outer_repetition"].unique())
    merged["participant_id"] = merged["participant_id"].astype(str)
    ordered = merged.set_index(keys).reindex(
        pd.MultiIndex.from_product(
            [participant_ids, repetitions], names=keys
        )
    )
    if ordered.isna().any().any():
        raise RuntimeError("Incomplete repeated-CV prediction grid")
    observed = ordered["observed"].to_numpy(float)
    baseline_prediction = ordered["predicted_baseline"].to_numpy(float)
    augmented_prediction = ordered["predicted_augmented"].to_numpy(float)
    participant_count = len(participant_ids)
    repetition_count = len(repetitions)
    rng = np.random.default_rng(seed)
    participant_weights = rng.multinomial(
        participant_count,
        np.repeat(1.0 / participant_count, participant_count),
        size=repeats,
    )
    weights = np.repeat(
        participant_weights, repetition_count, axis=1
    ).astype(float)
    weight_sum = weights.sum(axis=1)
    weighted_y = weights.dot(observed)
    sst = weights.dot(observed * observed) - weighted_y**2 / weight_sum
    baseline_sse = weights.dot(
        (observed - baseline_prediction) ** 2
    )
    augmented_sse = weights.dot(
        (observed - augmented_prediction) ** 2
    )
    baseline_r2 = 1.0 - baseline_sse / sst
    augmented_r2 = 1.0 - augmented_sse / sst
    baseline_mae = weights.dot(
        np.abs(observed - baseline_prediction)
    ) / weight_sum
    augmented_mae = weights.dot(
        np.abs(observed - augmented_prediction)
    ) / weight_sum
    return {
        "delta_r2": tuple(
            np.percentile(augmented_r2 - baseline_r2, [2.5, 97.5])
        ),
        "delta_mae": tuple(
            np.percentile(augmented_mae - baseline_mae, [2.5, 97.5])
        ),
    }


def simple_bootstrap_delta(
    baseline: pd.DataFrame,
    augmented: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    merged = baseline[
        ["participant_id", "observed", "predicted"]
    ].merge(
        augmented[["participant_id", "predicted"]],
        on="participant_id",
        suffixes=("_baseline", "_augmented"),
        validate="one_to_one",
    )
    participant_count = len(merged)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        participant_count,
        np.repeat(1.0 / participant_count, participant_count),
        size=repeats,
    ).astype(float)
    observed = merged["observed"].to_numpy(float)
    baseline_prediction = merged["predicted_baseline"].to_numpy(float)
    augmented_prediction = merged["predicted_augmented"].to_numpy(float)
    weight_sum = weights.sum(axis=1)
    weighted_y = weights.dot(observed)
    sst = weights.dot(observed * observed) - weighted_y**2 / weight_sum
    baseline_sse = weights.dot(
        (observed - baseline_prediction) ** 2
    )
    augmented_sse = weights.dot(
        (observed - augmented_prediction) ** 2
    )
    baseline_r2 = 1.0 - baseline_sse / sst
    augmented_r2 = 1.0 - augmented_sse / sst
    baseline_mae = weights.dot(
        np.abs(observed - baseline_prediction)
    ) / weight_sum
    augmented_mae = weights.dot(
        np.abs(observed - augmented_prediction)
    ) / weight_sum
    return {
        "delta_r2": tuple(
            np.percentile(augmented_r2 - baseline_r2, [2.5, 97.5])
        ),
        "delta_mae": tuple(
            np.percentile(augmented_mae - baseline_mae, [2.5, 97.5])
        ),
    }


def performance_table(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for split, frame in (
        ("validation_nested_cv", validation_predictions),
        ("test_transport", test_predictions),
    ):
        for feature_set, group in frame.groupby("feature_set"):
            result = metrics(group["observed"], group["predicted"])
            rows.append(
                {
                    "split": split,
                    "analysis_role": (
                        "development nested CV"
                        if split == "validation_nested_cv"
                        else "targeted positive-control predictive transport"
                    ),
                    "target": TARGET_LABEL,
                    "target_unit": TARGET_UNIT,
                    "feature_set": feature_set,
                    "n_participants": group["participant_id"].nunique(),
                    "n_prediction_rows": len(group),
                    **result,
                }
            )
    return pd.DataFrame(rows)


def incremental_table(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = (
        ("glycemic_plus_full_state", "glycemic_baseline"),
        ("glycemic_plus_neutral_state", "glycemic_baseline"),
        (
            "glycemic_plus_residual_neutral_state",
            "glycemic_baseline",
        ),
        ("adjusted_plus_full_state", "adjusted_baseline"),
        ("adjusted_plus_neutral_state", "adjusted_baseline"),
    )
    rows = []
    for split, frame in (
        ("validation_nested_cv", validation_predictions),
        ("test_transport", test_predictions),
    ):
        for augmented_name, baseline_name in comparisons:
            baseline = frame[frame["feature_set"].eq(baseline_name)]
            augmented = frame[frame["feature_set"].eq(augmented_name)]
            baseline_metric = metrics(
                baseline["observed"], baseline["predicted"]
            )
            augmented_metric = metrics(
                augmented["observed"], augmented["predicted"]
            )
            intervals = (
                weighted_bootstrap_delta(
                    baseline,
                    augmented,
                    bootstrap_repeats,
                    stable_seed(
                        split,
                        augmented_name,
                        "bootstrap",
                        base_seed=seed,
                    ),
                )
                if split == "validation_nested_cv"
                else simple_bootstrap_delta(
                    baseline,
                    augmented,
                    bootstrap_repeats,
                    stable_seed(
                        split,
                        augmented_name,
                        "bootstrap",
                        base_seed=seed,
                    ),
                )
            )
            rows.append(
                {
                    "split": split,
                    "analysis_role": (
                        "development nested CV"
                        if split == "validation_nested_cv"
                        else "targeted positive-control predictive transport"
                    ),
                    "target": TARGET_LABEL,
                    "feature_set": augmented_name,
                    "reference_feature_set": baseline_name,
                    "n_participants": baseline["participant_id"].nunique(),
                    "baseline_r2": baseline_metric["r2"],
                    "augmented_r2": augmented_metric["r2"],
                    "delta_r2":
                        augmented_metric["r2"] - baseline_metric["r2"],
                    "delta_r2_ci_low": intervals["delta_r2"][0],
                    "delta_r2_ci_high": intervals["delta_r2"][1],
                    "baseline_mae": baseline_metric["mae"],
                    "augmented_mae": augmented_metric["mae"],
                    "delta_mae":
                        augmented_metric["mae"] - baseline_metric["mae"],
                    "delta_mae_ci_low": intervals["delta_mae"][0],
                    "delta_mae_ci_high": intervals["delta_mae"][1],
                    "direction_agreement": np.nan,
                }
            )
    result = pd.DataFrame(rows)
    for feature_set in (
        "glycemic_plus_full_state",
        "glycemic_plus_neutral_state",
        "glycemic_plus_residual_neutral_state",
        "adjusted_plus_full_state",
        "adjusted_plus_neutral_state",
    ):
        mask = result["feature_set"].eq(feature_set)
        directions = np.sign(result.loc[mask, "delta_r2"].to_numpy(float))
        agreement = bool(len(directions) == 2 and directions[0] == directions[1])
        result.loc[mask, "direction_agreement"] = agreement
    return result


def make_incremental_figure(
    incremental: pd.DataFrame,
    output_path: Path,
) -> None:
    selected = incremental[
        incremental["feature_set"].isin(
            [
                "glycemic_plus_full_state",
                "glycemic_plus_neutral_state",
                "glycemic_plus_residual_neutral_state",
            ]
        )
    ].copy()
    label_map = {
        "glycemic_plus_full_state": "Full profile",
        "glycemic_plus_neutral_state": "Static neutral",
        "glycemic_plus_residual_neutral_state":
            "Residualized static neutral",
    }
    split_map = {
        "validation_nested_cv": "Validation nested CV",
        "test_transport": "Targeted test transport",
    }
    selected["label"] = selected["feature_set"].map(label_map)
    y_order = [
        "Full profile",
        "Static neutral",
        "Residualized static neutral",
    ]
    y_positions = {
        label: len(y_order) - 1 - index
        for index, label in enumerate(y_order)
    }
    offsets = {
        "validation_nested_cv": 0.11,
        "test_transport": -0.11,
    }
    colors = {
        "validation_nested_cv": HOUSE_NAVY,
        "test_transport": HOUSE_CRIMSON,
    }
    markers = {
        "validation_nested_cv": "o",
        "test_transport": "s",
    }
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(10.5, 5.6))
    for split in ("validation_nested_cv", "test_transport"):
        group = selected[selected["split"].eq(split)]
        for _, row in group.iterrows():
            y_value = y_positions[row["label"]] + offsets[split]
            estimate = row["delta_r2"]
            axis.errorbar(
                estimate,
                y_value,
                xerr=np.asarray(
                    [
                        [estimate - row["delta_r2_ci_low"]],
                        [row["delta_r2_ci_high"] - estimate],
                    ]
                ),
                fmt=markers[split],
                color=colors[split],
                ecolor=colors[split],
                capsize=4,
                markersize=7,
                label=split_map[split]
                if row["label"] == y_order[0]
                else None,
            )
            axis.annotate(
                f"{estimate:+.3f}",
                (estimate, y_value),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8.5,
                color=colors[split],
            )
    axis.axvline(0.0, color=HOUSE_GRAY, linestyle="--", linewidth=1.2)
    axis.set_yticks(
        [y_positions[label] for label in y_order], labels=y_order
    )
    axis.set_xlabel(
        "HbA1c incremental $R^2$ beyond conventional glycemic summaries"
    )
    axis.set_title(
        "Targeted HbA1c positive-control predictive transport\n"
        "95% participant-bootstrap confidence intervals",
        loc="left",
    )
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def frozen_projection(
    representations: pd.DataFrame,
    participant_ids: list[str],
    step3_directory: Path,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = representations.loc[
        representations["representation_type"].eq(condition),
        ["participant_id", *RAW_HIDDEN_COLUMNS],
    ].copy()
    selected["participant_id"] = selected["participant_id"].astype(str)
    selected = selected.set_index("participant_id").reindex(participant_ids)
    values = selected.loc[:, RAW_HIDDEN_COLUMNS].to_numpy(float)
    frozen = step3_directory / "frozen_validation_pipeline" / condition
    feature_order = json.loads((frozen / "feature_order.json").read_text())
    kept = np.load(frozen / "kept_dimensions.npy")
    if feature_order["source_dimensions"] != list(RAW_HIDDEN_COLUMNS):
        raise RuntimeError(f"Frozen feature order mismatch for {condition}")
    scaler = joblib.load(frozen / f"{condition}_scaler.joblib")
    pca = joblib.load(frozen / f"{condition}_pca.joblib")
    scores = pca.transform(scaler.transform(values[:, kept]))[:, :2]
    variance = np.asarray(pca.explained_variance_ratio_[:2], dtype=float)
    return scores, variance


def make_manifold_figure(
    test_representations: pd.DataFrame,
    test_frame: pd.DataFrame,
    step3_directory: Path,
    output_path: Path,
) -> None:
    participant_ids = test_frame["participant_id"].astype(str).tolist()
    full_scores, full_variance = frozen_projection(
        test_representations,
        participant_ids,
        step3_directory,
        "full_all",
    )
    neutral_scores, neutral_variance = frozen_projection(
        test_representations,
        participant_ids,
        step3_directory,
        "neutral_all",
    )
    hba1c = pd.to_numeric(
        test_frame[TARGET_COLUMN], errors="coerce"
    ).to_numpy(float)
    observed = np.isfinite(hba1c)
    normalization = Normalize(
        vmin=float(np.nanmin(hba1c)),
        vmax=float(np.nanmax(hba1c)),
    )
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    scatter = None
    for axis, scores, variance, title in (
        (
            axes[0],
            full_scores,
            full_variance,
            "Full-profile test manifold",
        ),
        (
            axes[1],
            neutral_scores,
            neutral_variance,
            "Static-neutral test manifold",
        ),
    ):
        scatter = axis.scatter(
            scores[observed, 0],
            scores[observed, 1],
            c=hba1c[observed],
            cmap="viridis",
            norm=normalization,
            s=28,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.25,
        )
        axis.set_title(f"{title}\nHbA1c available n={observed.sum()}")
        axis.set_xlabel(
            f"PC1 ({100 * variance[0]:.1f}% validation variance)"
        )
        axis.set_ylabel(
            f"PC2 ({100 * variance[1]:.1f}% validation variance)"
        )
    colorbar = figure.colorbar(
        scatter, ax=axes, fraction=0.035, pad=0.04
    )
    colorbar.set_label("HbA1c (%)")
    figure.suptitle(
        "HbA1c across matching frozen validation PCA spaces",
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.5,
        0.02,
        "Full and neutral panels use their matching frozen validation PCA spaces.",
        ha="center",
        color=HOUSE_GRAY,
        fontsize=9.5,
    )
    figure.subplots_adjust(
        left=0.07, right=0.91, bottom=0.16, top=0.81, wspace=0.25
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def scan_em_dash(paths: list[Path]) -> list[str]:
    affected: list[str] = []
    for root in paths:
        candidates = root.rglob("*") if root.is_dir() else [root]
        for path in candidates:
            if path.is_file() and path.suffix.lower() in {
                ".csv", ".json", ".log", ".md", ".py", ".tex"
            }:
                try:
                    if NO_EM_DASH in path.read_text():
                        affected.append(str(path))
                except UnicodeDecodeError:
                    continue
    return affected


def run_hba1c_stage(
    run_directory: Path,
    step2_directory: Path,
    step3_directory: Path,
    step4_directory: Path,
    step5_directory: Path,
    static_table: Path,
    bootstrap_replicates: int = N_BOOTSTRAP,
    permutation_replicates: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
    n_jobs: int = -1,
) -> dict[str, Any]:
    output_directory = run_directory / "hba1c_positive_control"
    manifest_path = run_directory / "step7_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("neighbor_stage", {}).get("status") != "GATE2_COMPLETE":
        raise RuntimeError("Gate 2 is not complete")
    if manifest.get("hba1c_stage", {}).get("status") == "QC_COMPLETE":
        raise RuntimeError("HbA1c stage is already complete")
    expected_partial_names = {
        "frozen_hba1c_models",
        "hba1c_analysis_plan_frozen.json",
        "hba1c_feature_sets.json",
    }
    existing_names = {path.name for path in output_directory.iterdir()}
    if existing_names - expected_partial_names:
        raise RuntimeError(
            f"Unexpected pre-existing HbA1c outputs: {existing_names}"
        )
    validation_ids, test_ids = load_participant_ids(
        step2_directory, step4_directory
    )
    existing_hba1c = False
    for filename in (
        "probe_incremental_value.csv",
        "probe_performance_summary.csv",
    ):
        path = step5_directory / filename
        if path.exists():
            frame = pd.read_csv(path)
            existing_hba1c = existing_hba1c or (
                "target" in frame
                and frame["target"].astype(str).str.contains(
                    "hba1c", case=False
                ).any()
            )
    for filename in (
        "validation_probe_predictions.parquet",
        "test_probe_predictions.parquet",
    ):
        path = step5_directory / filename
        if path.exists():
            frame = pd.read_parquet(path, columns=["target"])
            existing_hba1c = existing_hba1c or frame["target"].astype(
                str
            ).str.contains("hba1c", case=False).any()
    if existing_hba1c:
        raise RuntimeError(
            "HbA1c exists in Step 5 and requires protocol verification before reuse"
        )
    feature_specifications = feature_specs()
    plan = {
        "analysis_role": "targeted positive-control closing analysis",
        "test_role": "targeted positive-control predictive transport",
        "target": TARGET_COLUMN,
        "target_label": TARGET_LABEL,
        "target_unit": TARGET_UNIT,
        "target_is_full_profile_model_input": True,
        "external_biomarkers_are_model_inputs": False,
        "primary_headline":
            "neutral-state HbA1c delta R2 over glycemic summaries",
        "feature_sets": feature_specifications,
        "outer_cv": {
            "folds": OUTER_FOLDS,
            "repetitions": OUTER_REPETITIONS,
            "same_folds_across_feature_sets": True,
        },
        "inner_cv": {"folds": INNER_FOLDS},
        "ridge_alpha_grid": RIDGE_ALPHA_GRID.tolist(),
        "preprocessing": {
            "numeric":
                "fold-fitted median imputation with indicators and scaling",
            "categorical":
                "fold-fitted most-frequent imputation and one-hot encoding",
            "hidden": "finite verification and fold-fitted scaling",
        },
        "bootstrap": {
            "unit": "participant retaining all repeated OOF rows",
            "replicates": bootstrap_replicates,
        },
        "permutation": {
            "feature": "neutral hidden-state rows",
            "replicates": permutation_replicates,
            "method": "Step 5 outer-fold state-row permutation",
            "separate_positive_control_test": True,
        },
        "prohibitions": [
            "no forecasting checkpoint load",
            "no model replay",
            "no hidden-state regeneration",
            "no scaler or PCA fitting on test",
            "no test tuning",
            "no causal learned-versus-injected percentage",
        ],
    }
    write_json(
        output_directory / "hba1c_analysis_plan_frozen.json", plan
    )
    write_json(
        output_directory / "hba1c_feature_sets.json",
        feature_specifications,
    )
    validation_base, test_base, static = build_baseline_frames(
        step3_directory,
        step4_directory,
        static_table,
        validation_ids,
        test_ids,
        PANEL_PATH,
    )
    representations = representation_frames(
        step2_directory,
        step3_directory,
        step4_directory,
        validation_ids,
        test_ids,
    )
    validation = validation_base.loc[
        validation_base[TARGET_COLUMN].notna()
    ].reset_index(drop=True)
    test = test_base.loc[test_base[TARGET_COLUMN].notna()].reset_index(
        drop=True
    )
    if validation["participant_id"].duplicated().any():
        raise RuntimeError("Duplicate validation HbA1c participants")
    if test["participant_id"].duplicated().any():
        raise RuntimeError("Duplicate test HbA1c participants")
    cohort_rows = []
    for split, canonical_ids, frame in (
        ("validation", validation_ids, validation),
        ("test", test_ids, test),
    ):
        cohort_rows.append(
            {
                "split": split,
                "canonical_participants": len(canonical_ids),
                "target_column": TARGET_COLUMN,
                "target_unit": TARGET_UNIT,
                "participants_with_hba1c": len(frame),
                "coverage": len(frame) / len(canonical_ids),
                "missing_hba1c": len(canonical_ids) - len(frame),
                "duplicate_target_rows": int(
                    frame["participant_id"].duplicated().sum()
                ),
                "minimum_hba1c": frame[TARGET_COLUMN].min(),
                "maximum_hba1c": frame[TARGET_COLUMN].max(),
                "median_hba1c": frame[TARGET_COLUMN].median(),
                "participants_with_multiple_source_records": int(
                    frame["hba1c_percent_n_records"].fillna(0).gt(1).sum()
                ),
                "participants_with_measurement_date": int(
                    frame[f"{TARGET_COLUMN}_date"].notna().sum()
                ),
                "participants_with_timing_value": int(
                    frame["hba1c_percent_days_to_cgm_start"].notna().sum()
                ),
                "analysis_role": (
                    "development nested CV"
                    if split == "validation"
                    else "targeted positive-control predictive transport"
                ),
            }
        )
    cohort_audit = pd.DataFrame(cohort_rows)
    cohort_audit.to_csv(
        output_directory / "hba1c_cohort_audit.csv", index=False
    )
    validation_target = validation[TARGET_COLUMN].to_numpy(float)
    outer_split_sets = []
    fold_strategies = []
    for repetition in range(OUTER_REPETITIONS):
        split_set, strategy = outer_folds(
            validation_target,
            validation["clinical_site"],
            OUTER_FOLDS,
            seed + repetition,
        )
        outer_split_sets.append(split_set)
        fold_strategies.append(strategy)
    validation_prediction_rows: list[dict[str, Any]] = []
    selected_hyperparameters: list[dict[str, Any]] = []
    neutral_permutation_cache: dict[tuple[int, int], dict[str, Any]] = {}
    model_directory = output_directory / "frozen_hba1c_models"
    model_directory.mkdir(exist_ok=True)
    for feature_set, specification in feature_specifications.items():
        representation_name = specification["representation"]
        validation_data = merge_representation(
            validation,
            representations["validation"].get(representation_name),
        )
        if set(validation_data["participant_id"]) != set(
            validation["participant_id"]
        ):
            raise RuntimeError(
                f"Validation cohort changed for {feature_set}"
            )
        for repetition, split_set in enumerate(outer_split_sets):
            for fold_index, (train_index, held_out_index) in enumerate(
                split_set
            ):
                pipeline, parameter_grid = estimator(
                    tuple(specification["numeric"]),
                    tuple(specification["categorical"]),
                    tuple(specification["hidden"]),
                )
                inner = KFold(
                    INNER_FOLDS,
                    shuffle=True,
                    random_state=seed + 1000 * repetition + fold_index,
                )
                search = GridSearchCV(
                    pipeline,
                    parameter_grid,
                    cv=inner,
                    scoring="neg_mean_squared_error",
                    n_jobs=n_jobs,
                    refit=True,
                    error_score="raise",
                )
                search.fit(
                    validation_data.iloc[train_index],
                    validation_target[train_index],
                )
                predictions = search.predict(
                    validation_data.iloc[held_out_index]
                )
                alpha = search.best_params_["model__alpha"]
                selected_hyperparameters.append(
                    {
                        "stage": "validation_outer",
                        "feature_set": feature_set,
                        "outer_repetition": repetition,
                        "outer_fold": fold_index,
                        "selected_alpha": alpha,
                        "inner_best_neg_mse": search.best_score_,
                        "fold_strategy": fold_strategies[repetition],
                    }
                )
                for position, prediction in zip(
                    held_out_index, predictions
                ):
                    validation_prediction_rows.append(
                        {
                            "participant_id": validation_data.iloc[
                                position
                            ]["participant_id"],
                            "target": TARGET_LABEL,
                            "target_column": TARGET_COLUMN,
                            "outer_repetition": repetition,
                            "outer_fold": fold_index,
                            "feature_set": feature_set,
                            "observed": validation_target[position],
                            "predicted": float(prediction),
                            "selected_alpha": alpha,
                            "model_status": "nested_validation_held_out",
                        }
                    )
                if feature_set == "glycemic_plus_neutral_state":
                    baseline_preprocessor = preprocessing_pipeline(
                        GLYCEMIC_NUMERIC, (), ()
                    )
                    baseline_train = baseline_preprocessor.fit_transform(
                        validation_data.iloc[train_index]
                    )
                    baseline_held_out = baseline_preprocessor.transform(
                        validation_data.iloc[held_out_index]
                    )
                    hidden_scaler = StandardScaler()
                    hidden_train = hidden_scaler.fit_transform(
                        validation_data.iloc[train_index].loc[
                            :, HIDDEN_COLUMNS
                        ]
                    )
                    hidden_held_out = hidden_scaler.transform(
                        validation_data.iloc[held_out_index].loc[
                            :, HIDDEN_COLUMNS
                        ]
                    )
                    strata = (
                        validation_data.iloc[train_index][
                            "clinical_site"
                        ]
                        .fillna("<missing>")
                        .astype(str)
                        + "|"
                        + validation_data.iloc[train_index][
                            "study_group"
                        ]
                        .fillna("<missing>")
                        .astype(str)
                    ).to_numpy()
                    groups = [
                        np.flatnonzero(strata == level)
                        for level in np.unique(strata)
                    ]
                    permutation_strategy = (
                        "within_clinical_site_x_study_group"
                        if min(map(len, groups)) >= 2
                        else "global_fallback"
                    )
                    neutral_permutation_cache[
                        (repetition, fold_index)
                    ] = {
                        "baseline_train": baseline_train,
                        "baseline_held_out": baseline_held_out,
                        "hidden_train": hidden_train,
                        "hidden_held_out": hidden_held_out,
                        "target_train": validation_target[train_index],
                        "target_held_out": validation_target[
                            held_out_index
                        ],
                        "alpha": alpha,
                        "groups": groups,
                        "strategy": permutation_strategy,
                        "held_out_ids": validation_data.iloc[
                            held_out_index
                        ]["participant_id"].to_numpy(),
                    }
    validation_predictions = pd.DataFrame(validation_prediction_rows)
    atomic_parquet(
        validation_predictions,
        output_directory / "hba1c_validation_predictions.parquet",
    )
    test_prediction_rows: list[dict[str, Any]] = []
    for feature_set, specification in feature_specifications.items():
        representation_name = specification["representation"]
        validation_data = merge_representation(
            validation,
            representations["validation"].get(representation_name),
        )
        test_data = merge_representation(
            test,
            representations["test"].get(representation_name),
        )
        pipeline, parameter_grid = estimator(
            tuple(specification["numeric"]),
            tuple(specification["categorical"]),
            tuple(specification["hidden"]),
        )
        inner = KFold(
            INNER_FOLDS, shuffle=True, random_state=seed + 9000
        )
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            cv=inner,
            scoring="neg_mean_squared_error",
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(validation_data, validation_data[TARGET_COLUMN])
        model = search.best_estimator_
        joblib.dump(model, model_directory / f"{feature_set}.joblib")
        predictions = model.predict(test_data)
        selected_hyperparameters.append(
            {
                "stage": "final_validation_fit",
                "feature_set": feature_set,
                "outer_repetition": np.nan,
                "outer_fold": np.nan,
                "selected_alpha": search.best_params_["model__alpha"],
                "inner_best_neg_mse": search.best_score_,
                "fold_strategy": "validation_only_inner_cv",
            }
        )
        for row_index, prediction in enumerate(predictions):
            test_prediction_rows.append(
                {
                    "participant_id": test_data.iloc[row_index][
                        "participant_id"
                    ],
                    "target": TARGET_LABEL,
                    "target_column": TARGET_COLUMN,
                    "outer_repetition": np.nan,
                    "outer_fold": np.nan,
                    "feature_set": feature_set,
                    "observed": test_data.iloc[row_index][TARGET_COLUMN],
                    "predicted": float(prediction),
                    "selected_alpha": search.best_params_["model__alpha"],
                    "model_status":
                        "frozen_validation_pipeline_targeted_test_transport",
                }
            )
    test_predictions = pd.DataFrame(test_prediction_rows)
    atomic_parquet(
        test_predictions,
        output_directory / "hba1c_test_predictions.parquet",
    )
    pd.DataFrame(selected_hyperparameters).to_csv(
        output_directory / "hba1c_model_hyperparameters.csv",
        index=False,
    )
    performance = performance_table(
        validation_predictions, test_predictions
    )
    performance.to_csv(
        output_directory / "hba1c_probe_performance.csv", index=False
    )
    incremental = incremental_table(
        validation_predictions,
        test_predictions,
        bootstrap_replicates,
        seed,
    )
    baseline_validation = validation_predictions[
        validation_predictions["feature_set"].eq("glycemic_baseline")
    ][["participant_id", "outer_repetition", "predicted"]]

    def one_permutation(permutation_index: int) -> tuple[float, str]:
        rng = np.random.default_rng(
            seed + 50000 + permutation_index
        )
        rows = []
        strategies = []
        for repetition in range(OUTER_REPETITIONS):
            for fold_index in range(OUTER_FOLDS):
                cached = neutral_permutation_cache[
                    (repetition, fold_index)
                ]
                order = np.arange(len(cached["hidden_train"]))
                strategies.append(cached["strategy"])
                if cached["strategy"].startswith("within"):
                    for group in cached["groups"]:
                        order[group] = rng.permutation(group)
                else:
                    order = rng.permutation(order)
                model = Ridge(
                    alpha=cached["alpha"], solver="lsqr"
                ).fit(
                    np.hstack(
                        [
                            cached["baseline_train"],
                            cached["hidden_train"][order],
                        ]
                    ),
                    cached["target_train"],
                )
                prediction = model.predict(
                    np.hstack(
                        [
                            cached["baseline_held_out"],
                            cached["hidden_held_out"],
                        ]
                    )
                )
                rows.extend(
                    (
                        participant_id,
                        repetition,
                        observed,
                        predicted,
                    )
                    for participant_id, observed, predicted in zip(
                        cached["held_out_ids"],
                        cached["target_held_out"],
                        prediction,
                    )
                )
        permuted = pd.DataFrame(
            rows,
            columns=[
                "participant_id",
                "outer_repetition",
                "observed",
                "predicted_augmented",
            ],
        ).merge(
            baseline_validation,
            on=["participant_id", "outer_repetition"],
            validate="one_to_one",
        )
        delta = r2_score(
            permuted["observed"], permuted["predicted_augmented"]
        ) - r2_score(permuted["observed"], permuted["predicted"])
        return float(delta), "+".join(sorted(set(strategies)))

    null_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(one_permutation)(index)
        for index in range(permutation_replicates)
    )
    null_values = np.asarray([value for value, _ in null_results])
    observed_delta = incremental.loc[
        incremental["split"].eq("validation_nested_cv")
        & incremental["feature_set"].eq(
            "glycemic_plus_neutral_state"
        ),
        "delta_r2",
    ].iloc[0]
    empirical_p = float(
        (1 + np.sum(null_values >= observed_delta))
        / (1 + len(null_values))
    )
    permutation = pd.DataFrame(
        [
            {
                "target": TARGET_LABEL,
                "feature_set": "glycemic_plus_neutral_state",
                "reference_feature_set": "glycemic_baseline",
                "observed_delta_r2": observed_delta,
                "null_mean": null_values.mean(),
                "null_sd": null_values.std(ddof=1),
                "null_q025": np.quantile(null_values, 0.025),
                "null_q975": np.quantile(null_values, 0.975),
                "empirical_p_value": empirical_p,
                "permutation_strategy":
                    "all five outer repetitions; "
                    + ";".join(
                        sorted(
                            set(strategy for _, strategy in null_results)
                        )
                    ),
                "permutation_replicates": len(null_values),
                "separate_hba1c_positive_control_test": True,
                "null_values_json": json.dumps(null_values.tolist()),
            }
        ]
    )
    permutation.to_csv(
        output_directory / "hba1c_permutation_test.csv", index=False
    )
    incremental["permutation_p"] = np.nan
    incremental.loc[
        incremental["split"].eq("validation_nested_cv")
        & incremental["feature_set"].eq(
            "glycemic_plus_neutral_state"
        ),
        "permutation_p",
    ] = empirical_p
    incremental.to_csv(
        output_directory / "hba1c_incremental_value.csv", index=False
    )
    summary_rows = []
    for split in ("validation_nested_cv", "test_transport"):
        performance_split = performance[performance["split"].eq(split)]
        increments_split = incremental[incremental["split"].eq(split)]

        def performance_row(feature_set: str) -> pd.Series:
            return performance_split[
                performance_split["feature_set"].eq(feature_set)
            ].iloc[0]

        def increment_row(feature_set: str) -> pd.Series:
            return increments_split[
                increments_split["feature_set"].eq(feature_set)
            ].iloc[0]

        baseline_row = performance_row("glycemic_baseline")
        full_only_row = performance_row("full_state_only")
        neutral_only_row = performance_row("neutral_state_only")
        full_increment = increment_row("glycemic_plus_full_state")
        neutral_increment = increment_row(
            "glycemic_plus_neutral_state"
        )
        summary_rows.append(
            {
                "split": split,
                "analysis_role": (
                    "development nested CV"
                    if split == "validation_nested_cv"
                    else "targeted positive-control predictive transport"
                ),
                "n_participants": baseline_row["n_participants"],
                "glycemic_baseline_r2": baseline_row["r2"],
                "full_state_only_r2": full_only_row["r2"],
                "neutral_state_only_r2": neutral_only_row["r2"],
                "glycemic_plus_full_r2": full_increment["augmented_r2"],
                "full_delta_r2": full_increment["delta_r2"],
                "full_delta_r2_ci_low":
                    full_increment["delta_r2_ci_low"],
                "full_delta_r2_ci_high":
                    full_increment["delta_r2_ci_high"],
                "glycemic_plus_neutral_r2":
                    neutral_increment["augmented_r2"],
                "neutral_delta_r2": neutral_increment["delta_r2"],
                "neutral_delta_r2_ci_low":
                    neutral_increment["delta_r2_ci_low"],
                "neutral_delta_r2_ci_high":
                    neutral_increment["delta_r2_ci_high"],
                "neutral_permutation_p": (
                    empirical_p
                    if split == "validation_nested_cv"
                    else np.nan
                ),
                "full_minus_neutral_delta_r2":
                    full_increment["delta_r2"]
                    - neutral_increment["delta_r2"],
                "causal_decomposition_permitted": False,
            }
        )
    full_neutral_summary = pd.DataFrame(summary_rows)
    full_neutral_summary.to_csv(
        output_directory / "hba1c_full_vs_neutral_summary.csv",
        index=False,
    )
    validation_summary = full_neutral_summary[
        full_neutral_summary["split"].eq("validation_nested_cv")
    ].iloc[0]
    test_summary = full_neutral_summary[
        full_neutral_summary["split"].eq("test_transport")
    ].iloc[0]
    direction_agreement = bool(
        np.sign(validation_summary["neutral_delta_r2"])
        == np.sign(test_summary["neutral_delta_r2"])
    )
    transported = bool(
        validation_summary["neutral_delta_r2"] > 0
        and test_summary["neutral_delta_r2"] > 0
        and validation_summary["neutral_delta_r2_ci_low"] > 0
        and test_summary["neutral_delta_r2_ci_low"] > 0
    )
    if transported:
        interpretation = (
            "HbA1c-related information remained recoverable after "
            "participant-specific static conditioning was replaced, "
            "indicating that CGM and wearable dynamics reconstructed part "
            "of the long-term glycemic phenotype beyond conventional "
            "glucose summaries."
        )
    elif (
        validation_summary["full_delta_r2"]
        > validation_summary["neutral_delta_r2"]
        and test_summary["full_delta_r2"]
        > test_summary["neutral_delta_r2"]
    ):
        interpretation = (
            "Strong HbA1c recovery in the full-profile state was largely "
            "attributable to direct static conditioning, while limited "
            "incremental signal remained after neutralization."
        )
    else:
        interpretation = (
            "HbA1c remained visually associated with the manifold, but the "
            "static-neutralized representation did not improve prediction "
            "beyond conventional glycemic summaries."
        )
    report_lines = [
        "# Targeted HbA1c positive-control closing analysis",
        "",
        "## Objective",
        "",
        "Estimate how much HbA1c-related information remains recoverable "
        "after participant-specific static conditioning is replaced by the "
        "common reference profile.",
        "",
        "## Input clarification",
        "",
        "HbA1c was supplied directly to the full-profile forecasting model. "
        "It is therefore a direct-input positive control. The full-versus-"
        "neutral contrast is not an exact causal decomposition into learned "
        "and injected percentages.",
        "",
        "## Cohort",
        "",
        f"Validation included {len(validation)} participants with HbA1c; "
        f"targeted test transport included {len(test)} participants.",
        "",
        "## Protocol",
        "",
        "The Step 5 Ridge protocol was reused with five deterministic "
        "repetitions of nested 5 by 5 participant-level validation CV. "
        "Preprocessing was fitted inside each training fold. Final models "
        "were selected and fitted on validation only and applied once to "
        "test.",
        "",
        "## Primary results",
        "",
        f"Validation glycemic baseline R2 was "
        f"{validation_summary['glycemic_baseline_r2']:.4f}. Adding the "
        f"full-profile state changed R2 by "
        f"{validation_summary['full_delta_r2']:+.4f} "
        f"[{validation_summary['full_delta_r2_ci_low']:+.4f}, "
        f"{validation_summary['full_delta_r2_ci_high']:+.4f}]. Adding the "
        f"static-neutral state changed R2 by "
        f"{validation_summary['neutral_delta_r2']:+.4f} "
        f"[{validation_summary['neutral_delta_r2_ci_low']:+.4f}, "
        f"{validation_summary['neutral_delta_r2_ci_high']:+.4f}], with "
        f"positive-control permutation p={empirical_p:.4f}.",
        "",
        f"On targeted test transport, glycemic baseline R2 was "
        f"{test_summary['glycemic_baseline_r2']:.4f}; full-state delta R2 "
        f"was {test_summary['full_delta_r2']:+.4f} "
        f"[{test_summary['full_delta_r2_ci_low']:+.4f}, "
        f"{test_summary['full_delta_r2_ci_high']:+.4f}], and neutral-state "
        f"delta R2 was {test_summary['neutral_delta_r2']:+.4f} "
        f"[{test_summary['neutral_delta_r2_ci_low']:+.4f}, "
        f"{test_summary['neutral_delta_r2_ci_high']:+.4f}].",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "The test analysis is targeted positive-control predictive "
        "transport, not untouched confirmation.",
    ]
    report_path = output_directory / "hba1c_positive_control_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    incremental_figure = (
        output_directory / "figure_hba1c_incremental_value.png"
    )
    make_incremental_figure(incremental, incremental_figure)
    test_representations = pd.read_parquet(
        step4_directory / "test_participant_representations.parquet"
    )
    manifold_figure = (
        output_directory / "figure_hba1c_full_vs_neutral_manifold.png"
    )
    make_manifold_figure(
        test_representations,
        test,
        step3_directory,
        manifold_figure,
    )
    required_outputs = [
        output_directory / "hba1c_cohort_audit.csv",
        output_directory / "hba1c_feature_sets.json",
        output_directory / "hba1c_validation_predictions.parquet",
        output_directory / "hba1c_test_predictions.parquet",
        output_directory / "hba1c_probe_performance.csv",
        output_directory / "hba1c_incremental_value.csv",
        output_directory / "hba1c_permutation_test.csv",
        output_directory / "hba1c_full_vs_neutral_summary.csv",
        output_directory / "hba1c_positive_control_report.md",
        incremental_figure,
        manifold_figure,
    ]
    if any(not path.exists() for path in required_outputs):
        raise RuntimeError("Required HbA1c output is missing")
    if set(validation_predictions["model_status"]) != {
        "nested_validation_held_out"
    }:
        raise RuntimeError("Validation predictions are not all held out")
    if set(test_predictions["model_status"]) != {
        "frozen_validation_pipeline_targeted_test_transport"
    }:
        raise RuntimeError("Test transport model status is invalid")
    em_dash_files = scan_em_dash([Path(__file__), output_directory])
    if em_dash_files:
        raise RuntimeError(
            "Forbidden Unicode U+2014 found: " + ", ".join(em_dash_files)
        )
    warnings = []
    if not transported:
        warnings.append(
            "Neutral HbA1c incremental value did not meet the transported "
            "positive-control criterion."
        )
    manifest["hba1c_stage"] = {
        "status": "QC_COMPLETE",
        "analysis_role": "targeted positive-control closing analysis",
        "validation_participants": len(validation),
        "test_participants": len(test),
        "protocol": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "outer_repetitions": OUTER_REPETITIONS,
            "bootstrap_replicates": bootstrap_replicates,
            "permutation_replicates": permutation_replicates,
            "seed": seed,
            "ridge_alpha_grid": RIDGE_ALPHA_GRID.tolist(),
        },
        "headline": {
            "validation_baseline_r2":
                validation_summary["glycemic_baseline_r2"],
            "validation_full_delta_r2":
                validation_summary["full_delta_r2"],
            "validation_full_ci": [
                validation_summary["full_delta_r2_ci_low"],
                validation_summary["full_delta_r2_ci_high"],
            ],
            "validation_neutral_delta_r2":
                validation_summary["neutral_delta_r2"],
            "validation_neutral_ci": [
                validation_summary["neutral_delta_r2_ci_low"],
                validation_summary["neutral_delta_r2_ci_high"],
            ],
            "test_baseline_r2": test_summary["glycemic_baseline_r2"],
            "test_full_delta_r2": test_summary["full_delta_r2"],
            "test_full_ci": [
                test_summary["full_delta_r2_ci_low"],
                test_summary["full_delta_r2_ci_high"],
            ],
            "test_neutral_delta_r2":
                test_summary["neutral_delta_r2"],
            "test_neutral_ci": [
                test_summary["neutral_delta_r2_ci_low"],
                test_summary["neutral_delta_r2_ci_high"],
            ],
            "neutral_validation_test_direction_agreement":
                direction_agreement,
            "neutral_transport_criterion_met": transported,
            "permutation_p": empirical_p,
        },
        "interpretation": interpretation,
        "test_called_untouched_confirmation": False,
        "causal_full_neutral_percentage_reported": False,
        "forecasting_checkpoint_loaded": False,
        "model_replay_run": False,
        "output_paths": {
            path.name: str(path) for path in required_outputs
        },
        "output_hashes": {
            path.name: sha256_file(path) for path in required_outputs
        },
        "warnings": warnings,
        "blockers": [],
    }
    write_json(manifest_path, manifest)
    with (run_directory / "step7_run.log").open("a") as handle:
        handle.write("STEP 7 HbA1c positive-control stage completed\n")
        handle.write(
            f"Validation HbA1c participants: {len(validation)}\n"
        )
        handle.write(f"Test HbA1c participants: {len(test)}\n")
        handle.write("No forecasting checkpoint or model replay used\n")
    return {
        "output_directory": str(output_directory),
        "validation_participants": len(validation),
        "test_participants": len(test),
        "validation_baseline_r2":
            validation_summary["glycemic_baseline_r2"],
        "validation_full_delta_r2":
            validation_summary["full_delta_r2"],
        "validation_full_ci": [
            validation_summary["full_delta_r2_ci_low"],
            validation_summary["full_delta_r2_ci_high"],
        ],
        "validation_neutral_delta_r2":
            validation_summary["neutral_delta_r2"],
        "validation_neutral_ci": [
            validation_summary["neutral_delta_r2_ci_low"],
            validation_summary["neutral_delta_r2_ci_high"],
        ],
        "test_baseline_r2": test_summary["glycemic_baseline_r2"],
        "test_full_delta_r2": test_summary["full_delta_r2"],
        "test_full_ci": [
            test_summary["full_delta_r2_ci_low"],
            test_summary["full_delta_r2_ci_high"],
        ],
        "test_neutral_delta_r2": test_summary["neutral_delta_r2"],
        "test_neutral_ci": [
            test_summary["neutral_delta_r2_ci_low"],
            test_summary["neutral_delta_r2_ci_high"],
        ],
        "permutation_p": empirical_p,
        "transported": transported,
        "interpretation": interpretation,
        "warnings": warnings,
        "blockers": [],
    }
