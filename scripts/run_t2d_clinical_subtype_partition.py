#!/usr/bin/env python3
"""Build the validation-fitted T2D clinical subtype partition."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


STATIC_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "participant_static_features.parquet"
)
FINAL_MULTIMODAL_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "final_multimodal_dataset_20260515_184339.parquet"
)
SPLIT_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/"
    "experiment_c_split_adapt6h_seed42/split_participants.csv"
)
VALIDATION_FEATURES_PATH = Path(
    "/home/myriamcharfeddine/CGM/SSM-CGM/outputs/"
    "hidden_state_phenotype/step3_validation_clustering/"
    "20260725T001123Z/validation_glycemic_nuisance_features.parquet"
)
TEST_FEATURES_PATH = Path(
    "/home/myriamcharfeddine/CGM/SSM-CGM/outputs/"
    "hidden_state_phenotype/step4_test_confirmation/"
    "20260725T010440Z/test_glycemic_nuisance_features.parquet"
)
OUTPUT_DIRECTORY = Path(
    "/home/myriamcharfeddine/CGM/SSM-CGM/subtype_partition"
)
MODEL_DIRECTORY = OUTPUT_DIRECTORY / "frozen_models"
LABELS_PATH = OUTPUT_DIRECTORY / "subtype_labels.parquet"
PROFILES_PATH = OUTPUT_DIRECTORY / "cluster_profiles.csv"
STABILITY_PATH = OUTPUT_DIRECTORY / "subtype_cluster_stability.csv"
MANIFEST_PATH = OUTPUT_DIRECTORY / "subtype_manifest.json"
FIGURE_PATH = OUTPUT_DIRECTORY / "fig2_subtype_cluster_profiles.png"

PARTICIPANT_COLUMN = "participant_id"
STUDY_GROUP_COLUMN = "participants_study_group"
AGE_COLUMN = "participants_age"
BMI_COLUMN = "bmi_baseline"
C_PEPTIDE_COLUMN = "c_peptide_ngml_baseline"
TRIGLYCERIDES_COLUMN = "triglycerides_mgdl_baseline"
HDL_COLUMN = "hdl_cholesterol_mgdl_baseline"
HBA1C_COLUMN = "hba1c_percent_baseline"
TG_HDL_COLUMN = "tg_hdl"
CLUSTER_FEATURES = [
    BMI_COLUMN,
    TG_HDL_COLUMN,
    C_PEPTIDE_COLUMN,
    AGE_COLUMN,
]
PROFILE_METRICS = [
    BMI_COLUMN,
    TG_HDL_COLUMN,
    C_PEPTIDE_COLUMN,
    AGE_COLUMN,
    HBA1C_COLUMN,
    "mean_glucose",
]
T2D_STUDY_GROUPS = (
    "insulin_dependent",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
)
K_PRIMARY = 3
K_SENSITIVITY = 4
K_VALUES = (K_PRIMARY, K_SENSITIVITY)
N_INIT = 50
RANDOM_SEED = 42
BOOTSTRAP_REPLICATES = 2000
STABILITY_REPLICATES = 200
STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
NONFASTING_CAVEAT = (
    "C-peptide and triglycerides were not confirmed fasting measurements. "
    "TG/HDL and subtype profiles require this non-fasting caveat."
)
AGE_DEFINITION = (
    "Direct integer participant age at study visit from participants_age in "
    "the final enriched multimodal dataset. It is not age at diagnosis and "
    "was not imputed."
)
K4_WARNING = (
    "The k=4 validation solution isolates a one-participant extreme TG/HDL "
    "cluster rather than an age-related split. It is unstable, descriptive "
    "only, and must not be interpreted as a fourth clinical subtype."
)
NO_EM_DASH = "\u2014"


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


def write_labels(path: Path, frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"nonfasting_caveat"] = NONFASTING_CAVEAT.encode()
    metadata[b"age_definition"] = AGE_DEFINITION.encode()
    metadata[b"validation_test_rule"] = (
        b"Scaler and k-means fit on validation only; test assigned without refit."
    )
    table = table.replace_schema_metadata(metadata)
    temporary = path.with_suffix(".tmp.parquet")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="little"
    )
    return int((value + RANDOM_SEED) % (2**32 - 1))


def load_final_age(participant_ids: set[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(FINAL_MULTIMODAL_PATH)
    for batch in parquet.iter_batches(
        columns=[
            PARTICIPANT_COLUMN,
            AGE_COLUMN,
            "participants_study_visit_date",
        ],
        batch_size=500000,
    ):
        frame = batch.to_pandas()
        frame[PARTICIPANT_COLUMN] = frame[PARTICIPANT_COLUMN].astype(str)
        selected = frame[frame[PARTICIPANT_COLUMN].isin(participant_ids)]
        if len(selected):
            rows.append(selected.drop_duplicates(PARTICIPANT_COLUMN))
    output = pd.concat(rows, ignore_index=True).drop_duplicates(
        PARTICIPANT_COLUMN
    )
    if set(output[PARTICIPANT_COLUMN]) != participant_ids:
        raise RuntimeError("Final multimodal age cohort mismatch")
    return output


def build_complete_case_frame() -> pd.DataFrame:
    static_columns = [
        PARTICIPANT_COLUMN,
        STUDY_GROUP_COLUMN,
        AGE_COLUMN,
        BMI_COLUMN,
        C_PEPTIDE_COLUMN,
        TRIGLYCERIDES_COLUMN,
        HDL_COLUMN,
        HBA1C_COLUMN,
    ]
    static = pd.read_parquet(STATIC_PATH, columns=static_columns)
    static[PARTICIPANT_COLUMN] = static[PARTICIPANT_COLUMN].astype(str)
    split = pd.read_csv(SPLIT_PATH, dtype={PARTICIPANT_COLUMN: str})
    split["split"] = split["split"].replace({"val": "validation"})
    split = split[split["split"].isin(["validation", "test"])]
    if static[PARTICIPANT_COLUMN].duplicated().any():
        raise RuntimeError("Duplicate participant in static table")
    if split[PARTICIPANT_COLUMN].duplicated().any():
        raise RuntimeError("Duplicate participant in split table")
    age = load_final_age(set(split[PARTICIPANT_COLUMN]))
    age = age.rename(columns={AGE_COLUMN: "final_multimodal_age"})
    frame = split.merge(
        static, on=PARTICIPANT_COLUMN, validate="one_to_one"
    ).merge(
        age, on=PARTICIPANT_COLUMN, validate="one_to_one"
    )
    if not np.array_equal(
        frame[AGE_COLUMN].to_numpy(),
        frame["final_multimodal_age"].to_numpy(),
    ):
        raise RuntimeError("Static and final multimodal age differ")
    frame[AGE_COLUMN] = frame["final_multimodal_age"]
    frame = frame[frame[STUDY_GROUP_COLUMN].isin(T2D_STUDY_GROUPS)].copy()
    frame[TG_HDL_COLUMN] = (
        frame[TRIGLYCERIDES_COLUMN] / frame[HDL_COLUMN]
    )
    frame.loc[
        frame[HDL_COLUMN].isna() | (frame[HDL_COLUMN] <= 0),
        TG_HDL_COLUMN,
    ] = np.nan
    frame = frame[frame[CLUSTER_FEATURES].notna().all(axis=1)].copy()
    expected = {"validation": 91, "test": 83}
    observed = frame.groupby("split")[PARTICIPANT_COLUMN].nunique().to_dict()
    if observed != expected:
        raise RuntimeError(
            f"Complete-case cohort changed: {observed}, expected {expected}"
        )
    validation_features = pd.read_parquet(
        VALIDATION_FEATURES_PATH,
        columns=[PARTICIPANT_COLUMN, "mean_glucose"],
    )
    test_features = pd.read_parquet(
        TEST_FEATURES_PATH,
        columns=[PARTICIPANT_COLUMN, "mean_glucose"],
    )
    glycemic = pd.concat(
        [
            validation_features.assign(split="validation"),
            test_features.assign(split="test"),
        ],
        ignore_index=True,
    )
    glycemic[PARTICIPANT_COLUMN] = glycemic[PARTICIPANT_COLUMN].astype(str)
    frame = frame.merge(
        glycemic,
        on=[PARTICIPANT_COLUMN, "split"],
        validate="one_to_one",
    )
    return frame


def align_centroids(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[int, int]:
    cost = np.linalg.norm(
        reference[:, None, :] - candidate[None, :, :], axis=2
    )
    reference_index, candidate_index = linear_sum_assignment(cost)
    return {
        int(candidate_value): int(reference_value)
        for reference_value, candidate_value in zip(
            reference_index, candidate_index
        )
    }


def bootstrap_stability(
    values: np.ndarray,
    reference_labels: np.ndarray,
    reference_centroids: np.ndarray,
    k: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(stable_seed("subtype", k, "stability"))
    rows: list[dict[str, Any]] = []
    for replicate in range(STABILITY_REPLICATES):
        indices = rng.integers(0, len(values), size=len(values))
        model = KMeans(
            n_clusters=k,
            n_init=N_INIT,
            random_state=stable_seed(k, replicate, "kmeans"),
            algorithm="lloyd",
        ).fit(values[indices])
        mapping = align_centroids(reference_centroids, model.cluster_centers_)
        predicted = np.asarray(
            [mapping[int(value)] for value in model.predict(values)]
        )
        rows.append(
            {
                "k": k,
                "replicate": replicate,
                "adjusted_rand_index": adjusted_rand_score(
                    reference_labels, predicted
                ),
                "minimum_bootstrap_cluster_size": int(
                    pd.Series(model.labels_).value_counts().min()
                ),
                "nonfasting_caveat": NONFASTING_CAVEAT,
                "age_definition": AGE_DEFINITION,
                "k4_warning": K4_WARNING if k == K_SENSITIVITY else "",
            }
        )
    return pd.DataFrame(rows)


def interpretive_labels(
    k: int, centers: pd.DataFrame, validation_counts: pd.Series
) -> dict[int, str]:
    if k == K_PRIMARY:
        insulin_resistant = int(
            (
                centers[BMI_COLUMN]
                + centers[TG_HDL_COLUMN]
                + centers[C_PEPTIDE_COLUMN]
            ).idxmax()
        )
        remaining = [index for index in range(k) if index != insulin_resistant]
        insulin_deficient = int(
            centers.loc[remaining, C_PEPTIDE_COLUMN].idxmin()
        )
        overlapping = next(
            index
            for index in range(k)
            if index not in {insulin_resistant, insulin_deficient}
        )
        return {
            insulin_deficient: "insulin-deficient profile",
            insulin_resistant: "insulin-resistant profile",
            overlapping: "overlapping obesity-dominant profile",
        }
    singleton = int(validation_counts.idxmin())
    labels: dict[int, str] = {
        singleton: "unlabeled extreme TG/HDL singleton"
    }
    remaining = [index for index in range(k) if index != singleton]
    insulin_resistant = int(
        (
            centers.loc[remaining, BMI_COLUMN]
            + centers.loc[remaining, TG_HDL_COLUMN]
            + centers.loc[remaining, C_PEPTIDE_COLUMN]
        ).idxmax()
    )
    labels[insulin_resistant] = "insulin-resistant profile"
    remaining = [
        index
        for index in remaining
        if index != insulin_resistant
    ]
    insulin_deficient = int(
        centers.loc[remaining, C_PEPTIDE_COLUMN].idxmin()
    )
    labels[insulin_deficient] = "insulin-deficient profile"
    for index in remaining:
        if index != insulin_deficient:
            labels[index] = "overlapping obesity-dominant profile"
    return labels


def bootstrap_summary(
    values: np.ndarray, split: str, k: int, cluster_id: int, metric: str
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(
        stable_seed(split, k, cluster_id, metric, "bootstrap")
    )
    indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    estimates = values[indices].mean(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def build_profiles(
    frame: pd.DataFrame,
    label_maps: dict[int, dict[int, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for k in K_VALUES:
            cluster_column = f"cluster_k{k}"
            for cluster_id, group in frame[frame["split"] == split].groupby(
                cluster_column
            ):
                for metric in PROFILE_METRICS:
                    values = pd.to_numeric(group[metric], errors="coerce")
                    finite = values[np.isfinite(values)]
                    ci_low, ci_high = bootstrap_summary(
                        finite.to_numpy(),
                        split,
                        k,
                        int(cluster_id),
                        metric,
                    )
                    rows.append(
                        {
                            "split": split,
                            "k": k,
                            "cluster_id": int(cluster_id),
                            "subtype_label": label_maps[k][int(cluster_id)],
                            "metric": metric,
                            "category": "",
                            "n_cluster": len(group),
                            "n_nonmissing": len(finite),
                            "mean": finite.mean(),
                            "sd": finite.std(ddof=1),
                            "median": finite.median(),
                            "bootstrap_ci_low": ci_low,
                            "bootstrap_ci_high": ci_high,
                            "proportion": np.nan,
                            "nonfasting_caveat": NONFASTING_CAVEAT,
                            "age_definition": AGE_DEFINITION,
                            "k4_warning": (
                                K4_WARNING if k == K_SENSITIVITY else ""
                            ),
                        }
                    )
                for category, count in group[
                    STUDY_GROUP_COLUMN
                ].value_counts().items():
                    rows.append(
                        {
                            "split": split,
                            "k": k,
                            "cluster_id": int(cluster_id),
                            "subtype_label": label_maps[k][int(cluster_id)],
                            "metric": "study_group_composition",
                            "category": category,
                            "n_cluster": len(group),
                            "n_nonmissing": len(group),
                            "mean": np.nan,
                            "sd": np.nan,
                            "median": np.nan,
                            "bootstrap_ci_low": np.nan,
                            "bootstrap_ci_high": np.nan,
                            "proportion": count / len(group),
                            "nonfasting_caveat": NONFASTING_CAVEAT,
                            "age_definition": AGE_DEFINITION,
                            "k4_warning": (
                                K4_WARNING if k == K_SENSITIVITY else ""
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def make_figure(frame: pd.DataFrame, label_maps: dict[int, dict[int, str]]) -> None:
    metric_labels = {
        BMI_COLUMN: "BMI",
        TG_HDL_COLUMN: "TG/HDL",
        C_PEPTIDE_COLUMN: "C-peptide (ng/mL)",
        AGE_COLUMN: "Direct age (years)",
    }
    figure, axes = plt.subplots(2, 4, figsize=(20, 10))
    palette = STRATUM_COLORS[:4]
    for row_index, k in enumerate(K_VALUES):
        cluster_column = f"cluster_k{k}"
        plot_frame = frame.copy()
        plot_frame["cluster_label"] = plot_frame[cluster_column].map(
            {
                cluster_id: f"C{cluster_id}: {label}"
                for cluster_id, label in label_maps[k].items()
            }
        )
        order = [
            f"C{cluster_id}: {label_maps[k][cluster_id]}"
            for cluster_id in sorted(label_maps[k])
        ]
        for column_index, metric in enumerate(CLUSTER_FEATURES):
            axis = axes[row_index, column_index]
            sns.boxplot(
                data=plot_frame,
                x="cluster_label",
                y=metric,
                order=order,
                palette=palette[:k],
                showfliers=True,
                ax=axis,
            )
            axis.set_title(f"k={k}: {metric_labels[metric]}")
            axis.set_xlabel("")
            axis.set_ylabel(metric_labels[metric])
            axis.tick_params(axis="x", rotation=28, labelsize=8)
    figure.suptitle(
        "T2D clinical partition profiles\n"
        "C-peptide and triglycerides are not confirmed fasting; direct age "
        "is study-visit age, not diagnosis age.",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            "k=4 isolates an extreme TG/HDL singleton rather than an "
            "age-related group and is sensitivity-only."
        ),
        ha="center",
        fontsize=10,
        color=STRATUM_COLORS[0],
    )
    figure.tight_layout(rect=[0, 0.04, 1, 0.94])
    figure.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def scan_em_dash(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        if path.suffix.lower() in {".py", ".csv", ".json", ".md", ".txt"}:
            if NO_EM_DASH in path.read_text(errors="ignore"):
                failures.append(str(path))
    return failures


def main() -> None:
    started = datetime.now(timezone.utc)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    frame = build_complete_case_frame()
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()
    scaler = StandardScaler().fit(validation[CLUSTER_FEATURES])
    validation_values = scaler.transform(validation[CLUSTER_FEATURES])
    test_values = scaler.transform(test[CLUSTER_FEATURES])
    joblib.dump(scaler, MODEL_DIRECTORY / "t2d_validation_scaler.joblib")
    label_maps: dict[int, dict[int, str]] = {}
    stability_frames: list[pd.DataFrame] = []
    model_metadata: dict[str, Any] = {}
    for k in K_VALUES:
        model = KMeans(
            n_clusters=k,
            n_init=N_INIT,
            random_state=RANDOM_SEED,
            algorithm="lloyd",
        ).fit(validation_values)
        validation[f"cluster_k{k}"] = model.labels_
        test[f"cluster_k{k}"] = model.predict(test_values)
        counts = validation[f"cluster_k{k}"].value_counts().sort_index()
        centers = pd.DataFrame(
            model.cluster_centers_, columns=CLUSTER_FEATURES
        )
        label_maps[k] = interpretive_labels(k, centers, counts)
        joblib.dump(model, MODEL_DIRECTORY / f"t2d_validation_k{k}.joblib")
        stability_frames.append(
            bootstrap_stability(
                validation_values,
                model.labels_,
                model.cluster_centers_,
                k,
            )
        )
        model_metadata[str(k)] = {
            "inertia": model.inertia_,
            "validation_cluster_counts": counts.to_dict(),
            "test_cluster_counts": test[
                f"cluster_k{k}"
            ].value_counts().sort_index().to_dict(),
            "standardized_centroids": centers.to_dict("records"),
            "raw_centroids": pd.DataFrame(
                scaler.inverse_transform(model.cluster_centers_),
                columns=CLUSTER_FEATURES,
            ).to_dict("records"),
            "interpretive_labels": label_maps[k],
        }
    combined = pd.concat([validation, test], ignore_index=True)
    labels = combined[[PARTICIPANT_COLUMN, "split"]].copy()
    labels["subtype_k3"] = [
        f"k3_c{cluster}: {label_maps[K_PRIMARY][int(cluster)]}"
        for cluster in combined["cluster_k3"]
    ]
    labels["subtype_k4"] = [
        f"k4_c{cluster}: {label_maps[K_SENSITIVITY][int(cluster)]}"
        for cluster in combined["cluster_k4"]
    ]
    write_labels(LABELS_PATH, labels)
    profiles = build_profiles(combined, label_maps)
    write_csv(PROFILES_PATH, profiles)
    stability = pd.concat(stability_frames, ignore_index=True)
    write_csv(STABILITY_PATH, stability)
    make_figure(combined, label_maps)

    stability_summary = (
        stability.groupby("k")["adjusted_rand_index"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
        .to_dict("records")
    )
    output_paths = [
        LABELS_PATH,
        PROFILES_PATH,
        STABILITY_PATH,
        FIGURE_PATH,
        MODEL_DIRECTORY / "t2d_validation_scaler.joblib",
        MODEL_DIRECTORY / "t2d_validation_k3.joblib",
        MODEL_DIRECTORY / "t2d_validation_k4.joblib",
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "QC_COMPLETE",
        "analysis": "T2D-only external clinical partition",
        "t2d_definition_field": STUDY_GROUP_COLUMN,
        "t2d_definition_values": list(T2D_STUDY_GROUPS),
        "complete_case_counts": {"validation": 91, "test": 83},
        "cluster_features": CLUSTER_FEATURES,
        "age_definition": AGE_DEFINITION,
        "age_source": (
            f"{FINAL_MULTIMODAL_PATH}::{AGE_COLUMN}"
        ),
        "age_crosscheck": (
            "Exact equality with the participant-level static table for all "
            "validation and test participants."
        ),
        "nonfasting_caveat": NONFASTING_CAVEAT,
        "validation_fit_test_frozen": True,
        "scaler_fit_population": "T2D validation complete cases only",
        "k_values": {
            "primary": K_PRIMARY,
            "sensitivity": K_SENSITIVITY,
        },
        "kmeans_n_init": N_INIT,
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "stability_replicates": STABILITY_REPLICATES,
        "stability_summary": stability_summary,
        "k4_warning": K4_WARNING,
        "models": model_metadata,
        "input_paths": {
            "static": str(STATIC_PATH),
            "final_multimodal": str(FINAL_MULTIMODAL_PATH),
            "split": str(SPLIT_PATH),
            "validation_features": str(VALIDATION_FEATURES_PATH),
            "test_features": str(TEST_FEATURES_PATH),
        },
        "input_hashes": {
            "static": sha256_file(STATIC_PATH),
            "split": sha256_file(SPLIT_PATH),
            "validation_features": sha256_file(VALIDATION_FEATURES_PATH),
            "test_features": sha256_file(TEST_FEATURES_PATH),
        },
        "output_hashes": {
            path.name: sha256_file(path) for path in output_paths
        },
        "runtime_seconds": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "blockers": [],
    }
    write_json(MANIFEST_PATH, manifest)
    em_dash_failures = scan_em_dash(
        [Path(__file__), PROFILES_PATH, STABILITY_PATH, MANIFEST_PATH]
    )
    if em_dash_failures:
        raise RuntimeError(f"Em dash found: {em_dash_failures}")
    print(
        json.dumps(
            {
                "output_directory": str(OUTPUT_DIRECTORY),
                "counts": manifest["complete_case_counts"],
                "models": model_metadata,
                "stability_summary": stability_summary,
                "k4_warning": K4_WARNING,
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
