#!/usr/bin/env python3
"""Cluster frozen h0 and ht participant representations in two populations."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
DATA_ROOT = Path("/home/myriamcharfeddine/CGM/Data")
STATIC_PATH = DATA_ROOT / "enriched_multimodal/participant_static_features.parquet"
SPLIT_PATH = DATA_ROOT / "experiment_c_split_adapt6h_seed42/split_participants.csv"
STEP2_ROOT = PROJECT_ROOT / "outputs/hidden_state_phenotype/step2_validation_export/20260724T231513Z"
STEP3_ROOT = PROJECT_ROOT / "outputs/hidden_state_phenotype/step3_validation_clustering/20260725T001123Z"
STEP4_ROOT = PROJECT_ROOT / "outputs/hidden_state_phenotype/step4_test_confirmation/20260725T010440Z"
SUBTYPE_ROOT = PROJECT_ROOT / "subtype_partition"
OUTPUT_ROOT = PROJECT_ROOT / "hidden_state_clusters"
MODEL_ROOT = OUTPUT_ROOT / "frozen_models"

VALIDATION_REPRESENTATIONS = STEP2_ROOT / "participant_representations.parquet"
TEST_REPRESENTATIONS = STEP4_ROOT / "test_participant_representations.parquet"
VALIDATION_H0_ROOT = STEP2_ROOT / "validation_hidden_states/condition=full_profile"
TEST_H0_ROOT = STEP4_ROOT / "test_hidden_states/condition=full_profile"
VALIDATION_PROFILE_PATH = STEP3_ROOT / "validation_glycemic_nuisance_features.parquet"
TEST_PROFILE_PATH = STEP4_ROOT / "test_glycemic_nuisance_features.parquet"
SUBTYPE_LABELS_PATH = SUBTYPE_ROOT / "subtype_labels.parquet"
SUBTYPE_MANIFEST_PATH = SUBTYPE_ROOT / "subtype_manifest.json"

RANDOM_SEED = 42
K_VALUES = (3, 4)
KMEANS_N_INIT = 50
BOOTSTRAP_REPLICATES = 1000
STABILITY_REPLICATES = 200
STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
T2D_VALUES = {
    "insulin_dependent",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
}
H_COLS = [f"h_{i:03d}" for i in range(128)]
R_COLS = [f"r_{i:03d}" for i in range(128)]
SPACE_TO_PCA = {"h0": "full_all", "full_ht": "full_all", "neutral_ht": "neutral_all"}
PCA_N90 = {"full_all": 23, "neutral_all": 8}
PCA_VARIANCE_FIRST_TWO = {
    "full_all": (0.29084004460939367, 0.1376024154199507),
    "neutral_all": (0.47412725431311026, 0.12961192458280574),
}
ALL_NUMERIC_METRICS = ["hba1c", "mean_glucose", "glucose_cv", "tir_70_180"]
T2D_NUMERIC_METRICS = ALL_NUMERIC_METRICS + [
    "bmi_baseline",
    "tg_hdl",
    "c_peptide_ngml_baseline",
    "participants_age",
]
NONFASTING_CAVEAT = (
    "C-peptide and triglycerides were not confirmed fasting measurements. "
    "TG/HDL and related cluster profiles require this non-fasting caveat."
)
AGE_DEFINITION = (
    "participants_age is direct integer participant age at the study visit. "
    "It is not age at diabetes diagnosis and was not imputed."
)
MANIFOLD_CAVEAT = (
    "k is imposed on a previously established continuous manifold for like-for-like "
    "comparison. These partitions are not evidence of natural clusters."
)
ALLCOHORT_LABEL = "broad metabolic organization, not T2D subtypes"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_parquet_with_metadata(frame: pd.DataFrame, path: Path, metadata: dict[str, str]) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    existing = dict(table.schema.metadata or {})
    existing.update({str(k).encode(): str(v).encode() for k, v in metadata.items()})
    pq.write_table(table.replace_schema_metadata(existing), path, compression="zstd")


def load_static_and_splits() -> pd.DataFrame:
    static = pd.read_parquet(STATIC_PATH)
    split = pd.read_csv(SPLIT_PATH)
    static["participant_id"] = static["participant_id"].astype(str)
    split["participant_id"] = split["participant_id"].astype(str)
    split_col = "split" if "split" in split.columns else "partition"
    if split_col != "split":
        split = split.rename(columns={split_col: "split"})
    split["split"] = split["split"].replace({"val": "validation"})
    required = {
        "participant_id",
        "participants_study_group",
        "bmi_baseline",
        "c_peptide_ngml_baseline",
        "triglycerides_mgdl_baseline",
        "hdl_cholesterol_mgdl_baseline",
        "participants_age",
    }
    missing = sorted(required - set(static.columns))
    if missing:
        raise RuntimeError(f"Missing required static columns: {missing}")
    data = split[["participant_id", "split"]].merge(static, on="participant_id", how="left", validate="one_to_one")
    data["tg_hdl"] = data["triglycerides_mgdl_baseline"] / data["hdl_cholesterol_mgdl_baseline"]
    return data


def load_ht(path: Path, split_name: str) -> dict[str, pd.DataFrame]:
    frame = pd.read_parquet(path)
    frame["participant_id"] = frame["participant_id"].astype(str)
    if set(frame["split"].astype(str)) != {split_name}:
        raise RuntimeError(f"Unexpected split in {path}")
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
    for representation, space in (("full_all", "full_ht"), ("neutral_all", "neutral_ht")):
        part = frame[frame["representation_type"] == representation][["participant_id", *R_COLS]].copy()
        if part["participant_id"].duplicated().any():
            raise RuntimeError(f"Duplicate {space} rows in {split_name}")
        if not np.isfinite(part[R_COLS].to_numpy(float)).all():
            raise RuntimeError(f"Nonfinite {space} rows in {split_name}")
        result[space] = part.sort_values("participant_id").reset_index(drop=True)
    return result


def load_h0(root: Path, split_name: str) -> tuple[pd.DataFrame, dict]:
    paths = sorted(root.glob("participant_id=*/data.parquet"))
    if not paths:
        raise RuntimeError(f"No h0 files under {root}")
    rows = []
    total_h0_rows = 0
    h0_counts = []
    repeated_equal = True
    segments_match = True
    for path in paths:
        part = pd.read_parquet(path, columns=["participant_id", "segment_id", "is_h0_row", *H_COLS])
        h0 = part[part["is_h0_row"].astype(bool)]
        total_h0_rows += len(h0)
        h0_counts.append(len(h0))
        if h0.empty:
            raise RuntimeError(f"No h0 row in {path}")
        segments_match &= h0["segment_id"].nunique() == len(h0)
        values = h0[H_COLS].to_numpy(float)
        repeated_equal &= bool(np.allclose(values, values[[0]], atol=0.0, rtol=0.0))
        rows.append({"participant_id": str(h0["participant_id"].iloc[0]), **dict(zip(H_COLS, values[0]))})
    frame = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    if frame["participant_id"].duplicated().any() or not np.isfinite(frame[H_COLS].to_numpy(float)).all():
        raise RuntimeError(f"Invalid h0 participant rows in {split_name}")
    audit = {
        "path": str(root),
        "participants": len(frame),
        "total_h0_rows": total_h0_rows,
        "h0_rows_per_participant_range": [int(min(h0_counts)), int(max(h0_counts))],
        "one_h0_per_segment": bool(segments_match),
        "within_participant_repeats_exactly_equal": bool(repeated_equal),
        "finite": True,
        "model_forward_pass_run": False,
    }
    return frame, audit


def load_frozen_pca() -> dict[str, dict]:
    output = {}
    for name in ("full_all", "neutral_all"):
        root = STEP3_ROOT / "frozen_validation_pipeline" / name
        output[name] = {
            "scaler": joblib.load(root / f"{name}_scaler.joblib"),
            "pca": joblib.load(root / f"{name}_pca.joblib"),
            "keep": np.load(root / "kept_dimensions.npy"),
            "n90": PCA_N90[name],
        }
    return output


def project(frame: pd.DataFrame, value_cols: list[str], frozen: dict, pca_name: str) -> pd.DataFrame:
    pipe = frozen[pca_name]
    values = frame[value_cols].to_numpy(float)
    scores = pipe["pca"].transform(pipe["scaler"].transform(values[:, pipe["keep"]]))[:, : pipe["n90"]]
    result = frame[["participant_id"]].copy()
    for index in range(scores.shape[1]):
        result[f"pc_{index + 1:03d}"] = scores[:, index]
    return result


def make_profiles(static_split: pd.DataFrame) -> dict[str, pd.DataFrame]:
    val = pd.read_parquet(VALIDATION_PROFILE_PATH)
    test = pd.read_parquet(TEST_PROFILE_PATH)
    val["participant_id"] = val["participant_id"].astype(str)
    test["participant_id"] = test["participant_id"].astype(str)
    common = ["participant_id", *ALL_NUMERIC_METRICS, "study_group"]
    val = val[common].copy()
    test = test[common].copy()
    val["split"] = "validation"
    test["split"] = "test"
    combined = pd.concat([val, test], ignore_index=True)
    clinical = static_split[
        [
            "participant_id",
            "split",
            "participants_study_group",
            "bmi_baseline",
            "c_peptide_ngml_baseline",
            "participants_age",
            "tg_hdl",
        ]
    ].copy()
    clinical = clinical.rename(columns={"participants_study_group": "diagnosis_study_group"})
    combined = combined.merge(clinical, on=["participant_id", "split"], how="left", validate="one_to_one")
    return {name: group.reset_index(drop=True) for name, group in combined.groupby("split")}


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[indexes].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_prop_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    return bootstrap_mean_ci(values.astype(float), seed)


def fit_partitions(
    projected: dict[str, dict[str, pd.DataFrame]],
    ids: dict[str, set[str]],
    population: str,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    label_rows = []
    model_summary = {}
    score_rows = []
    for space_index, space in enumerate(("h0", "full_ht", "neutral_ht")):
        val = projected["validation"][space]
        test = projected["test"][space]
        val = val[val["participant_id"].isin(ids["validation"])].sort_values("participant_id").reset_index(drop=True)
        test = test[test["participant_id"].isin(ids["test"])].sort_values("participant_id").reset_index(drop=True)
        pc_cols = [column for column in val.columns if column.startswith("pc_")]
        scaler = StandardScaler().fit(val[pc_cols])
        val_z = scaler.transform(val[pc_cols])
        test_z = scaler.transform(test[pc_cols])
        scaler_path = MODEL_ROOT / f"{population}_{space}_validation_pc_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        for split_name, source, zvalues in (("validation", val, val_z), ("test", test, test_z)):
            for row_index, participant_id in enumerate(source["participant_id"]):
                score_rows.append(
                    {
                        "participant_id": participant_id,
                        "split": split_name,
                        "population": population,
                        "space": space,
                        **{column: source.iloc[row_index][column] for column in pc_cols},
                        **{f"z_{column}": zvalues[row_index, j] for j, column in enumerate(pc_cols)},
                    }
                )
        for k in K_VALUES:
            model = KMeans(
                n_clusters=k,
                n_init=KMEANS_N_INIT,
                random_state=RANDOM_SEED + 1000 * space_index + k,
                algorithm="lloyd",
            ).fit(val_z)
            val_labels = model.labels_.astype(int)
            test_labels = model.predict(test_z).astype(int)
            model_path = MODEL_ROOT / f"{population}_{space}_validation_k{k}.joblib"
            joblib.dump(model, model_path)
            for split_name, source, labels in (
                ("validation", val, val_labels),
                ("test", test, test_labels),
            ):
                label_rows.extend(
                    {
                        "participant_id": participant_id,
                        "split": split_name,
                        "space": space,
                        "k": k,
                        "cluster_id": int(label),
                    }
                    for participant_id, label in zip(source["participant_id"], labels)
                )
            stability = []
            rng = np.random.default_rng(RANDOM_SEED + 10000 * space_index + k)
            for replicate in range(STABILITY_REPLICATES):
                draw = rng.integers(0, len(val_z), size=len(val_z))
                boot = KMeans(
                    n_clusters=k,
                    n_init=KMEANS_N_INIT,
                    random_state=RANDOM_SEED + replicate,
                    algorithm="lloyd",
                ).fit(val_z[draw])
                stability.append(adjusted_rand_score(val_labels, boot.predict(val_z)))
            key = f"{space}_k{k}"
            model_summary[key] = {
                "validation_n": len(val),
                "test_n": len(test),
                "validation_cluster_counts": pd.Series(val_labels).value_counts().sort_index().to_dict(),
                "test_cluster_counts": pd.Series(test_labels).value_counts().sort_index().to_dict(),
                "inertia": float(model.inertia_),
                "validation_bootstrap_ari_mean": float(np.mean(stability)),
                "validation_bootstrap_ari_median": float(np.median(stability)),
                "validation_bootstrap_ari_ci": [
                    float(np.quantile(stability, 0.025)),
                    float(np.quantile(stability, 0.975)),
                ],
                "scaler_path": str(scaler_path),
                "model_path": str(model_path),
            }
    labels = pd.DataFrame(label_rows)
    return labels, model_summary, pd.DataFrame(score_rows)


def profile_clusters(
    labels: pd.DataFrame,
    profiles: dict[str, pd.DataFrame],
    subtype_labels: pd.DataFrame,
    population: str,
) -> pd.DataFrame:
    rows = []
    numeric_metrics = ALL_NUMERIC_METRICS if population == "allcohort" else T2D_NUMERIC_METRICS
    for (split_name, space, k, cluster_id), label_group in labels.groupby(["split", "space", "k", "cluster_id"]):
        data = label_group[["participant_id"]].merge(
            profiles[split_name], on="participant_id", how="left", validate="one_to_one"
        )
        cluster_n = len(data)
        for metric_index, metric in enumerate(numeric_metrics):
            values = pd.to_numeric(data[metric], errors="coerce").to_numpy(float)
            valid = values[np.isfinite(values)]
            low, high = bootstrap_mean_ci(
                valid,
                RANDOM_SEED + metric_index * 100000 + int(k) * 1000 + int(cluster_id) * 10 + (split_name == "test"),
            )
            rows.append(
                {
                    "population": population,
                    "split": split_name,
                    "space": space,
                    "k": int(k),
                    "cluster_id": int(cluster_id),
                    "profile_type": "numeric",
                    "metric": metric,
                    "category": "",
                    "cluster_n": cluster_n,
                    "n_nonmissing": len(valid),
                    "mean": float(np.mean(valid)) if len(valid) else np.nan,
                    "sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
                    "median": float(np.median(valid)) if len(valid) else np.nan,
                    "ci_low": low,
                    "ci_high": high,
                    "proportion": np.nan,
                    "analysis_label": ALLCOHORT_LABEL if population == "allcohort" else "T2D-only imposed partition",
                    "nonfasting_caveat": NONFASTING_CAVEAT,
                    "age_definition": AGE_DEFINITION,
                    "manifold_caveat": MANIFOLD_CAVEAT,
                }
            )
        for category, count in data["study_group"].fillna("missing").value_counts().items():
            binary = (data["study_group"].fillna("missing") == category).to_numpy()
            low, high = bootstrap_prop_ci(
                binary, RANDOM_SEED + int(k) * 1000 + int(cluster_id) * 10 + len(str(category))
            )
            rows.append(
                {
                    "population": population,
                    "split": split_name,
                    "space": space,
                    "k": int(k),
                    "cluster_id": int(cluster_id),
                    "profile_type": "study_group_composition",
                    "metric": "study_group",
                    "category": category,
                    "cluster_n": cluster_n,
                    "n_nonmissing": cluster_n,
                    "mean": np.nan,
                    "sd": np.nan,
                    "median": np.nan,
                    "ci_low": low,
                    "ci_high": high,
                    "proportion": count / cluster_n,
                    "analysis_label": ALLCOHORT_LABEL if population == "allcohort" else "T2D-only imposed partition",
                    "nonfasting_caveat": NONFASTING_CAVEAT,
                    "age_definition": AGE_DEFINITION,
                    "manifold_caveat": MANIFOLD_CAVEAT,
                }
            )
        if population == "t2d":
            subtype_col = f"subtype_k{int(k)}"
            overlap = data[["participant_id"]].merge(
                subtype_labels[["participant_id", "split", subtype_col]],
                left_on="participant_id",
                right_on="participant_id",
                how="left",
            )
            for category, count in overlap[subtype_col].fillna("not complete case").value_counts().items():
                binary = (overlap[subtype_col].fillna("not complete case") == category).to_numpy()
                low, high = bootstrap_prop_ci(
                    binary, RANDOM_SEED + 500000 + int(k) * 1000 + int(cluster_id) * 10 + len(str(category))
                )
                rows.append(
                    {
                        "population": population,
                        "split": split_name,
                        "space": space,
                        "k": int(k),
                        "cluster_id": int(cluster_id),
                        "profile_type": "clinical_subtype_overlap",
                        "metric": subtype_col,
                        "category": category,
                        "cluster_n": cluster_n,
                        "n_nonmissing": int(overlap[subtype_col].notna().sum()),
                        "mean": np.nan,
                        "sd": np.nan,
                        "median": np.nan,
                        "ci_low": low,
                        "ci_high": high,
                        "proportion": count / cluster_n,
                        "analysis_label": "T2D-only imposed partition",
                        "nonfasting_caveat": NONFASTING_CAVEAT,
                        "age_definition": AGE_DEFINITION,
                        "manifold_caveat": MANIFOLD_CAVEAT,
                    }
                )
    return pd.DataFrame(rows)


def figure_3(labels: pd.DataFrame, scores: pd.DataFrame, profiles: dict[str, pd.DataFrame]) -> None:
    points = (
        scores.query("population == 'allcohort' and split == 'test' and space == 'full_ht'")
        .merge(labels.query("split == 'test' and space == 'full_ht' and k == 3"), on=["participant_id", "split", "space"])
        .merge(profiles["test"][["participant_id", "mean_glucose"]], on="participant_id")
    )
    v1, v2 = PCA_VARIANCE_FIRST_TWO["full_all"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=True, sharey=True)
    for cluster_id in range(3):
        group = points[points["cluster_id"] == cluster_id]
        axes[0].scatter(
            group["pc_001"], group["pc_002"], s=34, alpha=0.82,
            color=STRATUM_COLORS[int(cluster_id)], label=f"Cluster {cluster_id} (n={len(group)})",
        )
    axes[0].legend(frameon=True, fontsize=9)
    axes[0].set_title("Imposed k=3 partition")
    scatter = axes[1].scatter(
        points["pc_001"], points["pc_002"], c=points["mean_glucose"],
        cmap="viridis", s=34, alpha=0.85,
    )
    colorbar = fig.colorbar(scatter, ax=axes[1])
    colorbar.set_label("Participant mean glucose (mg/dL)")
    axes[1].set_title("Continuous glycemic reference")
    for axis in axes:
        axis.set_xlabel(f"PC1 ({v1 * 100:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({v2 * 100:.1f}% variance)")
    fig.suptitle("All-cohort test full ht: broad metabolic organization, not T2D subtypes", fontsize=14)
    fig.text(0.5, 0.015, MANIFOLD_CAVEAT, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(OUTPUT_ROOT / "fig3_allcohort_clusters_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_4(labels: pd.DataFrame, scores: pd.DataFrame, profiles: dict[str, pd.DataFrame]) -> None:
    base = scores.query("population == 't2d' and split == 'test' and space == 'full_ht'")
    base = base.merge(profiles["test"][["participant_id", "mean_glucose"]], on="participant_id")
    v1, v2 = PCA_VARIANCE_FIRST_TWO["full_all"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharex=True, sharey=True)
    for axis_index, k in enumerate(K_VALUES):
        points = base.merge(
            labels.query("split == 'test' and space == 'full_ht' and k == @k"),
            on=["participant_id", "split", "space"],
        )
        for cluster_id in range(k):
            group = points[points["cluster_id"] == cluster_id]
            axes[axis_index].scatter(
                group["pc_001"], group["pc_002"], s=38, alpha=0.84,
                color=STRATUM_COLORS[int(cluster_id)], label=f"C{cluster_id} (n={len(group)})",
            )
        axes[axis_index].set_title(f"Imposed k={k} partition")
        axes[axis_index].legend(frameon=True, fontsize=8)
    scatter = axes[2].scatter(
        base["pc_001"], base["pc_002"], c=base["mean_glucose"],
        cmap="viridis", s=38, alpha=0.85,
    )
    colorbar = fig.colorbar(scatter, ax=axes[2])
    colorbar.set_label("Participant mean glucose (mg/dL)")
    axes[2].set_title("Continuous glycemic reference")
    for axis in axes:
        axis.set_xlabel(f"PC1 ({v1 * 100:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({v2 * 100:.1f}% variance)")
    fig.suptitle("T2D-only test full ht: imposed partitions on the frozen validation PCA", fontsize=14)
    fig.text(0.5, 0.015, MANIFOLD_CAVEAT, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(OUTPUT_ROOT / "fig4_t2d_clusters_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_5(profiles: pd.DataFrame) -> None:
    metrics = ["bmi_baseline", "tg_hdl", "c_peptide_ngml_baseline", "hba1c", "mean_glucose"]
    names = {
        "bmi_baseline": "BMI (kg/m²)",
        "tg_hdl": "TG/HDL ratio",
        "c_peptide_ngml_baseline": "C-peptide (ng/mL)",
        "hba1c": "HbA1c (%)",
        "mean_glucose": "Mean glucose (mg/dL)",
    }
    plot = profiles.query(
        "split == 'test' and k == 3 and profile_type == 'numeric' and metric in @metrics"
    ).copy()
    plot["cluster"] = "C" + plot["cluster_id"].astype(str)
    space_order = ["h0", "full_ht", "neutral_ht"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.8))
    for axis, metric in zip(axes, metrics):
        sub = plot[plot["metric"] == metric]
        clusters = sorted(sub["cluster"].unique())
        x = np.arange(len(clusters))
        width = 0.24
        for index, space in enumerate(space_order):
            block = sub[sub["space"] == space].set_index("cluster").reindex(clusters)
            offset = (index - 1) * width
            axis.bar(
                x + offset,
                block["mean"],
                width,
                label=space.replace("_", " "),
                color=STRATUM_COLORS[index],
                alpha=0.88,
                yerr=np.vstack([block["mean"] - block["ci_low"], block["ci_high"] - block["mean"]]),
                capsize=2,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(clusters)
        axis.set_xlabel("Space-specific cluster ID")
        axis.set_ylabel(names[metric])
        axis.set_title(names[metric])
    axes[0].legend(title="Representation", fontsize=9)
    fig.suptitle("T2D-only test k=3 profiles: h0 baseline versus streamed ht partitions", fontsize=14)
    fig.text(
        0.5,
        0.015,
        f"Cluster IDs are space-specific. Error bars are participant-bootstrap 95% CIs. {NONFASTING_CAVEAT}",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(OUTPUT_ROOT / "fig5_cluster_profile_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    required = [
        STATIC_PATH, SPLIT_PATH, VALIDATION_REPRESENTATIONS, TEST_REPRESENTATIONS,
        VALIDATION_PROFILE_PATH, TEST_PROFILE_PATH, SUBTYPE_LABELS_PATH, SUBTYPE_MANIFEST_PATH,
        STEP3_ROOT / "pca_loadings.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prerequisite files: {missing}")

    static_split = load_static_and_splits()
    subtype_manifest = json.loads(SUBTYPE_MANIFEST_PATH.read_text())
    if set(subtype_manifest["t2d_definition_values"]) != T2D_VALUES:
        raise RuntimeError("T2D definition does not match Prompt 1 manifest")
    subtype_labels = pd.read_parquet(SUBTYPE_LABELS_PATH)
    subtype_labels["participant_id"] = subtype_labels["participant_id"].astype(str)

    ht_val = load_ht(VALIDATION_REPRESENTATIONS, "validation")
    ht_test = load_ht(TEST_REPRESENTATIONS, "test")
    h0_val, h0_val_audit = load_h0(VALIDATION_H0_ROOT, "validation")
    h0_test, h0_test_audit = load_h0(TEST_H0_ROOT, "test")
    if not all(
        [
            h0_val_audit["one_h0_per_segment"],
            h0_val_audit["within_participant_repeats_exactly_equal"],
            h0_test_audit["one_h0_per_segment"],
            h0_test_audit["within_participant_repeats_exactly_equal"],
        ]
    ):
        raise RuntimeError("h0 availability QC failed")

    raw = {
        "validation": {"h0": h0_val, **ht_val},
        "test": {"h0": h0_test, **ht_test},
    }
    eligible_ids = {
        split_name: set.intersection(*(set(frame["participant_id"]) for frame in spaces.values()))
        for split_name, spaces in raw.items()
    }
    if len(eligible_ids["validation"]) != 239 or len(eligible_ids["test"]) != 221:
        raise RuntimeError(f"Unexpected all-cohort counts: { {key: len(value) for key, value in eligible_ids.items()} }")
    t2d_ids = {
        split_name: set(
            static_split[
                (static_split["split"] == split_name)
                & static_split["participants_study_group"].isin(T2D_VALUES)
            ]["participant_id"]
        )
        & eligible_ids[split_name]
        for split_name in ("validation", "test")
    }
    if len(t2d_ids["validation"]) != 94 or len(t2d_ids["test"]) != 87:
        raise RuntimeError(f"Unexpected T2D counts: { {key: len(value) for key, value in t2d_ids.items()} }")

    frozen = load_frozen_pca()
    projected = {"validation": {}, "test": {}}
    for split_name in ("validation", "test"):
        projected[split_name]["h0"] = project(raw[split_name]["h0"], H_COLS, frozen, "full_all")
        projected[split_name]["full_ht"] = project(raw[split_name]["full_ht"], R_COLS, frozen, "full_all")
        projected[split_name]["neutral_ht"] = project(raw[split_name]["neutral_ht"], R_COLS, frozen, "neutral_all")

    profiles = make_profiles(static_split)
    all_labels, all_models, all_scores = fit_partitions(projected, eligible_ids, "allcohort")
    t2d_labels, t2d_models, t2d_scores = fit_partitions(projected, t2d_ids, "t2d")
    label_metadata = {
        "protocol": "validation-fit, test-frozen",
        "manifold_caveat": MANIFOLD_CAVEAT,
        "age_definition": AGE_DEFINITION,
        "nonfasting_caveat": NONFASTING_CAVEAT,
    }
    write_parquet_with_metadata(
        all_labels[["participant_id", "split", "space", "k", "cluster_id"]],
        OUTPUT_ROOT / "allcohort_cluster_labels.parquet",
        {**label_metadata, "analysis_label": ALLCOHORT_LABEL},
    )
    write_parquet_with_metadata(
        t2d_labels[["participant_id", "split", "space", "k", "cluster_id"]],
        OUTPUT_ROOT / "t2d_cluster_labels.parquet",
        {**label_metadata, "analysis_label": "T2D-only imposed partition"},
    )
    all_profiles = profile_clusters(all_labels, profiles, subtype_labels, "allcohort")
    t2d_profiles = profile_clusters(t2d_labels, profiles, subtype_labels, "t2d")
    all_profiles.to_csv(OUTPUT_ROOT / "allcohort_cluster_profiles.csv", index=False)
    t2d_profiles.to_csv(OUTPUT_ROOT / "t2d_cluster_profiles.csv", index=False)
    all_scores.to_parquet(OUTPUT_ROOT / "allcohort_pca_scores.parquet", index=False)
    t2d_scores.to_parquet(OUTPUT_ROOT / "t2d_pca_scores.parquet", index=False)

    figure_3(all_labels, all_scores, profiles)
    figure_4(t2d_labels, t2d_scores, profiles)
    figure_5(t2d_profiles)

    output_files = [
        "allcohort_cluster_labels.parquet",
        "t2d_cluster_labels.parquet",
        "allcohort_cluster_profiles.csv",
        "t2d_cluster_profiles.csv",
        "allcohort_pca_scores.parquet",
        "t2d_pca_scores.parquet",
        "fig3_allcohort_clusters_pca.png",
        "fig4_t2d_clusters_pca.png",
        "fig5_cluster_profile_comparison.png",
    ]
    manifest = {
        "analysis": "Frozen h0 and ht imposed partitions in all-cohort and diagnosis-defined T2D populations",
        "status": "QC_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_fit_test_frozen": True,
        "model_forward_pass_run": False,
        "h0_availability_gate": {"validation": h0_val_audit, "test": h0_test_audit, "status": "PASS"},
        "population_counts": {
            "allcohort": {key: len(value) for key, value in eligible_ids.items()},
            "t2d": {key: len(value) for key, value in t2d_ids.items()},
            "t2d_clinical_complete_case": subtype_manifest["complete_case_counts"],
        },
        "t2d_definition_field": "participants_study_group",
        "t2d_definition_values": sorted(T2D_VALUES),
        "spaces": {
            "h0": "full-profile static initialization before any CGM update",
            "full_ht": "full-profile all-recording streamed participant representation",
            "neutral_ht": "static-neutralized all-recording streamed participant representation",
        },
        "pca_transport": {
            "h0": "frozen full_all validation PCA, 23 PCs reaching at least 90% variance in validation ht",
            "full_ht": "frozen full_all validation PCA, 23 PCs",
            "neutral_ht": "frozen neutral_all validation PCA, 8 PCs",
            "post_pca_standardization": "fit separately on the relevant validation population and space, then frozen on test",
            "loadings_path": str(STEP3_ROOT / "pca_loadings.parquet"),
        },
        "k_values": {"primary": 3, "sensitivity": 4},
        "kmeans_n_init": KMEANS_N_INIT,
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "stability_replicates": STABILITY_REPLICATES,
        "models": {"allcohort": all_models, "t2d": t2d_models},
        "interpretation": {
            "allcohort": ALLCOHORT_LABEL,
            "t2d": "Primary population for later comparison with clinical T2D subtype partition.",
            "h0": "Descriptive baseline because h0 encodes the static profile.",
            "manifold": MANIFOLD_CAVEAT,
            "clinical_k4": subtype_manifest["k4_warning"],
        },
        "age_definition": AGE_DEFINITION,
        "nonfasting_caveat": NONFASTING_CAVEAT,
        "input_paths": {path.name: str(path) for path in required},
        "input_hashes": {path.name: sha256(path) for path in required},
        "output_hashes": {name: sha256(OUTPUT_ROOT / name) for name in output_files},
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "runtime_seconds": time.time() - started,
    }
    (OUTPUT_ROOT / "clusters_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    print(
        json.dumps(
            {
                "output_directory": str(OUTPUT_ROOT),
                "population_counts": manifest["population_counts"],
                "h0_gate": manifest["h0_availability_gate"],
                "allcohort_models": all_models,
                "t2d_models": t2d_models,
                "manifest": str(OUTPUT_ROOT / "clusters_manifest.json"),
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
