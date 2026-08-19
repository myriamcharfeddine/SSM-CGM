#!/usr/bin/env python3
"""Build validation-fitted continuous clinical targets for T2D participants."""

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
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/continuous_clinical"
MODEL_ROOT = OUTPUT_ROOT / "frozen_models"
STATIC_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "participant_static_features.parquet"
)
SPLIT_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/"
    "experiment_c_split_adapt6h_seed42/split_participants.csv"
)
COVERAGE_AUDIT_PATH = PROJECT_ROOT / "subtype_partition/t2d_coverage_audit.csv"
SUBTYPE_LABELS_PATH = PROJECT_ROOT / "subtype_partition/subtype_labels.parquet"
SUBTYPE_MANIFEST_PATH = PROJECT_ROOT / "subtype_partition/subtype_manifest.json"
HIDDEN_CLUSTER_LABELS_PATH = (
    PROJECT_ROOT / "outputs/hidden_state_clusters/t2d_cluster_labels.parquet"
)
HIDDEN_CLUSTER_MANIFEST_PATH = (
    PROJECT_ROOT / "outputs/hidden_state_clusters/clusters_manifest.json"
)
TEST_GLYCEMIC_PATH = (
    PROJECT_ROOT
    / "outputs/hidden_state_phenotype/step4_test_confirmation/"
    "20260725T010440Z/test_glycemic_nuisance_features.parquet"
)

TARGETS_PATH = OUTPUT_ROOT / "clinical_targets.parquet"
LOADINGS_PATH = OUTPUT_ROOT / "pca_loadings.csv"
NEGATIVE_RESULT_PATH = OUTPUT_ROOT / "discrete_clustering_negative_result.md"
MANIFEST_PATH = OUTPUT_ROOT / "clinical_targets_manifest.json"
FIGURE_LOADINGS_PATH = OUTPUT_ROOT / "fig1_clinical_pca_loadings.png"
FIGURE_SCATTER_PATH = OUTPUT_ROOT / "fig2_clinical_pc1_pc2_scatter.png"
SCALER_PATH = MODEL_ROOT / "clinical_marker_validation_scaler.joblib"
PCA_PATH = MODEL_ROOT / "clinical_marker_validation_pca.joblib"

RANDOM_SEED = 42
STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
T2D_VALUES = {
    "insulin_dependent",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
}
RAW_FEATURES = ["bmi", "log_tg_hdl", "log_c_peptide"]
STANDARDIZED_FEATURES = ["z_bmi", "z_log_tg_hdl", "z_log_c_peptide"]
EXPECTED_COUNTS = {"validation": 91, "test": 83}
NONTRIVIAL_LOADING_THRESHOLD = 0.30
AGE_DEFINITION = (
    "Participant age at study visit, NOT age at diabetes diagnosis. "
    "The value comes directly from participants_age and was not imputed."
)
NONFASTING_CAVEAT = (
    "C-peptide and triglycerides were not confirmed fasting measurements. "
    "log TG/HDL and log C-peptide require this non-fasting caveat."
)
NEGATIVE_RESULT_TEMPLATE = (
    "When clustering was restricted to diagnosis-defined T2D participants, "
    "validation-fitted k=3 and k=4 partitions remained unstable, produced empty "
    "or near-empty test clusters, and showed near-chance agreement with the "
    "clinical-marker partition (test ARI: h0 {h0:.3f}, full h_t {full_ht:.3f}, "
    "neutral h_t {neutral_ht:.3f}). This indicates the representation does not "
    "contain reproducible discrete T2D subtype regions at this sample size and "
    "motivates the continuous analysis below."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_targets(
    frame: pd.DataFrame,
    component_names: dict[str, str],
    component_descriptions: dict[str, str],
) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"analysis": b"Continuous clinical phenotype targets, T2D-only",
            b"fit_rule": b"Scaler and PCA fit on T2D validation only; frozen on test.",
            b"age_definition": AGE_DEFINITION.encode(),
            b"nonfasting_caveat": NONFASTING_CAVEAT.encode(),
            b"raw_marker_columns": b"bmi, log_tg_hdl, log_c_peptide",
            b"standardized_target_columns": (
                b"z_bmi, z_log_tg_hdl, z_log_c_peptide"
            ),
            b"pca_target_columns": b"clinical_pc1, clinical_pc2",
            b"clinical_pc1_name": component_names["PC1"].encode(),
            b"clinical_pc2_name": component_names["PC2"].encode(),
        }
    )
    fields = []
    for field in table.schema:
        field_metadata = dict(field.metadata or {})
        if field.name == "participants_age":
            field_metadata.update(
                {
                    b"description": AGE_DEFINITION.encode(),
                    b"role": b"labeled covariate",
                    b"is_age_at_diabetes_diagnosis": b"false",
                    b"source_column": b"participants_age",
                }
            )
        elif field.name == "clinical_pc1":
            field_metadata.update(
                {
                    b"component_name": component_names["PC1"].encode(),
                    b"description": component_descriptions["PC1"].encode(),
                    b"fit_rule": b"T2D validation fit; frozen application to test",
                }
            )
        elif field.name == "clinical_pc2":
            field_metadata.update(
                {
                    b"component_name": component_names["PC2"].encode(),
                    b"description": component_descriptions["PC2"].encode(),
                    b"fit_rule": b"T2D validation fit; frozen application to test",
                }
            )
        fields.append(field.with_metadata(field_metadata))
    schema = pa.schema(fields, metadata=metadata)
    table = pa.Table.from_arrays(table.columns, schema=schema)
    pq.write_table(
        table,
        TARGETS_PATH,
        compression="zstd",
    )


def load_complete_cases() -> pd.DataFrame:
    static = pd.read_parquet(STATIC_PATH)
    split = pd.read_csv(SPLIT_PATH, dtype={"participant_id": str})
    static["participant_id"] = static["participant_id"].astype(str)
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
        raise RuntimeError(f"Missing static columns: {missing}")
    frame = split[["participant_id", "split"]].merge(
        static[list(required)],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    frame = frame[
        frame["split"].isin(["validation", "test"])
        & frame["participants_study_group"].isin(T2D_VALUES)
    ].copy()
    frame["tg_hdl"] = (
        frame["triglycerides_mgdl_baseline"]
        / frame["hdl_cholesterol_mgdl_baseline"]
    )
    if (
        (frame["tg_hdl"].dropna() <= 0).any()
        or (frame["c_peptide_ngml_baseline"].dropna() <= 0).any()
    ):
        raise RuntimeError("Log targets require strictly positive TG/HDL and C-peptide")
    frame["bmi"] = frame["bmi_baseline"]
    frame["log_tg_hdl"] = np.log(frame["tg_hdl"])
    frame["log_c_peptide"] = np.log(frame["c_peptide_ngml_baseline"])
    frame = frame.dropna(
        subset=[*RAW_FEATURES, "participants_age"]
    ).copy()
    counts = frame.groupby("split").size().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Complete-case counts changed: observed {counts}, expected {EXPECTED_COUNTS}"
        )
    if frame["participant_id"].duplicated().any():
        raise RuntimeError("Duplicate participant IDs in complete-case cohort")
    return frame.sort_values(["split", "participant_id"]).reset_index(drop=True)


def classify_components(
    components: np.ndarray,
) -> tuple[dict[str, str], dict[str, str], dict[str, bool]]:
    names = {}
    descriptions = {}
    qualifies = {}
    marker_names = ["BMI", "log TG/HDL", "log C-peptide"]
    for component_index in range(2):
        values = components[component_index]
        same_sign = bool(np.all(values > 0) or np.all(values < 0))
        all_nontrivial = bool(
            np.all(np.abs(values) >= NONTRIVIAL_LOADING_THRESHOLD)
        )
        qualifies[f"PC{component_index + 1}"] = same_sign and all_nontrivial
        if qualifies[f"PC{component_index + 1}"]:
            names[f"PC{component_index + 1}"] = "insulin-resistance axis"
            direction = "higher" if np.all(values > 0) else "lower"
            descriptions[f"PC{component_index + 1}"] = (
                f"High scores reflect {direction} BMI, TG/HDL, and C-peptide together."
            )
        elif component_index == 0:
            names["PC1"] = "continuous T2D clinical phenotype axis"
            descriptions["PC1"] = describe_direction(values, marker_names)
        else:
            names["PC2"] = (
                "BMI/TG-HDL dissociation axis (independent of C-peptide)"
            )
            descriptions["PC2"] = describe_direction(values, marker_names)
    return names, descriptions, qualifies


def describe_direction(values: np.ndarray, marker_names: list[str]) -> str:
    positive = [
        marker_names[index]
        for index, value in enumerate(values)
        if value >= NONTRIVIAL_LOADING_THRESHOLD
    ]
    negative = [
        marker_names[index]
        for index, value in enumerate(values)
        if value <= -NONTRIVIAL_LOADING_THRESHOLD
    ]
    near_zero = [
        marker_names[index]
        for index, value in enumerate(values)
        if abs(value) < NONTRIVIAL_LOADING_THRESHOLD
    ]
    parts = []
    if positive:
        parts.append("higher " + " and ".join(positive))
    if negative:
        parts.append("lower " + " and ".join(negative))
    text = "High scores reflect " + ", with ".join(parts) + "."
    if near_zero:
        text += " " + " and ".join(near_zero) + " contributes little."
    return text


def compute_prior_aris() -> dict[str, float]:
    hidden = pd.read_parquet(HIDDEN_CLUSTER_LABELS_PATH)
    clinical = pd.read_parquet(SUBTYPE_LABELS_PATH)
    hidden["participant_id"] = hidden["participant_id"].astype(str)
    clinical["participant_id"] = clinical["participant_id"].astype(str)
    results = {}
    for space in ("h0", "full_ht", "neutral_ht"):
        assignment = hidden[
            (hidden["split"] == "test")
            & (hidden["space"] == space)
            & (hidden["k"] == 3)
        ]
        merged = assignment.merge(
            clinical[["participant_id", "split", "subtype_k3"]],
            on=["participant_id", "split"],
            how="inner",
            validate="one_to_one",
        )
        clinical_id = (
            merged["subtype_k3"].str.extract(r"k3_c(\d+)")[0].astype(int)
        )
        results[space] = float(
            adjusted_rand_score(clinical_id, merged["cluster_id"])
        )
    return results


def plot_loadings(
    loading_table: pd.DataFrame,
    variance: np.ndarray,
    component_names: dict[str, str],
) -> None:
    plot = loading_table.copy()
    labels = {
        "bmi": "BMI",
        "log_tg_hdl": "Log TG/HDL",
        "log_c_peptide": "Log C-peptide",
    }
    plot["marker_label"] = plot["marker"].map(labels)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    for component_index, axis in enumerate(axes):
        component = f"PC{component_index + 1}"
        block = plot[plot["component"] == component]
        bars = axis.bar(
            block["marker_label"],
            block["loading"],
            color=STRATUM_COLORS[component_index],
            alpha=0.9,
        )
        axis.axhline(0, color="#333333", linewidth=0.9)
        axis.set_ylim(-0.85, 0.85)
        axis.set_ylabel("PCA loading")
        axis.set_title(
            f"{component}: {component_names[component]}\n"
            f"{variance[component_index] * 100:.1f}% variance explained"
        )
        axis.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, block["loading"]):
            offset = 0.035 if value >= 0 else -0.065
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    fig.suptitle("Validation-fitted continuous T2D clinical PCA loadings", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_LOADINGS_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_test_scatter(
    targets: pd.DataFrame,
    variance: np.ndarray,
    component_names: dict[str, str],
) -> int:
    hba1c = pd.read_parquet(TEST_GLYCEMIC_PATH)[
        ["participant_id", "hba1c"]
    ].copy()
    hba1c["participant_id"] = hba1c["participant_id"].astype(str)
    plot = targets[targets["split"] == "test"].merge(
        hba1c, on="participant_id", how="left", validate="one_to_one"
    )
    valid = plot["hba1c"].notna()
    fig, axis = plt.subplots(figsize=(8.2, 6.6))
    scatter = axis.scatter(
        plot.loc[valid, "clinical_pc1"],
        plot.loc[valid, "clinical_pc2"],
        c=plot.loc[valid, "hba1c"],
        cmap="viridis",
        s=52,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.35,
    )
    if (~valid).any():
        axis.scatter(
            plot.loc[~valid, "clinical_pc1"],
            plot.loc[~valid, "clinical_pc2"],
            color=STRATUM_COLORS[4],
            s=52,
            alpha=0.8,
            label=f"HbA1c missing (n={(~valid).sum()})",
        )
        axis.legend(frameon=True)
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("HbA1c (%)")
    axis.set_xlabel(
        f"PC1: {component_names['PC1']} "
        f"({variance[0] * 100:.1f}% variance)"
    )
    axis.set_ylabel(
        f"PC2: {component_names['PC2']} "
        f"({variance[1] * 100:.1f}% variance)"
    )
    axis.set_title(
        "Diagnosis-defined T2D test participants\n"
        "Frozen validation clinical PCA"
    )
    fig.tight_layout()
    fig.savefig(FIGURE_SCATTER_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return int(valid.sum())


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    np.random.seed(RANDOM_SEED)

    input_paths = {
        "static_table": STATIC_PATH,
        "split_manifest": SPLIT_PATH,
        "coverage_audit": COVERAGE_AUDIT_PATH,
        "clinical_subtype_labels": SUBTYPE_LABELS_PATH,
        "clinical_subtype_manifest": SUBTYPE_MANIFEST_PATH,
        "hidden_cluster_labels": HIDDEN_CLUSTER_LABELS_PATH,
        "hidden_cluster_manifest": HIDDEN_CLUSTER_MANIFEST_PATH,
        "test_glycemic_features": TEST_GLYCEMIC_PATH,
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required prior artifacts: {missing}")

    coverage = pd.read_csv(COVERAGE_AUDIT_PATH)
    audit_counts = (
        coverage[coverage["metric"] == "complete_case_four_marker"]
        .set_index("split")["count"]
        .astype(int)
        .to_dict()
    )
    if audit_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Coverage audit counts changed: {audit_counts} != {EXPECTED_COUNTS}"
        )

    frame = load_complete_cases()
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()

    scaler = StandardScaler().fit(validation[RAW_FEATURES])
    validation_z = scaler.transform(validation[RAW_FEATURES])
    test_z = scaler.transform(test[RAW_FEATURES])
    pca = PCA(n_components=3, random_state=RANDOM_SEED).fit(validation_z)
    validation_pc = pca.transform(validation_z)
    test_pc = pca.transform(test_z)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, PCA_PATH)

    for target, z_values, pc_values in (
        (validation, validation_z, validation_pc),
        (test, test_z, test_pc),
    ):
        target[STANDARDIZED_FEATURES] = z_values
        target["clinical_pc1"] = pc_values[:, 0]
        target["clinical_pc2"] = pc_values[:, 1]

    target_columns = [
        "participant_id",
        "split",
        "bmi",
        "log_tg_hdl",
        "log_c_peptide",
        "clinical_pc1",
        "clinical_pc2",
        "participants_age",
        *STANDARDIZED_FEATURES,
    ]
    targets = pd.concat(
        [validation[target_columns], test[target_columns]],
        ignore_index=True,
    )
    if not np.isfinite(
        targets.drop(columns=["participant_id", "split"]).to_numpy(float)
    ).all():
        raise RuntimeError("Nonfinite values in clinical target table")
    component_names, descriptions, qualifies = classify_components(
        pca.components_
    )
    write_targets(targets, component_names, descriptions)
    loadings = []
    marker_names = ["bmi", "log_tg_hdl", "log_c_peptide"]
    for component_index in range(2):
        component = f"PC{component_index + 1}"
        for marker, value in zip(
            marker_names, pca.components_[component_index]
        ):
            loadings.append(
                {
                    "component": component,
                    "component_name": component_names[component],
                    "marker": marker,
                    "loading": float(value),
                    "absolute_loading": float(abs(value)),
                    "variance_explained": float(
                        pca.explained_variance_ratio_[component_index]
                    ),
                    "variance_explained_percent": float(
                        100 * pca.explained_variance_ratio_[component_index]
                    ),
                    "description": descriptions[component],
                    "qualifies_as_insulin_resistance_axis": qualifies[component],
                    "nontrivial_loading_threshold": (
                        NONTRIVIAL_LOADING_THRESHOLD
                    ),
                    "nonfasting_caveat": NONFASTING_CAVEAT,
                }
            )
    loading_table = pd.DataFrame(loadings)
    loading_table.to_csv(LOADINGS_PATH, index=False)

    prior_aris = compute_prior_aris()
    negative_result = NEGATIVE_RESULT_TEMPLATE.format(**prior_aris)
    NEGATIVE_RESULT_PATH.write_text(negative_result + "\n")

    plot_loadings(
        loading_table, pca.explained_variance_ratio_, component_names
    )
    test_hba1c_n = plot_test_scatter(
        targets, pca.explained_variance_ratio_, component_names
    )

    output_paths = {
        "clinical_targets": TARGETS_PATH,
        "pca_loadings": LOADINGS_PATH,
        "negative_result": NEGATIVE_RESULT_PATH,
        "figure_loadings": FIGURE_LOADINGS_PATH,
        "figure_scatter": FIGURE_SCATTER_PATH,
        "frozen_scaler": SCALER_PATH,
        "frozen_pca": PCA_PATH,
    }
    pc1_under_50 = bool(pca.explained_variance_ratio_[0] < 0.50)
    manifest = {
        "analysis": "Continuous clinical phenotype targets, diagnosis-defined T2D only",
        "status": "QC_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(OUTPUT_ROOT),
        "cohort": {
            "definition_field": "participants_study_group",
            "definition_values": sorted(T2D_VALUES),
            "complete_case_counts": EXPECTED_COUNTS,
            "participant_total": len(targets),
        },
        "transforms": {
            "tg_hdl": "triglycerides_mgdl_baseline / hdl_cholesterol_mgdl_baseline",
            "log_tg_hdl": "natural logarithm of tg_hdl",
            "log_c_peptide": "natural logarithm of c_peptide_ngml_baseline",
            "standardization": (
                "StandardScaler fit on diagnosis-defined T2D validation "
                "complete cases only, then frozen on test"
            ),
            "scaler_validation_means": dict(
                zip(RAW_FEATURES, scaler.mean_.tolist())
            ),
            "scaler_validation_scales": dict(
                zip(RAW_FEATURES, scaler.scale_.tolist())
            ),
        },
        "target_columns": {
            "raw_transformed": RAW_FEATURES,
            "standardized_primary": STANDARDIZED_FEATURES,
            "pca_co_primary": ["clinical_pc1", "clinical_pc2"],
            "age_covariate": "participants_age",
        },
        "age_definition": AGE_DEFINITION,
        "nonfasting_caveat": NONFASTING_CAVEAT,
        "pca": {
            "fit_population": "T2D validation complete cases only",
            "test_application": "frozen, without refit",
            "explained_variance_ratio": {
                f"PC{index + 1}": float(value)
                for index, value in enumerate(pca.explained_variance_ratio_)
            },
            "pc1_pc2_combined_variance_ratio": float(
                pca.explained_variance_ratio_[:2].sum()
            ),
            "component_names": component_names,
            "component_descriptions": descriptions,
            "qualifies_as_insulin_resistance_axis": qualifies,
            "nontrivial_loading_threshold": (
                NONTRIVIAL_LOADING_THRESHOLD
            ),
            "pc1_under_50_percent": pc1_under_50,
            "downstream_rule": (
                "PC1 and PC2 are co-primary in all downstream analyses."
                if pc1_under_50
                else "PC1 is primary and PC2 is secondary."
            ),
        },
        "loading_table": loading_table[
            ["component", "marker", "loading"]
        ].to_dict(orient="records"),
        "test_scatter": {
            "test_complete_case_n": EXPECTED_COUNTS["test"],
            "hba1c_nonmissing_n": test_hba1c_n,
            "color_variable": "HbA1c (%)",
        },
        "discrete_clustering_negative_result": {
            "paragraph": negative_result,
            "test_ari": prior_aris,
            "source": (
                "Read from prior frozen cluster and subtype assignments; "
                "clustering was not rerun."
            ),
        },
        "validation_fit_test_frozen": True,
        "input_paths": {
            name: str(path) for name, path in input_paths.items()
        },
        "input_hashes": {
            name: sha256_file(path) for name, path in input_paths.items()
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

    print("PCA loading table, inspected before component naming:")
    print(
        loading_table[
            [
                "component",
                "component_name",
                "marker",
                "loading",
                "variance_explained_percent",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        json.dumps(
            {
                "output_directory": str(OUTPUT_ROOT),
                "counts": EXPECTED_COUNTS,
                "component_names": component_names,
                "component_descriptions": descriptions,
                "variance_explained_percent": {
                    "PC1": 100 * pca.explained_variance_ratio_[0],
                    "PC2": 100 * pca.explained_variance_ratio_[1],
                    "combined": 100 * pca.explained_variance_ratio_[:2].sum(),
                },
                "downstream_rule": manifest["pca"]["downstream_rule"],
                "prior_test_ari": prior_aris,
                "manifest": str(MANIFEST_PATH),
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
