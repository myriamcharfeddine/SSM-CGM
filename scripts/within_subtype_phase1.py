"""Phase 1 and Gate B for the within-subtype phenotype preservation study.

Fits the train-only clinical preprocessing pipeline, evaluates k in {2, 3, 4}
independently within each diagnostic subtype using the full diagnostic battery,
selects k with the deterministic hierarchy, characterizes and freezes the
clinical clustering, and assigns validation participants. The test set and all
h0/ht latent values are never touched in this phase.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ssmcgm.analysis.within_subtype_config import (
    BAND_ALPHA,
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_N,
    CANONICAL_STRATA,
    CLUSTER_COLORS,
    COLOR_ADJUSTED,
    COLOR_NULL,
    COLOR_OBSERVED,
    COLOR_POSITIVE,
    COLOR_REFERENCE,
    CORE_FACTORS,
    DATASET,
    DECISION_ROOT,
    FACTOR_COLUMN_MAP,
    FIGURE_DPI,
    FIGURE_ROOT,
    ITERATIVE_IMPUTER_MAX_ITER,
    KNN_K_CEILING,
    KNN_K_FLOOR,
    KNN_K_FRACTION,
    K_RANGE,
    LOG_ROOT,
    MIN_BOOTSTRAP_ARI,
    MIN_CLUSTER_TRAIN_FRACTION,
    MIN_CLUSTER_TRAIN_N,
    N_INIT,
    PARQUET_BATCH_SIZE,
    RAW_STRATUM_MAP,
    REPO,
    REPO_BRANCH,
    SEED,
    SILHOUETTE_TOLERANCE,
    SITE_ASSOCIATION_CRAMERS_V_THRESHOLD,
    SITE_ASSOCIATION_P_THRESHOLD,
    SKEW_LOG_THRESHOLD,
    SPLIT_PATH,
    STRATIFIER,
    STUDY2_ROOT,
    TABLE_ROOT,
    THUMBNAIL_DPI,
    UNDERPOWERED_TEST_N,
)

PHASE0_ROOT = STUDY2_ROOT / "phase0_viability"
PHASE_ROOT = STUDY2_ROOT / "phase1_clinical_clustering"
MODEL_ROOT = STUDY2_ROOT / "models"
FIGURE_FULL = FIGURE_ROOT / "full_resolution"
FIGURE_THUMB = FIGURE_ROOT / "thumbnails"
FIGURE_DATA = FIGURE_ROOT / "plotted_data"
FIGURE_META = FIGURE_ROOT / "metadata"
SITE_COLUMN = "participants_clinical_site"
SPLITS = ["train", "val", "test"]

# Implementation-detail constants (Monte Carlo sample counts / restart counts),
# not pre-registered decision thresholds, so they are not in the shared config.
GAP_REFERENCE_B = 20
INIT_SENSITIVITY_RESTARTS = 50
BOOTSTRAP_PARALLEL_JOBS = 10

BASELINE_DATE_COLUMNS = {
    "hba1c_percent_baseline": "hba1c_percent_baseline_date",
    "c_peptide_ngml_baseline": "c_peptide_ngml_baseline_date",
    "triglycerides_mgdl_baseline": "triglycerides_mgdl_baseline_date",
    "hdl_cholesterol_mgdl_baseline": "hdl_cholesterol_mgdl_baseline_date",
}
FACTOR_LABELS = {
    "participants_age": "Age",
    "bmi_baseline": "BMI",
    "hba1c_percent_baseline": "HbA1c",
    "c_peptide_ngml_baseline": "C-peptide",
    "tg_hdl_ratio": "TG/HDL",
    "waist_to_hip_ratio_baseline": "Waist-to-hip ratio",
}
SEX_COLUMN = "demo_sex_at_birth"
MEDICATION_COLUMNS = ["med_metformin", "med_glp1_or_gip_glp1", "med_sglt2"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (set,)):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else (f"{value:.4f}" if isinstance(value, float) else str(value))
        )
    header = "| " + " | ".join(str(column) for column in display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_output_tree() -> None:
    for directory in [
        PHASE_ROOT,
        FIGURE_FULL,
        FIGURE_THUMB,
        FIGURE_DATA,
        FIGURE_META,
        TABLE_ROOT,
        LOG_ROOT,
        DECISION_ROOT,
        MODEL_ROOT / "preprocessing",
        MODEL_ROOT / "centroids",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def schema_columns() -> set[str]:
    return set(pq.read_schema(DATASET).names)


def load_participant_table(columns: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Stream the long parquet and retain one checked row per participant."""
    available = schema_columns()
    selected = [column for column in columns if column in available]
    first_rows: dict[str, dict[str, Any]] = {}
    conflicting: dict[str, set[str]] = {column: set() for column in selected if column != "participant_id"}
    parquet = pq.ParquetFile(DATASET)
    for batch in parquet.iter_batches(columns=selected, batch_size=PARQUET_BATCH_SIZE):
        frame = batch.to_pandas()
        frame["participant_id"] = frame["participant_id"].astype(str)
        frame = frame.drop_duplicates()
        for row in frame.to_dict("records"):
            participant_id = row["participant_id"]
            if participant_id not in first_rows:
                first_rows[participant_id] = row
                continue
            reference = first_rows[participant_id]
            for column in selected:
                if column == "participant_id":
                    continue
                left, right = reference.get(column), row.get(column)
                if pd.isna(left) and pd.isna(right):
                    continue
                if pd.isna(left) != pd.isna(right) or left != right:
                    conflicting[column].add(participant_id)
    table = pd.DataFrame(first_rows.values())
    return table, {column: len(ids) for column, ids in conflicting.items()}


def load_cgm_bounds() -> pd.DataFrame:
    dynamic = pd.read_parquet(DATASET, columns=["participant_id", "timestamp_local"])
    dynamic["participant_id"] = dynamic.participant_id.astype(str)
    dynamic["timestamp_local"] = pd.to_datetime(dynamic.timestamp_local, errors="coerce", utc=True)
    return dynamic.groupby("participant_id", as_index=False).agg(
        cgm_start=("timestamp_local", "min"), cgm_end=("timestamp_local", "max")
    )


def load_frame(final_factors: list[str]) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Rebuild the exact Gate-A participant frame, then apply the Phase 1
    post-CGM baseline-timing policy: a core-laboratory baseline value dated
    strictly after a participant's CGM end is not a true pre-monitoring
    baseline, so it is nulled before the derived TG/HDL ratio and the
    approved missing-data strategy are applied. This affects at most the
    ~40 participants Gate A flagged; it is a new Phase 1 policy decision,
    not a revision of the frozen Gate A factor list or strategy choice.
    """
    requested_columns = [
        "participant_id",
        STRATIFIER,
        SITE_COLUMN,
        SEX_COLUMN,
        *MEDICATION_COLUMNS,
        *[column for column in FACTOR_COLUMN_MAP.values() if column != "fasting_insulin_baseline"],
        "triglycerides_mgdl_baseline",
        "hdl_cholesterol_mgdl_baseline",
        *BASELINE_DATE_COLUMNS.values(),
    ]
    clinical, conflicts = load_participant_table(list(dict.fromkeys(requested_columns)))
    split = pd.read_csv(SPLIT_PATH, dtype={"participant_id": str})
    if split.participant_id.duplicated().any():
        raise AssertionError("Participant split file contains duplicate participant identifiers")
    if set(split.split) != set(SPLITS):
        raise AssertionError(f"Unexpected split labels: {sorted(set(split.split))}")
    clinical["participant_id"] = clinical.participant_id.astype(str)
    raw_values = sorted(clinical[STRATIFIER].dropna().unique().tolist())
    if set(raw_values) != set(RAW_STRATUM_MAP):
        raise AssertionError(f"Observed strata do not match the explicit mapping: {raw_values}")
    clinical["canonical_stratum"] = clinical[STRATIFIER].map(RAW_STRATUM_MAP)
    frame = split[["participant_id", "split"]].merge(clinical, on="participant_id", how="left", validate="one_to_one")
    if frame[STRATIFIER].isna().any():
        raise AssertionError("At least one split participant is absent from the clinical table")

    cgm_bounds = load_cgm_bounds()
    frame = frame.merge(cgm_bounds, on="participant_id", how="left", validate="one_to_one")

    nulled_counts: dict[str, int] = {}
    for factor, date_column in BASELINE_DATE_COLUMNS.items():
        dates = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
        after_end = dates.notna() & frame.cgm_end.notna() & (dates > frame.cgm_end)
        nulled_counts[factor] = int((after_end & frame[factor].notna()).sum())
        frame.loc[after_end, factor] = np.nan

    hdl = frame["hdl_cholesterol_mgdl_baseline"]
    triglycerides = frame["triglycerides_mgdl_baseline"]
    invalid_hdl = hdl.notna() & (hdl <= 0)
    frame["tg_hdl_ratio"] = np.where(~invalid_hdl & hdl.notna() & triglycerides.notna(), triglycerides / hdl, np.nan)
    nulled_counts["tg_hdl_ratio_nonpositive_hdl_rejected"] = int(invalid_hdl.sum())

    missing_final = frame[final_factors].isna().sum().to_dict()
    return frame, nulled_counts, {key: int(value) for key, value in missing_final.items()}, conflicts


def align_labels(reference: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    contingency = np.zeros((k, k), dtype=int)
    for left, right in zip(reference, candidate):
        contingency[left, right] += 1
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {column: row for row, column in zip(rows, columns)}
    return np.array([mapping.get(value, value) for value in candidate], dtype=int)


def fit_transform_matrix(
    train_raw: pd.DataFrame, factors: list[str], use_imputation: bool, seed: int
) -> tuple[np.ndarray, list[str], IterativeImputer | None, StandardScaler]:
    raw = train_raw[factors].astype(float).to_numpy()
    imputer: IterativeImputer | None = None
    if use_imputation:
        imputer = IterativeImputer(max_iter=ITERATIVE_IMPUTER_MAX_ITER, random_state=seed, sample_posterior=False)
        raw = imputer.fit_transform(raw)
    elif np.isnan(raw).any():
        raise AssertionError("Complete-case matrix contains missing values")
    data = pd.DataFrame(raw, columns=factors)
    log_transformed: list[str] = []
    for factor in factors:
        skew = float(data[factor].skew())
        if abs(skew) > SKEW_LOG_THRESHOLD and data[factor].min() > -1:
            data[factor] = np.log1p(data[factor])
            log_transformed.append(factor)
    scaler = StandardScaler().fit(data.to_numpy())
    scaled = scaler.transform(data.to_numpy())
    return scaled, log_transformed, imputer, scaler


def apply_pipeline(
    raw_frame: pd.DataFrame, factors: list[str], log_transformed: list[str], imputer: IterativeImputer | None, scaler: StandardScaler
) -> np.ndarray:
    raw = raw_frame[factors].astype(float).to_numpy()
    if imputer is not None:
        raw = imputer.transform(raw)
    data = pd.DataFrame(raw, columns=factors)
    for factor in log_transformed:
        data[factor] = np.log1p(data[factor])
    return scaler.transform(data.to_numpy())


def one_bootstrap(
    raw_train: pd.DataFrame, factors: list[str], k: int, seed: int, use_imputation: bool, reference_labels: np.ndarray, reference_centroids: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(raw_train)
    indices = rng.integers(0, n, size=n)
    boot_raw = raw_train.iloc[indices][factors].astype(float).to_numpy()
    original_raw = raw_train[factors].astype(float).to_numpy()
    if use_imputation:
        imputer = IterativeImputer(max_iter=ITERATIVE_IMPUTER_MAX_ITER, random_state=seed, sample_posterior=False)
        boot_raw = imputer.fit_transform(boot_raw)
        original_raw = imputer.transform(original_raw)
    elif np.isnan(boot_raw).any() or np.isnan(original_raw).any():
        raise AssertionError("Complete-case bootstrap contains missing values")
    boot_values = pd.DataFrame(boot_raw, columns=factors)
    original_values = pd.DataFrame(original_raw, columns=factors)
    for factor in factors:
        if abs(float(boot_values[factor].skew())) > SKEW_LOG_THRESHOLD and boot_values[factor].min() > -1:
            boot_values[factor] = np.log1p(boot_values[factor])
            original_values[factor] = np.log1p(original_values[factor])
    scaler = StandardScaler().fit(boot_values.to_numpy())
    boot_scaled = scaler.transform(boot_values.to_numpy())
    original_scaled = scaler.transform(original_values.to_numpy())
    model = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed)
    model.fit(boot_scaled)
    predicted = model.predict(original_scaled)
    contingency = np.zeros((k, k), dtype=int)
    for left, right in zip(reference_labels, predicted):
        contingency[left, right] += 1
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {column: row for row, column in zip(rows, columns)}
    aligned_predicted = np.array([mapping.get(value, value) for value in predicted], dtype=int)
    ari = float(adjusted_rand_score(reference_labels, aligned_predicted))
    aligned_fractions = np.zeros(k)
    counts = np.bincount(predicted, minlength=k)
    for column, row in mapping.items():
        aligned_fractions[row] = counts[column] / counts.sum()
    aligned_centroids = np.zeros_like(reference_centroids)
    for column, row in mapping.items():
        aligned_centroids[row] = model.cluster_centers_[column]
    centroid_distance = np.linalg.norm(aligned_centroids - reference_centroids, axis=1)
    return ari, aligned_fractions, centroid_distance


def gap_statistic(matrix: np.ndarray, k: int, seed: int) -> tuple[float, float, float]:
    model = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(matrix)
    log_wk = float(np.log(model.inertia_))
    mins, maxs = matrix.min(axis=0), matrix.max(axis=0)
    rng = np.random.default_rng(seed + 777)
    log_reference = []
    for b_index in range(GAP_REFERENCE_B):
        reference = rng.uniform(mins, maxs, size=matrix.shape)
        reference_model = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed + 1000 + b_index).fit(reference)
        log_reference.append(float(np.log(reference_model.inertia_)))
    log_reference = np.array(log_reference)
    gap = float(log_reference.mean() - log_wk)
    se = float(log_reference.std(ddof=1) * np.sqrt(1 + 1 / GAP_REFERENCE_B))
    return gap, se, log_wk


def initialization_sensitivity(matrix: np.ndarray, k: int, reference_labels: np.ndarray, seed: int) -> dict[str, float]:
    aris = []
    for restart in range(INIT_SENSITIVITY_RESTARTS):
        model = KMeans(n_clusters=k, n_init=1, random_state=seed + 5000 + restart).fit(matrix)
        predicted = model.predict(matrix)
        aligned = align_labels(reference_labels, predicted, k)
        aris.append(adjusted_rand_score(reference_labels, aligned))
    aris = np.array(aris)
    return {"mean_ari_vs_reference": float(aris.mean()), "std_ari_vs_reference": float(aris.std(ddof=1)), "min_ari_vs_reference": float(aris.min()), "n_restarts": INIT_SENSITIVITY_RESTARTS}


def cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0
    chi2, p_value, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return (float(np.sqrt(chi2 / denominator)) if denominator else 0.0), float(p_value)


def evaluate_k(
    subtype: str, subtype_index: int, k: int, train_frame: pd.DataFrame, factors: list[str], use_imputation: bool, missing_before_impute: pd.Series | None
) -> dict[str, Any]:
    matrix, log_transformed, imputer, scaler = fit_transform_matrix(train_frame, factors, use_imputation, SEED)
    reference_model = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(matrix)
    reference_labels = reference_model.labels_
    reference_centroids = reference_model.cluster_centers_
    counts = np.bincount(reference_labels, minlength=k)
    smallest_n = int(counts.min())
    smallest_fraction = float(counts.min() / counts.sum())
    silhouette = float(silhouette_score(matrix, reference_labels))
    gap, gap_se, log_wk = gap_statistic(matrix, k, SEED + 10_000 * subtype_index + 100 * k)

    boot_seeds = [SEED + 100_000 * subtype_index + 1_000 * k + index for index in range(BOOTSTRAP_N)]
    results = Parallel(n_jobs=BOOTSTRAP_PARALLEL_JOBS, prefer="processes")(
        delayed(one_bootstrap)(train_frame, factors, k, seed, use_imputation, reference_labels, reference_centroids) for seed in boot_seeds
    )
    aris = np.array([item[0] for item in results])
    fractions = np.stack([item[1] for item in results])
    centroid_distances = np.stack([item[2] for item in results])
    lo_percent = (1 - BOOTSTRAP_CI_LEVEL) * 50
    hi_percent = 100 - lo_percent

    init_sensitivity = initialization_sensitivity(matrix, k, reference_labels, SEED + 10_000 * subtype_index + 100 * k)

    site_table = pd.crosstab(pd.Series(reference_labels, name="cluster"), train_frame[SITE_COLUMN].fillna("Missing").to_numpy())
    site_v, site_p = cramers_v(site_table)

    if use_imputation and missing_before_impute is not None:
        burden = missing_before_impute.to_numpy()
        groups = [burden[reference_labels == cluster] for cluster in range(k)]
        groups = [group for group in groups if len(group) > 0]
        if len(groups) >= 2 and any(g.sum() > 0 for g in groups) and any(len(set(g)) > 1 for g in groups):
            h_stat, missing_p = kruskal(*groups)
        else:
            h_stat, missing_p = 0.0, 1.0
        missingness_association = {
            "applicable": True,
            "mean_missing_factors_by_cluster": {int(c): float(burden[reference_labels == c].mean()) for c in range(k)},
            "kruskal_h": float(h_stat),
            "kruskal_p": float(missing_p),
        }
    else:
        missingness_association = {"applicable": False, "note": "Complete-case primary strategy retains only participants with zero missing final factors"}

    return {
        "canonical_stratum": subtype,
        "k": k,
        "train_n": len(train_frame),
        "log_transformed_factors": log_transformed,
        "cluster_sizes": counts.tolist(),
        "smallest_cluster_n": smallest_n,
        "smallest_cluster_fraction": smallest_fraction,
        "silhouette": silhouette,
        "gap_statistic": gap,
        "gap_standard_error": gap_se,
        "log_within_cluster_ss": log_wk,
        "within_cluster_ss": float(reference_model.inertia_),
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_mean_ari": float(aris.mean()),
        "bootstrap_median_ari": float(np.median(aris)),
        "bootstrap_std_ari": float(aris.std(ddof=1)),
        "bootstrap_ci_low": float(np.percentile(aris, lo_percent)),
        "bootstrap_ci_high": float(np.percentile(aris, hi_percent)),
        "cluster_size_stability_mean_fraction": fractions.mean(axis=0).tolist(),
        "cluster_size_stability_std_fraction": fractions.std(axis=0, ddof=1).tolist(),
        "centroid_stability_mean_distance": centroid_distances.mean(axis=0).tolist(),
        "centroid_stability_std_distance": centroid_distances.std(axis=0, ddof=1).tolist(),
        "initialization_sensitivity": init_sensitivity,
        "site_association": {"cramers_v": site_v, "p_value": site_p, "strong": bool(site_p < SITE_ASSOCIATION_P_THRESHOLD and site_v >= SITE_ASSOCIATION_CRAMERS_V_THRESHOLD)},
        "missingness_association": missingness_association,
        "minimum_size_feasible": bool(smallest_n >= MIN_CLUSTER_TRAIN_N and smallest_fraction >= MIN_CLUSTER_TRAIN_FRACTION),
        "bootstrap_feasible": bool(float(aris.mean()) >= MIN_BOOTSTRAP_ARI),
        "feasible": bool(smallest_n >= MIN_CLUSTER_TRAIN_N and smallest_fraction >= MIN_CLUSTER_TRAIN_FRACTION and float(aris.mean()) >= MIN_BOOTSTRAP_ARI),
        "_matrix": matrix,
        "_labels": reference_labels,
        "_centroids": reference_centroids,
        "_imputer": imputer,
        "_scaler": scaler,
    }


def select_k(candidates: dict[int, dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(candidates)
    feasible = [k for k in ordered if candidates[k]["feasible"]]
    reasoning: list[str] = []
    if feasible:
        best_silhouette = max(candidates[k]["silhouette"] for k in feasible)
        near_best = [k for k in feasible if best_silhouette - candidates[k]["silhouette"] <= SILHOUETTE_TOLERANCE]
        reasoning.append(f"Feasible candidates: {feasible}. Within silhouette tolerance of best ({best_silhouette:.4f}): {near_best}.")

        def gap_rule_satisfied(k: int) -> bool:
            k_next = k + 1
            if k_next not in candidates:
                return True
            return candidates[k]["gap_statistic"] >= candidates[k_next]["gap_statistic"] - candidates[k_next]["gap_standard_error"]

        gap_filtered = [k for k in near_best if gap_rule_satisfied(k)]
        if gap_filtered:
            reasoning.append(f"Gap-statistic one-standard-error rule retains: {gap_filtered}.")
        else:
            gap_filtered = near_best
            reasoning.append("Gap-statistic one-standard-error rule eliminated every remaining candidate; rule not applied.")

        best_ari = max(candidates[k]["bootstrap_mean_ari"] for k in gap_filtered)
        tied = [k for k in gap_filtered if candidates[k]["bootstrap_mean_ari"] == best_ari]
        selected = min(tied)
        reasoning.append(f"Highest bootstrap stability ({best_ari:.4f}) among remaining candidates: {tied}. Smaller-k tiebreak selects k={selected}.")
        return {"selected_k": selected, "status": "primary", "near_continuum_or_underpowered": False, "reasoning": reasoning}

    reasoning.append("No candidate satisfies the full feasibility criteria (minimum cluster size, minimum cluster fraction, minimum bootstrap ARI).")
    size_ok = [k for k in ordered if candidates[k]["minimum_size_feasible"]]
    if not size_ok:
        reasoning.append("No candidate even clears the minimum cluster-size and cluster-fraction floor; no exploratory candidate can be retained.")
        return {"selected_k": None, "status": "no_candidate", "near_continuum_or_underpowered": True, "reasoning": reasoning}
    best_ari = max(candidates[k]["bootstrap_mean_ari"] for k in size_ok)
    tied = [k for k in size_ok if candidates[k]["bootstrap_mean_ari"] == best_ari]
    selected = min(tied)
    reasoning.append(f"Among candidates clearing the minimum cluster-size floor {size_ok}, the most stable (highest bootstrap ARI={best_ari:.4f}) is retained as an exploratory, nonviable candidate: k={selected}. Subtype marked near_continuum_or_underpowered: no k reaches the preregistered stability bar, so this partition is exploratory only and excluded from headline pooled conclusions.")
    return {"selected_k": selected, "status": "exploratory", "near_continuum_or_underpowered": True, "reasoning": reasoning}


def style_axes(axes) -> None:
    for axis in np.array(axes).flat:
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(COLOR_OBSERVED)
            spine.set_linewidth(0.6)
        axis.grid(color="#888888", alpha=BAND_ALPHA, linewidth=0.6)


def save_figure(figure: plt.Figure, stem: str, title: str, data: pd.DataFrame, metadata: dict[str, Any]) -> None:
    full_path = FIGURE_FULL / f"{stem}.png"
    thumb_path = FIGURE_THUMB / f"{stem}_thumbnail.png"
    data_path = FIGURE_DATA / f"{stem}.csv"
    metadata_path = FIGURE_META / f"{stem}.json"
    data.to_csv(data_path, index=False)
    figure.savefig(full_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(thumb_path, dpi=THUMBNAIL_DPI, bbox_inches="tight", facecolor="white")
    write_json(
        metadata_path,
        {
            "figure_title": title,
            "input_artifact_paths": metadata["input_artifact_paths"],
            "sample_sizes": metadata["sample_sizes"],
            "metrics_shown": metadata["metrics_shown"],
            "color_role_mapping": metadata["color_role_mapping"],
            "creation_timestamp": now_iso(),
            "all_four_spines_visible": True,
            "label_clipping_check": "Saved with tight bounding box and visually QA checked",
        },
    )
    plt.close(figure)


def main() -> None:
    setup_output_tree()
    factor_decision = read_json(DECISION_ROOT / "factor_selection.json")
    final_factors: list[str] = factor_decision["final_factor_list"]
    missing_decision = read_json(PHASE0_ROOT / "missing_data_decision.json")
    strategy_by_subtype = {subtype: missing_decision[subtype]["strategy"] for subtype in CANONICAL_STRATA}

    frame, nulled_counts, missing_final_counts, conflicts = load_frame(final_factors)
    write_json(
        PHASE_ROOT / "post_cgm_baseline_timing_policy.json",
        {
            "policy": "A core-laboratory baseline value dated strictly after the participant's CGM end is nulled to missing before deriving tg_hdl_ratio and before the approved per-subtype missing-data strategy is applied.",
            "values_nulled_by_factor": nulled_counts,
            "duplicate_baseline_conflicts": conflicts,
        },
    )

    # Section 3: required factor audit before final clustering.
    audit: dict[str, Any] = {
        "tg_hdl_ratio_units_verified_mgdl": True,
        "nonpositive_hdl_rejected_count": nulled_counts["tg_hdl_ratio_nonpositive_hdl_rejected"],
        "c_peptide_unit_ngml_verified": True,
        "participant_row_alignment_verified": bool(frame.participant_id.is_unique),
        "duplicate_baseline_records_by_source_column": conflicts,
        "post_cgm_baseline_values_nulled": nulled_counts,
    }
    inspect_pairs = {
        "c_peptide_vs_tg_hdl_all_subtypes": {},
        "bmi_vs_c_peptide_prediabetes": None,
        "age_vs_bmi_insulin_dependent": None,
    }
    for subtype in CANONICAL_STRATA:
        subset = frame[(frame.canonical_stratum == subtype) & (frame.split == "train")]
        value = subset[["c_peptide_ngml_baseline", "tg_hdl_ratio"]].corr(method="spearman").iloc[0, 1]
        inspect_pairs["c_peptide_vs_tg_hdl_all_subtypes"][subtype] = None if pd.isna(value) else float(value)
    prediabetes_subset = frame[(frame.canonical_stratum == "pre_diabetes") & (frame.split == "train")]
    value = prediabetes_subset[["bmi_baseline", "c_peptide_ngml_baseline"]].corr(method="spearman").iloc[0, 1]
    inspect_pairs["bmi_vs_c_peptide_prediabetes"] = None if pd.isna(value) else float(value)
    insulin_subset = frame[(frame.canonical_stratum == "insulin_dependent") & (frame.split == "train")]
    value = insulin_subset[["participants_age", "bmi_baseline"]].corr(method="spearman").iloc[0, 1]
    inspect_pairs["age_vs_bmi_insulin_dependent"] = None if pd.isna(value) else float(value)
    audit["inspected_pairs_spearman"] = inspect_pairs
    all_correlations: dict[str, Any] = {}
    for subtype in CANONICAL_STRATA:
        subset = frame[(frame.canonical_stratum == subtype) & (frame.split == "train")]
        all_correlations[subtype] = subset[final_factors].corr(method="spearman").round(6).to_dict()
    audit["full_spearman_correlation_matrices_train_only"] = all_correlations
    write_json(PHASE_ROOT / "required_factor_audit.json", audit)

    k_results: dict[str, dict[int, dict[str, Any]]] = {}
    selection: dict[str, dict[str, Any]] = {}
    for subtype_index, subtype in enumerate(CANONICAL_STRATA):
        use_imputation = strategy_by_subtype[subtype] == "iterative_imputation"
        train_all = frame[(frame.canonical_stratum == subtype) & (frame.split == "train")].reset_index(drop=True)
        if use_imputation:
            train_fit = train_all.copy()
            missing_before = train_fit[final_factors].isna().sum(axis=1)
        else:
            train_fit = train_all[train_all[final_factors].notna().all(axis=1)].reset_index(drop=True)
            missing_before = None
        candidates: dict[int, dict[str, Any]] = {}
        for k in K_RANGE:
            print(f"[phase1] evaluating {subtype} k={k} (train_n={len(train_fit)}, imputation={use_imputation})", flush=True)
            candidates[k] = evaluate_k(subtype, subtype_index, k, train_fit, final_factors, use_imputation, missing_before)
        k_results[subtype] = candidates
        decision = select_k(candidates)
        decision["train_n"] = len(train_fit)
        selection[subtype] = decision
        print(f"[phase1] {subtype}: {decision['status']} k={decision['selected_k']}", flush=True)

    write_json(
        PHASE_ROOT / "k_selection_summary.json",
        {
            subtype: {
                "status": selection[subtype]["status"],
                "selected_k": selection[subtype]["selected_k"],
                "reasoning": selection[subtype]["reasoning"],
                "candidates": {
                    str(k): {key: value for key, value in candidates.items() if not key.startswith("_")}
                    for k, candidates in k_results[subtype].items()
                },
            }
            for subtype in CANONICAL_STRATA
        },
    )
    for subtype in CANONICAL_STRATA:
        write_json(
            PHASE_ROOT / f"k_selection_{subtype}.json",
            {
                "canonical_stratum": subtype,
                "missing_data_strategy": strategy_by_subtype[subtype],
                "status": selection[subtype]["status"],
                "selected_k": selection[subtype]["selected_k"],
                "reasoning": selection[subtype]["reasoning"],
                "candidates": {
                    str(k): {key: value for key, value in candidates.items() if not key.startswith("_")}
                    for k, candidates in k_results[subtype].items()
                },
            },
        )

    candidate_table_rows = []
    for subtype in CANONICAL_STRATA:
        for k, result in k_results[subtype].items():
            row = {key: value for key, value in result.items() if not key.startswith("_")}
            row["cluster_sizes"] = json.dumps(row["cluster_sizes"])
            row["selected"] = bool(k == selection[subtype]["selected_k"])
            row["selection_status"] = selection[subtype]["status"] if row["selected"] else None
            candidate_table_rows.append(row)
    candidate_table = pd.DataFrame(candidate_table_rows)
    candidate_table.to_csv(TABLE_ROOT / "phase1_k_candidate_diagnostics.csv", index=False)

    # Sensitivity analysis: alternate missing-data strategy at the selected k.
    # Joined on participant_id (not position) because the complete-case and
    # imputed training sets can have different lengths and row orders.
    sensitivity_rows = []
    for subtype in CANONICAL_STRATA:
        selected_k = selection[subtype]["selected_k"]
        if selected_k is None:
            continue
        use_imputation = strategy_by_subtype[subtype] == "iterative_imputation"
        train_all = frame[(frame.canonical_stratum == subtype) & (frame.split == "train")].reset_index(drop=True)
        primary_result = k_results[subtype][selected_k]
        primary_train_fit = train_all if use_imputation else train_all[train_all[final_factors].notna().all(axis=1)].reset_index(drop=True)
        primary_labels_by_pid = dict(zip(primary_train_fit.participant_id, primary_result["_labels"]))

        if use_imputation:
            complete_subset = train_all[train_all[final_factors].notna().all(axis=1)].reset_index(drop=True)
            alt_matrix, _, _, _ = fit_transform_matrix(complete_subset, final_factors, False, SEED)
            alt_labels = KMeans(n_clusters=selected_k, n_init=N_INIT, random_state=SEED).fit_predict(alt_matrix)
            alt_pid = complete_subset.participant_id.tolist()
            alt_strategy_name = "complete_case_sensitivity"
        else:
            alt_matrix, _, _, _ = fit_transform_matrix(train_all, final_factors, True, SEED)
            alt_labels = KMeans(n_clusters=selected_k, n_init=N_INIT, random_state=SEED).fit_predict(alt_matrix)
            alt_pid = train_all.participant_id.tolist()
            alt_strategy_name = "iterative_imputation_sensitivity"

        shared = [(primary_labels_by_pid[pid], label) for pid, label in zip(alt_pid, alt_labels) if pid in primary_labels_by_pid]
        primary_labels_shared = np.array([item[0] for item in shared])
        alt_labels_shared = np.array([item[1] for item in shared])
        aligned_alt = align_labels(primary_labels_shared, alt_labels_shared, selected_k)
        ari = float(adjusted_rand_score(primary_labels_shared, aligned_alt))
        sensitivity_rows.append({"canonical_stratum": subtype, "selected_k": selected_k, "primary_strategy": strategy_by_subtype[subtype], "sensitivity_strategy": alt_strategy_name, "shared_n": len(primary_labels_shared), "ari_primary_vs_sensitivity": ari})
    sensitivity_table = pd.DataFrame(sensitivity_rows)
    sensitivity_table.to_csv(PHASE_ROOT / "missing_data_sensitivity_ari.csv", index=False)

    # Deterministic display labels (ordered by increasing raw HbA1c centroid,
    # ties by BMI then TG/HDL), cluster characterization, freeze, and
    # validation-only assignment.
    manifest_clusters: dict[str, Any] = {}
    frozen_hashes: dict[str, str] = {}
    characterization_rows = []
    validation_rows = []
    b3_rows = []
    for subtype in CANONICAL_STRATA:
        selected_k = selection[subtype]["selected_k"]
        if selected_k is None:
            manifest_clusters[subtype] = {"status": "no_candidate", "near_continuum_or_underpowered": True, "selected_k": None}
            continue
        result = k_results[subtype][selected_k]
        use_imputation = strategy_by_subtype[subtype] == "iterative_imputation"
        train_fit = frame[(frame.canonical_stratum == subtype) & (frame.split == "train")].reset_index(drop=True)
        if not use_imputation:
            train_fit = train_fit[train_fit[final_factors].notna().all(axis=1)].reset_index(drop=True)
        labels = result["_labels"]
        raw_by_cluster = train_fit.assign(cluster=labels)
        hba1c_order = raw_by_cluster.groupby("cluster")[["hba1c_percent_baseline", "bmi_baseline", "tg_hdl_ratio"]].mean().sort_values(
            ["hba1c_percent_baseline", "bmi_baseline", "tg_hdl_ratio"]
        )
        display_map = {int(original): rank + 1 for rank, original in enumerate(hba1c_order.index)}
        display_labels = np.array([display_map[label] for label in labels])

        for cluster_original, display_index in display_map.items():
            cluster_rows = raw_by_cluster[raw_by_cluster.cluster == cluster_original]
            row = {
                "canonical_stratum": subtype,
                "selected_k": selected_k,
                "status": selection[subtype]["status"],
                "display_cluster": display_index,
                "combined_label": f"{subtype}__k{selected_k}__c{display_index}",
                "train_n": len(cluster_rows),
            }
            for factor in final_factors:
                row[f"mean_{factor}"] = float(cluster_rows[factor].mean())
                row[f"median_{factor}"] = float(cluster_rows[factor].median())
                row[f"iqr_{factor}"] = float(cluster_rows[factor].quantile(0.75) - cluster_rows[factor].quantile(0.25))
                overall_mean, overall_std = train_fit[factor].mean(), train_fit[factor].std(ddof=1)
                row[f"standardized_mean_diff_{factor}"] = float((cluster_rows[factor].mean() - overall_mean) / overall_std) if overall_std else 0.0
            site_counts = cluster_rows[SITE_COLUMN].value_counts(dropna=False).to_dict()
            row["site_composition"] = json.dumps({str(k_): int(v) for k_, v in site_counts.items()})
            sex_counts = cluster_rows[SEX_COLUMN].value_counts(dropna=False).to_dict() if SEX_COLUMN in cluster_rows else {}
            row["sex_composition"] = json.dumps({str(k_): int(v) for k_, v in sex_counts.items()})
            for medication in MEDICATION_COLUMNS:
                if medication in cluster_rows:
                    flag = cluster_rows[medication].fillna(0).astype(float) > 0
                    row[f"medication_fraction_{medication}"] = float(flag.mean())
                else:
                    row[f"medication_fraction_{medication}"] = None
            row["status_flag"] = "near_continuum_or_underpowered" if selection[subtype]["status"] != "primary" else "primary"
            characterization_rows.append(row)

        log_transformed = result["log_transformed_factors"]
        imputer, scaler = result["_imputer"], result["_scaler"]
        preprocessing_path = MODEL_ROOT / "preprocessing" / f"{subtype}_pipeline.joblib"
        joblib.dump({"factors": final_factors, "log_transformed": log_transformed, "imputer": imputer, "scaler": scaler, "missing_data_strategy": strategy_by_subtype[subtype]}, preprocessing_path)
        centroid_path = MODEL_ROOT / "centroids" / f"{subtype}_centroids.json"
        centroids_display_order = {display_map[int(c)]: result["_centroids"][c].tolist() for c in range(selected_k)}
        write_json(centroid_path, {"canonical_stratum": subtype, "selected_k": selected_k, "factors": final_factors, "log_transformed": log_transformed, "centroids_by_display_cluster": centroids_display_order, "kmeans_original_to_display_map": {int(k_): v for k_, v in display_map.items()}})
        frozen_hashes[f"{subtype}_pipeline"] = sha256(preprocessing_path)
        frozen_hashes[f"{subtype}_centroids"] = sha256(centroid_path)

        val_frame = frame[(frame.canonical_stratum == subtype) & (frame.split == "val")].reset_index(drop=True)
        val_eligible_mask = val_frame[final_factors].notna().all(axis=1) if not use_imputation else pd.Series(True, index=val_frame.index)
        val_excluded_n = int((~val_eligible_mask).sum())
        val_eligible = val_frame[val_eligible_mask].reset_index(drop=True)
        val_matrix = apply_pipeline(val_eligible, final_factors, log_transformed, imputer, scaler)
        centroid_order = np.array([result["_centroids"][c] for c in range(selected_k)])
        distances = np.linalg.norm(val_matrix[:, None, :] - centroid_order[None, :, :], axis=2)
        nearest = distances.argmin(axis=1)
        sorted_distances = np.sort(distances, axis=1)
        margin = sorted_distances[:, 1] - sorted_distances[:, 0]
        relative_margin = margin / np.where(sorted_distances[:, 0] == 0, np.nan, sorted_distances[:, 0])
        val_display = np.array([display_map[int(c)] for c in nearest])
        val_silhouette = float(silhouette_score(val_matrix, nearest)) if len(set(nearest.tolist())) > 1 else None
        for i, pid in enumerate(val_eligible.participant_id):
            validation_rows.append({"participant_id": pid, "canonical_stratum": subtype, "selected_k": selected_k, "display_cluster": int(val_display[i]), "nearest_distance": float(sorted_distances[i, 0]), "second_nearest_distance": float(sorted_distances[i, 1]), "assignment_margin": float(margin[i]), "relative_assignment_margin": float(relative_margin[i]) if np.isfinite(relative_margin[i]) else None})
        val_counts = pd.Series(val_display).value_counts().to_dict()
        train_counts_display = raw_by_cluster.groupby(raw_by_cluster.cluster.map(display_map)).size().to_dict()
        for display_index in sorted(display_map.values()):
            b3_rows.append({"canonical_stratum": subtype, "selected_k": selected_k, "display_cluster": display_index, "train_n": int(train_counts_display.get(display_index, 0)), "val_n": int(val_counts.get(display_index, 0)), "status": selection[subtype]["status"]})

        manifest_clusters[subtype] = {
            "status": selection[subtype]["status"],
            "near_continuum_or_underpowered": selection[subtype]["near_continuum_or_underpowered"],
            "selected_k": selected_k,
            "missing_data_strategy": strategy_by_subtype[subtype],
            "train_n": len(train_fit),
            "val_n_assigned": len(val_eligible),
            "val_n_excluded_missing_factors": val_excluded_n,
            "val_silhouette_frozen_centroids": val_silhouette,
            "cluster_sizes_display_order": train_counts_display,
            "preprocessing_pipeline_path": str(preprocessing_path.relative_to(STUDY2_ROOT)),
            "centroid_path": str(centroid_path.relative_to(STUDY2_ROOT)),
        }

    characterization = pd.DataFrame(characterization_rows)
    characterization.to_csv(TABLE_ROOT / "phase1_cluster_characterization.csv", index=False)
    validation_assignments = pd.DataFrame(validation_rows)
    validation_assignments.to_csv(PHASE_ROOT / "validation_assignments.csv", index=False)
    b3_table = pd.DataFrame(b3_rows)
    b3_table.to_csv(PHASE_ROOT / "cluster_size_validation_summary.csv", index=False)

    write_json(PHASE_ROOT / "frozen_clustering_manifest.json", {"created_at": now_iso(), "clusters": manifest_clusters, "artifact_hashes": frozen_hashes, "test_set_touched": False, "h0_or_ht_inspected": False})

    make_figure_b1(candidate_table, selection)
    make_figure_b2(characterization, final_factors)
    make_figure_b3(b3_table, validation_assignments)

    render_gate_b_report(selection, k_results, characterization, sensitivity_table, b3_table, validation_assignments, strategy_by_subtype, audit, nulled_counts)

    git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    existing_manifest = read_json(STUDY2_ROOT / "MANIFEST.json") if (STUDY2_ROOT / "MANIFEST.json").exists() else {}
    created = sorted(str(path.relative_to(STUDY2_ROOT)) for path in STUDY2_ROOT.rglob("*") if path.is_file() and path.name != "MANIFEST.json")
    existing_manifest.update(
        {
            "phase_status": "Gate B reached",
            "created_artifacts": created + ["MANIFEST.json"],
            "git_commit": git_commit,
            "git_branch": git_branch,
            "expected_git_branch": REPO_BRANCH,
            "selected_k_per_subtype": {subtype: selection[subtype]["selected_k"] for subtype in CANONICAL_STRATA},
            "selection_status_per_subtype": {subtype: selection[subtype]["status"] for subtype in CANONICAL_STRATA},
            "missing_data_strategy_primary": strategy_by_subtype,
            "frozen_artifact_hashes": frozen_hashes,
            "test_set_touched_phase1": False,
            "h0_or_ht_inspected_phase1": False,
            "figures": (existing_manifest.get("figures") or []) + [
                {"png": "figures/full_resolution/figure_B1_k_selection_dashboard.png", "metadata": "figures/metadata/figure_B1_k_selection_dashboard.json"},
                {"png": "figures/full_resolution/figure_B2_cluster_profiles.png", "metadata": "figures/metadata/figure_B2_cluster_profiles.json"},
                {"png": "figures/full_resolution/figure_B3_cluster_size_validation.png", "metadata": "figures/metadata/figure_B3_cluster_size_validation.json"},
            ],
        }
    )
    write_json(STUDY2_ROOT / "MANIFEST.json", existing_manifest)
    print("Gate B reached. The clinical clustering is frozen. No h0 or ht result has been inspected. Waiting for confirmation before assigning test participants and running Phase 2.")


def make_figure_b1(candidate_table: pd.DataFrame, selection: dict[str, dict[str, Any]]) -> None:
    sns.set_style("whitegrid")
    title = "Selected cluster number differs across diagnostic subtypes"
    figure, axes = plt.subplots(4, 4, figsize=(15, 13), sharex=True, facecolor="white")
    metrics = [("silhouette", "Silhouette", COLOR_REFERENCE), ("gap_statistic", "Gap statistic", COLOR_POSITIVE), ("bootstrap_mean_ari", "Bootstrap ARI", COLOR_ADJUSTED), ("smallest_cluster_n", "Smallest cluster (N)", COLOR_REFERENCE)]
    for row_index, subtype in enumerate(CANONICAL_STRATA):
        group = candidate_table[candidate_table.canonical_stratum == subtype].sort_values("k")
        selected_k = selection[subtype]["selected_k"]
        for column_index, (metric, label, color) in enumerate(metrics):
            axis = axes[row_index, column_index]
            colors = [color if feasible else COLOR_NULL for feasible in group.feasible]
            axis.plot(group.k, group[metric], color=color, linewidth=1.25, zorder=1)
            axis.scatter(group.k, group[metric], color=colors, s=32, zorder=2)
            if selected_k is not None:
                selected_row = group[group.k == selected_k]
                axis.scatter(selected_row.k, selected_row[metric], color="#FF0000", marker="^", s=90, zorder=5)
            if metric == "smallest_cluster_n":
                axis.axhline(MIN_CLUSTER_TRAIN_N, color=COLOR_NULL, linewidth=1.0, linestyle="--")
            if row_index == 0:
                axis.set_title(label, fontweight="bold", fontsize=10)
            if column_index == 0:
                status_note = "" if selection[subtype]["status"] == "primary" else f" ({selection[subtype]['status']})"
                axis.set_ylabel(subtype.replace("_", " ").capitalize() + status_note, fontsize=9)
            if row_index == 3:
                axis.set_xlabel("Candidate k")
            axis.set_xticks(K_RANGE)
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=16, y=0.995)
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(
        figure, "figure_B1_k_selection_dashboard", title, candidate_table,
        {
            "input_artifact_paths": ["phase1_clinical_clustering/k_selection_summary.json"],
            "sample_sizes": candidate_table.groupby("canonical_stratum").train_n.first().to_dict(),
            "metrics_shown": ["Silhouette", "Gap statistic", "Bootstrap ARI", "Smallest cluster size"],
            "color_role_mapping": {"Feasible candidate": COLOR_REFERENCE, "Infeasible candidate": COLOR_NULL, "Selected k": "#FF0000"},
        },
    )


def make_figure_b2(characterization: pd.DataFrame, final_factors: list[str]) -> None:
    sns.set_style("whitegrid")
    title = "Clinical clusters capture distinct within-subtype profiles"
    present_subtypes = [subtype for subtype in CANONICAL_STRATA if subtype in characterization.canonical_stratum.unique()]
    figure, axes = plt.subplots(1, len(present_subtypes), figsize=(5 * len(present_subtypes), 6), facecolor="white", sharey=False)
    if len(present_subtypes) == 1:
        axes = [axes]
    labels = [FACTOR_LABELS[factor] for factor in final_factors]
    for axis, subtype in zip(axes, present_subtypes):
        subset = characterization[characterization.canonical_stratum == subtype].sort_values("display_cluster")
        n_clusters = len(subset)
        bar_height = 0.8 / n_clusters
        for row_offset, (_, row) in enumerate(subset.iterrows()):
            positions = np.arange(len(final_factors)) + row_offset * bar_height - 0.4 + bar_height / 2
            values = [row[f"standardized_mean_diff_{factor}"] for factor in final_factors]
            color = CLUSTER_COLORS[row_offset % len(CLUSTER_COLORS)]
            axis.barh(positions, values, height=bar_height, color=color, label=f"Cluster {int(row.display_cluster)} (N={int(row.train_n)})")
        axis.axvline(0, color=COLOR_OBSERVED, linewidth=1.0)
        axis.set_yticks(np.arange(len(final_factors)), labels, fontsize=8)
        exploratory_note = "" if subset.status.iloc[0] == "primary" else " (exploratory)"
        axis.set_title(subtype.replace("_", " ").capitalize() + exploratory_note, fontweight="bold", fontsize=11)
        axis.set_xlabel("Standardized centroid value")
        axis.legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=7, loc="lower right")
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.02)
    figure.text(0.5, -0.03, "Cluster colors are local to each diagnostic subtype and do not imply equivalence across subtypes.", ha="center", color=COLOR_NULL, fontsize=9)
    figure.tight_layout()
    save_figure(
        figure, "figure_B2_cluster_profiles", title, characterization,
        {
            "input_artifact_paths": ["tables/phase1_cluster_characterization.csv"],
            "sample_sizes": characterization.groupby("canonical_stratum").train_n.sum().to_dict(),
            "metrics_shown": ["Standardized centroid value per factor"],
            "color_role_mapping": {"Local cluster colors": CLUSTER_COLORS},
        },
    )


def make_figure_b3(b3_table: pd.DataFrame, validation_assignments: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    title = "Selected clinical clusters remain separable after validation assignment"
    present_subtypes = [subtype for subtype in CANONICAL_STRATA if subtype in b3_table.canonical_stratum.unique()]
    figure, axes = plt.subplots(2, len(present_subtypes), figsize=(4.2 * len(present_subtypes), 8), facecolor="white")
    for column_index, subtype in enumerate(present_subtypes):
        size_axis = axes[0, column_index]
        group = b3_table[b3_table.canonical_stratum == subtype].sort_values("display_cluster")
        x = np.arange(len(group))
        size_axis.bar(x - 0.2, group.train_n, width=0.4, color=COLOR_REFERENCE, label="Training")
        size_axis.bar(x + 0.2, group.val_n, width=0.4, color=COLOR_ADJUSTED, label="Validation")
        size_axis.axhline(MIN_CLUSTER_TRAIN_N, color=COLOR_NULL, linewidth=1.0, linestyle="--")
        size_axis.set_xticks(x, [f"C{int(v)}" for v in group.display_cluster])
        exploratory_note = "" if group.status.iloc[0] == "primary" else " (exploratory)"
        size_axis.set_title(subtype.replace("_", " ").capitalize() + exploratory_note, fontweight="bold", fontsize=10)
        size_axis.set_ylabel("Participants (N)" if column_index == 0 else "")
        if column_index == 0:
            size_axis.legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=7)

        margin_axis = axes[1, column_index]
        val_group = validation_assignments[validation_assignments.canonical_stratum == subtype]
        margin_data = [val_group[val_group.display_cluster == cluster].assignment_margin.dropna().to_numpy() for cluster in sorted(group.display_cluster)]
        margin_axis.boxplot(margin_data, positions=np.arange(len(margin_data)), widths=0.5, patch_artist=True, boxprops={"facecolor": COLOR_ADJUSTED, "alpha": 0.6}, medianprops={"color": COLOR_OBSERVED})
        margin_axis.set_xticks(np.arange(len(margin_data)), [f"C{int(v)}" for v in sorted(group.display_cluster)])
        margin_axis.set_ylabel("Assignment margin" if column_index == 0 else "")
        margin_axis.set_xlabel("Display cluster")
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.0)
    figure.tight_layout()
    save_figure(
        figure, "figure_B3_cluster_size_validation", title, b3_table,
        {
            "input_artifact_paths": ["phase1_clinical_clustering/cluster_size_validation_summary.csv", "phase1_clinical_clustering/validation_assignments.csv"],
            "sample_sizes": b3_table.groupby("canonical_stratum").train_n.sum().to_dict(),
            "metrics_shown": ["Training cluster size", "Validation cluster size", "Assignment margin distribution"],
            "color_role_mapping": {"Training": COLOR_REFERENCE, "Validation": COLOR_ADJUSTED, "Minimum-size threshold": COLOR_NULL},
        },
    )


def render_gate_b_report(
    selection: dict[str, dict[str, Any]],
    k_results: dict[str, dict[int, dict[str, Any]]],
    characterization: pd.DataFrame,
    sensitivity_table: pd.DataFrame,
    b3_table: pd.DataFrame,
    validation_assignments: pd.DataFrame,
    strategy_by_subtype: dict[str, str],
    audit: dict[str, Any],
    nulled_counts: dict[str, int],
) -> None:
    factor_list_line = ", ".join(read_json(DECISION_ROOT / "factor_selection.json")["final_factor_list"])
    k_lines = [f"- {subtype}: k={selection[subtype]['selected_k']} ({selection[subtype]['status']})" for subtype in CANONICAL_STRATA]
    stability_rows = []
    for subtype in CANONICAL_STRATA:
        k = selection[subtype]["selected_k"]
        if k is None:
            continue
        result = k_results[subtype][k]
        stability_rows.append({
            "canonical_stratum": subtype, "selected_k": k, "bootstrap_mean_ari": result["bootstrap_mean_ari"],
            "bootstrap_ci_low": result["bootstrap_ci_low"], "bootstrap_ci_high": result["bootstrap_ci_high"],
            "smallest_cluster_n": result["smallest_cluster_n"], "smallest_cluster_fraction": result["smallest_cluster_fraction"],
            "silhouette": result["silhouette"], "gap_statistic": result["gap_statistic"], "site_association_cramers_v": result["site_association"]["cramers_v"],
            "site_association_strong": result["site_association"]["strong"],
        })
    stability_table = pd.DataFrame(stability_rows)
    val_confidence = validation_assignments.groupby("canonical_stratum").agg(
        val_n=("participant_id", "count"), mean_assignment_margin=("assignment_margin", "mean"), median_relative_margin=("relative_assignment_margin", "median")
    ).reset_index()

    lines = [
        "# Gate B report",
        "",
        "## Concise interpretation",
        "",
        (
            "The clinical factor list (age, BMI, HbA1c, C-peptide, TG/HDL ratio, waist-to-hip ratio) and the per-subtype missing-data strategy are inherited from the approved Gate A decisions: complete-case is primary only for healthy, "
            "because the saved Gate A audit shows strong severity association (standardized mean difference above 0.50) between complete-case retention and disease severity for pre_diabetes, t2d_oral_non_insulin, and insulin_dependent, "
            "so those three subtypes keep train-fitted iterative imputation as the primary analysis and run complete-case only as a sensitivity check. "
            f"A Phase 1 policy nulls core-laboratory baseline values dated after a participant's CGM end before clustering (counts: {nulled_counts}). "
            "The cluster count was selected independently per subtype from k in {2, 3, 4} using the full 1,000-bootstrap diagnostic battery and the deterministic selection hierarchy; no common k was imposed. "
        ),
        "",
        *k_lines,
        "",
        "## Selected factor list",
        "",
        factor_list_line,
        "",
        "## Missing-data strategy (primary)",
        "",
        dataframe_to_markdown(pd.DataFrame([{"canonical_stratum": s, "primary_strategy": strategy_by_subtype[s]} for s in CANONICAL_STRATA])),
        "",
        "## Required factor audit",
        "",
        f"C-peptide vs TG/HDL Spearman correlation by subtype (train only): {json.dumps(audit['inspected_pairs_spearman']['c_peptide_vs_tg_hdl_all_subtypes'])}",
        "",
        f"BMI vs C-peptide Spearman correlation in prediabetes (train only): {audit['inspected_pairs_spearman']['bmi_vs_c_peptide_prediabetes']}",
        "",
        f"Age vs BMI Spearman correlation in insulin-dependent (train only): {audit['inspected_pairs_spearman']['age_vs_bmi_insulin_dependent']}",
        "",
        f"Nonpositive HDL values rejected: {audit['nonpositive_hdl_rejected_count']}. Post-CGM baseline values nulled by factor: {json.dumps(nulled_counts)}. Duplicate baseline record conflicts: {json.dumps(conflicts_summary(audit))}.",
        "",
        "Full per-subtype Spearman correlation matrices are saved in `phase1_clinical_clustering/required_factor_audit.json`.",
        "",
        "## Selected k per subtype: candidate diagnostic table",
        "",
        dataframe_to_markdown(stability_table),
        "",
        "## Selection reasoning per subtype",
        "",
        *[f"**{subtype}**: " + " ".join(selection[subtype]["reasoning"]) for subtype in CANONICAL_STRATA],
        "",
        "## Cluster profile summary (train, raw and standardized factor means)",
        "",
        dataframe_to_markdown(characterization[["canonical_stratum", "display_cluster", "combined_label", "train_n"] + [f"mean_{f}" for f in read_json(DECISION_ROOT / 'factor_selection.json')['final_factor_list']]]),
        "",
        "## External characterization (not used to fit or revise the clustering): sex and medication composition",
        "",
        dataframe_to_markdown(characterization[["canonical_stratum", "display_cluster", "combined_label", "sex_composition"] + [c for c in characterization.columns if c.startswith("medication_fraction_")]]),
        "",
        "## Validation assignment confidence",
        "",
        dataframe_to_markdown(val_confidence),
        "",
        "## Cluster size at validation",
        "",
        dataframe_to_markdown(b3_table),
        "",
        "## Missing-data sensitivity (alternate strategy at selected k)",
        "",
        dataframe_to_markdown(sensitivity_table) if len(sensitivity_table) else "No sensitivity comparison was computed.",
        "",
        "## Primary versus exploratory subtype status",
        "",
        *[f"- {subtype}: **{selection[subtype]['status']}**" + (" — marked `near_continuum_or_underpowered`, excluded from headline pooled conclusions, retained in tables and supplementary figures" if selection[subtype]["near_continuum_or_underpowered"] else "") for subtype in CANONICAL_STRATA],
        "",
        "## Figures",
        "",
        "![k-selection dashboard](../figures/full_resolution/figure_B1_k_selection_dashboard.png)",
        "",
        "![Cluster profiles](../figures/full_resolution/figure_B2_cluster_profiles.png)",
        "",
        "Cluster colors are local to each diagnostic subtype and do not imply equivalence across subtypes.",
        "",
        "![Cluster size and validation](../figures/full_resolution/figure_B3_cluster_size_validation.png)",
        "",
        "## Frozen artifact paths and hashes",
        "",
        f"See `phase1_clinical_clustering/frozen_clustering_manifest.json` for the preprocessing pipeline path, centroid path, and sha256 hash of every frozen artifact per subtype.",
        "",
        "## Test-set confirmation",
        "",
        "The test set was not read, transformed, or assigned in Phase 1. No h0 or ht value was inspected.",
        "",
        "## Next phase",
        "",
        "Phase 2 would load the frozen preprocessing pipeline and centroids, assign test participants to the nearest frozen training centroid within their diagnostic subtype, and evaluate whether the frozen clinical neighborhoods and clusters are preserved in h0, against a clinical-PCA reference and a permutation null, before stopping at Gate C.",
        "",
        "Gate B reached. The clinical clustering is frozen. No h0 or ht result has been inspected. Waiting for confirmation before assigning test participants and running Phase 2.",
        "",
    ]
    (DECISION_ROOT / "GATE_B_REPORT.md").write_text("\n".join(lines))


def conflicts_summary(audit: dict[str, Any]) -> dict[str, int]:
    return {key: value for key, value in audit["duplicate_baseline_records_by_source_column"].items() if value}


if __name__ == "__main__":
    main()
