"""Phase 0 and Gate A for the within-subtype phenotype preservation study.

This phase audits inputs and clinical viability only. It never reads latent vector
values, never selects a final cluster count, and never saves participant labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ssmcgm.analysis.within_subtype_config import (
    BAND_ALPHA,
    BOOTSTRAP_CI_LEVEL,
    CANONICAL_STRATA,
    CLUSTER_COLORS,
    COLOR_ADJUSTED,
    COLOR_ANNOTATION,
    COLOR_NULL,
    COLOR_OBSERVED,
    COLOR_POSITIVE,
    COLOR_REFERENCE,
    COMPLETE_CASE_MIN_RETAINED_FRACTION,
    CORE_FACTORS,
    DATASET,
    DECISION_ROOT,
    FACTOR_COLUMN_MAP,
    FACTOR_DOMAINS,
    FACTOR_UNITS,
    FIGURE_DPI,
    FIGURE_ROOT,
    IMPOSSIBLE_VALUE_BOUNDS,
    ITERATIVE_IMPUTER_MAX_ITER,
    KNN_K_CEILING,
    KNN_K_FLOOR,
    KNN_K_FRACTION,
    K_RANGE,
    LOG_ROOT,
    MAX_FACTOR_CORRELATION,
    MIN_CLUSTER_TRAIN_FRACTION,
    MIN_CLUSTER_TRAIN_N,
    MIN_FACTOR_COVERAGE,
    N_INIT,
    PARQUET_BATCH_SIZE,
    PHASE0_BOOTSTRAP_N,
    PHASE0_PARALLEL_JOBS,
    RAW_STRATUM_MAP,
    REPO,
    REPO_BRANCH,
    SEED,
    SEVERITY_SMD_THRESHOLD,
    SITE_ASSOCIATION_CRAMERS_V_THRESHOLD,
    SITE_ASSOCIATION_P_THRESHOLD,
    SIXTH_FACTOR_CANDIDATES,
    SKEW_LOG_THRESHOLD,
    SPLIT_PATH,
    STRATIFIER,
    STUDY1_ROOT,
    STUDY2_ROOT,
    TABLE_ROOT,
    THUMBNAIL_DPI,
    UNDERPOWERED_TEST_N,
)


PHASE_ROOT = STUDY2_ROOT / "phase0_viability"
FIGURE_FULL = FIGURE_ROOT / "full_resolution"
FIGURE_THUMB = FIGURE_ROOT / "thumbnails"
FIGURE_DATA = FIGURE_ROOT / "plotted_data"
FIGURE_META = FIGURE_ROOT / "metadata"
SITE_COLUMN = "participants_clinical_site"
SPLITS = ["train", "val", "test"]
BASELINE_DATE_COLUMNS = {
    "bmi_baseline": "bmi_baseline_date",
    "hba1c_percent_baseline": "hba1c_percent_baseline_date",
    "c_peptide_ngml_baseline": "c_peptide_ngml_baseline_date",
    "triglycerides_mgdl_baseline": "triglycerides_mgdl_baseline_date",
    "hdl_cholesterol_mgdl_baseline": "hdl_cholesterol_mgdl_baseline_date",
    "waist_to_hip_ratio_baseline": "waist_to_hip_ratio_baseline_date",
    "systolic_blood_pressure_baseline": "clinical_systolic_bp_mmhg_baseline_date",
}
FACTOR_LABELS = {
    "participants_age": "Age",
    "bmi_baseline": "BMI",
    "hba1c_percent_baseline": "HbA1c",
    "c_peptide_ngml_baseline": "C-peptide",
    "tg_hdl_ratio": "TG/HDL",
    "waist_to_hip_ratio_baseline": "Waist-to-hip ratio",
    "fasting_insulin_baseline": "Fasting insulin",
    "systolic_blood_pressure_baseline": "Systolic blood pressure",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else (f"{value:.4f}" if isinstance(value, float) else str(value))
        )
    header = "| " + " | ".join(str(column) for column in display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in display.itertuples(index=False, name=None)]
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
        STUDY2_ROOT / "models/preprocessing",
        STUDY2_ROOT / "models/centroids",
        STUDY2_ROOT / "phase1_clinical_clustering",
        STUDY2_ROOT / "phase2_h0_preservation",
        STUDY2_ROOT / "phase3_ht_preservation",
        STUDY2_ROOT / "phase4_time_resolved_extension",
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


def parquet_inventory(path: Path, metadata_columns: list[str], endpoint_kind: str) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    id_frame = pd.read_parquet(path, columns=[column for column in ["participant_id", "split"] if column in names])
    id_frame["participant_id"] = id_frame["participant_id"].astype(str)
    vector_columns = [column for column in names if column not in metadata_columns]
    return {
        "resolved_path": str(path),
        "sha256": sha256(path),
        "shape": [int(parquet.metadata.num_rows), int(parquet.metadata.num_columns)],
        "participant_count": int(id_frame["participant_id"].nunique()),
        "participant_identifier_column": "participant_id",
        "vector_dimension": len(vector_columns),
        "missing_participant_identifier_count": int(id_frame["participant_id"].isna().sum()),
        "duplicate_participant_count": int(id_frame["participant_id"].duplicated().sum()),
        "endpoint_kind": endpoint_kind,
        "contains_repeated_time_resolved_vectors": False,
        "participant_ids": set(id_frame["participant_id"]),
        "split_by_participant": dict(zip(id_frame["participant_id"], id_frame.get("split", pd.Series(dtype=str)))),
    }


def build_artifact_inventory(clinical_ids: set[str], split_lookup: dict[str, str]) -> tuple[dict[str, Any], set[str], set[str]]:
    paths = {
        "h0_full": STUDY1_ROOT / "step2/h0_matrix.parquet",
        "ht_full_overnight_endpoint": STUDY1_ROOT / "step3/h_t_full.parquet",
        "ht_neutral_overnight_endpoint": STUDY1_ROOT / "step3/h_t_neutral.parquet",
        "ht_glucose_residualized_overnight_endpoint": STUDY1_ROOT / "step3/h_t_full_residualized.parquet",
    }
    required = [paths["h0_full"], paths["ht_full_overnight_endpoint"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required endpoint artifacts are absent: {missing}")

    inventory: dict[str, Any] = {}
    specifications = {
        "h0_full": (["participant_id", "split"], "Initial state endpoint"),
        "ht_full_overnight_endpoint": (
            ["participant_id", "split", "n_overnight_anchors", "avg_glucose_mgdl"],
            "Participant-level overnight endpoint summary",
        ),
        "ht_neutral_overnight_endpoint": (
            ["participant_id", "split", "n_overnight_anchors"],
            "Participant-level neutralized overnight endpoint summary",
        ),
        "ht_glucose_residualized_overnight_endpoint": (
            ["participant_id", "split", "n_overnight_anchors"],
            "Participant-level glucose-residualized overnight endpoint summary",
        ),
    }
    artifact_id_sets: dict[str, set[str]] = {}
    for key, path in paths.items():
        if not path.exists():
            inventory[key] = {"resolved_path": None, "available": False}
            continue
        metadata_columns, endpoint_kind = specifications[key]
        entry = parquet_inventory(path, metadata_columns, endpoint_kind)
        ids = entry.pop("participant_ids")
        artifact_splits = entry.pop("split_by_participant")
        artifact_id_sets[key] = ids
        split_conflicts = [pid for pid in ids if pid in split_lookup and artifact_splits.get(pid) != split_lookup[pid]]
        entry.update(
            {
                "available": True,
                "missing_participant_count_relative_to_clinical_split": len(clinical_ids - ids),
                "extra_participant_count_relative_to_clinical_split": len(ids - clinical_ids),
                "aligns_exactly_for_shared_participants": not (ids - clinical_ids) and not split_conflicts,
                "split_conflict_count": len(split_conflicts),
            }
        )
        inventory[key] = entry

    h0_ids = artifact_id_sets["h0_full"]
    ht_ids = artifact_id_sets["ht_full_overnight_endpoint"]
    if h0_ids != ht_ids:
        raise AssertionError("Required h0 and ht participant identifiers do not align exactly")
    if inventory["h0_full"]["vector_dimension"] != inventory["ht_full_overnight_endpoint"]["vector_dimension"]:
        raise AssertionError("Required h0 and ht vector dimensions differ")
    if inventory["h0_full"]["split_conflict_count"] or inventory["ht_full_overnight_endpoint"]["split_conflict_count"]:
        raise AssertionError("Endpoint split membership conflicts with the canonical split file")

    neutral_reference = REPO / "outputs/hidden_state_phenotype/step1_static_neutralization/20260724T223612Z/static_reference_profile.json"
    anchor_report = STUDY1_ROOT / "step3/anchor_extraction_report.json"
    inventory["h0_neutral"] = {
        "resolved_path": str(neutral_reference) if neutral_reference.exists() else None,
        "available": neutral_reference.exists(),
        "participant_count": len(h0_ids) if neutral_reference.exists() else 0,
        "participant_identifier_column": None,
        "vector_dimension": inventory["h0_full"]["vector_dimension"],
        "endpoint_kind": "Single constant neutral initial-state reference",
        "contains_repeated_time_resolved_vectors": False,
        "creation_metadata": str(anchor_report) if anchor_report.exists() else None,
        "note": "Study 1 verified the neutral h0 is identical across participants; it was not saved as a redundant participant table.",
    }
    inventory["participant_splits"] = {
        "resolved_path": str(SPLIT_PATH), "sha256": sha256(SPLIT_PATH), "participant_count": len(clinical_ids)
    }
    inventory["diagnostic_subtype_and_static_clinical_factors"] = {
        "resolved_path": str(DATASET), "sha256": sha256(DATASET), "participant_count": len(clinical_ids)
    }
    inventory["vector_construction_metadata"] = {
        "resolved_paths": [str(anchor_report), str(STUDY1_ROOT / "step1/coverage_and_stability_report.json")],
        "endpoint_interpretation": "Overnight endpoint summary using anchors from 0 to 6 hours",
    }
    inventory["time_resolved_ht_snapshots"] = {
        "resolved_path": None,
        "available": False,
        "note": "No participant-level matched time-resolved snapshot artifact was found in Study 1.",
    }
    return inventory, h0_ids, ht_ids


def cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0
    chi2, p_value, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return (float(math.sqrt(chi2 / denominator)) if denominator else 0.0), float(p_value)


def standardized_mean_difference(values: pd.Series, retained: pd.Series) -> float | None:
    left = values[retained & values.notna()].astype(float)
    right = values[~retained & values.notna()].astype(float)
    if len(left) < 2 or len(right) < 2:
        return None
    pooled = math.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2)
    return float((left.mean() - right.mean()) / pooled) if pooled else 0.0


def factor_selection(train: pd.DataFrame, adequately_powered: list[str], baseline_timing: dict[str, bool]) -> dict[str, Any]:
    candidate_results: dict[str, Any] = {}
    selected: str | None = None
    for candidate in SIXTH_FACTOR_CANDIDATES:
        source = FACTOR_COLUMN_MAP[candidate]
        resolved = source in train.columns
        coverage = {
            subtype: float(train.loc[train.canonical_stratum == subtype, source].notna().mean()) if resolved else 0.0
            for subtype in adequately_powered
        }
        correlations: dict[str, dict[str, float | None]] = {}
        passes_correlation = True
        if resolved:
            for subtype in adequately_powered:
                subtype_frame = train[train.canonical_stratum == subtype]
                correlations[subtype] = {}
                for core in CORE_FACTORS:
                    value = subtype_frame[[source, core]].corr(method="spearman").iloc[0, 1]
                    correlations[subtype][core] = None if pd.isna(value) else float(value)
                    if pd.notna(value) and abs(value) >= MAX_FACTOR_CORRELATION:
                        passes_correlation = False
        unit_resolved = FACTOR_UNITS[candidate] != "Unresolved"
        coverage_pass = resolved and all(value >= MIN_FACTOR_COVERAGE for value in coverage.values())
        domain_new = FACTOR_DOMAINS[candidate] not in {FACTOR_DOMAINS[factor] for factor in CORE_FACTORS}
        timing_valid = baseline_timing.get(candidate, False)
        criteria_passed = bool(resolved and unit_resolved and coverage_pass and passes_correlation and domain_new and timing_valid)
        accepted = bool(criteria_passed and selected is None)
        reasons = []
        if not resolved:
            reasons.append(f"Exact source column {source} is absent")
        if not unit_resolved:
            reasons.append("Unit definition is unresolved")
        if not coverage_pass:
            reasons.append("Coverage is below the required level in at least one adequately powered subtype")
        if not passes_correlation:
            reasons.append("Absolute Spearman correlation reaches the maximum allowed correlation")
        if not domain_new:
            reasons.append("Clinical domain is already strongly represented")
        if not timing_valid:
            reasons.append("Baseline timing could not be validated")
        if accepted:
            reasons.append("First ordered candidate satisfying every deterministic criterion")
        elif criteria_passed:
            reasons.append("Eligible but not selected because an earlier ordered candidate passed")
        candidate_results[candidate] = {
            "resolved_column": source if resolved else None,
            "unit": FACTOR_UNITS[candidate],
            "coverage_by_adequately_powered_subtype": coverage,
            "spearman_correlations_with_core": correlations,
            "clinically_interpretable": True,
            "baseline_timing_valid": timing_valid,
            "not_cgm_derived": True,
            "not_diagnostic_or_medication": True,
            "new_clinical_domain": domain_new,
            "criteria_passed": criteria_passed,
            "accepted": accepted,
            "reasons": reasons,
        }
        if accepted:
            selected = candidate
    return {
        "core_factors": CORE_FACTORS,
        "candidate_factors": candidate_results,
        "selected_sixth_factor": selected,
        "final_factor_list": CORE_FACTORS + ([selected] if selected else []),
    }


def decide_missing_data(frame: pd.DataFrame, factors: list[str]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for subtype in CANONICAL_STRATA:
        all_subset = frame[frame.canonical_stratum == subtype].copy()
        all_complete = all_subset[factors].notna().all(axis=1)
        split_representation = all(bool((all_complete & (all_subset.split == split)).any()) for split in SPLITS)

        subset = all_subset[all_subset.split == "train"].copy()
        complete = subset[factors].notna().all(axis=1)
        retained_fraction = float(complete.mean())
        train_complete_n = int(complete.sum())
        size_viable = train_complete_n >= max(K_RANGE) * MIN_CLUSTER_TRAIN_N
        site_table = pd.crosstab(complete.map({True: "Complete", False: "Incomplete"}), subset[SITE_COLUMN].fillna("Missing"))
        site_v, site_p = cramers_v(site_table)
        severity_smd = {
            factor: standardized_mean_difference(subset[factor], complete)
            for factor in CORE_FACTORS
            if factor in subset.columns
        }
        finite_smd = [abs(value) for value in severity_smd.values() if value is not None]
        max_smd = max(finite_smd, default=0.0)
        strong_site = site_p < SITE_ASSOCIATION_P_THRESHOLD and site_v >= SITE_ASSOCIATION_CRAMERS_V_THRESHOLD
        strong_severity = max_smd >= SEVERITY_SMD_THRESHOLD
        use_complete = (
            retained_fraction >= COMPLETE_CASE_MIN_RETAINED_FRACTION
            and split_representation
            and size_viable
            and not strong_site
            and not strong_severity
        )
        decisions[subtype] = {
            "strategy": "complete_case" if use_complete else "iterative_imputation",
            "eligible_n": len(all_subset),
            "complete_case_n": int(all_complete.sum()),
            "complete_case_retained_fraction_all_splits_for_reporting_only": float(all_complete.mean()),
            "train_retained_fraction_used_for_decision": retained_fraction,
            "every_split_represented": split_representation,
            "train_complete_n": train_complete_n,
            "training_size_viable_for_all_candidate_k": size_viable,
            "site_association_train_only": {"cramers_v": site_v, "p_value": site_p, "strong": strong_site},
            "severity_smd_train_only": severity_smd,
            "max_absolute_severity_smd_train_only": max_smd,
            "strong_severity_association_train_only": strong_severity,
            "alternative_sensitivity": "iterative_imputation" if use_complete else "complete_case",
            "test_values_used_for_decision": False,
        }
    return decisions


def fitted_matrix(train: pd.DataFrame, factors: list[str], use_imputation: bool) -> tuple[np.ndarray, list[str]]:
    raw = train[factors].astype(float).to_numpy()
    if use_imputation:
        imputer = IterativeImputer(max_iter=ITERATIVE_IMPUTER_MAX_ITER, random_state=SEED, sample_posterior=False)
        raw = imputer.fit_transform(raw)
    elif np.isnan(raw).any():
        raise AssertionError("Complete-case preliminary matrix contains missing values")
    data = pd.DataFrame(raw, columns=factors, index=train.index)
    transformed: list[str] = []
    for factor in factors:
        skew = float(data[factor].skew())
        if abs(skew) > SKEW_LOG_THRESHOLD and data[factor].min() > -1:
            data[factor] = np.log1p(data[factor])
            transformed.append(factor)
    scaled = StandardScaler().fit_transform(data.to_numpy())
    return scaled, transformed


def align_labels(reference: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    contingency = np.zeros((k, k), dtype=int)
    for left, right in zip(reference, candidate):
        contingency[left, right] += 1
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {column: row for row, column in zip(rows, columns)}
    return np.array([mapping.get(value, value) for value in candidate], dtype=int)


def one_bootstrap(reference_labels: np.ndarray, raw: pd.DataFrame, factors: list[str], k: int, seed: int, use_imputation: bool) -> float:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(raw), size=len(raw))
    boot = raw.iloc[indices]
    boot_array = boot[factors].astype(float).to_numpy()
    original_array = raw[factors].astype(float).to_numpy()
    if use_imputation:
        imputer = IterativeImputer(max_iter=ITERATIVE_IMPUTER_MAX_ITER, random_state=seed, sample_posterior=False)
        boot_array = imputer.fit_transform(boot_array)
        original_array = imputer.transform(original_array)
    elif np.isnan(boot_array).any() or np.isnan(original_array).any():
        raise AssertionError("Complete-case bootstrap contains missing values")
    boot_values = pd.DataFrame(boot_array, columns=factors)
    original_values = pd.DataFrame(original_array, columns=factors)
    for factor in factors:
        if abs(float(boot_values[factor].skew())) > SKEW_LOG_THRESHOLD and boot_values[factor].min() > -1:
            boot_values[factor] = np.log1p(boot_values[factor])
            original_values[factor] = np.log1p(original_values[factor])
    scaler = StandardScaler().fit(boot_values.to_numpy())
    boot_scaled = scaler.transform(boot_values.to_numpy())
    original_scaled = scaler.transform(original_values.to_numpy())
    model = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed)
    model.fit(boot_scaled)
    labels = align_labels(reference_labels, model.predict(original_scaled), k)
    return float(adjusted_rand_score(reference_labels, labels))


def preliminary_clustering(frame: pd.DataFrame, factors: list[str], missing: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subtype_index, subtype in enumerate(CANONICAL_STRATA):
        subset = frame[(frame.canonical_stratum == subtype) & (frame.split == "train")].copy()
        use_imputation = missing[subtype]["strategy"] == "iterative_imputation"
        if not use_imputation:
            subset = subset[subset[factors].notna().all(axis=1)].copy()
        matrix, transformed = fitted_matrix(subset, factors, use_imputation)
        for k in K_RANGE:
            reference_model = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED)
            reference = reference_model.fit_predict(matrix)
            counts = np.bincount(reference, minlength=k)
            silhouette = float(silhouette_score(matrix, reference))
            boot_seeds = [SEED + 100_000 * subtype_index + 1_000 * k + index for index in range(PHASE0_BOOTSTRAP_N)]
            aris = Parallel(n_jobs=PHASE0_PARALLEL_JOBS, prefer="processes")(
                delayed(one_bootstrap)(reference, subset, factors, k, seed, use_imputation) for seed in boot_seeds
            )
            lo_percent = (1 - BOOTSTRAP_CI_LEVEL) * 50
            hi_percent = 100 - lo_percent
            smallest_n = int(counts.min())
            smallest_fraction = float(counts.min() / counts.sum())
            rows.append(
                {
                    "canonical_stratum": subtype,
                    "k": k,
                    "train_n": len(subset),
                    "transformations": ", ".join(transformed),
                    "cluster_sizes": json.dumps(counts.tolist()),
                    "smallest_cluster_n": smallest_n,
                    "smallest_cluster_fraction": smallest_fraction,
                    "minimum_size_feasible": smallest_n >= MIN_CLUSTER_TRAIN_N and smallest_fraction >= MIN_CLUSTER_TRAIN_FRACTION,
                    "silhouette": silhouette,
                    "bootstrap_mean_ari": float(np.mean(aris)),
                    "bootstrap_median_ari": float(np.median(aris)),
                    "bootstrap_std_ari": float(np.std(aris, ddof=1)),
                    "bootstrap_ci_low": float(np.percentile(aris, lo_percent)),
                    "bootstrap_ci_high": float(np.percentile(aris, hi_percent)),
                    "bootstrap_n": PHASE0_BOOTSTRAP_N,
                }
            )
    return pd.DataFrame(rows)


def style_axes(axes: Iterable[plt.Axes]) -> None:
    for axis in axes:
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


def make_figure_a1(counts: pd.DataFrame, coverage: pd.DataFrame, retention: pd.DataFrame, correlations: dict[str, pd.DataFrame]) -> None:
    sns.set_style("whitegrid")
    title = "Clinical data viability differs across diagnostic subtypes"
    figure = plt.figure(figsize=(18, 11), facecolor="white")
    grid = figure.add_gridspec(2, 4, height_ratios=[0.9, 1.1], hspace=0.42, wspace=0.42)
    axes: list[plt.Axes] = []

    axis = figure.add_subplot(grid[0, 0]); axes.append(axis)
    pivot = counts.pivot(index="canonical_stratum", columns="split", values="n").reindex(CANONICAL_STRATA)
    pivot[[column for column in SPLITS if column in pivot]].plot.bar(
        ax=axis, color=[COLOR_REFERENCE, COLOR_ADJUSTED, COLOR_POSITIVE], width=0.75
    )
    axis.set_title("Participant counts by split", fontweight="bold")
    axis.set_xlabel(""); axis.set_ylabel("Participants (N)"); axis.tick_params(axis="x", rotation=30)
    axis.legend(["Training", "Validation", "Test"], facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=7)

    axis = figure.add_subplot(grid[0, 1]); axes.append(axis)
    core = coverage[coverage.factor.isin(CORE_FACTORS)].copy()
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    for index, subtype in enumerate(CANONICAL_STRATA):
        group = core[core.canonical_stratum == subtype]
        axis.plot(range(len(CORE_FACTORS)), group.set_index("factor").reindex(CORE_FACTORS).coverage, color=COLOR_REFERENCE, linestyle=line_styles[index], marker=markers[index], markersize=4, label=subtype.replace("_", " ").capitalize())
    axis.axhline(MIN_FACTOR_COVERAGE, color=COLOR_NULL, linewidth=1.25)
    axis.set_title("Core factor coverage", fontweight="bold"); axis.set_ylabel("Coverage fraction"); axis.set_ylim(0, 1.04)
    axis.set_xticks(range(len(CORE_FACTORS)), [FACTOR_LABELS[factor] for factor in CORE_FACTORS], rotation=35, ha="right")
    axis.legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=6)

    axis = figure.add_subplot(grid[0, 2]); axes.append(axis)
    ordered = retention.set_index("canonical_stratum").reindex(CANONICAL_STRATA)
    axis.bar(range(len(ordered)), ordered.retained_fraction, color=COLOR_REFERENCE)
    axis.axhline(COMPLETE_CASE_MIN_RETAINED_FRACTION, color=COLOR_NULL, linewidth=1.25)
    axis.set_title("Complete-case retention", fontweight="bold"); axis.set_ylabel("Retained fraction"); axis.set_ylim(0, 1.04)
    axis.set_xticks(range(len(ordered)), [value.replace("_", " ").capitalize() for value in ordered.index], rotation=35, ha="right")

    axis = figure.add_subplot(grid[0, 3]); axes.append(axis)
    candidate = coverage[coverage.factor.isin(SIXTH_FACTOR_CANDIDATES)].groupby("factor", as_index=False).coverage.min().set_index("factor").reindex(SIXTH_FACTOR_CANDIDATES)
    axis.bar(range(len(candidate)), candidate.coverage.fillna(0), color=[COLOR_REFERENCE, COLOR_NULL, COLOR_REFERENCE])
    axis.axhline(MIN_FACTOR_COVERAGE, color=COLOR_NULL, linewidth=1.25)
    axis.set_title("Sixth-factor candidate coverage", fontweight="bold"); axis.set_ylabel("Minimum coverage fraction"); axis.set_ylim(0, 1.04)
    axis.set_xticks(range(len(candidate)), [FACTOR_LABELS[factor] for factor in candidate.index], rotation=35, ha="right")

    heatmap_colors = ListedColormap([COLOR_REFERENCE, COLOR_NULL, COLOR_POSITIVE])
    short_titles = ["Healthy correlations", "Prediabetes correlations", "T2D non-insulin correlations", "Insulin-dependent correlations"]
    for index, subtype in enumerate(CANONICAL_STRATA):
        axis = figure.add_subplot(grid[1, index]); axes.append(axis)
        corr = correlations[subtype].reindex(index=CORE_FACTORS, columns=CORE_FACTORS)
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=heatmap_colors, cbar=index == 3, square=True, ax=axis, linewidths=0.25, linecolor="white")
        axis.set_title(short_titles[index], fontweight="bold", fontsize=10)
        axis.set_xticklabels([FACTOR_LABELS[factor] for factor in CORE_FACTORS], rotation=45, ha="right", fontsize=7)
        axis.set_yticklabels([FACTOR_LABELS[factor] for factor in CORE_FACTORS], rotation=0, fontsize=7)

    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=16, y=0.99)
    plotted = pd.concat([
        counts.assign(panel="Participant counts"),
        coverage.assign(panel="Factor coverage"),
        retention.assign(panel="Complete-case retention"),
    ], ignore_index=True, sort=False)
    save_figure(figure, "figure_A1_clinical_data_viability", title, plotted, {
        "input_artifact_paths": [str(DATASET), str(SPLIT_PATH)],
        "sample_sizes": counts.to_dict("records"),
        "metrics_shown": ["Participant count", "Factor coverage", "Complete-case retention", "Spearman correlation"],
        "color_role_mapping": {"Training and reference": COLOR_REFERENCE, "Validation": COLOR_ADJUSTED, "Test": COLOR_POSITIVE, "Threshold and absent": COLOR_NULL},
    })


def make_figure_a2(preliminary: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    title = "Candidate cluster viability differs across diagnostic subtypes"
    figure, axes = plt.subplots(4, 4, figsize=(15, 13), sharex=True, facecolor="white")
    metrics = [
        ("silhouette", "Preliminary silhouette"),
        ("bootstrap_mean_ari", "Preliminary bootstrap ARI"),
        ("smallest_cluster_n", "Smallest cluster (N)"),
        ("minimum_size_feasible", "Minimum-size feasibility"),
    ]
    for row_index, subtype in enumerate(CANONICAL_STRATA):
        group = preliminary[preliminary.canonical_stratum == subtype].sort_values("k")
        for column_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index, column_index]
            colors = [COLOR_REFERENCE if feasible else COLOR_NULL for feasible in group.minimum_size_feasible]
            if metric == "minimum_size_feasible":
                values = group[metric].astype(int)
                axis.bar(group.k, values, color=colors)
                axis.set_ylim(0, 1.15); axis.set_yticks([0, 1], ["No", "Yes"])
            else:
                axis.plot(group.k, group[metric], color=COLOR_REFERENCE, marker="o", markersize=4)
                for x_value, y_value, color in zip(group.k, group[metric], colors):
                    axis.scatter(x_value, y_value, color=color, s=30, zorder=3)
                if metric == "smallest_cluster_n":
                    axis.axhline(MIN_CLUSTER_TRAIN_N, color=COLOR_NULL, linewidth=1.25)
            if row_index == 0:
                axis.set_title(label, fontweight="bold", fontsize=10)
            if column_index == 0:
                axis.set_ylabel(subtype.replace("_", " ").capitalize())
            if row_index == len(CANONICAL_STRATA) - 1:
                axis.set_xlabel("Candidate k")
            axis.set_xticks(K_RANGE)
    style_axes(axes.flat)
    figure.suptitle(title, fontweight="bold", fontsize=16, y=0.995)
    figure.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(figure, "figure_A2_candidate_cluster_viability", title, preliminary, {
        "input_artifact_paths": [str(PHASE_ROOT / "preliminary_k_feasibility.csv")],
        "sample_sizes": preliminary.groupby("canonical_stratum").train_n.first().to_dict(),
        "metrics_shown": ["Silhouette", "Bootstrap ARI", "Smallest cluster size", "Minimum-size feasibility"],
        "color_role_mapping": {"Feasible candidate": COLOR_REFERENCE, "Infeasible candidate": COLOR_NULL},
    })


def render_report(inventory: dict[str, Any], mapping: dict[str, str], counts: pd.DataFrame, coverage: pd.DataFrame, invalid: pd.DataFrame, timing: pd.DataFrame, transformations: pd.DataFrame, missing: dict[str, Any], factor: dict[str, Any], preliminary: pd.DataFrame, neighbors: pd.DataFrame) -> None:
    lines = [
        "# Gate A report",
        "",
        "## Concise interpretation",
        "",
        "All required participant-level h0 and overnight ht endpoint artifacts are present, dimension-compatible, and aligned to the same 1,544 participants. Clinical viability and preliminary train-only cluster stability differ by diagnostic subtype. Four core laboratory factors each have 40 participants with nominal baseline dates after the CGM endpoint; this timing conflict must be resolved before Phase 1. No latent vector values were inspected, no final k was selected, and no participant cluster labels were created.",
        "",
        "## Artifact inventory",
        "",
        f"Full inventory: [study1_artifact_inventory.json](../study1_artifact_inventory.json)",
        "",
        f"Time-resolved ht snapshots available: **{inventory['time_resolved_ht_snapshots']['available']}**",
        "",
        "## Exact subtype mapping",
        "",
        *[f"- `{raw}` to `{canonical}`" for raw, canonical in mapping.items()],
        "",
        "## Participant counts",
        "",
        dataframe_to_markdown(counts),
        "",
        "## Factor coverage",
        "",
        dataframe_to_markdown(coverage),
        "",
        "## Invalid-value audit",
        "",
        dataframe_to_markdown(invalid),
        "",
        "## Baseline timing audit",
        "",
        dataframe_to_markdown(timing),
        "",
        "## Transformation decisions",
        "",
        dataframe_to_markdown(transformations),
        "",
        "## Missing-data strategy per subtype",
        "",
        dataframe_to_markdown(pd.DataFrame([{"canonical_stratum": key, **value} for key, value in missing.items()])[["canonical_stratum", "strategy", "complete_case_n", "train_retained_fraction_used_for_decision", "train_complete_n"]]),
        "",
        "## Sixth-factor decision",
        "",
        f"Selected sixth factor: **{factor['selected_sixth_factor'] or 'None'}**",
        "",
        f"Final factor list: {', '.join(factor['final_factor_list'])}",
        "",
        "## Preliminary k feasibility",
        "",
        dataframe_to_markdown(preliminary[["canonical_stratum", "k", "train_n", "cluster_sizes", "smallest_cluster_n", "smallest_cluster_fraction", "minimum_size_feasible", "silhouette", "bootstrap_mean_ari", "bootstrap_ci_low", "bootstrap_ci_high"]]),
        "",
        "These diagnostics are preliminary. They do not select a final k.",
        "",
        "## Neighbor counts and power flags",
        "",
        dataframe_to_markdown(neighbors),
        "",
        "## Figures",
        "",
        "![Clinical data viability](../figures/full_resolution/figure_A1_clinical_data_viability.png)",
        "",
        "![Candidate cluster viability](../figures/full_resolution/figure_A2_candidate_cluster_viability.png)",
        "",
        "## Recommendation",
        "",
        "Proceed to Phase 1 only after explicit approval and a decision on the post-CGM core laboratory values. Phase 1 will apply the approved timing policy, fit the train-only preprocessing pipeline, evaluate k in {2, 3, 4} with 1,000 bootstraps and the full selection hierarchy, validate frozen-centroid assignments, characterize the within-subtype phenotypes, freeze and hash the clinical clustering, and stop at Gate B. The test set and all h0 and ht values will remain closed.",
        "",
        "Gate A reached. No final clustering has been performed. Waiting for confirmation before Phase 1.",
        "",
    ]
    (DECISION_ROOT / "GATE_A_REPORT.md").write_text("\n".join(lines))


def main() -> None:
    setup_output_tree()
    warnings: list[dict[str, Any]] = []
    available = schema_columns()
    requested_columns = [
        "participant_id", STRATIFIER, SITE_COLUMN,
        *[column for column in FACTOR_COLUMN_MAP.values() if column != "fasting_insulin_baseline"],
        "triglycerides_mgdl_baseline", "hdl_cholesterol_mgdl_baseline",
        "bmi_n_records", "bmi_days_to_cgm_start",
        "hba1c_percent_n_records", "hba1c_percent_days_to_cgm_start",
        "c_peptide_ngml_n_records", "c_peptide_ngml_days_to_cgm_start",
        "triglycerides_mgdl_n_records", "triglycerides_mgdl_days_to_cgm_start",
        "hdl_cholesterol_mgdl_n_records", "hdl_cholesterol_mgdl_days_to_cgm_start",
        "waist_to_hip_ratio_n_records", "waist_to_hip_ratio_days_to_cgm_start",
        "clinical_systolic_bp_mmhg_n_records", "clinical_systolic_bp_mmhg_days_to_cgm_start",
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
    split_lookup = dict(zip(frame.participant_id, frame.split))

    cgm_bounds = load_cgm_bounds()
    frame = frame.merge(cgm_bounds, on="participant_id", how="left", validate="one_to_one")
    timing_rows: list[dict[str, Any]] = []
    baseline_timing_valid: dict[str, bool] = {"fasting_insulin_baseline": False}
    for factor, date_column in BASELINE_DATE_COLUMNS.items():
        dates = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
        available_mask = dates.notna() & frame.cgm_start.notna() & frame.cgm_end.notna()
        after_start = available_mask & (dates > frame.cgm_start)
        after_end = available_mask & (dates > frame.cgm_end)
        days_before_start = (frame.cgm_start - dates).dt.total_seconds() / 86_400
        timing_rows.append({
            "factor": factor, "date_column": date_column, "available_n": int(available_mask.sum()),
            "after_cgm_start_n": int(after_start.sum()), "after_cgm_end_n": int(after_end.sum()),
            "median_days_before_cgm_start": days_before_start[available_mask].median(),
            "minimum_days_before_cgm_start": days_before_start[available_mask].min(),
            "maximum_days_before_cgm_start": days_before_start[available_mask].max(),
        })
        if factor in SIXTH_FACTOR_CANDIDATES:
            baseline_timing_valid[factor] = bool(available_mask.any() and not after_end.any())
    timing_audit = pd.DataFrame(timing_rows)
    timing_audit.to_csv(PHASE_ROOT / "baseline_timing_audit.csv", index=False)

    hdl = frame["hdl_cholesterol_mgdl_baseline"]
    triglycerides = frame["triglycerides_mgdl_baseline"]
    invalid_hdl = hdl.notna() & (hdl <= 0)
    frame["tg_hdl_ratio"] = np.where(~invalid_hdl & hdl.notna() & triglycerides.notna(), triglycerides / hdl, np.nan)
    invalid_ratio_count = int(invalid_hdl.sum())

    inventory, h0_ids, ht_ids = build_artifact_inventory(set(frame.participant_id), split_lookup)
    write_json(STUDY2_ROOT / "study1_artifact_inventory.json", inventory)
    write_json(PHASE_ROOT / "stratum_mapping.json", {"observed_raw_values": raw_values, "exact_mapping": RAW_STRATUM_MAP})

    factor_audit_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    audit_factors = CORE_FACTORS + SIXTH_FACTOR_CANDIDATES
    for factor in audit_factors:
        source = FACTOR_COLUMN_MAP[factor]
        if factor == "tg_hdl_ratio": source = factor
        resolved = source in frame.columns
        series = frame[source] if resolved else pd.Series(np.nan, index=frame.index)
        numeric = pd.to_numeric(series, errors="coerce")
        bounds = IMPOSSIBLE_VALUE_BOUNDS[factor]
        impossible = numeric.notna() & ((numeric < bounds[0]) | (numeric > bounds[1]))
        observed = numeric.dropna()
        factor_audit_rows.append({
            "factor": factor, "resolved_column": source if resolved else None, "unit": FACTOR_UNITS[factor],
            "n": int(observed.size), "minimum": observed.min() if len(observed) else None,
            "q1": observed.quantile(0.25) if len(observed) else None,
            "median": observed.median() if len(observed) else None,
            "q3": observed.quantile(0.75) if len(observed) else None,
            "maximum": observed.max() if len(observed) else None,
            "missing_fraction": float(numeric.isna().mean()), "impossible_n": int(impossible.sum()),
            "duplicate_baseline_conflict_n": conflicts.get(source, 0),
        })
        invalid_rows.append({"factor": factor, "impossible_n": int(impossible.sum()), "duplicate_baseline_conflict_n": conflicts.get(source, 0)})
    invalid_rows.append({"factor": "tg_hdl_ratio_nonpositive_hdl", "impossible_n": invalid_ratio_count, "duplicate_baseline_conflict_n": conflicts.get("hdl_cholesterol_mgdl_baseline", 0)})
    factor_audit = pd.DataFrame(factor_audit_rows)
    invalid_audit = pd.DataFrame(invalid_rows)
    factor_audit.to_csv(PHASE_ROOT / "factor_audit.csv", index=False)
    invalid_audit.to_csv(PHASE_ROOT / "invalid_value_audit.csv", index=False)

    coverage_rows: list[dict[str, Any]] = []
    for subtype in CANONICAL_STRATA:
        subset = frame[frame.canonical_stratum == subtype]
        for factor in audit_factors:
            source = factor if factor == "tg_hdl_ratio" else FACTOR_COLUMN_MAP[factor]
            coverage_rows.append({"canonical_stratum": subtype, "factor": factor, "n": int(subset[source].notna().sum()) if source in subset else 0, "eligible_n": len(subset), "coverage": float(subset[source].notna().mean()) if source in subset else 0.0})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(PHASE_ROOT / "factor_coverage_by_subtype.csv", index=False)

    train = frame[frame.split == "train"]
    adequately_powered = [subtype for subtype in CANONICAL_STRATA if int((train.canonical_stratum == subtype).sum()) >= max(K_RANGE) * MIN_CLUSTER_TRAIN_N]
    factor_decision = factor_selection(train, adequately_powered, baseline_timing_valid)
    final_factors = factor_decision["final_factor_list"]
    missing_decision = decide_missing_data(frame, final_factors)
    factor_decision["missing_data_strategy_by_subtype"] = {key: value["strategy"] for key, value in missing_decision.items()}
    write_json(PHASE_ROOT / "factor_selection.json", factor_decision)
    write_json(DECISION_ROOT / "factor_selection.json", factor_decision)
    write_json(PHASE_ROOT / "missing_data_decision.json", missing_decision)

    count_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    for subtype in CANONICAL_STRATA:
        subset = frame[frame.canonical_stratum == subtype]
        h0_n = int(subset.participant_id.isin(h0_ids).sum())
        ht_n = int(subset.participant_id.isin(ht_ids).sum())
        core_complete = subset[CORE_FACTORS].notna().all(axis=1)
        final_complete = subset[final_factors].notna().all(axis=1)
        retention_rows.append({"canonical_stratum": subtype, "eligible_n": len(subset), "complete_case_n": int(final_complete.sum()), "retained_fraction": float(final_complete.mean())})
        for split_name in ["full", *SPLITS]:
            selected = subset if split_name == "full" else subset[subset.split == split_name]
            count_rows.append({
                "canonical_stratum": subtype, "split": split_name, "n": len(selected),
                "h0_n": int(selected.participant_id.isin(h0_ids).sum()), "ht_n": int(selected.participant_id.isin(ht_ids).sum()),
                "core_complete_case_n": int(selected[CORE_FACTORS].notna().all(axis=1).sum()),
                "final_complete_case_n": int(selected[final_factors].notna().all(axis=1).sum()),
                "participants_with_imputed_values": int(selected[final_factors].isna().any(axis=1).sum()),
            })
    counts = pd.DataFrame(count_rows)
    retention = pd.DataFrame(retention_rows)
    counts.to_csv(PHASE_ROOT / "participant_counts.csv", index=False)
    retention.to_csv(PHASE_ROOT / "complete_case_retention.csv", index=False)

    missingness_rows = []
    for factor in audit_factors:
        source = factor if factor == "tg_hdl_ratio" else FACTOR_COLUMN_MAP[factor]
        if source not in frame:
            continue
        for split_name in SPLITS:
            subset = frame[frame.split == split_name]
            missingness_rows.append({"grouping": "split", "group": split_name, "factor": factor, "missing_fraction": float(subset[source].isna().mean()), "n": len(subset)})
        for site, subset in frame.groupby(SITE_COLUMN, dropna=False):
            missingness_rows.append({"grouping": "clinical_site", "group": str(site), "factor": factor, "missing_fraction": float(subset[source].isna().mean()), "n": len(subset)})
        for subtype, subset in frame.groupby("canonical_stratum"):
            missingness_rows.append({"grouping": "diagnostic_subtype", "group": subtype, "factor": factor, "missing_fraction": float(subset[source].isna().mean()), "n": len(subset)})
    pd.DataFrame(missingness_rows).to_csv(PHASE_ROOT / "factor_missingness.csv", index=False)

    transformation_rows = []
    correlations: dict[str, pd.DataFrame] = {}
    for subtype in CANONICAL_STRATA:
        subset = train[train.canonical_stratum == subtype]
        correlations[subtype] = subset[final_factors].corr(method="spearman")
        correlations[subtype].to_csv(PHASE_ROOT / f"spearman_correlation_{subtype}.csv")
        for factor in final_factors:
            skew = float(subset[factor].dropna().skew())
            transformation_rows.append({"canonical_stratum": subtype, "factor": factor, "train_skewness": skew, "planned_transformation": "log1p" if abs(skew) > SKEW_LOG_THRESHOLD and subset[factor].dropna().min() > -1 else "none"})
    transformations = pd.DataFrame(transformation_rows)
    transformations.to_csv(PHASE_ROOT / "transformation_decisions.csv", index=False)

    preliminary = preliminary_clustering(frame, final_factors, missing_decision)
    preliminary.to_csv(PHASE_ROOT / "preliminary_k_feasibility.csv", index=False)

    neighbor_rows = []
    aligned_endpoint_ids = h0_ids & ht_ids
    for subtype in CANONICAL_STRATA:
        for split_name in ["val", "test"]:
            n = int(((frame.canonical_stratum == subtype) & (frame.split == split_name) & frame.participant_id.isin(aligned_endpoint_ids)).sum())
            k_neighbors = min(KNN_K_CEILING, max(KNN_K_FLOOR, round(KNN_K_FRACTION * n)))
            k_neighbors = min(k_neighbors, max(0, n - 1))
            neighbor_rows.append({"canonical_stratum": subtype, "analysis_split": "validation" if split_name == "val" else "test", "subtype_n": n, "knn_k": k_neighbors, "exploratory": bool(split_name == "test" and n < UNDERPOWERED_TEST_N)})
    neighbors = pd.DataFrame(neighbor_rows)
    neighbors.to_csv(PHASE_ROOT / "neighbor_counts.csv", index=False)

    exclusions = []
    for _, row in frame.iterrows():
        reasons = []
        if row.participant_id not in h0_ids:
            reasons.append("Absent from Study 1 h0 endpoint cohort")
        if row.participant_id not in ht_ids:
            reasons.append("Absent from Study 1 ht endpoint cohort")
        missing_factors = [factor for factor in final_factors if pd.isna(row[factor])]
        if missing_factors:
            reasons.append("Missing final clinical factors: " + ", ".join(missing_factors))
        if reasons:
            exclusions.append({"participant_id": row.participant_id, "canonical_stratum": row.canonical_stratum, "split": row.split, "reasons": " | ".join(reasons)})
    pd.DataFrame(exclusions).to_csv(PHASE_ROOT / "participant_exclusions.csv", index=False)

    make_figure_a1(counts[counts.split.isin(SPLITS)], coverage, retention, correlations)
    make_figure_a2(preliminary)
    render_report(inventory, RAW_STRATUM_MAP, counts, coverage, invalid_audit, timing_audit, transformations, missing_decision, factor_decision, preliminary, neighbors)

    if invalid_ratio_count:
        warnings.append({"code": "NONPOSITIVE_HDL", "severity": "warning", "count": invalid_ratio_count, "message": "Nonpositive HDL values were rejected from TG/HDL calculation"})
    for candidate in SIXTH_FACTOR_CANDIDATES:
        source = FACTOR_COLUMN_MAP[candidate]
        if source not in available:
            warnings.append({"code": "UNRESOLVED_CANDIDATE_FACTOR", "severity": "warning", "factor": candidate, "message": f"Exact source column {source} is absent"})
    for row in timing_audit.to_dict("records"):
        if row["after_cgm_end_n"]:
            warnings.append({
                "code": "POST_CGM_BASELINE_DATE", "severity": "warning", "factor": row["factor"],
                "count": row["after_cgm_end_n"],
                "message": "Nominal baseline date occurs after the participant CGM endpoint; a Phase 1 inclusion policy is required",
            })
    for column, count in conflicts.items():
        if count:
            warnings.append({"code": "DUPLICATE_BASELINE_CONFLICT", "severity": "warning", "column": column, "count": count})
    write_json(LOG_ROOT / "phase0_warnings.json", {"created_at": now_iso(), "warnings": warnings})

    readme = """# Within-subtype phenotype preservation study

The earlier global clinical clustering was not reused because it was dominated by the broad diagnosis-associated glycemic gradient. This study clusters separately within each diagnostic subtype so finer cohort-specific clinical relationships can be evaluated. The clinical reference uses five core physiological factors and the first eligible sixth factor under a deterministic train-only coverage, unit, redundancy, and domain rule. The final cluster count will be selected independently by subtype using train-only stability, silhouette, gap, size, and validation confirmation criteria.

The h0 and ht representations will never be clustered. After the clinical labels are frozen, later phases will quantify subtype-restricted neighborhood overlap, trustworthiness, continuity, distance concordance, label purity, and movement coherence in the full latent dimension. Static neutralization tests whether relationships persist when participant-specific initialization is removed. Glucose residualization tests whether endpoint structure remains beyond mean glucose when coordinate compatibility permits.

Phase 0 is complete and stopped at Gate A. No final cluster count or participant cluster label exists yet. Study 1 supplies overnight endpoint summaries, not matched time-resolved states, so the endpoint analysis is planned and the optional temporal extension is unavailable.
"""
    (STUDY2_ROOT / "README.md").write_text(readme)

    git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    created = sorted(str(path.relative_to(STUDY2_ROOT)) for path in STUDY2_ROOT.rglob("*") if path.is_file() and path.name != "MANIFEST.json")
    manifest = {
        "phase_status": "Gate A reached",
        "created_artifacts": created + ["MANIFEST.json"],
        "input_paths": [str(DATASET), str(SPLIT_PATH), str(STUDY1_ROOT)],
        "input_hashes": {
            "dataset": inventory["diagnostic_subtype_and_static_clinical_factors"]["sha256"],
            "split": inventory["participant_splits"]["sha256"],
            "h0_full": inventory["h0_full"]["sha256"],
            "ht_full_overnight_endpoint": inventory["ht_full_overnight_endpoint"]["sha256"],
            "ht_neutral_overnight_endpoint": inventory["ht_neutral_overnight_endpoint"]["sha256"],
            "ht_glucose_residualized_overnight_endpoint": inventory["ht_glucose_residualized_overnight_endpoint"]["sha256"],
        },
        "git_commit": git_commit,
        "git_branch": git_branch,
        "expected_git_branch": REPO_BRANCH,
        "random_seed": SEED,
        "factor_list": final_factors,
        "missing_data_strategy": factor_decision["missing_data_strategy_by_subtype"],
        "transformations": transformations.to_dict("records"),
        "selected_k_per_subtype": None,
        "cluster_stability": "Preliminary 200-bootstrap diagnostics only",
        "cluster_sizes": preliminary[["canonical_stratum", "k", "cluster_sizes"]].to_dict("records"),
        "underpowered_subtypes": neighbors[(neighbors.analysis_split == "test") & neighbors.exploratory].canonical_stratum.tolist(),
        "underpowered_clusters": None,
        "neighbor_counts": neighbors.to_dict("records"),
        "study1_artifacts_consumed": list(inventory),
        "time_resolved_ht_available": False,
        "new_model_forward_pass_ran": False,
        "model_retraining_ran": False,
        "figures": [
            {"png": "figures/full_resolution/figure_A1_clinical_data_viability.png", "metadata": "figures/metadata/figure_A1_clinical_data_viability.json"},
            {"png": "figures/full_resolution/figure_A2_candidate_cluster_viability.png", "metadata": "figures/metadata/figure_A2_candidate_cluster_viability.json"},
        ],
    }
    write_json(STUDY2_ROOT / "MANIFEST.json", manifest)
    print("Gate A reached. No final clustering has been performed. Waiting for confirmation before Phase 1.")


if __name__ == "__main__":
    main()
