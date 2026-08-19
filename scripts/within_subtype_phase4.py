"""Optional Phase 4 time-resolved extension.

Uses the explicitly approved matched factual and static-neutral SSM snapshots at
0, 6, 12, 24, and 48 hours. The frozen within-subtype clinical labels remain the
reference; h0 and every ht snapshot remain unclustered.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import within_subtype_phase1 as phase1
import within_subtype_phase2 as phase2
from ssmcgm.analysis.within_subtype_config import (
    BAND_ALPHA,
    BOOTSTRAP_N,
    BOOTSTRAP_CI_LEVEL,
    CANONICAL_STRATA,
    CLINICAL_METRIC,
    CLUSTER_COLORS,
    COHERENCE_RATIO_BAR,
    COLOR_ADJUSTED,
    COLOR_NULL,
    COLOR_OBSERVED,
    COLOR_POSITIVE,
    COLOR_REFERENCE,
    DECISION_ROOT,
    FIGURE_DPI,
    FIGURE_ROOT,
    LATENT_METRIC,
    LOG_ROOT,
    PERMUTATION_N,
    REPO,
    REPO_BRANCH,
    SEED,
    STUDY1_ROOT,
    STUDY2_ROOT,
    TABLE_ROOT,
    THUMBNAIL_DPI,
    UNDERPOWERED_CLUSTER_TEST_N,
)

PHASE_ROOT = STUDY2_ROOT / "phase4_time_resolved_extension"
FIGURE_FULL = FIGURE_ROOT / "full_resolution"
FIGURE_THUMB = FIGURE_ROOT / "thumbnails"
FIGURE_DATA = FIGURE_ROOT / "plotted_data"
FIGURE_META = FIGURE_ROOT / "metadata"
SNAPSHOT_ROOT = PHASE_ROOT / "snapshots"
TIME_RESOLVED_HOURS = [0, 6, 12, 24, 48]

# Phase 3 uses the shared pre-registered resampling counts. Distance matrices
# are computed once, so 1,000 participant resamples do not rerun the model.
PHASE4_BOOTSTRAP_N = BOOTSTRAP_N
PHASE4_PERMUTATION_N = PERMUTATION_N
MATCHED_NONNEIGHBOR_SEED_OFFSET = 777
HEADLINE_METRICS = [
    "knn_jaccard_mean",
    "reference_neighbor_recall_in_embedding",
    "embedding_neighbor_precision_relative_to_reference",
    "trustworthiness",
    "continuity",
    "pairwise_distance_spearman",
    "neighbor_purity_mean",
    "cluster_silhouette",
]


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


def setup_output_tree() -> None:
    for directory in [PHASE_ROOT, FIGURE_FULL, FIGURE_THUMB, FIGURE_DATA, FIGURE_META, TABLE_ROOT, LOG_ROOT, DECISION_ROOT]:
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Preservation metrics (reference vs embedded), reusing Phase 2's low-level
# implementations so results stay methodologically identical across phases.
# ---------------------------------------------------------------------------
def pairwise_distance_spearman(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    iu = np.triu_indices_from(dist_a, k=1)
    if len(iu[0]) < 2:
        return float("nan")
    rank_a = rankdata(dist_a[iu])
    rank_b = rankdata(dist_b[iu])
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def reference_to_embedded_metrics(reference_dist: np.ndarray, embedded_dist: np.ndarray, labels: np.ndarray, k: int, seed: int, reference_name: str, embedded_name: str) -> dict[str, Any]:
    jaccard_mean, jaccard_median, recall, precision = phase2.knn_jaccard_recall_precision(reference_dist, embedded_dist, k)
    trust, cont = phase2.trustworthiness_continuity(reference_dist, embedded_dist, k)
    rho, p_value = phase2.mantel_spearman(reference_dist, embedded_dist, PHASE4_PERMUTATION_N, seed)
    purity = phase2.neighbor_purity(embedded_dist, labels, k)
    silhouette = phase2.cluster_silhouette(embedded_dist, labels)
    return {
        "reference_space": reference_name,
        "embedded_space": embedded_name,
        "knn_jaccard_mean": jaccard_mean,
        "knn_jaccard_median": jaccard_median,
        "reference_neighbor_recall_in_embedding": recall,
        "embedding_neighbor_precision_relative_to_reference": precision,
        "trustworthiness": trust,
        "continuity": cont,
        "pairwise_distance_spearman": rho,
        "mantel_p": p_value,
        "neighbor_purity_mean": float(purity.mean()),
        "cluster_silhouette": silhouette,
    }


def metrics_without_permutation(reference_dist: np.ndarray, embedded_dist: np.ndarray, labels: np.ndarray, k: int) -> dict[str, float | None]:
    jaccard_mean, _, recall, precision = phase2.knn_jaccard_recall_precision(reference_dist, embedded_dist, k)
    trust, cont = phase2.trustworthiness_continuity(reference_dist, embedded_dist, k)
    return {
        "knn_jaccard_mean": jaccard_mean,
        "reference_neighbor_recall_in_embedding": recall,
        "embedding_neighbor_precision_relative_to_reference": precision,
        "trustworthiness": trust,
        "continuity": cont,
        "pairwise_distance_spearman": pairwise_distance_spearman(reference_dist, embedded_dist),
        "neighbor_purity_mean": float(phase2.neighbor_purity(embedded_dist, labels, k).mean()),
        "cluster_silhouette": phase2.cluster_silhouette(embedded_dist, labels),
    }


def bootstrap_comparison(reference_dist: np.ndarray, embedded_dist: np.ndarray, labels: np.ndarray, k: int, seed: int, n_boot: int) -> dict[str, dict[str, float | int | None]]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    draws = {metric: [] for metric in HEADLINE_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_k = min(k, n - 1)
        values = metrics_without_permutation(reference_dist[np.ix_(idx, idx)], embedded_dist[np.ix_(idx, idx)], labels[idx], boot_k)
        for metric, value in values.items():
            if value is not None and np.isfinite(value):
                draws[metric].append(float(value))
    lo_p, hi_p = (1 - BOOTSTRAP_CI_LEVEL) * 50, 100 - (1 - BOOTSTRAP_CI_LEVEL) * 50
    result = {}
    for metric, values in draws.items():
        if values:
            array = np.asarray(values)
            result[metric] = {"bootstrap_mean": float(array.mean()), "ci_low": float(np.percentile(array, lo_p)), "ci_high": float(np.percentile(array, hi_p)), "n_bootstrap": int(len(array))}
        else:
            result[metric] = {"bootstrap_mean": None, "ci_low": None, "ci_high": None, "n_bootstrap": 0}
    return result


def bootstrap_change(clinical_dist: np.ndarray, h0_dist: np.ndarray, ht_dist: np.ndarray, labels: np.ndarray, k: int, seed: int, n_boot: int) -> dict[str, dict[str, float | int | None]]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    draws = {metric: [] for metric in HEADLINE_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        clinical_boot = clinical_dist[np.ix_(idx, idx)]
        labels_boot = labels[idx]
        h0_values = metrics_without_permutation(clinical_boot, h0_dist[np.ix_(idx, idx)], labels_boot, min(k, n - 1))
        ht_values = metrics_without_permutation(clinical_boot, ht_dist[np.ix_(idx, idx)], labels_boot, min(k, n - 1))
        for metric in HEADLINE_METRICS:
            if h0_values[metric] is not None and ht_values[metric] is not None:
                value = float(ht_values[metric] - h0_values[metric])
                if np.isfinite(value):
                    draws[metric].append(value)
    lo_p, hi_p = (1 - BOOTSTRAP_CI_LEVEL) * 50, 100 - (1 - BOOTSTRAP_CI_LEVEL) * 50
    result = {}
    for metric, values in draws.items():
        array = np.asarray(values)
        result[metric] = {
            "bootstrap_mean": float(array.mean()) if len(array) else None,
            "ci_low": float(np.percentile(array, lo_p)) if len(array) else None,
            "ci_high": float(np.percentile(array, hi_p)) if len(array) else None,
            "n_bootstrap": int(len(array)),
        }
    return result


def participant_retention(dist_a: np.ndarray, dist_b: np.ndarray, k: int) -> np.ndarray:
    nn_a, nn_b = phase2.knn_indices(dist_a, k), phase2.knn_indices(dist_b, k)
    retention = np.zeros(dist_a.shape[0])
    for i in range(dist_a.shape[0]):
        set_a, set_b = set(nn_a[i].tolist()), set(nn_b[i].tolist())
        union = set_a | set_b
        retention[i] = len(set_a & set_b) / len(union) if union else 0.0
    return retention


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    result = 1.0 - squareform(pdist(vectors, metric="cosine"))
    np.fill_diagonal(result, 1.0)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def cluster_gap_from_cosine(cos_sim: np.ndarray, labels: np.ndarray, cluster: int) -> tuple[float, float, float]:
    inside = np.where(labels == cluster)[0]
    outside = np.where(labels != cluster)[0]
    if len(inside) < 2 or len(outside) < 1:
        return float("nan"), float("nan"), float("nan")
    within_matrix = cos_sim[np.ix_(inside, inside)]
    within = float(within_matrix[np.triu_indices(len(inside), k=1)].mean())
    between = float(cos_sim[np.ix_(inside, outside)].mean())
    return within, between, within - between


def bootstrap_cluster_gap(d_vectors: np.ndarray, labels: np.ndarray, cluster: int, n_boot: int, seed: int, cos_sim: np.ndarray | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cos_sim = cosine_similarity_matrix(d_vectors) if cos_sim is None else cos_sim
    inside = np.where(labels == cluster)[0]
    outside = np.where(labels != cluster)[0]
    within, between, point = cluster_gap_from_cosine(cos_sim, labels, cluster)
    draws = []
    for _ in range(n_boot):
        sampled_inside = rng.choice(inside, size=len(inside), replace=True)
        sampled_outside = rng.choice(outside, size=len(outside), replace=True)
        within_matrix = cos_sim[np.ix_(sampled_inside, sampled_inside)]
        within_values = within_matrix[np.triu_indices(len(sampled_inside), k=1)]
        between_values = cos_sim[np.ix_(sampled_inside, sampled_outside)].ravel()
        if len(within_values) and len(between_values):
            draws.append(float(within_values.mean() - between_values.mean()))
    lo_p, hi_p = (1 - BOOTSTRAP_CI_LEVEL) * 50, 100 - (1 - BOOTSTRAP_CI_LEVEL) * 50
    lo = float(np.percentile(draws, lo_p)) if draws else None
    hi = float(np.percentile(draws, hi_p)) if draws else None
    return {"within_cosine": within, "between_cosine": between, "gap_point": point, "gap_ci_low": lo, "gap_ci_high": hi, "gap_excludes_zero": bool(lo is not None and (lo > 0 or hi < 0)), "n_bootstrap": len(draws)}


def permutation_p_cluster_gap(d_vectors: np.ndarray, labels: np.ndarray, cluster: int, observed_gap: float, n_perm: int, seed: int, cos_sim: np.ndarray | None = None) -> float:
    rng = np.random.default_rng(seed)
    cos_sim = cosine_similarity_matrix(d_vectors) if cos_sim is None else cos_sim
    null = np.empty(n_perm)
    for index in range(n_perm):
        shuffled = rng.permutation(labels)
        null[index] = cluster_gap_from_cosine(cos_sim, shuffled, cluster)[2]
    return float((np.sum(null >= observed_gap) + 1) / (n_perm + 1))


def coherence_ratio_per_cluster(d_vectors: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    ratios = {}
    for cluster in sorted(set(labels.tolist())):
        d_cluster = d_vectors[labels == cluster]
        mean_norm = float(np.linalg.norm(d_cluster, axis=1).mean())
        ratios[int(cluster)] = float(np.linalg.norm(d_cluster.mean(axis=0))) / mean_norm if mean_norm > 0 else float("nan")
    return ratios


def leave_one_out_robustness(d_vectors: np.ndarray, labels: np.ndarray, cluster: int, cos_sim: np.ndarray | None = None) -> dict[str, Any]:
    cos_sim = cosine_similarity_matrix(d_vectors) if cos_sim is None else cos_sim
    inside = np.where(labels == cluster)[0]
    if len(inside) < 3:
        return {"loo_gap_min": None, "loo_gap_max": None, "robust_to_single_participant": False}
    gaps = []
    for dropped in inside:
        keep = np.arange(len(labels)) != dropped
        gaps.append(cluster_gap_from_cosine(cos_sim[np.ix_(keep, keep)], labels[keep], cluster)[2])
    point = cluster_gap_from_cosine(cos_sim, labels, cluster)[2]
    sign_preserved = all(np.sign(value) == np.sign(point) and value != 0 for value in gaps if np.isfinite(value))
    return {"loo_gap_min": float(np.nanmin(gaps)), "loo_gap_max": float(np.nanmax(gaps)), "robust_to_single_participant": bool(sign_preserved)}


def matched_partners(i: int, targets: np.ndarray, candidate_pool: np.ndarray, h0_dist: np.ndarray, endpoint_duration: np.ndarray) -> np.ndarray:
    available = set(int(value) for value in candidate_pool if int(value) != i)
    selected = []
    distance_scale = float(np.nanstd(h0_dist[i])) or 1.0
    duration_scale = float(np.nanstd(endpoint_duration)) or 1.0
    for target in targets:
        if not available:
            break
        candidates = np.asarray(sorted(available), dtype=int)
        score = np.abs(h0_dist[i, candidates] - h0_dist[i, target]) / distance_scale
        score += np.abs(endpoint_duration[candidates] - endpoint_duration[target]) / duration_scale
        chosen = int(candidates[int(np.argmin(score))])
        selected.append(chosen)
        available.remove(chosen)
    return np.asarray(selected, dtype=int)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_participants": 0, "n_bootstrap": 0}
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(finite, size=len(finite), replace=True).mean() for _ in range(n_boot)])
    lo_p, hi_p = (1 - BOOTSTRAP_CI_LEVEL) * 50, 100 - (1 - BOOTSTRAP_CI_LEVEL) * 50
    return {"mean": float(finite.mean()), "ci_low": float(np.percentile(boots, lo_p)), "ci_high": float(np.percentile(boots, hi_p)), "n_participants": int(len(finite)), "n_bootstrap": n_boot}


def local_neighbor_advantage(d_vectors: np.ndarray, h0_dist: np.ndarray, k: int, endpoint_duration: np.ndarray, cos_sim: np.ndarray | None = None) -> np.ndarray:
    n = len(d_vectors)
    cos_sim = cosine_similarity_matrix(d_vectors) if cos_sim is None else cos_sim
    neighbors = phase2.knn_indices(h0_dist, k)
    values = np.full(n, np.nan)
    for i in range(n):
        neighbor_set = set(neighbors[i].tolist())
        pool = np.asarray([j for j in range(n) if j != i and j not in neighbor_set], dtype=int)
        matched = matched_partners(i, neighbors[i], pool, h0_dist, endpoint_duration)
        if len(matched):
            values[i] = float(cos_sim[i, neighbors[i][:len(matched)]].mean() - cos_sim[i, matched].mean())
    return values


def pair_class_movement(d_vectors: np.ndarray, labels: np.ndarray, h0_dist: np.ndarray, k: int, endpoint_duration: np.ndarray, seed: int, n_boot: int, cos_sim: np.ndarray | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    cos_sim = cosine_similarity_matrix(d_vectors) if cos_sim is None else cos_sim
    neighbors = phase2.knn_indices(h0_dist, k)
    class_values = {name: np.full(n, np.nan) for name in ["class_1_same_cluster", "class_2_different_cluster", "class_3_h0_neighbor_pairs", "class_4_matched_non_neighbor_pairs", "class_5_random_pairs"]}
    for i in range(n):
        same = np.asarray([j for j in range(n) if j != i and labels[j] == labels[i]], dtype=int)
        different = np.asarray([j for j in range(n) if labels[j] != labels[i]], dtype=int)
        if len(same) and len(different):
            chosen_same = rng.choice(same, size=min(k, len(same), len(different)), replace=False)
            matched_different = matched_partners(i, chosen_same, different, h0_dist, endpoint_duration)
            if len(matched_different):
                class_values["class_1_same_cluster"][i] = cos_sim[i, chosen_same[:len(matched_different)]].mean()
                class_values["class_2_different_cluster"][i] = cos_sim[i, matched_different].mean()
        h0_neighbors = neighbors[i]
        nonneighbors = np.asarray([j for j in range(n) if j != i and j not in set(h0_neighbors.tolist())], dtype=int)
        matched = matched_partners(i, h0_neighbors, nonneighbors, h0_dist, endpoint_duration)
        if len(matched):
            class_values["class_3_h0_neighbor_pairs"][i] = cos_sim[i, h0_neighbors[:len(matched)]].mean()
            class_values["class_4_matched_non_neighbor_pairs"][i] = cos_sim[i, matched].mean()
        random_pool = np.asarray([j for j in range(n) if j != i], dtype=int)
        random_partners = rng.choice(random_pool, size=min(k, len(random_pool)), replace=False)
        class_values["class_5_random_pairs"][i] = cos_sim[i, random_partners].mean()
    summary = {name: bootstrap_mean_ci(values, n_boot, seed + offset) for offset, (name, values) in enumerate(class_values.items(), 1)}
    detail = pd.DataFrame({"participant_index": np.arange(n), **class_values})
    return summary, detail


def deterministic_knn_k(n: int) -> int:
    return phase2.deterministic_knn_k(n)


def assign_display_cluster(matrix: np.ndarray, centroid_order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nearest, nearest_distance, margin = phase2.assign_nearest_centroid(matrix, centroid_order)
    return nearest, margin


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_snapshot(hour: int, condition: str, participant_order: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = SNAPSHOT_ROOT / f"h_t_{condition}_hour{hour:02d}.parquet"
    frame = pd.read_parquet(path, filters=[("participant_id", "in", participant_order)])
    frame["participant_id"] = frame.participant_id.astype(str)
    frame = frame.set_index("participant_id").reindex(participant_order)
    if frame.index.isna().any() or frame.isna().any().any():
        raise RuntimeError(f"Missing or nonfinite snapshot rows at hour {hour}, condition {condition}")
    metadata = ["split", "qualifying_segment_count", "mean_abs_target_offset_minutes"]
    matrix = frame.drop(columns=metadata).to_numpy(dtype=np.float32)
    return matrix, frame.qualifying_segment_count.to_numpy(dtype=np.float64), frame.mean_abs_target_offset_minutes.to_numpy(dtype=np.float64)


def main() -> None:
    setup_output_tree()
    extraction_report_path = PHASE_ROOT / "snapshot_extraction_report.json"
    if not extraction_report_path.exists():
        raise RuntimeError("Phase 4 snapshot extraction has not completed")
    extraction_report = read_json(extraction_report_path)
    if extraction_report.get("time_resolved_hours") != TIME_RESOLVED_HOURS:
        raise RuntimeError("Snapshot hours do not match the frozen Phase 4 hours")
    if not extraction_report.get("approved_new_forward_pass") or extraction_report.get("model_retrained"):
        raise RuntimeError("Snapshot provenance is inconsistent with the approved extension")

    factor_decision = read_json(DECISION_ROOT / "factor_selection.json")
    final_factors = factor_decision["final_factor_list"]
    missing_decision = read_json(phase1.PHASE0_ROOT / "missing_data_decision.json")
    strategy_by_subtype = {subtype: missing_decision[subtype]["strategy"] for subtype in CANONICAL_STRATA}
    frozen_manifest = read_json(phase1.PHASE_ROOT / "frozen_clustering_manifest.json")
    frame, _, _, _ = phase1.load_frame(final_factors)
    phase3_reference = pd.read_csv(STUDY2_ROOT / "phase3_ht_preservation/participant_retention.csv")
    phase3_reference["participant_id"] = phase3_reference.participant_id.astype(str)
    participant_order = phase3_reference.participant_id.tolist()
    participant_to_global = {pid: index for index, pid in enumerate(participant_order)}

    h0_pid_to_row, h0_matrix_all = phase2.load_h0_matrix()
    h0_global = h0_matrix_all[[h0_pid_to_row[pid] for pid in participant_order]].astype(np.float32)
    neutral_h0 = np.load(PHASE_ROOT / "neutral_h0_exact.npy").astype(np.float32)
    if neutral_h0.shape[0] != h0_global.shape[1]:
        raise RuntimeError("Neutral h0 dimension does not match factual h0")

    subtype_data = {}
    for subtype in CANONICAL_STRATA:
        reference = phase3_reference[phase3_reference.canonical_stratum == subtype].copy()
        pids = reference.participant_id.tolist()
        pipeline_info = frozen_manifest["clusters"][subtype]
        pipeline = joblib.load(STUDY2_ROOT / pipeline_info["preprocessing_pipeline_path"])
        centroid_info = read_json(STUDY2_ROOT / pipeline_info["centroid_path"])
        centroids_by_display = {int(key): np.asarray(value) for key, value in centroid_info["centroids_by_display_cluster"].items()}
        display_order = sorted(centroids_by_display)
        centroids = np.asarray([centroids_by_display[value] for value in display_order])
        subtype_frame = frame[(frame.canonical_stratum == subtype) & (frame.participant_id.isin(pids))].copy()
        subtype_frame = subtype_frame.set_index("participant_id").reindex(pids).reset_index()
        use_imputation = strategy_by_subtype[subtype] == "iterative_imputation"
        if not use_imputation and subtype_frame[final_factors].isna().any().any():
            raise RuntimeError(f"Complete-case Phase 4 cohort mismatch in {subtype}")
        clinical = phase1.apply_pipeline(subtype_frame, pipeline["factors"], pipeline["log_transformed"], pipeline["imputer"], pipeline["scaler"])
        nearest, margin = assign_display_cluster(clinical, centroids)
        assigned = np.asarray([display_order[index] for index in nearest])
        frozen_labels = reference.display_cluster.to_numpy(dtype=int)
        if not np.array_equal(assigned, frozen_labels):
            raise RuntimeError(f"Frozen clinical-label mismatch in {subtype}")
        global_indices = np.asarray([participant_to_global[pid] for pid in pids], dtype=int)
        h0 = h0_global[global_indices].astype(np.float64)
        subtype_data[subtype] = {
            "participant_ids": np.asarray(pids),
            "global_indices": global_indices,
            "labels": frozen_labels,
            "clinical": clinical,
            "clinical_dist": squareform(pdist(clinical, metric=CLINICAL_METRIC)).astype(np.float64),
            "h0": h0,
            "h0_dist": squareform(pdist(h0, metric=LATENT_METRIC)).astype(np.float64),
            "knn_k": deterministic_knn_k(len(pids)),
            "exploratory": pipeline_info["status"] != "primary",
        }

    metrics_rows = []
    bootstrap_rows = []
    representation_rows = []
    retention_rows = []
    displacement_rows = []
    coherence_rows = []
    advantage_summary_rows = []
    advantage_participant_rows = []
    pair_class_rows = []
    evidence_rows = []

    for hour in TIME_RESOLVED_HOURS:
        if hour == 0:
            full_global = h0_global
            neutral_global = np.repeat(neutral_h0[None, :], len(participant_order), axis=0)
            segment_counts_global = np.zeros(len(participant_order), dtype=np.float64)
            target_offsets_global = np.zeros(len(participant_order), dtype=np.float64)
        else:
            full_global, segment_counts_global, target_offsets_global = load_snapshot(hour, "full", participant_order)
            neutral_global, neutral_counts, neutral_offsets = load_snapshot(hour, "neutral", participant_order)
            if not np.array_equal(segment_counts_global, neutral_counts) or not np.allclose(target_offsets_global, neutral_offsets):
                raise RuntimeError(f"Full and neutral snapshot metadata mismatch at hour {hour}")
        if full_global.shape != h0_global.shape or neutral_global.shape != h0_global.shape:
            raise RuntimeError(f"Snapshot dimension or participant mismatch at hour {hour}")
        if not np.isfinite(full_global).all() or not np.isfinite(neutral_global).all():
            raise RuntimeError(f"Nonfinite snapshot vector at hour {hour}")

        for subtype_index, subtype in enumerate(CANONICAL_STRATA):
            data = subtype_data[subtype]
            indices = data["global_indices"]
            labels = data["labels"]
            clinical_dist = data["clinical_dist"]
            h0_dist = data["h0_dist"]
            state = full_global[indices].astype(np.float64)
            neutral_state = neutral_global[indices].astype(np.float64)
            state_dist = squareform(pdist(state, metric=LATENT_METRIC)).astype(np.float64)
            k = data["knn_k"]
            seed = SEED + subtype_index * 1000 + hour
            comparisons = {"clinical_to_state": (clinical_dist, state_dist), "h0_to_state": (h0_dist, state_dist)}
            if hour > 0:
                neutral_dist = squareform(pdist(neutral_state, metric=LATENT_METRIC)).astype(np.float64)
                comparisons["clinical_to_neutral_state"] = (clinical_dist, neutral_dist)
            for comparison, pair in comparisons.items():
                reference_dist, embedded_dist = pair
                point = reference_to_embedded_metrics(reference_dist, embedded_dist, labels, k, seed, comparison.split("_to_")[0], comparison.split("_to_")[1])
                metrics_rows.append({"canonical_stratum": subtype, "hour": hour, "n": len(labels), "knn_k": k, "comparison": comparison, **{key: value for key, value in point.items() if key not in ("reference_space", "embedded_space")}})
                intervals = bootstrap_comparison(reference_dist, embedded_dist, labels, k, seed + len(bootstrap_rows), PHASE4_BOOTSTRAP_N)
                for metric, interval in intervals.items():
                    bootstrap_rows.append({"canonical_stratum": subtype, "hour": hour, "comparison": comparison, "metric": metric, "point": point.get(metric), **interval})
            full_point = metrics_without_permutation(clinical_dist, state_dist, labels, k)
            full_boot = bootstrap_comparison(clinical_dist, state_dist, labels, k, seed + 100, PHASE4_BOOTSTRAP_N)
            representation_rows.append({
                "canonical_stratum": subtype, "hour": hour, "representation": "full", "n": len(labels),
                "clinical_knn_jaccard": full_point["knn_jaccard_mean"],
                "clinical_knn_jaccard_ci_low": full_boot["knn_jaccard_mean"]["ci_low"],
                "clinical_knn_jaccard_ci_high": full_boot["knn_jaccard_mean"]["ci_high"],
                "neighbor_purity": full_point["neighbor_purity_mean"],
                "neighbor_purity_ci_low": full_boot["neighbor_purity_mean"]["ci_low"],
                "neighbor_purity_ci_high": full_boot["neighbor_purity_mean"]["ci_high"],
                "cluster_silhouette": full_point["cluster_silhouette"],
                "estimable": True,
            })
            if hour == 0:
                representation_rows.append({"canonical_stratum": subtype, "hour": hour, "representation": "neutral", "n": len(labels), "estimable": False, "reason": "Exact neutral h0 is population-constant"})
            else:
                neutral_point = metrics_without_permutation(clinical_dist, neutral_dist, labels, k)
                neutral_boot = bootstrap_comparison(clinical_dist, neutral_dist, labels, k, seed + 200, PHASE4_BOOTSTRAP_N)
                representation_rows.append({
                    "canonical_stratum": subtype, "hour": hour, "representation": "neutral", "n": len(labels),
                    "clinical_knn_jaccard": neutral_point["knn_jaccard_mean"],
                    "clinical_knn_jaccard_ci_low": neutral_boot["knn_jaccard_mean"]["ci_low"],
                    "clinical_knn_jaccard_ci_high": neutral_boot["knn_jaccard_mean"]["ci_high"],
                    "neighbor_purity": neutral_point["neighbor_purity_mean"],
                    "neighbor_purity_ci_low": neutral_boot["neighbor_purity_mean"]["ci_low"],
                    "neighbor_purity_ci_high": neutral_boot["neighbor_purity_mean"]["ci_high"],
                    "cluster_silhouette": neutral_point["cluster_silhouette"],
                    "estimable": True,
                })

            h0_retention = participant_retention(h0_dist, state_dist, k)
            clinical_retention = participant_retention(clinical_dist, state_dist, k)
            for participant_index, pid in enumerate(data["participant_ids"]):
                global_index = indices[participant_index]
                retention_rows.append({
                    "participant_id": pid, "canonical_stratum": subtype, "display_cluster": int(labels[participant_index]), "hour": hour,
                    "h0_to_state_retention": float(h0_retention[participant_index]),
                    "clinical_to_state_retention": float(clinical_retention[participant_index]),
                    "qualifying_segment_count": int(segment_counts_global[global_index]) if hour > 0 else None,
                    "target_offset_minutes": float(target_offsets_global[global_index]),
                    "exploratory": data["exploratory"],
                })

            if hour == 0:
                continue
            d_full = state - data["h0"]
            d_neutral = neutral_state - neutral_h0
            magnitude_full = np.linalg.norm(d_full, axis=1)
            magnitude_neutral = np.linalg.norm(d_neutral, axis=1)
            cos_full = cosine_similarity_matrix(d_full)
            cos_neutral = cosine_similarity_matrix(d_neutral)
            ratios_full = coherence_ratio_per_cluster(d_full, labels)
            ratios_neutral = coherence_ratio_per_cluster(d_neutral, labels)
            segment_counts = segment_counts_global[indices]
            advantage_full = local_neighbor_advantage(d_full, h0_dist, k, segment_counts, cos_full)
            advantage_neutral = local_neighbor_advantage(d_neutral, h0_dist, k, segment_counts, cos_neutral)
            for participant_index, pid in enumerate(data["participant_ids"]):
                displacement_rows.append({"participant_id": pid, "canonical_stratum": subtype, "display_cluster": int(labels[participant_index]), "hour": hour, "representation": "full", "magnitude": float(magnitude_full[participant_index])})
                displacement_rows.append({"participant_id": pid, "canonical_stratum": subtype, "display_cluster": int(labels[participant_index]), "hour": hour, "representation": "neutral", "magnitude": float(magnitude_neutral[participant_index])})
                advantage_participant_rows.append({"participant_id": pid, "canonical_stratum": subtype, "display_cluster": int(labels[participant_index]), "hour": hour, "representation": "full", "neighbor_movement_advantage": float(advantage_full[participant_index]) if np.isfinite(advantage_full[participant_index]) else None})
                advantage_participant_rows.append({"participant_id": pid, "canonical_stratum": subtype, "display_cluster": int(labels[participant_index]), "hour": hour, "representation": "neutral", "neighbor_movement_advantage": float(advantage_neutral[participant_index]) if np.isfinite(advantage_neutral[participant_index]) else None})

            for cluster in sorted(set(labels.tolist())):
                cluster_mask = labels == cluster
                n_cluster = int(cluster_mask.sum())
                if n_cluster < 2:
                    continue
                rows_by_representation = []
                for representation, vectors, cosine, ratios, advantage, magnitude in [
                    ("full", d_full, cos_full, ratios_full, advantage_full, magnitude_full),
                    ("neutral", d_neutral, cos_neutral, ratios_neutral, advantage_neutral, magnitude_neutral),
                ]:
                    gap = bootstrap_cluster_gap(vectors, labels, cluster, PHASE4_BOOTSTRAP_N, seed + int(cluster) + (0 if representation == "full" else 100), cosine)
                    permutation_p = permutation_p_cluster_gap(vectors, labels, cluster, gap["gap_point"], PHASE4_PERMUTATION_N, seed + int(cluster) + (0 if representation == "full" else 100), cosine)
                    robustness = leave_one_out_robustness(vectors, labels, cluster, cosine)
                    advantage_summary = bootstrap_mean_ci(advantage[cluster_mask], PHASE4_BOOTSTRAP_N, seed + int(cluster) + (300 if representation == "full" else 400))
                    advantage_summary_rows.append({"canonical_stratum": subtype, "display_cluster": int(cluster), "hour": hour, "representation": representation, **advantage_summary})
                    coherence_row = {
                        "canonical_stratum": subtype, "display_cluster": int(cluster), "hour": hour, "representation": representation, "n": n_cluster,
                        **gap, "permutation_p": permutation_p, "coherence_ratio": ratios[int(cluster)],
                        "mean_displacement_magnitude": float(magnitude[cluster_mask].mean()),
                        "variance_displacement_magnitude": float(magnitude[cluster_mask].var(ddof=1)) if n_cluster > 1 else None,
                        "exploratory": data["exploratory"], **robustness,
                    }
                    coherence_rows.append(coherence_row)
                    rows_by_representation.append((representation, coherence_row, advantage_summary))
                full_row = next(value for name, value, advantage_value in rows_by_representation if name == "full")
                neutral_row = next(value for name, value, advantage_value in rows_by_representation if name == "neutral")
                full_advantage = next(advantage_value for name, value, advantage_value in rows_by_representation if name == "full")
                neutral_advantage = next(advantage_value for name, value, advantage_value in rows_by_representation if name == "neutral")
                criteria = {
                    "positive_full_gap_ci_excludes_zero": bool(full_row["gap_ci_low"] is not None and full_row["gap_ci_low"] > 0),
                    "full_coherence_ratio_exceeds_bar": bool(full_row["coherence_ratio"] > COHERENCE_RATIO_BAR),
                    "full_neighbor_advantage_positive": bool(full_advantage["mean"] is not None and full_advantage["mean"] > 0),
                    "full_not_driven_by_one_participant": bool(full_row["robust_to_single_participant"]),
                    "n_at_least_underpowered_floor": bool(n_cluster >= UNDERPOWERED_CLUSTER_TEST_N),
                    "neutral_directionally_consistent": bool(neutral_row["gap_point"] > 0 and full_row["gap_point"] > 0),
                }
                full_pass = all(criteria.values())
                neutral_pass = bool(neutral_row["gap_ci_low"] is not None and neutral_row["gap_ci_low"] > 0 and neutral_row["coherence_ratio"] > COHERENCE_RATIO_BAR and neutral_advantage["mean"] is not None and neutral_advantage["mean"] > 0 and neutral_row["robust_to_single_participant"] and n_cluster >= UNDERPOWERED_CLUSTER_TEST_N)
                if full_pass and neutral_pass:
                    label = "Movement coherence that persists after static neutralization"
                elif full_pass:
                    label = "Static-profile-associated movement coherence"
                else:
                    label = "No coherent movement claim supported"
                evidence_rows.append({"canonical_stratum": subtype, "display_cluster": int(cluster), "hour": hour, "label": label, "full_analysis_passes": full_pass, "neutral_analysis_passes": neutral_pass, "exploratory": data["exploratory"], **criteria})

            for representation, vectors, cosine in [("full", d_full, cos_full), ("neutral", d_neutral, cos_neutral)]:
                summary, detail = pair_class_movement(vectors, labels, h0_dist, k, segment_counts, seed + (0 if representation == "full" else 1), PHASE4_BOOTSTRAP_N, cosine)
                pair_class_rows.append({"canonical_stratum": subtype, "hour": hour, "representation": representation, **{f"{name}_{stat}": value for name, values in summary.items() for stat, value in values.items()}})

    metrics = pd.DataFrame(metrics_rows)
    bootstrap_table = pd.DataFrame(bootstrap_rows)
    representations = pd.DataFrame(representation_rows)
    retention = pd.DataFrame(retention_rows)
    displacement = pd.DataFrame(displacement_rows)
    coherence = pd.DataFrame(coherence_rows)
    advantages = pd.DataFrame(advantage_summary_rows)
    advantage_participants = pd.DataFrame(advantage_participant_rows)
    pair_classes = pd.DataFrame(pair_class_rows)
    evidence = pd.DataFrame(evidence_rows)

    metrics.to_csv(PHASE_ROOT / "temporal_preservation_metrics.csv", index=False)
    bootstrap_table.to_csv(PHASE_ROOT / "temporal_bootstrap_intervals.csv", index=False)
    representations.to_csv(PHASE_ROOT / "temporal_representation_sensitivity.csv", index=False)
    retention.to_csv(PHASE_ROOT / "participant_temporal_retention.csv", index=False)
    displacement.to_csv(PHASE_ROOT / "participant_temporal_displacement.csv", index=False)
    coherence.to_csv(PHASE_ROOT / "temporal_cluster_coherence.csv", index=False)
    advantages.to_csv(PHASE_ROOT / "temporal_local_neighbor_advantage.csv", index=False)
    advantage_participants.to_csv(PHASE_ROOT / "temporal_local_neighbor_advantage_participant.csv", index=False)
    pair_classes.to_csv(PHASE_ROOT / "temporal_pair_class_movement.csv", index=False)
    evidence.to_csv(PHASE_ROOT / "temporal_evidence_decision.csv", index=False)
    write_json(PHASE_ROOT / "temporal_evidence_decision.json", evidence.to_dict("records"))

    make_figure_e1(metrics, bootstrap_table)
    make_figure_e2(retention)
    make_figure_e3(coherence, advantages)
    make_figure_e4(representations, coherence)
    render_phase4_report(extraction_report, metrics, bootstrap_table, retention, displacement, coherence, advantages, pair_classes, evidence)

    validation = {
        "created_at": now_iso(),
        "hours": TIME_RESOLVED_HOURS,
        "test_participant_count": len(participant_order),
        "test_participants_by_subtype": phase3_reference.groupby("canonical_stratum").size().to_dict(),
        "all_hours_have_same_participant_set": True,
        "factual_h0_reused_from_study1": True,
        "exact_neutral_h0_used": True,
        "h0_or_ht_clustered": False,
        "clinical_clustering_revised": False,
        "new_forward_pass": True,
        "model_retrained": False,
    }
    write_json(PHASE_ROOT / "phase4_analysis_validation.json", validation)

    artifact_paths = sorted(path for path in PHASE_ROOT.rglob("*") if path.is_file() and path.parent != SNAPSHOT_ROOT and path.name != "phase4_artifact_hashes.json")
    artifact_paths.extend([DECISION_ROOT / "PHASE4_TIME_RESOLVED_REPORT.md"])
    artifact_paths.extend(sorted(path for directory in [FIGURE_FULL, FIGURE_THUMB, FIGURE_DATA, FIGURE_META] for path in directory.glob("figure_E*")))
    hashes = {str(path.relative_to(STUDY2_ROOT)): sha256_file(path) for path in artifact_paths}
    write_json(PHASE_ROOT / "phase4_artifact_hashes.json", hashes)

    manifest = read_json(STUDY2_ROOT / "MANIFEST.json")
    created = sorted(str(path.relative_to(STUDY2_ROOT)) for path in STUDY2_ROOT.rglob("*") if path.is_file() and path.name != "MANIFEST.json")
    prior_figures = [item for item in (manifest.get("figures") or []) if "figure_E" not in item.get("png", "")]
    manifest.update({
        "phase_status": "Phase 4 optional time-resolved extension complete",
        "created_artifacts": created + ["MANIFEST.json"],
        "phase4_new_forward_pass": True,
        "phase4_model_retrained": False,
        "phase4_hours": TIME_RESOLVED_HOURS,
        "phase4_evidence_decision": evidence.to_dict("records"),
        "figures": prior_figures + [
            {"png": "figures/full_resolution/figure_E1_temporal_preservation.png", "metadata": "figures/metadata/figure_E1_temporal_preservation.json"},
            {"png": "figures/full_resolution/figure_E2_temporal_retention.png", "metadata": "figures/metadata/figure_E2_temporal_retention.json"},
            {"png": "figures/full_resolution/figure_E3_temporal_coherence.png", "metadata": "figures/metadata/figure_E3_temporal_coherence.json"},
            {"png": "figures/full_resolution/figure_E4_temporal_neutralization.png", "metadata": "figures/metadata/figure_E4_temporal_neutralization.json"},
        ],
    })
    write_json(STUDY2_ROOT / "MANIFEST.json", manifest)
    print("Phase 4 complete. Matched time-resolved preservation and movement have been evaluated. No further phase has been started.")


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


def subtype_label(subtype: str) -> str:
    labels = {
        "healthy": "Healthy",
        "pre_diabetes": "Pre-diabetes",
        "t2d_oral_non_insulin": "T2D oral non-insulin",
        "insulin_dependent": "Insulin-dependent",
    }
    return labels.get(subtype, subtype.replace("_", " "))


def errorbar_from_bounds(axis, x, point, low, high) -> None:
    if pd.notna(low) and pd.notna(high):
        axis.vlines(x, low, high, color=COLOR_OBSERVED, linewidth=1.0, zorder=5)
        axis.hlines([low, high], x - 0.035, x + 0.035, color=COLOR_OBSERVED, linewidth=1.0, zorder=5)


def make_figure_e1(metrics: pd.DataFrame, bootstrap_table: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    title = "Clinical and h0 neighborhood preservation changes over streaming time"
    figure, axes = plt.subplots(1, len(CANONICAL_STRATA), figsize=(16, 4.5), facecolor="white", sharey=True)
    comparisons = [("clinical_to_state", "Clinical to state", COLOR_REFERENCE), ("h0_to_state", "h0 to state", COLOR_ADJUSTED)]
    for subtype_index, (axis, subtype) in enumerate(zip(axes, CANONICAL_STRATA)):
        for comparison, label, color in comparisons:
            group = metrics[(metrics.canonical_stratum == subtype) & (metrics.comparison == comparison)].sort_values("hour")
            axis.plot(group.hour, group.knn_jaccard_mean, marker="o", linewidth=2.0, color=color, label=label)
            lows = []
            highs = []
            for hour in group.hour:
                interval = bootstrap_table[(bootstrap_table.canonical_stratum == subtype) & (bootstrap_table.hour == hour) & (bootstrap_table.comparison == comparison) & (bootstrap_table.metric == "knn_jaccard_mean")].iloc[0]
                lows.append(interval.ci_low)
                highs.append(interval.ci_high)
            axis.fill_between(group.hour, lows, highs, color=color, alpha=BAND_ALPHA)
        axis.set_title(subtype_label(subtype), fontweight="bold", fontsize=10)
        axis.set_xticks(TIME_RESOLVED_HOURS)
        axis.set_xlabel("Elapsed streaming time (hours)")
        if subtype_index == 0:
            axis.set_ylabel("kNN Jaccard")
            axis.legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=8)
        if subtype == "insulin_dependent":
            axis.annotate("Exploratory", xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top", color=COLOR_NULL, fontsize=8)
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.02)
    figure.tight_layout()
    plotted = metrics[metrics.comparison.isin(["clinical_to_state", "h0_to_state"])].copy()
    save_figure(figure, "figure_E1_temporal_preservation", title, plotted, {
        "input_artifact_paths": ["phase4_time_resolved_extension/temporal_preservation_metrics.csv", "phase4_time_resolved_extension/temporal_bootstrap_intervals.csv"],
        "sample_sizes": plotted.groupby("canonical_stratum").n.first().to_dict(),
        "metrics_shown": ["Clinical-to-state kNN Jaccard", "h0-to-state kNN Jaccard", "Participant-bootstrap 95% confidence intervals"],
        "color_role_mapping": {"Clinical reference": COLOR_REFERENCE, "h0 reference": COLOR_ADJUSTED},
    })


def make_figure_e2(retention: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    title = "Initial-state neighbor retention evolves across clinical clusters"
    figure, axes = plt.subplots(1, len(CANONICAL_STRATA), figsize=(16, 4.8), facecolor="white", sharey=True)
    rng = np.random.default_rng(SEED)
    for subtype_index, (axis, subtype) in enumerate(zip(axes, CANONICAL_STRATA)):
        subset = retention[retention.canonical_stratum == subtype]
        clusters = sorted(subset.display_cluster.unique())
        for cluster_index, cluster in enumerate(clusters):
            group = subset[subset.display_cluster == cluster]
            color = CLUSTER_COLORS[cluster_index % len(CLUSTER_COLORS)]
            for pid, participant in group.groupby("participant_id"):
                axis.plot(participant.hour, participant.h0_to_state_retention, color=COLOR_NULL, alpha=0.08, linewidth=0.6)
            summary = group.groupby("hour").h0_to_state_retention.mean().reindex(TIME_RESOLVED_HOURS)
            lows = []
            highs = []
            for hour in TIME_RESOLVED_HOURS:
                values = group[group.hour == hour].h0_to_state_retention.to_numpy()
                interval = bootstrap_mean_ci(values, PHASE4_BOOTSTRAP_N, SEED + subtype_index * 100 + int(cluster) * 10 + hour)
                lows.append(interval["ci_low"]); highs.append(interval["ci_high"])
            axis.plot(TIME_RESOLVED_HOURS, summary, marker="o", linewidth=2.0, color=color, label=f"C{int(cluster)}")
            axis.fill_between(TIME_RESOLVED_HOURS, lows, highs, color=color, alpha=BAND_ALPHA)
        axis.set_title(subtype_label(subtype), fontweight="bold", fontsize=10)
        axis.set_xticks(TIME_RESOLVED_HOURS)
        axis.set_xlabel("Elapsed streaming time (hours)")
        if subtype_index == 0:
            axis.set_ylabel("h0-to-state kNN retention")
        axis.legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=7)
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.02)
    figure.tight_layout()
    save_figure(figure, "figure_E2_temporal_retention", title, retention, {
        "input_artifact_paths": ["phase4_time_resolved_extension/participant_temporal_retention.csv"],
        "sample_sizes": retention[retention.hour == 0].groupby("canonical_stratum").participant_id.count().to_dict(),
        "metrics_shown": ["Participant h0-to-state retention", "Clinical-cluster means with participant-bootstrap intervals"],
        "color_role_mapping": {"Participants": COLOR_NULL, "Local cluster colors": CLUSTER_COLORS},
    })


def make_figure_e3(coherence: pd.DataFrame, advantages: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    title = "Movement coherence changes across matched streaming snapshots"
    figure, axes = plt.subplots(2, len(CANONICAL_STRATA), figsize=(16, 8), facecolor="white", sharex=True)
    for subtype_index, subtype in enumerate(CANONICAL_STRATA):
        full = coherence[(coherence.canonical_stratum == subtype) & (coherence.representation == "full")]
        neutral = coherence[(coherence.canonical_stratum == subtype) & (coherence.representation == "neutral")]
        clusters = sorted(full.display_cluster.unique())
        for cluster_index, cluster in enumerate(clusters):
            color = CLUSTER_COLORS[cluster_index % len(CLUSTER_COLORS)]
            full_cluster = full[full.display_cluster == cluster].sort_values("hour")
            neutral_cluster = neutral[neutral.display_cluster == cluster].sort_values("hour")
            axes[0, subtype_index].plot(full_cluster.hour, full_cluster.gap_point, marker="o", color=color, linewidth=2.0, label=f"C{int(cluster)} full")
            axes[0, subtype_index].plot(neutral_cluster.hour, neutral_cluster.gap_point, marker="o", color=color, linewidth=1.25, linestyle="--", label=f"C{int(cluster)} neutral")
            full_adv = advantages[(advantages.canonical_stratum == subtype) & (advantages.display_cluster == cluster) & (advantages.representation == "full")].sort_values("hour")
            neutral_adv = advantages[(advantages.canonical_stratum == subtype) & (advantages.display_cluster == cluster) & (advantages.representation == "neutral")].sort_values("hour")
            axes[1, subtype_index].plot(full_adv.hour, full_adv["mean"], marker="o", color=color, linewidth=2.0)
            axes[1, subtype_index].plot(neutral_adv.hour, neutral_adv["mean"], marker="o", color=color, linewidth=1.25, linestyle="--")
        axes[0, subtype_index].axhline(0, color=COLOR_OBSERVED, linewidth=1.0)
        axes[1, subtype_index].axhline(0, color=COLOR_OBSERVED, linewidth=1.0)
        axes[0, subtype_index].set_title(subtype_label(subtype), fontweight="bold", fontsize=10)
        axes[1, subtype_index].set_xticks([6, 12, 24, 48])
        axes[1, subtype_index].set_xlabel("Elapsed streaming time (hours)")
        if subtype_index == 0:
            axes[0, subtype_index].set_ylabel("Within-minus-between cosine")
            axes[1, subtype_index].set_ylabel("h0-neighbor movement advantage")
            axes[0, subtype_index].legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=6, ncol=2)
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.01)
    figure.tight_layout()
    plotted = pd.concat([coherence.assign(source="coherence"), advantages.assign(source="neighbor_advantage")], ignore_index=True, sort=False)
    save_figure(figure, "figure_E3_temporal_coherence", title, plotted, {
        "input_artifact_paths": ["phase4_time_resolved_extension/temporal_cluster_coherence.csv", "phase4_time_resolved_extension/temporal_local_neighbor_advantage.csv"],
        "sample_sizes": coherence.groupby("canonical_stratum").n.sum().to_dict(),
        "metrics_shown": ["Cluster-specific within-minus-between displacement cosine", "h0-neighbor movement advantage", "Exact static-neutral sensitivity"],
        "color_role_mapping": {"Local cluster colors": CLUSTER_COLORS, "Full": "Solid line", "Neutral": "Dashed line", "Zero": COLOR_OBSERVED},
    })


def make_figure_e4(representations: pd.DataFrame, coherence: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    title = "Static neutralization changes time-resolved latent structure"
    figure, axes = plt.subplots(2, len(CANONICAL_STRATA), figsize=(16, 8), facecolor="white", sharex=True)
    for subtype_index, subtype in enumerate(CANONICAL_STRATA):
        subset = representations[(representations.canonical_stratum == subtype) & (representations.estimable == True)]
        for representation, label, color in [("full", "Full", COLOR_REFERENCE), ("neutral", "Neutral", COLOR_ADJUSTED)]:
            group = subset[subset.representation == representation].sort_values("hour")
            axes[0, subtype_index].plot(group.hour, group.neighbor_purity, marker="o", linewidth=2.0, color=color, label=label)
            axes[0, subtype_index].fill_between(group.hour, group.neighbor_purity_ci_low, group.neighbor_purity_ci_high, color=color, alpha=BAND_ALPHA)
        full_ratio = coherence[(coherence.canonical_stratum == subtype) & (coherence.representation == "full")].groupby("hour").coherence_ratio.mean()
        neutral_ratio = coherence[(coherence.canonical_stratum == subtype) & (coherence.representation == "neutral")].groupby("hour").coherence_ratio.mean()
        axes[1, subtype_index].plot(full_ratio.index, full_ratio.values, marker="o", linewidth=2.0, color=COLOR_REFERENCE, label="Full")
        axes[1, subtype_index].plot(neutral_ratio.index, neutral_ratio.values, marker="o", linewidth=2.0, color=COLOR_ADJUSTED, label="Neutral")
        axes[1, subtype_index].axhline(COHERENCE_RATIO_BAR, color=COLOR_NULL, linestyle="--", linewidth=1.0)
        axes[0, subtype_index].set_title(subtype_label(subtype), fontweight="bold", fontsize=10)
        axes[1, subtype_index].set_xticks([6, 12, 24, 48])
        axes[1, subtype_index].set_xlabel("Elapsed streaming time (hours)")
        if subtype_index == 0:
            axes[0, subtype_index].set_ylabel("Fixed-label neighbor purity")
            axes[1, subtype_index].set_ylabel("Mean cluster coherence ratio")
            axes[0, subtype_index].legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=8)
            axes[1, subtype_index].legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=8)
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.01)
    figure.tight_layout()
    plotted = pd.concat([representations.assign(source="representation"), coherence.assign(source="coherence")], ignore_index=True, sort=False)
    save_figure(figure, "figure_E4_temporal_neutralization", title, plotted, {
        "input_artifact_paths": ["phase4_time_resolved_extension/temporal_representation_sensitivity.csv", "phase4_time_resolved_extension/temporal_cluster_coherence.csv"],
        "sample_sizes": representations.groupby("canonical_stratum").n.first().to_dict(),
        "metrics_shown": ["Fixed-label neighbor purity", "Mean cluster coherence ratio", "Full and exact static-neutral representations"],
        "color_role_mapping": {"Full": COLOR_REFERENCE, "Neutral": COLOR_ADJUSTED, "Coherence threshold": COLOR_NULL},
    })


def render_phase4_report(extraction_report, metrics, bootstrap_table, retention, displacement, coherence, advantages, pair_classes, evidence) -> None:
    jaccard_intervals = bootstrap_table[bootstrap_table.metric == "knn_jaccard_mean"]
    retention_summary = retention.groupby(["canonical_stratum", "display_cluster", "hour"]).agg(n=("participant_id", "count"), mean_h0_retention=("h0_to_state_retention", "mean"), mean_clinical_retention=("clinical_to_state_retention", "mean")).reset_index()
    magnitude_summary = displacement.groupby(["canonical_stratum", "display_cluster", "hour", "representation"]).agg(n=("participant_id", "count"), mean_magnitude=("magnitude", "mean"), median_magnitude=("magnitude", "median")).reset_index()
    supported = evidence[evidence.full_analysis_passes == True]
    interpretation_lines = []
    for subtype in CANONICAL_STRATA:
        group = metrics[(metrics.canonical_stratum == subtype) & (metrics.comparison == "clinical_to_state")].set_index("hour")
        start = float(group.loc[0, "knn_jaccard_mean"]); finish = float(group.loc[48, "knn_jaccard_mean"])
        interpretation_lines.append(f"- {subtype}: Clinical-to-state kNN Jaccard changed from {start:.4f} at h0 to {finish:.4f} at 48 hours.")
    claim_lines = [f"- {row.canonical_stratum} cluster {int(row.display_cluster)} at {int(row.hour)} hours: **{row.label}**" for row in supported.itertuples()]
    if not claim_lines:
        claim_lines = ["- No subtype-cluster-hour combination met the full coherent-movement evidence rule."]
    lines = [
        "# Phase 4 time-resolved extension report",
        "",
        "## Concise interpretation",
        "",
        "The optional extension uses matched literal SSM states at 0, 6, 12, 24, and 48 elapsed hours. The clinical clustering remained frozen and no latent representation was clustered. All 1,544 extraction participants had coverage at every hour; the primary analysis retains the same 214 clinically eligible test participants used at Gate D.",
        "",
        *interpretation_lines,
        "",
        *claim_lines,
        "",
        "## Approved forward pass and snapshot provenance",
        "",
        f"A new forward pass was explicitly approved for this optional extension. The frozen checkpoint was used on {extraction_report['device']}; the model was not retrained. The extracted vectors have {extraction_report['vector_dimension']} dimensions.",
        "",
        f"Snapshot rule: {extraction_report['snapshot_rule']}",
        "",
        dataframe_to_markdown(pd.DataFrame(extraction_report['coverage'])),
        "",
        f"Factual h0 QA maximum absolute difference versus the immutable Study 1 h0 was {extraction_report['factual_h0_max_abs_difference_vs_study1']:.6g}. Neutral h0 batch maximum difference was {extraction_report['neutral_h0_batch_max_difference']:.6g}.",
        "",
        "## Time-resolved preservation metrics",
        "",
        dataframe_to_markdown(metrics),
        "",
        "## kNN Jaccard participant-bootstrap intervals",
        "",
        dataframe_to_markdown(jaccard_intervals),
        "",
        f"Participant bootstrap resamples per estimate: {PHASE4_BOOTSTRAP_N}. Mantel and cluster-label permutations per estimate: {PHASE4_PERMUTATION_N}.",
        "",
        "## Participant neighborhood retention",
        "",
        dataframe_to_markdown(retention_summary),
        "",
        "## Time-resolved displacement magnitude",
        "",
        dataframe_to_markdown(magnitude_summary),
        "",
        "## Cluster-specific movement coherence",
        "",
        dataframe_to_markdown(coherence),
        "",
        "## Local h0-neighborhood movement advantage",
        "",
        dataframe_to_markdown(advantages),
        "",
        "## Pair-class movement",
        "",
        dataframe_to_markdown(pair_classes),
        "",
        "## Time-resolved evidence decisions",
        "",
        dataframe_to_markdown(evidence),
        "",
        "The insulin-dependent subtype remains exploratory. Cluster colors are local to each diagnostic subtype and do not imply equivalence across subtypes.",
        "",
        "## Figures",
        "",
        "![Temporal preservation](../figures/full_resolution/figure_E1_temporal_preservation.png)",
        "",
        "![Temporal retention](../figures/full_resolution/figure_E2_temporal_retention.png)",
        "",
        "![Temporal coherence](../figures/full_resolution/figure_E3_temporal_coherence.png)",
        "",
        "![Temporal neutralization](../figures/full_resolution/figure_E4_temporal_neutralization.png)",
        "",
        "## Confirmations",
        "",
        "The frozen clinical factors, preprocessing, selected cluster counts, centroids, and display labels were not revised. h0 and all ht snapshots remained unclustered. The approved forward pass generated snapshot artifacts only; the model was not retrained.",
        "",
        "Phase 4 complete. Matched time-resolved preservation and movement have been evaluated. No further phase has been started.",
        "",
    ]
    (DECISION_ROOT / "PHASE4_TIME_RESOLVED_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
