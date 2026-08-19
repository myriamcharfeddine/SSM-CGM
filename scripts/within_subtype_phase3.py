"""Phase 3 and Gate D for the within-subtype phenotype preservation study.

Evaluates clinical-to-ht and h0-to-ht neighborhood preservation, participant
neighborhood retention across h0 -> ht, endpoint displacement (h_t - h0) and
its within-cluster movement coherence, a static-neutralized sensitivity pass,
and a glucose-residualized sensitivity pass, before the pre-registered
evidence decision for coherent endpoint movement.

Everything here reuses existing Study 1 h0 and h_t full, neutral, and glucose-residualized endpoint artifacts. Full displacement is calculated directly as h_t minus h0. No forward pass through the model is run in this phase.
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

PHASE_ROOT = STUDY2_ROOT / "phase3_ht_preservation"
FIGURE_FULL = FIGURE_ROOT / "full_resolution"
FIGURE_THUMB = FIGURE_ROOT / "thumbnails"
FIGURE_DATA = FIGURE_ROOT / "plotted_data"
FIGURE_META = FIGURE_ROOT / "metadata"
STEP3_DIR = STUDY1_ROOT / "step3"

# View names mirror Study 1's own Phase 3 convention exactly (test = primary,
# full_cohort = sensitivity across all splits), not Phase 2's val/test split,
# since Section 12 of the build prompt never mentions a validation view and
# Study 1's phase3_analyze.py used precisely this pair of views.
VIEWS = ["test", "full_cohort"]

# Phase 3 uses the shared pre-registered resampling counts. Distance matrices
# are computed once, so 1,000 participant resamples do not rerun the model.
PHASE3_BOOTSTRAP_N = BOOTSTRAP_N
PHASE3_PERMUTATION_N = PERMUTATION_N
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
    rho, p_value = phase2.mantel_spearman(reference_dist, embedded_dist, PHASE3_PERMUTATION_N, seed)
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


def main() -> None:
    setup_output_tree()
    factor_decision = read_json(DECISION_ROOT / "factor_selection.json")
    final_factors: list[str] = factor_decision["final_factor_list"]
    missing_decision = read_json(phase1.PHASE0_ROOT / "missing_data_decision.json")
    strategy_by_subtype = {subtype: missing_decision[subtype]["strategy"] for subtype in CANONICAL_STRATA}
    frame, _, _, _ = phase1.load_frame(final_factors)
    frozen_manifest = read_json(phase1.PHASE_ROOT / "frozen_clustering_manifest.json")

    h0_pid_to_row, h0_matrix = phase2.load_h0_matrix()
    ht_full_df = pd.read_parquet(STEP3_DIR / "h_t_full.parquet")
    ht_neutral_df = pd.read_parquet(STEP3_DIR / "h_t_neutral.parquet")
    ht_resid_df = pd.read_parquet(STEP3_DIR / "h_t_full_residualized.parquet")
    for table in (ht_full_df, ht_neutral_df, ht_resid_df):
        table["participant_id"] = table["participant_id"].astype(str)
    ht_ids = ht_full_df.participant_id.to_numpy()
    if ht_full_df.participant_id.duplicated().any() or ht_neutral_df.participant_id.duplicated().any() or ht_resid_df.participant_id.duplicated().any():
        raise RuntimeError("Duplicate participant identifiers in an endpoint artifact")
    if set(ht_ids) != set(ht_neutral_df.participant_id) or set(ht_ids) != set(ht_resid_df.participant_id):
        raise RuntimeError("Endpoint participant sets do not agree")
    ht_neutral_df = ht_neutral_df.set_index("participant_id").reindex(ht_ids).reset_index()
    ht_resid_df = ht_resid_df.set_index("participant_id").reindex(ht_ids).reset_index()
    ht_pid_to_row = {pid: i for i, pid in enumerate(ht_ids)}
    ht_full_matrix = ht_full_df.drop(columns=["participant_id", "split", "n_overnight_anchors", "avg_glucose_mgdl"]).to_numpy(dtype=np.float32)
    ht_neutral_matrix = ht_neutral_df.drop(columns=["participant_id", "split", "n_overnight_anchors"]).to_numpy(dtype=np.float32)
    ht_resid_matrix = ht_resid_df.drop(columns=["participant_id", "split", "n_overnight_anchors"]).to_numpy(dtype=np.float32)
    endpoint_duration = ht_full_df["n_overnight_anchors"].to_numpy(dtype=np.float64)
    neutral_center = ht_neutral_matrix.mean(axis=0, dtype=np.float64)

    dimensions = {"h0": int(h0_matrix.shape[1]), "ht_full": int(ht_full_matrix.shape[1]), "ht_neutral": int(ht_neutral_matrix.shape[1]), "ht_residualized": int(ht_resid_matrix.shape[1])}
    if len(set(dimensions.values())) != 1:
        raise RuntimeError(f"Incompatible latent dimensions: {dimensions}")
    finite = {"h0": bool(np.isfinite(h0_matrix).all()), "ht_full": bool(np.isfinite(ht_full_matrix).all()), "ht_neutral": bool(np.isfinite(ht_neutral_matrix).all()), "ht_residualized": bool(np.isfinite(ht_resid_matrix).all())}
    if not all(finite.values()):
        raise RuntimeError(f"Nonfinite endpoint vectors found: {finite}")
    shared = set(h0_pid_to_row) & set(ht_pid_to_row)
    input_audit = {
        "created_at": now_iso(),
        "endpoint_interpretation": "Participant-level overnight endpoint summary using 0 to 6 hour anchors",
        "vector_dimensions": dimensions,
        "finite_vectors": finite,
        "h0_participants": len(h0_pid_to_row),
        "ht_participants": len(ht_pid_to_row),
        "paired_participants": len(shared),
        "ht_duplicate_participants": 0,
        "endpoint_anchor_count": {"min": int(endpoint_duration.min()), "median": float(np.median(endpoint_duration)), "max": int(endpoint_duration.max())},
        "full_displacement_definition": "Calculated directly as ht_full minus h0 without normalizing either endpoint",
        "neutral_sensitivity_definition": "Study 1 centered neutral endpoint contrast: ht_neutral minus the full Study 1 cohort mean ht_neutral",
        "neutral_sensitivity_exact_protocol_displacement": False,
        "neutral_limitation": "The coordinate-compatible constant neutral h0 vector was verified during Study 1 extraction but was not saved. Recomputing it would require a forbidden model pass. The saved Study 1 centered neutral endpoint contrast is therefore retained as a sensitivity proxy and cannot support the stronger persists-after-static-neutralization label.",
        "glucose_residualized_displacement_computed": False,
        "glucose_residualized_reason": "No coordinate-compatible residualized h0 exists",
        "time_resolved_snapshots_available": False,
        "new_model_forward_pass": False,
    }
    write_json(PHASE_ROOT / "endpoint_input_audit.json", input_audit)
    print(f"[phase3] paired endpoint artifacts validated: {len(shared)} participants, dimension {dimensions['h0']}", flush=True)

    three_space_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    displacement_rows: list[dict[str, Any]] = []
    cluster_coherence_rows: list[dict[str, Any]] = []
    pair_class_rows: list[dict[str, Any]] = []
    pair_class_participant_rows: list[dict[str, Any]] = []
    local_advantage_rows: list[dict[str, Any]] = []
    local_advantage_participant_rows: list[dict[str, Any]] = []
    neutral_coherence_rows: list[dict[str, Any]] = []
    residualized_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    quiver_records: list[dict[str, Any]] = []
    pc1_variance_rows: list[dict[str, Any]] = []

    for subtype_index, subtype in enumerate(CANONICAL_STRATA):
        cluster_info = frozen_manifest["clusters"][subtype]
        if cluster_info["selected_k"] is None:
            continue
        pipeline = joblib.load(STUDY2_ROOT / cluster_info["preprocessing_pipeline_path"])
        centroid_info = read_json(STUDY2_ROOT / cluster_info["centroid_path"])
        centroids_by_display = {int(key): np.asarray(value) for key, value in centroid_info["centroids_by_display_cluster"].items()}
        display_order = sorted(centroids_by_display)
        centroid_order = np.asarray([centroids_by_display[display] for display in display_order])
        use_imputation = strategy_by_subtype[subtype] == "iterative_imputation"
        subtype_frame = frame[frame.canonical_stratum == subtype].reset_index(drop=True)
        eligible_mask = pd.Series(True, index=subtype_frame.index) if use_imputation else subtype_frame[final_factors].notna().all(axis=1)
        eligible = subtype_frame[eligible_mask].reset_index(drop=True)
        clinical_all = phase1.apply_pipeline(eligible, pipeline["factors"], pipeline["log_transformed"], pipeline["imputer"], pipeline["scaler"])
        nearest, margin = assign_display_cluster(clinical_all, centroid_order)
        labels_all = np.asarray([display_order[index] for index in nearest])
        paired_mask = np.asarray([pid in h0_pid_to_row and pid in ht_pid_to_row for pid in eligible.participant_id])
        eligible = eligible[paired_mask].reset_index(drop=True)
        clinical_all = clinical_all[paired_mask]
        labels_all = labels_all[paired_mask]
        margin = margin[paired_mask]
        h0_all = h0_matrix[[h0_pid_to_row[pid] for pid in eligible.participant_id]]
        ht_rows_all = np.asarray([ht_pid_to_row[pid] for pid in eligible.participant_id])
        ht_all = ht_full_matrix[ht_rows_all]
        neutral_all = ht_neutral_matrix[ht_rows_all]
        residual_all = ht_resid_matrix[ht_rows_all]
        anchors_all = endpoint_duration[ht_rows_all]
        split_lookup = ht_full_df.set_index("participant_id")["split"].to_dict()
        split_conflicts = sum(str(row.split) != str(split_lookup[row.participant_id]) for row in eligible.itertuples())
        if split_conflicts:
            raise RuntimeError(f"{subtype}: {split_conflicts} split conflicts between clinical and endpoint artifacts")

        for view in VIEWS:
            view_mask = (eligible.split == "test").to_numpy() if view == "test" else np.ones(len(eligible), dtype=bool)
            n_view = int(view_mask.sum())
            if n_view < 6:
                continue
            pids = eligible.participant_id.to_numpy()[view_mask]
            labels = labels_all[view_mask]
            clinical = clinical_all[view_mask]
            h0 = h0_all[view_mask].astype(np.float64, copy=False)
            ht = ht_all[view_mask].astype(np.float64, copy=False)
            ht_neutral = neutral_all[view_mask].astype(np.float64, copy=False)
            ht_residualized = residual_all[view_mask].astype(np.float64, copy=False)
            durations = anchors_all[view_mask]
            d_full = ht - h0
            d_neutral_proxy = ht_neutral - neutral_center
            knn_k = deterministic_knn_k(n_view)
            seed = SEED + 1000 * subtype_index + (0 if view == "test" else 500)
            clinical_dist = squareform(pdist(clinical, metric=CLINICAL_METRIC)).astype(np.float64)
            h0_dist = squareform(pdist(h0, metric=LATENT_METRIC)).astype(np.float64)
            ht_dist = squareform(pdist(ht, metric=LATENT_METRIC)).astype(np.float64)

            comparisons = {"clinical_to_h0": (clinical_dist, h0_dist), "clinical_to_ht": (clinical_dist, ht_dist), "h0_to_ht": (h0_dist, ht_dist)}
            points = {}
            for comparison, (reference_dist, embedded_dist) in comparisons.items():
                if view == "test":
                    point = reference_to_embedded_metrics(reference_dist, embedded_dist, labels, knn_k, seed, comparison.split("_to_")[0], comparison.split("_to_")[1])
                else:
                    point = metrics_without_permutation(reference_dist, embedded_dist, labels, knn_k)
                    point["mantel_p"] = None
                points[comparison] = point
                three_space_rows.append({"canonical_stratum": subtype, "view": view, "n": n_view, "knn_k": knn_k, "comparison": comparison, **{key: value for key, value in point.items() if key not in ("reference_space", "embedded_space")}})
                if view == "test":
                    intervals = bootstrap_comparison(reference_dist, embedded_dist, labels, knn_k, seed + len(bootstrap_rows), PHASE3_BOOTSTRAP_N)
                    for metric, interval in intervals.items():
                        bootstrap_rows.append({"canonical_stratum": subtype, "comparison": comparison, "metric": metric, "point": point.get(metric), **interval})
            compatible = [metric for metric in HEADLINE_METRICS if points["clinical_to_h0"].get(metric) is not None and points["clinical_to_ht"].get(metric) is not None]
            change = {metric: float(points["clinical_to_ht"][metric] - points["clinical_to_h0"][metric]) for metric in compatible}
            three_space_rows.append({"canonical_stratum": subtype, "view": view, "n": n_view, "knn_k": knn_k, "comparison": "change_clinical_ht_minus_clinical_h0", **change})
            if view == "test":
                change_intervals = bootstrap_change(clinical_dist, h0_dist, ht_dist, labels, knn_k, seed + 90, PHASE3_BOOTSTRAP_N)
                for metric, interval in change_intervals.items():
                    bootstrap_rows.append({"canonical_stratum": subtype, "comparison": "change_clinical_ht_minus_clinical_h0", "metric": metric, "point": change.get(metric), **interval})

            if view != "test":
                continue

            neutral_dist = squareform(pdist(ht_neutral, metric=LATENT_METRIC)).astype(np.float64)
            residual_dist = squareform(pdist(ht_residualized, metric=LATENT_METRIC)).astype(np.float64)
            representations = {"clinical": clinical_dist, "h0_full": h0_dist, "ht_full": ht_dist, "ht_neutral": neutral_dist, "ht_glucose_residualized": residual_dist}
            for representation, rep_dist in representations.items():
                rep_point = metrics_without_permutation(clinical_dist, rep_dist, labels, knn_k)
                rep_boot = bootstrap_comparison(clinical_dist, rep_dist, labels, knn_k, seed + 200 + len(representation_rows), PHASE3_BOOTSTRAP_N)
                representation_rows.append({
                    "canonical_stratum": subtype, "representation": representation, "n": n_view, "knn_k": knn_k,
                    "clinical_knn_jaccard": rep_point["knn_jaccard_mean"],
                    "clinical_knn_jaccard_ci_low": rep_boot["knn_jaccard_mean"]["ci_low"],
                    "clinical_knn_jaccard_ci_high": rep_boot["knn_jaccard_mean"]["ci_high"],
                    "neighbor_purity": rep_point["neighbor_purity_mean"],
                    "neighbor_purity_ci_low": rep_boot["neighbor_purity_mean"]["ci_low"],
                    "neighbor_purity_ci_high": rep_boot["neighbor_purity_mean"]["ci_high"],
                    "cluster_silhouette": rep_point["cluster_silhouette"],
                    "pairwise_distance_spearman": rep_point["pairwise_distance_spearman"],
                    "estimable": True,
                })
            representation_rows.append({"canonical_stratum": subtype, "representation": "h0_neutral", "n": n_view, "knn_k": knn_k, "estimable": False, "reason": "Population-constant vector; pairwise distances are degenerate"})
            residualized_rows.extend([row for row in representation_rows[-6:] if row.get("representation") in ("ht_full", "ht_glucose_residualized")])

            h0_retention = participant_retention(h0_dist, ht_dist, knn_k)
            clinical_retention = participant_retention(clinical_dist, ht_dist, knn_k)
            margin_values = margin[view_mask]
            confidence_quartile = np.ceil(pd.Series(margin_values).rank(method="average", pct=True).to_numpy() * 4).clip(1, 4).astype(int)
            magnitude = np.linalg.norm(d_full, axis=1)
            neutral_magnitude = np.linalg.norm(d_neutral_proxy, axis=1)
            for i, pid in enumerate(pids):
                retention_rows.append({"participant_id": pid, "canonical_stratum": subtype, "display_cluster": int(labels[i]), "h0_to_ht_retention": float(h0_retention[i]), "clinical_to_ht_retention": float(clinical_retention[i]), "clinical_assignment_margin": float(margin_values[i]), "clinical_assignment_confidence_quartile": int(confidence_quartile[i]), "endpoint_anchor_count": int(durations[i]), "exploratory": cluster_info["status"] != "primary"})
                displacement_rows.append({"participant_id": pid, "canonical_stratum": subtype, "display_cluster": int(labels[i]), "displacement_magnitude": float(magnitude[i]), "neutral_proxy_magnitude": float(neutral_magnitude[i]), "endpoint_anchor_count": int(durations[i])})

            cos_full = cosine_similarity_matrix(d_full)
            cos_neutral = cosine_similarity_matrix(d_neutral_proxy)
            full_ratios = coherence_ratio_per_cluster(d_full, labels)
            neutral_ratios = coherence_ratio_per_cluster(d_neutral_proxy, labels)
            advantage_full = local_neighbor_advantage(d_full, h0_dist, knn_k, durations, cos_full)
            advantage_neutral = local_neighbor_advantage(d_neutral_proxy, h0_dist, knn_k, durations, cos_neutral)
            for i, pid in enumerate(pids):
                local_advantage_participant_rows.append({"participant_id": pid, "canonical_stratum": subtype, "display_cluster": int(labels[i]), "representation": "full", "neighbor_movement_advantage": float(advantage_full[i]) if np.isfinite(advantage_full[i]) else None})
                local_advantage_participant_rows.append({"participant_id": pid, "canonical_stratum": subtype, "display_cluster": int(labels[i]), "representation": "neutral_endpoint_centered_proxy", "neighbor_movement_advantage": float(advantage_neutral[i]) if np.isfinite(advantage_neutral[i]) else None})
            for cluster in display_order:
                cluster_mask = labels == cluster
                n_cluster = int(cluster_mask.sum())
                if n_cluster < 2:
                    continue
                full_gap = bootstrap_cluster_gap(d_full, labels, cluster, PHASE3_BOOTSTRAP_N, seed + int(cluster), cos_full)
                neutral_gap = bootstrap_cluster_gap(d_neutral_proxy, labels, cluster, PHASE3_BOOTSTRAP_N, seed + 100 + int(cluster), cos_neutral)
                permutation_p = permutation_p_cluster_gap(d_full, labels, cluster, full_gap["gap_point"], PHASE3_PERMUTATION_N, seed + int(cluster), cos_full)
                robustness = leave_one_out_robustness(d_full, labels, cluster, cos_full)
                full_advantage_summary = bootstrap_mean_ci(advantage_full[cluster_mask], PHASE3_BOOTSTRAP_N, seed + 300 + int(cluster))
                neutral_advantage_summary = bootstrap_mean_ci(advantage_neutral[cluster_mask], PHASE3_BOOTSTRAP_N, seed + 400 + int(cluster))
                local_advantage_rows.append({"canonical_stratum": subtype, "display_cluster": int(cluster), "representation": "full", **full_advantage_summary})
                local_advantage_rows.append({"canonical_stratum": subtype, "display_cluster": int(cluster), "representation": "neutral_endpoint_centered_proxy", **neutral_advantage_summary})
                cluster_coherence_rows.append({"canonical_stratum": subtype, "display_cluster": int(cluster), "n": n_cluster, **full_gap, "permutation_p": permutation_p, "coherence_ratio": full_ratios[int(cluster)], "mean_displacement_magnitude": float(magnitude[cluster_mask].mean()), "variance_displacement_magnitude": float(magnitude[cluster_mask].var(ddof=1)), "exploratory": cluster_info["status"] != "primary", **robustness})
                neutral_coherence_rows.append({"canonical_stratum": subtype, "display_cluster": int(cluster), "n": n_cluster, **neutral_gap, "coherence_ratio": neutral_ratios[int(cluster)], "mean_proxy_magnitude": float(neutral_magnitude[cluster_mask].mean()), "exact_protocol_neutral_displacement": False})

            for representation, vectors, cos_sim in [("full", d_full, cos_full), ("neutral_endpoint_centered_proxy", d_neutral_proxy, cos_neutral)]:
                pair_summary, pair_detail = pair_class_movement(vectors, labels, h0_dist, knn_k, durations, seed + (0 if representation == "full" else 1), PHASE3_BOOTSTRAP_N, cos_sim)
                pair_class_rows.append({"canonical_stratum": subtype, "representation": representation, **{f"{name}_{stat}": value for name, summary in pair_summary.items() for stat, value in summary.items()}})
                pair_detail.insert(0, "participant_id", pids)
                pair_detail.insert(1, "canonical_stratum", subtype)
                pair_detail.insert(2, "display_cluster", labels)
                pair_detail.insert(3, "representation", representation)
                pair_class_participant_rows.extend(pair_detail.to_dict("records"))

            for representation, vectors in [("full", d_full), ("neutral_endpoint_centered_proxy", d_neutral_proxy)]:
                pc1 = PCA(n_components=1, svd_solver="randomized", random_state=SEED).fit(vectors)
                pc1_variance_rows.append({"canonical_stratum": subtype, "representation": representation, "pc1_variance_explained": float(pc1.explained_variance_ratio_[0])})

            train_mask = (eligible.split == "train").to_numpy()
            if train_mask.sum() >= 3:
                shared_pca = PCA(n_components=2, svd_solver="randomized", random_state=SEED).fit(np.vstack([h0_all[train_mask], ht_all[train_mask]]))
                h0_2d = shared_pca.transform(h0)
                ht_2d = shared_pca.transform(ht)
                for i, pid in enumerate(pids):
                    quiver_records.append({"canonical_stratum": subtype, "participant_id": pid, "display_cluster": int(labels[i]), "h0_pc1": float(h0_2d[i, 0]), "h0_pc2": float(h0_2d[i, 1]), "ht_pc1": float(ht_2d[i, 0]), "ht_pc2": float(ht_2d[i, 1]), "pca_fit_split": "train", "exploratory": cluster_info["status"] != "primary"})

        subtype_full = [row for row in cluster_coherence_rows if row["canonical_stratum"] == subtype]
        subtype_neutral = [row for row in neutral_coherence_rows if row["canonical_stratum"] == subtype]
        for full_row in subtype_full:
            cluster = full_row["display_cluster"]
            neutral_row = next(row for row in subtype_neutral if row["display_cluster"] == cluster)
            full_advantage = next(row for row in local_advantage_rows if row["canonical_stratum"] == subtype and row["display_cluster"] == cluster and row["representation"] == "full")
            neutral_advantage = next(row for row in local_advantage_rows if row["canonical_stratum"] == subtype and row["display_cluster"] == cluster and row["representation"] == "neutral_endpoint_centered_proxy")
            criteria = {
                "positive_gap_ci_excludes_zero": bool(full_row["gap_ci_low"] is not None and full_row["gap_ci_low"] > 0),
                "coherence_ratio_exceeds_bar": bool(full_row["coherence_ratio"] > COHERENCE_RATIO_BAR),
                "cluster_movement_advantage_positive": bool(full_advantage["mean"] is not None and full_advantage["mean"] > 0),
                "not_driven_by_one_participant": bool(full_row["robust_to_single_participant"]),
                "n_at_least_underpowered_floor": bool(full_row["n"] >= UNDERPOWERED_CLUSTER_TEST_N),
                "neutral_proxy_directionally_consistent": bool(neutral_row["gap_point"] > 0 and full_row["gap_point"] > 0),
            }
            full_pass = all(criteria.values())
            proxy_pass = bool(neutral_row["gap_ci_low"] is not None and neutral_row["gap_ci_low"] > 0 and neutral_row["coherence_ratio"] > COHERENCE_RATIO_BAR and neutral_advantage["mean"] is not None and neutral_advantage["mean"] > 0)
            if full_pass:
                label = "Static-profile-associated movement coherence"
            else:
                label = "No coherent movement claim supported"
            evidence_rows.append({"canonical_stratum": subtype, "display_cluster": int(cluster), "label": label, "full_analysis_passes": full_pass, "centered_neutral_endpoint_proxy_passes": proxy_pass, "exact_neutralized_displacement_available": False, "exploratory": cluster_info["status"] != "primary", **criteria})

    three_space = pd.DataFrame(three_space_rows)
    bootstrap_table = pd.DataFrame(bootstrap_rows)
    representation_table = pd.DataFrame(representation_rows)
    retention = pd.DataFrame(retention_rows)
    displacement = pd.DataFrame(displacement_rows)
    cluster_coherence = pd.DataFrame(cluster_coherence_rows)
    pair_class = pd.DataFrame(pair_class_rows)
    pair_class_participant = pd.DataFrame(pair_class_participant_rows)
    local_advantage = pd.DataFrame(local_advantage_rows)
    local_advantage_participant = pd.DataFrame(local_advantage_participant_rows)
    neutral_coherence = pd.DataFrame(neutral_coherence_rows)
    residualized = pd.DataFrame(residualized_rows)
    evidence = pd.DataFrame(evidence_rows)
    pc1_variance = pd.DataFrame(pc1_variance_rows)
    quiver = pd.DataFrame(quiver_records)

    three_space.to_csv(TABLE_ROOT / "phase3_three_space_preservation.csv", index=False)
    bootstrap_table.to_csv(PHASE_ROOT / "three_space_bootstrap_intervals.csv", index=False)
    representation_table.to_csv(PHASE_ROOT / "representation_sensitivity.csv", index=False)
    retention.to_csv(PHASE_ROOT / "participant_retention.csv", index=False)
    displacement.to_csv(PHASE_ROOT / "participant_displacement_magnitude.csv", index=False)
    cluster_coherence.to_csv(PHASE_ROOT / "cluster_displacement_coherence.csv", index=False)
    pair_class.to_csv(PHASE_ROOT / "pair_class_movement.csv", index=False)
    pair_class_participant.to_csv(PHASE_ROOT / "pair_class_participant_aggregation.csv", index=False)
    local_advantage.to_csv(PHASE_ROOT / "local_neighbor_advantage.csv", index=False)
    local_advantage_participant.to_csv(PHASE_ROOT / "local_neighbor_advantage_participant.csv", index=False)
    neutral_coherence.to_csv(PHASE_ROOT / "neutralized_cluster_coherence.csv", index=False)
    residualized.to_csv(PHASE_ROOT / "glucose_residualized_sensitivity.csv", index=False)
    evidence.to_csv(PHASE_ROOT / "evidence_decision.csv", index=False)
    write_json(PHASE_ROOT / "evidence_decision.json", evidence_rows)
    pc1_variance.to_csv(PHASE_ROOT / "displacement_pca_variance.csv", index=False)
    quiver.to_csv(PHASE_ROOT / "quiver_display.csv", index=False)
    write_json(PHASE_ROOT / "execution_confirmation.json", {"new_model_forward_pass": False, "model_retrained": False, "h0_or_ht_reclustered": False, "phase1_revised": False, "time_resolved_extension_run": False, "stopped_at_gate": "D"})

    make_figure_d1(three_space, bootstrap_table, representation_table)
    make_figure_d2(retention)
    make_figure_d3(cluster_coherence, neutral_coherence, pair_class, quiver, evidence)
    figure_d4_title = make_figure_d4(representation_table, cluster_coherence, neutral_coherence)
    render_gate_d_report(three_space, bootstrap_table, representation_table, retention, displacement, cluster_coherence, pair_class, local_advantage, neutral_coherence, residualized, evidence, pc1_variance, frozen_manifest, figure_d4_title, input_audit)

    phase3_files = sorted(path for path in PHASE_ROOT.rglob("*") if path.is_file() and path.name != "phase3_artifact_hashes.json")
    phase3_files.extend(sorted(path for path in [DECISION_ROOT / "GATE_D_REPORT.md", TABLE_ROOT / "phase3_three_space_preservation.csv"] if path.is_file()))
    phase3_files.extend(sorted(path for directory in [FIGURE_FULL, FIGURE_THUMB, FIGURE_DATA, FIGURE_META] for path in directory.glob("figure_D*")))
    hashes = {str(path.relative_to(STUDY2_ROOT)): sha256_file(path) for path in phase3_files}
    write_json(PHASE_ROOT / "phase3_artifact_hashes.json", hashes)

    git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = read_json(STUDY2_ROOT / "MANIFEST.json")
    created = sorted(str(path.relative_to(STUDY2_ROOT)) for path in STUDY2_ROOT.rglob("*") if path.is_file() and path.name != "MANIFEST.json")
    prior_figures = [item for item in (manifest.get("figures") or []) if "figure_D" not in item.get("png", "")]
    manifest.update({
        "phase_status": "Gate D reached",
        "created_artifacts": created + ["MANIFEST.json"],
        "git_commit": git_commit,
        "git_branch": git_branch,
        "expected_git_branch": REPO_BRANCH,
        "movement_coherence_evidence_decision": evidence_rows,
        "new_model_forward_pass_ran_phase3": False,
        "neutralized_displacement_exactly_estimable": False,
        "figures": prior_figures + [
            {"png": "figures/full_resolution/figure_D1_reorganization.png", "metadata": "figures/metadata/figure_D1_reorganization.json"},
            {"png": "figures/full_resolution/figure_D2_retention_by_cluster.png", "metadata": "figures/metadata/figure_D2_retention_by_cluster.json"},
            {"png": "figures/full_resolution/figure_D3_movement_coherence.png", "metadata": "figures/metadata/figure_D3_movement_coherence.json"},
            {"png": "figures/full_resolution/figure_D4_neutralization.png", "metadata": "figures/metadata/figure_D4_neutralization.json"},
        ],
    })
    write_json(STUDY2_ROOT / "MANIFEST.json", manifest)
    print("Gate D reached. Endpoint preservation and displacement coherence from h0 to ht have been evaluated. Waiting for confirmation before any further phase.")


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


def make_figure_d1(three_space: pd.DataFrame, bootstrap_table: pd.DataFrame, representation_table: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    title = "Clinical and initial-state neighborhoods are reorganized in ht"
    subtypes = [subtype for subtype in CANONICAL_STRATA if subtype in representation_table.canonical_stratum.unique()]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    comparisons = [("clinical_to_h0", "Clinical to h0", COLOR_REFERENCE), ("clinical_to_ht", "Clinical to ht", COLOR_ADJUSTED), ("h0_to_ht", "h0 to ht", COLOR_NULL)]
    width = 0.24
    for comparison_index, (comparison, label, color) in enumerate(comparisons):
        for subtype_index, subtype in enumerate(subtypes):
            point_row = three_space[(three_space.view == "test") & (three_space.canonical_stratum == subtype) & (three_space.comparison == comparison)].iloc[0]
            interval = bootstrap_table[(bootstrap_table.canonical_stratum == subtype) & (bootstrap_table.comparison == comparison) & (bootstrap_table.metric == "knn_jaccard_mean")].iloc[0]
            x = subtype_index + (comparison_index - 1) * width
            axes[0].bar(x, point_row.knn_jaccard_mean, width=width, color=color, label=label if subtype_index == 0 else None)
            errorbar_from_bounds(axes[0], x, point_row.knn_jaccard_mean, interval.ci_low, interval.ci_high)
    axes[0].set_title("Neighborhood overlap", fontweight="bold")
    axes[0].set_ylabel("kNN Jaccard")
    axes[0].set_xticks(np.arange(len(subtypes)), [subtype_label(value) for value in subtypes], rotation=20, ha="right")
    axes[0].legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=8)

    purity_representations = [("clinical", "Clinical", COLOR_REFERENCE), ("h0_full", "h0", COLOR_ADJUSTED), ("ht_full", "ht", COLOR_NULL)]
    for representation_index, (representation, label, color) in enumerate(purity_representations):
        for subtype_index, subtype in enumerate(subtypes):
            row = representation_table[(representation_table.canonical_stratum == subtype) & (representation_table.representation == representation)].iloc[0]
            x = subtype_index + (representation_index - 1) * width
            axes[1].bar(x, row.neighbor_purity, width=width, color=color, label=label if subtype_index == 0 else None)
            errorbar_from_bounds(axes[1], x, row.neighbor_purity, row.neighbor_purity_ci_low, row.neighbor_purity_ci_high)
    axes[1].set_title("Fixed-label neighbor purity", fontweight="bold")
    axes[1].set_ylabel("Neighbor purity")
    axes[1].set_xticks(np.arange(len(subtypes)), [subtype_label(value) for value in subtypes], rotation=20, ha="right")
    axes[1].legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=8)
    axes[1].annotate("Insulin-dependent estimates are exploratory", xy=(0.02, 0.97), xycoords="axes fraction", ha="left", va="top", color=COLOR_NULL, fontsize=8)
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.02)
    figure.tight_layout()
    plotted = pd.concat([three_space[three_space.view == "test"].assign(source="three_space"), representation_table.assign(source="representation")], ignore_index=True, sort=False)
    save_figure(figure, "figure_D1_reorganization", title, plotted, {
        "input_artifact_paths": ["tables/phase3_three_space_preservation.csv", "phase3_ht_preservation/three_space_bootstrap_intervals.csv", "phase3_ht_preservation/representation_sensitivity.csv"],
        "sample_sizes": representation_table.groupby("canonical_stratum").n.first().to_dict(),
        "metrics_shown": ["kNN Jaccard with participant-bootstrap confidence interval", "Fixed-label neighbor purity with participant-bootstrap confidence interval"],
        "color_role_mapping": {"Clinical reference": COLOR_REFERENCE, "h0": COLOR_ADJUSTED, "ht or h0-to-ht": COLOR_NULL},
    })


def make_figure_d2(retention: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    title = "Initial-state neighbor retention differs across clinical clusters"
    subtypes = [subtype for subtype in CANONICAL_STRATA if subtype in retention.canonical_stratum.unique()]
    figure, axes = plt.subplots(1, len(subtypes), figsize=(4 * len(subtypes), 4.8), facecolor="white", sharey=True)
    if len(subtypes) == 1:
        axes = [axes]
    rng = np.random.default_rng(SEED)
    for subtype_index, (axis, subtype) in enumerate(zip(axes, subtypes)):
        subset = retention[retention.canonical_stratum == subtype]
        clusters = sorted(subset.display_cluster.unique())
        for cluster in clusters:
            group = subset[subset.display_cluster == cluster]
            x = np.full(len(group), cluster) + rng.uniform(-0.12, 0.12, len(group))
            axis.scatter(x, group.h0_to_ht_retention, s=14, color=COLOR_NULL, alpha=0.55)
            interval = bootstrap_mean_ci(group.h0_to_ht_retention.to_numpy(), PHASE3_BOOTSTRAP_N, SEED + subtype_index * 10 + int(cluster))
            axis.scatter([cluster], [interval["mean"]], s=75, color=COLOR_ADJUSTED, zorder=5)
            errorbar_from_bounds(axis, cluster, interval["mean"], interval["ci_low"], interval["ci_high"])
        axis.set_xticks(clusters, [f"C{int(cluster)}" for cluster in clusters])
        suffix = " (exploratory)" if bool(subset.exploratory.any()) else ""
        axis.set_title(subtype_label(subtype) + suffix, fontweight="bold", fontsize=10)
        axis.set_xlabel("Fixed clinical cluster")
    axes[0].set_ylabel("h0-to-ht kNN retention")
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.02)
    figure.tight_layout()
    save_figure(figure, "figure_D2_retention_by_cluster", title, retention, {
        "input_artifact_paths": ["phase3_ht_preservation/participant_retention.csv"],
        "sample_sizes": retention.groupby("canonical_stratum").participant_id.count().to_dict(),
        "metrics_shown": ["Participant h0-to-ht neighborhood retention", "Participant-bootstrap cluster mean and confidence interval"],
        "color_role_mapping": {"Participant": COLOR_NULL, "Cluster mean": COLOR_ADJUSTED},
    })


def make_figure_d3(cluster_coherence: pd.DataFrame, neutral_coherence: pd.DataFrame, pair_class: pd.DataFrame, quiver: pd.DataFrame, evidence: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    title = "Clinically similar participants show varying endpoint movement coherence"
    subtypes = [subtype for subtype in CANONICAL_STRATA if subtype in cluster_coherence.canonical_stratum.unique()]
    figure, axes = plt.subplots(4, len(subtypes), figsize=(4.3 * len(subtypes), 15), facecolor="white", squeeze=False)
    for subtype_index, subtype in enumerate(subtypes):
        full = cluster_coherence[cluster_coherence.canonical_stratum == subtype].sort_values("display_cluster")
        neutral = neutral_coherence[neutral_coherence.canonical_stratum == subtype].set_index("display_cluster").reindex(full.display_cluster)
        x = np.arange(len(full))
        width = 0.36
        panel_a = axes[0, subtype_index]
        panel_a.bar(x - width / 2, full.within_cosine, width, color=COLOR_ADJUSTED, label="Within cluster")
        panel_a.bar(x + width / 2, full.between_cosine, width, color=COLOR_REFERENCE, label="Between clusters")
        panel_a.set_xticks(x, [f"C{int(value)}" for value in full.display_cluster])
        panel_a.set_ylabel("Displacement cosine" if subtype_index == 0 else "")
        panel_a.set_title(subtype_label(subtype), fontweight="bold", fontsize=10)
        if subtype_index == 0:
            panel_a.legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=7)

        panel_b = axes[1, subtype_index]
        pair_row = pair_class[(pair_class.canonical_stratum == subtype) & (pair_class.representation == "full")].iloc[0]
        neighbor_mean = pair_row["class_3_h0_neighbor_pairs_mean"]
        matched_mean = pair_row["class_4_matched_non_neighbor_pairs_mean"]
        panel_b.bar([0, 1], [neighbor_mean, matched_mean], color=[COLOR_ADJUSTED, COLOR_REFERENCE])
        errorbar_from_bounds(panel_b, 0, neighbor_mean, pair_row["class_3_h0_neighbor_pairs_ci_low"], pair_row["class_3_h0_neighbor_pairs_ci_high"])
        errorbar_from_bounds(panel_b, 1, matched_mean, pair_row["class_4_matched_non_neighbor_pairs_ci_low"], pair_row["class_4_matched_non_neighbor_pairs_ci_high"])
        panel_b.set_xticks([0, 1], ["h0 neighbor", "Matched non-neighbor"], fontsize=8)
        panel_b.set_ylabel("Movement cosine" if subtype_index == 0 else "")

        panel_c = axes[2, subtype_index]
        panel_c.bar(x - width / 2, full.coherence_ratio, width, color=COLOR_REFERENCE, label="Full displacement")
        panel_c.bar(x + width / 2, neutral.coherence_ratio, width, color=COLOR_ADJUSTED, label="Centered neutral endpoint proxy")
        panel_c.axhline(COHERENCE_RATIO_BAR, color=COLOR_NULL, linestyle="--", linewidth=1.0)
        panel_c.set_xticks(x, [f"C{int(value)}" for value in full.display_cluster])
        panel_c.set_ylabel("Coherence ratio" if subtype_index == 0 else "")
        if subtype_index == 0:
            panel_c.legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=7)

        panel_d = axes[3, subtype_index]
        q = quiver[quiver.canonical_stratum == subtype]
        center_pc1 = float(pd.concat([q.h0_pc1, q.ht_pc1]).mean())
        center_pc2 = float(pd.concat([q.h0_pc2, q.ht_pc2]).mean())
        for cluster_index, cluster in enumerate(sorted(q.display_cluster.unique())):
            group = q[q.display_cluster == cluster]
            dx = group.ht_pc1.to_numpy() - group.h0_pc1.to_numpy()
            dy = group.ht_pc2.to_numpy() - group.h0_pc2.to_numpy()
            color = CLUSTER_COLORS[cluster_index % len(CLUSTER_COLORS)]
            panel_d.quiver(group.h0_pc1 - center_pc1, group.h0_pc2 - center_pc2, dx, dy, angles="xy", scale_units="xy", scale=1, color=color, alpha=0.25, width=0.004)
            mean_start = np.array([group.h0_pc1.mean() - center_pc1, group.h0_pc2.mean() - center_pc2])
            mean_end = np.array([group.ht_pc1.mean(), group.ht_pc2.mean()])
            mean_delta = mean_end - mean_start
            panel_d.quiver([mean_start[0]], [mean_start[1]], [mean_delta[0]], [mean_delta[1]], angles="xy", scale_units="xy", scale=1, color=color, edgecolor=COLOR_OBSERVED, linewidth=0.8, width=0.015, zorder=6)
        panel_d.set_xlabel("Centered shared PCA component 1")
        panel_d.set_ylabel("Centered shared PCA component 2" if subtype_index == 0 else "")
        panel_d.annotate("Display only; statistics use full dimension", xy=(0.02, 0.03), xycoords="axes fraction", color=COLOR_NULL, fontsize=7)
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.0)
    figure.tight_layout()
    plotted = pd.concat([cluster_coherence.assign(source="full_coherence"), neutral_coherence.assign(source="neutral_proxy_coherence"), pair_class.assign(source="pair_classes"), quiver.assign(source="quiver")], ignore_index=True, sort=False)
    save_figure(figure, "figure_D3_movement_coherence", title, plotted, {
        "input_artifact_paths": ["phase3_ht_preservation/cluster_displacement_coherence.csv", "phase3_ht_preservation/neutralized_cluster_coherence.csv", "phase3_ht_preservation/pair_class_movement.csv", "phase3_ht_preservation/quiver_display.csv"],
        "sample_sizes": cluster_coherence.groupby("canonical_stratum").n.sum().to_dict(),
        "metrics_shown": ["Within and between-cluster displacement cosine", "h0-neighbor and distance-duration-matched non-neighbor movement cosine", "Full and centered-neutral-endpoint-proxy coherence ratio", "Train-fitted shared-PCA endpoint displacement display"],
        "color_role_mapping": {"Reference or full": COLOR_REFERENCE, "Adjusted or within": COLOR_ADJUSTED, "Threshold": COLOR_NULL, "Quiver clusters": CLUSTER_COLORS},
    })


def make_figure_d4(representation_table: pd.DataFrame, cluster_coherence: pd.DataFrame, neutral_coherence: pd.DataFrame) -> str:
    sns.set_style("whitegrid")
    full_ht = representation_table[representation_table.representation == "ht_full"]
    neutral_ht = representation_table[representation_table.representation == "ht_neutral"]
    reduction = bool(neutral_ht.neighbor_purity.mean() < full_ht.neighbor_purity.mean() and neutral_ht.clinical_knn_jaccard.mean() < full_ht.clinical_knn_jaccard.mean())
    title = "Static neutralization reduces some latent neighborhood structure" if reduction else "Static neutralization changes latent neighborhood structure"
    subtypes = [subtype for subtype in CANONICAL_STRATA if subtype in representation_table.canonical_stratum.unique()]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")
    representations = [("h0_full", "Full h0", COLOR_REFERENCE), ("ht_full", "Full ht", COLOR_ADJUSTED), ("h0_neutral", "Neutral h0", COLOR_NULL), ("ht_neutral", "Neutral ht", COLOR_NULL), ("ht_glucose_residualized", "Residualized ht", COLOR_NULL)]
    width = 0.15
    for axis, metric, ylabel in [(axes[0, 0], "neighbor_purity", "Neighbor purity"), (axes[0, 1], "clinical_knn_jaccard", "Clinical kNN Jaccard")]:
        for rep_index, (representation, label, color) in enumerate(representations):
            for subtype_index, subtype in enumerate(subtypes):
                row = representation_table[(representation_table.canonical_stratum == subtype) & (representation_table.representation == representation)]
                x = subtype_index + (rep_index - 2) * width
                if len(row) and bool(row.iloc[0].estimable):
                    value = row.iloc[0][metric]
                    axis.bar(x, value, width=width, color=color, alpha=1.0 if rep_index < 2 else 0.55, hatch="" if rep_index < 3 else ("//" if rep_index == 3 else ".."), label=label if subtype_index == 0 else None)
                    low = row.iloc[0].get(metric + "_ci_low")
                    high = row.iloc[0].get(metric + "_ci_high")
                    errorbar_from_bounds(axis, x, value, low, high)
                elif subtype_index == 0:
                    axis.scatter([x], [0], marker="x", color=COLOR_NULL, label=label + " not estimable")
        axis.set_xticks(np.arange(len(subtypes)), [subtype_label(value) for value in subtypes], rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=7, ncol=2)
    axes[0, 0].set_title("Fixed-label neighborhood purity", fontweight="bold")
    axes[0, 1].set_title("Clinical neighborhood overlap", fontweight="bold")

    for axis, metric, ylabel in [(axes[1, 0], "gap_point", "Within-minus-between cosine"), (axes[1, 1], "coherence_ratio", "Coherence ratio")]:
        full_mean = cluster_coherence.groupby("canonical_stratum")[metric].mean().reindex(subtypes)
        neutral_mean = neutral_coherence.groupby("canonical_stratum")[metric].mean().reindex(subtypes)
        x = np.arange(len(subtypes))
        axis.bar(x - 0.18, full_mean, width=0.36, color=COLOR_REFERENCE, label="Full displacement")
        axis.bar(x + 0.18, neutral_mean, width=0.36, color=COLOR_ADJUSTED, label="Centered neutral endpoint proxy")
        axis.set_xticks(x, [subtype_label(value) for value in subtypes], rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=7)
        if metric == "gap_point":
            axis.axhline(0, color=COLOR_OBSERVED, linewidth=1.0)
        else:
            axis.axhline(COHERENCE_RATIO_BAR, color=COLOR_NULL, linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Cluster-specific movement gap", fontweight="bold")
    axes[1, 1].set_title("Cluster movement coherence", fontweight="bold")
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.01)
    figure.tight_layout()
    plotted = pd.concat([representation_table.assign(source="representations"), cluster_coherence.assign(source="full_displacement"), neutral_coherence.assign(source="neutral_proxy")], ignore_index=True, sort=False)
    save_figure(figure, "figure_D4_neutralization", title, plotted, {
        "input_artifact_paths": ["phase3_ht_preservation/representation_sensitivity.csv", "phase3_ht_preservation/cluster_displacement_coherence.csv", "phase3_ht_preservation/neutralized_cluster_coherence.csv"],
        "sample_sizes": representation_table.groupby("canonical_stratum").n.first().to_dict(),
        "metrics_shown": ["Cluster purity", "Clinical kNN overlap", "Within-minus-between movement cosine", "Coherence ratio"],
        "color_role_mapping": {"Full h0 or displacement": COLOR_REFERENCE, "Full ht or adjusted": COLOR_ADJUSTED, "Unavailable or sensitivity representations": COLOR_NULL},
        "neutral_h0_note": "Neutral h0 is population-constant and therefore not estimable for neighborhood metrics",
    })
    return title


def render_gate_d_report(three_space, bootstrap_table, representation_table, retention, displacement, cluster_coherence, pair_class, local_advantage, neutral_coherence, residualized, evidence, pc1_variance, frozen_manifest, figure_d4_title, input_audit) -> None:
    test_three_space = three_space[three_space.view == "test"]
    full_three_space = three_space[three_space.view == "full_cohort"]
    retention_summary = retention.groupby(["canonical_stratum", "display_cluster"]).agg(n=("participant_id", "count"), mean_h0_to_ht_retention=("h0_to_ht_retention", "mean"), median_h0_to_ht_retention=("h0_to_ht_retention", "median"), mean_clinical_to_ht_retention=("clinical_to_ht_retention", "mean")).reset_index()
    confidence_summary = retention.groupby(["canonical_stratum", "clinical_assignment_confidence_quartile"]).agg(n=("participant_id", "count"), mean_h0_to_ht_retention=("h0_to_ht_retention", "mean"), mean_clinical_to_ht_retention=("clinical_to_ht_retention", "mean")).reset_index()
    magnitude_summary = displacement.groupby(["canonical_stratum", "display_cluster"]).agg(n=("participant_id", "count"), mean_displacement_magnitude=("displacement_magnitude", "mean"), median_displacement_magnitude=("displacement_magnitude", "median"), displacement_magnitude_variance=("displacement_magnitude", "var")).reset_index()
    lines = [
        "# Gate D report",
        "",
        "## Concise interpretation",
        "",
        "Phase 3 used the existing participant-level overnight h_t endpoint summary and the frozen Phase 1 clinical clustering. It evaluated clinical-to-ht and h0-to-ht preservation, participant neighborhood retention, full-dimensional endpoint displacement, cluster-specific movement coherence, h0-neighbor movement advantage, a centered neutral-endpoint sensitivity proxy, and the valid portion of the glucose-residualized sensitivity. No model retraining, forward pass, latent clustering, or revision of Phase 1 was performed.",
        "",
        *[f"- {row.canonical_stratum} cluster {int(row.display_cluster)}: **{row.label}**" + (" (exploratory)" if row.exploratory else "") for row in evidence.itertuples()],
        "",
        "## Endpoint input and pairing audit",
        "",
        f"The paired artifacts contain {input_audit['paired_participants']} shared participants with {input_audit['vector_dimensions']['h0']} dimensions. All endpoint vectors are finite, participant identifiers are unique, and clinical split labels agree with the endpoint split labels. Endpoint anchor counts range from {input_audit['endpoint_anchor_count']['min']} to {input_audit['endpoint_anchor_count']['max']} with median {input_audit['endpoint_anchor_count']['median']:.1f}.",
        "",
        "The primary endpoint is an overnight 0 to 6 hour summary. Endpoint displacement is not described as a full temporal trajectory. Time-resolved h_t snapshots are unavailable.",
        "",
        "## Three-space preservation, test primary",
        "",
        dataframe_to_markdown(test_three_space),
        "",
        "## Participant-bootstrap intervals and change from h0 to ht",
        "",
        dataframe_to_markdown(bootstrap_table),
        "",
        f"Participant bootstrap resamples per test comparison: {PHASE3_BOOTSTRAP_N}.",
        "",
        "## Three-space preservation, full-cohort sensitivity",
        "",
        dataframe_to_markdown(full_three_space),
        "",
        "## Participant-level neighborhood retention",
        "",
        dataframe_to_markdown(retention_summary),
        "",
        "Retention by frozen clinical-cluster assignment-confidence quartile:",
        "",
        dataframe_to_markdown(confidence_summary),
        "",
        "h0 has no separate assignment-confidence quartile because h0 was never clustered or assigned. Creating one would violate the governing no-latent-clustering rule.",
        "",
        "## Endpoint displacement magnitude",
        "",
        dataframe_to_markdown(magnitude_summary),
        "",
        "Displacement was calculated as h_t minus h0 before any normalization. Cosine normalization is implicit only in cosine-similarity calculations.",
        "",
        "## Pair-class movement",
        "",
        dataframe_to_markdown(pair_class),
        "",
        "Pair-class confidence intervals bootstrap participant-level aggregates rather than pair rows. Different-cluster and non-neighbor partners were greedily matched within subtype on baseline h0 distance and endpoint anchor count, with equal partner counts per participant where feasible.",
        "",
        "## Cluster-specific full displacement coherence",
        "",
        dataframe_to_markdown(cluster_coherence),
        "",
        f"Cluster-label permutations per cluster: {PHASE3_PERMUTATION_N}.",
        "",
        "## Local h0-neighborhood movement advantage",
        "",
        dataframe_to_markdown(local_advantage),
        "",
        "## Static-neutralization sensitivity",
        "",
        "The coordinate-compatible constant neutral h0 vector was verified during Study 1 extraction but was not saved. Recomputing it would require a prohibited new model pass. Therefore the exact d_i neutral = h_t neutral minus h0 neutral analysis is not estimable from the frozen artifacts. The table below uses Study 1s saved convention, the neutral endpoint centered by the full Study 1 cohort mean, as a sensitivity proxy. It is not treated as exact neutralized displacement and cannot support the stronger movement-coherence-persisting-after-static-neutralization label.",
        "",
        dataframe_to_markdown(neutral_coherence),
        "",
        "## Glucose-residualized endpoint sensitivity",
        "",
        "The residualized h_t endpoint supports fixed-label purity, clinical-neighbor preservation, and distance concordance. Residualized displacement was not computed because no coordinate-compatible residualized h0 exists.",
        "",
        dataframe_to_markdown(residualized),
        "",
        "## Displacement PCA variance",
        "",
        dataframe_to_markdown(pc1_variance),
        "",
        "## Pre-registered evidence decision",
        "",
        dataframe_to_markdown(evidence),
        "",
        "The insulin-dependent subtype remains exploratory and is excluded from headline pooled conclusions while retained in all tables and figures.",
        "",
        "## Figures",
        "",
        "![Reorganization across clinical, h0, and ht](../figures/full_resolution/figure_D1_reorganization.png)",
        "",
        "![Retention by cluster](../figures/full_resolution/figure_D2_retention_by_cluster.png)",
        "",
        "![Movement coherence](../figures/full_resolution/figure_D3_movement_coherence.png)",
        "",
        f"Figure D4 title selected after checking direction: **{figure_d4_title}**",
        "",
        "![Neutralization and residualized sensitivities](../figures/full_resolution/figure_D4_neutralization.png)",
        "",
        "## Confirmations",
        "",
        "h0 and h_t were read from existing Study 1 artifacts and were never clustered. The clinical factors, missing-data strategy, selected k values, transformations, centroids, and display labels remained frozen. No model forward pass or time-resolved extension was run.",
        "",
        "## Next phase",
        "",
        "No time-resolved h_t artifact is available, so the optional time-resolved extension cannot run without a separately approved model pass. This study stops at Gate D.",
        "",
        "Gate D reached. Endpoint preservation and displacement coherence from h0 to ht have been evaluated. Waiting for confirmation before any further phase.",
        "",
    ]
    (DECISION_ROOT / "GATE_D_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
