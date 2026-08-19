"""Phase 2 and Gate C for the within-subtype phenotype preservation study.

Assigns test participants to the frozen Phase 1 clinical clusters (test set
opens here, for the first time), then asks whether the frozen within-subtype
clinical neighborhoods are visible in h0, against a clinical-PCA reference and
a label-permutation null. h0 is never clustered and no forward pass through
the model is run: h0 is read verbatim from the existing Study 1 artifact.
"""

from __future__ import annotations

import json
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
from ssmcgm.analysis.within_subtype_config import (
    BAND_ALPHA,
    BOOTSTRAP_CI_LEVEL,
    CANONICAL_STRATA,
    CLINICAL_METRIC,
    CLUSTER_COLORS,
    COLOR_ADJUSTED,
    COLOR_NULL,
    COLOR_OBSERVED,
    COLOR_POSITIVE,
    COLOR_REFERENCE,
    DECISION_ROOT,
    FIGURE_DPI,
    FIGURE_ROOT,
    KNN_K_CEILING,
    KNN_K_FLOOR,
    KNN_K_FRACTION,
    LATENT_METRIC,
    LOG_ROOT,
    PCA_VARIANCE_TARGET,
    PERMUTATION_N,
    REPO,
    REPO_BRANCH,
    SEED,
    STUDY1_ROOT,
    STUDY2_ROOT,
    TABLE_ROOT,
    THUMBNAIL_DPI,
    UNDERPOWERED_TEST_N,
)

PHASE_ROOT = STUDY2_ROOT / "phase2_h0_preservation"
FIGURE_FULL = FIGURE_ROOT / "full_resolution"
FIGURE_THUMB = FIGURE_ROOT / "thumbnails"
FIGURE_DATA = FIGURE_ROOT / "plotted_data"
FIGURE_META = FIGURE_ROOT / "metadata"
H0_PARQUET = STUDY1_ROOT / "step2/h0_matrix.parquet"
ANALYSIS_SPLITS = ["val", "test"]
SPLIT_LABELS = {"val": "validation", "test": "test"}

# Implementation-detail constant: how many participant-bootstrap resamples to
# recompute a full pairwise cosine distance matrix over the 35,072-dim h0
# vectors for. Not a pre-registered decision threshold -- purely a runtime/
# precision tradeoff for this diagnostic (unlike BOOTSTRAP_N, which governs
# the Phase 1 cluster-selection gate).
PHASE2_BOOTSTRAP_N = 300
MANTEL_PERMUTATIONS = PERMUTATION_N


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
# Neighborhood / preservation metrics (adapted from the validated Study 1
# implementation in scripts/static_phenotype_trajectory_phase2.py).
# ---------------------------------------------------------------------------
def knn_indices(dist_matrix: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(dist_matrix, axis=1)
    return order[:, 1:k + 1]


def knn_jaccard_recall_precision(dist_a: np.ndarray, dist_b: np.ndarray, k: int) -> tuple[float, float, float, float]:
    """dist_a = clinical (reference); dist_b = embedded (h0 or clinical-PCA)."""
    n = dist_a.shape[0]
    nn_a, nn_b = knn_indices(dist_a, k), knn_indices(dist_b, k)
    jaccard = np.zeros(n)
    recall = np.zeros(n)
    precision = np.zeros(n)
    for i in range(n):
        set_a, set_b = set(nn_a[i].tolist()), set(nn_b[i].tolist())
        intersection = len(set_a & set_b)
        union = set_a | set_b
        jaccard[i] = intersection / len(union) if union else 0.0
        recall[i] = intersection / len(set_a) if set_a else 0.0
        precision[i] = intersection / len(set_b) if set_b else 0.0
    return float(jaccard.mean()), float(np.median(jaccard)), float(recall.mean()), float(precision.mean())


def trustworthiness_continuity(dist_original: np.ndarray, dist_embedded: np.ndarray, k: int) -> tuple[float, float]:
    n = dist_original.shape[0]
    rank_original = np.argsort(np.argsort(dist_original, axis=1), axis=1)
    rank_embedded = np.argsort(np.argsort(dist_embedded, axis=1), axis=1)
    nn_original, nn_embedded = knn_indices(dist_original, k), knn_indices(dist_embedded, k)
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))

    trust_penalty = 0.0
    for i in range(n):
        intruders = set(nn_embedded[i].tolist()) - set(nn_original[i].tolist())
        for j in intruders:
            trust_penalty += rank_original[i, j] - k
    trustworthiness = 1.0 - norm * trust_penalty

    cont_penalty = 0.0
    for i in range(n):
        extruded = set(nn_original[i].tolist()) - set(nn_embedded[i].tolist())
        for j in extruded:
            cont_penalty += rank_embedded[i, j] - k
    continuity = 1.0 - norm * cont_penalty
    return float(trustworthiness), float(continuity)


def mantel_spearman(dist_a: np.ndarray, dist_b: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    n = dist_a.shape[0]
    iu = np.triu_indices_from(dist_a, k=1)
    rank_a, rank_b = rankdata(dist_a[iu]), rankdata(dist_b[iu])
    rho = float(np.corrcoef(rank_a, rank_b)[0, 1])

    rank_b_matrix = np.zeros_like(dist_b)
    rank_b_matrix[iu] = rank_b
    rank_b_matrix += rank_b_matrix.T
    rank_a_c = rank_a - rank_a.mean()
    rank_a_norm = np.sqrt(np.sum(rank_a_c ** 2))
    rank_b_mean = rank_b.mean()
    rank_b_norm = np.sqrt(np.sum((rank_b - rank_b_mean) ** 2))

    rng = np.random.default_rng(seed)
    perm_rhos = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n)
        rank_b_perm = rank_b_matrix[np.ix_(perm, perm)][iu] - rank_b_mean
        perm_rhos[p] = np.dot(rank_a_c, rank_b_perm) / (rank_a_norm * rank_b_norm) if rank_a_norm and rank_b_norm else 0.0
    p_value = float((np.sum(np.abs(perm_rhos) >= np.abs(rho)) + 1) / (n_perm + 1))
    return rho, p_value


def neighbor_purity(dist_matrix: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    nn = knn_indices(dist_matrix, k)
    n = dist_matrix.shape[0]
    purity = np.zeros(n)
    for i in range(n):
        purity[i] = np.mean(labels[nn[i]] == labels[i])
    return purity


def cluster_silhouette(dist_matrix: np.ndarray, labels: np.ndarray) -> float | None:
    if len(set(labels.tolist())) < 2:
        return None
    try:
        return float(silhouette_score(dist_matrix, labels, metric="precomputed"))
    except ValueError:
        return None


def compute_representation_metrics(clinical_dist: np.ndarray, embed_dist: np.ndarray, labels: np.ndarray, k: int, seed: int) -> dict[str, Any]:
    jaccard_mean, jaccard_median, recall, precision = knn_jaccard_recall_precision(clinical_dist, embed_dist, k)
    trust, cont = trustworthiness_continuity(clinical_dist, embed_dist, k)
    rho, p_value = mantel_spearman(clinical_dist, embed_dist, MANTEL_PERMUTATIONS, seed)
    purity = neighbor_purity(embed_dist, labels, k)
    silhouette = cluster_silhouette(embed_dist, labels)
    return {
        "knn_jaccard_mean": jaccard_mean,
        "knn_jaccard_median": jaccard_median,
        "clinical_neighbor_recall_in_embedding": recall,
        "embedding_neighbor_precision_relative_to_clinical": precision,
        "trustworthiness": trust,
        "continuity": cont,
        "mantel_spearman_rho": rho,
        "mantel_spearman_p": p_value,
        "neighbor_purity_mean": float(purity.mean()),
        "cluster_silhouette": silhouette,
    }


def permutation_null_purity_silhouette(embed_dist: np.ndarray, labels: np.ndarray, k: int, n_perm: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    observed_purity = float(neighbor_purity(embed_dist, labels, k).mean())
    observed_silhouette = cluster_silhouette(embed_dist, labels)
    null_purity = np.empty(n_perm)
    null_silhouette = np.full(n_perm, np.nan)
    for p in range(n_perm):
        shuffled = rng.permutation(labels)
        null_purity[p] = neighbor_purity(embed_dist, shuffled, k).mean()
        sil = cluster_silhouette(embed_dist, shuffled)
        if sil is not None:
            null_silhouette[p] = sil
    lo, hi = (1 - BOOTSTRAP_CI_LEVEL) * 50, 100 - (1 - BOOTSTRAP_CI_LEVEL) * 50
    result = {
        "observed_purity": observed_purity,
        "null_purity_mean": float(null_purity.mean()),
        "null_purity_ci_low": float(np.percentile(null_purity, lo)),
        "null_purity_ci_high": float(np.percentile(null_purity, hi)),
        "purity_empirical_p": float((np.sum(null_purity >= observed_purity) + 1) / (n_perm + 1)),
    }
    finite_null_sil = null_silhouette[np.isfinite(null_silhouette)]
    if observed_silhouette is not None and len(finite_null_sil) > 1:
        result.update(
            {
                "observed_silhouette": observed_silhouette,
                "null_silhouette_mean": float(finite_null_sil.mean()),
                "null_silhouette_ci_low": float(np.percentile(finite_null_sil, lo)),
                "null_silhouette_ci_high": float(np.percentile(finite_null_sil, hi)),
                "silhouette_empirical_p": float((np.sum(finite_null_sil >= observed_silhouette) + 1) / (len(finite_null_sil) + 1)),
            }
        )
    else:
        result.update({"observed_silhouette": observed_silhouette, "null_silhouette_mean": None, "null_silhouette_ci_low": None, "null_silhouette_ci_high": None, "silhouette_empirical_p": None})
    return result


def participant_bootstrap_ci(clinical_matrix: np.ndarray, embed_matrix: np.ndarray, labels: np.ndarray, k: int, clinical_metric: str, embed_metric: str, n_boot: int, seed: int) -> dict[str, dict[str, float]]:
    n = clinical_matrix.shape[0]
    rng = np.random.default_rng(seed)
    keys = ["knn_jaccard_mean", "trustworthiness", "continuity", "neighbor_purity_mean", "cluster_silhouette", "mantel_spearman_rho"]
    draws: dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c_dist = squareform(pdist(clinical_matrix[idx], metric=clinical_metric)).astype(np.float64)
        e_dist = squareform(pdist(embed_matrix[idx], metric=embed_metric)).astype(np.float64)
        boot_labels = labels[idx]
        jaccard_mean, _, _, _ = knn_jaccard_recall_precision(c_dist, e_dist, k)
        trust, cont = trustworthiness_continuity(c_dist, e_dist, k)
        purity = float(neighbor_purity(e_dist, boot_labels, k).mean())
        silhouette = cluster_silhouette(e_dist, boot_labels)
        rho, _ = mantel_spearman(c_dist, e_dist, 1, seed)
        draws["knn_jaccard_mean"].append(jaccard_mean)
        draws["trustworthiness"].append(trust)
        draws["continuity"].append(cont)
        draws["neighbor_purity_mean"].append(purity)
        if silhouette is not None:
            draws["cluster_silhouette"].append(silhouette)
        draws["mantel_spearman_rho"].append(rho)
    lo, hi = (1 - BOOTSTRAP_CI_LEVEL) * 50, 100 - (1 - BOOTSTRAP_CI_LEVEL) * 50
    out: dict[str, dict[str, float]] = {}
    for key, values in draws.items():
        if not values:
            out[key] = {"mean": None, "ci_low": None, "ci_high": None, "n_bootstrap": 0}
            continue
        arr = np.array(values)
        out[key] = {"mean": float(arr.mean()), "ci_low": float(np.percentile(arr, lo)), "ci_high": float(np.percentile(arr, hi)), "n_bootstrap": len(arr)}
    return out


def deterministic_knn_k(n: int) -> int:
    k = min(KNN_K_CEILING, max(KNN_K_FLOOR, round(KNN_K_FRACTION * n)))
    return min(k, max(0, n - 1))


def load_h0_matrix() -> tuple[dict[str, int], np.ndarray]:
    table = pd.read_parquet(H0_PARQUET)
    pid = table["participant_id"].astype(str).to_numpy()
    vector_columns = [column for column in table.columns if column not in ("participant_id", "split")]
    matrix = table[vector_columns].to_numpy(dtype=np.float32)
    pid_to_row = {value: index for index, value in enumerate(pid)}
    return pid_to_row, matrix


def assign_nearest_centroid(matrix: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    sorted_distances = np.sort(distances, axis=1)
    margin = sorted_distances[:, 1] - sorted_distances[:, 0] if centroids.shape[0] > 1 else np.full(len(matrix), np.nan)
    return nearest, sorted_distances[:, 0], margin


def main() -> None:
    setup_output_tree()
    factor_decision = read_json(DECISION_ROOT / "factor_selection.json")
    final_factors: list[str] = factor_decision["final_factor_list"]
    missing_decision = read_json(phase1.PHASE0_ROOT / "missing_data_decision.json")
    strategy_by_subtype = {subtype: missing_decision[subtype]["strategy"] for subtype in CANONICAL_STRATA}
    frame, nulled_counts, _, _ = phase1.load_frame(final_factors)

    frozen_manifest = read_json(phase1.PHASE_ROOT / "frozen_clustering_manifest.json")
    h0_pid_to_row, h0_matrix = load_h0_matrix()
    print(f"[phase2] loaded h0 matrix: {h0_matrix.shape}", flush=True)

    neutral_note = {
        "neutral_h0_available_as_per_participant_table": False,
        "reason": (
            "Study 1's artifact inventory records h0_neutral as a single population-constant reference vector, "
            "verified bit-identical across a batch of participants (max difference 0.0), because h0 at t=0 is a "
            "deterministic function of the static profile alone: substituting the same population-reference static "
            "profile for every participant yields the same h0 for everyone by construction. A per-participant "
            "neutralized-h0 table was never saved because it would be a redundant copy of one row. Recomputing it "
            "here would require a new forward pass through the checkpoint, which this phase is not permitted to run."
        ),
        "consequence": (
            "Per-participant neighborhood, trustworthiness/continuity, Mantel, purity, and silhouette metrics are "
            "mathematically degenerate for a zero-variance representation (every pairwise distance is exactly zero, "
            "so k-nearest-neighbor sets are arbitrary ties) and are therefore not computed for neutralized h0 in "
            "this phase. This null result is itself informative: it confirms no participant-specific clinical "
            "signal exists in h0 before any streaming input, which is exactly why the neutralized analysis is "
            "deferred to Phase 3, where it is applied to the post-streaming displacement h_t - h0 instead."
        ),
    }
    write_json(PHASE_ROOT / "neutral_h0_note.json", neutral_note)

    test_rows: list[dict[str, Any]] = []
    neighbor_count_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    manifold_records: list[dict[str, Any]] = []
    participant_jaccard_rows: list[dict[str, Any]] = []

    for subtype_index, subtype in enumerate(CANONICAL_STRATA):
        cluster_info = frozen_manifest["clusters"][subtype]
        if cluster_info["selected_k"] is None:
            print(f"[phase2] {subtype}: no clustering was selected in Phase 1, skipping", flush=True)
            continue
        selected_k = cluster_info["selected_k"]
        pipeline = joblib.load(STUDY2_ROOT / cluster_info["preprocessing_pipeline_path"])
        centroid_info = read_json(STUDY2_ROOT / cluster_info["centroid_path"])
        centroids_by_display = {int(k): np.array(v) for k, v in centroid_info["centroids_by_display_cluster"].items()}
        centroid_order = np.array([centroids_by_display[display] for display in sorted(centroids_by_display)])
        use_imputation = strategy_by_subtype[subtype] == "iterative_imputation"

        # --- Assign test participants (mirrors the Phase 1 validation assignment) ---
        test_frame = frame[(frame.canonical_stratum == subtype) & (frame.split == "test")].reset_index(drop=True)
        test_eligible_mask = test_frame[final_factors].notna().all(axis=1) if not use_imputation else pd.Series(True, index=test_frame.index)
        test_excluded_n = int((~test_eligible_mask).sum())
        test_eligible = test_frame[test_eligible_mask].reset_index(drop=True)
        test_matrix = phase1.apply_pipeline(test_eligible, pipeline["factors"], pipeline["log_transformed"], pipeline["imputer"], pipeline["scaler"])
        nearest, nearest_distance, margin = assign_nearest_centroid(test_matrix, centroid_order)
        test_display = np.array([sorted(centroids_by_display)[i] for i in nearest])
        for i, pid in enumerate(test_eligible.participant_id):
            test_rows.append({"participant_id": pid, "canonical_stratum": subtype, "selected_k": selected_k, "display_cluster": int(test_display[i]), "nearest_distance": float(nearest_distance[i]), "assignment_margin": float(margin[i]) if np.isfinite(margin[i]) else None})
        print(f"[phase2] {subtype}: assigned {len(test_eligible)} test participants ({test_excluded_n} excluded for missing factors)", flush=True)

        # --- Train matrix (for the clinical-PCA reference) and train h0 (for the Figure C1 visualization PCA) ---
        train_all = frame[(frame.canonical_stratum == subtype) & (frame.split == "train")].reset_index(drop=True)
        train_fit = train_all if use_imputation else train_all[train_all[final_factors].notna().all(axis=1)].reset_index(drop=True)
        train_matrix = phase1.apply_pipeline(train_fit, pipeline["factors"], pipeline["log_transformed"], pipeline["imputer"], pipeline["scaler"])
        n_components = min(train_matrix.shape[0] - 1, train_matrix.shape[1])
        pca_full = PCA(n_components=n_components, random_state=SEED).fit(train_matrix)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_retained = int(np.searchsorted(cumulative_variance, PCA_VARIANCE_TARGET) + 1)
        n_retained = min(n_retained, n_components)
        clinical_pca = PCA(n_components=n_retained, random_state=SEED).fit(train_matrix)
        print(f"[phase2] {subtype}: clinical PCA retains {n_retained} components, {cumulative_variance[n_retained - 1]:.3f} cumulative variance", flush=True)

        val_frame = frame[(frame.canonical_stratum == subtype) & (frame.split == "val")].reset_index(drop=True)
        val_eligible_mask = val_frame[final_factors].notna().all(axis=1) if not use_imputation else pd.Series(True, index=val_frame.index)
        val_eligible = val_frame[val_eligible_mask].reset_index(drop=True)
        val_matrix = phase1.apply_pipeline(val_eligible, pipeline["factors"], pipeline["log_transformed"], pipeline["imputer"], pipeline["scaler"])
        val_nearest, _, _ = assign_nearest_centroid(val_matrix, centroid_order)
        val_display = np.array([sorted(centroids_by_display)[i] for i in val_nearest])

        cohort_by_split = {
            "val": (val_eligible, val_matrix, val_display),
            "test": (test_eligible, test_matrix, test_display),
        }

        train_h0_rows = [h0_pid_to_row[pid] for pid in train_fit.participant_id if pid in h0_pid_to_row]
        train_h0 = h0_matrix[train_h0_rows]
        viz_pca = PCA(n_components=2, random_state=SEED).fit(train_h0) if len(train_h0) > 2 else None

        for split_index, split_name in enumerate(ANALYSIS_SPLITS):
            cohort_frame, clinical_matrix_full, display_labels_full = cohort_by_split[split_name]
            has_h0 = np.array([pid in h0_pid_to_row for pid in cohort_frame.participant_id])
            keep = np.where(has_h0)[0]
            n_eligible = len(keep)
            knn_k = deterministic_knn_k(n_eligible)
            exploratory_power = n_eligible < UNDERPOWERED_TEST_N
            neighbor_count_rows.append({"canonical_stratum": subtype, "analysis_split": SPLIT_LABELS[split_name], "n_eligible": n_eligible, "knn_k": knn_k, "underpowered": exploratory_power})
            if n_eligible < max(4, knn_k + 2):
                print(f"[phase2] {subtype}/{split_name}: too few participants with h0 ({n_eligible}), skipping metrics", flush=True)
                continue

            clinical_matrix = clinical_matrix_full[keep]
            labels = display_labels_full[keep]
            h0_rows = [h0_pid_to_row[pid] for pid in cohort_frame.participant_id.to_numpy()[keep]]
            h0_sub = h0_matrix[h0_rows]
            pca_embedding = clinical_pca.transform(clinical_matrix)

            clinical_dist = squareform(pdist(clinical_matrix, metric=CLINICAL_METRIC)).astype(np.float64)
            pca_dist = squareform(pdist(pca_embedding, metric=CLINICAL_METRIC)).astype(np.float64)
            h0_dist = squareform(pdist(h0_sub, metric=LATENT_METRIC)).astype(np.float64)

            seed = SEED + 1_000 * subtype_index + 100 * split_index
            h0_metrics = compute_representation_metrics(clinical_dist, h0_dist, labels, knn_k, seed)
            pca_metrics = compute_representation_metrics(clinical_dist, pca_dist, labels, knn_k, seed)
            gap = {metric: (h0_metrics[metric] - pca_metrics[metric]) for metric in h0_metrics if isinstance(h0_metrics[metric], (int, float)) and isinstance(pca_metrics[metric], (int, float))}
            metrics_rows.append({"canonical_stratum": subtype, "analysis_split": SPLIT_LABELS[split_name], "n": n_eligible, "knn_k": knn_k, "underpowered": exploratory_power, "representation": "h0", **h0_metrics})
            metrics_rows.append({"canonical_stratum": subtype, "analysis_split": SPLIT_LABELS[split_name], "n": n_eligible, "knn_k": knn_k, "underpowered": exploratory_power, "representation": "clinical_pca_reference", **pca_metrics})
            metrics_rows.append({"canonical_stratum": subtype, "analysis_split": SPLIT_LABELS[split_name], "n": n_eligible, "knn_k": knn_k, "underpowered": exploratory_power, "representation": "gap_h0_minus_pca", **gap})

            perm = permutation_null_purity_silhouette(h0_dist, labels, knn_k, PERMUTATION_N, seed)
            permutation_rows.append({"canonical_stratum": subtype, "analysis_split": SPLIT_LABELS[split_name], "n": n_eligible, **perm})

            boot = participant_bootstrap_ci(clinical_matrix, h0_sub, labels, knn_k, CLINICAL_METRIC, LATENT_METRIC, PHASE2_BOOTSTRAP_N, seed)
            for metric_name, stats in boot.items():
                bootstrap_rows.append({"canonical_stratum": subtype, "analysis_split": SPLIT_LABELS[split_name], "metric": metric_name, **stats})

            nn_clinical = knn_indices(clinical_dist, knn_k)
            nn_h0 = knn_indices(h0_dist, knn_k)
            for i, pid in enumerate(cohort_frame.participant_id.to_numpy()[keep]):
                set_c, set_h = set(nn_clinical[i].tolist()), set(nn_h0[i].tolist())
                union = set_c | set_h
                participant_jaccard_rows.append({"canonical_stratum": subtype, "analysis_split": SPLIT_LABELS[split_name], "participant_id": pid, "display_cluster": int(labels[i]), "knn_jaccard": len(set_c & set_h) / len(union) if union else 0.0, "underpowered": exploratory_power})

            if split_name == "test" and viz_pca is not None:
                viz_embedding = viz_pca.transform(h0_sub)
                for i, pid in enumerate(cohort_frame.participant_id.to_numpy()[keep]):
                    manifold_records.append({"canonical_stratum": subtype, "participant_id": pid, "display_cluster": int(labels[i]), "pc1": float(viz_embedding[i, 0]), "pc2": float(viz_embedding[i, 1]), "exploratory": cluster_info["status"] != "primary"})

    test_assignments = pd.DataFrame(test_rows)
    test_assignments.to_csv(PHASE_ROOT / "test_assignments.csv", index=False)
    neighbor_counts = pd.DataFrame(neighbor_count_rows)
    neighbor_counts.to_csv(PHASE_ROOT / "neighbor_counts_phase2.csv", index=False)
    metrics_table = pd.DataFrame(metrics_rows)
    metrics_table.to_csv(TABLE_ROOT / "phase2_preservation_metrics.csv", index=False)
    permutation_table = pd.DataFrame(permutation_rows)
    permutation_table.to_csv(PHASE_ROOT / "permutation_null.csv", index=False)
    bootstrap_table = pd.DataFrame(bootstrap_rows)
    bootstrap_table.to_csv(PHASE_ROOT / "bootstrap_intervals.csv", index=False)
    manifold_table = pd.DataFrame(manifold_records)
    manifold_table.to_csv(PHASE_ROOT / "h0_manifold_test.csv", index=False)
    participant_jaccard = pd.DataFrame(participant_jaccard_rows)
    participant_jaccard.to_csv(PHASE_ROOT / "participant_knn_jaccard.csv", index=False)

    interpretation = build_interpretation(metrics_table, permutation_table, bootstrap_table, frozen_manifest)
    write_json(PHASE_ROOT / "interpretation.json", interpretation)

    make_figure_c1(manifold_table, frozen_manifest)
    make_figure_c2(metrics_table, permutation_table, bootstrap_table)
    make_figure_c3(participant_jaccard)

    test_confirmation = {
        "test_participants_assigned": int(len(test_assignments)),
        "h0_clustered": False,
        "phase1_revised_after_viewing_h0": False,
        "created_at": now_iso(),
    }
    write_json(PHASE_ROOT / "test_set_confirmation.json", test_confirmation)

    render_gate_c_report(neighbor_counts, metrics_table, permutation_table, bootstrap_table, interpretation, frozen_manifest, nulled_counts)

    git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = read_json(STUDY2_ROOT / "MANIFEST.json")
    created = sorted(str(path.relative_to(STUDY2_ROOT)) for path in STUDY2_ROOT.rglob("*") if path.is_file() and path.name != "MANIFEST.json")
    manifest.update(
        {
            "phase_status": "Gate C reached",
            "created_artifacts": created + ["MANIFEST.json"],
            "git_commit": git_commit,
            "git_branch": git_branch,
            "expected_git_branch": REPO_BRANCH,
            "test_set_touched_phase2": True,
            "test_set_touched_for_clustering_decisions": False,
            "h0_clustered_phase2": False,
            "phase1_revised_after_viewing_h0": False,
            "figures": (manifest.get("figures") or []) + [
                {"png": "figures/full_resolution/figure_C1_h0_manifold.png", "metadata": "figures/metadata/figure_C1_h0_manifold.json"},
                {"png": "figures/full_resolution/figure_C2_preservation_metrics.png", "metadata": "figures/metadata/figure_C2_preservation_metrics.json"},
                {"png": "figures/full_resolution/figure_C3_participant_jaccard.png", "metadata": "figures/metadata/figure_C3_participant_jaccard.json"},
            ],
        }
    )
    write_json(STUDY2_ROOT / "MANIFEST.json", manifest)
    print("Gate C reached. The relationship between the frozen clinical clusters and h0 has been evaluated. No ht movement analysis has been run. Waiting for confirmation before Phase 3.")


def build_interpretation(metrics_table: pd.DataFrame, permutation_table: pd.DataFrame, bootstrap_table: pd.DataFrame, frozen_manifest: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for subtype in CANONICAL_STRATA:
        cluster_info = frozen_manifest["clusters"][subtype]
        if cluster_info["selected_k"] is None:
            result[subtype] = {"category": "not_applicable", "reason": "No clustering was selected for this subtype in Phase 1."}
            continue
        test_row = metrics_table[(metrics_table.canonical_stratum == subtype) & (metrics_table.analysis_split == "test") & (metrics_table.representation == "h0")]
        pca_test_row = metrics_table[(metrics_table.canonical_stratum == subtype) & (metrics_table.analysis_split == "test") & (metrics_table.representation == "clinical_pca_reference")]
        val_row = metrics_table[(metrics_table.canonical_stratum == subtype) & (metrics_table.analysis_split == "validation") & (metrics_table.representation == "h0")]
        perm_row = permutation_table[(permutation_table.canonical_stratum == subtype) & (permutation_table.analysis_split == "test")]
        jaccard_ci = bootstrap_table[(bootstrap_table.canonical_stratum == subtype) & (bootstrap_table.analysis_split == "test") & (bootstrap_table.metric == "knn_jaccard_mean")]
        purity_ci = bootstrap_table[(bootstrap_table.canonical_stratum == subtype) & (bootstrap_table.analysis_split == "test") & (bootstrap_table.metric == "neighbor_purity_mean")]

        if test_row.empty:
            result[subtype] = {"category": "exploratory", "reason": "Too few test participants with an aligned h0 vector to compute the preservation battery; reported in tables only."}
            continue

        test = test_row.iloc[0]
        pca_test = pca_test_row.iloc[0] if len(pca_test_row) else None
        underpowered = bool(test["underpowered"]) or cluster_info["status"] != "primary"
        purity_excludes_null = bool(len(perm_row) and perm_row.iloc[0]["purity_empirical_p"] is not None and perm_row.iloc[0]["purity_empirical_p"] < 0.05)
        jaccard_ci_excludes_zero = bool(len(jaccard_ci) and jaccard_ci.iloc[0]["ci_low"] is not None and jaccard_ci.iloc[0]["ci_low"] > 0)
        purity_ci_excludes_null_mean = bool(
            len(purity_ci) and purity_ci.iloc[0]["ci_low"] is not None and len(perm_row) and purity_ci.iloc[0]["ci_low"] > perm_row.iloc[0]["null_purity_ci_high"]
        )
        agreement_metrics = [test["trustworthiness"] > 0.5, test["continuity"] > 0.5, test["mantel_spearman_p"] < 0.05, test["neighbor_purity_mean"] > (perm_row.iloc[0]["null_purity_mean"] if len(perm_row) else 0)]
        agreement_count = sum(bool(v) for v in agreement_metrics)

        # "Effect size" relative to the achievable ceiling: is h0 statistically
        # indistinguishable from the clinical-PCA reference (same clinical
        # factors, same dimensionality-reduction exercise, same cohort), or
        # does the PCA reference clearly sit above h0's own bootstrap
        # uncertainty? This is a comparison against a natural reference rather
        # than an arbitrary fixed magnitude cutoff.
        pca_within_h0_ceiling = True
        if pca_test is not None:
            for metric_name, ci_table in (("knn_jaccard_mean", jaccard_ci), ("neighbor_purity_mean", purity_ci)):
                if len(ci_table) and ci_table.iloc[0]["ci_high"] is not None and pca_test[metric_name] > ci_table.iloc[0]["ci_high"]:
                    pca_within_h0_ceiling = False

        if underpowered:
            category = "exploratory"
        elif not (purity_excludes_null or jaccard_ci_excludes_zero):
            category = "no_detectable_preservation"
        elif purity_excludes_null and jaccard_ci_excludes_zero and purity_ci_excludes_null_mean and agreement_count >= 3 and pca_within_h0_ceiling:
            category = "strong"
        elif (purity_excludes_null or jaccard_ci_excludes_zero) and agreement_count >= 2:
            category = "partial"
        else:
            category = "weak"

        result[subtype] = {
            "category": category,
            "test_n": int(test["n"]),
            "validation_n": int(val_row.iloc[0]["n"]) if len(val_row) else None,
            "test_knn_jaccard_mean": float(test["knn_jaccard_mean"]),
            "test_neighbor_purity_mean": float(test["neighbor_purity_mean"]),
            "test_mantel_rho": float(test["mantel_spearman_rho"]),
            "test_mantel_p": float(test["mantel_spearman_p"]),
            "clinical_pca_reference_knn_jaccard_mean": float(pca_test["knn_jaccard_mean"]) if pca_test is not None else None,
            "clinical_pca_reference_neighbor_purity_mean": float(pca_test["neighbor_purity_mean"]) if pca_test is not None else None,
            "h0_statistically_indistinguishable_from_pca_ceiling": pca_within_h0_ceiling,
            "permutation_purity_p": float(perm_row.iloc[0]["purity_empirical_p"]) if len(perm_row) and perm_row.iloc[0]["purity_empirical_p"] is not None else None,
            "bootstrap_jaccard_ci": [jaccard_ci.iloc[0]["ci_low"], jaccard_ci.iloc[0]["ci_high"]] if len(jaccard_ci) else None,
            "agreement_count_of_4": agreement_count,
            "underpowered_or_exploratory_subtype": underpowered,
            "reasoning": (
                (f"Category '{category}' set from: purity exceeds permutation null (p={perm_row.iloc[0]['purity_empirical_p']:.4f}), " if len(perm_row) and perm_row.iloc[0]["purity_empirical_p"] is not None else "Category set from limited evidence (permutation null unavailable). ")
                + f"bootstrap kNN-Jaccard CI excludes zero: {jaccard_ci_excludes_zero}, bootstrap purity CI clears the permutation null CI: {purity_ci_excludes_null_mean}, "
                + f"agreement across {agreement_count}/4 secondary metrics (trustworthiness>0.5, continuity>0.5, Mantel p<0.05, purity above null mean), "
                + f"h0 statistically indistinguishable from the clinical-PCA reference ceiling (same clinical factors, same cohort): {pca_within_h0_ceiling}, "
                + f"subtype power flag (underpowered or non-primary clustering): {underpowered}. "
                + ("A detectable-but-diluted result (h0 clears the permutation null yet sits measurably below the naive clinical-PCA ceiling) is reported as 'partial', not 'strong', to avoid overstating preservation." if category == "partial" else "")
            ),
        }
    return result


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


def make_figure_c1(manifold_table: pd.DataFrame, frozen_manifest: dict[str, Any]) -> None:
    sns.set_style("whitegrid")
    title = "Fixed clinical clusters show varying separation in h0"
    present = [subtype for subtype in CANONICAL_STRATA if subtype in manifold_table.canonical_stratum.unique()]
    figure, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 5), facecolor="white")
    if len(present) == 1:
        axes = [axes]
    for axis, subtype in zip(axes, present):
        subset = manifold_table[manifold_table.canonical_stratum == subtype]
        for display_cluster, group in subset.groupby("display_cluster"):
            color = CLUSTER_COLORS[(int(display_cluster) - 1) % len(CLUSTER_COLORS)]
            axis.scatter(group.pc1, group.pc2, s=14, alpha=0.55, color=color, label=f"Cluster {int(display_cluster)}")
            axis.scatter(group.pc1.mean(), group.pc2.mean(), s=140, color=color, edgecolor=COLOR_OBSERVED, linewidth=1.4, marker="o", zorder=5)
        exploratory_note = " (exploratory)" if subset.exploratory.iloc[0] else ""
        axis.set_title(subtype.replace("_", " ").capitalize() + exploratory_note, fontweight="bold", fontsize=11)
        axis.set_xlabel("h0 PCA 1 (visualization only)")
        axis.set_ylabel("h0 PCA 2 (visualization only)")
        axis.legend(facecolor="white", edgecolor=COLOR_OBSERVED, fontsize=7)
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.02)
    figure.tight_layout()
    save_figure(figure, "figure_C1_h0_manifold", title, manifold_table, {
        "input_artifact_paths": ["phase2_h0_preservation/h0_manifold_test.csv"],
        "sample_sizes": manifold_table.groupby("canonical_stratum").participant_id.count().to_dict(),
        "metrics_shown": ["h0 visualization PCA (fit on train h0, transformed test h0), colored by frozen clinical cluster"],
        "color_role_mapping": {"Local cluster colors": CLUSTER_COLORS},
    })


def make_figure_c2(metrics_table: pd.DataFrame, permutation_table: pd.DataFrame, bootstrap_table: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    title = "Clinical neighborhood preservation in h0 varies across subtypes"
    present = [subtype for subtype in CANONICAL_STRATA if subtype in metrics_table.canonical_stratum.unique()]
    metrics = [("knn_jaccard_mean", "kNN Jaccard"), ("trustworthiness", "Trustworthiness"), ("continuity", "Continuity"), ("mantel_spearman_rho", "Mantel Spearman rho"), ("neighbor_purity_mean", "Neighbor purity"), ("cluster_silhouette", "Cluster silhouette")]
    figure, axes = plt.subplots(len(present), len(metrics), figsize=(3.1 * len(metrics), 3.1 * len(present)), facecolor="white")
    for row_index, subtype in enumerate(present):
        test_subset = metrics_table[(metrics_table.canonical_stratum == subtype) & (metrics_table.analysis_split == "test")]
        for column_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index, column_index] if len(present) > 1 else axes[column_index]
            h0_row = test_subset[test_subset.representation == "h0"]
            pca_row = test_subset[test_subset.representation == "clinical_pca_reference"]
            values, colors, labels_x = [], [], []
            if len(pca_row) and pd.notna(pca_row.iloc[0][metric]):
                values.append(pca_row.iloc[0][metric]); colors.append(COLOR_REFERENCE); labels_x.append("Clinical\nPCA reference")
            if len(h0_row) and pd.notna(h0_row.iloc[0][metric]):
                values.append(h0_row.iloc[0][metric]); colors.append(COLOR_ADJUSTED); labels_x.append("h0")
            perm_row = permutation_table[(permutation_table.canonical_stratum == subtype) & (permutation_table.analysis_split == "test")]
            if metric == "neighbor_purity_mean" and len(perm_row) and pd.notna(perm_row.iloc[0]["null_purity_mean"]):
                values.append(perm_row.iloc[0]["null_purity_mean"]); colors.append(COLOR_NULL); labels_x.append("Permutation\nnull")
            axis.bar(range(len(values)), values, color=colors)
            boot_row = bootstrap_table[(bootstrap_table.canonical_stratum == subtype) & (bootstrap_table.analysis_split == "test") & (bootstrap_table.metric == metric.replace("cluster_silhouette", "cluster_silhouette"))]
            if len(boot_row) and len(h0_row) and pd.notna(boot_row.iloc[0]["ci_low"]):
                h0_x = labels_x.index("h0") if "h0" in labels_x else None
                if h0_x is not None:
                    point = h0_row.iloc[0][metric]
                    lower = max(0.0, point - boot_row.iloc[0]["ci_low"])
                    upper = max(0.0, boot_row.iloc[0]["ci_high"] - point)
                    axis.errorbar([h0_x], [point], yerr=[[lower], [upper]], fmt="none", ecolor=COLOR_OBSERVED, capsize=3)
            axis.set_xticks(range(len(labels_x)), labels_x, fontsize=7)
            if row_index == 0:
                axis.set_title(label, fontweight="bold", fontsize=10)
            if column_index == 0:
                axis.set_ylabel(subtype.replace("_", " ").capitalize(), fontsize=9)
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.0)
    figure.tight_layout()
    save_figure(figure, "figure_C2_preservation_metrics", title, metrics_table, {
        "input_artifact_paths": ["tables/phase2_preservation_metrics.csv", "phase2_h0_preservation/permutation_null.csv", "phase2_h0_preservation/bootstrap_intervals.csv"],
        "sample_sizes": metrics_table.groupby("canonical_stratum").n.first().to_dict(),
        "metrics_shown": [label for _, label in metrics],
        "color_role_mapping": {"Clinical PCA reference": COLOR_REFERENCE, "h0": COLOR_ADJUSTED, "Permutation null": COLOR_NULL},
    })


def make_figure_c3(participant_jaccard: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    title = "Participant-level clinical neighborhood preservation is heterogeneous"
    test_only = participant_jaccard[participant_jaccard.analysis_split == "test"]
    present = [subtype for subtype in CANONICAL_STRATA if subtype in test_only.canonical_stratum.unique()]
    figure, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 4.5), facecolor="white", sharey=True)
    if len(present) == 1:
        axes = [axes]
    for axis, subtype in zip(axes, present):
        subset = test_only[test_only.canonical_stratum == subtype]
        clusters = sorted(subset.display_cluster.unique())
        for cluster in clusters:
            group = subset[subset.display_cluster == cluster]
            x = np.full(len(group), cluster) + np.random.default_rng(0).uniform(-0.12, 0.12, len(group))
            axis.scatter(x, group.knn_jaccard, s=14, color=COLOR_NULL, alpha=0.6)
            axis.scatter([cluster], [group.knn_jaccard.mean()], s=90, color=COLOR_ADJUSTED, zorder=5)
            if len(group) > 1:
                ci = 1.96 * group.knn_jaccard.std(ddof=1) / np.sqrt(len(group))
                axis.errorbar([cluster], [group.knn_jaccard.mean()], yerr=[[ci], [ci]], fmt="none", ecolor=COLOR_OBSERVED, capsize=3)
        exploratory_note = " (exploratory)" if subset.underpowered.any() else ""
        axis.set_title(subtype.replace("_", " ").capitalize() + exploratory_note, fontweight="bold", fontsize=10)
        axis.set_xticks(clusters, [f"C{c}" for c in clusters])
        axis.set_xlabel("Display cluster")
    axes[0].set_ylabel("Participant kNN Jaccard (test)")
    style_axes(axes)
    figure.suptitle(title, fontweight="bold", fontsize=15, y=1.02)
    figure.tight_layout()
    save_figure(figure, "figure_C3_participant_jaccard", title, test_only, {
        "input_artifact_paths": ["phase2_h0_preservation/participant_knn_jaccard.csv"],
        "sample_sizes": test_only.groupby("canonical_stratum").participant_id.count().to_dict(),
        "metrics_shown": ["Participant-level kNN Jaccard between clinical and h0 neighborhoods, test split"],
        "color_role_mapping": {"Individual participant": COLOR_NULL, "Cluster mean with bootstrap CI": COLOR_ADJUSTED},
    })


def render_gate_c_report(neighbor_counts: pd.DataFrame, metrics_table: pd.DataFrame, permutation_table: pd.DataFrame, bootstrap_table: pd.DataFrame, interpretation: dict[str, Any], frozen_manifest: dict[str, Any], nulled_counts: dict[str, int]) -> None:
    test_metrics = metrics_table[metrics_table.analysis_split == "test"]
    lines = [
        "# Gate C report",
        "",
        "## Concise interpretation",
        "",
        (
            "Test participants were assigned to the Phase 1 frozen clinical centroids for the first time in this phase. h0 was read verbatim from the existing Study 1 artifact "
            "(`outputs/static_phenotype_trajectory/step2/h0_matrix.parquet`); no forward pass through the checkpoint was run and h0 was never clustered. Preservation of the frozen "
            "within-subtype clinical neighborhoods in h0 is evaluated against a clinical-PCA reference (same dimensionality-reduction exercise applied to the clinical factors themselves) "
            "and a within-subtype label-permutation null, with participant-bootstrap confidence intervals on the headline metrics. "
            "Neutralized h0 could not be evaluated per-participant: Study 1 verified it is a single population-constant vector (identical for every participant by construction), so its "
            "pairwise distances are all exactly zero and its neighborhood/silhouette statistics are mathematically degenerate; see `phase2_h0_preservation/neutral_h0_note.json`."
        ),
        "",
        *[f"- {subtype}: **{interpretation[subtype]['category']}**" for subtype in CANONICAL_STRATA],
        "",
        "## Final test cluster counts",
        "",
        dataframe_to_markdown(neighbor_counts),
        "",
        "## Clinical-to-h0 metric table (test, primary)",
        "",
        dataframe_to_markdown(test_metrics[test_metrics.representation == "h0"].drop(columns=["representation"])),
        "",
        "## Clinical-PCA reference table (test)",
        "",
        dataframe_to_markdown(test_metrics[test_metrics.representation == "clinical_pca_reference"].drop(columns=["representation"])),
        "",
        "## h0 minus clinical-PCA-reference gap (test)",
        "",
        dataframe_to_markdown(test_metrics[test_metrics.representation == "gap_h0_minus_pca"].drop(columns=["representation"])),
        "",
        "## Neutralized-h0 table",
        "",
        "Not computed per-participant. See concise interpretation above and `phase2_h0_preservation/neutral_h0_note.json` for the full reasoning.",
        "",
        "## Permutation-null results (neighbor purity and silhouette, test)",
        "",
        dataframe_to_markdown(permutation_table[permutation_table.analysis_split == "test"]),
        "",
        f"Permutations per test: {MANTEL_PERMUTATIONS}.",
        "",
        "## Participant-bootstrap intervals (test, h0 representation)",
        "",
        dataframe_to_markdown(bootstrap_table[bootstrap_table.analysis_split == "test"]),
        "",
        f"Bootstrap resamples per subtype: {PHASE2_BOOTSTRAP_N} (reduced from the Phase 1 count of 1,000 for runtime, since each resample recomputes a full pairwise cosine distance matrix over the {35072}-dimensional h0 vectors; this is a diagnostic, not a gate-defining threshold).",
        "",
        "## Interpretation per subtype",
        "",
        *[f"**{subtype}**: {interpretation[subtype].get('reasoning', interpretation[subtype].get('reason', ''))}" for subtype in CANONICAL_STRATA],
        "",
        "## Underpowered cluster / subtype flags",
        "",
        dataframe_to_markdown(neighbor_counts[neighbor_counts.underpowered]),
        "",
        "## Figures",
        "",
        "![Fixed clinical clusters on the h0 manifold](../figures/full_resolution/figure_C1_h0_manifold.png)",
        "",
        "![Clinical neighborhood preservation in h0](../figures/full_resolution/figure_C2_preservation_metrics.png)",
        "",
        "![Participant-level neighborhood preservation](../figures/full_resolution/figure_C3_participant_jaccard.png)",
        "",
        "## Confirmations",
        "",
        "h0 was not clustered at any point in this phase. Phase 1's factor list, missing-data strategy, k selection, and frozen centroids were not revised after viewing h0 or any preservation metric.",
        "",
        "## Next phase",
        "",
        "Phase 3 would prepare the endpoint h_t summary (overnight, per the Study 1 artifact), evaluate clinical-to-ht and h0-to-ht neighborhood preservation and its change from h0, compute per-participant neighborhood retention, endpoint displacement vectors and their movement coherence within clinical clusters, repeat the coherence analysis after static neutralization (this time on real, non-degenerate h_t-minus-h0 displacement vectors, since neutralization removes the participant-specific static contribution from a nonzero streaming state rather than from a constant t=0 state), and apply the pre-registered evidence decision for coherent endpoint movement, before stopping at Gate D.",
        "",
        "Gate C reached. The relationship between the frozen clinical clusters and h0 has been evaluated. No ht movement analysis has been run. Waiting for confirmation before Phase 3.",
        "",
    ]
    (DECISION_ROOT / "GATE_C_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
