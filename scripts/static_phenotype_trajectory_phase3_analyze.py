"""Phase 3, analysis: three-space preservation, glucose-residualized sensitivity,
membership flow, coherent drift, neutralized control, figures, manifest, README.

Static-only clinical phenotype partition project. See build prompt for full spec.
Consumes the outputs of static_phenotype_trajectory_phase3_extract.py.
"""

import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Named constants (identical to extract script / Study 1)
# ---------------------------------------------------------------------------
SEED = 42
KNN_K = 15
BOOTSTRAP_N = 1000
LATENT_METRIC = "cosine"
CLINICAL_METRIC = "euclidean"
COHERENCE_RATIO_BAR = 0.30
PRIMARY_K = 2
EXPLORATORY_K = 4
RECOVERY_METRICS = ["neighbor_purity", "knn_jaccard"]
MANTEL_PERMUTATIONS = 999

FACTORS = ["age", "bmi", "hba1c_baseline", "c_peptide_baseline", "tg_hdl_ratio"]

STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
COLOR_H0 = "#003366"
COLOR_PCA = "#888888"
COLOR_HT = "#5BBABA"
COLOR_POS = "#BA2828"
COLOR_NEG = "#003366"
COLOR_EVENT = "#FF0000"
COLOR_NULL = "#888888"

STUDY1_ROOT = f"{ROOT}/outputs/static_phenotype_trajectory"
STEP1_DIR = f"{STUDY1_ROOT}/step1"
STEP2_DIR = f"{STUDY1_ROOT}/step2"
STEP3_DIR = f"{STUDY1_ROOT}/step3"
FIG_DIR = f"{STUDY1_ROOT}/figures"

sns.set_style("whitegrid", {"axes.edgecolor": "0.3", "grid.color": "0.85"})


def apply_figure_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


# ---------------------------------------------------------------------------
# Metric helpers (mirrors Phase 2, generalized for reuse)
# ---------------------------------------------------------------------------
def knn_indices(dist_matrix, k):
    order = np.argsort(dist_matrix, axis=1)
    order = order[:, 1:k + 1] if order.shape[1] > k else order[:, 1:]
    return order


def knn_jaccard_per_participant(dist_a, dist_b, k):
    n = dist_a.shape[0]
    nn_a = knn_indices(dist_a, k)
    nn_b = knn_indices(dist_b, k)
    scores = np.zeros(n)
    for i in range(n):
        set_a, set_b = set(nn_a[i].tolist()), set(nn_b[i].tolist())
        union = set_a | set_b
        scores[i] = len(set_a & set_b) / len(union) if union else 0.0
    return scores


def trustworthiness_continuity(dist_original, dist_embedded, k):
    n = dist_original.shape[0]
    rank_original = np.argsort(np.argsort(dist_original, axis=1), axis=1)
    rank_embedded = np.argsort(np.argsort(dist_embedded, axis=1), axis=1)
    nn_original = knn_indices(dist_original, k)
    nn_embedded = knn_indices(dist_embedded, k)
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


def mantel_spearman(dist_a, dist_b, n_perm, seed):
    n = dist_a.shape[0]
    iu = np.triu_indices_from(dist_a, k=1)
    vec_a, vec_b = dist_a[iu], dist_b[iu]
    rank_a, rank_b = rankdata(vec_a), rankdata(vec_b)
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
        perm_rhos[p] = np.dot(rank_a_c, rank_b_perm) / (rank_a_norm * rank_b_norm)
    p_value = float((np.sum(np.abs(perm_rhos) >= np.abs(rho)) + 1) / (n_perm + 1))
    return rho, p_value


def neighbor_purity_per_participant(dist_matrix, labels, k):
    n = dist_matrix.shape[0]
    nn = knn_indices(dist_matrix, k)
    purity = np.zeros(n)
    for i in range(n):
        purity[i] = np.mean(labels[nn[i]] == labels[i])
    return purity


def compute_metric_arrays(clinical_dist, embed_dist, labels_by_k, k_knn):
    """Return per-participant arrays (for bootstrap) plus point-estimate scalars."""
    jaccard_arr = knn_jaccard_per_participant(clinical_dist, embed_dist, k_knn)
    trust, cont = trustworthiness_continuity(clinical_dist, embed_dist, k_knn)
    rho, p_val = mantel_spearman(clinical_dist, embed_dist, MANTEL_PERMUTATIONS, SEED)
    purity_arrs = {k: neighbor_purity_per_participant(embed_dist, labels, k_knn) for k, labels in labels_by_k.items()}
    sil = {}
    for k, labels in labels_by_k.items():
        try:
            sil[k] = float(silhouette_score(embed_dist, labels, metric="precomputed"))
        except ValueError:
            sil[k] = float("nan")
    return {
        "knn_jaccard_arr": jaccard_arr,
        "trustworthiness": trust,
        "continuity": cont,
        "mantel_spearman_rho": rho,
        "mantel_spearman_p": p_val,
        "purity_arrs": purity_arrs,
        "silhouette": sil,
    }


def scalarize(metrics, k_primary):
    return {
        "knn_jaccard": float(metrics["knn_jaccard_arr"].mean()),
        "trustworthiness": metrics["trustworthiness"],
        "continuity": metrics["continuity"],
        "mantel_spearman_rho": metrics["mantel_spearman_rho"],
        "mantel_spearman_p": metrics["mantel_spearman_p"],
        f"neighbor_purity_k{k_primary}_mean": float(metrics["purity_arrs"][k_primary].mean()),
        f"cluster_silhouette_k{k_primary}": metrics["silhouette"][k_primary],
        f"neighbor_purity_k{EXPLORATORY_K}_mean": float(metrics["purity_arrs"][EXPLORATORY_K].mean()),
        f"cluster_silhouette_k{EXPLORATORY_K}": metrics["silhouette"][EXPLORATORY_K],
    }


def bootstrap_gap_ci(arr_ht, arr_h0, n_boot, seed):
    """Paired participant-bootstrap CI on mean(arr_ht) - mean(arr_h0)."""
    rng = np.random.default_rng(seed)
    n = len(arr_ht)
    point = float(arr_ht.mean() - arr_h0.mean())
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = arr_ht[idx].mean() - arr_h0[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    excludes_zero = bool(lo > 0 or hi < 0)
    return point, float(lo), float(hi), excludes_zero


def nearest_centroid_labels(vectors, cluster_labels, metric):
    unique_clusters = sorted(np.unique(cluster_labels))
    centroids = np.stack([vectors[cluster_labels == c].mean(axis=0) for c in unique_clusters])
    if metric == "cosine":
        d = squareform(pdist(np.vstack([vectors, centroids]), metric="cosine"))[: len(vectors), len(vectors):]
    else:
        d = squareform(pdist(np.vstack([vectors, centroids]), metric="euclidean"))[: len(vectors), len(vectors):]
    nearest = np.argmin(d, axis=1)
    return np.array([unique_clusters[i] for i in nearest]), centroids, unique_clusters


def transition_matrix(labels_a, labels_b, unique_labels):
    n_k = len(unique_labels)
    mat = np.zeros((n_k, n_k), dtype=np.int64)
    label_to_idx = {c: i for i, c in enumerate(unique_labels)}
    for a, b in zip(labels_a, labels_b):
        mat[label_to_idx[a], label_to_idx[b]] += 1
    return mat


def bootstrap_crossing_rate(labels_a, labels_b, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(labels_a)
    stayed = (labels_a == labels_b).astype(float)
    point = float(1.0 - stayed.mean())
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = 1.0 - stayed[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def within_between_cosine(d_vectors, cluster_labels):
    """Mean within-cluster vs between-cluster pairwise cosine similarity of displacement vectors."""
    cos_sim = 1.0 - squareform(pdist(d_vectors, metric="cosine"))
    n = len(cluster_labels)
    same = cluster_labels[:, None] == cluster_labels[None, :]
    iu = np.triu_indices(n, k=1)
    same_iu = same[iu]
    sims_iu = cos_sim[iu]
    within = float(sims_iu[same_iu].mean())
    between = float(sims_iu[~same_iu].mean())
    return within, between, sims_iu, same_iu


def bootstrap_within_between_gap(d_vectors, cluster_labels, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(cluster_labels)
    _, _, sims_iu, same_iu = within_between_cosine(d_vectors, cluster_labels)
    point = float(sims_iu[same_iu].mean() - sims_iu[~same_iu].mean())
    boots = np.empty(n_boot)
    cos_full = 1.0 - squareform(pdist(d_vectors, metric="cosine"))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub_labels = cluster_labels[idx]
        sub_cos = cos_full[np.ix_(idx, idx)]
        m = len(idx)
        iu = np.triu_indices(m, k=1)
        same = (sub_labels[:, None] == sub_labels[None, :])[iu]
        sims = sub_cos[iu]
        if same.sum() == 0 or (~same).sum() == 0:
            boots[b] = np.nan
            continue
        boots[b] = sims[same].mean() - sims[~same].mean()
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    excludes_zero = bool(lo > 0 or hi < 0)
    return point, float(lo), float(hi), excludes_zero


def coherence_ratio_per_cluster(d_vectors, cluster_labels):
    ratios = {}
    for c in sorted(np.unique(cluster_labels)):
        d_c = d_vectors[cluster_labels == c]
        mean_disp_norm = np.linalg.norm(d_c.mean(axis=0))
        mean_norm = np.linalg.norm(d_c, axis=1).mean()
        ratios[int(c)] = float(mean_disp_norm / mean_norm) if mean_norm > 0 else float("nan")
    return ratios


def main():
    print("=" * 80)
    print("PHASE 3 ANALYSIS STEP 0: Load extraction outputs, align cohorts")
    print("=" * 80)
    with open(f"{STEP3_DIR}/anchor_extraction_report.json") as f:
        anchor_report = json.load(f)
    print(f"  Retained: {anchor_report['n_retained']} / (retained+dropped)  dropped={anchor_report['n_dropped']}")
    print(f"  Retained per split: {anchor_report['retained_per_split']}")
    print(f"  PC1-glucose correlation: {anchor_report['pc1_glucose_correlation']:.4f}")
    print(f"  Neutral h0 batch max diff: {anchor_report['neutral_h0_batch5_max_diff']:.3e}")

    h_t_full_df = pd.read_parquet(f"{STEP3_DIR}/h_t_full.parquet")
    h_t_neutral_df = pd.read_parquet(f"{STEP3_DIR}/h_t_neutral.parquet")
    h_t_resid_df = pd.read_parquet(f"{STEP3_DIR}/h_t_full_residualized.parquet")
    retained_pids = h_t_full_df["participant_id"].astype(str).tolist()
    n = len(retained_pids)
    print(f"  h_t retained cohort n={n}")

    meta_cols_full = ["participant_id", "split", "n_overnight_anchors", "avg_glucose_mgdl"]
    meta_cols_other = ["participant_id", "split", "n_overnight_anchors"]
    h_t_full = h_t_full_df.drop(columns=meta_cols_full).to_numpy(dtype=np.float64)
    h_t_neutral = h_t_neutral_df.drop(columns=meta_cols_other).to_numpy(dtype=np.float64)
    h_t_resid = h_t_resid_df.drop(columns=meta_cols_other).to_numpy(dtype=np.float64)
    split_arr = h_t_full_df["split"].to_numpy()
    avg_glucose = h_t_full_df["avg_glucose_mgdl"].to_numpy(dtype=np.float64)

    labels_df = pd.read_parquet(f"{STEP1_DIR}/participant_cluster_labels.parquet")
    z_df = pd.read_parquet(f"{STEP1_DIR}/zscored_factor_matrix.parquet")
    h0_full_df = pd.read_parquet(f"{STEP2_DIR}/h0_matrix.parquet")
    labels_df["participant_id"] = labels_df["participant_id"].astype(str)
    z_df["participant_id"] = z_df["participant_id"].astype(str)
    h0_full_df["participant_id"] = h0_full_df["participant_id"].astype(str)

    order_df = pd.DataFrame({"participant_id": retained_pids})
    labels_aligned = order_df.merge(labels_df, on="participant_id", how="left")
    z_aligned = order_df.merge(z_df, on="participant_id", how="left")
    h0_aligned = order_df.merge(h0_full_df, on="participant_id", how="left")
    assert labels_aligned["participant_id"].tolist() == retained_pids
    assert z_aligned["participant_id"].tolist() == retained_pids
    assert h0_aligned["participant_id"].tolist() == retained_pids

    cluster_k2 = labels_aligned[f"cluster_k{PRIMARY_K}"].to_numpy().astype(int)
    cluster_k4 = labels_aligned[f"cluster_k{EXPLORATORY_K}"].to_numpy().astype(int)
    labels_by_k = {PRIMARY_K: cluster_k2, EXPLORATORY_K: cluster_k4}
    clinical_z = z_aligned[FACTORS].to_numpy(dtype=np.float64)
    h0_vecs = h0_aligned.drop(columns=["participant_id", "split"]).to_numpy(dtype=np.float64)
    test_mask = split_arr == "test"
    print(f"  Test subset n={test_mask.sum()}")

    print("\n" + "=" * 80)
    print("STEP 1: Three-space preservation, test primary + full-cohort sensitivity")
    print("=" * 80)
    with open(f"{STEP2_DIR}/h0_vs_pca_preservation_metrics.json") as f:
        phase2_metrics = json.load(f)

    preservation_results = {}
    for view_name, mask in [("test", test_mask), ("full_cohort", np.ones(n, dtype=bool))]:
        idx = np.where(mask)[0]
        clinical_sub = squareform(pdist(clinical_z[idx], metric=CLINICAL_METRIC)).astype(np.float64)
        h0_sub = squareform(pdist(h0_vecs[idx], metric=LATENT_METRIC)).astype(np.float64)
        ht_sub = squareform(pdist(h_t_full[idx], metric=LATENT_METRIC)).astype(np.float64)
        labels_sub = {k: v[idx] for k, v in labels_by_k.items()}

        h0_metrics_full_arrays = compute_metric_arrays(clinical_sub, h0_sub, labels_sub, KNN_K)
        ht_metrics_full_arrays = compute_metric_arrays(clinical_sub, ht_sub, labels_sub, KNN_K)
        h0_scalar = scalarize(h0_metrics_full_arrays, PRIMARY_K)
        ht_scalar = scalarize(ht_metrics_full_arrays, PRIMARY_K)
        pca_scalar = phase2_metrics[view_name]["pca_baseline"]

        recovery = {}
        for metric_name in RECOVERY_METRICS:
            if metric_name == "neighbor_purity":
                arr_ht = ht_metrics_full_arrays["purity_arrs"][PRIMARY_K]
                arr_h0 = h0_metrics_full_arrays["purity_arrs"][PRIMARY_K]
            elif metric_name == "knn_jaccard":
                arr_ht = ht_metrics_full_arrays["knn_jaccard_arr"]
                arr_h0 = h0_metrics_full_arrays["knn_jaccard_arr"]
            point, lo, hi, excludes_zero = bootstrap_gap_ci(arr_ht, arr_h0, BOOTSTRAP_N, SEED)
            recovery[metric_name] = {"gap_ht_minus_h0": point, "ci_lo": lo, "ci_hi": hi, "recovery_holds": excludes_zero and point > 0}
            print(f"  [{view_name}] recovery check {metric_name}: h_t-h0 gap={point:+.4f} "
                  f"CI=[{lo:+.4f},{hi:+.4f}] recovery_holds={recovery[metric_name]['recovery_holds']}")

        preservation_results[view_name] = {
            "h0": h0_scalar, "pca_baseline": pca_scalar, "h_t": ht_scalar,
            "gap_ht_minus_h0": {m: ht_scalar[m] - h0_scalar[m] for m in ht_scalar},
            "recovery_decision": recovery, "n": int(len(idx)),
        }
        for m in ht_scalar:
            print(f"  [{view_name}] {m:30s} h0={h0_scalar[m]:+.4f} pca={pca_scalar.get(m, float('nan')):+.4f} "
                  f"h_t={ht_scalar[m]:+.4f}")

    print("\n" + "=" * 80)
    print("STEP 2: Glucose-residualized sensitivity (full cohort)")
    print("=" * 80)
    idx_full = np.arange(n)
    clinical_full_d = squareform(pdist(clinical_z, metric=CLINICAL_METRIC)).astype(np.float64)
    h0_full_d = squareform(pdist(h0_vecs, metric=LATENT_METRIC)).astype(np.float64)
    ht_resid_d = squareform(pdist(h_t_resid, metric=LATENT_METRIC)).astype(np.float64)
    resid_metrics_arrays = compute_metric_arrays(clinical_full_d, ht_resid_d, labels_by_k, KNN_K)
    resid_scalar = scalarize(resid_metrics_arrays, PRIMARY_K)
    nonresid_scalar = preservation_results["full_cohort"]["h_t"]
    print("  Residualized vs non-residualized h_t (full cohort):")
    for m in resid_scalar:
        print(f"    {m:30s} non-resid={nonresid_scalar[m]:+.4f}  resid={resid_scalar[m]:+.4f}  "
              f"delta={resid_scalar[m] - nonresid_scalar[m]:+.4f}")

    print("\n" + "=" * 80)
    print("STEP 3: Membership flow (k=2 primary, k=4 exploratory), full pass")
    print("=" * 80)
    membership_results = {}
    for k in (PRIMARY_K, EXPLORATORY_K):
        labels = labels_by_k[k]
        label_clinical, _, uniq = nearest_centroid_labels(clinical_z, labels, CLINICAL_METRIC)
        label_h0, _, _ = nearest_centroid_labels(h0_vecs, labels, LATENT_METRIC)
        label_ht, _, _ = nearest_centroid_labels(h_t_full, labels, LATENT_METRIC)

        self_consistency = float((label_clinical == labels).mean())
        trans_clin_h0 = transition_matrix(labels, label_h0, uniq)
        trans_h0_ht = transition_matrix(label_h0, label_ht, uniq)
        crossing_clin_h0, ci_lo1, ci_hi1 = bootstrap_crossing_rate(labels, label_h0, BOOTSTRAP_N, SEED)
        crossing_h0_ht, ci_lo2, ci_hi2 = bootstrap_crossing_rate(label_h0, label_ht, BOOTSTRAP_N, SEED)
        stayed_frac_clin_h0 = np.diag(trans_clin_h0) / trans_clin_h0.sum(axis=1)
        stayed_frac_h0_ht = np.diag(trans_h0_ht) / trans_h0_ht.sum(axis=1)

        membership_results[k] = {
            "cluster_labels": uniq,
            "self_consistency_clinical_relabel": self_consistency,
            "transition_clinical_to_h0": trans_clin_h0.tolist(),
            "transition_h0_to_ht": trans_h0_ht.tolist(),
            "crossing_rate_clinical_to_h0": {"point": crossing_clin_h0, "ci_lo": ci_lo1, "ci_hi": ci_hi1},
            "crossing_rate_h0_to_ht": {"point": crossing_h0_ht, "ci_lo": ci_lo2, "ci_hi": ci_hi2},
            "stayed_fraction_clinical_to_h0": stayed_frac_clin_h0.tolist(),
            "stayed_fraction_h0_to_ht": stayed_frac_h0_ht.tolist(),
            "labels_h0": label_h0, "labels_ht": label_ht, "labels_clinical_relabel": label_clinical,
        }
        print(f"  k={k}: self-consistency of clinical relabel = {self_consistency:.4f} (sanity, expect ~1.0)")
        print(f"  k={k}: clinical->h0 crossing rate = {crossing_clin_h0:.4f} CI=[{ci_lo1:.4f},{ci_hi1:.4f}]")
        print(f"  k={k}: h0->h_t crossing rate = {crossing_h0_ht:.4f} CI=[{ci_lo2:.4f},{ci_hi2:.4f}]")

    print("\n" + "=" * 80)
    print("STEP 4: Coherent drift, full pass")
    print("=" * 80)
    d_full = h_t_full - h0_vecs
    drift_results = {}
    for k in (PRIMARY_K, EXPLORATORY_K):
        labels = labels_by_k[k]
        within, between, _, _ = within_between_cosine(d_full, labels)
        point, lo, hi, excludes_zero = bootstrap_within_between_gap(d_full, labels, BOOTSTRAP_N, SEED)
        coh_ratio = coherence_ratio_per_cluster(d_full, labels)
        drift_results[k] = {
            "within_cosine": within, "between_cosine": between,
            "within_minus_between_gap": {"point": point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": excludes_zero},
            "coherence_ratio": coh_ratio,
        }
        print(f"  k={k}: within={within:.4f} between={between:.4f} gap={point:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
        print(f"  k={k}: coherence ratio per cluster: {coh_ratio}")

    pca_drift = PCA(n_components=min(10, d_full.shape[1]), random_state=SEED)
    pca_drift.fit(d_full)
    pc1_var_full = float(pca_drift.explained_variance_ratio_[0])
    print(f"  Global drift PCA: PC1 explains {pc1_var_full * 100:.2f}% of displacement variance (full pass)")

    print("\n" + "=" * 80)
    print("STEP 5: Neutralized control")
    print("=" * 80)
    neutral_h0_qa_diff = anchor_report["neutral_h0_batch5_max_diff"]
    print(f"  Neutral h0 batch max diff (from extraction QA): {neutral_h0_qa_diff:.3e} -> treated as a single constant vector")

    ht_neutral_d_full = squareform(pdist(h_t_neutral, metric=LATENT_METRIC)).astype(np.float64)
    neutral_metrics_arrays = compute_metric_arrays(clinical_full_d, ht_neutral_d_full, labels_by_k, KNN_K)
    neutral_scalar = scalarize(neutral_metrics_arrays, PRIMARY_K)
    print("  Neutralized h_t preservation metrics (full cohort) vs clinical space:")
    for m in neutral_scalar:
        print(f"    {m:30s} full={nonresid_scalar[m]:+.4f}  neutral={neutral_scalar[m]:+.4f}  "
              f"delta={neutral_scalar[m] - nonresid_scalar[m]:+.4f}")

    d_neutral = h_t_neutral - h_t_neutral.mean(axis=0, keepdims=True)
    drift_results_neutral = {}
    for k in (PRIMARY_K, EXPLORATORY_K):
        labels = labels_by_k[k]
        within, between, _, _ = within_between_cosine(d_neutral, labels)
        point, lo, hi, excludes_zero = bootstrap_within_between_gap(d_neutral, labels, BOOTSTRAP_N, SEED)
        coh_ratio = coherence_ratio_per_cluster(d_neutral, labels)
        meets_bar = all(v >= COHERENCE_RATIO_BAR for v in coh_ratio.values())
        drift_results_neutral[k] = {
            "within_cosine": within, "between_cosine": between,
            "within_minus_between_gap": {"point": point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": excludes_zero},
            "coherence_ratio": coh_ratio,
            "meets_coherence_bar": meets_bar,
            "decision": bool(excludes_zero and point > 0 and meets_bar),
        }
        print(f"  [neutral] k={k}: within={within:.4f} between={between:.4f} gap={point:+.4f} "
              f"CI=[{lo:+.4f},{hi:+.4f}] coherence_ratio={coh_ratio} meets_bar={meets_bar}")

    pca_drift_neutral = PCA(n_components=min(10, d_neutral.shape[1]), random_state=SEED)
    pca_drift_neutral.fit(d_neutral)
    pc1_var_neutral = float(pca_drift_neutral.explained_variance_ratio_[0])
    print(f"  Global drift PCA (neutral): PC1 explains {pc1_var_neutral * 100:.2f}% of displacement variance")

    print("\n" + "=" * 80)
    print("SAVE: JSON artifacts")
    print("=" * 80)
    with open(f"{STEP3_DIR}/preservation_metrics.json", "w") as f:
        json.dump(preservation_results, f, indent=2)
    with open(f"{STEP3_DIR}/glucose_residualized_sensitivity.json", "w") as f:
        json.dump({"non_residualized": nonresid_scalar, "residualized": resid_scalar}, f, indent=2)
    membership_json = {
        str(k): {kk: vv for kk, vv in v.items() if kk not in ("labels_h0", "labels_ht", "labels_clinical_relabel")}
        for k, v in membership_results.items()
    }
    with open(f"{STEP3_DIR}/membership_flow.json", "w") as f:
        json.dump(membership_json, f, indent=2, default=str)
    drift_json = {
        "full": {str(k): v for k, v in drift_results.items()},
        "neutral": {str(k): v for k, v in drift_results_neutral.items()},
        "global_drift_pc1_variance_explained_full": pc1_var_full,
        "global_drift_pc1_variance_explained_neutral": pc1_var_neutral,
    }
    with open(f"{STEP3_DIR}/coherent_drift.json", "w") as f:
        json.dump(drift_json, f, indent=2)
    print(f"  Wrote preservation_metrics.json, glucose_residualized_sensitivity.json, "
          f"membership_flow.json, coherent_drift.json")

    np.savez_compressed(
        f"{STEP3_DIR}/analysis_arrays.npz",
        participant_id=np.array(retained_pids), split=split_arr,
        cluster_k2=cluster_k2, cluster_k4=cluster_k4,
        d_full=d_full, d_neutral=d_neutral,
        h_t_full=h_t_full, h_t_neutral=h_t_neutral, h0_vecs=h0_vecs,
        pc1_var_full=pca_drift.explained_variance_ratio_,
        pc1_var_neutral=pca_drift_neutral.explained_variance_ratio_,
    )

    print("\nAnalysis stage done (figures generated by phase3_figures.py).")


if __name__ == "__main__":
    main()
