"""Phase 3, figures + manifest + README. Consumes analysis_arrays.npz and the JSON
artifacts written by static_phenotype_trajectory_phase3_analyze.py.

Static-only clinical phenotype partition project. See build prompt for full spec.
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
from sklearn.decomposition import PCA

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
KNN_K = 15
BOOTSTRAP_N = 1000
LATENT_METRIC = "cosine"
COHERENCE_RATIO_BAR = 0.30
PRIMARY_K = 2
EXPLORATORY_K = 4

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
SPLIT_PATH = "/home/myriamcharfeddine/CGM/Data/experiment_c_split_adapt6h_seed42/split_participants.csv"

sns.set_style("whitegrid", {"axes.edgecolor": "0.3", "grid.color": "0.85"})


def apply_figure_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def bootstrap_within_ci_per_cluster(d_vectors, cluster_labels, n_boot, seed):
    rng = np.random.default_rng(seed)
    cos_full = 1.0 - squareform(pdist(d_vectors, metric="cosine"))
    results = {}
    for c in sorted(np.unique(cluster_labels)):
        idx_c = np.where(cluster_labels == c)[0]
        m = len(idx_c)
        sub_cos = cos_full[np.ix_(idx_c, idx_c)]
        iu = np.triu_indices(m, k=1)
        point = float(sub_cos[iu].mean())
        boots = np.empty(n_boot)
        for b in range(n_boot):
            bidx = rng.integers(0, m, size=m)
            bsub = sub_cos[np.ix_(bidx, bidx)]
            biu = np.triu_indices(m, k=1)
            boots[b] = bsub[biu].mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        results[int(c)] = (point, float(lo), float(hi))
    return results


def main():
    print("=" * 80)
    print("FIGURES: loading analysis arrays and JSON artifacts")
    print("=" * 80)
    arrays = np.load(f"{STEP3_DIR}/analysis_arrays.npz", allow_pickle=True)
    participant_id = arrays["participant_id"]
    split_arr = arrays["split"]
    cluster_k2 = arrays["cluster_k2"]
    cluster_k4 = arrays["cluster_k4"]
    d_full = arrays["d_full"]
    d_neutral = arrays["d_neutral"]
    h_t_full = arrays["h_t_full"]
    h_t_neutral = arrays["h_t_neutral"]
    h0_vecs = arrays["h0_vecs"]

    with open(f"{STEP3_DIR}/preservation_metrics.json") as f:
        preservation = json.load(f)
    with open(f"{STEP3_DIR}/coherent_drift.json") as f:
        drift = json.load(f)
    with open(f"{STEP3_DIR}/membership_flow.json") as f:
        membership = json.load(f)
    with open(f"{STEP3_DIR}/glucose_residualized_sensitivity.json") as f:
        resid_sens = json.load(f)
    with open(f"{STEP3_DIR}/anchor_extraction_report.json") as f:
        anchor_report = json.load(f)

    strata_df = pd.read_csv(SPLIT_PATH, dtype={"participant_id": str})[["participant_id", "stratum"]]
    h0_full_df = pd.read_parquet(f"{STEP2_DIR}/h0_matrix.parquet", columns=["participant_id"])
    z_df = pd.read_parquet(f"{STEP1_DIR}/zscored_factor_matrix.parquet")
    order_df = pd.DataFrame({"participant_id": participant_id.astype(str)})
    strata_aligned = order_df.merge(strata_df, on="participant_id", how="left")["stratum"].to_numpy()
    z_aligned = order_df.merge(z_df, on="participant_id", how="left")
    hba1c_aligned = z_aligned["hba1c_baseline"].to_numpy()

    labels_by_k = {PRIMARY_K: cluster_k2, EXPLORATORY_K: cluster_k4}

    # -----------------------------------------------------------------
    # FIG 6: three-space preservation, test view primary
    # -----------------------------------------------------------------
    print("FIG 6: three-space preservation")
    test_res = preservation["test"]
    metric_order = [
        ("knn_jaccard", f"kNN Jaccard (k={KNN_K})"),
        ("trustworthiness", "Trustworthiness"),
        ("continuity", "Continuity"),
        ("mantel_spearman_rho", "Mantel Spearman rho"),
        (f"neighbor_purity_k{PRIMARY_K}_mean", f"Neighbor purity (k={PRIMARY_K})"),
        (f"cluster_silhouette_k{PRIMARY_K}", f"Cluster silhouette (k={PRIMARY_K})"),
    ]
    h0_vals = [test_res["h0"][m] for m, _ in metric_order]
    pca_vals = [test_res["pca_baseline"].get(m, np.nan) for m, _ in metric_order]
    ht_vals = [test_res["h_t"][m] for m, _ in metric_order]

    purity_recovery = test_res["recovery_decision"]["neighbor_purity"]
    jaccard_recovery = test_res["recovery_decision"]["knn_jaccard"]
    n_recovered = sum(1 for r in (purity_recovery, jaccard_recovery) if r["recovery_holds"])
    if n_recovered == 2:
        verdict = "the stream recovers clinical structure the encoder had diluted (both recovery metrics improve)"
    elif n_recovered == 1:
        verdict = "the stream partially recovers clinical structure (one of two recovery metrics improves)"
    else:
        verdict = "the stream does not recover clinical structure lost at h0 (replicates the continuum thesis)"

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_order))
    width = 0.26
    ax.bar(x - width, h0_vals, width, color=COLOR_H0, label="h0 (floor)")
    ax.bar(x, pca_vals, width, color=COLOR_PCA, label="PCA baseline")
    ax.bar(x + width, ht_vals, width, color=COLOR_HT, label="h_t", zorder=3)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_order], rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    gap_p = purity_recovery["gap_ht_minus_h0"]
    gap_j = jaccard_recovery["gap_ht_minus_h0"]
    ax.set_title(
        f"Three-space preservation on test: {verdict}\n"
        f"h_t-h0 gap: purity {gap_p:+.3f} (CI excl. 0: {purity_recovery['recovery_holds']}), "
        f"Jaccard {gap_j:+.3f} (CI excl. 0: {jaccard_recovery['recovery_holds']})"
    )
    ax.legend(frameon=False)
    apply_figure_frame(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig6_three_space_preservation.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # FIG 7: four-panel PCA of h_t summary
    # -----------------------------------------------------------------
    print("FIG 7: PCA of h_t summary")
    pca_ht = PCA(n_components=2, random_state=SEED)
    ht_pcs = pca_ht.fit_transform(h_t_full)
    var_exp = pca_ht.explained_variance_ratio_
    drift_mag = np.linalg.norm(d_full, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    ax = axes[0, 0]
    for cl in sorted(np.unique(cluster_k2)):
        m = cluster_k2 == cl
        ax.scatter(ht_pcs[m, 0], ht_pcs[m, 1], s=10, alpha=0.6,
                   color=STRATUM_COLORS[cl % len(STRATUM_COLORS)], label=f"Cluster {cl}")
    ax.set_title(f"A: PCA of h_t colored by cluster label (k={PRIMARY_K})")
    ax.legend(frameon=False, markerscale=2)

    ax = axes[0, 1]
    for i, sg in enumerate(sorted(pd.unique(strata_aligned))):
        m = strata_aligned == sg
        ax.scatter(ht_pcs[m, 0], ht_pcs[m, 1], s=10, alpha=0.6, color=STRATUM_COLORS[i % len(STRATUM_COLORS)], label=sg)
    ax.set_title("B: PCA of h_t colored by study group")
    ax.legend(frameon=False, markerscale=2, fontsize=8)

    ax = axes[1, 0]
    sc = ax.scatter(ht_pcs[:, 0], ht_pcs[:, 1], s=10, c=hba1c_aligned, cmap="viridis", alpha=0.8)
    ax.set_title("C: PCA of h_t colored by HbA1c")
    fig.colorbar(sc, ax=ax, label="HbA1c (z-scored)")

    ax = axes[1, 1]
    sc = ax.scatter(ht_pcs[:, 0], ht_pcs[:, 1], s=10, c=drift_mag, cmap="viridis", alpha=0.8)
    ax.set_title("D: PCA of h_t colored by drift magnitude")
    fig.colorbar(sc, ax=ax, label="||h_t - h0||")

    for ax in axes.flat:
        ax.set_xlabel(f"PC1 ({var_exp[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({var_exp[1] * 100:.1f}% var)")
        apply_figure_frame(ax)
    fig.suptitle("Overnight streaming state h_t: cluster label vs known covariates and drift")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig7_ht_pca.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # FIG 8: money figure
    # -----------------------------------------------------------------
    print("FIG 8: money figure")
    within_ci_k2 = bootstrap_within_ci_per_cluster(d_full, cluster_k2, BOOTSTRAP_N, SEED)
    between_full_k2 = drift["full"][str(PRIMARY_K)]["between_cosine"]
    coh_full = drift["full"][str(PRIMARY_K)]["coherence_ratio"]
    coh_neutral = drift["neutral"][str(PRIMARY_K)]["coherence_ratio"]

    # h0 and h_t live at very different scales/locations (one is a static-only reset
    # state, the other has evolved through a full night of stream updates), so a PCA
    # fit on their concatenation is dominated by "which cloud a point belongs to" and
    # produces an unreadable hairball. Instead, project each participant's own
    # displacement vector d_i from the origin, using the same PCA already needed for
    # the scree plot -- this directly shows whether same-cluster arrows share a
    # direction, which is what "coherent drift" actually means.
    pca_scree = PCA(n_components=10, random_state=SEED)
    d_pcs = pca_scree.fit_transform(d_full)
    scree = pca_scree.explained_variance_ratio_
    d_2d = d_pcs[:, :2]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    ax = axes[0, 0]
    clusters_sorted = sorted(within_ci_k2.keys())
    for i, c in enumerate(clusters_sorted):
        point, lo, hi = within_ci_k2[c]
        ax.bar(i, point, color=COLOR_POS, width=0.5, zorder=3)
        ax.errorbar(i, point, yerr=[[point - lo], [hi - point]], color="black", capsize=4, zorder=4)
    ax.axhline(between_full_k2, color=COLOR_NEG, linestyle="--", linewidth=1.5, label="Between-cluster cosine (pooled)")
    ax.set_xticks(range(len(clusters_sorted)))
    ax.set_xticklabels([f"Cluster {c}" for c in clusters_sorted])
    ax.set_ylabel("Mean pairwise cosine of displacement")
    ax.set_title(f"A: Within-cluster displacement cosine per cluster (k={PRIMARY_K})")
    ax.legend(frameon=False)
    apply_figure_frame(ax)

    ax = axes[0, 1]
    clusters_c = sorted(coh_full.keys(), key=lambda x: int(x))
    x = np.arange(len(clusters_c))
    width = 0.35
    ax.bar(x - width / 2, [coh_full[c] for c in clusters_c], width, color=COLOR_HT, label="Full")
    ax.bar(x + width / 2, [coh_neutral[c] for c in clusters_c], width, color=COLOR_NULL, label="Neutral")
    ax.axhline(COHERENCE_RATIO_BAR, color=COLOR_EVENT, linestyle="--", linewidth=1.5, label="Pre-registered bar")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in clusters_c])
    ax.set_ylabel("Coherence ratio")
    ax.set_title(f"B: Coherence ratio, full vs neutral (k={PRIMARY_K})")
    ax.legend(frameon=False)
    apply_figure_frame(ax)

    ax = axes[1, 0]
    ax.bar(np.arange(1, len(scree) + 1), scree, color=COLOR_HT)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained")
    ax.set_title("C: Global drift PCA scree (full pass)")
    apply_figure_frame(ax)

    ax = axes[1, 1]
    origin = np.zeros_like(d_2d)
    for cl in sorted(np.unique(cluster_k2)):
        m = cluster_k2 == cl
        ax.quiver(origin[m, 0], origin[m, 1], d_2d[m, 0], d_2d[m, 1],
                  angles="xy", scale_units="xy", scale=1, width=0.0015, alpha=0.35,
                  color=STRATUM_COLORS[cl % len(STRATUM_COLORS)], label=f"Cluster {cl}")
    ax.set_xlabel(f"Drift PC1 ({scree[0] * 100:.1f}% var)")
    ax.set_ylabel(f"Drift PC2 ({scree[1] * 100:.1f}% var)")
    ax.set_title(f"D: h0 to h_t displacement direction from origin (k={PRIMARY_K})")
    ax.legend(frameon=False, markerscale=2)
    apply_figure_frame(ax)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig8_money_figure.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # FIG 9: full vs neutral delta
    # -----------------------------------------------------------------
    print("FIG 9: full vs neutral delta")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    deltas_coh = [coh_full[c] - coh_neutral[c] for c in clusters_c]
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in deltas_coh]
    order = np.argsort(np.abs(deltas_coh))[::-1]
    ax.barh([f"Cluster {clusters_c[i]}" for i in order], [deltas_coh[i] for i in order],
            color=[colors[i] for i in order])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Coherence ratio delta (full minus neutral)")
    ax.set_title("Coherence ratio: what the stream adds")
    apply_figure_frame(ax)

    ht_dist = squareform(pdist(h_t_full, metric=LATENT_METRIC))
    neutral_dist = squareform(pdist(h_t_neutral, metric=LATENT_METRIC))
    def knn_purity(dist, labels, k):
        order_ = np.argsort(dist, axis=1)[:, 1:k + 1]
        return np.array([np.mean(labels[order_[i]] == labels[i]) for i in range(len(labels))])
    purity_full_arr = knn_purity(ht_dist, cluster_k2, KNN_K)
    purity_neutral_arr = knn_purity(neutral_dist, cluster_k2, KNN_K)
    purity_delta = [purity_full_arr[cluster_k2 == int(c)].mean() - purity_neutral_arr[cluster_k2 == int(c)].mean()
                    for c in clusters_c]
    ax = axes[1]
    colors2 = [COLOR_POS if v >= 0 else COLOR_NEG for v in purity_delta]
    order2 = np.argsort(np.abs(purity_delta))[::-1]
    ax.barh([f"Cluster {clusters_c[i]}" for i in order2], [purity_delta[i] for i in order2],
            color=[colors2[i] for i in order2])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(f"Neighbor purity delta (full minus neutral, k={PRIMARY_K})")
    ax.set_title("Neighbor purity: what the stream adds")
    apply_figure_frame(ax)

    fig.suptitle("Full pass vs neutralized pass: isolating the stream's contribution")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig9_full_vs_neutral_delta.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # FIG 10: membership flow
    # -----------------------------------------------------------------
    print("FIG 10: membership flow")
    def plot_membership(k, ax_heat1, ax_heat2, ax_stack):
        mk = membership[str(k)]
        labels_ = mk["cluster_labels"]
        trans1 = np.array(mk["transition_clinical_to_h0"])
        trans2 = np.array(mk["transition_h0_to_ht"])
        trans1_norm = trans1 / trans1.sum(axis=1, keepdims=True)
        trans2_norm = trans2 / trans2.sum(axis=1, keepdims=True)

        sns.heatmap(trans1_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=[f"h0 {c}" for c in labels_], yticklabels=[f"Clinical {c}" for c in labels_], ax=ax_heat1)
        ax_heat1.set_title(f"Clinical to h0 (k={k})")

        sns.heatmap(trans2_norm, annot=True, fmt=".2f", cmap="Greens", cbar=False,
                    xticklabels=[f"h_t {c}" for c in labels_], yticklabels=[f"h0 {c}" for c in labels_], ax=ax_heat2)
        ax_heat2.set_title(f"h0 to h_t (k={k})")

        bottom = np.zeros(len(labels_))
        for j, c in enumerate(labels_):
            ax_stack.bar([f"{c}" for c in labels_], trans2_norm[:, j], bottom=bottom,
                        color=STRATUM_COLORS[j % len(STRATUM_COLORS)], label=f"-> h_t {c}")
            bottom += trans2_norm[:, j]
        ax_stack.set_xlabel("Source h0 cluster")
        ax_stack.set_ylabel("Destination share")
        ax_stack.set_title(f"h0 to h_t destination shares (k={k})")
        ax_stack.legend(frameon=False, fontsize=7)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plot_membership(PRIMARY_K, axes[0, 0], axes[0, 1], axes[0, 2])
    plot_membership(EXPLORATORY_K, axes[1, 0], axes[1, 1], axes[1, 2])
    for ax in axes.flat:
        apply_figure_frame(ax)
    fig.suptitle(f"Membership flow: k={PRIMARY_K} primary (top), k={EXPLORATORY_K} exploratory (bottom, supplementary)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig10_membership_flow.png", dpi=200)
    plt.close(fig)

    print("\nAll Phase 3 figures saved to", FIG_DIR)

    return {
        "verdict": verdict,
        "n_recovered": n_recovered,
        "gap_purity": gap_p,
        "gap_jaccard": gap_j,
        "purity_recovery": purity_recovery,
        "jaccard_recovery": jaccard_recovery,
    }


if __name__ == "__main__":
    main()
