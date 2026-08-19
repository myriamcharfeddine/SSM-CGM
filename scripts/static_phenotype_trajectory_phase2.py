"""Phase 2: is C visible in h0. Reports h0 preservation against a PCA baseline
(the tautology guard), since h0 = psi(g_phi(s_i)) is a deterministic function of
the static profile alone and is expected to preserve clinical structure by
construction.

Static-only clinical phenotype partition project. See build prompt for full spec.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssmcgm.data.aireadi import AireadiFeatureSpec, AireadiPreprocessor  # noqa: E402
from ssmcgm.models.aireadi_stream import AireadiStreamModel, AireadiStreamModelConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
SEED = 42
FACTORS = ["age", "bmi", "hba1c_baseline", "c_peptide_baseline", "tg_hdl_ratio"]
FACTOR_LABELS = {
    "age": "Age", "bmi": "BMI", "hba1c_baseline": "HbA1c",
    "c_peptide_baseline": "C-peptide", "tg_hdl_ratio": "TG/HDL ratio",
}
PRIMARY_K = 2
EXPLORATORY_K = 4
KNN_K = 15
LATENT_METRIC = "cosine"
CLINICAL_METRIC = "euclidean"
MANTEL_PERMUTATIONS = 999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_REFERENCE = "#003366"  # PCA baseline, reference series
COLOR_ADJUSTED = "#5BBABA"   # encoder, adjusted series
STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]

CHECKPOINT = f"{ROOT}/outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/best_model_checkpoint.pt"
EXPECTED_VAL_PINBALL = 3.286316
DATASET = "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet"
SPLIT_PATH = "/home/myriamcharfeddine/CGM/Data/experiment_c_split_adapt6h_seed42/split_participants.csv"
OUTPUT_ROOT = f"{ROOT}/outputs/static_phenotype_trajectory"
STEP1_DIR = f"{OUTPUT_ROOT}/step1"
STEP2_DIR = f"{OUTPUT_ROOT}/step2"
FIG_DIR = f"{OUTPUT_ROOT}/figures"

import seaborn as sns  # noqa: E402
sns.set_style("whitegrid", {"axes.edgecolor": "0.3", "grid.color": "0.85"})


def apply_figure_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def load_model_from_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    md = ckpt["metadata"]
    spec = AireadiFeatureSpec(**md["feature_spec"])
    pre = AireadiPreprocessor.from_jsonable(md["preprocessor"])
    mcfg = AireadiStreamModelConfig(**md["model_config"])
    model = AireadiStreamModel(spec, pre, mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, spec, pre, ckpt


def flatten_participant_state(state, i):
    parts = [x[i].detach().cpu().numpy().reshape(-1) for x in state.layer_states]
    parts += [x[i].detach().cpu().numpy().reshape(-1) for x in (state.conv_states or []) if x is not None]
    return np.concatenate(parts)


def knn_indices(dist_matrix, k):
    n = dist_matrix.shape[0]
    order = np.argsort(dist_matrix, axis=1)
    order = order[:, 1:k + 1] if order.shape[1] > k else order[:, 1:]
    return order


def knn_jaccard(dist_a, dist_b, k):
    n = dist_a.shape[0]
    nn_a = knn_indices(dist_a, k)
    nn_b = knn_indices(dist_b, k)
    scores = np.zeros(n)
    for i in range(n):
        set_a, set_b = set(nn_a[i].tolist()), set(nn_b[i].tolist())
        union = set_a | set_b
        scores[i] = len(set_a & set_b) / len(union) if union else 0.0
    return float(scores.mean())


def trustworthiness_continuity(dist_original, dist_embedded, k):
    n = dist_original.shape[0]
    rank_original = np.argsort(np.argsort(dist_original, axis=1), axis=1)
    rank_embedded = np.argsort(np.argsort(dist_embedded, axis=1), axis=1)
    nn_original = knn_indices(dist_original, k)
    nn_embedded = knn_indices(dist_embedded, k)

    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))

    trust_penalty = 0.0
    for i in range(n):
        embedded_set = set(nn_embedded[i].tolist())
        original_set = set(nn_original[i].tolist())
        intruders = embedded_set - original_set
        for j in intruders:
            trust_penalty += rank_original[i, j] - k
    trustworthiness = 1.0 - norm * trust_penalty

    cont_penalty = 0.0
    for i in range(n):
        embedded_set = set(nn_embedded[i].tolist())
        original_set = set(nn_original[i].tolist())
        extruded = original_set - embedded_set
        for j in extruded:
            cont_penalty += rank_embedded[i, j] - k
    continuity = 1.0 - norm * cont_penalty

    return float(trustworthiness), float(continuity)


def mantel_spearman(dist_a, dist_b, n_perm, seed):
    """Mantel test: Spearman correlation between two distance matrices' upper
    triangles, with a permutation null. Ranks are precomputed once and the
    rank matrix (not the raw distances) is permuted per draw, since permuting
    participant labels leaves the multiset of pairwise values unchanged and
    only reshuffles which pair each rank attaches to -- this avoids an
    O(n^2 log n) rankdata call inside the permutation loop.
    """
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


def neighbor_purity(dist_matrix, labels, k):
    n = dist_matrix.shape[0]
    nn = knn_indices(dist_matrix, k)
    purity = np.zeros(n)
    for i in range(n):
        neighbor_labels = labels[nn[i]]
        purity[i] = np.mean(neighbor_labels == labels[i])
    return purity


def compute_all_metrics(clinical_dist, embed_dist, labels_by_k, k_knn, mantel_perm, seed):
    jaccard = knn_jaccard(clinical_dist, embed_dist, k_knn)
    trust, cont = trustworthiness_continuity(clinical_dist, embed_dist, k_knn)
    rho, p_val = mantel_spearman(clinical_dist, embed_dist, mantel_perm, seed)
    out = {
        "knn_jaccard": jaccard,
        "trustworthiness": trust,
        "continuity": cont,
        "mantel_spearman_rho": rho,
        "mantel_spearman_p": p_val,
    }
    for k, labels in labels_by_k.items():
        purity = neighbor_purity(embed_dist, labels, k_knn)
        out[f"neighbor_purity_k{k}_mean"] = float(purity.mean())
        try:
            sil = silhouette_score(embed_dist, labels, metric="precomputed")
        except ValueError:
            sil = float("nan")
        out[f"cluster_silhouette_k{k}"] = float(sil)
    return out


def main():
    print("=" * 80)
    print("PHASE 2 STEP 0: Verify checkpoint, load model")
    print("=" * 80)
    model, spec, pre, ckpt = load_model_from_checkpoint(CHECKPOINT, DEVICE)
    reported = ckpt["metrics"]["val_pinball_mgdl"]
    if abs(reported - EXPECTED_VAL_PINBALL) > 1e-4:
        raise SystemExit(
            f"STOP: checkpoint val_pinball_mgdl={reported} does not match expected {EXPECTED_VAL_PINBALL}"
        )
    print(f"  Checkpoint val_pinball_mgdl={reported:.6f} confirmed OK")
    print(f"  Device: {DEVICE}")
    print(f"  static_reals ({len(spec.static_reals)}): {spec.static_reals}")
    print(f"  static_categoricals ({len(spec.static_categoricals)}): {spec.static_categoricals}")

    print("\n" + "=" * 80)
    print("PHASE 2 STEP 1: Load Phase 1 complete-case cohort, cluster labels, clinical distances")
    print("=" * 80)
    labels_df = pd.read_parquet(f"{STEP1_DIR}/participant_cluster_labels.parquet")
    z_df = pd.read_parquet(f"{STEP1_DIR}/zscored_factor_matrix.parquet")
    clin = np.load(f"{STEP1_DIR}/clinical_pairwise_distance.npz", allow_pickle=True)
    clinical_dist_full = clin["distance"].astype(np.float64)
    pid_order = clin["participant_id"].astype(str)
    print(f"  Complete-case cohort from Phase 1: {len(pid_order)} participants")

    labels_df["participant_id"] = labels_df["participant_id"].astype(str)
    z_df["participant_id"] = z_df["participant_id"].astype(str)
    strata_df = pd.read_csv(SPLIT_PATH, dtype={"participant_id": str})[["participant_id", "stratum"]]
    order_df = pd.DataFrame({"participant_id": pid_order})
    merged = (
        order_df.merge(labels_df, on="participant_id", how="left")
        .merge(z_df.drop(columns=["split"]), on="participant_id", how="left")
        .merge(strata_df, on="participant_id", how="left")
    )
    assert merged["participant_id"].tolist() == list(pid_order), "Row order mismatch after merge"
    split_arr = merged["split"].values
    cluster_k2 = merged[f"cluster_k{PRIMARY_K}"].values.astype(int)
    cluster_k4 = merged[f"cluster_k{EXPLORATORY_K}"].values.astype(int)
    test_mask = split_arr == "test"
    print(f"  Test subset size: {test_mask.sum()}  |  Full cohort size: {len(pid_order)}")

    print("\n" + "=" * 80)
    print("PHASE 2 STEP 2: Build static tensors and extract h0 per participant")
    print("=" * 80)
    read_cols = ["participant_id"] + spec.static_reals + spec.static_categoricals
    read_cols = list(dict.fromkeys(read_cols))
    raw = pd.read_parquet(DATASET, columns=read_cols)
    raw["participant_id"] = raw["participant_id"].astype(str)
    raw = raw.drop_duplicates(subset="participant_id").set_index("participant_id")
    raw = raw.loc[pid_order]
    print(f"  Static feature rows aligned to complete-case cohort order: {len(raw)}")

    cont_rows, cat_rows = [], []
    for i in range(len(raw)):
        row = raw.iloc[i]
        cont_rows.append(pre.transform_static_reals(row))
        cat_rows.append(pre.transform_static_categoricals(row))
    static_cont = torch.tensor(np.stack(cont_rows), dtype=torch.float32, device=DEVICE)
    static_cat = torch.tensor(np.stack(cat_rows), dtype=torch.long, device=DEVICE)
    print(f"  static_cont shape: {tuple(static_cont.shape)}  static_cat shape: {tuple(static_cat.shape)}")

    with torch.no_grad():
        sctx = model.encode_static(static_cat, static_cont)
        state = model.init_stream(sctx)
    h0_matrix = np.stack([flatten_participant_state(state, i) for i in range(len(raw))])
    print(f"  h0 matrix shape: {h0_matrix.shape}")

    h0_df = pd.DataFrame(h0_matrix)
    h0_df.insert(0, "participant_id", pid_order)
    h0_df.insert(1, "split", split_arr)
    h0_df.to_parquet(f"{STEP2_DIR}/h0_matrix.parquet", index=False)

    print("\n" + "=" * 80)
    print("PHASE 2 STEP 3: PCA baseline embedding (fit on train, same participant order)")
    print("=" * 80)
    train_mask = split_arr == "train"
    z_values = merged[FACTORS].values
    pca_baseline = PCA(n_components=len(FACTORS), random_state=SEED)
    pca_baseline.fit(z_values[train_mask])
    pca_embedding = pca_baseline.transform(z_values)
    print(f"  PCA baseline explained variance ratio: {pca_baseline.explained_variance_ratio_.round(3).tolist()}")

    print("\n" + "=" * 80)
    print("PHASE 2 STEP 4: Preservation metrics, h0 vs PCA baseline, test primary + full sensitivity")
    print("=" * 80)
    labels_by_k_full = {PRIMARY_K: cluster_k2, EXPLORATORY_K: cluster_k4}

    results = {}
    for view_name, mask in [("test", test_mask), ("full_cohort", np.ones_like(test_mask, dtype=bool))]:
        idx = np.where(mask)[0]
        clinical_sub = clinical_dist_full[np.ix_(idx, idx)]
        h0_sub_dist = squareform(pdist(h0_matrix[idx], metric=LATENT_METRIC)).astype(np.float64)
        pca_sub_dist = squareform(pdist(pca_embedding[idx], metric=LATENT_METRIC)).astype(np.float64)
        labels_sub = {k: v[idx] for k, v in labels_by_k_full.items()}

        print(f"\n  --- {view_name} (n={len(idx)}) ---")
        h0_metrics = compute_all_metrics(clinical_sub, h0_sub_dist, labels_sub, KNN_K, MANTEL_PERMUTATIONS, SEED)
        pca_metrics = compute_all_metrics(clinical_sub, pca_sub_dist, labels_sub, KNN_K, MANTEL_PERMUTATIONS, SEED)
        gap = {m: h0_metrics[m] - pca_metrics[m] for m in h0_metrics}
        for m in h0_metrics:
            print(f"    {m:30s}  h0={h0_metrics[m]:+.4f}  pca={pca_metrics[m]:+.4f}  gap={gap[m]:+.4f}")
        results[view_name] = {"h0": h0_metrics, "pca_baseline": pca_metrics, "gap_h0_minus_pca": gap, "n": int(len(idx))}

    with open(f"{STEP2_DIR}/h0_vs_pca_preservation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Wrote {STEP2_DIR}/h0_vs_pca_preservation_metrics.json")

    print("\n" + "=" * 80)
    print("PHASE 2 FIGURES")
    print("=" * 80)

    # FIG 4: tautology guard, grouped bars, test view (primary)
    metric_order = [
        ("knn_jaccard", f"kNN Jaccard (k={KNN_K})"),
        ("trustworthiness", "Trustworthiness"),
        ("continuity", "Continuity"),
        ("mantel_spearman_rho", "Mantel Spearman rho"),
        (f"neighbor_purity_k{PRIMARY_K}_mean", f"Neighbor purity (k={PRIMARY_K})"),
        (f"cluster_silhouette_k{PRIMARY_K}", f"Cluster silhouette (k={PRIMARY_K})"),
    ]
    test_h0 = results["test"]["h0"]
    test_pca = results["test"]["pca_baseline"]
    h0_beats_pca = sum(1 for m, _ in metric_order if test_h0[m] >= test_pca[m])
    title_verdict = (
        f"h0 matches or exceeds the PCA baseline on {h0_beats_pca}/{len(metric_order)} metrics on test"
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(metric_order))
    width = 0.35
    h0_vals = [test_h0[m] for m, _ in metric_order]
    pca_vals = [test_pca[m] for m, _ in metric_order]
    ax.bar(x - width / 2, h0_vals, width, color=COLOR_ADJUSTED, label="h0 (encoder)")
    ax.bar(x + width / 2, pca_vals, width, color=COLOR_REFERENCE, label="PCA baseline")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_order], rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title(f"h0 preservation of the clinical partition vs a PCA baseline: {title_verdict}")
    ax.legend(frameon=False)
    apply_figure_frame(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig4_tautology_guard.png", dpi=200)
    plt.close(fig)
    print("  Saved FIG 4: fig4_tautology_guard.png")

    # FIG 5: four-panel PCA of h0 (visualization only), full cohort
    pca_vis = PCA(n_components=2, random_state=SEED)
    h0_pcs = pca_vis.fit_transform(h0_matrix)
    var_exp = pca_vis.explained_variance_ratio_
    full_purity_k2 = neighbor_purity(
        squareform(pdist(h0_matrix, metric=LATENT_METRIC)).astype(np.float64), cluster_k2, KNN_K
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    ax = axes[0, 0]
    for cl in sorted(np.unique(cluster_k2)):
        m = cluster_k2 == cl
        ax.scatter(h0_pcs[m, 0], h0_pcs[m, 1], s=10, alpha=0.6,
                   color=STRATUM_COLORS[cl % len(STRATUM_COLORS)], label=f"Cluster {cl}")
    ax.set_title(f"A: PCA of h0 colored by cluster label (k={PRIMARY_K})")
    ax.legend(frameon=False, markerscale=2)

    ax = axes[0, 1]
    study_groups = merged["stratum"].values
    for i, sg in enumerate(sorted(pd.unique(study_groups))):
        m = study_groups == sg
        ax.scatter(h0_pcs[m, 0], h0_pcs[m, 1], s=10, alpha=0.6, color=STRATUM_COLORS[i % len(STRATUM_COLORS)], label=sg)
    ax.set_title("B: PCA of h0 colored by study group")
    ax.legend(frameon=False, markerscale=2, fontsize=8)

    ax = axes[1, 0]
    sc = ax.scatter(h0_pcs[:, 0], h0_pcs[:, 1], s=10, c=merged["hba1c_baseline"].values, cmap="viridis", alpha=0.8)
    ax.set_title("C: PCA of h0 colored by HbA1c")
    fig.colorbar(sc, ax=ax, label="HbA1c (z-scored)")

    ax = axes[1, 1]
    sc = ax.scatter(h0_pcs[:, 0], h0_pcs[:, 1], s=10, c=full_purity_k2, cmap="viridis", alpha=0.8)
    ax.set_title(f"D: PCA of h0 colored by neighbor purity (k={PRIMARY_K})")
    fig.colorbar(sc, ax=ax, label="Neighbor purity")

    for ax in axes.flat:
        ax.set_xlabel(f"PC1 ({var_exp[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({var_exp[1] * 100:.1f}% var)")
        apply_figure_frame(ax)
    fig.suptitle("Static-init hidden state h0: cluster label vs known covariates")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig5_h0_pca.png", dpi=200)
    plt.close(fig)
    print("  Saved FIG 5: fig5_h0_pca.png")

    print("\nDone. Outputs in:", STEP2_DIR, "and", FIG_DIR)


if __name__ == "__main__":
    main()
