"""Phase 1: static reference partition C.

Static-only clinical phenotype partition project. See build prompt for full spec.
Confirmed at Gate A: complete-case rule, applied across all splits (train/val/test);
log1p skewed factors before z-scoring (skew computed on train); k=2 primary,
k=4 exploratory (k=3,5,6 dropped).
"""

import json

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
SEED = 42
FACTORS = ["age", "bmi", "hba1c_baseline", "c_peptide_baseline", "tg_hdl_ratio"]
FACTOR_LABELS = {
    "age": "Age",
    "bmi": "BMI",
    "hba1c_baseline": "HbA1c",
    "c_peptide_baseline": "C-peptide",
    "tg_hdl_ratio": "TG/HDL ratio",
}

K_RANGE = list(range(2, 7))          # 2..6 inclusive, self-check only
PRIMARY_K = 2                        # reported partition
EXPLORATORY_K = 4                    # literature-comparison partition (Ahlqvist)
N_INIT = 25
BOOTSTRAP_N = 1000                   # cluster-stability ARI
GAP_N_REFS = 20
SKEW_LOG_THRESHOLD = 1.0             # abs(train skew) above this -> log1p before z-score
STABILITY_ARI_FLAG_BELOW = 0.5       # flag cluster solution as unstable below this mean ARI

STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
COLOR_POS = "#BA2828"
COLOR_NEG = "#003366"
COLOR_NULL = "#888888"

DATASET = "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet"
SPLIT_PATH = "/home/myriamcharfeddine/CGM/Data/experiment_c_split_adapt6h_seed42/split_participants.csv"
OUTPUT_ROOT = "/home/myriamcharfeddine/CGM/SSM-CGM/outputs/static_phenotype_trajectory"
STEP1_DIR = f"{OUTPUT_ROOT}/step1"
FIG_DIR = f"{OUTPUT_ROOT}/figures"

FACTOR_COLUMN_MAP = {
    "age": "participants_age",
    "bmi": "bmi_baseline",
    "hba1c_baseline": "hba1c_percent_baseline",
    "c_peptide_baseline": "c_peptide_ngml_baseline",
    "triglycerides_baseline": "triglycerides_mgdl_baseline",
    "hdl_cholesterol_baseline": "hdl_cholesterol_mgdl_baseline",
}

sns.set_style("whitegrid", {"axes.edgecolor": "0.3", "grid.color": "0.85"})


def apply_figure_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def gap_statistic(x_scaled, k_range, n_refs, seed):
    rng = np.random.default_rng(seed)
    mins = x_scaled.min(axis=0)
    maxs = x_scaled.max(axis=0)
    gaps, sk = {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed)
        km.fit(x_scaled)
        log_wk = np.log(km.inertia_)
        ref_log_wks = []
        for _ in range(n_refs):
            ref = rng.uniform(mins, maxs, size=x_scaled.shape)
            km_ref = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed)
            km_ref.fit(ref)
            ref_log_wks.append(np.log(km_ref.inertia_))
        ref_log_wks = np.array(ref_log_wks)
        gaps[k] = float(ref_log_wks.mean() - log_wk)
        sk[k] = float(ref_log_wks.std() * np.sqrt(1 + 1.0 / n_refs))
    return gaps, sk


def load_data():
    schema = pq.read_schema(DATASET)
    for col in FACTOR_COLUMN_MAP.values():
        assert col in schema.names, f"Column {col} missing from parquet schema"

    read_cols = ["participant_id", "participants_study_group", "participants_clinical_site", "med_insulin"] + list(
        FACTOR_COLUMN_MAP.values()
    )
    df = pd.read_parquet(DATASET, columns=read_cols)
    df = df.drop_duplicates(subset="participant_id").reset_index(drop=True)
    df = df.rename(columns={v: k for k, v in FACTOR_COLUMN_MAP.items()})
    df["tg_hdl_ratio"] = df["triglycerides_baseline"] / df["hdl_cholesterol_baseline"]

    split_df = pd.read_csv(SPLIT_PATH, dtype={"participant_id": str})
    assert "adapt48h" not in SPLIT_PATH

    merged = df.merge(split_df[["participant_id", "split", "stratum"]], on="participant_id", how="inner")
    return merged


def main():
    print("=" * 80)
    print("PHASE 1 STEP 0: Load, merge, complete-case across ALL splits")
    print("=" * 80)
    merged = load_data()
    split_counts_pre = merged["split"].value_counts().to_dict()

    complete_case_mask = merged[FACTORS].notna().all(axis=1)
    cc = merged[complete_case_mask].copy().reset_index(drop=True)
    split_counts_post = cc["split"].value_counts().to_dict()

    retained_report = {}
    for s in ["train", "val", "test"]:
        pre = int(split_counts_pre.get(s, 0))
        post = int(split_counts_post.get(s, 0))
        frac = post / pre if pre else float("nan")
        retained_report[s] = {"pre_complete_case": pre, "post_complete_case": post, "retained_fraction": frac}
        print(f"  {s:12s}: {post:5d} / {pre:5d} retained ({frac * 100:.2f}%)")

    train = cc[cc["split"] == "train"].copy()
    val = cc[cc["split"] == "val"].copy()
    test = cc[cc["split"] == "test"].copy()
    print(f"\n  Complete-case totals: train={len(train)}, val={len(val)}, test={len(test)}, all={len(cc)}")

    print("\n" + "=" * 80)
    print("PHASE 1 STEP 1: Skew rule and standardization (fit on train only)")
    print("=" * 80)
    skew_report = {}
    log_transformed = []
    for f in FACTORS:
        s_before = float(skew(train[f].values))
        do_log = abs(s_before) > SKEW_LOG_THRESHOLD
        if do_log:
            log_transformed.append(f)
            s_after = float(skew(np.log1p(train[f].values)))
        else:
            s_after = s_before
        skew_report[f] = {"skew_train": s_before, "log_transformed": do_log, "skew_train_after": s_after}
        print(f"  {f:20s} skew={s_before:+.3f}  log1p={'YES' if do_log else 'no'}  skew_after={s_after:+.3f}")

    def transform_factors(frame):
        out = frame[FACTORS].copy()
        for f in log_transformed:
            out[f] = np.log1p(out[f])
        return out.values

    x_train_raw = transform_factors(train)
    x_val_raw = transform_factors(val)
    x_test_raw = transform_factors(test)
    x_all_raw = transform_factors(cc)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    x_test = scaler.transform(x_test_raw)
    x_all = scaler.transform(x_all_raw)

    print("\n" + "=" * 80)
    print("PHASE 1 STEP 2: k-selection self-check on transformed factors")
    print("=" * 80)
    inertia, silhouette = {}, {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED)
        labels = km.fit_predict(x_train)
        inertia[k] = float(km.inertia_)
        silhouette[k] = float(silhouette_score(x_train, labels))
        print(f"  k={k}: inertia={inertia[k]:.2f}  silhouette={silhouette[k]:.4f}")

    gaps, sk = gap_statistic(x_train, K_RANGE, GAP_N_REFS, SEED)
    for k in K_RANGE:
        print(f"  k={k}: gap={gaps[k]:.4f}  s_k={sk[k]:.4f}")

    gap_k_star = None
    for i, k in enumerate(K_RANGE[:-1]):
        k_next = K_RANGE[i + 1]
        if gaps[k] >= gaps[k_next] - sk[k_next]:
            gap_k_star = k
            break
    gap_k_star = gap_k_star or K_RANGE[-1]
    silhouette_k_star = max(silhouette, key=silhouette.get)
    print(f"\n  Gap-selected k: {gap_k_star}  |  Best-silhouette k: {silhouette_k_star}")

    k2_confirmed = gap_k_star == 2 and silhouette_k_star == 2
    if not k2_confirmed:
        print("\n  STOP: k=2 is NOT confirmed as best under the transformed factors.")
        print(f"  Gap picked k={gap_k_star}, silhouette picked k={silhouette_k_star}.")
        with open(f"{STEP1_DIR}/phase1_k_selfcheck_FAILED.json", "w") as f:
            json.dump({"gap": gaps, "gap_sk": sk, "silhouette": silhouette,
                       "gap_selected_k": gap_k_star, "silhouette_selected_k": silhouette_k_star}, f, indent=2)
        raise SystemExit("Halting before saving any partition. See phase1_k_selfcheck_FAILED.json.")
    print("  k=2 self-check PASSED under log-transformed, z-scored factors. Proceeding.")

    silhouette_landscape_note = (
        f"Best silhouette across k=2..6 is {silhouette[2]:.3f} at k=2, and does not exceed "
        f"{max(silhouette.values()):.3f} anywhere in the range; values sag for k=3 "
        f"({silhouette[3]:.3f}) before partially recovering by k=6 ({silhouette[6]:.3f}). "
        "The raw clinical factor space is close to a continuum, not cleanly separated clusters, "
        "even before the model is involved. This bounds how much cluster crispness downstream "
        "geometry-preservation metrics can inherit, but it does not invalidate them: neighbor "
        "purity and drift coherence are well-defined on a continuum."
    )
    print(f"\n  Phase 1 headline: {silhouette_landscape_note}")

    print("\n" + "=" * 80)
    print("PHASE 1 STEP 3: Fit k=2 (primary) and k=4 (exploratory) on train, assign val/test")
    print("=" * 80)
    cluster_results = {}
    for k in (PRIMARY_K, EXPLORATORY_K):
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED)
        train_labels = km.fit_predict(x_train)
        val_labels = km.predict(x_val)
        test_labels = km.predict(x_test)
        all_labels = km.predict(x_all)
        cluster_results[k] = {
            "model": km,
            "train_labels": train_labels,
            "val_labels": val_labels,
            "test_labels": test_labels,
            "all_labels": all_labels,
        }
        sizes = np.bincount(train_labels, minlength=k)
        print(f"  k={k}: train cluster sizes = {sizes.tolist()}")

    print("\n" + "=" * 80)
    print(f"PHASE 1 STEP 4: Cluster stability, {BOOTSTRAP_N} bootstrap resamples of train")
    print("=" * 80)
    stability_report = {}
    rng = np.random.default_rng(SEED)
    n_train = x_train.shape[0]
    for k in (PRIMARY_K, EXPLORATORY_K):
        full_labels = cluster_results[k]["train_labels"]
        aris = []
        for _ in range(BOOTSTRAP_N):
            idx = rng.integers(0, n_train, size=n_train)
            km_b = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED)
            boot_labels = km_b.fit_predict(x_train[idx])
            aris.append(adjusted_rand_score(full_labels[idx], boot_labels))
        mean_ari = float(np.mean(aris))
        std_ari = float(np.std(aris))
        unstable = mean_ari < STABILITY_ARI_FLAG_BELOW
        stability_report[k] = {"mean_ari": mean_ari, "std_ari": std_ari, "unstable": unstable}
        flag = "  <-- FLAGGED UNSTABLE" if unstable else ""
        print(f"  k={k}: mean ARI={mean_ari:.4f}  std={std_ari:.4f}{flag}")

    print("\n" + "=" * 80)
    print("PHASE 1 STEP 5: Save labels, centroids, z-scored matrix, distance matrix, reports")
    print("=" * 80)
    labels_df = pd.DataFrame({"participant_id": cc["participant_id"].values, "split": cc["split"].values})
    # cc's row order interleaves train/val/test; use the "all_labels" prediction (scaler and
    # centroids fit on train, applied to every cc row) so label order matches cc exactly.
    for k in (PRIMARY_K, EXPLORATORY_K):
        labels_df[f"cluster_k{k}"] = cluster_results[k]["all_labels"]
    labels_df.to_parquet(f"{STEP1_DIR}/participant_cluster_labels.parquet", index=False)

    centroids = {str(k): cluster_results[k]["model"].cluster_centers_.tolist() for k in (PRIMARY_K, EXPLORATORY_K)}
    with open(f"{STEP1_DIR}/centroids.json", "w") as f:
        json.dump({"factors_order": FACTORS, "log_transformed_factors": log_transformed, "centroids": centroids}, f, indent=2)

    z_matrix_df = pd.DataFrame(x_all, columns=FACTORS)
    z_matrix_df.insert(0, "participant_id", cc["participant_id"].values)
    z_matrix_df.insert(1, "split", cc["split"].values)
    z_matrix_df.to_parquet(f"{STEP1_DIR}/zscored_factor_matrix.parquet", index=False)

    from scipy.spatial.distance import pdist, squareform
    clinical_dist = squareform(pdist(x_all, metric="euclidean")).astype(np.float32)
    np.savez_compressed(f"{STEP1_DIR}/clinical_pairwise_distance.npz",
                         distance=clinical_dist, participant_id=cc["participant_id"].values)

    coverage_stability_report = {
        "seed": SEED,
        "n_init": N_INIT,
        "factors": FACTORS,
        "skew_log_threshold": SKEW_LOG_THRESHOLD,
        "skew_report": skew_report,
        "log_transformed_factors": log_transformed,
        "complete_case_retained_per_split": retained_report,
        "n_complete_case_total": int(len(cc)),
        "k_selfcheck": {
            "k_range": K_RANGE, "inertia": inertia, "silhouette": silhouette,
            "gap": gaps, "gap_sk": sk, "gap_selected_k": gap_k_star,
            "silhouette_selected_k": silhouette_k_star, "k2_confirmed": k2_confirmed,
        },
        "silhouette_landscape_note": silhouette_landscape_note,
        "primary_k": PRIMARY_K,
        "exploratory_k": EXPLORATORY_K,
        "stability": {str(k): v for k, v in stability_report.items()},
        "stability_ari_flag_below": STABILITY_ARI_FLAG_BELOW,
    }
    with open(f"{STEP1_DIR}/coverage_and_stability_report.json", "w") as f:
        json.dump(coverage_stability_report, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 1 STEP 6: Overlays (interpretation only)")
    print("=" * 80)
    overlay_cc = cc.copy()
    for k in (PRIMARY_K, EXPLORATORY_K):
        overlay_cc[f"cluster_k{k}"] = cluster_results[k]["all_labels"]

    overlay_report = {}
    for k in (PRIMARY_K, EXPLORATORY_K):
        col = f"cluster_k{k}"
        overlay_report[str(k)] = {}
        for overlay_col in ["stratum", "med_insulin", "participants_clinical_site"]:
            ctab = pd.crosstab(overlay_cc[col], overlay_cc[overlay_col], normalize="index")
            print(f"\n  k={k} x {overlay_col} (row-normalized %):")
            print((ctab * 100).round(1).to_string())
            overlay_report[str(k)][overlay_col] = (ctab * 100).round(2).to_dict()
    with open(f"{STEP1_DIR}/cluster_overlay_composition.json", "w") as f:
        json.dump(overlay_report, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 1 FIGURES")
    print("=" * 80)

    # FIG 1: diagnostic, elbow / silhouette / gap over K_RANGE
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ks = K_RANGE
    axes[0].plot(ks, [inertia[k] for k in ks], marker="o", color=COLOR_NEG)
    axes[0].axvline(PRIMARY_K, color=COLOR_POS, linestyle="--", linewidth=1, label=f"Selected k={PRIMARY_K}")
    axes[0].set_title("Elbow: inertia vs k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].legend(frameon=False)

    axes[1].plot(ks, [silhouette[k] for k in ks], marker="o", color=COLOR_NEG)
    axes[1].axvline(PRIMARY_K, color=COLOR_POS, linestyle="--", linewidth=1)
    axes[1].set_title("Silhouette vs k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")

    axes[2].errorbar(ks, [gaps[k] for k in ks], yerr=[sk[k] for k in ks], marker="o", color=COLOR_NEG, capsize=3)
    axes[2].axvline(PRIMARY_K, color=COLOR_POS, linestyle="--", linewidth=1)
    axes[2].set_title("Gap statistic vs k")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Gap")
    for ax in axes:
        apply_figure_frame(ax)
    fig.suptitle("K-selection diagnostics on train, log-transformed and z-scored factors")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig1_k_selection_diagnostics.png", dpi=200)
    plt.close(fig)
    print("  Saved FIG 1: fig1_k_selection_diagnostics.png")

    # FIG 2: 4-panel PCA of z-scored factors, k=2 primary labeling
    pca = PCA(n_components=2, random_state=SEED)
    pcs = pca.fit_transform(x_all)
    var_exp = pca.explained_variance_ratio_

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    ax = axes[0, 0]
    labels_k2 = overlay_cc[f"cluster_k{PRIMARY_K}"].values
    for cl in sorted(np.unique(labels_k2)):
        m = labels_k2 == cl
        ax.scatter(pcs[m, 0], pcs[m, 1], s=10, alpha=0.6,
                   color=STRATUM_COLORS[cl % len(STRATUM_COLORS)], label=f"Cluster {cl}")
    ax.set_title("A: PCA colored by cluster label (k=2)")
    ax.legend(frameon=False, markerscale=2)

    ax = axes[0, 1]
    study_groups = overlay_cc["stratum"].values
    for i, sg in enumerate(sorted(pd.unique(study_groups))):
        m = study_groups == sg
        ax.scatter(pcs[m, 0], pcs[m, 1], s=10, alpha=0.6, color=STRATUM_COLORS[i % len(STRATUM_COLORS)], label=sg)
    ax.set_title("B: PCA colored by study group")
    ax.legend(frameon=False, markerscale=2, fontsize=8)

    ax = axes[1, 0]
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], s=10, c=overlay_cc["hba1c_baseline"].values, cmap="viridis", alpha=0.8)
    ax.set_title("C: PCA colored by HbA1c")
    fig.colorbar(sc, ax=ax, label="HbA1c (%)")

    ax = axes[1, 1]
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], s=10, c=overlay_cc["tg_hdl_ratio"].values, cmap="viridis", alpha=0.8)
    ax.set_title("D: PCA colored by triglyceride-to-HDL ratio")
    fig.colorbar(sc, ax=ax, label="TG/HDL ratio")

    for ax in axes.flat:
        ax.set_xlabel(f"PC1 ({var_exp[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({var_exp[1] * 100:.1f}% var)")
        apply_figure_frame(ax)
    fig.suptitle("Static clinical factor space: cluster label vs known covariates")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig2_clinical_pca_k2.png", dpi=200)
    plt.close(fig)
    print("  Saved FIG 2: fig2_clinical_pca_k2.png")

    # FIG 3: cluster factor profile heatmap, fixed factor order, for both k=2 and k=4
    diverging_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "navy_white_brickred", [COLOR_NEG, "#FFFFFF", COLOR_POS]
    )
    for k in (PRIMARY_K, EXPLORATORY_K):
        col = f"cluster_k{k}"
        z_profile = z_matrix_df.copy()
        z_profile[col] = overlay_cc[col].values
        means = z_profile.groupby(col)[FACTORS].mean().T  # rows=factors (fixed order), cols=clusters
        means = means.reindex(FACTORS)
        means.columns = [f"Cluster {c}" for c in means.columns]
        means.index = [FACTOR_LABELS[f] for f in means.index]

        vmax = float(np.abs(means.values).max())
        fig, ax = plt.subplots(figsize=(1.6 + 1.4 * k, 4.5))
        sns.heatmap(
            means, cmap=diverging_cmap, center=0, vmin=-vmax, vmax=vmax,
            annot=True, fmt=".2f", cbar_kws={"label": "Z-scored factor mean"},
            linewidths=0.5, linecolor="white", ax=ax,
        )
        ax.set_title(f"Cluster factor profile (k={k})")
        ax.set_xlabel("")
        ax.set_ylabel("")
        apply_figure_frame(ax)
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/fig3_cluster_profile_k{k}.png", dpi=200)
        plt.close(fig)
        print(f"  Saved FIG 3 (k={k}): fig3_cluster_profile_k{k}.png")

    print("\nDone. Outputs in:", STEP1_DIR, "and", FIG_DIR)


if __name__ == "__main__":
    main()
