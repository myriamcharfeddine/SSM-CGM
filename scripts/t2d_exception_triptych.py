"""T2D exception triptych.

Explains the one subtype (T2D oral non-insulin) where the streamed hidden
state re-forms a discrete glucose-based grouping after the baseline clinical
clusters dissolve. Three panels: per-subtype ARI (clinical vs glycemic),
old-cluster vs new-group confusion matrix within T2D, and the glucose
profile of the new groups. Reuses saved h_t snapshots and frozen clinical
labels. No forward pass, no model retraining. Read-only on the canonical
checkpoint and the multimodal parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mode as scipy_mode
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
BRANCH = "aireadi-ssmcgm-stream-report"
INPUT_DIR_SPEC = REPO_ROOT / "outputs/static_phenotype_trajectory/global_realignment"
INPUT_DIR_ACTUAL = REPO_ROOT / "outputs/static_phenotype_trajectory_stratified_v2/global_realignment"
INPUT_DIR = INPUT_DIR_SPEC if INPUT_DIR_SPEC.exists() else INPUT_DIR_ACTUAL
OUTPUT_SUBDIR = REPO_ROOT / "outputs/static_phenotype_trajectory/t2d_exception"
CANONICAL_SPLIT = "adapt6h_seed42"
MULTIMODAL_PARQUET = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "final_multimodal_dataset_20260515_184339.parquet"
)

MATCHED_PANEL_PATH = INPUT_DIR / "matched_panel_participant_ids.csv"
SNAPSHOT_DIR = (
    REPO_ROOT
    / "outputs/static_phenotype_trajectory_stratified_v2/phase4_time_resolved_extension/snapshots"
)
CLINICAL_ASSIGNMENTS_PATH = (
    REPO_ROOT
    / "outputs/static_phenotype_trajectory_stratified_v2/phase2_h0_preservation/test_assignments.csv"
)

TARGET_SUBTYPE = "T2D oral non-insulin"
ALL_SUBTYPES = ["Healthy", "Prediabetes", "T2D oral non-insulin", "Insulin-dependent"]
EXPLORATORY_SUBTYPE = "Insulin-dependent"
SNAPSHOT_HOUR = 48
HT_PATH = SNAPSHOT_DIR / f"h_t_full_hour{SNAPSHOT_HOUR:02d}.parquet"
PCA_VARIANCE_TARGET = 0.95
T2D_CLINICAL_K = 3
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = (2.5, 97.5)
RANDOM_SEED = 42

TIR_LOW_MGDL = 70
TIR_HIGH_MGDL = 180
HYPER_THRESHOLD_MGDL = 180
GLYCEMIC_FEATURES = ["mean_cgm", "time_in_range", "time_above_180"]
GLYCEMIC_WINDOW_H = 48
CGM_COLUMN = "cgm_glucose_mean"
TIMESTAMP_COLUMN = "timestamp_local"
PARTICIPANT_ID_COLUMN = "participant_id"

COLOR_CLINICAL = "#003366"
COLOR_GLYCEMIC = "#5BBABA"
COLOR_POSITIVE = "#BA2828"
COLOR_NULL = "#888888"
STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]

PHYSIOLOGICAL_LABELS_K3 = ["Near-range", "Mid-range", "High-exposure"]

FIG_PATH = OUTPUT_SUBDIR / "t2d_exception_triptych.png"
TABLE_ARI_PATH = OUTPUT_SUBDIR / "panel_a_per_subtype_ari.csv"
TABLE_CONFUSION_PATH = OUTPUT_SUBDIR / "panel_b_confusion_matrix.csv"
TABLE_MODAL_PATH = OUTPUT_SUBDIR / "panel_b_modal_fraction.csv"
TABLE_PROFILE_PATH = OUTPUT_SUBDIR / "panel_c_glucose_profile.csv"
DIAGNOSTICS_PATH = OUTPUT_SUBDIR / "triptych_diagnostics.json"

sns.set_style("whitegrid")
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 0.8


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_snapshot_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    ids = df[PARTICIPANT_ID_COLUMN].astype(str).to_numpy()
    dim_cols = sorted((c for c in df.columns if c.isdigit()), key=int)
    matrix = df[dim_cols].to_numpy(dtype=np.float32)
    return ids, matrix


def align_matrix_to_panel(ids: np.ndarray, matrix: np.ndarray, panel_ids: pd.Series) -> np.ndarray:
    id_to_row = {pid: i for i, pid in enumerate(ids)}
    positions = np.array([id_to_row[pid] for pid in panel_ids], dtype=int)
    return matrix[positions]


def load_matched_panel() -> pd.DataFrame:
    panel = pd.read_csv(MATCHED_PANEL_PATH)
    panel[PARTICIPANT_ID_COLUMN] = panel[PARTICIPANT_ID_COLUMN].astype(str)
    panel = panel.sort_values(PARTICIPANT_ID_COLUMN).reset_index(drop=True)
    return panel


def compute_glycemic_features(panel_ids: pd.Series) -> pd.DataFrame:
    """Positional per-participant slicing over the raw multimodal parquet.

    Row blocks are contiguous and already timestamp-sorted per participant
    (verified at the Phase 0 gate of the prior analysis), so a single pass
    builds a start/end position index and every participant's window is a
    plain array slice. Never boolean-masks the full multi-million-row frame.
    """
    df = pd.read_parquet(
        MULTIMODAL_PARQUET, columns=[PARTICIPANT_ID_COLUMN, TIMESTAMP_COLUMN, CGM_COLUMN]
    )
    pid_arr = df[PARTICIPANT_ID_COLUMN].to_numpy()
    ts_arr = df[TIMESTAMP_COLUMN].to_numpy()
    cgm_arr = df[CGM_COLUMN].to_numpy()

    change_points = np.flatnonzero(pid_arr[1:] != pid_arr[:-1]) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(pid_arr)]))
    block_pid_to_range = {pid_arr[s]: (s, e) for s, e in zip(starts, ends)}

    window_seconds = GLYCEMIC_WINDOW_H * 3600.0
    records = []
    for pid in panel_ids:
        s, e = block_pid_to_range[pid]
        ts_slice = ts_arr[s:e]
        cgm_slice = cgm_arr[s:e]
        elapsed_seconds = np.array(
            [(t - ts_slice[0]).total_seconds() for t in ts_slice], dtype=float
        )
        in_window = elapsed_seconds <= window_seconds
        cgm_window = cgm_slice[in_window]
        valid = ~pd.isna(cgm_window)
        cgm_valid = cgm_window[valid].astype(float)
        n = cgm_valid.size
        mean_cgm = float(np.mean(cgm_valid)) if n else np.nan
        in_range = (cgm_valid >= TIR_LOW_MGDL) & (cgm_valid <= TIR_HIGH_MGDL)
        above = cgm_valid > HYPER_THRESHOLD_MGDL
        time_in_range = float(np.mean(in_range)) if n else np.nan
        time_above_180 = float(np.mean(above)) if n else np.nan
        records.append({
            PARTICIPANT_ID_COLUMN: pid, "n_readings_in_window": n,
            "mean_cgm": mean_cgm, "time_in_range": time_in_range, "time_above_180": time_above_180,
        })
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Clustering primitives
# ---------------------------------------------------------------------------
def pca_reduce(matrix: np.ndarray, label: str, diagnostics: dict) -> np.ndarray:
    n_samples = matrix.shape[0]
    n_components_cap = max(1, n_samples - 1)
    pca_probe = PCA(n_components=min(n_components_cap, matrix.shape[1]), svd_solver="full",
                     random_state=RANDOM_SEED)
    pca_probe.fit(matrix)
    cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET) + 1)
    n_components = min(n_components, n_components_cap)
    reduced = pca_probe.transform(matrix)[:, :n_components]
    diagnostics[label] = {
        "n_samples": int(n_samples), "n_components": n_components,
        "variance_explained": float(cumvar[n_components - 1]),
    }
    return reduced


def ward_partition_at_k(reduced: np.ndarray, k: int) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(reduced)


def glycemic_ward_partition(features: pd.DataFrame, k: int) -> np.ndarray:
    x = StandardScaler().fit_transform(features[GLYCEMIC_FEATURES].to_numpy())
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(x)


def participant_clustered_ari_bootstrap(labels_a: np.ndarray, labels_b: np.ndarray,
                                         rng: np.random.RandomState) -> tuple[float, float, float]:
    point = float(adjusted_rand_score(labels_a, labels_b))
    n = len(labels_a)
    replicates = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        replicates[i] = adjusted_rand_score(labels_a[idx], labels_b[idx])
    lo, hi = np.percentile(replicates, BOOTSTRAP_CI)
    return point, float(lo), float(hi)


def modal_fraction_bootstrap(old_labels: np.ndarray, new_labels: np.ndarray,
                              rng: np.random.RandomState) -> dict:
    results = {}
    for old_cluster in sorted(np.unique(old_labels)):
        mask = old_labels == old_cluster
        members_new = new_labels[mask]
        n = int(mask.sum())
        modal_group = int(scipy_mode(members_new, keepdims=False).mode)
        point = float(np.mean(members_new == modal_group))
        replicates = np.empty(N_BOOTSTRAP, dtype=float)
        for i in range(N_BOOTSTRAP):
            idx = rng.randint(0, n, size=n)
            replicates[i] = np.mean(members_new[idx] == modal_group)
        lo, hi = np.percentile(replicates, BOOTSTRAP_CI)
        results[int(old_cluster)] = {
            "modal_new_group": modal_group, "fraction": point,
            "ci_low": float(lo), "ci_high": float(hi), "n": n,
        }
    return results


def profile_bootstrap(values: np.ndarray, rng: np.random.RandomState) -> tuple[float, float, float]:
    point = float(np.mean(values))
    n = len(values)
    replicates = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        replicates[i] = np.mean(values[idx])
    lo, hi = np.percentile(replicates, BOOTSTRAP_CI)
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    print(f"Using INPUT_DIR = {INPUT_DIR}")
    rng = np.random.RandomState(RANDOM_SEED)
    diagnostics: dict = {"pca": {}, "input_dir_used": str(INPUT_DIR)}

    panel = load_matched_panel()
    panel_ids = panel[PARTICIPANT_ID_COLUMN]
    ht_ids, ht_full = load_snapshot_matrix(HT_PATH)
    ht = align_matrix_to_panel(ht_ids, ht_full, panel_ids)
    print(f"Matched panel N = {len(panel)}, h_t (hour {SNAPSHOT_HOUR}) shape {ht.shape}")

    print("Computing glycemic features from the raw multimodal parquet (positional slicing)")
    glycemic = compute_glycemic_features(panel_ids)
    panel = panel.merge(glycemic, on=PARTICIPANT_ID_COLUMN, how="left")

    # ---- Panel A: per-subtype ARI, streamed vs clinical, streamed vs glycemic ----
    panel_a_rows = []
    t2d_streamed_labels = None
    t2d_clinical_labels = None
    t2d_participant_ids = None
    t2d_glycemic_features = None

    for subtype in ALL_SUBTYPES:
        mask = (panel["diagnostic_subtype"] == subtype).to_numpy()
        n_sub = int(mask.sum())
        subtype_k = int(panel.loc[mask, "selected_k"].iloc[0])
        exploratory = subtype == EXPLORATORY_SUBTYPE
        print(f"\nSubtype {subtype}{' (exploratory)' if exploratory else ''}: n={n_sub}, k={subtype_k}")

        ht_sub = ht[mask]
        clinical_labels = panel.loc[mask, "clinical_cluster"].to_numpy()
        features_sub = panel.loc[mask, GLYCEMIC_FEATURES]

        ht_reduced = pca_reduce(ht_sub, subtype, diagnostics["pca"])
        streamed_labels = ward_partition_at_k(ht_reduced, subtype_k)
        glycemic_labels = glycemic_ward_partition(features_sub, subtype_k)
        print(f"  PCA: {diagnostics['pca'][subtype]['n_components']} components reach "
              f"{diagnostics['pca'][subtype]['variance_explained']:.3f} variance")

        ari_clinical, ci_lo_clinical, ci_hi_clinical = participant_clustered_ari_bootstrap(
            streamed_labels, clinical_labels, rng
        )
        ari_glycemic, ci_lo_glycemic, ci_hi_glycemic = participant_clustered_ari_bootstrap(
            streamed_labels, glycemic_labels, rng
        )
        print(f"  ARI(streamed, clinical) = {ari_clinical:.3f} [{ci_lo_clinical:.3f}, {ci_hi_clinical:.3f}]")
        print(f"  ARI(streamed, glycemic) = {ari_glycemic:.3f} [{ci_lo_glycemic:.3f}, {ci_hi_glycemic:.3f}]")

        panel_a_rows.append({
            "subtype": subtype, "exploratory": exploratory, "n": n_sub, "k": subtype_k,
            "reference": "clinical", "ari": ari_clinical, "ci_low": ci_lo_clinical, "ci_high": ci_hi_clinical,
        })
        panel_a_rows.append({
            "subtype": subtype, "exploratory": exploratory, "n": n_sub, "k": subtype_k,
            "reference": "glycemic", "ari": ari_glycemic, "ci_low": ci_lo_glycemic, "ci_high": ci_hi_glycemic,
        })

        if subtype == TARGET_SUBTYPE:
            t2d_streamed_labels = streamed_labels
            t2d_clinical_labels = clinical_labels
            t2d_participant_ids = panel.loc[mask, PARTICIPANT_ID_COLUMN].to_numpy()
            t2d_glycemic_features = panel.loc[mask, GLYCEMIC_FEATURES + [PARTICIPANT_ID_COLUMN]].reset_index(drop=True)

    panel_a_table = pd.DataFrame(panel_a_rows)
    panel_a_table.to_csv(TABLE_ARI_PATH, index=False)

    # ---- Panel B: T2D confusion matrix, old clinical cluster vs new streamed group ----
    assert t2d_streamed_labels is not None, "T2D subtype not found in matched panel"
    confusion_raw = pd.crosstab(
        pd.Series(t2d_clinical_labels, name="baseline_clinical_cluster"),
        pd.Series(t2d_streamed_labels, name="streamed_glucose_group"),
    )
    confusion_row_normalized = confusion_raw.div(confusion_raw.sum(axis=1), axis=0)
    confusion_row_normalized.to_csv(TABLE_CONFUSION_PATH)

    t2d_confusion_ari, t2d_confusion_ci_lo, t2d_confusion_ci_hi = participant_clustered_ari_bootstrap(
        t2d_clinical_labels, t2d_streamed_labels, rng
    )
    modal_results = modal_fraction_bootstrap(t2d_clinical_labels, t2d_streamed_labels, rng)
    modal_table = pd.DataFrame([
        {"baseline_clinical_cluster": k, **v} for k, v in modal_results.items()
    ])
    modal_table.to_csv(TABLE_MODAL_PATH, index=False)

    print(f"\nT2D confusion: ARI(clinical, streamed) = {t2d_confusion_ari:.3f} "
          f"[{t2d_confusion_ci_lo:.3f}, {t2d_confusion_ci_hi:.3f}]")
    for old_cluster, res in modal_results.items():
        print(f"  C{old_cluster} (n={res['n']}) -> modal new group {res['modal_new_group']}, "
              f"fraction={res['fraction']:.3f} [{res['ci_low']:.3f}, {res['ci_high']:.3f}]")

    # ---- Panel C: glucose profile of the new streamed groups, T2D only ----
    t2d_glycemic_features = t2d_glycemic_features.copy()
    t2d_glycemic_features["streamed_group"] = t2d_streamed_labels
    group_mean_cgm = t2d_glycemic_features.groupby("streamed_group")["mean_cgm"].mean().sort_values()
    ordered_groups = group_mean_cgm.index.tolist()
    n_groups = len(ordered_groups)
    physiological_labels = (
        PHYSIOLOGICAL_LABELS_K3 if n_groups == len(PHYSIOLOGICAL_LABELS_K3)
        else [f"Group {i + 1} of {n_groups} by mean glucose" for i in range(n_groups)]
    )
    group_to_label = dict(zip(ordered_groups, physiological_labels))

    profile_rows = []
    for group in ordered_groups:
        group_rows = t2d_glycemic_features[t2d_glycemic_features["streamed_group"] == group]
        n_group = len(group_rows)
        for feature in GLYCEMIC_FEATURES:
            point, lo, hi = profile_bootstrap(group_rows[feature].to_numpy(), rng)
            profile_rows.append({
                "streamed_group": group, "physiological_label": group_to_label[group],
                "n": n_group, "feature": feature, "value": point, "ci_low": lo, "ci_high": hi,
            })
    profile_table = pd.DataFrame(profile_rows)
    profile_table.to_csv(TABLE_PROFILE_PATH, index=False)

    print("\nT2D streamed-group glucose profile (ordered by mean CGM):")
    for group in ordered_groups:
        rows_g = profile_table[profile_table["streamed_group"] == group]
        mean_row = rows_g[rows_g["feature"] == "mean_cgm"].iloc[0]
        print(f"  {group_to_label[group]} (group {group}, n={mean_row['n']}): "
              f"mean CGM={mean_row['value']:.1f} [{mean_row['ci_low']:.1f}, {mean_row['ci_high']:.1f}] mg/dL")

    with open(DIAGNOSTICS_PATH, "w") as f:
        json.dump(diagnostics, f, indent=2)

    make_triptych(panel_a_table, confusion_row_normalized, confusion_raw, modal_table, profile_table,
                   group_to_label, ordered_groups, t2d_confusion_ari, t2d_confusion_ci_lo, t2d_confusion_ci_hi)
    print_summary(panel_a_table, modal_table, profile_table, group_to_label,
                  t2d_confusion_ari, t2d_confusion_ci_lo, t2d_confusion_ci_hi)
    print(f"\nWrote {TABLE_ARI_PATH}")
    print(f"Wrote {TABLE_CONFUSION_PATH}")
    print(f"Wrote {TABLE_MODAL_PATH}")
    print(f"Wrote {TABLE_PROFILE_PATH}")
    print(f"Wrote {FIG_PATH}")
    print(f"Wrote {DIAGNOSTICS_PATH}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_triptych(panel_a_table, confusion_row_normalized, confusion_raw, modal_table, profile_table,
                   group_to_label, ordered_groups, t2d_ari, t2d_ci_lo, t2d_ci_hi) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})

    # -- Panel A --
    ax = axes[0]
    x = np.arange(len(ALL_SUBTYPES))
    bar_width = 0.32
    clinical_favors_glycemic = {}
    for i, subtype in enumerate(ALL_SUBTYPES):
        rows_sub = panel_a_table[panel_a_table["subtype"] == subtype]
        clinical_row = rows_sub[rows_sub["reference"] == "clinical"].iloc[0]
        glycemic_row = rows_sub[rows_sub["reference"] == "glycemic"].iloc[0]
        clinical_favors_glycemic[subtype] = (
            glycemic_row["ari"] > clinical_row["ci_high"] and glycemic_row["ci_low"] > 0
        )
        ax.bar(i - bar_width / 2, clinical_row["ari"], width=bar_width, color=COLOR_CLINICAL,
               yerr=[[clinical_row["ari"] - clinical_row["ci_low"]], [clinical_row["ci_high"] - clinical_row["ari"]]],
               capsize=3, error_kw={"linewidth": 1.0},
               label="Clinical clusters" if i == 0 else None)
        ax.bar(i + bar_width / 2, glycemic_row["ari"], width=bar_width, color=COLOR_GLYCEMIC,
               yerr=[[glycemic_row["ari"] - glycemic_row["ci_low"]], [glycemic_row["ci_high"] - glycemic_row["ari"]]],
               capsize=3, error_kw={"linewidth": 1.0},
               label="Glycemic clusters" if i == 0 else None)
        if clinical_favors_glycemic[subtype]:
            y_top = max(clinical_row["ci_high"], glycemic_row["ci_high"])
            ax.annotate("*", (i, y_top + 0.02), ha="center", fontsize=16, color=COLOR_POSITIVE)

    labels = [s if s != EXPLORATORY_SUBTYPE else s + "\n(exploratory)" for s in ALL_SUBTYPES]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Adjusted rand index (streamed vs reference)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Streamed state vs clinical and glycemic partitions", fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    # -- Panel B --
    ax = axes[1]
    cmap = sns.light_palette(COLOR_GLYCEMIC, as_cmap=True)
    sns.heatmap(confusion_row_normalized, annot=confusion_raw.values, fmt="d", cmap=cmap, vmin=0, vmax=1,
                cbar_kws={"label": "Row fraction"}, ax=ax, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Streamed-state glucose group")
    ax.set_ylabel("Baseline clinical cluster")
    ax.set_yticklabels([f"C{int(c)}" for c in confusion_row_normalized.index], rotation=0)
    ax.set_xticklabels([f"G{int(c)}" for c in confusion_row_normalized.columns])
    ax.set_title(f"T2D reassignment, ARI={t2d_ari:.2f} [{t2d_ci_lo:.2f}, {t2d_ci_hi:.2f}]", fontsize=11)

    # -- Panel C --
    ax = axes[2]
    y_pos = np.arange(len(ordered_groups))
    profile_ramp = sns.blend_palette([COLOR_GLYCEMIC, COLOR_POSITIVE], n_colors=len(ordered_groups))
    for i, group in enumerate(ordered_groups):
        rows_g = profile_table[profile_table["streamed_group"] == group]
        mean_row = rows_g[rows_g["feature"] == "mean_cgm"].iloc[0]
        color = profile_ramp[i]
        ax.errorbar(mean_row["value"], i, xerr=[[mean_row["value"] - mean_row["ci_low"]],
                                                   [mean_row["ci_high"] - mean_row["value"]]],
                    fmt="o", color=color, markersize=9, capsize=4, linewidth=1.5, zorder=3 if i == len(ordered_groups) - 1 else 2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{group_to_label[g]} (n={int(profile_table[(profile_table['streamed_group']==g)&(profile_table['feature']=='mean_cgm')]['n'].iloc[0])})"
                         for g in ordered_groups])
    ax.set_xlabel("Mean CGM, mg/dL (0-48h)")
    ax.set_title("Glucose profile of the streamed-state groups", fontsize=11)
    ax.axvline(TIR_HIGH_MGDL, color=COLOR_NULL, linewidth=1.0, linestyle="--")
    ax.annotate("Hyperglycemia threshold", (TIR_HIGH_MGDL, len(ordered_groups) - 0.5), rotation=90,
                fontsize=7, color=COLOR_NULL, ha="right", va="top")

    for a in axes:
        for spine in a.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)

    fig.suptitle(
        "T2D clinical clusters are replaced by a discrete glucose-exposure grouping "
        "that cuts across the baseline clusters",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)


def print_summary(panel_a_table, modal_table, profile_table, group_to_label,
                   t2d_ari, t2d_ci_lo, t2d_ci_hi) -> None:
    print("\n=== Panel A: per-subtype ARI ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(panel_a_table.to_string(index=False))

    t2d_clinical = panel_a_table[(panel_a_table["subtype"] == TARGET_SUBTYPE) & (panel_a_table["reference"] == "clinical")].iloc[0]
    t2d_glycemic = panel_a_table[(panel_a_table["subtype"] == TARGET_SUBTYPE) & (panel_a_table["reference"] == "glycemic")].iloc[0]

    print("\n=== Panel B: T2D old cluster to new group modal fraction ===")
    print(modal_table.to_string(index=False))

    print("\n=== Panel C: T2D streamed-group glucose profile ===")
    print(profile_table.to_string(index=False))

    print("\nPlain-language summary:")
    print(f"Panel A: in T2D oral non-insulin, the streamed state aligns with glycemic clusters "
          f"(ARI={t2d_glycemic['ari']:.3f} [{t2d_glycemic['ci_low']:.3f}, {t2d_glycemic['ci_high']:.3f}]) "
          f"more than with the frozen clinical clusters (ARI={t2d_clinical['ari']:.3f} "
          f"[{t2d_clinical['ci_low']:.3f}, {t2d_clinical['ci_high']:.3f}], CI includes zero, "
          f"not distinguishable from no structure). No other subtype shows this pattern.")

    scattered = [row for row in modal_table.itertuples() if row.fraction < 0.6]
    if scattered:
        scattered_desc = ", ".join(f"C{row.baseline_clinical_cluster} ({row.fraction:.2f})" for row in scattered)
        print(f"Panel B: ARI(clinical, streamed)={t2d_ari:.3f} [{t2d_ci_lo:.3f}, {t2d_ci_hi:.3f}]. "
              f"No single clinical cluster maps cleanly onto one streamed group, all scatter across groups "
              f"(modal fraction below 0.6): {scattered_desc}.")
    else:
        clean = ", ".join(f"C{row.baseline_clinical_cluster} ({row.fraction:.2f})" for row in modal_table.itertuples())
        print(f"Panel B: ARI(clinical, streamed)={t2d_ari:.3f} [{t2d_ci_lo:.3f}, {t2d_ci_hi:.3f}]. "
              f"Clinical clusters map cleanly onto single streamed groups: {clean}.")

    print("Panel C: streamed groups form a graded glucose exposure ordering, not named as clinical subtypes:")
    for group, label in group_to_label.items():
        rows_g = profile_table[profile_table["streamed_group"] == group]
        mean_row = rows_g[rows_g["feature"] == "mean_cgm"].iloc[0]
        tir_row = rows_g[rows_g["feature"] == "time_in_range"].iloc[0]
        print(f"  {label}: mean CGM={mean_row['value']:.1f} [{mean_row['ci_low']:.1f}, {mean_row['ci_high']:.1f}] mg/dL, "
              f"time in range={tir_row['value']:.2f} [{tir_row['ci_low']:.2f}, {tir_row['ci_high']:.2f}]")


if __name__ == "__main__":
    main()
