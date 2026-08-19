"""Phase 1: ARI realignment cross tab.

Tests whether the streamed hidden state h_t aligns with a glycemic-state
partition more than with the baseline clinical partition, while the coarse
diagnostic axis is comparatively retained. Reuses saved h0 and h_t snapshots
and frozen clinical/diagnostic labels. No forward pass, no model retraining.
Read-only on the canonical checkpoint and the multimodal parquet.
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
BRANCH = "aireadi-ssmcgm-stream-report"
OUTPUT_ROOT = REPO_ROOT / "outputs/static_phenotype_trajectory"
OUTPUT_SUBDIR = REPO_ROOT / "outputs/static_phenotype_trajectory/global_realignment"
CANONICAL_SPLIT = "adapt6h_seed42"
MULTIMODAL_PARQUET = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "final_multimodal_dataset_20260515_184339.parquet"
)

SNAPSHOT_HOURS = [0, 6, 12, 24, 48]
H0_PATH = OUTPUT_ROOT / "step2/h0_matrix.parquet"
SNAPSHOT_DIR = (
    REPO_ROOT
    / "outputs/static_phenotype_trajectory_stratified_v2/phase4_time_resolved_extension/snapshots"
)
HT_ANALYSIS_HOUR = 48
HT_PATH = SNAPSHOT_DIR / "h_t_full_hour48.parquet"
CLINICAL_ASSIGNMENTS_PATH = (
    REPO_ROOT
    / "outputs/static_phenotype_trajectory_stratified_v2/phase2_h0_preservation/test_assignments.csv"
)

CLINICAL_FACTORS = [
    "study_visit_age", "bmi", "hba1c", "c_peptide", "tg_hdl_ratio", "waist_to_hip_ratio",
]
FACTOR_COLUMN_MAP = {
    "study_visit_age": "participants_age",
    "bmi": "bmi_baseline",
    "hba1c": "hba1c_percent_baseline",
    "c_peptide": "c_peptide_ngml_baseline",
    "waist_to_hip_ratio": "waist_to_hip_ratio_baseline",
}
TG_COLUMN = "triglycerides_mgdl_baseline"
HDL_COLUMN = "hdl_cholesterol_mgdl_baseline"

DIAGNOSTIC_SUBTYPES = ["Healthy", "Prediabetes", "T2D oral non-insulin", "Insulin-dependent"]
STRATUM_KEY_TO_DISPLAY = {
    "healthy": "Healthy",
    "pre_diabetes": "Prediabetes",
    "t2d_oral_non_insulin": "T2D oral non-insulin",
    "insulin_dependent": "Insulin-dependent",
}
EXPLORATORY_SUBTYPE = "Insulin-dependent"
FROZEN_K_PER_SUBTYPE = {
    "Healthy": 2, "Prediabetes": 2, "T2D oral non-insulin": 3, "Insulin-dependent": 3,
}

PCA_VARIANCE_TARGET = 0.95
K_CANDIDATES = [2, 3, 4, 5, 6]
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = (2.5, 97.5)
RANDOM_SEED = 42

TIR_LOW_MGDL = 70
TIR_HIGH_MGDL = 180
HYPER_THRESHOLD_MGDL = 180
GLYCEMIC_FEATURES = ["mean_cgm", "time_in_range", "time_above_180"]
GLYCEMIC_WINDOW_HOURS = 48
CGM_COLUMN = "cgm_glucose_mean"
TIMESTAMP_COLUMN = "timestamp_local"
PARTICIPANT_ID_COLUMN = "participant_id"

STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
COLOR_H0 = "#003366"
COLOR_HT = "#5BBABA"
COLOR_CONTROL = "#888888"

REFERENCE_PARTITIONS = [
    "diagnostic subtype (global)",
    "clinical clusters (within subtype)",
    "glycemic clusters (within subtype)",
]

FIG1_PATH = OUTPUT_SUBDIR / "ari_realignment_crosstab.png"
TABLE1_PATH = OUTPUT_SUBDIR / "ari_realignment_crosstab.csv"
DIAGNOSTICS_PATH = OUTPUT_SUBDIR / "phase1_pca_and_k_selection.json"
MATCHED_PANEL_PATH = OUTPUT_SUBDIR / "matched_panel_participant_ids.csv"

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


def build_matched_panel() -> pd.DataFrame:
    h0_full = pd.read_parquet(H0_PATH, columns=[PARTICIPANT_ID_COLUMN, "split"])
    ht_full = pd.read_parquet(HT_PATH, columns=[PARTICIPANT_ID_COLUMN, "split"])
    assert list(h0_full[PARTICIPANT_ID_COLUMN]) == list(ht_full[PARTICIPANT_ID_COLUMN]), \
        "h0 and h_t snapshot ordering diverged"
    test_ids = h0_full.loc[h0_full["split"] == "test", PARTICIPANT_ID_COLUMN].astype(str)
    snapshot_set = set(test_ids)

    assignments = pd.read_csv(CLINICAL_ASSIGNMENTS_PATH)
    assignments[PARTICIPANT_ID_COLUMN] = assignments[PARTICIPANT_ID_COLUMN].astype(str)
    assignments["diagnostic_subtype"] = assignments["canonical_stratum"].map(STRATUM_KEY_TO_DISPLAY)
    assert assignments["diagnostic_subtype"].isna().sum() == 0, "unmapped canonical_stratum value"

    matched = assignments[assignments[PARTICIPANT_ID_COLUMN].isin(snapshot_set)].copy()
    matched = matched.rename(columns={"display_cluster": "clinical_cluster"})
    matched = matched[[PARTICIPANT_ID_COLUMN, "diagnostic_subtype", "clinical_cluster", "selected_k"]]
    matched = matched.sort_values(PARTICIPANT_ID_COLUMN).reset_index(drop=True)

    dropped_from_snapshots = snapshot_set - set(matched[PARTICIPANT_ID_COLUMN])
    dropped_from_assignments = set(assignments[PARTICIPANT_ID_COLUMN]) - snapshot_set
    print(f"Matched panel N = {len(matched)} (snapshot test N = {len(snapshot_set)}, "
          f"clinical assignment N = {len(assignments)})")
    print(f"  Dropped, present in snapshots but not clinical assignments ({len(dropped_from_snapshots)}): "
          f"{sorted(dropped_from_snapshots)}")
    print(f"  Dropped, present in clinical assignments but not snapshots ({len(dropped_from_assignments)}): "
          f"{sorted(dropped_from_assignments)}")
    return matched


def align_matrix_to_panel(ids: np.ndarray, matrix: np.ndarray, panel_ids: pd.Series) -> np.ndarray:
    id_to_row = {pid: i for i, pid in enumerate(ids)}
    positions = np.array([id_to_row[pid] for pid in panel_ids], dtype=int)
    return matrix[positions]


def compute_glycemic_features(panel_ids: pd.Series) -> pd.DataFrame:
    """Positional per-participant slicing over the raw multimodal parquet.

    Row blocks are contiguous and already timestamp-sorted per participant
    (verified at the Phase 0 gate), so a single pass builds a start/end
    position index and every participant's window is a plain array slice.
    Never boolean-masks the full 4.5M-row frame.
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

    window_seconds = GLYCEMIC_WINDOW_HOURS * 3600.0
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
            PARTICIPANT_ID_COLUMN: pid,
            "n_readings_in_window": n,
            "mean_cgm": mean_cgm,
            "time_in_range": time_in_range,
            "time_above_180": time_above_180,
        })
    out = pd.DataFrame.from_records(records)
    missing = out[GLYCEMIC_FEATURES].isna().any(axis=1).sum()
    if missing:
        print(f"  Warning: {missing} participants have no CGM readings in the 0-{GLYCEMIC_WINDOW_HOURS}h window")
    return out


def compute_clinical_factor_table(panel_ids: pd.Series) -> pd.DataFrame:
    cols = list(FACTOR_COLUMN_MAP.values()) + [TG_COLUMN, HDL_COLUMN]
    df = pd.read_parquet(MULTIMODAL_PARQUET, columns=[PARTICIPANT_ID_COLUMN] + cols)
    df = df.drop_duplicates(subset=PARTICIPANT_ID_COLUMN, keep="first").set_index(PARTICIPANT_ID_COLUMN)
    df = df.loc[panel_ids]
    out = pd.DataFrame(index=panel_ids)
    for factor, col in FACTOR_COLUMN_MAP.items():
        out[factor] = df[col].to_numpy()
    out["tg_hdl_ratio"] = (df[TG_COLUMN] / df[HDL_COLUMN]).to_numpy()
    out = out[CLINICAL_FACTORS]
    return out.reset_index()


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
        "n_samples": int(n_samples),
        "n_components": n_components,
        "variance_explained": float(cumvar[n_components - 1]),
        "variance_target": PCA_VARIANCE_TARGET,
    }
    print(f"  PCA[{label}]: n={n_samples}, {n_components} components reach "
          f"{cumvar[n_components - 1]:.3f} variance (target {PCA_VARIANCE_TARGET})")
    return reduced


def ward_partition_by_silhouette(reduced: np.ndarray, label: str, diagnostics: dict) -> np.ndarray:
    n_samples = reduced.shape[0]
    feasible_k = [k for k in K_CANDIDATES if 2 <= k <= n_samples - 1]
    scores = {}
    labels_by_k = {}
    for k in feasible_k:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(reduced)
        scores[k] = float(silhouette_score(reduced, labels))
        labels_by_k[k] = labels
    best_k = max(scores, key=scores.get)
    diagnostics[label] = {"feasible_k": feasible_k, "silhouette_by_k": scores, "selected_k": best_k}
    print(f"  Ward[{label}]: selected k={best_k} (silhouette={scores[best_k]:.4f}) "
          f"over candidates {feasible_k}")
    return labels_by_k[best_k]


def glycemic_ward_partition(features: pd.DataFrame, k: int) -> np.ndarray:
    x = StandardScaler().fit_transform(features[GLYCEMIC_FEATURES].to_numpy())
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(x)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
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


def pooled_ari_bootstrap(per_subtype_labels: dict, subtypes: list[str],
                          rng: np.random.RandomState) -> tuple[float, float, float]:
    """Participant-count-weighted pooled ARI across the given subtypes.

    Pools by weighted-averaging per-subtype ARI rather than concatenating
    cluster labels across subtypes, since cross-subtype pairs are trivially
    "different cluster" under both partitions and would otherwise inflate
    agreement. Bootstrap resamples participants independently within each
    subtype (participant-clustered) and recombines with fixed weights.
    """
    weights = np.array([len(per_subtype_labels[s][0]) for s in subtypes], dtype=float)
    weights = weights / weights.sum()
    points = np.array([
        adjusted_rand_score(per_subtype_labels[s][0], per_subtype_labels[s][1]) for s in subtypes
    ])
    point = float(np.dot(weights, points))

    replicates = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        rep_vals = np.empty(len(subtypes), dtype=float)
        for j, s in enumerate(subtypes):
            a, b = per_subtype_labels[s]
            n = len(a)
            idx = rng.randint(0, n, size=n)
            rep_vals[j] = adjusted_rand_score(a[idx], b[idx])
        replicates[i] = np.dot(weights, rep_vals)
    lo, hi = np.percentile(replicates, BOOTSTRAP_CI)
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    diagnostics: dict = {"pca": {}, "ward_k_selection": {}}
    rng = np.random.RandomState(RANDOM_SEED)

    print("Building matched panel")
    panel = build_matched_panel()
    panel.to_csv(MATCHED_PANEL_PATH, index=False)
    panel_ids = panel[PARTICIPANT_ID_COLUMN]

    print("Loading h0 and h_t snapshots for the matched panel")
    h0_ids, h0_full = load_snapshot_matrix(H0_PATH)
    ht_ids, ht_full = load_snapshot_matrix(HT_PATH)
    h0 = align_matrix_to_panel(h0_ids, h0_full, panel_ids)
    ht = align_matrix_to_panel(ht_ids, ht_full, panel_ids)
    print(f"  h0 shape {h0.shape}, h_t (hour {HT_ANALYSIS_HOUR}) shape {ht.shape}")

    print("Computing glycemic features from the raw multimodal parquet (positional slicing)")
    glycemic = compute_glycemic_features(panel_ids)
    panel = panel.merge(glycemic, on=PARTICIPANT_ID_COLUMN, how="left")

    rows = []

    # --- Global level ---
    print("\nGlobal level: PCA + Ward on pooled test panel")
    h0_reduced_global = pca_reduce(h0, "global_h0", diagnostics["pca"])
    ht_reduced_global = pca_reduce(ht, "global_ht", diagnostics["pca"])
    h0_part_global = ward_partition_by_silhouette(h0_reduced_global, "global_h0", diagnostics["ward_k_selection"])
    ht_part_global = ward_partition_by_silhouette(ht_reduced_global, "global_ht", diagnostics["ward_k_selection"])

    diag_codes = panel["diagnostic_subtype"].astype("category").cat.codes.to_numpy()
    for rep_name, rep_part in [("h0", h0_part_global), ("h_t", ht_part_global)]:
        point, lo, hi = participant_clustered_ari_bootstrap(rep_part, diag_codes, rng)
        rows.append({
            "level": "global", "subtype": "all", "reference_partition": "diagnostic subtype (global)",
            "representation": rep_name, "ari": point, "ci_low": lo, "ci_high": hi, "n": len(panel),
        })

    # --- Within-subtype level ---
    within_rep_clinical = {"h0": {}, "h_t": {}}
    within_rep_glycemic = {"h0": {}, "h_t": {}}
    within_clinical_glycemic = {}

    for subtype in DIAGNOSTIC_SUBTYPES:
        mask = (panel["diagnostic_subtype"] == subtype).to_numpy()
        n_sub = int(mask.sum())
        exploratory = subtype == EXPLORATORY_SUBTYPE
        tag = " (exploratory)" if exploratory else ""
        print(f"\nSubtype {subtype}{tag}: n={n_sub}")

        h0_sub = h0[mask]
        ht_sub = ht[mask]
        clinical_labels = panel.loc[mask, "clinical_cluster"].to_numpy()
        glycemic_features_sub = panel.loc[mask, GLYCEMIC_FEATURES + [PARTICIPANT_ID_COLUMN]]
        k_frozen = FROZEN_K_PER_SUBTYPE[subtype]
        glycemic_labels = glycemic_ward_partition(glycemic_features_sub, k_frozen)

        h0_reduced = pca_reduce(h0_sub, f"{subtype}_h0", diagnostics["pca"])
        ht_reduced = pca_reduce(ht_sub, f"{subtype}_ht", diagnostics["pca"])
        h0_part = ward_partition_by_silhouette(h0_reduced, f"{subtype}_h0", diagnostics["ward_k_selection"])
        ht_part = ward_partition_by_silhouette(ht_reduced, f"{subtype}_ht", diagnostics["ward_k_selection"])

        for rep_name, rep_part in [("h0", h0_part), ("h_t", ht_part)]:
            point, lo, hi = participant_clustered_ari_bootstrap(rep_part, clinical_labels, rng)
            rows.append({
                "level": "within_subtype", "subtype": subtype + tag,
                "reference_partition": "clinical clusters (within subtype)",
                "representation": rep_name, "ari": point, "ci_low": lo, "ci_high": hi, "n": n_sub,
            })
            within_rep_clinical[rep_name][subtype] = (rep_part, clinical_labels)

            point, lo, hi = participant_clustered_ari_bootstrap(rep_part, glycemic_labels, rng)
            rows.append({
                "level": "within_subtype", "subtype": subtype + tag,
                "reference_partition": "glycemic clusters (within subtype)",
                "representation": rep_name, "ari": point, "ci_low": lo, "ci_high": hi, "n": n_sub,
            })
            within_rep_glycemic[rep_name][subtype] = (rep_part, glycemic_labels)

        point, lo, hi = participant_clustered_ari_bootstrap(clinical_labels, glycemic_labels, rng)
        rows.append({
            "level": "within_subtype", "subtype": subtype + tag,
            "reference_partition": "clinical vs glycemic (reference)",
            "representation": "n/a", "ari": point, "ci_low": lo, "ci_high": hi, "n": n_sub,
        })
        within_clinical_glycemic[subtype] = (clinical_labels, glycemic_labels)

    # --- Pooled within-subtype (primary subtypes only, insulin-dependent excluded and reported separately) ---
    primary_subtypes = [s for s in DIAGNOSTIC_SUBTYPES if s != EXPLORATORY_SUBTYPE]
    for rep_name, ref_name, per_subtype in [
        ("h0", "clinical clusters (within subtype)", within_rep_clinical["h0"]),
        ("h_t", "clinical clusters (within subtype)", within_rep_clinical["h_t"]),
        ("h0", "glycemic clusters (within subtype)", within_rep_glycemic["h0"]),
        ("h_t", "glycemic clusters (within subtype)", within_rep_glycemic["h_t"]),
    ]:
        point, lo, hi = pooled_ari_bootstrap(per_subtype, primary_subtypes, rng)
        n_pooled = sum((panel["diagnostic_subtype"] == s).sum() for s in primary_subtypes)
        rows.append({
            "level": "pooled_primary", "subtype": "pooled (Healthy+Prediabetes+T2D oral non-insulin)",
            "reference_partition": ref_name, "representation": rep_name,
            "ari": point, "ci_low": lo, "ci_high": hi, "n": int(n_pooled),
        })

    point, lo, hi = pooled_ari_bootstrap(within_clinical_glycemic, primary_subtypes, rng)
    n_pooled = sum((panel["diagnostic_subtype"] == s).sum() for s in primary_subtypes)
    rows.append({
        "level": "pooled_primary", "subtype": "pooled (Healthy+Prediabetes+T2D oral non-insulin)",
        "reference_partition": "clinical vs glycemic (reference)", "representation": "n/a",
        "ari": point, "ci_low": lo, "ci_high": hi, "n": int(n_pooled),
    })

    table = pd.DataFrame(rows)
    table.to_csv(TABLE1_PATH, index=False)
    with open(DIAGNOSTICS_PATH, "w") as f:
        json.dump(diagnostics, f, indent=2)

    make_figure(table)
    print_summary(table)
    print(f"\nWrote {TABLE1_PATH}")
    print(f"Wrote {FIG1_PATH}")
    print(f"Wrote {DIAGNOSTICS_PATH}")


def make_figure(table: pd.DataFrame) -> None:
    group_source = {
        "diagnostic subtype (global)": table[(table["level"] == "global")],
        "clinical clusters (within subtype)": table[
            (table["level"] == "pooled_primary") & (table["reference_partition"] == "clinical clusters (within subtype)")
        ],
        "glycemic clusters (within subtype)": table[
            (table["level"] == "pooled_primary") & (table["reference_partition"] == "glycemic clusters (within subtype)")
        ],
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bar_width = 0.32
    x = np.arange(len(REFERENCE_PARTITIONS))

    h0_vals, h0_lo, h0_hi, h0_ci_high = [], [], [], []
    ht_vals, ht_lo, ht_hi, ht_ci_high = [], [], [], []
    for group in REFERENCE_PARTITIONS:
        sub = group_source[group]
        h0_row = sub[sub["representation"] == "h0"].iloc[0]
        ht_row = sub[sub["representation"] == "h_t"].iloc[0]
        h0_vals.append(h0_row["ari"]); h0_lo.append(h0_row["ari"] - h0_row["ci_low"]); h0_hi.append(h0_row["ci_high"] - h0_row["ari"])
        h0_ci_high.append(h0_row["ci_high"])
        ht_vals.append(ht_row["ari"]); ht_lo.append(ht_row["ari"] - ht_row["ci_low"]); ht_hi.append(ht_row["ci_high"] - ht_row["ari"])
        ht_ci_high.append(ht_row["ci_high"])

    ax.bar(x - bar_width / 2, h0_vals, width=bar_width, color=COLOR_H0, label="h0 (initial state)",
           yerr=[h0_lo, h0_hi], capsize=3, error_kw={"linewidth": 1.0})
    ax.bar(x + bar_width / 2, ht_vals, width=bar_width, color=COLOR_HT, label="h_t (streamed, 48h)",
           yerr=[ht_lo, ht_hi], capsize=3, error_kw={"linewidth": 1.0})

    wrapped_labels = [
        "Diagnostic subtype\n(global)",
        "Clinical clusters\n(within subtype)",
        "Glycemic clusters\n(within subtype)",
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels)
    ax.set_ylabel("Adjusted rand index")
    ax.axhline(0.0, color="black", linewidth=0.8)

    # Directional read: a leg counts as favoring a representation only when that
    # representation's point estimate clears the other representation's upper CI.
    clinical_favors_h0 = h0_vals[1] > ht_ci_high[1]
    glycemic_favors_ht = ht_vals[2] > h0_ci_high[2]
    if clinical_favors_h0 and glycemic_favors_ht:
        title = "Streamed state realigns from clinical to glycemic partition"
    elif clinical_favors_h0 and not glycemic_favors_ht:
        title = "Clinical partition dissolves in the streamed state, glycemic realignment not confirmed"
    elif glycemic_favors_ht:
        title = "Glycemic realignment appears without confirmed clinical dissolution"
    else:
        title = "No confirmed realignment from clinical to glycemic partition"
    ax.set_title(title, fontsize=11)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG1_PATH, dpi=200)
    plt.close(fig)


def print_summary(table: pd.DataFrame) -> None:
    print("\n=== ARI realignment cross tab ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(table.to_string(index=False))

    global_h0 = table[(table["level"] == "global") & (table["representation"] == "h0")].iloc[0]
    global_ht = table[(table["level"] == "global") & (table["representation"] == "h_t")].iloc[0]
    pooled_clin_h0 = table[(table["level"] == "pooled_primary") & (table["reference_partition"] == "clinical clusters (within subtype)") & (table["representation"] == "h0")].iloc[0]
    pooled_clin_ht = table[(table["level"] == "pooled_primary") & (table["reference_partition"] == "clinical clusters (within subtype)") & (table["representation"] == "h_t")].iloc[0]
    pooled_gly_h0 = table[(table["level"] == "pooled_primary") & (table["reference_partition"] == "glycemic clusters (within subtype)") & (table["representation"] == "h0")].iloc[0]
    pooled_gly_ht = table[(table["level"] == "pooled_primary") & (table["reference_partition"] == "glycemic clusters (within subtype)") & (table["representation"] == "h_t")].iloc[0]

    def describe(row_a, row_b, higher_is):
        gap = row_a["ari"] - row_b["ari"] if higher_is == "a" else row_b["ari"] - row_a["ari"]
        ci_excludes_zero = (row_a["ci_low"] > row_b["ci_high"]) or (row_b["ci_low"] > row_a["ci_high"])
        return gap, ci_excludes_zero

    print("\nPlain-language readout (prediction: h0 aligns more with clinical, h_t aligns more with glycemic, "
          "diagnostic axis retained):")
    print(f"1. Diagnostic subtype (global): h0 ARI={global_h0['ari']:.3f} [{global_h0['ci_low']:.3f}, "
          f"{global_h0['ci_high']:.3f}], h_t ARI={global_ht['ari']:.3f} [{global_ht['ci_low']:.3f}, "
          f"{global_ht['ci_high']:.3f}]. "
          + ("Diagnostic axis is retained under streaming." if global_ht["ari"] >= global_h0["ci_low"]
             else "Diagnostic axis erodes under streaming."))
    print(f"2. Clinical clusters (pooled within subtype): h0 ARI={pooled_clin_h0['ari']:.3f} "
          f"[{pooled_clin_h0['ci_low']:.3f}, {pooled_clin_h0['ci_high']:.3f}], h_t ARI={pooled_clin_ht['ari']:.3f} "
          f"[{pooled_clin_ht['ci_low']:.3f}, {pooled_clin_ht['ci_high']:.3f}]. "
          + ("h0 aligns more with the clinical partition, as predicted."
             if pooled_clin_h0["ari"] > pooled_clin_ht["ci_high"]
             else "The predicted h0-over-h_t clinical advantage is not confirmed at this CI."))
    print(f"3. Glycemic clusters (pooled within subtype): h0 ARI={pooled_gly_h0['ari']:.3f} "
          f"[{pooled_gly_h0['ci_low']:.3f}, {pooled_gly_h0['ci_high']:.3f}], h_t ARI={pooled_gly_ht['ari']:.3f} "
          f"[{pooled_gly_ht['ci_low']:.3f}, {pooled_gly_ht['ci_high']:.3f}]. "
          + ("h_t aligns more with the glycemic partition, as predicted."
             if pooled_gly_ht["ari"] > pooled_gly_h0["ci_high"]
             else "The predicted h_t-over-h0 glycemic advantage is not confirmed at this CI."))
    print("Insulin-dependent is exploratory (small n); its per-subtype rows are reported but excluded "
          "from the pooled_primary aggregate.")


if __name__ == "__main__":
    main()
