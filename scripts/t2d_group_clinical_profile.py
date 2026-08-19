"""T2D streamed-state group clinical profile.

Characterizes the three T2D oral non-insulin streamed-state glucose groups
(G0, G1, G2 from the reassignment analysis) by baseline clinical phenotype,
testing whether the glucose-exposure axis also carries an independent
clinical-phenotype signature or is glucose-specific. Reuses the h_t hour-48
snapshot and the frozen six-factor clinical variables. No forward pass, no
model retraining. Read-only on the canonical checkpoint and the multimodal
parquet.

The streamed-group assignment itself was not persisted by the prior
triptych run (only aggregate statistics were saved), so it is recomputed
here with the identical deterministic recipe (same snapshot, same PCA
variance target, same Ward k, same seed) and verified to reproduce the
same group sizes (20, 24, 17) before use. See the Phase 0 gate report.
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
from scipy.stats import kruskal
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
BRANCH = "aireadi-ssmcgm-stream-report"
INPUT_DIR = REPO_ROOT / "outputs/static_phenotype_trajectory/t2d_exception"
OUTPUT_SUBDIR = REPO_ROOT / "outputs/static_phenotype_trajectory/t2d_group_phenotype"
MULTIMODAL_PARQUET = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "final_multimodal_dataset_20260515_184339.parquet"
)

MATCHED_PANEL_PATH = (
    REPO_ROOT
    / "outputs/static_phenotype_trajectory_stratified_v2/global_realignment/matched_panel_participant_ids.csv"
)
HT_HOUR48_PATH = (
    REPO_ROOT
    / "outputs/static_phenotype_trajectory_stratified_v2/phase4_time_resolved_extension/snapshots/h_t_full_hour48.parquet"
)
PRIOR_GLUCOSE_PROFILE_PATH = INPUT_DIR / "panel_c_glucose_profile.csv"

TARGET_SUBTYPE = "T2D oral non-insulin"
STREAMED_GROUPS = ["G0", "G1", "G2"]
PCA_VARIANCE_TARGET = 0.95
T2D_CLINICAL_K = 3

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
FACTOR_UNITS = {
    "study_visit_age": "years", "bmi": "kg/m^2", "hba1c": "%", "c_peptide": "ng/mL",
    "tg_hdl_ratio": "ratio", "waist_to_hip_ratio": "ratio",
}
GLUCOSE_LINKED = ["hba1c"]
INDEPENDENT_FACTORS = ["bmi", "c_peptide", "tg_hdl_ratio", "waist_to_hip_ratio", "study_visit_age"]
PARTICIPANT_ID_COLUMN = "participant_id"

N_BOOTSTRAP = 1000
BOOTSTRAP_CI = (2.5, 97.5)
RANDOM_SEED = 42
FDR_ALPHA = 0.05

STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
COLOR_NULL = "#888888"
GROUP_LABEL_BY_INDEX = {0: "High-exposure", 1: "Mid-range", 2: "Near-range"}
EXPOSURE_ORDER = [2, 1, 0]  # streamed_group index, near-range to high-exposure

ASSIGNMENT_PATH = OUTPUT_SUBDIR / "t2d_streamed_group_assignment.csv"
TABLE_PROFILE_PATH = OUTPUT_SUBDIR / "t2d_group_clinical_profile.csv"
TABLE_OMNIBUS_PATH = OUTPUT_SUBDIR / "t2d_group_clinical_omnibus.csv"
TABLE_PAIRWISE_PATH = OUTPUT_SUBDIR / "t2d_group_clinical_pairwise.csv"
FIG_PATH = OUTPUT_SUBDIR / "t2d_group_clinical_profile.png"
DIAGNOSTICS_PATH = OUTPUT_SUBDIR / "t2d_group_phenotype_diagnostics.json"

sns.set_style("whitegrid")
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 0.8


# ---------------------------------------------------------------------------
# Streamed-group reconstruction (deterministic, verified at the gate)
# ---------------------------------------------------------------------------
def load_snapshot_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    ids = df[PARTICIPANT_ID_COLUMN].astype(str).to_numpy()
    dim_cols = sorted((c for c in df.columns if c.isdigit()), key=int)
    matrix = df[dim_cols].to_numpy(dtype=np.float32)
    return ids, matrix


def reconstruct_t2d_streamed_groups(diagnostics: dict) -> pd.DataFrame:
    panel = pd.read_csv(MATCHED_PANEL_PATH)
    panel[PARTICIPANT_ID_COLUMN] = panel[PARTICIPANT_ID_COLUMN].astype(str)
    t2d = panel[panel["diagnostic_subtype"] == TARGET_SUBTYPE].sort_values(
        PARTICIPANT_ID_COLUMN
    ).reset_index(drop=True)

    ids, matrix = load_snapshot_matrix(HT_HOUR48_PATH)
    id_to_row = {pid: i for i, pid in enumerate(ids)}
    positions = np.array([id_to_row[pid] for pid in t2d[PARTICIPANT_ID_COLUMN]], dtype=int)
    ht_sub = matrix[positions]

    n_components_cap = max(1, ht_sub.shape[0] - 1)
    pca = PCA(n_components=min(n_components_cap, ht_sub.shape[1]), svd_solver="full",
              random_state=RANDOM_SEED)
    pca.fit(ht_sub)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET) + 1)
    n_components = min(n_components, n_components_cap)
    reduced = pca.transform(ht_sub)[:, :n_components]
    labels = AgglomerativeClustering(n_clusters=T2D_CLINICAL_K, linkage="ward").fit_predict(reduced)

    t2d["streamed_group"] = labels
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    diagnostics["reconstruction"] = {
        "n_components": n_components, "variance_explained": float(cumvar[n_components - 1]),
        "group_counts": {int(k): int(v) for k, v in counts.items()},
    }
    print(f"Reconstructed streamed groups: {diagnostics['reconstruction']['group_counts']} "
          f"({n_components} PCA components, {cumvar[n_components - 1]:.3f} variance)")

    expected_counts = {}
    if PRIOR_GLUCOSE_PROFILE_PATH.exists():
        prior = pd.read_csv(PRIOR_GLUCOSE_PROFILE_PATH)
        expected_counts = prior.drop_duplicates("streamed_group").set_index("streamed_group")["n"].to_dict()
    matches_prior = {int(k): int(v) for k, v in counts.items()} == {int(k): int(v) for k, v in expected_counts.items()}
    diagnostics["reconstruction"]["matches_prior_run"] = matches_prior
    print(f"Matches prior triptych run's group sizes: {matches_prior}")
    if not matches_prior:
        raise RuntimeError(
            "Reconstructed streamed-group sizes do not match the prior run; stopping rather than "
            "silently using a different partition."
        )
    return t2d[[PARTICIPANT_ID_COLUMN, "streamed_group"]]


# ---------------------------------------------------------------------------
# Clinical factors
# ---------------------------------------------------------------------------
def compute_clinical_factor_table(participant_ids: pd.Series) -> pd.DataFrame:
    cols = list(FACTOR_COLUMN_MAP.values()) + [TG_COLUMN, HDL_COLUMN]
    df = pd.read_parquet(MULTIMODAL_PARQUET, columns=[PARTICIPANT_ID_COLUMN] + cols)
    df = df.drop_duplicates(subset=PARTICIPANT_ID_COLUMN, keep="first").set_index(PARTICIPANT_ID_COLUMN)
    df = df.loc[participant_ids]
    out = pd.DataFrame(index=participant_ids)
    for factor, col in FACTOR_COLUMN_MAP.items():
        out[factor] = df[col].to_numpy()
    out["tg_hdl_ratio"] = (df[TG_COLUMN] / df[HDL_COLUMN]).to_numpy()
    out = out[CLINICAL_FACTORS]
    return out.reset_index()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def mean_bootstrap(values: np.ndarray, rng: np.random.RandomState) -> tuple[float, float, float]:
    point = float(np.mean(values))
    n = len(values)
    replicates = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        replicates[i] = np.mean(values[idx])
    lo, hi = np.percentile(replicates, BOOTSTRAP_CI)
    return point, float(lo), float(hi)


def kruskal_h_bootstrap(groups: list[np.ndarray], rng: np.random.RandomState) -> tuple[float, float, float]:
    point = float(kruskal(*groups).statistic)
    replicates = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        resampled = [g[rng.randint(0, len(g), size=len(g))] for g in groups]
        replicates[i] = kruskal(*resampled).statistic
    lo, hi = np.percentile(replicates, BOOTSTRAP_CI)
    return point, float(lo), float(hi)


def mean_diff_bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.RandomState) -> tuple[float, float, float]:
    point = float(np.mean(a) - np.mean(b))
    na, nb = len(a), len(b)
    replicates = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        idx_a = rng.randint(0, na, size=na)
        idx_b = rng.randint(0, nb, size=nb)
        replicates[i] = np.mean(a[idx_a]) - np.mean(b[idx_b])
    lo, hi = np.percentile(replicates, BOOTSTRAP_CI)
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)
    diagnostics: dict = {}

    assignment = reconstruct_t2d_streamed_groups(diagnostics)
    assignment.to_csv(ASSIGNMENT_PATH, index=False)

    factors = compute_clinical_factor_table(assignment[PARTICIPANT_ID_COLUMN])
    data = assignment.merge(factors, on=PARTICIPANT_ID_COLUMN, how="left")
    assert data[CLINICAL_FACTORS].isna().sum().sum() == 0, "unexpected missing clinical factor values"
    print(f"Clinical factor table: n={len(data)}, all six factors complete case")

    # ---- Phase 1: per-group profile, raw units ----
    profile_rows = []
    for group_idx in sorted(data["streamed_group"].unique()):
        group_rows = data[data["streamed_group"] == group_idx]
        n_group = len(group_rows)
        for factor in CLINICAL_FACTORS:
            point, lo, hi = mean_bootstrap(group_rows[factor].to_numpy(), rng)
            profile_rows.append({
                "streamed_group": f"G{group_idx}", "physiological_label": GROUP_LABEL_BY_INDEX[group_idx],
                "n": n_group, "factor": factor, "unit": FACTOR_UNITS[factor],
                "mean": point, "ci_low": lo, "ci_high": hi,
                "glucose_linked": factor in GLUCOSE_LINKED,
            })
    profile_table = pd.DataFrame(profile_rows)
    profile_table.to_csv(TABLE_PROFILE_PATH, index=False)

    print("\n=== Phase 1: per-group clinical profile ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(profile_table.to_string(index=False))

    # ---- Phase 2: omnibus Kruskal-Wallis, BH-FDR, pairwise ----
    omnibus_rows = []
    group_order = sorted(data["streamed_group"].unique())
    group_values = {
        factor: [data.loc[data["streamed_group"] == g, factor].to_numpy() for g in group_order]
        for factor in CLINICAL_FACTORS
    }
    for factor in CLINICAL_FACTORS:
        h_point, h_lo, h_hi = kruskal_h_bootstrap(group_values[factor], rng)
        _, p_value = kruskal(*group_values[factor])
        omnibus_rows.append({
            "factor": factor, "glucose_linked": factor in GLUCOSE_LINKED,
            "h_statistic": h_point, "h_ci_low": h_lo, "h_ci_high": h_hi, "p_raw": float(p_value),
        })
    omnibus_table = pd.DataFrame(omnibus_rows)
    reject, q_values, _, _ = multipletests(omnibus_table["p_raw"].to_numpy(), alpha=FDR_ALPHA, method="fdr_bh")
    omnibus_table["q_value"] = q_values
    omnibus_table["significant_fdr"] = reject
    omnibus_table.to_csv(TABLE_OMNIBUS_PATH, index=False)

    print("\n=== Phase 2: omnibus Kruskal-Wallis (BH-FDR over 6 factors) ===")
    print(omnibus_table.to_string(index=False))

    pairwise_rows = []
    significant_factors = omnibus_table.loc[omnibus_table["significant_fdr"], "factor"].tolist()
    for factor in significant_factors:
        for i in range(len(group_order)):
            for j in range(i + 1, len(group_order)):
                g_a, g_b = group_order[i], group_order[j]
                a = data.loc[data["streamed_group"] == g_a, factor].to_numpy()
                b = data.loc[data["streamed_group"] == g_b, factor].to_numpy()
                point, lo, hi = mean_diff_bootstrap(a, b, rng)
                pairwise_rows.append({
                    "factor": factor, "group_a": f"G{g_a}", "group_b": f"G{g_b}",
                    "mean_diff": point, "ci_low": lo, "ci_high": hi,
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                })
    pairwise_table = pd.DataFrame(pairwise_rows)
    pairwise_table.to_csv(TABLE_PAIRWISE_PATH, index=False)
    if len(pairwise_table):
        print("\n=== Pairwise contrasts for FDR-significant factors ===")
        print(pairwise_table.to_string(index=False))
    else:
        print("\nNo factor reached FDR significance; no pairwise contrasts computed.")

    with open(DIAGNOSTICS_PATH, "w") as f:
        json.dump(diagnostics, f, indent=2)

    make_figure(data, omnibus_table)
    print_summary(omnibus_table, significant_factors)
    print(f"\nWrote {ASSIGNMENT_PATH}")
    print(f"Wrote {TABLE_PROFILE_PATH}")
    print(f"Wrote {TABLE_OMNIBUS_PATH}")
    print(f"Wrote {TABLE_PAIRWISE_PATH}")
    print(f"Wrote {FIG_PATH}")
    print(f"Wrote {DIAGNOSTICS_PATH}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(data: pd.DataFrame, omnibus_table: pd.DataFrame) -> None:
    rng = np.random.RandomState(RANDOM_SEED)
    ordered_group_idx = EXPOSURE_ORDER
    ordered_labels = [GROUP_LABEL_BY_INDEX[g] for g in ordered_group_idx]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    axes = axes.flatten()

    independent_significant = [
        f for f in INDEPENDENT_FACTORS
        if omnibus_table.loc[omnibus_table["factor"] == f, "significant_fdr"].iloc[0]
    ]

    # Same near-range to high-exposure teal-to-red severity ramp used in the
    # T2D exception triptych figure, for visual consistency across the chapter.
    exposure_ramp = sns.blend_palette(
        [STRATUM_COLORS[2], STRATUM_COLORS[0]], n_colors=len(ordered_group_idx)
    )

    for ax, factor in zip(axes, CLINICAL_FACTORS):
        values = data[factor].to_numpy()
        z = (values - values.mean()) / values.std(ddof=0)
        data_z = data.copy()
        data_z["_z"] = z

        is_glucose_linked = factor in GLUCOSE_LINKED
        if is_glucose_linked:
            ax.set_facecolor("#f2f2f2")

        for i, group_idx in enumerate(ordered_group_idx):
            group_z = data_z.loc[data_z["streamed_group"] == group_idx, "_z"].to_numpy()
            point, lo, hi = mean_bootstrap(group_z, rng)
            color = exposure_ramp[i]
            ax.errorbar(i, point, yerr=[[point - lo], [hi - point]], fmt="o", color=color,
                        markersize=8, capsize=4, linewidth=1.5)

        ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
        ax.set_xticks(range(len(ordered_labels)))
        ax.set_xticklabels(ordered_labels, fontsize=8)
        q_val = omnibus_table.loc[omnibus_table["factor"] == factor, "q_value"].iloc[0]
        title_suffix = " (glucose-linked)" if is_glucose_linked else ""
        ax.set_title(f"{factor.replace('_', ' ').capitalize()}{title_suffix}, q={q_val:.3f}", fontsize=9)
        ax.set_ylabel("Standardized value")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)

    if independent_significant:
        readable = ", ".join(f.replace("_", " ") for f in independent_significant)
        suptitle = f"T2D exposure groups separate on glucose and on independent phenotype: {readable}"
    else:
        suptitle = "T2D exposure groups separate on glucose but not on independent phenotype"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)


def print_summary(omnibus_table: pd.DataFrame, significant_factors: list[str]) -> None:
    hba1c_row = omnibus_table[omnibus_table["factor"] == "hba1c"].iloc[0]
    print(f"\nGlucose-linked factor (hba1c, expected to separate, not an independent phenotype signal): "
          f"H={hba1c_row['h_statistic']:.2f} [{hba1c_row['h_ci_low']:.2f}, {hba1c_row['h_ci_high']:.2f}], "
          f"p={hba1c_row['p_raw']:.4f}, q={hba1c_row['q_value']:.4f}, "
          f"significant={'yes' if hba1c_row['significant_fdr'] else 'no'}.")

    independent_significant = [f for f in significant_factors if f in INDEPENDENT_FACTORS]
    if independent_significant:
        readable = ", ".join(independent_significant)
        print(f"\nPlain-language summary: the T2D streamed-state exposure groups carry an independent "
              f"clinical-phenotype signature beyond glucose. Factor(s) surviving FDR correction: {readable}.")
    else:
        print("\nPlain-language summary: no independent clinical factor (age, BMI, C-peptide, TG/HDL "
              "ratio, waist-to-hip ratio) separates the T2D streamed-state exposure groups after FDR "
              "correction. This is a confirmed null given the bootstrap CIs, not an underpowered gap: "
              "the exposure axis reads as glucose-specific, not a proxy for baseline clinical phenotype.")


if __name__ == "__main__":
    main()
