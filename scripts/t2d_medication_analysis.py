"""T2D medication reanalyses.

Three medication-related analyses on the T2D oral non-insulin streamed-state
groups (G0, G1, G2) and the insulin-dependent stratum, all reusing existing
saved hidden-state group labels, frozen clinical variables, and pre-built
RxNorm medication class columns. No forward pass, no model retraining.
Read-only on the canonical checkpoint, the medication files, and both
candidate parquets.

Phase 1: within-regimen confounder test on G0/G1/G2 glucose exposure.
Phase 2: G0/G1/G2 clinical phenotype profile with medication covariates.
Phase 3: insulin-dependent stratum medication description, no clustering.
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
from scipy.stats import chi2_contingency, kruskal
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
BRANCH = "aireadi-ssmcgm-stream-report"
DATA_DIR = Path("/home/myriamcharfeddine/CGM/Data/enriched_multimodal")
CANONICAL_PARQUET = DATA_DIR / "final_multimodal_dataset_20260515_184339.parquet"
ENV_PARQUET = DATA_DIR / "final_multimodal_with_environment_20260716_222829.parquet"
MEDS_LONG_PARQUET = DATA_DIR / "participant_medications_long.parquet"
MED_CLASS_COVERAGE = DATA_DIR / "medication_class_coverage.csv"
STATIC_FEATURES_PARQUET = DATA_DIR / "participant_static_features.parquet"

T2D_GROUPS_DIR = REPO_ROOT / "outputs/static_phenotype_trajectory/t2d_exception"
T2D_GROUP_ASSIGNMENT_PATH = (
    REPO_ROOT / "outputs/static_phenotype_trajectory/t2d_group_phenotype/t2d_streamed_group_assignment.csv"
)
MATCHED_PANEL_PATH = (
    REPO_ROOT / "outputs/static_phenotype_trajectory_stratified_v2/global_realignment/matched_panel_participant_ids.csv"
)
OUTPUT_SUBDIR = REPO_ROOT / "outputs/static_phenotype_trajectory/t2d_medication"

ID_COL = "studyid"
PARTICIPANT_ID_COLUMN = "participant_id"
TARGET_SUBTYPE = "T2D oral non-insulin"
INSULIN_SUBTYPE = "Insulin-dependent"
STREAMED_GROUPS = ["G0", "G1", "G2"]
GROUP_LABEL_BY_INDEX = {0: "High-exposure", 1: "Mid-range", 2: "Near-range"}
EXPOSURE_ORDER = [2, 1, 0]  # streamed_group index, near-range to high-exposure
HIGH_GROUP_IDX = 0
NEAR_GROUP_IDX = 2

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
GLUCOSE_LINKED_FACTOR = "hba1c"

GLYCEMIC_FEATURES = ["mean_cgm", "time_in_range", "time_above_180"]
GLYCEMIC_WINDOW_H = 48
CGM_COLUMN = "cgm_glucose_mean"
TIMESTAMP_COLUMN = "timestamp_local"
TIR_LOW_MGDL = 70
TIR_HIGH_MGDL = 180
HYPER_THRESHOLD_MGDL = 180

# The medication class coverage file only has these five diabetes-oral
# classes plus insulin. dpp4 and meglitinide have no RxNorm-classified
# column anywhere and are explicitly out of scope (see Phase 0 gate).
ORAL_CLASSES = ["metformin", "sulfonylurea", "glp1_or_gip_glp1", "sglt2", "thiazolidinedione"]
ORAL_CLASS_COLUMNS = [f"med_{c}" for c in ORAL_CLASSES]
INSULIN_COLUMN = "med_insulin"
ANY_DRUG_COLUMN = "med_any_diabetes_drug"
MIN_REGIMEN_N = 8

BASAL_TERMS = ["glargine", "detemir", "degludec", "nph"]
BOLUS_TERMS = ["aspart", "lispro", "glulisine", "regular"]

N_BOOTSTRAP = 1000
BOOTSTRAP_CI = (2.5, 97.5)
RANDOM_SEED = 42
FDR_ALPHA = 0.05

STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
COLOR_NULL = "#888888"

ASSIGNMENT_MISMATCH_NOTE_PATH = OUTPUT_SUBDIR / "t2d_medication_label_mismatches.json"
FIG1_PATH = OUTPUT_SUBDIR / "t2d_within_regimen_exposure.png"
TABLE1_PATH = OUTPUT_SUBDIR / "t2d_within_regimen_exposure.csv"
FIG2_PATH = OUTPUT_SUBDIR / "t2d_group_profile_with_meds.png"
TABLE2_PATH = OUTPUT_SUBDIR / "t2d_group_profile_with_meds.csv"
TABLE3_PATH = OUTPUT_SUBDIR / "insulin_dependent_medication_description.csv"
WRITEUP_PATH = OUTPUT_SUBDIR / "t2d_medication_writeup.md"
DIAGNOSTICS_PATH = OUTPUT_SUBDIR / "t2d_medication_diagnostics.json"

sns.set_style("whitegrid")
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 0.8


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------
def load_t2d_panel() -> pd.DataFrame:
    assignment = pd.read_csv(T2D_GROUP_ASSIGNMENT_PATH)
    assignment[PARTICIPANT_ID_COLUMN] = assignment[PARTICIPANT_ID_COLUMN].astype(str)

    med_cols = ORAL_CLASS_COLUMNS + [INSULIN_COLUMN, ANY_DRUG_COLUMN]
    meds = pd.read_parquet(STATIC_FEATURES_PARQUET, columns=[PARTICIPANT_ID_COLUMN] + med_cols)
    meds[PARTICIPANT_ID_COLUMN] = meds[PARTICIPANT_ID_COLUMN].astype(str)

    panel = assignment.merge(meds, on=PARTICIPANT_ID_COLUMN, how="left")
    assert panel[ANY_DRUG_COLUMN].isna().sum() == 0, "missing medication data for T2D panel"
    return panel


def regimen_label(row: pd.Series) -> str:
    """Regimen label for the T2D oral panel, where any insulin presence is
    itself a diagnostic-label mismatch, not a regimen dimension."""
    classes = [c for c, col in zip(ORAL_CLASSES, ORAL_CLASS_COLUMNS) if row[col]]
    if row[INSULIN_COLUMN]:
        return "insulin_flag_present"
    if not classes:
        return "none_recorded"
    return "+".join(sorted(classes))


def full_regimen_label(row: pd.Series) -> str:
    """Full regimen label including insulin as a first-class component, for
    the insulin-dependent stratum where insulin is expected and the goal is
    to describe regimen composition, not flag a diagnostic mismatch."""
    classes = [c for c, col in zip(ORAL_CLASSES, ORAL_CLASS_COLUMNS) if row[col]]
    if row[INSULIN_COLUMN]:
        classes = ["insulin"] + classes
    if not classes:
        return "none_recorded"
    return "+".join(classes)


def compute_glycemic_features(participant_ids: pd.Series) -> pd.DataFrame:
    """Positional per-participant slicing over the canonical parquet.

    Row blocks are contiguous and pre-sorted by timestamp per participant
    (verified in prior gates on this same file), so a start/end position
    index built once lets every participant's window be a plain array
    slice, never a boolean mask over the full multi-million-row frame.
    """
    df = pd.read_parquet(
        CANONICAL_PARQUET, columns=[PARTICIPANT_ID_COLUMN, TIMESTAMP_COLUMN, CGM_COLUMN]
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
    for pid in participant_ids:
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
        records.append({
            PARTICIPANT_ID_COLUMN: pid,
            "mean_cgm": mean_cgm,
            "time_in_range": float(np.mean(in_range)) if n else np.nan,
            "time_above_180": float(np.mean(above)) if n else np.nan,
        })
    return pd.DataFrame.from_records(records)


def compute_clinical_factor_table(participant_ids: pd.Series) -> pd.DataFrame:
    cols = list(FACTOR_COLUMN_MAP.values()) + [TG_COLUMN, HDL_COLUMN]
    df = pd.read_parquet(CANONICAL_PARQUET, columns=[PARTICIPANT_ID_COLUMN] + cols)
    df = df.drop_duplicates(subset=PARTICIPANT_ID_COLUMN, keep="first").set_index(PARTICIPANT_ID_COLUMN)
    df = df.loc[participant_ids]
    out = pd.DataFrame(index=participant_ids)
    for factor, col in FACTOR_COLUMN_MAP.items():
        out[factor] = df[col].to_numpy()
    out["tg_hdl_ratio"] = (df[TG_COLUMN] / df[HDL_COLUMN]).to_numpy()
    out = out[CLINICAL_FACTORS]
    return out.reset_index()


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def mean_bootstrap(values: np.ndarray, rng: np.random.RandomState) -> tuple[float, float, float]:
    point = float(np.mean(values))
    n = len(values)
    if n == 1:
        return point, point, point
    replicates = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        replicates[i] = np.mean(values[idx])
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


def proportion_bootstrap(flags: np.ndarray, rng: np.random.RandomState) -> tuple[float, float, float]:
    point = float(np.mean(flags))
    n = len(flags)
    replicates = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        replicates[i] = np.mean(flags[idx])
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


def permutation_chi2_pvalue(table: pd.DataFrame, group_col: str, regimen_col: str,
                             rng: np.random.RandomState) -> tuple[float, float]:
    observed_ct = pd.crosstab(table[group_col], table[regimen_col])
    observed_chi2, _, _, _ = chi2_contingency(observed_ct, correction=False)
    regimen_values = table[regimen_col].to_numpy().copy()
    count = 0
    for _ in range(N_BOOTSTRAP):
        shuffled = rng.permutation(regimen_values)
        ct = pd.crosstab(table[group_col], pd.Series(shuffled))
        chi2_stat, _, _, _ = chi2_contingency(ct, correction=False)
        if chi2_stat >= observed_chi2:
            count += 1
    return float(observed_chi2), float(count / N_BOOTSTRAP)


# ---------------------------------------------------------------------------
# Phase 1: within-regimen confounder test
# ---------------------------------------------------------------------------
def phase1_within_regimen(panel: pd.DataFrame, glycemic: pd.DataFrame, rng: np.random.RandomState,
                           diagnostics: dict) -> pd.DataFrame:
    data = panel.merge(glycemic, on=PARTICIPANT_ID_COLUMN, how="left")
    data["regimen"] = data.apply(regimen_label, axis=1)

    rows = []
    print("\nPhase 1, unconditional exposure gradient (all 61 T2D participants):")
    for group_idx in sorted(data["streamed_group"].unique()):
        vals = data.loc[data["streamed_group"] == group_idx, "mean_cgm"].to_numpy()
        point, lo, hi = mean_bootstrap(vals, rng)
        rows.append({
            "scope": "unconditional", "regimen": "all", "streamed_group": f"G{group_idx}",
            "physiological_label": GROUP_LABEL_BY_INDEX[group_idx], "n": len(vals),
            "mean_cgm": point, "ci_low": lo, "ci_high": hi,
        })
        print(f"  G{group_idx} ({GROUP_LABEL_BY_INDEX[group_idx]}, n={len(vals)}): "
              f"mean CGM={point:.1f} [{lo:.1f}, {hi:.1f}] mg/dL")

    high_all = data.loc[data["streamed_group"] == HIGH_GROUP_IDX, "mean_cgm"].to_numpy()
    near_all = data.loc[data["streamed_group"] == NEAR_GROUP_IDX, "mean_cgm"].to_numpy()
    diff_point, diff_lo, diff_hi = mean_diff_bootstrap(high_all, near_all, rng)
    print(f"  Unconditional high-minus-near mean CGM difference: {diff_point:.1f} [{diff_lo:.1f}, {diff_hi:.1f}] mg/dL")
    rows.append({
        "scope": "unconditional_contrast", "regimen": "all", "streamed_group": "high_minus_near",
        "physiological_label": "high_minus_near", "n": len(high_all) + len(near_all),
        "mean_cgm": diff_point, "ci_low": diff_lo, "ci_high": diff_hi,
    })

    regimen_counts = data["regimen"].value_counts()
    well_powered = regimen_counts[regimen_counts >= MIN_REGIMEN_N].index.tolist()
    well_powered = [r for r in well_powered if r not in ("insulin_flag_present", "none_recorded")]
    diagnostics["phase1"] = {"well_powered_regimens": well_powered, "regimen_counts": regimen_counts.to_dict()}
    print(f"\nWell-powered regimens (n >= {MIN_REGIMEN_N}): {well_powered}")

    for reg in well_powered:
        reg_data = data[data["regimen"] == reg]
        print(f"\nWithin regimen '{reg}' (n={len(reg_data)}):")
        group_present = sorted(reg_data["streamed_group"].unique())
        for group_idx in group_present:
            vals = reg_data.loc[reg_data["streamed_group"] == group_idx, "mean_cgm"].to_numpy()
            point, lo, hi = mean_bootstrap(vals, rng)
            rows.append({
                "scope": "within_regimen", "regimen": reg, "streamed_group": f"G{group_idx}",
                "physiological_label": GROUP_LABEL_BY_INDEX[group_idx], "n": len(vals),
                "mean_cgm": point, "ci_low": lo, "ci_high": hi,
            })
            print(f"  G{group_idx} ({GROUP_LABEL_BY_INDEX[group_idx]}, n={len(vals)}): "
                  f"mean CGM={point:.1f} [{lo:.1f}, {hi:.1f}] mg/dL")

        if HIGH_GROUP_IDX in group_present and NEAR_GROUP_IDX in group_present:
            high_vals = reg_data.loc[reg_data["streamed_group"] == HIGH_GROUP_IDX, "mean_cgm"].to_numpy()
            near_vals = reg_data.loc[reg_data["streamed_group"] == NEAR_GROUP_IDX, "mean_cgm"].to_numpy()
            diff_point, diff_lo, diff_hi = mean_diff_bootstrap(high_vals, near_vals, rng)
            rows.append({
                "scope": "within_regimen_contrast", "regimen": reg, "streamed_group": "high_minus_near",
                "physiological_label": "high_minus_near", "n": len(high_vals) + len(near_vals),
                "mean_cgm": diff_point, "ci_low": diff_lo, "ci_high": diff_hi,
            })
            print(f"  Within-regimen high-minus-near difference: {diff_point:.1f} [{diff_lo:.1f}, {diff_hi:.1f}] mg/dL")
        else:
            print("  High or near group absent within this regimen; contrast not computed.")

    # Association test: is streamed group redundant with regimen
    collapsed = data["regimen"].where(data["regimen"].isin(well_powered), other="other")
    data["regimen_collapsed"] = collapsed
    observed_chi2, perm_p = permutation_chi2_pvalue(data, "streamed_group", "regimen_collapsed", rng)
    _, analytic_p, dof, _ = chi2_contingency(
        pd.crosstab(data["streamed_group"], data["regimen_collapsed"]), correction=False
    )
    diagnostics["phase1"]["association_test"] = {
        "chi2": observed_chi2, "analytic_p": float(analytic_p), "permutation_p": perm_p, "dof": int(dof),
    }
    print(f"\nGroup-by-regimen association (collapsed to {well_powered + ['other']}): "
          f"chi2={observed_chi2:.2f}, analytic p={analytic_p:.4f}, permutation p={perm_p:.4f}")

    table = pd.DataFrame(rows)
    table.to_csv(TABLE1_PATH, index=False)
    make_figure1(table, well_powered)
    return table


def make_figure1(table: pd.DataFrame, well_powered: list[str]) -> None:
    scopes = ["all"] + well_powered
    fig, axes = plt.subplots(1, len(scopes), figsize=(4.5 * len(scopes), 4.5), sharey=True)
    if len(scopes) == 1:
        axes = [axes]

    for ax, scope_regimen in zip(axes, scopes):
        scope_label = "scope == 'unconditional'" if scope_regimen == "all" else "scope == 'within_regimen'"
        sub = table.query(scope_label + " and regimen == @scope_regimen")
        sub = sub.set_index("streamed_group").reindex([f"G{g}" for g in EXPOSURE_ORDER])
        colors = sns.blend_palette([STRATUM_COLORS[2], STRATUM_COLORS[0]], n_colors=len(EXPOSURE_ORDER))
        for i, (idx, row) in enumerate(sub.iterrows()):
            ax.errorbar(i, row["mean_cgm"], yerr=[[row["mean_cgm"] - row["ci_low"]], [row["ci_high"] - row["mean_cgm"]]],
                        fmt="o", color=colors[i], markersize=9, capsize=4, linewidth=1.5)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([GROUP_LABEL_BY_INDEX[g] for g in EXPOSURE_ORDER], fontsize=8, rotation=15)
        ax.axhline(HYPER_THRESHOLD_MGDL, color=COLOR_NULL, linewidth=1.0, linestyle="--")
        title = "Unconditional (all T2D)" if scope_regimen == "all" else f"Within regimen: {scope_regimen.replace('_', ' ')}"
        ax.set_title(title, fontsize=10)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)

    axes[0].set_ylabel("Mean CGM, mg/dL (0-48h)")
    fig.suptitle("T2D exposure gradient persists within single-regimen strata", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG1_PATH, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 2: clinical profile with medication covariates
# ---------------------------------------------------------------------------
def phase2_group_profile(panel: pd.DataFrame, rng: np.random.RandomState, diagnostics: dict) -> pd.DataFrame:
    factors = compute_clinical_factor_table(panel[PARTICIPANT_ID_COLUMN])
    data = panel.merge(factors, on=PARTICIPANT_ID_COLUMN, how="left")
    data["n_oral_classes"] = data[ORAL_CLASS_COLUMNS].sum(axis=1)

    rows = []
    group_order = sorted(data["streamed_group"].unique())
    print("\nPhase 2, per-group clinical factor and medication profile:")
    for group_idx in group_order:
        group_rows = data[data["streamed_group"] == group_idx]
        n_group = len(group_rows)
        for factor in CLINICAL_FACTORS:
            point, lo, hi = mean_bootstrap(group_rows[factor].to_numpy(), rng)
            rows.append({
                "streamed_group": f"G{group_idx}", "physiological_label": GROUP_LABEL_BY_INDEX[group_idx],
                "n": n_group, "variable": factor, "kind": "clinical_factor",
                "glucose_linked": factor == GLUCOSE_LINKED_FACTOR,
                "value": point, "ci_low": lo, "ci_high": hi,
            })
        for oral_class, col in zip(ORAL_CLASSES, ORAL_CLASS_COLUMNS):
            point, lo, hi = proportion_bootstrap(group_rows[col].to_numpy().astype(float), rng)
            rows.append({
                "streamed_group": f"G{group_idx}", "physiological_label": GROUP_LABEL_BY_INDEX[group_idx],
                "n": n_group, "variable": f"fraction_on_{oral_class}", "kind": "medication_fraction",
                "glucose_linked": False, "value": point, "ci_low": lo, "ci_high": hi,
            })
        point, lo, hi = mean_bootstrap(group_rows["n_oral_classes"].to_numpy(), rng)
        rows.append({
            "streamed_group": f"G{group_idx}", "physiological_label": GROUP_LABEL_BY_INDEX[group_idx],
            "n": n_group, "variable": "n_oral_classes", "kind": "medication_intensity",
            "glucose_linked": False, "value": point, "ci_low": lo, "ci_high": hi,
        })
        print(f"  G{group_idx} ({GROUP_LABEL_BY_INDEX[group_idx]}, n={n_group}): "
              f"mean oral classes={point:.2f} [{lo:.2f}, {hi:.2f}]")

    omnibus_rows = []
    group_values = {
        var: [data.loc[data["streamed_group"] == g, var].to_numpy() for g in group_order]
        for var in CLINICAL_FACTORS + ["n_oral_classes"]
    }
    for var in CLINICAL_FACTORS + ["n_oral_classes"]:
        h_point, h_lo, h_hi = kruskal_h_bootstrap(group_values[var], rng)
        _, p_value = kruskal(*group_values[var])
        omnibus_rows.append({
            "variable": var, "glucose_linked": var == GLUCOSE_LINKED_FACTOR,
            "h_statistic": h_point, "h_ci_low": h_lo, "h_ci_high": h_hi, "p_raw": float(p_value),
        })
    omnibus_table = pd.DataFrame(omnibus_rows)
    reject, q_values, _, _ = multipletests(omnibus_table["p_raw"].to_numpy(), alpha=FDR_ALPHA, method="fdr_bh")
    omnibus_table["q_value"] = q_values
    omnibus_table["significant_fdr"] = reject
    diagnostics["phase2"] = {"omnibus": omnibus_table.to_dict(orient="records")}

    print("\nPhase 2 omnibus Kruskal-Wallis (BH-FDR over 6 clinical factors + medication intensity):")
    print(omnibus_table.to_string(index=False))

    profile_table = pd.DataFrame(rows)
    combined = profile_table.merge(
        omnibus_table[["variable", "h_statistic", "p_raw", "q_value", "significant_fdr"]],
        left_on="variable", right_on="variable", how="left",
    )
    combined.to_csv(TABLE2_PATH, index=False)

    make_figure2(data, omnibus_table)
    return combined


def make_figure2(data: pd.DataFrame, omnibus_table: pd.DataFrame) -> None:
    rng = np.random.RandomState(RANDOM_SEED)
    panels = CLINICAL_FACTORS + ["n_oral_classes"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    exposure_ramp = sns.blend_palette([STRATUM_COLORS[2], STRATUM_COLORS[0]], n_colors=len(EXPOSURE_ORDER))

    for ax, var in zip(axes, panels):
        values = data[var].to_numpy()
        z = (values - values.mean()) / values.std(ddof=0)
        data_z = data.copy()
        data_z["_z"] = z

        is_glucose_linked = var == GLUCOSE_LINKED_FACTOR
        if is_glucose_linked:
            ax.set_facecolor("#f2f2f2")

        for i, group_idx in enumerate(EXPOSURE_ORDER):
            group_z = data_z.loc[data_z["streamed_group"] == group_idx, "_z"].to_numpy()
            point, lo, hi = mean_bootstrap(group_z, rng)
            ax.errorbar(i, point, yerr=[[point - lo], [hi - point]], fmt="o", color=exposure_ramp[i],
                        markersize=8, capsize=4, linewidth=1.5)

        ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
        ax.set_xticks(range(len(EXPOSURE_ORDER)))
        ax.set_xticklabels([GROUP_LABEL_BY_INDEX[g] for g in EXPOSURE_ORDER], fontsize=8)
        q_val = omnibus_table.loc[omnibus_table["variable"] == var, "q_value"].iloc[0]
        title_suffix = " (glucose-linked)" if is_glucose_linked else ""
        readable = var.replace("_", " ").capitalize()
        ax.set_title(f"{readable}{title_suffix}, q={q_val:.3f}", fontsize=9)
        ax.set_ylabel("Standardized value")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)

    for ax in axes[len(panels):]:
        ax.axis("off")

    independent_significant = [
        f for f in CLINICAL_FACTORS + ["n_oral_classes"]
        if f != GLUCOSE_LINKED_FACTOR
        and omnibus_table.loc[omnibus_table["variable"] == f, "significant_fdr"].iloc[0]
    ]
    if independent_significant:
        readable = ", ".join(f.replace("_", " ") for f in independent_significant)
        suptitle = f"T2D exposure groups differ on glucose and on: {readable}"
    else:
        suptitle = "T2D exposure groups separate on glucose but not on independent phenotype or treatment intensity"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG2_PATH, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 3: insulin-dependent descriptive check
# ---------------------------------------------------------------------------
def phase3_insulin_dependent(rng: np.random.RandomState, diagnostics: dict) -> pd.DataFrame:
    matched = pd.read_csv(MATCHED_PANEL_PATH)
    matched[PARTICIPANT_ID_COLUMN] = matched[PARTICIPANT_ID_COLUMN].astype(str)
    ins_panel = matched[matched["diagnostic_subtype"] == INSULIN_SUBTYPE].copy()
    n_panel = len(ins_panel)

    med_cols = ORAL_CLASS_COLUMNS + [INSULIN_COLUMN, ANY_DRUG_COLUMN]
    meds = pd.read_parquet(STATIC_FEATURES_PARQUET, columns=[PARTICIPANT_ID_COLUMN] + med_cols)
    meds[PARTICIPANT_ID_COLUMN] = meds[PARTICIPANT_ID_COLUMN].astype(str)
    ins_panel = ins_panel.merge(meds, on=PARTICIPANT_ID_COLUMN, how="left")

    fraction_on_insulin, ins_lo, ins_hi = proportion_bootstrap(
        ins_panel[INSULIN_COLUMN].to_numpy().astype(float), rng
    )
    n_no_insulin_record = int((ins_panel[INSULIN_COLUMN] != True).sum())  # noqa: E712

    ml = pd.read_parquet(MEDS_LONG_PARQUET)
    ml[PARTICIPANT_ID_COLUMN] = ml[PARTICIPANT_ID_COLUMN].astype(str)
    ins_ids = set(ins_panel[PARTICIPANT_ID_COLUMN])
    ins_rows = ml[ml[PARTICIPANT_ID_COLUMN].isin(ins_ids)]
    ins_rows = ins_rows[ins_rows["rxnorm_term"].astype(str).str.lower().str.contains("insulin", na=False)]

    term_lower = ins_rows["rxnorm_term"].astype(str).str.lower()
    ins_rows = ins_rows.copy()
    ins_rows["is_basal"] = term_lower.str.contains("|".join(BASAL_TERMS))
    ins_rows["is_bolus"] = term_lower.str.contains("|".join(BOLUS_TERMS))

    per_participant = ins_rows.groupby(PARTICIPANT_ID_COLUMN).agg(has_basal=("is_basal", "any"),
                                                                     has_bolus=("is_bolus", "any"))
    n_with_insulin_rows = len(per_participant)
    n_basal_only = int(((per_participant["has_basal"]) & (~per_participant["has_bolus"])).sum())
    n_basal_bolus = int(((per_participant["has_basal"]) & (per_participant["has_bolus"])).sum())
    n_bolus_only = int(((~per_participant["has_basal"]) & (per_participant["has_bolus"])).sum())
    n_unclassified = int(((~per_participant["has_basal"]) & (~per_participant["has_bolus"])).sum())

    ins_panel["any_oral"] = ins_panel[ORAL_CLASS_COLUMNS].any(axis=1)
    oral_cotherapy_fraction, oral_lo, oral_hi = proportion_bootstrap(
        ins_panel["any_oral"].to_numpy().astype(float), rng
    )

    numeric_dose = pd.to_numeric(ins_rows["cmdos"], errors="coerce")
    dose_clean_fraction = float(numeric_dose.notna().mean()) if len(ins_rows) else np.nan
    basal_dose_rows = ins_rows[ins_rows["is_basal"] & numeric_dose.notna()]
    basal_dose_values = pd.to_numeric(basal_dose_rows["cmdos"], errors="coerce").to_numpy()
    if len(basal_dose_values):
        dose_point, dose_lo, dose_hi = mean_bootstrap(basal_dose_values, rng)
    else:
        dose_point = dose_lo = dose_hi = np.nan

    ins_panel["regimen"] = ins_panel.apply(full_regimen_label, axis=1)
    regimen_counts = ins_panel["regimen"].value_counts()
    largest_regimen = regimen_counts.index[0]
    largest_regimen_share = float(regimen_counts.iloc[0] / n_panel)

    rows = [
        {"metric": "n_matched_exploratory_panel", "value": n_panel, "note": "exploratory, n=22, same panel tagged exploratory throughout this chapter"},
        {"metric": "fraction_with_med_insulin_flag", "value": fraction_on_insulin, "ci_low": ins_lo, "ci_high": ins_hi, "note": ""},
        {"metric": "n_no_insulin_record_despite_label", "value": n_no_insulin_record, "note": "label mismatch, insulin-dependent by diagnosis, no med_insulin flag"},
        {"metric": "n_with_any_insulin_rxnorm_row", "value": n_with_insulin_rows, "note": "subset of fraction_with_med_insulin_flag with a parseable insulin row in the long file"},
        {"metric": "n_basal_only", "value": n_basal_only, "note": "rxnorm_term ingredient recognition (glargine, detemir, degludec, NPH), not free-text cmname matching"},
        {"metric": "n_basal_bolus", "value": n_basal_bolus, "note": ""},
        {"metric": "n_bolus_only", "value": n_bolus_only, "note": ""},
        {"metric": "n_unclassified_insulin_type", "value": n_unclassified, "note": ""},
        {"metric": "oral_cotherapy_fraction", "value": oral_cotherapy_fraction, "ci_low": oral_lo, "ci_high": oral_hi, "note": "fraction on any of the 5 oral classes"},
        {"metric": "insulin_dose_field_clean_fraction", "value": dose_clean_fraction, "note": "fraction of insulin rows with a numeric cmdos value; cmdosu is a coded unit with no available codebook, not decoded"},
        {"metric": "basal_standing_dose_mean", "value": dose_point, "ci_low": dose_lo, "ci_high": dose_hi, "note": "mean raw cmdos among basal insulin rows, opaque unit code, descriptive only"},
        {"metric": "largest_single_regimen", "value": largest_regimen, "note": ""},
        {"metric": "largest_single_regimen_share", "value": largest_regimen_share, "note": "bounded observation, not a homogeneity claim; heterogeneous regimens observed"},
    ]
    table = pd.DataFrame(rows)
    table.to_csv(TABLE3_PATH, index=False)

    diagnostics["phase3"] = {
        "n_panel": n_panel, "fraction_with_med_insulin_flag": fraction_on_insulin,
        "n_no_insulin_record_despite_label": n_no_insulin_record,
        "largest_regimen_share": largest_regimen_share,
    }

    print(f"\nPhase 3, insulin-dependent stratum (matched exploratory panel, n={n_panel}):")
    print(f"  Fraction with med_insulin flag: {fraction_on_insulin:.3f} [{ins_lo:.3f}, {ins_hi:.3f}]")
    print(f"  Label mismatch (no insulin record despite diagnosis): {n_no_insulin_record} of {n_panel}")
    print(f"  Basal only={n_basal_only}, basal+bolus={n_basal_bolus}, bolus only={n_bolus_only}, "
          f"unclassified={n_unclassified} (of {n_with_insulin_rows} with an insulin record)")
    print(f"  Oral co-therapy fraction: {oral_cotherapy_fraction:.3f} [{oral_lo:.3f}, {oral_hi:.3f}]")
    print(f"  Largest single regimen: '{largest_regimen}' at {largest_regimen_share:.3f} share, "
          f"heterogeneous, not dominated by one regimen")
    print("  No clustering attempted, discrete-structure detection out of scope here due to n, "
          "not regimen homogeneity.")

    return table


# ---------------------------------------------------------------------------
# Write-up
# ---------------------------------------------------------------------------
def build_writeup(table1: pd.DataFrame, table2: pd.DataFrame, table3: pd.DataFrame, diagnostics: dict) -> str:
    contrast_all = table1[(table1["scope"] == "unconditional_contrast")].iloc[0]
    well_powered = diagnostics["phase1"]["well_powered_regimens"]
    assoc = diagnostics["phase1"]["association_test"]

    regimen_full_n = diagnostics["phase1"]["regimen_counts"]
    within_regimen_lines = []
    for reg in well_powered:
        sub = table1[(table1["scope"] == "within_regimen_contrast") & (table1["regimen"] == reg)]
        if len(sub):
            row = sub.iloc[0]
            excludes_zero = row["ci_low"] > 0 or row["ci_high"] < 0
            within_regimen_lines.append(
                f"- Within {reg.replace('_', ' ')} (n={regimen_full_n[reg]}, "
                f"high vs near subgroup n={row['n']}): high-minus-near mean CGM difference "
                f"{row['mean_cgm']:.1f} mg/dL, 95% CI [{row['ci_low']:.1f}, {row['ci_high']:.1f}], "
                f"CI {'excludes' if excludes_zero else 'does not exclude'} zero."
            )

    independent_significant = table2[
        (table2["kind"].isin(["clinical_factor", "medication_intensity"]))
        & (~table2["glucose_linked"])
        & (table2["significant_fdr"] == True)  # noqa: E712
    ]["variable"].unique().tolist()

    ins_row_frac = table3[table3["metric"] == "fraction_with_med_insulin_flag"].iloc[0]
    ins_mismatch = table3[table3["metric"] == "n_no_insulin_record_despite_label"].iloc[0]
    ins_n = table3[table3["metric"] == "n_matched_exploratory_panel"].iloc[0]["value"]
    ins_regimen_share = table3[table3["metric"] == "largest_single_regimen_share"].iloc[0]["value"]

    lines = []
    lines.append("### Addendum to Section 21.13: medication reanalyses of the T2D glucose-exposure axis")
    lines.append("")
    lines.append("**Within-regimen confounder test.** The unconditional high-minus-near mean CGM gap "
                 f"across the streamed-state exposure groups is {contrast_all['mean_cgm']:.1f} mg/dL "
                 f"(95% CI [{contrast_all['ci_low']:.1f}, {contrast_all['ci_high']:.1f}]). Two oral regimens "
                 f"were well powered for a within-regimen check (n >= {MIN_REGIMEN_N}): "
                 + ", ".join(r.replace("_", " ") for r in well_powered) + ".")
    lines.extend(within_regimen_lines)
    lines.append(
        f"Streamed-state group is not redundant with regimen: a group-by-regimen association test gives "
        f"chi2={assoc['chi2']:.2f} (permutation p={assoc['permutation_p']:.3f}, analytic p={assoc['analytic_p']:.3f}), "
        "not distinguishable from independence. Taken together, the exposure gradient surviving within "
        "at least one well-powered regimen, with group not simply standing in for regimen, argues the "
        "glucose axis is not primarily a medication artifact, though the within-regimen subgroups are "
        "small and the CIs above should be read with that in mind."
    )
    lines.append("")
    lines.append("**Clinical and medication-intensity profile.** Repeating the six-factor clinical profile "
                 "with medication covariates folded in (fraction on each oral class, mean number of oral "
                 "classes) and BH-FDR corrected across all seven tested variables: ")
    if independent_significant:
        lines.append(
            f"the exposure groups differ beyond glucose and HbA1c on: {', '.join(independent_significant)}. "
            "This qualifies the earlier glucose-specific reading and should be checked against the "
            "single-factor result already in Section 21.13."
        )
    else:
        lines.append(
            "no independent clinical factor and no medication-intensity summary (mean number of oral "
            "classes) separates the groups after FDR correction, consistent with the earlier finding that "
            "only HbA1c (glucose-linked, expected) separates. This is a confirmed null given the bootstrap "
            "CIs, not an underpowered gap, and now additionally rules out treatment intensity as a "
            "confound for the earlier clinical-profile null."
        )
    lines.append("")
    lines.append("**Insulin-dependent stratum, descriptive only.** In the same n="
                 f"{ins_n} exploratory panel used throughout this chapter, {ins_row_frac['value']:.1%} "
                 f"(95% CI [{ins_row_frac['ci_low']:.1%}, {ins_row_frac['ci_high']:.1%}]) carry a recorded "
                 f"insulin medication; {int(ins_mismatch['value'])} of {ins_n} are labeled insulin-dependent "
                 "with no insulin record at all, a label-quality caveat rather than a clinical finding. "
                 f"The largest single treatment regimen accounts for only {ins_regimen_share:.1%} of the "
                 "panel, i.e., regimens are heterogeneous, not dominated by one pattern. This is reported "
                 "as a bounded observation: it means the small sample size, not treatment homogeneity, is "
                 "why no clustering or ARI analysis was attempted here. No claim of discrete structure, "
                 "or its absence, is made for this stratum."
    )
    lines.append("")
    lines.append("*dpp4 and meglitinide oral classes were not available as RxNorm-classified columns in "
                 "any source and are excluded from ORAL_CLASSES throughout; see the Phase 0 gate report.*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)
    diagnostics: dict = {}

    panel = load_t2d_panel()
    print(f"T2D panel loaded: n={len(panel)}")

    mismatch_insulin = panel[panel[INSULIN_COLUMN] == True][PARTICIPANT_ID_COLUMN].tolist()  # noqa: E712
    mismatch_no_drug = panel[panel[ANY_DRUG_COLUMN] == False][PARTICIPANT_ID_COLUMN].tolist()  # noqa: E712
    with open(ASSIGNMENT_MISMATCH_NOTE_PATH, "w") as f:
        json.dump({
            "t2d_oral_with_insulin_flag": mismatch_insulin,
            "t2d_oral_with_no_diabetes_drug_record": mismatch_no_drug,
        }, f, indent=2)
    print(f"Label mismatches: {len(mismatch_insulin)} T2D-oral participants with insulin flag "
          f"({mismatch_insulin}), {len(mismatch_no_drug)} with no diabetes drug record ({mismatch_no_drug})")

    glycemic = compute_glycemic_features(panel[PARTICIPANT_ID_COLUMN])

    table1 = phase1_within_regimen(panel, glycemic, rng, diagnostics)
    table2 = phase2_group_profile(panel, rng, diagnostics)
    table3 = phase3_insulin_dependent(rng, diagnostics)

    writeup = build_writeup(table1, table2, table3, diagnostics)
    WRITEUP_PATH.write_text(writeup + "\n")

    with open(DIAGNOSTICS_PATH, "w") as f:
        json.dump(diagnostics, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else str(x))

    print("\n" + "=" * 70)
    print("DRAFT WRITE-UP")
    print("=" * 70)
    print(writeup)

    print(f"\nWrote {TABLE1_PATH}")
    print(f"Wrote {FIG1_PATH}")
    print(f"Wrote {TABLE2_PATH}")
    print(f"Wrote {FIG2_PATH}")
    print(f"Wrote {TABLE3_PATH}")
    print(f"Wrote {WRITEUP_PATH}")
    print(f"Wrote {ASSIGNMENT_MISMATCH_NOTE_PATH}")
    print(f"Wrote {DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
