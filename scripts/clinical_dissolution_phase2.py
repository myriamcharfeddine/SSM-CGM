"""Phase 2: clinical-label dissolution curve.

Tests whether the frozen within-subtype clinical partition loses separation
in the latent geometry as the state streams. Frozen labels only, no
reclustering. Reuses saved h0 and h_t snapshots and frozen clinical labels.
No forward pass, no model retraining. Read-only on the canonical checkpoint
and the multimodal parquet.
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
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
BRANCH = "aireadi-ssmcgm-stream-report"
OUTPUT_ROOT = REPO_ROOT / "outputs/static_phenotype_trajectory"
OUTPUT_SUBDIR = REPO_ROOT / "outputs/static_phenotype_trajectory/global_realignment"
CANONICAL_SPLIT = "adapt6h_seed42"

SNAPSHOT_HOURS = [0, 6, 12, 24, 48]
H0_PATH = OUTPUT_ROOT / "step2/h0_matrix.parquet"
SNAPSHOT_DIR = (
    REPO_ROOT
    / "outputs/static_phenotype_trajectory_stratified_v2/phase4_time_resolved_extension/snapshots"
)
HT_PATH_BY_HOUR = {
    6: SNAPSHOT_DIR / "h_t_full_hour06.parquet",
    12: SNAPSHOT_DIR / "h_t_full_hour12.parquet",
    24: SNAPSHOT_DIR / "h_t_full_hour24.parquet",
    48: SNAPSHOT_DIR / "h_t_full_hour48.parquet",
}
CLINICAL_ASSIGNMENTS_PATH = (
    REPO_ROOT
    / "outputs/static_phenotype_trajectory_stratified_v2/phase2_h0_preservation/test_assignments.csv"
)

DIAGNOSTIC_SUBTYPES = ["Healthy", "Prediabetes", "T2D oral non-insulin", "Insulin-dependent"]
STRATUM_KEY_TO_DISPLAY = {
    "healthy": "Healthy",
    "pre_diabetes": "Prediabetes",
    "t2d_oral_non_insulin": "T2D oral non-insulin",
    "insulin_dependent": "Insulin-dependent",
}
EXPLORATORY_SUBTYPE = "Insulin-dependent"
TOP_LAYER_SUBTYPE = "T2D oral non-insulin"

PCA_VARIANCE_TARGET = 0.95
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = (2.5, 97.5)
RANDOM_SEED = 42
N_SHUFFLE_SEEDS = 20
MAX_BOOTSTRAP_RETRIES = 50

PARTICIPANT_ID_COLUMN = "participant_id"

STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
SUBTYPE_COLOR = dict(zip(DIAGNOSTIC_SUBTYPES, STRATUM_COLORS[:4]))
COLOR_CONTROL = "#888888"
BAND_ALPHA = 0.18
CONTROL_BAND_ALPHA = 0.15

FIG2_SILHOUETTE_PATH = OUTPUT_SUBDIR / "clinical_dissolution_curve.png"
FIG2_PSEUDOF_PATH = OUTPUT_SUBDIR / "clinical_dissolution_curve_pseudoF.png"
TABLE2_PATH = OUTPUT_SUBDIR / "clinical_dissolution_curve.csv"
DIAGNOSTICS_PATH = OUTPUT_SUBDIR / "phase2_pca_and_bootstrap_diagnostics.json"

sns.set_style("whitegrid")
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 0.8


# ---------------------------------------------------------------------------
# Data loading (mirrors global_realignment_phase1.py)
# ---------------------------------------------------------------------------
def load_snapshot_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    ids = df[PARTICIPANT_ID_COLUMN].astype(str).to_numpy()
    dim_cols = sorted((c for c in df.columns if c.isdigit()), key=int)
    matrix = df[dim_cols].to_numpy(dtype=np.float32)
    return ids, matrix


def build_matched_panel() -> pd.DataFrame:
    h0_full = pd.read_parquet(H0_PATH, columns=[PARTICIPANT_ID_COLUMN, "split"])
    ht48_full = pd.read_parquet(HT_PATH_BY_HOUR[48], columns=[PARTICIPANT_ID_COLUMN, "split"])
    assert list(h0_full[PARTICIPANT_ID_COLUMN]) == list(ht48_full[PARTICIPANT_ID_COLUMN]), \
        "h0 and h_t snapshot ordering diverged"
    test_ids = h0_full.loc[h0_full["split"] == "test", PARTICIPANT_ID_COLUMN].astype(str)
    snapshot_set = set(test_ids)

    assignments = pd.read_csv(CLINICAL_ASSIGNMENTS_PATH)
    assignments[PARTICIPANT_ID_COLUMN] = assignments[PARTICIPANT_ID_COLUMN].astype(str)
    assignments["diagnostic_subtype"] = assignments["canonical_stratum"].map(STRATUM_KEY_TO_DISPLAY)
    assert assignments["diagnostic_subtype"].isna().sum() == 0, "unmapped canonical_stratum value"

    matched = assignments[assignments[PARTICIPANT_ID_COLUMN].isin(snapshot_set)].copy()
    matched = matched.rename(columns={"display_cluster": "clinical_cluster"})
    matched = matched[[PARTICIPANT_ID_COLUMN, "diagnostic_subtype", "clinical_cluster"]]
    matched = matched.sort_values(PARTICIPANT_ID_COLUMN).reset_index(drop=True)
    print(f"Matched panel N = {len(matched)}")
    return matched


def align_matrix_to_panel(ids: np.ndarray, matrix: np.ndarray, panel_ids: pd.Series) -> np.ndarray:
    id_to_row = {pid: i for i, pid in enumerate(ids)}
    positions = np.array([id_to_row[pid] for pid in panel_ids], dtype=int)
    return matrix[positions]


def load_all_hour_matrices(panel_ids: pd.Series) -> dict:
    out = {}
    h0_ids, h0_full = load_snapshot_matrix(H0_PATH)
    out[0] = align_matrix_to_panel(h0_ids, h0_full, panel_ids)
    for hour, path in HT_PATH_BY_HOUR.items():
        ids, full = load_snapshot_matrix(path)
        out[hour] = align_matrix_to_panel(ids, full, panel_ids)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def safe_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels > len(labels) - 1:
        return np.nan
    return float(silhouette_score(x, labels, metric="euclidean"))


def safe_calinski_harabasz(x: np.ndarray, labels: np.ndarray) -> float:
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels > len(labels) - 1:
        return np.nan
    return float(calinski_harabasz_score(x, labels))


def bootstrap_metric(x: np.ndarray, labels: np.ndarray, metric_fn, rng: np.random.RandomState) -> tuple[float, float]:
    n = len(labels)
    replicates = []
    attempts = 0
    while len(replicates) < N_BOOTSTRAP and attempts < N_BOOTSTRAP * MAX_BOOTSTRAP_RETRIES:
        attempts += 1
        idx = rng.randint(0, n, size=n)
        value = metric_fn(x[idx], labels[idx])
        if not np.isnan(value):
            replicates.append(value)
    if not replicates:
        return np.nan, np.nan
    lo, hi = np.percentile(replicates, BOOTSTRAP_CI)
    return float(lo), float(hi)


def shuffled_control_mean(x: np.ndarray, labels: np.ndarray, metric_fn, base_seed: int) -> float:
    values = []
    for seed_offset in range(N_SHUFFLE_SEEDS):
        rng = np.random.RandomState(base_seed + seed_offset)
        shuffled = rng.permutation(labels)
        value = metric_fn(x, shuffled)
        if not np.isnan(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)
    diagnostics: dict = {"pca": {}}

    panel = build_matched_panel()
    panel_ids = panel[PARTICIPANT_ID_COLUMN]
    hour_matrices = load_all_hour_matrices(panel_ids)

    rows = []
    for subtype in DIAGNOSTIC_SUBTYPES:
        mask = (panel["diagnostic_subtype"] == subtype).to_numpy()
        n_sub = int(mask.sum())
        exploratory = subtype == EXPLORATORY_SUBTYPE
        clinical_labels = panel.loc[mask, "clinical_cluster"].to_numpy()
        print(f"\nSubtype {subtype}{' (exploratory)' if exploratory else ''}: n={n_sub}")

        subtype_hour_matrices = {hour: hour_matrices[hour][mask] for hour in SNAPSHOT_HOURS}
        pooled = np.concatenate([subtype_hour_matrices[hour] for hour in SNAPSHOT_HOURS], axis=0)

        n_components_cap = max(1, pooled.shape[0] - 1)
        pca_probe = PCA(n_components=min(n_components_cap, pooled.shape[1]), svd_solver="full",
                         random_state=RANDOM_SEED)
        pca_probe.fit(pooled)
        cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET) + 1)
        n_components = min(n_components, n_components_cap)
        diagnostics["pca"][subtype] = {
            "n_participants": n_sub, "n_pooled_rows": int(pooled.shape[0]),
            "n_components": n_components, "variance_explained": float(cumvar[n_components - 1]),
        }
        print(f"  PCA fit on pooled snapshots ({pooled.shape[0]} rows): {n_components} components "
              f"reach {cumvar[n_components - 1]:.3f} variance (target {PCA_VARIANCE_TARGET})")

        for hour in SNAPSHOT_HOURS:
            reduced = pca_probe.transform(subtype_hour_matrices[hour])[:, :n_components]

            silhouette_point = safe_silhouette(reduced, clinical_labels)
            silhouette_lo, silhouette_hi = bootstrap_metric(reduced, clinical_labels, safe_silhouette, rng)
            silhouette_control = shuffled_control_mean(reduced, clinical_labels, safe_silhouette,
                                                         base_seed=RANDOM_SEED + hour)

            pseudo_f_point = safe_calinski_harabasz(reduced, clinical_labels)
            pseudo_f_lo, pseudo_f_hi = bootstrap_metric(reduced, clinical_labels, safe_calinski_harabasz, rng)
            pseudo_f_control = shuffled_control_mean(reduced, clinical_labels, safe_calinski_harabasz,
                                                       base_seed=RANDOM_SEED + hour)

            rows.append({
                "subtype": subtype, "exploratory": exploratory, "hour": hour, "metric": "silhouette",
                "value": silhouette_point, "ci_low": silhouette_lo, "ci_high": silhouette_hi,
                "shuffled_control_mean": silhouette_control, "n": n_sub,
            })
            rows.append({
                "subtype": subtype, "exploratory": exploratory, "hour": hour, "metric": "pseudo_f",
                "value": pseudo_f_point, "ci_low": pseudo_f_lo, "ci_high": pseudo_f_hi,
                "shuffled_control_mean": pseudo_f_control, "n": n_sub,
            })
            print(f"  hour={hour:>2}: silhouette={silhouette_point:.4f} [{silhouette_lo:.4f}, {silhouette_hi:.4f}] "
                  f"control~{silhouette_control:.4f} | pseudo_f={pseudo_f_point:.2f} "
                  f"[{pseudo_f_lo:.2f}, {pseudo_f_hi:.2f}] control~{pseudo_f_control:.2f}")

    table = pd.DataFrame(rows)
    table.to_csv(TABLE2_PATH, index=False)
    with open(DIAGNOSTICS_PATH, "w") as f:
        json.dump(diagnostics, f, indent=2)

    make_figure(table, metric="silhouette", ylabel="Clinical-label silhouette",
                path=FIG2_SILHOUETTE_PATH)
    make_figure(table, metric="pseudo_f", ylabel="Clinical-label Calinski-Harabasz pseudo-F",
                path=FIG2_PSEUDOF_PATH)

    print_summary(table)
    print(f"\nWrote {TABLE2_PATH}")
    print(f"Wrote {FIG2_SILHOUETTE_PATH}")
    print(f"Wrote {FIG2_PSEUDOF_PATH}")
    print(f"Wrote {DIAGNOSTICS_PATH}")


def make_figure(table: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    sub_table = table[table["metric"] == metric]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    plot_order = [s for s in DIAGNOSTIC_SUBTYPES if s != TOP_LAYER_SUBTYPE] + [TOP_LAYER_SUBTYPE]

    control_by_hour = sub_table.groupby("hour")["shuffled_control_mean"].mean().reindex(SNAPSHOT_HOURS)
    control_band_half_width = sub_table.groupby("hour")["shuffled_control_mean"].std().reindex(SNAPSHOT_HOURS).fillna(0.0)
    ax.fill_between(
        SNAPSHOT_HOURS,
        (control_by_hour - control_band_half_width).to_numpy(),
        (control_by_hour + control_band_half_width).to_numpy(),
        color=COLOR_CONTROL, alpha=CONTROL_BAND_ALPHA, label="Shuffled label control", zorder=1,
    )
    ax.plot(SNAPSHOT_HOURS, control_by_hour.to_numpy(), color=COLOR_CONTROL, linewidth=1.0,
            linestyle="--", zorder=1)

    for subtype in plot_order:
        rows_sub = sub_table[sub_table["subtype"] == subtype].set_index("hour").reindex(SNAPSHOT_HOURS)
        exploratory = bool(rows_sub["exploratory"].iloc[0])
        label = subtype + (" (exploratory)" if exploratory else "")
        color = SUBTYPE_COLOR[subtype]
        linestyle = "--" if exploratory else "-"
        zorder = 3 if subtype == TOP_LAYER_SUBTYPE else 2
        ax.fill_between(SNAPSHOT_HOURS, rows_sub["ci_low"].to_numpy(), rows_sub["ci_high"].to_numpy(),
                         color=color, alpha=BAND_ALPHA, zorder=zorder)
        ax.plot(SNAPSHOT_HOURS, rows_sub["value"].to_numpy(), color=color, linewidth=2.0,
                linestyle=linestyle, marker="o", markersize=4, label=label, zorder=zorder + 1)

    ax.set_xlabel("Elapsed hours")
    ax.set_ylabel(ylabel)
    ax.set_xticks(SNAPSHOT_HOURS)

    # Title reflects the pooled pattern across the three primary subtypes
    # (Insulin-dependent excluded, exploratory), not any single line.
    primary_subtypes = [s for s in DIAGNOSTIC_SUBTYPES if s != EXPLORATORY_SUBTYPE]
    primary_rows = sub_table[sub_table["subtype"].isin(primary_subtypes)]
    hour0_avg = primary_rows.loc[primary_rows["hour"] == 0, "value"].mean()
    hour48_avg = primary_rows.loc[primary_rows["hour"] == 48, "value"].mean()
    control0_avg = primary_rows.loc[primary_rows["hour"] == 0, "shuffled_control_mean"].mean()
    control48_avg = primary_rows.loc[primary_rows["hour"] == 48, "shuffled_control_mean"].mean()

    above_control_at_0 = hour0_avg > control0_avg
    declines = hour0_avg > hour48_avg
    near_control_at_48 = abs(hour48_avg - control48_avg) < abs(hour0_avg - control0_avg)
    if above_control_at_0 and declines and near_control_at_48:
        title = "Clinical separation is highest at hour 0 and collapses toward the shuffled control"
    elif above_control_at_0 and declines:
        title = "Clinical separation declines from hour 0 but stays above the shuffled control"
    elif above_control_at_0:
        title = "Clinical separation persists above the shuffled control across streaming"
    else:
        title = "Clinical separation does not clearly exceed the shuffled control at hour 0"
    ax.set_title(title, fontsize=11)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.legend(frameon=False, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_summary(table: pd.DataFrame) -> None:
    print("\n=== Clinical-label dissolution curve ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(table.to_string(index=False))

    print("\nPlain-language readout (prediction: separation highest at hour 0 in h0, collapses toward a low "
          "plateau, stays above the shuffled control if coarse structure survives):")
    for subtype in DIAGNOSTIC_SUBTYPES:
        tag = " (exploratory)" if subtype == EXPLORATORY_SUBTYPE else ""
        sil = table[(table["subtype"] == subtype) & (table["metric"] == "silhouette")].set_index("hour").reindex(
            [0, 48]
        )
        hour0, hour48 = sil.loc[0], sil.loc[48]
        declined = hour48["value"] < hour0["ci_low"]
        above_control_at_48 = hour48["ci_low"] > hour48["shuffled_control_mean"]
        if declined and above_control_at_48:
            verdict = "separation declines but a coarse residual above the shuffled control survives to 48h"
        elif declined:
            verdict = "separation declines and is indistinguishable from the shuffled control by 48h"
        else:
            verdict = "separation does not show a confirmed decline by 48h at this CI"
        print(f"{subtype}{tag}: hour0 silhouette={hour0['value']:.3f} [{hour0['ci_low']:.3f}, "
              f"{hour0['ci_high']:.3f}], hour48 silhouette={hour48['value']:.3f} [{hour48['ci_low']:.3f}, "
              f"{hour48['ci_high']:.3f}], control~{hour48['shuffled_control_mean']:.3f}. {verdict}.")


if __name__ == "__main__":
    main()
