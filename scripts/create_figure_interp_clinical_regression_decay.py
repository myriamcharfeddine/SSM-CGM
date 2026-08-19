"""Figure: confound-controlled clinical-regression delta R^2 across h_t snapshots.

Plotting only. Loads the Freedman-Lane permutation results table written by
scripts/t2d_confound_permutation_test.py and renders it; does not regenerate
or touch the regression itself. Follows the project figure style (seaborn
whitegrid, named constants, no magic numbers) used across the Master's
Thesis Overleaf project's figures/generated/*.pdf.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
RESULTS_PATH = (
    REPO_ROOT
    / "outputs/static_phenotype_trajectory_stratified_v2/confound_permutation_test"
    / "t2d_confound_permutation_results.csv"
)
OUTPUT_PATH = (
    REPO_ROOT / "ReportMasterThesis-overleaf/figures/generated/fig_interp_clinical_regression_decay_v1.pdf"
)

# ---- named constants, no inline magic numbers ----
COLOR_LINE = "#5BBABA"        # corrected/adjusted series
COLOR_BAND = "#5BBABA"        # same, low alpha
BAND_ALPHA = 0.18
COLOR_SIG = "#BA2828"         # positive/significant marker
COLOR_NONSIG = "#888888"      # de-emphasized/non-significant marker
COLOR_ZERO_LINE = "#000000"
COLOR_ANNOTATION = "#888888"
LINE_WIDTH = 2.0
MARKER_SIZE = 7
FIGURE_DPI = 200

TARGET_ORDER = [
    "participants_age", "bmi_baseline", "hba1c_percent_baseline",
    "c_peptide_ngml_baseline", "tg_hdl_ratio",
    "waist_to_hip_ratio_baseline",
]
TARGET_LABELS = {
    "participants_age": "Age",
    "bmi_baseline": "BMI",
    "hba1c_percent_baseline": "HbA1c",
    "c_peptide_ngml_baseline": "C-peptide",
    "tg_hdl_ratio": "TG/HDL ratio",
    "waist_to_hip_ratio_baseline": "Waist-to-hip ratio",
}
HOURS = [0, 6, 12, 24, 48]


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)
    df["bonferroni_pass"] = df["bonferroni_pass"].astype(bool)

    negative_targets = sorted(df.loc[df["delta_r2"] < 0, "target"].unique().tolist())
    print(f"delta_r2 range: [{df['delta_r2'].min():.4f}, {df['delta_r2'].max():.4f}]")
    if negative_targets:
        print(f"delta_r2 goes negative for: {negative_targets}")
    else:
        print("delta_r2 never goes negative for any target.")
    print("Per-target delta_r2 range (for the shared- vs per-panel y-axis question):")
    for target in TARGET_ORDER:
        sub = df[df["target"] == target]
        print(f"  {TARGET_LABELS[target]}: [{sub['delta_r2'].min():.4f}, {sub['delta_r2'].max():.4f}] "
              f"(ci [{sub['ci_low'].min():.4f}, {sub['ci_high'].max():.4f}])")

    sns.set_style("whitegrid")
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=False)
    fig.suptitle(
        "Clinical-feature signal in h_t beyond glucose severity",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, target in zip(axes.flat, TARGET_ORDER):
        sub = df[df["target"] == target].sort_values("hour")
        ax.axhline(0, color=COLOR_ZERO_LINE, linewidth=1.0, zorder=1)
        ax.plot(
            sub["hour"], sub["delta_r2"],
            color=COLOR_LINE, linewidth=LINE_WIDTH, zorder=3,
        )
        ax.fill_between(
            sub["hour"], sub["ci_low"], sub["ci_high"],
            color=COLOR_BAND, alpha=BAND_ALPHA, zorder=2,
        )
        sig = sub["bonferroni_pass"]
        ax.scatter(
            sub.loc[sig, "hour"], sub.loc[sig, "delta_r2"],
            color=COLOR_SIG, s=MARKER_SIZE**2, zorder=4,
            label="Bonferroni significant",
        )
        ax.scatter(
            sub.loc[~sig, "hour"], sub.loc[~sig, "delta_r2"],
            facecolors="none", edgecolors=COLOR_NONSIG, s=MARKER_SIZE**2,
            zorder=4, label="Not significant",
        )
        ax.set_title(TARGET_LABELS[target], fontsize=11, fontweight="bold")
        ax.set_xticks(HOURS)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)
        if target == "hba1c_percent_baseline":
            ax.annotate(
                "Expected near zero,\noverlaps severity control",
                xy=(0.5, 0.08), xycoords="axes fraction",
                fontsize=8, color=COLOR_ANNOTATION, ha="center",
            )

    for ax in axes[1, :]:
        ax.set_xlabel("Hours since stream start")
    for ax in axes[:, 0]:
        ax.set_ylabel("Delta R² (severity-controlled)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=2, frameon=True, fontsize=9,
    )

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
