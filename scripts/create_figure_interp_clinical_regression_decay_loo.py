"""Figure: confound-controlled clinical-regression delta R^2 across h_t
snapshots, genuine leave-one-out out-of-sample scoring.

Plotting only. Loads the LOO diagnostic table written by
scripts/t2d_confound_loo_diagnostic.py and renders delta_r2_loo (not the
in-sample delta_r2 used by fig_interp_clinical_regression_decay_v1.pdf).
Does not regenerate or touch the regression itself.

No CI band or significance-based marker coloring: the diagnostic table's
ci_low/ci_high and p_value/bonferroni_pass columns are carried over
unchanged from the original in-sample permutation run and are not valid
for the LOO point estimate (mismatched estimator), so this figure plots
the LOO delta R^2 alone rather than pairing it with a CI/significance
encoding computed for a different quantity. Follows the project figure
style (seaborn whitegrid, named constants, no magic numbers) used across
the Master's Thesis Overleaf project's figures/generated/*.pdf.
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
    / "t2d_confound_loo_diagnostic.csv"
)
OUTPUT_PATH = (
    REPO_ROOT / "ReportMasterThesis-overleaf/figures/generated/fig_interp_clinical_regression_decay_v2.pdf"
)

# ---- named constants, no inline magic numbers ----
COLOR_LINE = "#5BBABA"
MARKER_FACE = "#5BBABA"
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

    negative_targets = sorted(df.loc[df["delta_r2_loo"] < 0, "target"].unique().tolist())
    print(f"delta_r2_loo range: [{df['delta_r2_loo'].min():.4f}, {df['delta_r2_loo'].max():.4f}]")
    if negative_targets:
        print(f"delta_r2_loo goes negative for: {negative_targets}")
    print("Per-target delta_r2_loo range:")
    for target in TARGET_ORDER:
        sub = df[df["target"] == target]
        print(f"  {TARGET_LABELS[target]}: [{sub['delta_r2_loo'].min():.4f}, {sub['delta_r2_loo'].max():.4f}]")

    sns.set_style("whitegrid")
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=False)
    fig.suptitle(
        "Clinical-feature signal in h_t beyond glucose severity, leave-one-out scoring",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, target in zip(axes.flat, TARGET_ORDER):
        sub = df[df["target"] == target].sort_values("hour")
        ax.axhline(0, color=COLOR_ZERO_LINE, linewidth=1.0, zorder=1)
        ax.plot(
            sub["hour"], sub["delta_r2_loo"],
            color=COLOR_LINE, linewidth=LINE_WIDTH, zorder=3,
        )
        ax.scatter(
            sub["hour"], sub["delta_r2_loo"],
            color=MARKER_FACE, s=MARKER_SIZE**2, zorder=4,
        )
        ax.set_title(TARGET_LABELS[target], fontsize=11, fontweight="bold")
        ax.set_xticks(HOURS)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)
        if target == "hba1c_percent_baseline":
            # LOO delta R^2 for HbA1c sits in a narrow, near-flat band close to the
            # bottom of its own axis range, so the note goes near the top instead
            # of the bottom (which fit fine in the wider-range in-sample figure).
            ax.annotate(
                "Expected near zero,\noverlaps severity control",
                xy=(0.5, 0.85), xycoords="axes fraction",
                fontsize=8, color=COLOR_ANNOTATION, ha="center",
            )

    for ax in axes[1, :]:
        ax.set_xlabel("Hours since stream start")
    for ax in axes[:, 0]:
        ax.set_ylabel("Delta R² (severity-controlled, LOO)")

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
