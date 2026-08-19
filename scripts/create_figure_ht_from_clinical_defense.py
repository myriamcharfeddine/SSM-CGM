"""Figure: h_t explained by clinical profile, out-of-sample, single panel.

Defense-slide figure only, not the thesis appendix. Plotting only -- loads
scripts/t2d_confound_h_on_clinical_regression.py's output table and renders
it, does not regenerate the regression. Uses R2_loo exclusively; R2_insample
is not read for any axis or annotation (that column is already known to be
inflated -- see the LOO diagnostic that motivated this whole reverse-
direction experiment).
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
    / "t2d_confound_h_on_clinical_results.csv"
)
OUTPUT_PATH = REPO_ROOT / "figures/generated/fig_interp_ht_from_clinical_defense_v1.pdf"

# ---- named constants, no inline magic numbers ----
COLOR_LINE = "#5BBABA"
COLOR_SIG = "#BA2828"
COLOR_NONSIG = "#888888"
COLOR_ZERO_LINE = "#000000"
COLOR_ANNOTATION = "#888888"
LINE_WIDTH = 2.0
MARKER_SIZE = 9
FIGURE_DPI = 200

HOURS = [0, 6, 12, 24, 48]
ANNOTATION_TEXT = "Mechanically expected, h0 is a direct\nfunction of these same clinical inputs"


def main() -> None:
    df = pd.read_csv(RESULTS_PATH, usecols=["hour", "R2_loo", "p_value", "pass_at_0.01"])
    df = df.sort_values("hour").reset_index(drop=True)
    df["pass_at_0.01"] = df["pass_at_0.01"].astype(bool)

    sns.set_style("whitegrid")
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color=COLOR_ZERO_LINE, linewidth=1.0, zorder=1)
    ax.plot(df["hour"], df["R2_loo"], color=COLOR_LINE, linewidth=LINE_WIDTH, zorder=2)

    sig = df["pass_at_0.01"]
    ax.scatter(
        df.loc[sig, "hour"], df.loc[sig, "R2_loo"],
        color=COLOR_SIG, s=MARKER_SIZE ** 2, zorder=3,
    )
    ax.scatter(
        df.loc[~sig, "hour"], df.loc[~sig, "R2_loo"],
        facecolors="none", edgecolors=COLOR_NONSIG, s=MARKER_SIZE ** 2, linewidths=1.5, zorder=3,
    )

    hour0 = df.loc[df["hour"] == 0].iloc[0]
    ax.annotate(
        ANNOTATION_TEXT,
        xy=(hour0["hour"], hour0["R2_loo"]),
        xytext=(10, -0.10), textcoords="offset fontsize",
        fontsize=9, color=COLOR_ANNOTATION,
    )

    ax.set_title("Clinical profile predicts h_t only at initialization", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hours since stream start")
    ax.set_ylabel("Delta R2, h_t explained by clinical profile (out-of-sample)")
    ax.set_xticks(HOURS)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
