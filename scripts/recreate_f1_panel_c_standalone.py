"""Render a standalone, fully visible version of F1 Panel C.

This figure-only script reads the saved Panel C contrast table. It does not
recompute estimates, confidence intervals, bootstrap results, FDR decisions,
states, neighborhoods, or models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(
    "/home/myriamcharfeddine/CGM/SSM-CGM/outputs/"
    "static_phenotype_trajectory_stratified_v2/neighbor_transition_drivers/"
    "direct_variable_level_figure_v2"
)
TABLE = ROOT / "tables/panel_C_gained_vs_matched_features.csv"
FIGURES = ROOT / "figures"

SUBTYPES = ["healthy", "pre_diabetes", "t2d_oral_non_insulin", "insulin_dependent"]
SUBTYPE_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes": "Prediabetes",
    "t2d_oral_non_insulin": "T2D oral non-insulin",
    "insulin_dependent": "Insulin-dependent, exploratory",
}
COLORS = {
    "healthy": "#17365D",
    "pre_diabetes": "#15989C",
    "t2d_oral_non_insulin": "#BE263B",
    "insulin_dependent": "#777777",
}

# Exact compact subset used in Panel C of the source F1 figure.
FEATURES = [
    "static__participants_age",
    "static__bmi_baseline",
    "static__hba1c_percent_baseline",
    "static__c_peptide_ngml_baseline",
    "static__tg_hdl_ratio",
    "dynamic__cgm_mean",
    "dynamic__cgm_sd",
    "dynamic__cgm_time_in_range",
    "dynamic__cgm_time_above_180",
    "dynamic__cgm_masd",
    "dynamic__heart_rate_mean_summary",
    "dynamic__spo2_mean_summary",
    "dynamic__total_steps",
    "dynamic__sleep_rem_proportion",
    "dynamic__sleep_continuity",
]
LABELS = {
    "static__participants_age": "Study-visit age",
    "static__bmi_baseline": "BMI",
    "static__hba1c_percent_baseline": "HbA1c",
    "static__c_peptide_ngml_baseline": "C-peptide",
    "static__tg_hdl_ratio": "TG/HDL",
    "dynamic__cgm_mean": "Mean CGM",
    "dynamic__cgm_sd": "CGM SD",
    "dynamic__cgm_time_in_range": "Time in range",
    "dynamic__cgm_time_above_180": "Time above 180 mg/dL",
    "dynamic__cgm_masd": "Mean absolute successive difference",
    "dynamic__heart_rate_mean_summary": "Mean heart rate",
    "dynamic__spo2_mean_summary": "Mean SpO2",
    "dynamic__total_steps": "Total steps",
    "dynamic__sleep_rem_proportion": "REM-sleep proportion",
    "dynamic__sleep_continuity": "Sleep continuity",
}


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="both", color="#D8D8D8", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)


def main() -> None:
    if not TABLE.exists():
        raise FileNotFoundError(f"Missing saved Panel C table: {TABLE}")

    source = pd.read_csv(TABLE)
    data = source[
        (source["hour"] == 48)
        & (source["comparison"] == "Gained_vs_Matched")
        & source["feature"].isin(FEATURES)
    ].copy()

    expected_pairs = {(subtype, feature) for subtype in SUBTYPES for feature in FEATURES}
    saved_pairs = set(zip(data["canonical_stratum"], data["feature"]))
    missing = sorted(expected_pairs - saved_pairs)
    if missing:
        raise RuntimeError(f"Missing saved Panel C values: {missing}")

    low = float(data["ci_low"].min())
    high = float(data["ci_high"].max())
    span = max(high - low, 0.1)
    xlim = (min(0.0, low) - 0.07 * span, max(0.0, high) + 0.07 * span)

    fig, axes = plt.subplots(1, 4, figsize=(22, 10), sharey=True, facecolor="white")
    y = np.arange(len(FEATURES))

    for index, (ax, subtype) in enumerate(zip(axes, SUBTYPES)):
        q = (
            data[data["canonical_stratum"] == subtype]
            .set_index("feature")
            .reindex(FEATURES)
            .reset_index()
        )
        color = COLORS[subtype]
        for yy, row in zip(y, q.itertuples(index=False)):
            filled = bool(row.fdr_supported)
            ax.errorbar(
                row.estimate,
                yy,
                xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                fmt="o",
                markersize=7,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.3,
                ecolor=color,
                elinewidth=1.3,
                capsize=3.2,
                zorder=3,
            )
        ax.axvline(0, color="#111111", linewidth=1.0, zorder=2)
        ax.set_xlim(*xlim)
        ax.set_ylim(len(FEATURES) - 0.4, -0.6)
        ax.set_title(SUBTYPE_LABELS[subtype], fontsize=11.5, fontweight="bold", pad=12)
        ax.set_xlabel("Gained minus matched\npairwise similarity", fontsize=10)
        ax.set_yticks(y)
        if index == 0:
            ax.set_yticklabels([LABELS[feature] for feature in FEATURES], fontsize=9.5)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="x", labelsize=9)
        style_axis(ax)

    legend = [
        Line2D(
            [0], [0], marker="o", color="#17365D", markerfacecolor="#17365D",
            linewidth=0, markersize=7,
            label="Filled: saved 95% participant-bootstrap CI excludes 0 and BH-FDR q < 0.05",
        ),
        Line2D(
            [0], [0], marker="o", color="#17365D", markerfacecolor="white",
            linewidth=0, markersize=7, label="Hollow: saved rule not satisfied",
        ),
    ]
    fig.legend(
        handles=legend, frameon=False, ncol=2, fontsize=9.5,
        loc="upper center", bbox_to_anchor=(0.61, 0.925),
    )
    fig.suptitle(
        "C  Static and dynamic similarities distinguishing gained neighbors from matched non-neighbors",
        fontsize=16, fontweight="bold", y=0.98,
    )
    fig.text(
        0.61, 0.035,
        "Positive: gained neighbors are more similar than matched non-neighbors.   "
        "Negative: gained neighbors are less similar than matched non-neighbors.\n"
        "Estimates, intervals, and marker decisions are copied from the saved participant-level 48-hour contrast table. "
        "The insulin-dependent stratum is exploratory.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.23, right=0.99, top=0.86, bottom=0.14, wspace=0.08)

    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / "figure_F1_panel_C_gained_vs_matched_standalone.png"
    pdf = FIGURES / "figure_F1_panel_C_gained_vs_matched_standalone.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
