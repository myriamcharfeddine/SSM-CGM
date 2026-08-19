"""Render standalone, fully visible versions of F1 Panels A and B.

Figure-only script: reads the already-saved panel tables and does not recompute
states, neighborhoods, models, bootstrap intervals, or inferential decisions.
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
TABLE_ROOT = ROOT / "tables"
FIGURE_ROOT = ROOT / "figures"

PANEL_A_TABLE = TABLE_ROOT / "panel_A_retention_over_time.csv"
PANEL_B_TABLE = TABLE_ROOT / "panel_B_static_retained_vs_lost.csv"

SUBTYPES = [
    "healthy",
    "pre_diabetes",
    "t2d_oral_non_insulin",
    "insulin_dependent",
]
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
STATIC_FEATURES = [
    "static__participants_age",
    "static__bmi_baseline",
    "static__hba1c_percent_baseline",
    "static__c_peptide_ngml_baseline",
    "static__tg_hdl_ratio",
    "static__waist_to_hip_ratio_baseline",
    "static__triglycerides_mgdl_baseline",
    "static__hdl_cholesterol_mgdl_baseline",
    "static__ldl_cholesterol_mgdl_baseline",
    "static__clinical_systolic_bp_mmhg_baseline",
    "static__clinical_diastolic_bp_mmhg_baseline",
    "static__med_any_diabetes_drug",
    "static__med_metformin",
    "static__med_insulin",
    "static__med_glp1_or_gip_glp1",
    "static__med_sglt2",
    "static__med_sulfonylurea",
    "static__med_thiazolidinedione",
    "static__participants_clinical_site",
    "static__demo_sex_at_birth",
]
FEATURE_LABELS = {
    "static__participants_age": "Study-visit age",
    "static__bmi_baseline": "BMI",
    "static__hba1c_percent_baseline": "HbA1c",
    "static__c_peptide_ngml_baseline": "C-peptide",
    "static__tg_hdl_ratio": "TG/HDL",
    "static__waist_to_hip_ratio_baseline": "Waist-to-hip ratio",
    "static__triglycerides_mgdl_baseline": "Triglycerides",
    "static__hdl_cholesterol_mgdl_baseline": "HDL cholesterol",
    "static__ldl_cholesterol_mgdl_baseline": "LDL cholesterol",
    "static__clinical_systolic_bp_mmhg_baseline": "Systolic BP",
    "static__clinical_diastolic_bp_mmhg_baseline": "Diastolic BP",
    "static__med_any_diabetes_drug": "Any diabetes-drug match",
    "static__med_metformin": "Metformin match",
    "static__med_insulin": "Insulin match",
    "static__med_glp1_or_gip_glp1": "GLP-1/GIP match",
    "static__med_sglt2": "SGLT2 match",
    "static__med_sulfonylurea": "Sulfonylurea match",
    "static__med_thiazolidinedione": "TZD match",
    "static__participants_clinical_site": "Clinical-site match",
    "static__demo_sex_at_birth": "Sex-at-birth match",
}


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="both", color="#D8D8D8", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_a(retention: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.2), facecolor="white")
    style_axis(ax)
    for subtype in SUBTYPES:
        d = retention[retention["canonical_stratum"] == subtype].sort_values("hour")
        color = COLORS[subtype]
        ax.plot(
            d["hour"], d["estimate"], marker="o", markersize=6.5,
            linewidth=2.4, color=color, label=SUBTYPE_LABELS[subtype], zorder=3,
        )
        ax.fill_between(
            d["hour"], d["ci_low"], d["ci_high"],
            color=color, alpha=0.16, linewidth=0, zorder=2,
        )
    ax.set_xticks([6, 12, 24, 48])
    ax.set_xlim(4, 50)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("Elapsed streaming time (hours)", fontsize=12)
    ax.set_ylabel("Retained-neighbor fraction", fontsize=12)
    ax.set_title(
        "A  How much of the original neighborhood remains after streaming?",
        loc="left", fontsize=15, fontweight="bold", pad=18,
    )
    ax.legend(
        title="Clinical subtype", frameon=False, ncol=2, fontsize=9.5,
        title_fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.01),
    )
    fig.text(
        0.5, 0.018,
        "Points are participant-level means; shaded bands are the saved 95% participant-bootstrap intervals. "
        "The insulin-dependent stratum is exploratory.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.83, bottom=0.15)
    save(fig, "figure_F1_panel_A_retention_standalone")


def panel_b(contrasts: pd.DataFrame) -> None:
    d = contrasts[
        (contrasts["hour"] == 48)
        & (contrasts["comparison"] == "Retained_vs_Lost")
        & contrasts["feature"].isin(STATIC_FEATURES)
    ].copy()
    expected = len(SUBTYPES) * len(STATIC_FEATURES)
    if len(d) != expected:
        missing = sorted(
            set((s, f) for s in SUBTYPES for f in STATIC_FEATURES)
            - set(zip(d["canonical_stratum"], d["feature"]))
        )
        raise RuntimeError(f"Missing saved Panel B values: {missing}")

    low = float(d["ci_low"].min())
    high = float(d["ci_high"].max())
    span = max(high - low, 0.1)
    xlim = (min(0.0, low) - 0.07 * span, max(0.0, high) + 0.07 * span)

    fig, axes = plt.subplots(1, 4, figsize=(22, 11.5), sharey=True, facecolor="white")
    y = np.arange(len(STATIC_FEATURES))
    for index, (ax, subtype) in enumerate(zip(axes, SUBTYPES)):
        q = (
            d[d["canonical_stratum"] == subtype]
            .set_index("feature")
            .reindex(STATIC_FEATURES)
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
                markersize=6.5,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.3,
                ecolor=color,
                elinewidth=1.25,
                capsize=3,
                zorder=3,
            )
        ax.axvline(0, color="#111111", linewidth=1.0, zorder=2)
        ax.set_xlim(*xlim)
        ax.set_ylim(len(STATIC_FEATURES) - 0.4, -0.6)
        ax.set_title(SUBTYPE_LABELS[subtype], fontsize=11.5, fontweight="bold", pad=12)
        ax.set_xlabel("Retained minus lost\npairwise similarity", fontsize=10)
        ax.set_yticks(y)
        if index == 0:
            ax.set_yticklabels([FEATURE_LABELS[f] for f in STATIC_FEATURES], fontsize=9.5)
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
        "B  Static similarities distinguishing retained from lost neighbors at 48 hours",
        fontsize=16, fontweight="bold", y=0.98,
    )
    fig.text(
        0.61, 0.035,
        "Positive: retained neighbors are more similar than lost neighbors.   "
        "Negative: retained neighbors are less similar than lost neighbors.\n"
        "Estimates and intervals are copied from the saved participant-level contrast table; "
        "the inferential marker decision is not recomputed.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.23, right=0.99, top=0.86, bottom=0.13, wspace=0.08)
    save(fig, "figure_F1_panel_B_static_contrasts_standalone")


def main() -> None:
    missing = [str(path) for path in [PANEL_A_TABLE, PANEL_B_TABLE] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing saved panel table(s): " + ", ".join(missing))
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    retention = pd.read_csv(PANEL_A_TABLE)
    contrasts = pd.read_csv(PANEL_B_TABLE)
    panel_a(retention)
    panel_b(contrasts)
    print(FIGURE_ROOT / "figure_F1_panel_A_retention_standalone.png")
    print(FIGURE_ROOT / "figure_F1_panel_B_static_contrasts_standalone.png")


if __name__ == "__main__":
    main()
