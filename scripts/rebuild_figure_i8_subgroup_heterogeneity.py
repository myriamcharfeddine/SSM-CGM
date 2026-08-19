#!/usr/bin/env python3
"""Re-render figure_i8_subgroup_heterogeneity.png as small multiples, one panel
per subgroup dimension, so it is readable at report size.

Standalone: only reads the already-exported
outputs/exercise_detector_model/interpretability/subgroup_effects.csv
(no model checkpoint / tensor cache needed). The original combined-heatmap
version (23 rows crammed into one alphabetically-sorted axis, with the
"p=<n>" participant count repeated in every one of the 3x23 cells even
though it is constant across the row, plus glucose strata sorted
alphabetically as "105-180, <105, >180") is replaced by:
  - one small heatmap per subgroup dimension (7 panels, shared color scale)
  - clinically ordered levels within each dimension (e.g. <105/105-180/>180
    instead of alphabetical) and a clinically ordered panel sequence
  - the participant/anchor count moved once into the row tick label
    (it never varies across strain, so repeating it per cell was pure
    clutter) instead of a second annotation line inside every cell
  - long category strings (e.g. the study-group medication label) wrapped
    instead of stretching the whole figure
  - a sequential white->navy colormap (all effects are negative, so the
    diverging red/blue scale in the original was only ever showing one
    half of the palette)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/exercise_detector_model/interpretability"
DATA = OUT / "subgroup_effects.csv"

STRAINS = ("light", "moderate", "vigorous")
NAVY = "#003366"
COLOR_GRID = "#D9D9D9"
COLOR_FRAME = "#000000"
# vmin (most negative / strongest glucose drop) -> NAVY, vmax (0, weakest
# effect) -> WHITE, so the darkest cells are the largest-magnitude effects.
CMAP = LinearSegmentedColormap.from_list("navy_white", [NAVY, "#FFFFFF"])

MIN_PARTICIPANTS = 10
MIN_ANCHORS = 20

DIMENSION_ORDER = [
    "study_group",
    "starting_glucose",
    "insulin_use",
    "hba1c_quartile",
    "bmi_quartile",
    "age_quartile",
    "clinical_site",
]
DIMENSION_TITLES = {
    "study_group": "Study group",
    "starting_glucose": "Starting glucose (mg/dL)",
    "insulin_use": "Insulin use",
    "hba1c_quartile": "HbA1c quartile",
    "bmi_quartile": "BMI quartile",
    "age_quartile": "Age quartile",
    "clinical_site": "Clinical site",
}
LEVEL_ORDER = {
    "starting_glucose": ["<105", "105-180", ">180"],
    "study_group": [
        "healthy",
        "pre_diabetes_lifestyle_controlled",
        "oral_medication_and_or_non_insulin_injectable_medication_controlled",
        "insulin_dependent",
    ],
    "insulin_use": ["0", "1"],
    "age_quartile": ["Q1", "Q2", "Q3", "Q4"],
    "bmi_quartile": ["Q1", "Q2", "Q3", "Q4"],
    "hba1c_quartile": ["Q1", "Q2", "Q3", "Q4"],
    "clinical_site": ["UAB", "UCSD", "UW"],
}
LEVEL_LABELS = {
    "0": "No insulin",
    "1": "Insulin",
    "healthy": "Healthy",
    "pre_diabetes_lifestyle_controlled": "Pre-diabetes,\nlifestyle-controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": (
        "Oral / non-insulin\ninjectable medication"
    ),
    "insulin_dependent": "Insulin-dependent",
}


def level_label(dimension: str, level: str) -> str:
    if level in LEVEL_LABELS:
        return LEVEL_LABELS[level]
    if len(level) > 14:
        return "\n".join(textwrap.wrap(level, 14))
    return level


def main() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.edgecolor": COLOR_FRAME,
            "axes.linewidth": 0.8,
            "font.size": 10.5,
        }
    )
    data = pd.read_csv(DATA)
    supported = data[data["supported"]].copy()

    global_min = supported["participant_mean_terminal_effect"].min()
    global_max = 0.0

    groups = []
    for dimension in DIMENSION_ORDER:
        block = supported[supported["subgroup_dimension"].eq(dimension)]
        if block.empty:
            continue
        order = [
            level
            for level in LEVEL_ORDER.get(dimension, sorted(block["subgroup_level"].unique()))
            if level in set(block["subgroup_level"])
        ]
        groups.append((dimension, block, order))

    norm = Normalize(vmin=global_min, vmax=global_max)

    row_heights = [len(order) for _, _, order in groups]
    fig = plt.figure(figsize=(11.0, 0.62 * sum(row_heights) + 2.9))
    gs = GridSpec(
        len(groups),
        2,
        width_ratios=[1.0, 0.035],
        height_ratios=row_heights,
        hspace=0.55,
        wspace=0.06,
        figure=fig,
        left=0.34,
        right=0.87,
        top=0.90,
        bottom=0.075,
    )
    cax = fig.add_subplot(gs[:, 1])

    mesh = None
    for row_index, (dimension, block, order) in enumerate(groups):
        ax = fig.add_subplot(gs[row_index, 0])
        matrix = (
            block.pivot(
                index="subgroup_level",
                columns="scenario_strain",
                values="participant_mean_terminal_effect",
            )
            .reindex(index=order, columns=STRAINS)
        )
        counts = (
            block.pivot(
                index="subgroup_level",
                columns="scenario_strain",
                values="participant_count",
            )
            .reindex(index=order, columns=STRAINS)
        )
        anchors = (
            block.pivot(
                index="subgroup_level",
                columns="scenario_strain",
                values="anchor_count",
            )
            .reindex(index=order, columns=STRAINS)
        )
        mesh = ax.pcolormesh(
            matrix.to_numpy(),
            cmap=CMAP,
            vmin=global_min,
            vmax=global_max,
            edgecolors="white",
            linewidth=1.5,
        )
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                value = matrix.iat[y, x]
                if pd.isna(value):
                    continue
                r, g, b, _ = CMAP(norm(value))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                color = "white" if luminance < 0.55 else "#111111"
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11.5,
                    fontweight="bold",
                    color=color,
                )
        ax.set_xlim(0, 3)
        ax.set_ylim(0, len(order))
        ax.invert_yaxis()
        ax.set_xticks(np.arange(3) + 0.5)
        row_labels = [
            f"{level_label(dimension, level)}\n"
            f"n={int(counts.loc[level].iloc[0])} · "
            f"{int(anchors.loc[level].iloc[0])} anc."
            for level in order
        ]
        ax.set_yticks(np.arange(len(order)) + 0.5)
        ax.set_yticklabels(row_labels, fontsize=8.6, linespacing=1.25)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.set_ylabel(
            DIMENSION_TITLES[dimension],
            fontsize=10.5,
            fontweight="bold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=14,
        )
        if row_index == len(groups) - 1:
            ax.set_xticklabels([s.capitalize() for s in STRAINS], fontsize=10)
            ax.set_xlabel("Planned exercise strain", fontsize=10.5, fontweight="bold")
        else:
            ax.set_xticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(COLOR_FRAME)
            spine.set_linewidth(0.8)

    fig.colorbar(mesh, cax=cax, label="Participant-mean 60-min\nglucose effect (mg/dL)")
    fig.suptitle(
        "Supported subgroup heterogeneity in the planned-exercise effect",
        fontsize=14.5,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.945,
        f"Shown only where a subgroup has ≥{MIN_PARTICIPANTS} participants and "
        f"≥{MIN_ANCHORS} anchors (69 of 75 subgroup×strain cells qualify)",
        ha="center",
        fontsize=11,
        color="#333333",
    )

    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"figure_i8_subgroup_heterogeneity.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    print(OUT / "figure_i8_subgroup_heterogeneity.png")


if __name__ == "__main__":
    main()
