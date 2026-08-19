#!/usr/bin/env python3
"""Render the final frozen-test per-variable static-conditioning result."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
RUN_ROOT = (
    REPO_ROOT
    / "outputs/hidden_state_phenotype/per_variable_static_conditioning_audit/"
    "20260727T214535Z"
)
TEST_ROOT = RUN_ROOT / "test_interventions"
VALIDATION_ROOT = RUN_ROOT / "validation_interventions"
FROZEN_ROOT = RUN_ROOT / "frozen_plan"
OUTPUT_ROOT = RUN_ROOT / "final_outputs"

TEST_PRESENTATION_PATH = (
    TEST_ROOT / "test_per_variable_static_effects_presentation.csv"
)
VALIDATION_PRESENTATION_PATH = (
    VALIDATION_ROOT
    / "validation_per_variable_static_effects_presentation.csv"
)
TEST_MEDICATION_PREVALENCE_PATH = TEST_ROOT / "medication_prevalence_test.csv"
TEST_MANIFEST_PATH = TEST_ROOT / "step_c_test_manifest.json"
FROZEN_PLAN_PATH = FROZEN_ROOT / "frozen_test_application_plan.json"

GLOBAL_LABEL = "all static"
VARIABLE_COLOR = "#BA2828"
ALL_STATIC_COLOR = "#888888"
NAVY_COLOR = "#003366"
STRATUM_COLORS = [
    VARIABLE_COLOR,
    NAVY_COLOR,
    "#5BBABA",
    "#FF0000",
    ALL_STATIC_COLOR,
]
FIGURE_WIDTH = 13.5
FIGURE_HEIGHT = 7.3
FIGURE_DPI = 300
BAR_HEIGHT = 0.68
ERROR_LINE_WIDTH = 1.4
ERROR_CAP_SIZE = 3
VALUE_LABEL_PADDING_FRACTION = 0.018
STATE_AXIS_HEADROOM = 1.18
REORGANIZATION_AXIS_MAX = 0.72
TITLE_FONT_SIZE = 15
PANEL_TITLE_FONT_SIZE = 12
AXIS_FONT_SIZE = 11
TICK_FONT_SIZE = 10
FOOTNOTE_FONT_SIZE = 9
HORIZONTAL_SPACE = 0.10
BOTTOM_MARGIN = 0.20
TOP_MARGIN = 0.86
LEFT_MARGIN = 0.18
RIGHT_MARGIN = 0.98
EXPECTED_TEST_N = 221
EXPECTED_TEST_SEX_NONREFERENCE_N = 87

DISPLAY_LABELS = {
    "all static": "All static",
    "study_group": "Study group",
    "sex": "Sex, M vs F reference (n=87)",
    "hba1c": "HbA1c",
    "age": "Age",
    "bmi": "BMI",
    "metformin": "Metformin",
    "glp1": "GLP-1",
    "insulin": "Insulin",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ranked_table() -> pd.DataFrame:
    table = pd.read_csv(TEST_PRESENTATION_PATH)
    if set(table["variable"]) != set(DISPLAY_LABELS):
        raise RuntimeError("Test presentation variables differ from figure contract")
    sex = table[table["variable"].eq("sex")].iloc[0]
    if int(sex["n_participants"]) != EXPECTED_TEST_SEX_NONREFERENCE_N:
        raise RuntimeError("Sex row does not use the frozen conditional test sample")
    other = table[~table["variable"].eq(GLOBAL_LABEL)].sort_values(
        "median_state_l2",
        ascending=False,
    )
    global_row = table[table["variable"].eq(GLOBAL_LABEL)]
    ranked = pd.concat([global_row, other], ignore_index=True)
    ranked["display_label"] = ranked["variable"].map(DISPLAY_LABELS)
    ranked["median_nn10_reorganization"] = (
        1 - ranked["median_nn10_overlap"]
    )
    ranked["nn10_reorganization_ci_low"] = 1 - ranked["nn10_ci_high"]
    ranked["nn10_reorganization_ci_high"] = 1 - ranked["nn10_ci_low"]
    ranked["rank_state_l2"] = np.arange(1, len(ranked) + 1)
    return ranked


def draw_panel(
    axis,
    ranked: pd.DataFrame,
    value_column: str,
    low_column: str,
    high_column: str,
    title: str,
    xlabel: str,
    axis_max: float,
    value_decimals: int,
) -> None:
    positions = np.arange(len(ranked))
    values = ranked[value_column].to_numpy(dtype=float)
    lows = ranked[low_column].to_numpy(dtype=float)
    highs = ranked[high_column].to_numpy(dtype=float)
    colors = [
        ALL_STATIC_COLOR if variable == GLOBAL_LABEL else VARIABLE_COLOR
        for variable in ranked["variable"]
    ]
    axis.barh(
        positions,
        values,
        height=BAR_HEIGHT,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    axis.errorbar(
        values,
        positions,
        xerr=np.vstack([values - lows, highs - values]),
        fmt="none",
        ecolor=NAVY_COLOR,
        elinewidth=ERROR_LINE_WIDTH,
        capsize=ERROR_CAP_SIZE,
        zorder=3,
    )
    label_padding = axis_max * VALUE_LABEL_PADDING_FRACTION
    for position, value in zip(positions, values):
        axis.text(
            value + label_padding,
            position,
            f"{value:.{value_decimals}f}",
            va="center",
            ha="left",
            fontsize=TICK_FONT_SIZE - 1,
            color="#222222",
        )
    axis.set_xlim(0, axis_max)
    axis.set_title(title, fontsize=PANEL_TITLE_FONT_SIZE, loc="left")
    axis.set_xlabel(xlabel, fontsize=AXIS_FONT_SIZE)
    axis.set_ylabel("")
    axis.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.8, zorder=0)
    axis.grid(axis="y", visible=False)
    axis.invert_yaxis()


def results_paragraph(ranked: pd.DataFrame) -> str:
    values = ranked.set_index("variable")
    medication_prevalence = pd.read_csv(TEST_MEDICATION_PREVALENCE_PATH).set_index(
        "variable"
    )
    validation = pd.read_csv(VALIDATION_PRESENTATION_PATH).set_index("variable")
    maximum_transport_difference = float(
        np.max(
            np.abs(
                values["median_state_l2"]
                - validation.loc[values.index, "median_state_l2"]
            )
        )
    )
    return (
        "In the frozen test application, neutralizing the full 44-input static "
        f"profile produced a median participant-representation displacement of "
        f"{values.loc['all static', 'median_state_l2']:.2f} and reduced median "
        f"NN10 overlap to {values.loc['all static', 'median_nn10_overlap']:.2f}. "
        "Among individual variables, study group had the largest all-cohort "
        f"effect (L2 {values.loc['study_group', 'median_state_l2']:.2f}), while "
        "sex among participants differing from the female reference had a "
        f"conditional median effect of {values.loc['sex', 'median_state_l2']:.2f}. "
        f"HbA1c ({values.loc['hba1c', 'median_state_l2']:.2f}) and age "
        f"({values.loc['age', 'median_state_l2']:.2f}) followed. Medication "
        f"effects were smaller: metformin {values.loc['metformin', 'median_state_l2']:.2f}, "
        f"GLP-1 {values.loc['glp1', 'median_state_l2']:.2f}, and insulin "
        f"{values.loc['insulin', 'median_state_l2']:.2f}. Insulin remained "
        f"low-confidence because only "
        f"{int(medication_prevalence.loc['insulin', 'exposed_n'])} test "
        "participants were exposed. The validation ranking transported closely, "
        f"with the largest absolute validation-test median L2 difference equal "
        f"to {maximum_transport_difference:.2f}. These are one-variable "
        "intervention effects and do not sum exactly to the global effect because "
        "the frozen network combines static inputs nonlinearly."
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    ranked = ranked_table()
    ranked_path = OUTPUT_ROOT / "final_ranked_static_effects.csv"
    ranked.to_csv(ranked_path, index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        sharey=True,
    )
    state_axis_max = float(ranked["l2_ci_high"].max()) * STATE_AXIS_HEADROOM
    draw_panel(
        axes[0],
        ranked,
        "median_state_l2",
        "l2_ci_low",
        "l2_ci_high",
        "A. State displacement",
        "Median participant representation L2",
        state_axis_max,
        2,
    )
    axes[0].set_yticks(
        np.arange(len(ranked)),
        ranked["display_label"],
        fontsize=TICK_FONT_SIZE,
    )
    draw_panel(
        axes[1],
        ranked,
        "median_nn10_reorganization",
        "nn10_reorganization_ci_low",
        "nn10_reorganization_ci_high",
        "B. Neighborhood reorganization",
        "Median 1 - NN10 overlap",
        REORGANIZATION_AXIS_MAX,
        2,
    )
    figure.suptitle(
        "Frozen test per-variable static-conditioning effects",
        fontsize=TITLE_FONT_SIZE,
        y=0.96,
    )
    figure.text(
        LEFT_MARGIN,
        0.065,
        (
            "Sex is conditional on test participants differing from the female "
            "reference (M, n=87); all other rows use n=221. "
            "Insulin exposed n=17 and is low-confidence. Error bars are "
            "participant-bootstrap 95% CIs."
        ),
        ha="left",
        va="bottom",
        fontsize=FOOTNOTE_FONT_SIZE,
        color="#333333",
    )
    figure.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        bottom=BOTTOM_MARGIN,
        top=TOP_MARGIN,
        wspace=HORIZONTAL_SPACE,
    )
    figure_path = OUTPUT_ROOT / "fig_ranked_static_conditioning_effects_test.png"
    pdf_path = OUTPUT_ROOT / "fig_ranked_static_conditioning_effects_test.pdf"
    figure.savefig(
        figure_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    paragraph = results_paragraph(ranked)
    paragraph_path = OUTPUT_ROOT / "final_results_paragraph.md"
    paragraph_path.write_text(paragraph + "\n")
    report_path = OUTPUT_ROOT / "final_per_variable_audit_report.md"
    report_path.write_text(
        "# Final per-variable static-conditioning audit\n\n"
        "## Result\n\n"
        f"{paragraph}\n\n"
        "## Presentation note\n\n"
        "The displayed sex result is the frozen conditional estimate among "
        "participants whose factual sex differs from the female reference. The "
        "all-participant median remains zero by construction and is retained in "
        "the audit outputs.\n"
    )

    input_paths = [
        TEST_PRESENTATION_PATH,
        VALIDATION_PRESENTATION_PATH,
        TEST_MEDICATION_PREVALENCE_PATH,
        TEST_MANIFEST_PATH,
        FROZEN_PLAN_PATH,
    ]
    output_paths = [
        ranked_path,
        figure_path,
        pdf_path,
        paragraph_path,
        report_path,
    ]
    manifest = {
        "stage": "step_d_final_ranked_figure",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_participant_count": EXPECTED_TEST_N,
        "sex_nonreference_participant_count": EXPECTED_TEST_SEX_NONREFERENCE_N,
        "figure_panels": [
            "state displacement",
            "neighborhood reorganization",
        ],
        "state_sort_order": ranked["variable"].tolist(),
        "colors": {
            "variable": VARIABLE_COLOR,
            "all_static": ALL_STATIC_COLOR,
            "error_bar": NAVY_COLOR,
            "stratum_colors": STRATUM_COLORS,
        },
        "input_hashes_sha256": {
            str(path): sha256_file(path) for path in input_paths
        },
        "output_hashes_sha256": {
            str(path): sha256_file(path) for path in output_paths
        },
    }
    manifest_path = OUTPUT_ROOT / "step_d_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(ranked.to_string(index=False))
    print()
    print(paragraph)
    print(f"Saved final figure to {figure_path}")


if __name__ == "__main__":
    main()
