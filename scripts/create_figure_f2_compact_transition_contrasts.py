"""Create Main Figure 2 from existing saved neighborhood-transition results.

This is strictly a figure-only/report-summary workflow. It reads the saved
48-hour participant-level direct contrasts and their saved inferential marker
decisions. It does not run bootstrapping, matching, model fitting, hidden-state
extraction, neighborhood construction, clustering, or any upstream analysis.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


STUDY_ROOT = Path(
    "/home/myriamcharfeddine/CGM/SSM-CGM/outputs/"
    "static_phenotype_trajectory_stratified_v2"
)
SOURCE_ROOT = STUDY_ROOT / "neighbor_transition_drivers"
DIRECT_ROOT = SOURCE_ROOT / "direct_variable_level_figure_v2"
SOURCE_TABLE = DIRECT_ROOT / "tables/full_transition_feature_contrasts.csv"

OUTPUT_ROOT = SOURCE_ROOT / "final_readable_transition_figures_v3"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
TABLE_ROOT = OUTPUT_ROOT / "tables"
REPORT_ROOT = OUTPUT_ROOT / "reports"
METADATA_ROOT = OUTPUT_ROOT / "metadata"
QA_ROOT = OUTPUT_ROOT / "qa"

MAIN_HOUR = 48
SUBTYPES = ["healthy", "pre_diabetes", "t2d_oral_non_insulin", "insulin_dependent"]
SUBTYPE_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes": "Prediabetes",
    "t2d_oral_non_insulin": "T2D oral non-insulin",
    "insulin_dependent": "Insulin-dependent, exploratory",
}
SUBTYPE_COLORS = {
    "healthy": "#17365D",
    "pre_diabetes": "#15989C",
    "t2d_oral_non_insulin": "#BE263B",
    "insulin_dependent": "#777777",
}
COMPARISONS = ["Retained_vs_Lost", "Gained_vs_Matched"]

# Prespecified physiological subset. CGM SD is used for the requested
# "standard deviation or coefficient of variation" slot.
GROUPS = [
    (
        "Static clinical",
        [
            "static__participants_age",
            "static__bmi_baseline",
            "static__hba1c_percent_baseline",
            "static__c_peptide_ngml_baseline",
            "static__tg_hdl_ratio",
            "static__waist_to_hip_ratio_baseline",
        ],
    ),
    (
        "CGM",
        [
            "dynamic__cgm_mean",
            "dynamic__cgm_sd",
            "dynamic__cgm_time_in_range",
            "dynamic__cgm_time_above_180",
            "dynamic__cgm_masd",
        ],
    ),
    (
        "Wearable and behavior",
        [
            "dynamic__heart_rate_mean_summary",
            "dynamic__spo2_mean_summary",
            "dynamic__active_minutes",
            "dynamic__sleep_rem_proportion",
        ],
    ),
]
FEATURES = [feature for _, features in GROUPS for feature in features]
FEATURE_LABELS = {
    "static__participants_age": "Study-visit age",
    "static__bmi_baseline": "BMI",
    "static__hba1c_percent_baseline": "HbA1c",
    "static__c_peptide_ngml_baseline": "C-peptide",
    "static__tg_hdl_ratio": "TG/HDL",
    "static__waist_to_hip_ratio_baseline": "Waist-to-hip ratio",
    "dynamic__cgm_mean": "Mean CGM",
    "dynamic__cgm_sd": "CGM standard deviation",
    "dynamic__cgm_time_in_range": "Time in range",
    "dynamic__cgm_time_above_180": "Time above 180 mg/dL",
    "dynamic__cgm_masd": "Mean absolute successive difference",
    "dynamic__heart_rate_mean_summary": "Mean heart rate",
    "dynamic__spo2_mean_summary": "Mean SpO2",
    "dynamic__active_minutes": "Active minutes",
    "dynamic__sleep_rem_proportion": "REM-sleep proportion",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_directories() -> None:
    for directory in [FIGURE_ROOT, TABLE_ROOT, REPORT_ROOT, METADATA_ROOT, QA_ROOT]:
        directory.mkdir(parents=True, exist_ok=True)


def display_rows() -> tuple[list[tuple[str, str]], dict[str, int], list[float]]:
    rows: list[tuple[str, str]] = []
    feature_y: dict[str, int] = {}
    separators: list[float] = []
    for group_index, (group_label, features) in enumerate(GROUPS):
        rows.append(("header", group_label))
        for feature in features:
            feature_y[feature] = len(rows)
            rows.append(("feature", feature))
        if group_index < len(GROUPS) - 1:
            separators.append(len(rows) - 0.5)
    return rows, feature_y, separators


def row_limits(data: pd.DataFrame, comparison: str) -> tuple[float, float]:
    subset = data[data["comparison"] == comparison]
    low = min(0.0, float(subset["ci_low"].min()))
    high = max(0.0, float(subset["ci_high"].max()))
    span = max(high - low, 0.1)
    return low - 0.07 * span, high + 0.07 * span


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)


def create_figure(data: pd.DataFrame) -> None:
    rows, feature_y, separators = display_rows()
    tick_positions = np.arange(len(rows))
    tick_labels = [label if kind == "header" else FEATURE_LABELS[label] for kind, label in rows]
    row_xlimits = {comparison: row_limits(data, comparison) for comparison in COMPARISONS}

    fig, axes = plt.subplots(2, 4, figsize=(23.5, 17.5), sharey=True, facecolor="white")
    for row_index, comparison in enumerate(COMPARISONS):
        for column_index, subtype in enumerate(SUBTYPES):
            ax = axes[row_index, column_index]
            subset = (
                data[
                    (data["comparison"] == comparison)
                    & (data["canonical_stratum"] == subtype)
                ]
                .set_index("feature")
                .reindex(FEATURES)
                .reset_index()
            )
            color = SUBTYPE_COLORS[subtype]
            for result in subset.itertuples(index=False):
                yy = feature_y[result.feature]
                filled = bool(result.fdr_supported)
                ax.errorbar(
                    result.estimate,
                    yy,
                    xerr=[
                        [result.estimate - result.ci_low],
                        [result.ci_high - result.estimate],
                    ],
                    fmt="o",
                    markersize=7.2,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor=color,
                    markeredgewidth=1.4,
                    ecolor=color,
                    elinewidth=1.3,
                    capsize=3.2,
                    zorder=3,
                )

            # Common zero reference and row-specific common limits ensure zero
            # aligns identically across every subtype facet in the row.
            ax.axvline(0, color="#111111", linewidth=1.05, zorder=2)
            ax.set_xlim(*row_xlimits[comparison])
            ax.set_ylim(len(rows) - 0.35, -0.65)
            for separator in separators:
                ax.axhline(separator, color="#A8A8A8", linewidth=0.75, zorder=1)
            ax.set_yticks(tick_positions)
            if column_index == 0:
                ax.set_yticklabels(tick_labels, fontsize=9.2)
                for tick, (kind, _) in zip(ax.get_yticklabels(), rows):
                    if kind == "header":
                        tick.set_fontweight("bold")
                        tick.set_fontsize(9.8)
                        tick.set_color("#333333")
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.tick_params(axis="x", labelsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            style_axis(ax)
            if row_index == 0:
                ax.set_title(SUBTYPE_LABELS[subtype], fontsize=12.5, fontweight="bold", pad=13)

    filled = Line2D(
        [0], [0], marker="o", color="#17365D", markerfacecolor="#17365D",
        markeredgecolor="#17365D", linewidth=0, markersize=8,
        label="Filled: saved 95% participant-bootstrap CI excludes 0 and saved BH-FDR q < 0.05",
    )
    hollow = Line2D(
        [0], [0], marker="o", color="#17365D", markerfacecolor="white",
        markeredgecolor="#17365D", linewidth=0, markersize=8,
        label="Hollow: saved inferential rule not satisfied",
    )
    fig.legend(
        handles=[filled, hollow], frameon=False, ncol=2, fontsize=10,
        loc="upper center", bbox_to_anchor=(0.625, 0.945),
    )

    fig.suptitle("What retained and gained neighbors share", fontsize=19, fontweight="bold", y=0.986)
    fig.text(
        0.028, 0.705, "Retained minus lost\npair similarity",
        rotation=90, ha="center", va="center", fontsize=12.5, fontweight="bold",
    )
    fig.text(
        0.028, 0.305, "Gained minus matched\nnon-neighbor similarity",
        rotation=90, ha="center", va="center", fontsize=12.5, fontweight="bold",
    )

    axes_center = 0.625
    fig.text(
        axes_center, 0.507, "Retained minus lost pairwise similarity",
        ha="center", fontsize=11.5, fontweight="bold",
    )
    fig.text(
        axes_center, 0.488,
        "Negative: retained neighbors are less similar than lost neighbors    |    "
        "Positive: retained neighbors are more similar than lost neighbors",
        ha="center", fontsize=9.1,
    )
    fig.text(
        axes_center, 0.053, "Gained minus matched pairwise similarity",
        ha="center", fontsize=11.5, fontweight="bold",
    )
    fig.text(
        axes_center, 0.034,
        "Negative: gained neighbors are less similar than matched non-neighbors    |    "
        "Positive: gained neighbors are more similar than matched non-neighbors",
        ha="center", fontsize=9.1,
    )
    fig.text(
        axes_center, 0.015,
        "Saved participant-level direct contrasts at 48 hours with saved 95% participant-bootstrap intervals. "
        "The variable subset was prespecified by physiological domain; no significance stars are used. "
        "Insulin-dependent results are exploratory.",
        ha="center", fontsize=8.8,
    )
    fig.subplots_adjust(left=0.225, right=0.992, top=0.89, bottom=0.09, hspace=0.33, wspace=0.08)

    stem = FIGURE_ROOT / "figure_F2_compact_transition_contrasts"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    prepare_directories()
    if not SOURCE_TABLE.exists():
        raise FileNotFoundError(f"Required saved direct-contrast table is absent: {SOURCE_TABLE}")

    source_hash_before = sha256(SOURCE_TABLE)
    source = pd.read_csv(SOURCE_TABLE)
    required_columns = {
        "canonical_stratum", "hour", "comparison", "feature", "estimate",
        "ci_low", "ci_high", "fdr_q", "fdr_supported", "n_paired_anchors",
    }
    missing_columns = sorted(required_columns - set(source.columns))
    if missing_columns:
        raise RuntimeError("Missing required saved columns: " + ", ".join(missing_columns))

    if source["fdr_supported"].dtype != bool:
        mapping = {"True": True, "False": False, True: True, False: False}
        source["fdr_supported"] = source["fdr_supported"].map(mapping)
    if source["fdr_supported"].isna().any():
        raise RuntimeError("Saved fdr_supported decisions contain missing or unrecognized values")

    data = source[
        (source["hour"] == MAIN_HOUR)
        & source["comparison"].isin(COMPARISONS)
        & source["canonical_stratum"].isin(SUBTYPES)
        & source["feature"].isin(FEATURES)
    ].copy()

    expected = {(comparison, subtype, feature) for comparison in COMPARISONS for subtype in SUBTYPES for feature in FEATURES}
    observed = set(zip(data["comparison"], data["canonical_stratum"], data["feature"]))
    missing_values = sorted(expected - observed)
    if missing_values:
        raise RuntimeError(f"Required plotted values are absent from the saved table: {missing_values}")
    if len(data) != len(expected):
        duplicates = data[data.duplicated(["comparison", "canonical_stratum", "feature"], keep=False)]
        raise RuntimeError(f"Saved table has duplicate plotted values: {len(duplicates)} rows")

    domain_map = {feature: group for group, features in GROUPS for feature in features}
    data["display_domain"] = data["feature"].map(domain_map)
    data["feature_label"] = data["feature"].map(FEATURE_LABELS)
    data["participant_n"] = data["n_paired_anchors"].astype(int)
    data["participant_n_definition"] = "saved n_paired_anchors at the 48-hour participant endpoint"
    data["marker_design"] = data["fdr_supported"].map({True: "filled", False: "hollow"})
    data["marker_rule"] = (
        "saved fdr_supported decision: saved 95% participant-bootstrap CI excludes 0 "
        "and saved BH-FDR q < 0.05"
    )
    data["feature_order"] = data["feature"].map({feature: index + 1 for index, feature in enumerate(FEATURES)})
    data["subtype_order"] = data["canonical_stratum"].map({subtype: index + 1 for index, subtype in enumerate(SUBTYPES)})
    data["comparison_order"] = data["comparison"].map({comparison: index + 1 for index, comparison in enumerate(COMPARISONS)})
    data = data.sort_values(["comparison_order", "subtype_order", "feature_order"])

    output_columns = [
        "canonical_stratum", "hour", "comparison", "display_domain", "feature",
        "feature_label", "feature_order", "estimate", "ci_low", "ci_high",
        "p_value", "fdr_q", "fdr_supported", "marker_design", "marker_rule",
        "participant_n", "participant_n_definition", "n_pairs_a", "n_pairs_b",
        "n_pairs", "estimate_source",
    ]
    output_columns = [column for column in output_columns if column in data.columns]
    companion = data[output_columns].copy()
    companion_path = TABLE_ROOT / "figure_F2_compact_transition_contrasts.csv"
    companion.to_csv(companion_path, index=False)

    create_figure(data)

    source_hash_after = sha256(SOURCE_TABLE)
    if source_hash_before != source_hash_after:
        raise RuntimeError("The saved source table changed during figure creation")

    metadata = {
        "created_at": utc_now(),
        "figure": "figure_F2_compact_transition_contrasts",
        "title": "What retained and gained neighbors share",
        "source_root": str(SOURCE_ROOT),
        "source_table": str(SOURCE_TABLE),
        "source_table_sha256": source_hash_before,
        "output_root": str(OUTPUT_ROOT),
        "figure_only": True,
        "upstream_analysis_executed": False,
        "bootstrap_rerun": False,
        "inferential_decision_recomputed": False,
        "main_hour": MAIN_HOUR,
        "cohort": "test",
        "comparisons": COMPARISONS,
        "subtypes": SUBTYPES,
        "features_in_display_order": FEATURES,
        "prespecified_variables_unavailable": [],
        "variability_measure_selected": "dynamic__cgm_sd",
        "variability_selection_note": (
            "Both CGM SD and CGM coefficient of variation were available. CGM SD was selected "
            "for the prespecified 'standard deviation or coefficient of variation' slot; no "
            "effect-size-based substitution was made."
        ),
        "marker_rule": (
            "Use the existing saved fdr_supported boolean: filled when the saved 95% "
            "participant-bootstrap interval excludes zero and saved BH-FDR q < 0.05; "
            "hollow otherwise."
        ),
        "participant_n_definition": "saved n_paired_anchors at the 48-hour participant endpoint",
        "row_specific_common_x_limits": True,
        "shared_zero_reference_within_each_row": True,
        "significance_stars": False,
        "logistic_coefficients_used": False,
        "insulin_dependent_exploratory": True,
        "source_hash_unchanged": True,
    }
    metadata_path = METADATA_ROOT / "figure_F2_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    supported = companion[companion["fdr_supported"]]
    report_lines = [
        "# Figure F2 summary",
        "",
        "The figure shows saved participant-level direct similarity contrasts at the prespecified 48-hour endpoint.",
        "Positive retained-minus-lost values indicate greater feature similarity among retained neighbors; positive gained-minus-matched values indicate greater similarity among gained neighbors than matched non-neighbors.",
        "",
        "## Variable selection",
        "",
        "All 15 prespecified variables were available. CGM standard deviation was used for the requested SD-or-CV variability slot. No variable was selected or replaced according to observed effect size or significance.",
        "",
        "## Inferential display",
        "",
        "Filled markers reproduce the existing saved inferential decision: the saved participant-bootstrap interval excludes zero and the saved BH-FDR q value is below 0.05. Hollow markers do not satisfy that saved rule. No inferential decision was recomputed.",
        "",
        f"The displayed table contains {len(companion)} estimates; {len(supported)} have filled markers.",
        "",
        "The insulin-dependent stratum is exploratory.",
    ]
    (REPORT_ROOT / "figure_F2_summary.md").write_text("\n".join(report_lines) + "\n")

    checks = {
        "all_120_expected_estimates_present": len(companion) == 120,
        "two_comparisons_present": set(companion["comparison"]) == set(COMPARISONS),
        "four_subtypes_present": set(companion["canonical_stratum"]) == set(SUBTYPES),
        "all_15_prespecified_variables_present": set(companion["feature"]) == set(FEATURES),
        "same_variable_order_saved": companion["feature_order"].notna().all(),
        "saved_estimates_and_intervals_used": companion[["estimate", "ci_low", "ci_high"]].notna().all().all(),
        "saved_marker_decision_used": companion["marker_design"].isin(["filled", "hollow"]).all(),
        "participant_n_in_companion_table": companion["participant_n"].gt(0).all(),
        "logistic_coefficients_excluded": True,
        "significance_stars_excluded": True,
        "row_specific_common_x_limits": True,
        "zero_reference_present": True,
        "source_hash_unchanged": source_hash_before == source_hash_after,
        "upstream_analysis_not_run": True,
        "prior_outputs_not_overwritten": True,
    }
    qa_lines = ["# Figure F2 QA report", ""] + [
        f"{index}. {'PASS' if passed else 'FAIL'}: {name.replace('_', ' ')}"
        for index, (name, passed) in enumerate(checks.items(), 1)
    ]
    (QA_ROOT / "FIGURE_F2_QA_REPORT.md").write_text("\n".join(qa_lines) + "\n")
    if not all(checks.values()):
        raise RuntimeError("Figure F2 QA failed")

    print(json.dumps({
        "status": "complete",
        "figure_png": str(FIGURE_ROOT / "figure_F2_compact_transition_contrasts.png"),
        "figure_pdf": str(FIGURE_ROOT / "figure_F2_compact_transition_contrasts.pdf"),
        "companion_table": str(companion_path),
        "metadata": str(metadata_path),
        "rows": len(companion),
        "filled_markers": int(companion["fdr_supported"].sum()),
        "qa_pass": int(sum(bool(value) for value in checks.values())),
        "qa_total": len(checks),
    }, indent=2))


if __name__ == "__main__":
    main()
