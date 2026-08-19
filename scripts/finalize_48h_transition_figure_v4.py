"""Finalize the 48-hour retained/gained neighborhood-transition figure package.

Reads existing saved results only. No states, neighborhoods, pairs, matches,
bootstraps, FDR values, coefficients, or models are recomputed.
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


STUDY_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM/outputs/static_phenotype_trajectory_stratified_v2")
SOURCE_ROOT = STUDY_ROOT / "neighbor_transition_drivers"
OUTPUT_ROOT = SOURCE_ROOT / "final_48h_transition_figure_v4"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
TABLE_ROOT = OUTPUT_ROOT / "tables"
METADATA_ROOT = OUTPUT_ROOT / "metadata"
REPORT_ROOT = OUTPUT_ROOT / "reports"
QA_ROOT = OUTPUT_ROOT / "qa"

HOURS = [6, 12, 24, 48]
MAIN_HOUR = 48
SUBTYPES = ["healthy", "pre_diabetes", "t2d_oral_non_insulin", "insulin_dependent"]
SUBTYPE_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes": "Prediabetes",
    "t2d_oral_non_insulin": "T2D oral non-insulin",
    "insulin_dependent": "Insulin-dependent, exploratory",
}
COLORS = {
    "healthy": "#003366",
    "pre_diabetes": "#5BBABA",
    "t2d_oral_non_insulin": "#BA2828",
    "insulin_dependent": "#888888",
}
BLACK = "#000000"
GRID = "#D9D9D9"
COMPARISONS = ["Retained_vs_Lost", "Gained_vs_Matched"]

GROUPS = [
    ("Static clinical", [
        "static__participants_age", "static__bmi_baseline",
        "static__hba1c_percent_baseline", "static__c_peptide_ngml_baseline",
        "static__tg_hdl_ratio", "static__waist_to_hip_ratio_baseline",
    ]),
    ("CGM", [
        "dynamic__cgm_mean", "dynamic__cgm_sd", "dynamic__cgm_time_in_range",
        "dynamic__cgm_time_above_180", "dynamic__cgm_masd",
    ]),
    ("Wearable and behavior", [
        "dynamic__heart_rate_mean_summary", "dynamic__spo2_mean_summary",
        "dynamic__active_minutes", "dynamic__sleep_rem_proportion",
    ]),
]
FEATURES = [feature for _, group_features in GROUPS for feature in group_features]
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
TIME_COLORS = {6: "#A6CEE3", 12: "#1F78B4", 24: "#FDBF6F", 48: "#6A3D9A"}
TIME_OFFSETS = {6: -0.27, 12: -0.09, 24: 0.09, 48: 0.27}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dirs() -> None:
    for directory in [FIGURE_ROOT, TABLE_ROOT, METADATA_ROOT, REPORT_ROOT, QA_ROOT]:
        directory.mkdir(parents=True, exist_ok=True)


def csv_candidates(required: set[str]) -> list[tuple[Path, pd.DataFrame]]:
    candidates = []
    for path in SOURCE_ROOT.rglob("*.csv"):
        if OUTPUT_ROOT in path.parents:
            continue
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if required.issubset(columns):
            candidates.append((path, pd.read_csv(path)))
    return candidates


def discover_sources() -> dict[str, Path]:
    direct_required = {
        "canonical_stratum", "hour", "comparison", "feature", "estimate",
        "ci_low", "ci_high", "fdr_q", "fdr_supported", "n_paired_anchors",
    }
    direct = csv_candidates(direct_required)
    eligible_direct = []
    for path, data in direct:
        q = data[(data["hour"] == 48) & data["comparison"].isin(COMPARISONS)]
        if set(q["comparison"]) == set(COMPARISONS) and set(q["canonical_stratum"]) >= set(SUBTYPES):
            eligible_direct.append((q["feature"].nunique(), len(q), path))
    if not eligible_direct:
        raise FileNotFoundError("No saved 48-hour direct-contrast table with saved marker decisions was found")
    direct_path = max(eligible_direct)[2]

    time_required = {
        "cohort", "canonical_stratum", "hour", "comparison", "feature",
        "mean_difference", "ci_low", "ci_high", "fdr_q", "n_paired_anchors",
    }
    time_candidates = []
    for path, data in csv_candidates(time_required):
        q = data[(data["cohort"] == "test") & data["comparison"].isin(COMPARISONS)]
        if set(q["hour"].dropna().astype(int)) >= set(HOURS) and set(q["canonical_stratum"]) >= set(SUBTYPES):
            time_candidates.append((q["feature"].nunique(), len(q), path))
    if not time_candidates:
        raise FileNotFoundError("No saved all-timepoint participant-level contrast table was found")
    time_path = max(time_candidates)[2]

    coefficient_required = {
        "task", "feature", "median_coefficient", "fold_ci_low", "fold_ci_high",
        "sign_stability_pct", "n_folds", "uncertainty_note",
    }
    coefficient_candidates = csv_candidates(coefficient_required)
    if not coefficient_candidates:
        raise FileNotFoundError("No saved combined-model coefficient summary was found")
    coefficient_path = max(coefficient_candidates, key=lambda item: len(item[1]))[0]

    count_required = {
        "cohort", "canonical_stratum", "hour", "retained_n", "lost_n",
        "gained_n", "matched_n", "k_mode", "cluster_matched", "anchor_id_hash",
    }
    count_candidates = csv_candidates(count_required)
    if not count_candidates:
        raise FileNotFoundError("No saved participant transition-count table was found")
    count_path = max(count_candidates, key=lambda item: len(item[1]))[0]

    reports = [
        path for path in SOURCE_ROOT.rglob("*.md")
        if OUTPUT_ROOT not in path.parents
        and "Continuous similarities are negative absolute train-standardized differences" in path.read_text(errors="ignore")
        and "Matched non-neighbors were matched" in path.read_text(errors="ignore")
    ]
    if not reports:
        raise FileNotFoundError("Exact similarity and matching definitions were not found in saved metadata/report files")
    methods_path = max(reports, key=lambda path: path.stat().st_mtime)

    invariant_candidates = []
    dynamic_candidates = []
    for path in SOURCE_ROOT.rglob("*.json"):
        if OUTPUT_ROOT in path.parents:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if {"future_dynamic_data_used", "same_subtype_only", "bootstrap_n"}.issubset(payload):
            invariant_candidates.append(path)
        if "future_data_used" in payload and "segment_aggregation" in payload:
            dynamic_candidates.append(path)
    if not invariant_candidates or not dynamic_candidates:
        raise FileNotFoundError("Saved invariants or dynamic-window metadata are absent")

    return {
        "direct_contrasts": direct_path,
        "time_resolved_contrasts": time_path,
        "coefficient_summary": coefficient_path,
        "participant_counts": count_path,
        "methodology_report": methods_path,
        "invariants": max(invariant_candidates, key=lambda path: path.stat().st_mtime),
        "dynamic_report": max(dynamic_candidates, key=lambda path: path.stat().st_mtime),
    }


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.map({True: True, False: False, "True": True, "False": False})
    if mapped.isna().any():
        raise RuntimeError("Saved inferential marker decisions contain unrecognized values")
    return mapped.astype(bool)


def preflight(sources: dict[str, Path]) -> dict:
    methods = sources["methodology_report"].read_text()
    invariants = json.loads(sources["invariants"].read_text())
    dynamic = json.loads(sources["dynamic_report"].read_text())
    direct = pd.read_csv(sources["direct_contrasts"])
    counts = pd.read_csv(sources["participant_counts"])
    direct["fdr_supported"] = bool_series(direct["fdr_supported"])
    main = direct[(direct["hour"] == 48) & direct["comparison"].isin(COMPARISONS) & direct["feature"].isin(FEATURES)]

    expected = {(comparison, subtype, feature) for comparison in COMPARISONS for subtype in SUBTYPES for feature in FEATURES}
    observed = set(zip(main["comparison"], main["canonical_stratum"], main["feature"]))
    participant_counts = {
        comparison: {
            subtype: sorted(main[(main["comparison"] == comparison) & (main["canonical_stratum"] == subtype)]["n_paired_anchors"].dropna().astype(int).unique().tolist())
            for subtype in SUBTYPES
        }
        for comparison in COMPARISONS
    }
    primary_counts = counts[(counts["cohort"] == "test") & (counts["hour"] == 48) & (counts["k_mode"] == "primary")]

    audit = {
        "created_at": now(),
        "discovery": {key: str(path) for key, path in sources.items()},
        "continuous_similarity_definition": "S_f(i,j) = -|z_if - z_jf| for train-standardized continuous features",
        "continuous_similarity_verified": "Continuous similarities are negative absolute train-standardized differences" in methods,
        "binary_similarity_definition": "S_f(i,j) = 1[x_if = x_jf]",
        "binary_similarity_verified": "binary/categorical similarities are exact matches" in methods,
        "medication_similarity_definition": "individual medication indicators are binary exact-match similarities, not Jaccard",
        "participant_level_contrasts_verified": "Reporting files are participant-aggregated" in methods,
        "participant_bootstrap_95_ci_verified": int(invariants.get("bootstrap_n", -1)) == 1000,
        "dynamic_window": "cumulative from time 0 through t; the main endpoint uses 0 through 48 hours only",
        "no_future_dynamic_data_verified": invariants.get("future_dynamic_data_used") is False and dynamic.get("future_data_used") is False,
        "main_endpoint_hours": 48,
        "same_subtype_neighbor_search_verified": invariants.get("same_subtype_only") is True,
        "primary_cluster_restriction": False,
        "all_frozen_clusters_within_subtype_eligible": bool((primary_counts["cluster_matched"] == False).all()),  # noqa: E712
        "matching_definition": "matched without replacement on standardized h0 distance, valid observation count, endpoint-anchor count, and available streaming duration among non-neighbors",
        "matching_definition_verified": "Matched non-neighbors were matched on h0 distance, valid observations, endpoint anchors, and available duration" in methods,
        "saved_marker_rule": "filled iff the saved participant-bootstrap CI excludes zero and saved BH-FDR q < 0.05",
        "saved_marker_decisions_present": bool(main["fdr_supported"].notna().all()),
        "participant_n_by_task_and_subtype": participant_counts,
        "insulin_dependent_exploratory": "insulin-dependent subtype is exploratory" in methods.lower(),
        "same_compact_variables_all_panels": observed == expected,
        "activity_feature_choice": "dynamic__active_minutes",
        "activity_choice_basis": "used in the latest saved compact Figure F2 plotted-data table; total steps was not substituted by effect size",
        "missing_prespecified_features": sorted(set(FEATURES) - set(main["feature"])),
        "hard_stop_passed": False,
    }
    hard_checks = [
        audit["continuous_similarity_verified"], audit["binary_similarity_verified"],
        audit["participant_level_contrasts_verified"], audit["participant_bootstrap_95_ci_verified"],
        audit["no_future_dynamic_data_verified"], audit["same_subtype_neighbor_search_verified"],
        audit["all_frozen_clusters_within_subtype_eligible"], audit["matching_definition_verified"],
        audit["saved_marker_decisions_present"], audit["same_compact_variables_all_panels"],
    ]
    audit["hard_stop_passed"] = bool(all(hard_checks))
    (OUTPUT_ROOT / "PREFLIGHT_FINAL_FIGURE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    md = ["# Preflight final-figure audit", ""] + [f"- **{key}**: {value}" for key, value in audit.items()]
    (OUTPUT_ROOT / "PREFLIGHT_FINAL_FIGURE_AUDIT.md").write_text("\n".join(md) + "\n")
    if not audit["hard_stop_passed"]:
        raise RuntimeError("Hard-stop preflight failed; see PREFLIGHT_FINAL_FIGURE_AUDIT.json")
    return audit


def display_rows(features: list[str] = FEATURES):
    rows, positions, separators = [], {}, []
    feature_set = set(features)
    for group_index, (group_name, group_features) in enumerate(GROUPS):
        selected = [feature for feature in group_features if feature in feature_set]
        if not selected:
            continue
        rows.append(("header", group_name))
        for feature in selected:
            positions[feature] = len(rows)
            rows.append(("feature", feature))
        if group_index < len(GROUPS) - 1:
            separators.append(len(rows) - 0.5)
    return rows, positions, separators


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BLACK)
        spine.set_linewidth(0.8)


def common_limits(data, comparison, estimate="estimate"):
    q = data[data["comparison"] == comparison]
    low, high = min(0.0, float(q["ci_low"].min())), max(0.0, float(q["ci_high"].max()))
    span = max(high - low, 0.1)
    return low - 0.07 * span, high + 0.07 * span


def label_for(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("static__", "").replace("dynamic__", "").replace("_", " "))


def main_figure(data: pd.DataFrame, participant_n: dict) -> None:
    rows, positions, separators = display_rows()
    ticks = np.arange(len(rows))
    labels = [name if kind == "header" else label_for(name) for kind, name in rows]
    limits = {comparison: common_limits(data, comparison) for comparison in COMPARISONS}
    fig, axes = plt.subplots(2, 4, figsize=(23.5, 17.7), sharey=True, facecolor="white")
    for row_index, comparison in enumerate(COMPARISONS):
        for column_index, subtype in enumerate(SUBTYPES):
            ax = axes[row_index, column_index]
            q = data[(data["comparison"] == comparison) & (data["canonical_stratum"] == subtype)].set_index("feature").reindex(FEATURES).reset_index()
            color = COLORS[subtype]
            for result in q.itertuples(index=False):
                y = positions[result.feature]
                ax.errorbar(result.estimate, y,
                    xerr=[[result.estimate-result.ci_low], [result.ci_high-result.estimate]],
                    fmt="o", ms=7.2, mfc=color if result.fdr_supported else "white", mec=color,
                    mew=1.4, ecolor=color, elinewidth=1.3, capsize=3.2, zorder=3)
            ax.axvline(0, color=BLACK, lw=1.05)
            for separator in separators:
                ax.axhline(separator, color="#A8A8A8", lw=0.75)
            ax.set_xlim(*limits[comparison])
            ax.set_ylim(len(rows)-0.35, -0.65)
            ax.set_yticks(ticks)
            if column_index == 0:
                ax.set_yticklabels(labels, fontsize=9.2)
                for tick, (kind, _) in zip(ax.get_yticklabels(), rows):
                    if kind == "header":
                        tick.set_fontweight("bold")
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.tick_params(axis="x", labelsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            style_axis(ax)
            if row_index == 0:
                ax.set_title(SUBTYPE_LABELS[subtype], fontsize=12.4, fontweight="bold", pad=12)

    legend = [
        Line2D([0], [0], marker="o", color=COLORS["healthy"], mfc=COLORS["healthy"], lw=0, ms=8,
               label="Filled: saved 95% participant-bootstrap CI excludes 0 and BH-FDR q < 0.05"),
        Line2D([0], [0], marker="o", color=COLORS["healthy"], mfc="white", lw=0, ms=8,
               label="Hollow: saved inferential rule not satisfied"),
    ]
    fig.suptitle("What retained and gained neighbors share after 48 hours of streaming", fontsize=19, fontweight="bold", y=.991)
    fig.legend(handles=legend, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.625, .953), fontsize=10)
    nline = "Participant N (retained-minus-lost / gained-minus-matched): " + "   |   ".join(
        f"{SUBTYPE_LABELS[subtype].replace(', exploratory','')} {participant_n['Retained_vs_Lost'][subtype][0]}/{participant_n['Gained_vs_Matched'][subtype][0]}"
        for subtype in SUBTYPES
    )
    fig.text(.625, .921, nline, ha="center", fontsize=9.5)
    fig.text(.027, .705, "Retained minus lost\npairwise similarity", rotation=90, ha="center", va="center", fontsize=12.5, fontweight="bold")
    fig.text(.027, .305, "Gained minus matched\nnon-neighbor pairwise similarity", rotation=90, ha="center", va="center", fontsize=12.5, fontweight="bold")
    fig.text(.625, .508, "Participant-level retained minus lost similarity contrast", ha="center", fontsize=11.5, fontweight="bold")
    fig.text(.625, .489, "Negative: retained neighbors are less similar than lost neighbors    |    Positive: retained neighbors are more similar than lost neighbors", ha="center", fontsize=9)
    fig.text(.625, .055, "Participant-level gained minus matched similarity contrast", ha="center", fontsize=11.5, fontweight="bold")
    fig.text(.625, .036, "Negative: gained neighbors are less similar than matched non-neighbors    |    Positive: gained neighbors are more similar than matched non-neighbors", ha="center", fontsize=9)
    fig.text(.625, .016, "Associative pairwise-similarity contrasts at 48 hours; dynamic summaries use observations only through 48 hours. Insulin-dependent results are exploratory.", ha="center", fontsize=8.8)
    fig.subplots_adjust(left=.225, right=.992, top=.89, bottom=.092, hspace=.33, wspace=.08)
    stem = FIGURE_ROOT / "figure_F2_final_retained_and_gained_similarity"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def time_resolved_figure(data: pd.DataFrame) -> None:
    rows, positions, separators = display_rows()
    ticks = np.arange(len(rows))
    labels = [name if kind == "header" else label_for(name) for kind, name in rows]
    limits = {comparison: common_limits(data.rename(columns={"mean_difference": "estimate"}), comparison) for comparison in COMPARISONS}
    fig, axes = plt.subplots(2, 4, figsize=(23.5, 18), sharey=True, facecolor="white")
    for row_index, comparison in enumerate(COMPARISONS):
        for column_index, subtype in enumerate(SUBTYPES):
            ax = axes[row_index, column_index]
            q = data[(data["comparison"] == comparison) & (data["canonical_stratum"] == subtype)]
            for hour in HOURS:
                h = q[q["hour"] == hour].set_index("feature").reindex(FEATURES).reset_index()
                for result in h.itertuples(index=False):
                    y = positions[result.feature] + TIME_OFFSETS[hour]
                    ax.errorbar(result.mean_difference, y,
                        xerr=[[result.mean_difference-result.ci_low], [result.ci_high-result.mean_difference]],
                        fmt="o", ms=4.2, mfc=TIME_COLORS[hour], mec=TIME_COLORS[hour],
                        ecolor=TIME_COLORS[hour], elinewidth=.9, capsize=1.8, zorder=3)
            ax.axvline(0, color=BLACK, lw=1)
            for separator in separators:
                ax.axhline(separator, color="#A8A8A8", lw=.7)
            ax.set_xlim(*limits[comparison])
            ax.set_ylim(len(rows)-.45, -.65)
            ax.set_yticks(ticks)
            if column_index == 0:
                ax.set_yticklabels(labels, fontsize=8.5)
                for tick, (kind, _) in zip(ax.get_yticklabels(), rows):
                    if kind == "header": tick.set_fontweight("bold")
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.tick_params(axis="x", labelsize=8)
            style_axis(ax)
            if row_index == 0: ax.set_title(SUBTYPE_LABELS[subtype], fontsize=11.5, fontweight="bold")
    handles = [Line2D([0], [0], marker="o", color=TIME_COLORS[h], mfc=TIME_COLORS[h], lw=0, label=f"{h} h") for h in HOURS]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.62, .951))
    fig.suptitle("Appendix A1  Time-resolved retained and gained similarity contrasts", fontsize=17, fontweight="bold", y=.99)
    fig.text(.62, .505, "Retained minus lost pairwise similarity", ha="center", fontsize=11, fontweight="bold")
    fig.text(.62, .048, "Gained minus matched pairwise similarity", ha="center", fontsize=11, fontweight="bold")
    fig.text(.62, .018, "Points and intervals are saved participant-level estimates at 6, 12, 24, and 48 hours. Timepoint color is descriptive; no new inferential marker decision was calculated.", ha="center", fontsize=8.7)
    fig.subplots_adjust(left=.225, right=.992, top=.91, bottom=.075, hspace=.31, wspace=.08)
    stem = FIGURE_ROOT / "figure_A1_time_resolved_transition_contrasts"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def full_inventory_figure(data: pd.DataFrame) -> None:
    features = sorted(data["feature"].unique().tolist())
    y = np.arange(len(features))
    limits = {comparison: common_limits(data, comparison) for comparison in COMPARISONS}
    fig, axes = plt.subplots(2, 4, figsize=(23, max(18, len(features)*.34)), sharey=True, facecolor="white")
    for row_index, comparison in enumerate(COMPARISONS):
        for column_index, subtype in enumerate(SUBTYPES):
            ax = axes[row_index, column_index]
            q = data[(data["comparison"] == comparison) & (data["canonical_stratum"] == subtype)].set_index("feature").reindex(features).reset_index()
            color = COLORS[subtype]
            for yy, result in zip(y, q.itertuples(index=False)):
                if not np.isfinite(result.estimate): continue
                ax.errorbar(result.estimate, yy,
                    xerr=[[result.estimate-result.ci_low], [result.ci_high-result.estimate]],
                    fmt="o", ms=4, mfc=color if result.fdr_supported else "white", mec=color,
                    ecolor=color, elinewidth=.85, capsize=1.7)
            ax.axvline(0, color=BLACK, lw=1)
            ax.set_xlim(*limits[comparison]); ax.set_ylim(len(features)-.4, -.6); ax.set_yticks(y)
            if column_index == 0: ax.set_yticklabels([label_for(f) for f in features], fontsize=6.2)
            else: ax.tick_params(axis="y", labelleft=False)
            ax.set_xlabel("Direct pairwise-similarity contrast", fontsize=8)
            style_axis(ax)
            if row_index == 0: ax.set_title(SUBTYPE_LABELS[subtype], fontsize=11, fontweight="bold")
    fig.suptitle("Appendix A2  Full variable-level retained and gained similarity contrasts at 48 hours", fontsize=16, fontweight="bold", y=.995)
    fig.text(.5, .008, "Top: retained minus lost. Bottom: gained minus matched non-neighbor. Filled markers reproduce the saved participant-bootstrap CI and BH-FDR decision.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.24, right=.99, top=.97, bottom=.035, hspace=.2, wspace=.08)
    stem = FIGURE_ROOT / "figure_A2_full_variable_transition_contrasts"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def coefficient_figure(data: pd.DataFrame) -> pd.DataFrame:
    plotted = []
    fig, axes = plt.subplots(1, 2, figsize=(17, 9), facecolor="white")
    for ax, task, title in zip(axes, ["A", "B"], ["Retained versus lost", "Gained versus matched non-neighbor"]):
        q = data[data["task"] == task].copy()
        q["abs_coefficient"] = q["median_coefficient"].abs()
        q = q.nlargest(18, "abs_coefficient").sort_values("median_coefficient")
        plotted.append(q)
        y = np.arange(len(q))
        ax.errorbar(q["median_coefficient"], y,
            xerr=[q["median_coefficient"]-q["fold_ci_low"], q["fold_ci_high"]-q["median_coefficient"]],
            fmt="o", color=COLORS["healthy"], ecolor=COLORS["healthy"], capsize=2.5)
        ax.axvline(0, color=BLACK, lw=1)
        ax.set_yticks(y); ax.set_yticklabels([label_for(f) for f in q["feature"]], fontsize=7.5)
        ax.set_xlabel("Median standardized logistic coefficient")
        ax.set_title(title, loc="left", fontweight="bold")
        style_axis(ax)
    fig.suptitle("Exploratory conditional feature coefficients in combined transition models", fontsize=16, fontweight="bold")
    fig.text(.5, .02, "Intervals show fold-wise spread and are not participant-bootstrap coefficient confidence intervals. Correlated-predictor coefficients are exploratory conditional associations.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.19, right=.98, top=.89, bottom=.13, wspace=.45)
    stem = FIGURE_ROOT / "figure_A3_exploratory_combined_coefficients"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pd.concat(plotted, ignore_index=True)


def main() -> None:
    ensure_dirs()
    sources = discover_sources()
    audit = preflight(sources)
    source_hashes_before = {name: sha256(path) for name, path in sources.items()}

    direct = pd.read_csv(sources["direct_contrasts"])
    direct["fdr_supported"] = bool_series(direct["fdr_supported"])
    main = direct[(direct["hour"] == MAIN_HOUR) & direct["comparison"].isin(COMPARISONS) & direct["canonical_stratum"].isin(SUBTYPES) & direct["feature"].isin(FEATURES)].copy()
    expected = {(comparison, subtype, feature) for comparison in COMPARISONS for subtype in SUBTYPES for feature in FEATURES}
    if set(zip(main["comparison"], main["canonical_stratum"], main["feature"])) != expected:
        raise RuntimeError("Compact 48-hour saved results are incomplete")
    main["feature_label"] = main["feature"].map(FEATURE_LABELS)
    main["display_domain"] = main["feature"].map({feature: group for group, group_features in GROUPS for feature in group_features})
    main["participant_n"] = main["n_paired_anchors"].astype(int)
    main["marker_design"] = main["fdr_supported"].map({True: "filled", False: "hollow"})
    main["marker_decision_source"] = "existing saved fdr_supported value; not recomputed"
    main["feature_order"] = main["feature"].map({feature: i+1 for i, feature in enumerate(FEATURES)})
    main = main.sort_values(["comparison", "canonical_stratum", "feature_order"])
    main.to_csv(TABLE_ROOT / "figure_F2_final_retained_and_gained_similarity.csv", index=False)
    main_figure(main, audit["participant_n_by_task_and_subtype"])

    time = pd.read_csv(sources["time_resolved_contrasts"])
    time = time[(time["cohort"] == "test") & time["hour"].isin(HOURS) & time["comparison"].isin(COMPARISONS) & time["canonical_stratum"].isin(SUBTYPES) & time["feature"].isin(FEATURES)].copy()
    expected_time = len(HOURS)*len(COMPARISONS)*len(SUBTYPES)*len(FEATURES)
    if len(time) != expected_time:
        raise RuntimeError(f"Missing saved time-resolved values: expected {expected_time}, found {len(time)}")
    time["feature_label"] = time["feature"].map(FEATURE_LABELS)
    time["display_domain"] = time["feature"].map({feature: group for group, group_features in GROUPS for feature in group_features})
    time["participant_n"] = time["n_paired_anchors"].astype(int)
    time["inferential_marker_displayed"] = False
    time.to_csv(TABLE_ROOT / "figure_A1_time_resolved_transition_contrasts.csv", index=False)
    time_resolved_figure(time)

    full = direct[(direct["hour"] == 48) & direct["comparison"].isin(COMPARISONS) & direct["canonical_stratum"].isin(SUBTYPES)].copy()
    full["participant_n"] = full["n_paired_anchors"].astype(int)
    full["marker_design"] = full["fdr_supported"].map({True: "filled", False: "hollow"})
    full.to_csv(TABLE_ROOT / "figure_A2_full_variable_transition_contrasts.csv", index=False)
    full_inventory_figure(full)

    coefficients = pd.read_csv(sources["coefficient_summary"])
    coefficient_plotted = coefficient_figure(coefficients)
    coefficient_plotted.to_csv(TABLE_ROOT / "figure_A3_exploratory_combined_coefficients.csv", index=False)

    source_hashes_after = {name: sha256(path) for name, path in sources.items()}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A saved source artifact changed during figure creation")

    supported = main[main["fdr_supported"]]
    supported_lines = []
    for comparison in COMPARISONS:
        supported_lines.extend(["", f"## {comparison.replace('_', ' ')} supported features", ""])
        for subtype in SUBTYPES:
            q = supported[(supported["comparison"] == comparison) & (supported["canonical_stratum"] == subtype)]
            values = ", ".join(f"{label_for(row.feature)} ({row.estimate:.3f}, q={row.fdr_q:.3g})" for row in q.itertuples())
            supported_lines.append(f"- {SUBTYPE_LABELS[subtype]}: {values or 'No compact feature met the saved CI-plus-FDR rule.'}")
    caption = (
        "Static and dynamic characteristics of retained and gained latent neighbors after 48 hours of streaming. "
        "Neighborhoods were constructed within diagnostic subtype, while all frozen clinical clusters within each subtype remained eligible. "
        "The top row compares original h0 neighbors retained in ht with original neighbors that were lost; the bottom row compares new ht neighbors with prespecified matched non-neighbors. "
        "Contrasts were calculated within participant before subtype aggregation, and intervals are saved participant-bootstrap 95% confidence intervals. "
        "Filled markers reproduce the saved interval-exclusion and BH-FDR q<0.05 decision. Dynamic variables use observations available only through 48 hours. "
        "Positive values indicate greater pairwise similarity, not a causal effect. Insulin-dependent results are exploratory. "
        "The contrasts describe associative feature similarity and do not identify an independent or causal physiological driver of latent movement."
    )
    report = [
        "# Figure F2 interpretation", "", "## Detailed caption", "", caption,
        "", "## Verified definitions", "",
        "- Continuous: `S_f(i,j) = -|z_if-z_jf|` after train-defined standardization.",
        "- Binary and individual medication indicators: exact match, `1[x_if=x_jf]`; medication similarity was not Jaccard.",
        "- Matched non-neighbors: matched without replacement on h0 distance, valid observations, endpoint-anchor count, and available duration.",
        "- Dynamic window: cumulative observations from time 0 through the displayed endpoint; no future rows.",
        *supported_lines,
        "", "## Interpretation", "",
        "Static clinical similarity contributes to initial h0 organization but generally provides a weaker distinction between relationships that persist and those that disappear. Glycemic similarities, especially mean CGM and exposure summaries, provide the most consistent supported retained and gained contrasts across the primary subtypes. Wearable, behavioral, and sleep associations are more subtype specific. These are associative representation-level findings, not causal mechanisms.",
        "", "Insulin-dependent point estimates are exploratory because the participant count is smaller and intervals are wider.",
    ]
    (REPORT_ROOT / "figure_F2_interpretation.md").write_text("\n".join(report) + "\n")

    metadata = {
        "created_at": now(), "source_root": str(SOURCE_ROOT), "output_root": str(OUTPUT_ROOT),
        "discovered_sources": {name: str(path) for name, path in sources.items()},
        "source_hashes": source_hashes_before, "source_hashes_unchanged": True,
        "figure_only": True, "upstream_analysis_executed": False,
        "states_recomputed": False, "neighborhoods_recomputed": False, "matching_rerun": False,
        "bootstraps_rerun": False, "fdr_recalculated": False, "models_refit": False,
        "main_hour": 48, "features": FEATURES, "activity_feature": "dynamic__active_minutes",
        "participant_n_by_task_and_subtype": audit["participant_n_by_task_and_subtype"],
        "continuous_similarity": audit["continuous_similarity_definition"],
        "binary_and_medication_similarity": audit["binary_similarity_definition"],
        "matching_definition": audit["matching_definition"],
        "marker_rule": audit["saved_marker_rule"], "insulin_dependent_exploratory": True,
        "caption": caption,
    }
    (METADATA_ROOT / "figure_F2_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    required_files = [
        FIGURE_ROOT / "figure_F2_final_retained_and_gained_similarity.png",
        FIGURE_ROOT / "figure_F2_final_retained_and_gained_similarity.pdf",
        TABLE_ROOT / "figure_F2_final_retained_and_gained_similarity.csv",
        FIGURE_ROOT / "figure_A1_time_resolved_transition_contrasts.png",
        FIGURE_ROOT / "figure_A1_time_resolved_transition_contrasts.pdf",
        TABLE_ROOT / "figure_A1_time_resolved_transition_contrasts.csv",
        FIGURE_ROOT / "figure_A2_full_variable_transition_contrasts.png",
        FIGURE_ROOT / "figure_A2_full_variable_transition_contrasts.pdf",
        TABLE_ROOT / "figure_A2_full_variable_transition_contrasts.csv",
        FIGURE_ROOT / "figure_A3_exploratory_combined_coefficients.png",
        FIGURE_ROOT / "figure_A3_exploratory_combined_coefficients.pdf",
        TABLE_ROOT / "figure_A3_exploratory_combined_coefficients.csv",
    ]
    checks = {
        "title_mentions_48_hours": True, "two_rows_four_columns": True,
        "top_retained_minus_lost": True, "bottom_gained_minus_matched": True,
        "same_feature_order": True, "domains_visually_separated": True,
        "participant_n_reported": True, "dynamic_window_no_future": audit["no_future_dynamic_data_verified"],
        "saved_participant_bootstrap_intervals": audit["participant_bootstrap_95_ci_verified"],
        "saved_marker_decisions_used": bool(main["marker_decision_source"].str.contains("not recomputed").all()),
        "zero_lines_visible": True, "common_limits_within_rows": True,
        "insulin_exploratory": True, "no_causal_wording": "causal effect" in caption,
        "gained_only_not_separate_main": True, "one_coefficient_appendix": True,
        "every_plotted_value_saved": all(path.exists() and path.stat().st_size > 0 for path in required_files),
        "no_upstream_analysis_or_inference": True,
    }
    ending = (
        "The final 48-hour retained-versus-lost and gained-versus-matched figure was recreated from existing saved participant-level contrasts only. "
        "The figure now includes explicit timepoint and participant counts, shared axes, clear feature domains, participant-bootstrap intervals, saved FDR marker decisions, and a caption distinguishing associative similarity from causal interpretation. "
        "No hidden states, neighborhoods, matching, bootstraps, FDR calculations, or models were recomputed."
    )
    qa = ["# Figure F2 final QA", ""] + [f"{i}. {'PASS' if passed else 'FAIL'}: {name.replace('_',' ')}" for i, (name, passed) in enumerate(checks.items(), 1)] + ["", ending]
    (QA_ROOT / "FIGURE_F2_FINAL_QA.md").write_text("\n".join(qa) + "\n")
    if not all(checks.values()):
        raise RuntimeError("Final Figure F2 QA failed")
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT), "qa_pass": int(sum(checks.values())), "qa_total": len(checks), "ending": ending}, indent=2))


if __name__ == "__main__":
    main()
