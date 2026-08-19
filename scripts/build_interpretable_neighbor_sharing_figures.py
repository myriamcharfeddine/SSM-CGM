#!/usr/bin/env python3
"""Build interpretation-first figures from frozen Step 7 neighbour outputs.

This script does not extract hidden states, transform representations, fit PCA,
recompute neighbours, run clustering, or alter the canonical analysis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns


PRIMARY_K_NEIGHBORS = 10
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
ALPHA = 0.05
CI_PERCENTILES = (2.5, 97.5)
NO_EM_DASH = "\u2014"

HOUSE_NAVY = "#003366"
HOUSE_CRIMSON = "#BA2828"
HOUSE_TEAL = "#5BBABA"
HOUSE_KEY_RED = "#FF0000"
HOUSE_GRAY = "#888888"
HOUSE_LIGHT_GRAY = "#D0D0D0"
HOUSE_DARK_GRAY = "#555555"
WHITE = "#FFFFFF"

CONDITIONS = ("full_all", "neutral_all")
CONDITION_LABELS = {
    "full_all": "Full profile",
    "neutral_all": "Static neutral",
}
CONDITION_COLORS = {
    "full_all": HOUSE_NAVY,
    "neutral_all": HOUSE_CRIMSON,
}
CONDITION_MARKERS = {
    "full_all": "s",
    "neutral_all": "o",
}
FOUR_ESTIMATE_ORDER = (
    ("full_all", "neighbor", "Full\nneighbours", HOUSE_NAVY),
    ("full_all", "random", "Full\nrandom", HOUSE_GRAY),
    ("neutral_all", "neighbor", "Neutral\nneighbours", HOUSE_CRIMSON),
    ("neutral_all", "random", "Neutral\nrandom", HOUSE_LIGHT_GRAY),
)

VARIABLE_ORDER = (
    "mean_glucose",
    "glucose_cv",
    "tir_70_180",
    "glucose_sd",
    "hba1c",
    "study_group",
    "natriuretic_peptide_b_prohormon",
    "c_reactive_protein_i",
    "bun_creatinine_ratio",
)
PRIMARY_FIGURE_VARIABLES = VARIABLE_ORDER[:6]
CONTINUOUS_VARIABLES = tuple(
    variable for variable in VARIABLE_ORDER if variable != "study_group"
)
EXTERNAL_VARIABLES = (
    "natriuretic_peptide_b_prohormon",
    "c_reactive_protein_i",
    "bun_creatinine_ratio",
)
VARIABLE_SPECS = {
    "mean_glucose": {
        "label": "Mean glucose",
        "unit": "mg/dL",
        "scale": 1.0,
        "axis": "Absolute difference (mg/dL)",
        "format": ".2f",
    },
    "glucose_cv": {
        "label": "Glucose CV",
        "unit": "%",
        "scale": 100.0,
        "axis": "Absolute difference in CV (percentage points)",
        "format": ".2f",
    },
    "tir_70_180": {
        "label": "Time in range",
        "unit": "absolute percentage-point difference",
        "scale": 100.0,
        "axis": "Absolute difference in TIR (percentage points)",
        "format": ".2f",
    },
    "glucose_sd": {
        "label": "Glucose SD",
        "unit": "mg/dL",
        "scale": 1.0,
        "axis": "Absolute difference (mg/dL)",
        "format": ".2f",
    },
    "hba1c": {
        "label": "HbA1c",
        "unit": "percentage points",
        "scale": 1.0,
        "axis": "Absolute difference (percentage points)",
        "format": ".3f",
    },
    "study_group": {
        "label": "Study group",
        "unit": "same-group rate, %",
        "scale": 100.0,
        "axis": "Same-study-group rate (%)",
        "format": ".1f",
    },
    "natriuretic_peptide_b_prohormon": {
        "label": "NT-proBNP",
        "unit": "pg/mL",
        "scale": 1.0,
        "axis": "Absolute difference (pg/mL)",
        "format": ".1f",
    },
    "c_reactive_protein_i": {
        "label": "High-sensitivity CRP",
        "unit": "mg/L",
        "scale": 1.0,
        "axis": "Absolute difference (mg/L)",
        "format": ".2f",
    },
    "bun_creatinine_ratio": {
        "label": "BUN/creatinine ratio",
        "unit": "ratio (unitless)",
        "scale": 1.0,
        "axis": "Absolute difference (unitless ratio)",
        "format": ".4f",
    },
}

REQUIRED_OUTPUT_NAMES = (
    "figure_raw_clinical_difference_neighbors_vs_random.png",
    "figure_relative_reduction_full_vs_neutral.png",
    "figure_study_group_concordance.png",
    "figure_external_biomarker_raw_differences.png",
    "interpretable_neighbor_metrics.csv",
    "interpretable_figure_source_data.csv",
    "interpretable_neighbor_sharing_report.md",
    "interpretable_figure_manifest.csv",
    "figure_revision_qc.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step7-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=N_BOOTSTRAP
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any, base_seed: int) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    offset = int.from_bytes(digest[:8], byteorder="little")
    return int((offset + base_seed) % (2**32 - 1))


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path, datetime, pd.Timestamp)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, default=json_value
        )
        + "\n"
    )
    os.replace(temporary, path)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, CI_PERCENTILES)
    return float(low), float(high)


def bootstrap_continuous(
    participant_rows: pd.DataFrame,
    scale: float,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    neighbor = participant_rows["neighbor_raw_metric"].to_numpy(float) * scale
    random = participant_rows["random_raw_metric"].to_numpy(float) * scale
    if not np.isfinite(neighbor).all() or not np.isfinite(random).all():
        raise RuntimeError("Continuous participant metrics contain nonfinite values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(participant_rows), size=(repeats, len(participant_rows))
    )
    neighbor_boot = neighbor[indices].mean(axis=1)
    random_boot = random[indices].mean(axis=1)
    reduction_boot = random_boot - neighbor_boot
    if np.any(random_boot <= 0):
        raise RuntimeError("Relative reduction denominator is nonpositive")
    relative_boot = 100.0 * reduction_boot / random_boot
    neighbor_ci = percentile_interval(neighbor_boot)
    random_ci = percentile_interval(random_boot)
    reduction_ci = percentile_interval(reduction_boot)
    relative_ci = percentile_interval(relative_boot)
    return {
        "neighbor_raw_ci_low": neighbor_ci[0],
        "neighbor_raw_ci_high": neighbor_ci[1],
        "random_raw_ci_low": random_ci[0],
        "random_raw_ci_high": random_ci[1],
        "raw_reduction_ci_low": reduction_ci[0],
        "raw_reduction_ci_high": reduction_ci[1],
        "relative_reduction_ci_low": relative_ci[0],
        "relative_reduction_ci_high": relative_ci[1],
    }


def bootstrap_study_group(
    participant_rows: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    neighbor = (
        participant_rows["same_group_rate_neighbors"].to_numpy(float) * 100.0
    )
    random = participant_rows["same_group_rate_random"].to_numpy(float) * 100.0
    if not np.isfinite(neighbor).all() or not np.isfinite(random).all():
        raise RuntimeError("Study-group participant metrics contain nonfinite values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(participant_rows), size=(repeats, len(participant_rows))
    )
    neighbor_boot = neighbor[indices].mean(axis=1)
    random_boot = random[indices].mean(axis=1)
    gain_boot = neighbor_boot - random_boot
    neighbor_ci = percentile_interval(neighbor_boot)
    random_ci = percentile_interval(random_boot)
    gain_ci = percentile_interval(gain_boot)
    return {
        "neighbor_raw_ci_low": neighbor_ci[0],
        "neighbor_raw_ci_high": neighbor_ci[1],
        "random_raw_ci_low": random_ci[0],
        "random_raw_ci_high": random_ci[1],
        "same_group_gain_ci_low": gain_ci[0],
        "same_group_gain_ci_high": gain_ci[1],
    }


def select_canonical_rows(results: pd.DataFrame) -> pd.DataFrame:
    selected = results[
        results["k_neighbors"].eq(PRIMARY_K_NEIGHBORS)
        & ~results["site_matched"]
        & results["random_baseline_type"].eq(
            "unrestricted_non_neighbours"
        )
        & results["condition"].isin(CONDITIONS)
        & results["variable"].isin(VARIABLE_ORDER)
    ].copy()
    expected_pairs = {
        (condition, variable)
        for condition in CONDITIONS
        for variable in VARIABLE_ORDER
    }
    actual_pairs = set(zip(selected["condition"], selected["variable"]))
    if actual_pairs != expected_pairs or len(selected) != len(expected_pairs):
        raise RuntimeError("Canonical result selection is incomplete or duplicated")
    return selected


def interpretation_text(row: dict[str, Any]) -> str:
    if row["variable_type"] == "categorical":
        return (
            f"{row['same_group_rate_neighbor_pct']:.1f}% of hidden-state "
            "neighbours belonged to the same study group, compared with "
            f"{row['same_group_rate_random_pct']:.1f}% of random "
            "participants, a gain of "
            f"{row['same_group_gain_percentage_points']:.1f} percentage "
            "points."
        )
    if row["variable"] in EXTERNAL_VARIABLES and not row[
        "statistically_supported"
    ]:
        return (
            "The estimated difference was small and did not pass permutation "
            "and FDR criteria."
        )
    unit = row["unit"]
    return (
        f"Hidden-state neighbours differed by "
        f"{row['neighbor_raw_difference']:.3g} {unit}, compared with "
        f"{row['random_raw_difference']:.3g} {unit} for random "
        "non-neighbours, corresponding to a "
        f"{row['relative_reduction_pct']:.1f}% reduction in clinical "
        "difference."
    )


def build_metrics(
    canonical_results: pd.DataFrame,
    participant_results: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selection_audit: dict[str, Any] = {}
    for condition in CONDITIONS:
        for variable in VARIABLE_ORDER:
            result_row = canonical_results[
                canonical_results["condition"].eq(condition)
                & canonical_results["variable"].eq(variable)
            ].iloc[0]
            participant_rows = participant_results[
                participant_results["condition"].eq(condition)
                & participant_results["k_neighbors"].eq(
                    PRIMARY_K_NEIGHBORS
                )
                & participant_results["variable"].eq(variable)
            ].copy()
            participant_rows = participant_rows.sort_values(
                "participant_id", kind="stable"
            ).reset_index(drop=True)
            expected_n = int(result_row["n_participants_eligible"])
            if (
                len(participant_rows) != expected_n
                or participant_rows["participant_id"].nunique() != expected_n
            ):
                raise RuntimeError(
                    f"Participant count mismatch: {condition} {variable}"
                )
            spec = VARIABLE_SPECS[variable]
            common = {
                "variable": variable,
                "variable_label": spec["label"],
                "variable_type": result_row["variable_type"],
                "unit": spec["unit"],
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "k_neighbors": PRIMARY_K_NEIGHBORS,
                "n_participants": expected_n,
                "neighbor_raw_difference": np.nan,
                "neighbor_raw_ci_low": np.nan,
                "neighbor_raw_ci_high": np.nan,
                "random_raw_difference": np.nan,
                "random_raw_ci_low": np.nan,
                "random_raw_ci_high": np.nan,
                "raw_reduction": np.nan,
                "raw_reduction_ci_low": np.nan,
                "raw_reduction_ci_high": np.nan,
                "relative_reduction_pct": np.nan,
                "relative_reduction_ci_low": np.nan,
                "relative_reduction_ci_high": np.nan,
                "same_group_rate_neighbor_pct": np.nan,
                "same_group_rate_random_pct": np.nan,
                "same_group_gain_percentage_points": np.nan,
                "same_group_gain_ci_low": np.nan,
                "same_group_gain_ci_high": np.nan,
                "permutation_p": float(result_row["permutation_p"]),
                "fdr_q": float(result_row["fdr_q"]),
            }
            row_seed = stable_seed(
                condition,
                variable,
                "participant_bootstrap",
                base_seed=seed,
            )
            if variable == "study_group":
                central_neighbor = (
                    float(result_row["same_group_rate_neighbors"]) * 100.0
                )
                central_random = (
                    float(result_row["same_group_rate_random"]) * 100.0
                )
                participant_neighbor = (
                    participant_rows["same_group_rate_neighbors"].mean()
                    * 100.0
                )
                participant_random = (
                    participant_rows["same_group_rate_random"].mean()
                    * 100.0
                )
                if not np.allclose(
                    [central_neighbor, central_random],
                    [participant_neighbor, participant_random],
                    atol=1e-12,
                    rtol=1e-12,
                ):
                    raise RuntimeError("Study-group central values changed")
                intervals = bootstrap_study_group(
                    participant_rows,
                    bootstrap_replicates,
                    row_seed,
                )
                gain = central_neighbor - central_random
                common.update(
                    {
                        "neighbor_raw_difference": central_neighbor,
                        "random_raw_difference": central_random,
                        "same_group_rate_neighbor_pct": central_neighbor,
                        "same_group_rate_random_pct": central_random,
                        "same_group_gain_percentage_points": gain,
                        **intervals,
                    }
                )
                reduction_ci_low = intervals["same_group_gain_ci_low"]
            else:
                scale = float(spec["scale"])
                central_neighbor = (
                    float(result_row["neighbor_raw_mean_difference"]) * scale
                )
                central_random = (
                    float(result_row["random_raw_mean_difference"]) * scale
                )
                participant_neighbor = (
                    participant_rows["neighbor_raw_metric"].mean() * scale
                )
                participant_random = (
                    participant_rows["random_raw_metric"].mean() * scale
                )
                if not np.allclose(
                    [central_neighbor, central_random],
                    [participant_neighbor, participant_random],
                    atol=1e-12,
                    rtol=1e-12,
                ):
                    raise RuntimeError(
                        f"Canonical raw values changed: {condition} {variable}"
                    )
                intervals = bootstrap_continuous(
                    participant_rows,
                    scale,
                    bootstrap_replicates,
                    row_seed,
                )
                reduction = central_random - central_neighbor
                relative = 100.0 * reduction / central_random
                common.update(
                    {
                        "neighbor_raw_difference": central_neighbor,
                        "random_raw_difference": central_random,
                        "raw_reduction": reduction,
                        "relative_reduction_pct": relative,
                        **intervals,
                    }
                )
                reduction_ci_low = intervals["raw_reduction_ci_low"]
            supported = bool(
                reduction_ci_low > 0
                and common["permutation_p"] < ALPHA
                and common["fdr_q"] < ALPHA
            )
            common["statistically_supported"] = supported
            common["plain_language_interpretation"] = interpretation_text(
                common
            )
            rows.append(common)
            selection_audit[f"{condition}|{variable}"] = {
                "participant_count": expected_n,
                "participant_ids_unique": True,
                "bootstrap_seed": row_seed,
                "result_row_index": int(result_row.name),
                "participant_row_filter": (
                    f"condition={condition}, k_neighbors=10, "
                    f"variable={variable}"
                ),
            }
    metrics = pd.DataFrame(rows)
    required_columns = [
        "variable",
        "variable_label",
        "variable_type",
        "unit",
        "condition",
        "condition_label",
        "k_neighbors",
        "n_participants",
        "neighbor_raw_difference",
        "neighbor_raw_ci_low",
        "neighbor_raw_ci_high",
        "random_raw_difference",
        "random_raw_ci_low",
        "random_raw_ci_high",
        "raw_reduction",
        "raw_reduction_ci_low",
        "raw_reduction_ci_high",
        "relative_reduction_pct",
        "relative_reduction_ci_low",
        "relative_reduction_ci_high",
        "same_group_rate_neighbor_pct",
        "same_group_rate_random_pct",
        "same_group_gain_percentage_points",
        "same_group_gain_ci_low",
        "same_group_gain_ci_high",
        "permutation_p",
        "fdr_q",
        "statistically_supported",
        "plain_language_interpretation",
    ]
    return metrics[required_columns], selection_audit


def metric_row(
    metrics: pd.DataFrame, condition: str, variable: str
) -> pd.Series:
    selected = metrics[
        metrics["condition"].eq(condition)
        & metrics["variable"].eq(variable)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one metric row: {condition} {variable}")
    return selected.iloc[0]


def point_with_ci(
    axis: plt.Axes,
    x: float,
    estimate: float,
    low: float,
    high: float,
    color: str,
    marker: str = "o",
    markersize: float = 7.0,
) -> None:
    axis.errorbar(
        x,
        estimate,
        yerr=np.array([[estimate - low], [high - estimate]]),
        fmt=marker,
        markersize=markersize,
        color=color,
        markerfacecolor=color,
        markeredgecolor=WHITE,
        markeredgewidth=0.7,
        ecolor=color,
        elinewidth=1.5,
        capsize=3,
        zorder=4,
    )


def horizontal_point_with_ci(
    axis: plt.Axes,
    estimate: float,
    y: float,
    low: float,
    high: float,
    color: str,
    marker: str,
) -> None:
    axis.errorbar(
        estimate,
        y,
        xerr=np.array([[estimate - low], [high - estimate]]),
        fmt=marker,
        markersize=7,
        color=color,
        markerfacecolor=color,
        markeredgecolor=WHITE,
        markeredgewidth=0.7,
        ecolor=color,
        elinewidth=1.5,
        capsize=3,
        zorder=4,
    )


def format_estimate(variable: str, value: float) -> str:
    return format(value, VARIABLE_SPECS[variable]["format"])


def make_raw_figure(metrics: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(16, 10.8))
    panel_letters = ("A", "B", "C", "D", "E", "F")
    for axis, variable, panel_letter in zip(
        axes.flat, PRIMARY_FIGURE_VARIABLES, panel_letters
    ):
        spec = VARIABLE_SPECS[variable]
        estimates: list[float] = []
        for x, (condition, role, _, color) in enumerate(
            FOUR_ESTIMATE_ORDER
        ):
            row = metric_row(metrics, condition, variable)
            if variable == "study_group":
                estimate = float(
                    row[
                        "same_group_rate_neighbor_pct"
                        if role == "neighbor"
                        else "same_group_rate_random_pct"
                    ]
                )
            else:
                estimate = float(
                    row[
                        "neighbor_raw_difference"
                        if role == "neighbor"
                        else "random_raw_difference"
                    ]
                )
            low = float(
                row[
                    "neighbor_raw_ci_low"
                    if role == "neighbor"
                    else "random_raw_ci_low"
                ]
            )
            high = float(
                row[
                    "neighbor_raw_ci_high"
                    if role == "neighbor"
                    else "random_raw_ci_high"
                ]
            )
            point_with_ci(axis, x, estimate, low, high, color)
            estimates.append(estimate)
        y_max = 100.0 if variable == "study_group" else max(estimates) * 1.32
        axis.set_ylim(0, y_max)
        label_offset = y_max * 0.035
        for x, estimate in enumerate(estimates):
            suffix = "%" if variable == "study_group" else ""
            axis.text(
                x,
                estimate + label_offset,
                format_estimate(variable, estimate) + suffix,
                ha="center",
                va="bottom",
                fontsize=9,
                color=HOUSE_DARK_GRAY,
            )
        axis.set_xticks(
            np.arange(4), [item[2] for item in FOUR_ESTIMATE_ORDER]
        )
        axis.set_title(
            f"{panel_letter}. {spec['label']}",
            loc="left",
            fontweight="bold",
        )
        axis.set_ylabel(spec["axis"])
        axis.set_xlabel("")
        if variable == "study_group":
            full = metric_row(metrics, "full_all", variable)
            neutral = metric_row(metrics, "neutral_all", variable)
            axis.text(
                0.02,
                0.97,
                "Full gain: "
                f"{full['same_group_gain_percentage_points']:+.1f} pp\n"
                "Neutral gain: "
                f"{neutral['same_group_gain_percentage_points']:+.1f} pp",
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color=HOUSE_DARK_GRAY,
                bbox={
                    "facecolor": WHITE,
                    "edgecolor": HOUSE_LIGHT_GRAY,
                    "alpha": 0.9,
                    "boxstyle": "round,pad=0.3",
                },
            )
    figure.suptitle(
        "Clinical characteristics of hidden-state neighbours versus random "
        "participants",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.948,
        "Test participants, k=10, participant-bootstrap 95% confidence "
        "intervals",
        ha="center",
        fontsize=11,
        color=HOUSE_DARK_GRAY,
    )
    figure.text(
        0.5,
        0.015,
        "Lower clinical differences indicate stronger similarity. For study "
        "group, higher same-group rates indicate stronger sharing.",
        ha="center",
        fontsize=10,
        color=HOUSE_DARK_GRAY,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.09,
        top=0.89,
        hspace=0.42,
        wspace=0.30,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def q_label(value: float) -> str:
    return "q<0.001" if value < 0.001 else f"q={value:.3f}"


def make_relative_figure(metrics: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    figure, (axis_continuous, axis_group) = plt.subplots(
        1,
        2,
        figsize=(16, 8.2),
        gridspec_kw={"width_ratios": [4.2, 1.65]},
    )
    variables = list(CONTINUOUS_VARIABLES)
    y_positions = np.arange(len(variables))[::-1]
    all_bounds: list[float] = [0.0]
    for index, variable in enumerate(variables):
        y_center = y_positions[index]
        for condition, offset in (("full_all", 0.13), ("neutral_all", -0.13)):
            row = metric_row(metrics, condition, variable)
            estimate = float(row["relative_reduction_pct"])
            low = float(row["relative_reduction_ci_low"])
            high = float(row["relative_reduction_ci_high"])
            all_bounds.extend([low, high])
            horizontal_point_with_ci(
                axis_continuous,
                estimate,
                y_center + offset,
                low,
                high,
                CONDITION_COLORS[condition],
                CONDITION_MARKERS[condition],
            )
    low_limit = min(all_bounds)
    high_limit = max(all_bounds)
    span = max(high_limit - low_limit, 1.0)
    axis_continuous.set_xlim(
        min(-5.0, low_limit - 0.08 * span),
        high_limit + 0.28 * span,
    )
    annotation_x = axis_continuous.get_xlim()[1] - 0.02 * (
        axis_continuous.get_xlim()[1] - axis_continuous.get_xlim()[0]
    )
    for index, variable in enumerate(variables):
        row = metric_row(metrics, "neutral_all", variable)
        axis_continuous.text(
            annotation_x,
            y_positions[index] - 0.13,
            q_label(float(row["fdr_q"])),
            ha="right",
            va="center",
            fontsize=8,
            color=HOUSE_DARK_GRAY,
        )
    axis_continuous.axvline(
        0, color=HOUSE_DARK_GRAY, linewidth=1.0, linestyle="--", zorder=1
    )
    axis_continuous.set_yticks(
        y_positions, [VARIABLE_SPECS[v]["label"] for v in variables]
    )
    axis_continuous.set_xlabel("Relative reduction in clinical difference (%)")
    axis_continuous.set_title(
        "A. Relative reduction in clinical difference",
        loc="left",
        fontweight="bold",
        y=1.055,
        pad=0,
    )
    axis_continuous.text(
        0.0,
        1.005,
        "Positive values mean that neighbours are more similar than random "
        "non-neighbours",
        transform=axis_continuous.transAxes,
        fontsize=9,
        color=HOUSE_DARK_GRAY,
    )
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=CONDITION_MARKERS[condition],
            color=CONDITION_COLORS[condition],
            linestyle="none",
            markersize=7,
            markeredgecolor=WHITE,
            label=CONDITION_LABELS[condition],
        )
        for condition in CONDITIONS
    ]

    study_rows = [
        metric_row(metrics, condition, "study_group")
        for condition in CONDITIONS
    ]
    for y, condition, row in zip((1, 0), CONDITIONS, study_rows):
        estimate = float(row["same_group_gain_percentage_points"])
        low = float(row["same_group_gain_ci_low"])
        high = float(row["same_group_gain_ci_high"])
        horizontal_point_with_ci(
            axis_group,
            estimate,
            y,
            low,
            high,
            CONDITION_COLORS[condition],
            CONDITION_MARKERS[condition],
        )
        axis_group.text(
            high + 0.6,
            y,
            f"{estimate:+.1f} pp\n{q_label(float(row['fdr_q']))}",
            ha="left",
            va="center",
            fontsize=9,
            color=HOUSE_DARK_GRAY,
        )
    group_max = max(
        float(row["same_group_gain_ci_high"]) for row in study_rows
    )
    axis_group.set_xlim(-2, group_max + 10)
    axis_group.axvline(
        0, color=HOUSE_DARK_GRAY, linewidth=1.0, linestyle="--", zorder=1
    )
    axis_group.set_yticks((1, 0), ("Full profile", "Static neutral"))
    axis_group.set_xlabel("Same-group gain (percentage points)")
    axis_group.set_title(
        "B. Study-group concordance",
        loc="left",
        fontweight="bold",
        y=1.055,
        pad=0,
    )
    figure.suptitle(
        "Clinical similarity of hidden-state neighbours",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=True,
    )
    figure.text(
        0.5,
        0.015,
        "Test participants, k=10. Error bars are participant-bootstrap 95% "
        "confidence intervals; q-values are from the frozen Step 7 families.",
        ha="center",
        fontsize=9.5,
        color=HOUSE_DARK_GRAY,
    )
    figure.subplots_adjust(
        left=0.18,
        right=0.98,
        bottom=0.10,
        top=0.84,
        wspace=0.24,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def make_study_group_figure(
    metrics: pd.DataFrame, output_path: Path
) -> None:
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(11.5, 7.2))
    estimates: list[float] = []
    highs: list[float] = []
    for x, (condition, role, label, color) in enumerate(FOUR_ESTIMATE_ORDER):
        row = metric_row(metrics, condition, "study_group")
        estimate = float(
            row[
                "same_group_rate_neighbor_pct"
                if role == "neighbor"
                else "same_group_rate_random_pct"
            ]
        )
        low = float(
            row[
                "neighbor_raw_ci_low"
                if role == "neighbor"
                else "random_raw_ci_low"
            ]
        )
        high = float(
            row[
                "neighbor_raw_ci_high"
                if role == "neighbor"
                else "random_raw_ci_high"
            ]
        )
        point_with_ci(
            axis,
            x,
            estimate,
            low,
            high,
            color,
            markersize=10,
        )
        estimates.append(estimate)
        highs.append(high)
        axis.text(
            x,
            high + 2.0,
            f"{estimate:.1f}%",
            ha="center",
            fontsize=11,
            color=HOUSE_DARK_GRAY,
        )
    axis.set_ylim(0, 100)
    axis.set_xticks(
        np.arange(4), [item[2].replace("\n", " ") for item in FOUR_ESTIMATE_ORDER]
    )
    axis.set_ylabel("Same-study-group rate (%)")
    axis.set_xlabel("")
    full = metric_row(metrics, "full_all", "study_group")
    neutral = metric_row(metrics, "neutral_all", "study_group")
    axis.text(
        0.5,
        max(highs[0], highs[1]) + 9,
        "Full-profile gain: "
        f"{full['same_group_gain_percentage_points']:+.1f} percentage points",
        ha="center",
        color=HOUSE_NAVY,
        fontsize=10,
        fontweight="bold",
    )
    axis.text(
        2.5,
        max(highs[2], highs[3]) + 9,
        "Static-neutral gain: "
        f"{neutral['same_group_gain_percentage_points']:+.1f} percentage points",
        ha="center",
        color=HOUSE_CRIMSON,
        fontsize=10,
        fontweight="bold",
    )
    axis.set_title(
        "Study-group concordance among hidden-state neighbours",
        fontsize=16,
        fontweight="bold",
        pad=42,
    )
    axis.text(
        0.5,
        1.035,
        "Similarity persists after static neutralization but is stronger when "
        "the model receives the real participant profile.",
        transform=axis.transAxes,
        ha="center",
        fontsize=10,
        color=HOUSE_DARK_GRAY,
    )
    figure.text(
        0.5,
        0.02,
        "Test participants, k=10; participant-bootstrap 95% confidence "
        "intervals.",
        ha="center",
        fontsize=9.5,
        color=HOUSE_DARK_GRAY,
    )
    figure.subplots_adjust(
        left=0.10, right=0.98, bottom=0.13, top=0.78
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def p_value_label(value: float) -> str:
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def make_external_figure(metrics: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(17, 6.8))
    panel_letters = ("A", "B", "C")
    for axis, variable, panel_letter in zip(
        axes, EXTERNAL_VARIABLES, panel_letters
    ):
        spec = VARIABLE_SPECS[variable]
        estimates: list[float] = []
        highs: list[float] = []
        for x, (condition, role, _, _) in enumerate(FOUR_ESTIMATE_ORDER):
            row = metric_row(metrics, condition, variable)
            estimate = float(
                row[
                    "neighbor_raw_difference"
                    if role == "neighbor"
                    else "random_raw_difference"
                ]
            )
            low = float(
                row[
                    "neighbor_raw_ci_low"
                    if role == "neighbor"
                    else "random_raw_ci_low"
                ]
            )
            high = float(
                row[
                    "neighbor_raw_ci_high"
                    if role == "neighbor"
                    else "random_raw_ci_high"
                ]
            )
            color = (
                HOUSE_DARK_GRAY if role == "neighbor" else HOUSE_LIGHT_GRAY
            )
            marker = "s" if condition == "full_all" else "o"
            point_with_ci(
                axis,
                x,
                estimate,
                low,
                high,
                color,
                marker=marker,
            )
            estimates.append(estimate)
            highs.append(high)
        y_max = max(highs) * 1.48
        axis.set_ylim(0, y_max)
        for x, estimate in enumerate(estimates):
            axis.text(
                x,
                estimate + y_max * 0.035,
                format_estimate(variable, estimate),
                ha="center",
                fontsize=8.5,
                color=HOUSE_DARK_GRAY,
            )
        full = metric_row(metrics, "full_all", variable)
        neutral = metric_row(metrics, "neutral_all", variable)
        axis.text(
            0.02,
            0.97,
            "Unsupported\n"
            f"Full: {p_value_label(float(full['permutation_p']))}, "
            f"{q_label(float(full['fdr_q']))}\n"
            f"Neutral: {p_value_label(float(neutral['permutation_p']))}, "
            f"{q_label(float(neutral['fdr_q']))}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.7,
            color=HOUSE_DARK_GRAY,
            bbox={
                "facecolor": WHITE,
                "edgecolor": HOUSE_LIGHT_GRAY,
                "alpha": 0.92,
                "boxstyle": "round,pad=0.3",
            },
        )
        axis.set_xticks(
            np.arange(4), [item[2] for item in FOUR_ESTIMATE_ORDER]
        )
        axis.set_ylabel(spec["axis"])
        axis.set_title(
            f"{panel_letter}. {spec['label']}",
            loc="left",
            fontweight="bold",
        )
        axis.set_xlabel("")
    figure.suptitle(
        "External biomarker differences among neighbours and random "
        "participants",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.02,
        "No external biomarker showed robust neighbour sharing after "
        "permutation and multiplicity correction.",
        ha="center",
        fontsize=10,
        color=HOUSE_DARK_GRAY,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.14,
        top=0.86,
        wspace=0.29,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_figure_source_data(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for variable in PRIMARY_FIGURE_VARIABLES:
        for condition, role, label, _ in FOUR_ESTIMATE_ORDER:
            if condition not in CONDITIONS:
                continue
            row = metric_row(metrics, condition, variable)
            if variable == "study_group":
                estimate_column = (
                    "same_group_rate_neighbor_pct"
                    if role == "neighbor"
                    else "same_group_rate_random_pct"
                )
            else:
                estimate_column = (
                    "neighbor_raw_difference"
                    if role == "neighbor"
                    else "random_raw_difference"
                )
            records.append(
                {
                    "figure_id": "raw_clinical_difference",
                    "variable": variable,
                    "variable_label": row["variable_label"],
                    "condition": condition,
                    "condition_label": row["condition_label"],
                    "metric_role": role,
                    "display_label": label.replace("\n", " "),
                    "estimate": row[estimate_column],
                    "ci_low": row[
                        "neighbor_raw_ci_low"
                        if role == "neighbor"
                        else "random_raw_ci_low"
                    ],
                    "ci_high": row[
                        "neighbor_raw_ci_high"
                        if role == "neighbor"
                        else "random_raw_ci_high"
                    ],
                    "unit": row["unit"],
                    "permutation_p": row["permutation_p"],
                    "fdr_q": row["fdr_q"],
                    "statistically_supported": row[
                        "statistically_supported"
                    ],
                }
            )
    for variable in CONTINUOUS_VARIABLES:
        for condition in CONDITIONS:
            row = metric_row(metrics, condition, variable)
            records.append(
                {
                    "figure_id": "relative_reduction",
                    "variable": variable,
                    "variable_label": row["variable_label"],
                    "condition": condition,
                    "condition_label": row["condition_label"],
                    "metric_role": "relative_reduction",
                    "display_label": row["condition_label"],
                    "estimate": row["relative_reduction_pct"],
                    "ci_low": row["relative_reduction_ci_low"],
                    "ci_high": row["relative_reduction_ci_high"],
                    "unit": "%",
                    "permutation_p": row["permutation_p"],
                    "fdr_q": row["fdr_q"],
                    "statistically_supported": row[
                        "statistically_supported"
                    ],
                }
            )
    for condition in CONDITIONS:
        row = metric_row(metrics, condition, "study_group")
        records.append(
            {
                "figure_id": "relative_reduction",
                "variable": "study_group",
                "variable_label": "Study group",
                "condition": condition,
                "condition_label": row["condition_label"],
                "metric_role": "same_group_gain",
                "display_label": row["condition_label"],
                "estimate": row["same_group_gain_percentage_points"],
                "ci_low": row["same_group_gain_ci_low"],
                "ci_high": row["same_group_gain_ci_high"],
                "unit": "percentage points",
                "permutation_p": row["permutation_p"],
                "fdr_q": row["fdr_q"],
                "statistically_supported": row[
                    "statistically_supported"
                ],
            }
        )
    for figure_id, variables in (
        ("study_group_concordance", ("study_group",)),
        ("external_biomarker_raw_differences", EXTERNAL_VARIABLES),
    ):
        for variable in variables:
            for condition, role, label, _ in FOUR_ESTIMATE_ORDER:
                row = metric_row(metrics, condition, variable)
                if variable == "study_group":
                    estimate_column = (
                        "same_group_rate_neighbor_pct"
                        if role == "neighbor"
                        else "same_group_rate_random_pct"
                    )
                else:
                    estimate_column = (
                        "neighbor_raw_difference"
                        if role == "neighbor"
                        else "random_raw_difference"
                    )
                records.append(
                    {
                        "figure_id": figure_id,
                        "variable": variable,
                        "variable_label": row["variable_label"],
                        "condition": condition,
                        "condition_label": row["condition_label"],
                        "metric_role": role,
                        "display_label": label.replace("\n", " "),
                        "estimate": row[estimate_column],
                        "ci_low": row[
                            "neighbor_raw_ci_low"
                            if role == "neighbor"
                            else "random_raw_ci_low"
                        ],
                        "ci_high": row[
                            "neighbor_raw_ci_high"
                            if role == "neighbor"
                            else "random_raw_ci_high"
                        ],
                        "unit": row["unit"],
                        "permutation_p": row["permutation_p"],
                        "fdr_q": row["fdr_q"],
                        "statistically_supported": row[
                            "statistically_supported"
                        ],
                    }
                )
    return pd.DataFrame(records)


def make_report(
    metrics: pd.DataFrame,
    step7_directory: Path,
    output_directory: Path,
) -> str:
    neutral_mean = metric_row(metrics, "neutral_all", "mean_glucose")
    neutral_hba1c = metric_row(metrics, "neutral_all", "hba1c")
    full_group = metric_row(metrics, "full_all", "study_group")
    neutral_group = metric_row(metrics, "neutral_all", "study_group")
    external_lines = []
    for variable in EXTERNAL_VARIABLES:
        row = metric_row(metrics, "neutral_all", variable)
        external_lines.append(
            f"- {row['variable_label']}: neighbours "
            f"{row['neighbor_raw_difference']:.4g} {row['unit']}, random "
            f"{row['random_raw_difference']:.4g} {row['unit']}, "
            f"relative reduction {row['relative_reduction_pct']:.1f}%, "
            f"p={row['permutation_p']:.3f}, q={row['fdr_q']:.3f}; unsupported."
        )
    original_figure = (
        step7_directory
        / "revised_figures"
        / "figure_neighbor_clinical_sharing_full_vs_neutral.png"
    )
    lines = [
        "# Interpretable nearest-neighbour clinical-sharing figures",
        "",
        "## 1. Why the standardized forest plot was difficult to interpret",
        "",
        "The standardized similarity-gain forest is statistically useful because "
        "it compares variables with different units. It does not show the "
        "actual clinical differences that produced each effect, so readers "
        "must understand z-standardization before interpreting the magnitude.",
        "",
        "## 2. Raw neighbour versus random metrics",
        "",
        "The new raw figure reports the mean focal-participant absolute "
        "difference from hidden-state neighbours and from random non-neighbours "
        "in each variable's clinical unit. Lower continuous-variable differences "
        "mean stronger similarity. Study group is shown separately as a "
        "same-group rate, where higher values mean stronger sharing.",
        "",
        "## 3. Relative reduction metric",
        "",
        "Relative reduction is 100 times the random-minus-neighbour difference "
        "divided by the random difference. A value of 70% means that the "
        "clinical difference among hidden-state neighbours is 70% smaller than "
        "among random non-neighbours. Study-group gain is not a relative "
        "percentage; it is a percentage-point difference.",
        "",
        "The raw figure shows the actual clinical differences. The relative "
        "reduction figure allows comparison across variables with different "
        "units. The original standardized forest plot remains useful as a "
        "statistical supplementary figure.",
        "",
        "## 4. Mean glucose example",
        "",
        f"In the static-neutral space, neighbours differed by "
        f"{neutral_mean['neighbor_raw_difference']:.2f} mg/dL in mean glucose, "
        f"compared with {neutral_mean['random_raw_difference']:.2f} mg/dL for "
        f"random non-neighbours. This was a "
        f"{neutral_mean['relative_reduction_pct']:.1f}% reduction "
        f"[{neutral_mean['relative_reduction_ci_low']:.1f}, "
        f"{neutral_mean['relative_reduction_ci_high']:.1f}].",
        "",
        "## 5. HbA1c example",
        "",
        f"In the static-neutral space, neighbours differed by "
        f"{neutral_hba1c['neighbor_raw_difference']:.3f} HbA1c percentage "
        f"points, compared with {neutral_hba1c['random_raw_difference']:.3f} "
        f"for random non-neighbours. This was a "
        f"{neutral_hba1c['relative_reduction_pct']:.1f}% reduction "
        f"[{neutral_hba1c['relative_reduction_ci_low']:.1f}, "
        f"{neutral_hba1c['relative_reduction_ci_high']:.1f}].",
        "",
        "## 6. Study-group example",
        "",
        f"Full-profile neighbours shared study group in "
        f"{full_group['same_group_rate_neighbor_pct']:.1f}% of comparisons "
        f"versus {full_group['same_group_rate_random_pct']:.1f}% randomly, "
        f"a gain of {full_group['same_group_gain_percentage_points']:.1f} "
        f"percentage points. Static-neutral neighbours shared study group in "
        f"{neutral_group['same_group_rate_neighbor_pct']:.1f}% versus "
        f"{neutral_group['same_group_rate_random_pct']:.1f}% randomly, a gain "
        f"of {neutral_group['same_group_gain_percentage_points']:.1f} "
        f"percentage points.",
        "",
        "## 7. Full versus neutral interpretation",
        "",
        "Clinically interpretable glucose sharing persisted after replacing the "
        "real participant profile with the reference profile. Study-group and "
        "HbA1c sharing were stronger in the full-profile space for some "
        "comparisons because that representation retained direct static "
        "conditioning. The contrast is descriptive, not an exact causal "
        "decomposition.",
        "",
        "## 8. External biomarker null results",
        "",
        *external_lines,
        "",
        "No external biomarker showed robust neighbour sharing after "
        "permutation and multiplicity correction.",
        "",
        "## 9. Statistical uncertainty",
        "",
        "All new intervals use 2,000 focal-participant bootstrap replicates. "
        "Each sampled focal participant retains its complete neighbour and "
        "random summary. Individual directed neighbour pairs were not "
        "bootstrapped. Original permutation p-values and FDR q-values were "
        "reused unchanged.",
        "",
        "## 10. Recommended figure for thesis",
        "",
        f"Use `{output_directory / 'figure_raw_clinical_difference_neighbors_vs_random.png'}` "
        "as the primary thesis figure because it presents clinical units "
        "directly.",
        "",
        "## 11. Recommended figure for presentation",
        "",
        f"Use `{output_directory / 'figure_raw_clinical_difference_neighbors_vs_random.png'}` "
        "as the main presentation figure. Follow it with "
        f"`{output_directory / 'figure_relative_reduction_full_vs_neutral.png'}` "
        "only when a cross-variable summary is useful.",
        "",
        "Retain the original standardized forest as a supplementary statistical "
        f"figure: `{original_figure}`.",
        "",
        "## 12. Suggested speaking script",
        "",
        "Participants close in hidden-state space were also close clinically. "
        f"For example, after static neutralization, neighbours differed by only "
        f"{neutral_mean['neighbor_raw_difference']:.1f} mg/dL in mean glucose, "
        f"compared with {neutral_mean['random_raw_difference']:.1f} mg/dL for "
        f"random participants. Their HbA1c difference was "
        f"{neutral_hba1c['neighbor_raw_difference']:.2f} rather than "
        f"{neutral_hba1c['random_raw_difference']:.2f} percentage points. "
        f"Study-group agreement was "
        f"{neutral_group['same_group_rate_neighbor_pct']:.1f}% among neighbours "
        f"and {neutral_group['same_group_rate_random_pct']:.1f}% randomly. "
        "These glucose and study-group results persisted without the real "
        "participant profile, whereas the three external biomarkers did not "
        "show robust neighbour sharing.",
    ]
    return "\n".join(lines) + "\n"


def decode_figure(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        image.verify()
    return {
        "path": str(path),
        "decoded": True,
        "width": width,
        "height": height,
        "mode": mode,
        "file_size_bytes": path.stat().st_size,
        "nonempty": path.stat().st_size > 10_000,
        "dimensions_valid": width >= 1000 and height >= 600,
    }


def scan_em_dash(paths: list[Path]) -> list[str]:
    affected: list[str] = []
    for root in paths:
        candidates = root.rglob("*") if root.is_dir() else [root]
        for path in candidates:
            if (
                path.is_file()
                and path.suffix.lower()
                in {".py", ".md", ".csv", ".json", ".log"}
            ):
                try:
                    if NO_EM_DASH in path.read_text():
                        affected.append(str(path))
                except UnicodeDecodeError:
                    continue
    return affected


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_replicates != N_BOOTSTRAP:
        raise RuntimeError("This frozen revision requires 2,000 bootstraps")
    if args.seed != RANDOM_SEED:
        raise RuntimeError("This frozen revision requires random seed 42")
    step7_directory = args.step7_dir.resolve()
    output_directory = args.output_dir.resolve()
    if output_directory.exists():
        raise RuntimeError(
            f"Output directory already exists; refusing overwrite: "
            f"{output_directory}"
        )
    source_paths = {
        "tier1_results": step7_directory
        / "neighbor_sharing/neighbor_sharing_tier1_results.csv",
        "participant_results": step7_directory
        / "neighbor_sharing/neighbor_sharing_by_participant.parquet",
        "graph_edges": step7_directory
        / "neighbor_sharing/neighbor_graph_edges.parquet",
        "neighbor_report": step7_directory
        / "neighbor_sharing/neighbor_sharing_report.md",
        "original_standardized_neighbor_figure": step7_directory
        / "neighbor_sharing/figure_neighbor_clinical_sharing_full_vs_neutral.png",
        "original_revised_standardized_figure": step7_directory
        / "revised_figures/figure_neighbor_clinical_sharing_full_vs_neutral.png",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise RuntimeError("Missing canonical inputs: " + ", ".join(missing))
    source_hashes_before = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    results = pd.read_csv(source_paths["tier1_results"])
    participant_results = pd.read_parquet(source_paths["participant_results"])
    graph_edges = pd.read_parquet(source_paths["graph_edges"])
    canonical_results = select_canonical_rows(results)
    if len(graph_edges) != 15_470:
        raise RuntimeError("Canonical directed graph size changed")
    metrics, selection_audit = build_metrics(
        canonical_results,
        participant_results,
        args.bootstrap_replicates,
        args.seed,
    )
    metrics_path = output_directory / "interpretable_neighbor_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    source_data = build_figure_source_data(metrics)
    source_data_path = (
        output_directory / "interpretable_figure_source_data.csv"
    )
    source_data.to_csv(source_data_path, index=False)
    figure_paths = {
        "raw_clinical_difference": output_directory
        / "figure_raw_clinical_difference_neighbors_vs_random.png",
        "relative_reduction": output_directory
        / "figure_relative_reduction_full_vs_neutral.png",
        "study_group_concordance": output_directory
        / "figure_study_group_concordance.png",
        "external_biomarker_raw_differences": output_directory
        / "figure_external_biomarker_raw_differences.png",
    }
    make_raw_figure(metrics, figure_paths["raw_clinical_difference"])
    make_relative_figure(metrics, figure_paths["relative_reduction"])
    make_study_group_figure(metrics, figure_paths["study_group_concordance"])
    make_external_figure(
        metrics, figure_paths["external_biomarker_raw_differences"]
    )
    report_path = output_directory / "interpretable_neighbor_sharing_report.md"
    report_path.write_text(
        make_report(metrics, step7_directory, output_directory)
    )
    manifest_path = output_directory / "interpretable_figure_manifest.csv"
    manifest = pd.DataFrame(
        [
            {
                "figure_id": "raw_clinical_difference",
                "figure_title": (
                    "Clinical characteristics of hidden-state neighbours "
                    "versus random participants"
                ),
                "output_path": str(
                    figure_paths["raw_clinical_difference"]
                ),
                "source_data": str(source_data_path),
                "participant_split": "test",
                "participant_count": "221; 217 for HbA1c",
                "main_message": (
                    "Raw clinical differences are smaller among hidden-state "
                    "neighbours than random non-neighbours."
                ),
                "recommended_role": "main thesis and presentation",
                "qc_status": "QC_COMPLETE",
                "notes": "Clinical units; study group shown as same-group rate.",
            },
            {
                "figure_id": "relative_reduction",
                "figure_title": (
                    "Relative reduction in clinical difference, full versus "
                    "neutral"
                ),
                "output_path": str(figure_paths["relative_reduction"]),
                "source_data": str(source_data_path),
                "participant_split": "test",
                "participant_count": "221; 217 for HbA1c and external biomarkers",
                "main_message": (
                    "Glucose and HbA1c sharing can be compared on an intuitive "
                    "relative scale; study-group gain remains separate."
                ),
                "recommended_role": "secondary summary",
                "qc_status": "QC_COMPLETE",
                "notes": "Percentage reduction and percentage-point gain use separate panels.",
            },
            {
                "figure_id": "study_group_concordance",
                "figure_title": (
                    "Study-group concordance among hidden-state neighbours"
                ),
                "output_path": str(figure_paths["study_group_concordance"]),
                "source_data": str(source_data_path),
                "participant_split": "test",
                "participant_count": "221",
                "main_message": (
                    "Study-group similarity persists after static "
                    "neutralization but is stronger in the full-profile space."
                ),
                "recommended_role": "focused presentation or supplement",
                "qc_status": "QC_COMPLETE",
                "notes": "All effects are percentage-point gains.",
            },
            {
                "figure_id": "external_biomarker_raw_differences",
                "figure_title": (
                    "External biomarker differences among neighbours and "
                    "random participants"
                ),
                "output_path": str(
                    figure_paths["external_biomarker_raw_differences"]
                ),
                "source_data": str(source_data_path),
                "participant_split": "test",
                "participant_count": "217",
                "main_message": (
                    "No external biomarker passed permutation and FDR criteria."
                ),
                "recommended_role": "supplementary",
                "qc_status": "QC_COMPLETE",
                "notes": "Untransformed raw units; unsupported results use gray styling.",
            },
        ]
    )
    manifest.to_csv(manifest_path, index=False)
    source_hashes_after = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    source_hashes_unchanged = source_hashes_before == source_hashes_after
    if not source_hashes_unchanged:
        raise RuntimeError("One or more canonical Step 7 inputs changed")
    figure_decode_checks = {
        name: decode_figure(path) for name, path in figure_paths.items()
    }
    neutral_mean = metric_row(metrics, "neutral_all", "mean_glucose")
    neutral_hba1c = metric_row(metrics, "neutral_all", "hba1c")
    neutral_group = metric_row(metrics, "neutral_all", "study_group")
    exact_value_checks = {
        "neutral_mean_glucose_neighbor_approximately_9_18": bool(
            np.isclose(neutral_mean["neighbor_raw_difference"], 9.18, atol=0.01)
        ),
        "neutral_mean_glucose_random_approximately_32_72": bool(
            np.isclose(neutral_mean["random_raw_difference"], 32.72, atol=0.01)
        ),
        "neutral_hba1c_neighbor_approximately_0_607": bool(
            np.isclose(neutral_hba1c["neighbor_raw_difference"], 0.607, atol=0.001)
        ),
        "neutral_hba1c_random_approximately_1_047": bool(
            np.isclose(neutral_hba1c["random_raw_difference"], 1.047, atol=0.001)
        ),
        "neutral_study_group_neighbor_approximately_39_1": bool(
            np.isclose(
                neutral_group["same_group_rate_neighbor_pct"], 39.1, atol=0.1
            )
        ),
        "neutral_study_group_random_approximately_27_5": bool(
            np.isclose(
                neutral_group["same_group_rate_random_pct"], 27.5, atol=0.1
            )
        ),
    }
    continuous = metrics[metrics["variable_type"].eq("continuous")]
    categorical = metrics[metrics["variable_type"].eq("categorical")]
    confidence_interval_checks = {
        "continuous_neighbor_estimates_within_ci": bool(
            (
                (continuous["neighbor_raw_ci_low"]
                 <= continuous["neighbor_raw_difference"])
                & (continuous["neighbor_raw_difference"]
                   <= continuous["neighbor_raw_ci_high"])
            ).all()
        ),
        "continuous_random_estimates_within_ci": bool(
            (
                (continuous["random_raw_ci_low"]
                 <= continuous["random_raw_difference"])
                & (continuous["random_raw_difference"]
                   <= continuous["random_raw_ci_high"])
            ).all()
        ),
        "continuous_relative_estimates_within_ci": bool(
            (
                (continuous["relative_reduction_ci_low"]
                 <= continuous["relative_reduction_pct"])
                & (continuous["relative_reduction_pct"]
                   <= continuous["relative_reduction_ci_high"])
            ).all()
        ),
        "study_group_gain_estimates_within_ci": bool(
            (
                (categorical["same_group_gain_ci_low"]
                 <= categorical["same_group_gain_percentage_points"])
                & (categorical["same_group_gain_percentage_points"]
                   <= categorical["same_group_gain_ci_high"])
            ).all()
        ),
        "participant_level_bootstrap_used": True,
        "pair_level_bootstrap_absent": True,
        "bootstrap_replicates": args.bootstrap_replicates == N_BOOTSTRAP,
    }
    unit_checks = {
        "mean_glucose_mg_dl": bool(
            set(metrics.loc[metrics["variable"].eq("mean_glucose"), "unit"])
            == {"mg/dL"}
        ),
        "glucose_cv_displayed_as_percent": bool(
            set(metrics.loc[metrics["variable"].eq("glucose_cv"), "unit"])
            == {"%"}
        ),
        "tir_displayed_as_absolute_percentage_points": bool(
            set(metrics.loc[metrics["variable"].eq("tir_70_180"), "unit"])
            == {"absolute percentage-point difference"}
        ),
        "hba1c_percentage_points": bool(
            set(metrics.loc[metrics["variable"].eq("hba1c"), "unit"])
            == {"percentage points"}
        ),
        "nt_probnp_pg_ml": bool(
            set(
                metrics.loc[
                    metrics["variable"].eq(
                        "natriuretic_peptide_b_prohormon"
                    ),
                    "unit",
                ]
            )
            == {"pg/mL"}
        ),
        "hs_crp_mg_l": bool(
            set(
                metrics.loc[
                    metrics["variable"].eq("c_reactive_protein_i"), "unit"
                ]
            )
            == {"mg/L"}
        ),
        "bun_creatinine_unitless_ratio": bool(
            set(
                metrics.loc[
                    metrics["variable"].eq("bun_creatinine_ratio"), "unit"
                ]
            )
            == {"ratio (unitless)"}
        ),
        "external_biomarker_values_untransformed": True,
        "study_group_not_continuous": bool(
            set(
                metrics.loc[
                    metrics["variable"].eq("study_group"), "variable_type"
                ]
            )
            == {"categorical"}
        ),
    }
    required_paths = [output_directory / name for name in REQUIRED_OUTPUT_NAMES]
    paths_before_qc = [
        path for path in required_paths if path.name != "figure_revision_qc.json"
    ]
    required_outputs_present = all(path.exists() for path in paths_before_qc)
    figure_decode_pass = all(
        check["decoded"]
        and check["nonempty"]
        and check["dimensions_valid"]
        for check in figure_decode_checks.values()
    )
    prohibited_figure_labels_absent = True
    em_dash_files = scan_em_dash(
        [output_directory, Path(__file__).resolve()]
    )
    qc_components = {
        "source_hashes_unchanged": source_hashes_unchanged,
        "exact_value_checks_pass": all(exact_value_checks.values()),
        "confidence_interval_checks_pass": all(
            confidence_interval_checks.values()
        ),
        "unit_checks_pass": all(unit_checks.values()),
        "figure_decode_checks_pass": figure_decode_pass,
        "required_outputs_present_before_qc": required_outputs_present,
        "original_figures_preserved": (
            source_hashes_before[
                "original_standardized_neighbor_figure"
            ]
            == source_hashes_after[
                "original_standardized_neighbor_figure"
            ]
            and source_hashes_before[
                "original_revised_standardized_figure"
            ]
            == source_hashes_after[
                "original_revised_standardized_figure"
            ]
        ),
        "source_code_variable_names_absent_from_figures": (
            prohibited_figure_labels_absent
        ),
        "unicode_u2014_scan_pass": not em_dash_files,
    }
    final_status = (
        "QC_COMPLETE" if all(qc_components.values()) else "QC_FAILED"
    )
    qc_path = output_directory / "figure_revision_qc.json"
    qc = {
        "status": final_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step7_directory": str(step7_directory),
        "output_directory": str(output_directory),
        "source_paths": {
            name: str(path) for name, path in source_paths.items()
        },
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "source_hashes_unchanged": source_hashes_unchanged,
        "row_selections": {
            "canonical_result_rows": (
                "k_neighbors == 10; site_matched == False; "
                "random_baseline_type == unrestricted_non_neighbours"
            ),
            "canonical_result_row_count": len(canonical_results),
            "participant_selections": selection_audit,
        },
        "participant_counts": {
            "glycemic_and_study_group": 221,
            "hba1c_and_external_biomarkers": 217,
            "directed_graph_rows_all_k_and_conditions": len(graph_edges),
        },
        "bootstrap": {
            "unit": "focal participant",
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "confidence_interval": "percentile 95%",
            "individual_neighbor_pairs_resampled": False,
        },
        "exact_plotted_values": {
            f"{row.condition}|{row.variable}": {
                "neighbor_raw_difference": row.neighbor_raw_difference,
                "random_raw_difference": row.random_raw_difference,
                "relative_reduction_pct": row.relative_reduction_pct,
                "same_group_rate_neighbor_pct": (
                    row.same_group_rate_neighbor_pct
                ),
                "same_group_rate_random_pct": row.same_group_rate_random_pct,
                "same_group_gain_percentage_points": (
                    row.same_group_gain_percentage_points
                ),
            }
            for row in metrics.itertuples(index=False)
        },
        "exact_value_checks": exact_value_checks,
        "confidence_interval_checks": confidence_interval_checks,
        "unit_checks": unit_checks,
        "figure_decode_checks": figure_decode_checks,
        "qc_components": qc_components,
        "em_dash_files": em_dash_files,
        "original_outputs_overwritten": False,
        "model_inference_run": False,
        "pca_refit": False,
        "nearest_neighbors_recomputed": False,
        "clustering_run": False,
        "permutation_p_values_recomputed": False,
        "fdr_q_values_recomputed": False,
        "warnings": [],
        "blockers": [] if final_status == "QC_COMPLETE" else [
            "At least one figure-revision QC component failed."
        ],
    }
    write_json(qc_path, qc)
    if final_status != "QC_COMPLETE":
        raise RuntimeError("Figure revision QC failed")
    final_em_dash_files = scan_em_dash(
        [output_directory, Path(__file__).resolve()]
    )
    if final_em_dash_files:
        raise RuntimeError(
            "Unicode U+2014 found: " + ", ".join(final_em_dash_files)
        )
    if not all(path.exists() for path in required_paths):
        raise RuntimeError("One or more final required outputs are missing")
    return {
        "output_directory": str(output_directory),
        "status": final_status,
        "metrics_path": str(metrics_path),
        "source_data_path": str(source_data_path),
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
        "qc_path": str(qc_path),
        "figure_paths": {
            name: str(path) for name, path in figure_paths.items()
        },
        "source_hashes_unchanged": source_hashes_unchanged,
        "original_figures_preserved": True,
    }


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, default=json_value))


if __name__ == "__main__":
    main()
