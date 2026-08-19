#!/usr/bin/env python3
"""Run Steps A and B of the standalone T2D clinical landscape analysis."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, kurtosis, skew
import seaborn as sns


PROJECT_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/clinical_landscape"
TARGETS_PATH = (
    PROJECT_ROOT / "outputs/continuous_clinical/clinical_targets.parquet"
)
LOADINGS_PATH = PROJECT_ROOT / "outputs/continuous_clinical/pca_loadings.csv"
COVERAGE_AUDIT_PATH = PROJECT_ROOT / "subtype_partition/t2d_coverage_audit.csv"
STATIC_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "participant_static_features.parquet"
)
PRIOR_NEGATIVE_RESULT_PATH = (
    PROJECT_ROOT
    / "outputs/continuous_clinical/discrete_clustering_negative_result.md"
)

RESULTS_PATH = OUTPUT_ROOT / "density_landscape_results.csv"
DENSITY_FIGURE_PATH = OUTPUT_ROOT / "fig_density_landscape.png"
MARGINAL_FIGURE_PATH = OUTPUT_ROOT / "fig_marginal_distributions.png"
MANIFEST_PATH = OUTPUT_ROOT / "clinical_landscape_manifest.json"

PARTICIPANT_COLUMN = "participant_id"
SPLIT_COLUMN = "split"
PC1_COLUMN = "clinical_pc1"
PC2_COLUMN = "clinical_pc2"
HBA1C_COLUMN = "hba1c_percent_baseline"
ANALYSIS_SPLITS = ("validation", "test")
EXPECTED_SPLIT_COUNTS = {"validation": 91, "test": 83}
EXPECTED_POOLED_N = 174

KDE_BANDWIDTH_METHOD = "scott"
KDE_PRIMARY_MULTIPLIER = 1.0
KDE_SENSITIVITY_MULTIPLIERS = (0.8, 1.0, 1.2)
JOINT_GRID_SIZE = 250
JOINT_GRID_PADDING_SD = 0.50
JOINT_PEAK_WINDOW_GRID_POINTS = 21
PEAK_MIN_RELATIVE_DENSITY = 0.05
MARGINAL_GRID_SIZE = 500
MARGINAL_PEAK_MIN_DISTANCE_GRID_POINTS = 40
MARGINAL_PEAK_MIN_PROMINENCE_FRACTION = 0.05
HISTOGRAM_BIN_METHOD = "fd"

STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
NO_EM_DASH = "\u2014"
POOLING_JUSTIFICATION = (
    "Validation and test were pooled only for standalone descriptive "
    "characterization. No model is fitted for a downstream hidden-state "
    "comparison, so the pooled description creates no leakage risk."
)
MODE_UNCERTAINTY_CAVEAT = (
    "Mode counting from a two-dimensional KDE with n=174 is sensitive to "
    "bandwidth, grid resolution, and sparse observations. It must not be "
    "interpreted as evidence for validated discrete subtypes."
)
NONFASTING_CAVEAT = (
    "C-peptide and triglycerides were not confirmed fasting measurements. "
    "Log TG/HDL and log C-peptide require this non-fasting caveat."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_figure(fig: plt.Figure, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(
        temporary,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    os.replace(temporary, path)
    plt.close(fig)


def scott_bandwidth(kde: gaussian_kde, multiplier: float) -> float:
    return float(kde.scotts_factor() * multiplier)


def load_inputs() -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    targets = pd.read_parquet(TARGETS_PATH)
    required_targets = {
        PARTICIPANT_COLUMN,
        SPLIT_COLUMN,
        "bmi",
        "log_tg_hdl",
        "log_c_peptide",
        PC1_COLUMN,
        PC2_COLUMN,
        "participants_age",
    }
    missing_targets = sorted(required_targets - set(targets.columns))
    if missing_targets:
        raise RuntimeError(f"Missing clinical target columns: {missing_targets}")
    targets[PARTICIPANT_COLUMN] = targets[PARTICIPANT_COLUMN].astype(str)
    if targets[PARTICIPANT_COLUMN].duplicated().any():
        raise RuntimeError("Duplicate participant IDs in clinical targets")
    if targets[list(required_targets)].isna().any().any():
        raise RuntimeError("Clinical target complete-case cohort contains missing values")
    split_counts = targets[SPLIT_COLUMN].value_counts().to_dict()
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            f"Unexpected clinical target split counts: {split_counts}"
        )
    if len(targets) != EXPECTED_POOLED_N:
        raise RuntimeError(
            f"Expected pooled n={EXPECTED_POOLED_N}, observed n={len(targets)}"
        )

    coverage = pd.read_csv(COVERAGE_AUDIT_PATH)
    observed_coverage = (
        coverage[coverage["metric"] == "complete_case_four_marker"]
        .set_index("split")["count"]
        .astype(int)
        .to_dict()
    )
    if observed_coverage != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            f"Coverage audit does not match target cohort: {observed_coverage}"
        )

    static = pd.read_parquet(
        STATIC_PATH,
        columns=[PARTICIPANT_COLUMN, HBA1C_COLUMN],
    )
    static[PARTICIPANT_COLUMN] = static[PARTICIPANT_COLUMN].astype(str)
    if static[PARTICIPANT_COLUMN].duplicated().any():
        raise RuntimeError("Duplicate participant IDs in static feature table")
    pooled = targets.merge(
        static,
        on=PARTICIPANT_COLUMN,
        how="left",
        validate="one_to_one",
    )

    loadings = pd.read_csv(LOADINGS_PATH)
    component_metadata: dict[str, dict[str, object]] = {}
    for component in ("PC1", "PC2"):
        current = loadings[loadings["component"] == component]
        if current.empty:
            raise RuntimeError(f"Missing {component} rows in PCA loadings")
        component_metadata[component] = {
            "name": str(current["component_name"].iloc[0]),
            "variance_explained": float(
                current["variance_explained"].iloc[0]
            ),
            "variance_explained_percent": float(
                current["variance_explained_percent"].iloc[0]
            ),
        }
    return pooled, component_metadata


def evaluate_joint_kde(
    frame: pd.DataFrame,
    bandwidth_multiplier: float,
) -> dict[str, object]:
    raw = frame[[PC1_COLUMN, PC2_COLUMN]].to_numpy(dtype=float)
    pooled_mean = raw.mean(axis=0)
    pooled_sd = raw.std(axis=0, ddof=1)
    standardized = (raw - pooled_mean) / pooled_sd

    x_grid_z = np.linspace(
        standardized[:, 0].min() - JOINT_GRID_PADDING_SD,
        standardized[:, 0].max() + JOINT_GRID_PADDING_SD,
        JOINT_GRID_SIZE,
    )
    y_grid_z = np.linspace(
        standardized[:, 1].min() - JOINT_GRID_PADDING_SD,
        standardized[:, 1].max() + JOINT_GRID_PADDING_SD,
        JOINT_GRID_SIZE,
    )
    grid_x_z, grid_y_z = np.meshgrid(x_grid_z, y_grid_z)
    positions = np.vstack([grid_x_z.ravel(), grid_y_z.ravel()])
    kde = gaussian_kde(
        standardized.T,
        bw_method=lambda estimator: (
            estimator.scotts_factor() * bandwidth_multiplier
        ),
    )
    density = kde(positions).reshape(JOINT_GRID_SIZE, JOINT_GRID_SIZE)
    local_maximum = maximum_filter(
        density,
        size=JOINT_PEAK_WINDOW_GRID_POINTS,
        mode="nearest",
    )
    peak_mask = (
        (density == local_maximum)
        & (density >= PEAK_MIN_RELATIVE_DENSITY * density.max())
    )
    half_window = JOINT_PEAK_WINDOW_GRID_POINTS // 2
    peak_mask[:half_window, :] = False
    peak_mask[-half_window:, :] = False
    peak_mask[:, :half_window] = False
    peak_mask[:, -half_window:] = False
    peak_indices = np.argwhere(peak_mask)
    peak_densities = density[peak_indices[:, 0], peak_indices[:, 1]]
    order = np.argsort(peak_densities)[::-1]
    peaks: list[dict[str, float]] = []
    for rank, index in enumerate(order, start=1):
        row_index, column_index = peak_indices[index]
        raw_location = (
            np.array(
                [x_grid_z[column_index], y_grid_z[row_index]],
                dtype=float,
            )
            * pooled_sd
            + pooled_mean
        )
        peaks.append(
            {
                "mode_index": rank,
                "mode_pc1": float(raw_location[0]),
                "mode_pc2": float(raw_location[1]),
                "density": float(peak_densities[index]),
                "relative_density": float(
                    peak_densities[index] / density.max()
                ),
            }
        )
    return {
        "bandwidth_multiplier": bandwidth_multiplier,
        "effective_bandwidth_factor": scott_bandwidth(
            kde, bandwidth_multiplier
        ),
        "mode_count": len(peaks),
        "peaks": peaks,
        "grid_x_raw": x_grid_z * pooled_sd[0] + pooled_mean[0],
        "grid_y_raw": y_grid_z * pooled_sd[1] + pooled_mean[1],
        "density": density,
    }


def evaluate_marginal_kde(
    values: np.ndarray,
    bandwidth_multiplier: float,
) -> dict[str, object]:
    sd = float(np.std(values, ddof=1))
    grid = np.linspace(
        float(values.min() - JOINT_GRID_PADDING_SD * sd),
        float(values.max() + JOINT_GRID_PADDING_SD * sd),
        MARGINAL_GRID_SIZE,
    )
    kde = gaussian_kde(
        values,
        bw_method=lambda estimator: (
            estimator.scotts_factor() * bandwidth_multiplier
        ),
    )
    density = kde(grid)
    peak_indices, _ = find_peaks(
        density,
        distance=MARGINAL_PEAK_MIN_DISTANCE_GRID_POINTS,
        prominence=(
            MARGINAL_PEAK_MIN_PROMINENCE_FRACTION * density.max()
        ),
    )
    peak_indices = peak_indices[
        np.argsort(density[peak_indices])[::-1]
    ]
    modes = [
        {
            "mode_index": mode_index,
            "location": float(grid[peak_index]),
            "density": float(density[peak_index]),
            "relative_density": float(
                density[peak_index] / density.max()
            ),
        }
        for mode_index, peak_index in enumerate(peak_indices, start=1)
    ]
    return {
        "bandwidth_multiplier": bandwidth_multiplier,
        "effective_bandwidth_factor": scott_bandwidth(
            kde, bandwidth_multiplier
        ),
        "mode_count": len(modes),
        "modes": modes,
        "grid": grid,
        "density": density,
    }


def density_classification(
    joint_results: dict[float, dict[str, object]],
    marginal_results: dict[str, dict[float, dict[str, object]]],
) -> tuple[str, str]:
    primary_joint_count = int(
        joint_results[KDE_PRIMARY_MULTIPLIER]["mode_count"]
    )
    primary_marginal_counts = [
        int(
            marginal_results[axis][KDE_PRIMARY_MULTIPLIER][
                "mode_count"
            ]
        )
        for axis in (PC1_COLUMN, PC2_COLUMN)
    ]
    if primary_joint_count == 1 and primary_marginal_counts == [1, 1]:
        classification = "one continuous dominant mode"
    elif primary_joint_count > 1:
        classification = "multiple KDE modes"
    else:
        classification = "continuous core with a sparse tail"

    undersmoothed_count = int(joint_results[0.8]["mode_count"])
    if undersmoothed_count > primary_joint_count:
        sensitivity = (
            f"At 0.8 times Scott bandwidth, {undersmoothed_count} local "
            f"maxima were detected versus {primary_joint_count} at Scott "
            "bandwidth. The additional low-density peaks are treated as "
            "undersmoothing sensitivity, not validated subtypes."
        )
    else:
        sensitivity = (
            "The local-mode count was unchanged under the 0.8 times Scott "
            "undersmoothing sensitivity."
        )
    return classification, sensitivity


def build_results_table(
    frame: pd.DataFrame,
    joint_results: dict[float, dict[str, object]],
    marginal_results: dict[str, dict[float, dict[str, object]]],
    classification: str,
    sensitivity_note: str,
) -> pd.DataFrame:
    split_counts = frame[SPLIT_COLUMN].value_counts().to_dict()
    base = {
        "analysis_scope": "standalone pooled T2D clinical characterization",
        "n_pooled": len(frame),
        "validation_n": int(split_counts["validation"]),
        "test_n": int(split_counts["test"]),
        "hba1c_nonmissing_n": int(frame[HBA1C_COLUMN].notna().sum()),
        "hba1c_missing_n": int(frame[HBA1C_COLUMN].isna().sum()),
        "bandwidth_method": KDE_BANDWIDTH_METHOD,
        "density_classification": classification,
        "pooling_justification": POOLING_JUSTIFICATION,
        "mode_uncertainty_caveat": MODE_UNCERTAINTY_CAVEAT,
        "bandwidth_sensitivity_note": sensitivity_note,
        "hidden_state_used": False,
        "discrete_clustering_rerun": False,
    }
    rows: list[dict[str, object]] = []
    rows.append(
        {
            **base,
            "record_type": "cohort_summary",
            "axis": "joint_pc1_pc2",
            "is_primary_bandwidth": True,
        }
    )
    for multiplier, result in joint_results.items():
        rows.append(
            {
                **base,
                "record_type": "joint_summary",
                "axis": "joint_pc1_pc2",
                "bandwidth_multiplier": multiplier,
                "effective_bandwidth_factor": result[
                    "effective_bandwidth_factor"
                ],
                "mode_count": result["mode_count"],
                "is_primary_bandwidth": (
                    multiplier == KDE_PRIMARY_MULTIPLIER
                ),
            }
        )
        for peak in result["peaks"]:
            rows.append(
                {
                    **base,
                    "record_type": "joint_mode",
                    "axis": "joint_pc1_pc2",
                    "bandwidth_multiplier": multiplier,
                    "effective_bandwidth_factor": result[
                        "effective_bandwidth_factor"
                    ],
                    "mode_count": result["mode_count"],
                    "mode_index": peak["mode_index"],
                    "mode_pc1": peak["mode_pc1"],
                    "mode_pc2": peak["mode_pc2"],
                    "mode_density": peak["density"],
                    "mode_relative_density": peak["relative_density"],
                    "is_primary_bandwidth": (
                        multiplier == KDE_PRIMARY_MULTIPLIER
                    ),
                }
            )

    for axis in (PC1_COLUMN, PC2_COLUMN):
        values = frame[axis].to_numpy(dtype=float)
        quantiles = np.quantile(values, [0.25, 0.50, 0.75])
        for multiplier, result in marginal_results[axis].items():
            rows.append(
                {
                    **base,
                    "record_type": "marginal_summary",
                    "axis": axis,
                    "bandwidth_multiplier": multiplier,
                    "effective_bandwidth_factor": result[
                        "effective_bandwidth_factor"
                    ],
                    "mode_count": result["mode_count"],
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "q25": float(quantiles[0]),
                    "median": float(quantiles[1]),
                    "q75": float(quantiles[2]),
                    "skewness": float(skew(values, bias=False)),
                    "excess_kurtosis": float(
                        kurtosis(values, bias=False)
                    ),
                    "is_primary_bandwidth": (
                        multiplier == KDE_PRIMARY_MULTIPLIER
                    ),
                }
            )
            for mode in result["modes"]:
                rows.append(
                    {
                        **base,
                        "record_type": "marginal_mode",
                        "axis": axis,
                        "bandwidth_multiplier": multiplier,
                        "effective_bandwidth_factor": result[
                            "effective_bandwidth_factor"
                        ],
                        "mode_count": result["mode_count"],
                        "mode_index": mode["mode_index"],
                        "mode_axis_location": mode["location"],
                        "mode_density": mode["density"],
                        "mode_relative_density": mode[
                            "relative_density"
                        ],
                        "is_primary_bandwidth": (
                            multiplier == KDE_PRIMARY_MULTIPLIER
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_density_landscape(
    frame: pd.DataFrame,
    component_metadata: dict[str, dict[str, object]],
    primary_joint: dict[str, object],
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axis = plt.subplots(figsize=(9.4, 7.8))
    grid_x, grid_y = np.meshgrid(
        primary_joint["grid_x_raw"],
        primary_joint["grid_y_raw"],
    )
    density = primary_joint["density"]
    contour_levels = np.linspace(
        float(density.max()) * 0.10,
        float(density.max()) * 0.95,
        9,
    )
    axis.contour(
        grid_x,
        grid_y,
        density,
        levels=contour_levels,
        colors=STRATUM_COLORS[1],
        linewidths=1.15,
        alpha=0.78,
        zorder=1,
    )

    observed = frame[frame[HBA1C_COLUMN].notna()]
    missing = frame[frame[HBA1C_COLUMN].isna()]
    normalization = Normalize(
        vmin=float(observed[HBA1C_COLUMN].min()),
        vmax=float(observed[HBA1C_COLUMN].max()),
    )
    scatter = axis.scatter(
        observed[PC1_COLUMN],
        observed[PC2_COLUMN],
        c=observed[HBA1C_COLUMN],
        cmap="viridis",
        norm=normalization,
        s=38,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    if not missing.empty:
        axis.scatter(
            missing[PC1_COLUMN],
            missing[PC2_COLUMN],
            color=STRATUM_COLORS[4],
            s=55,
            alpha=0.95,
            edgecolor="white",
            linewidth=0.45,
            label=f"HbA1c unavailable (n={len(missing)})",
            zorder=4,
        )
    for peak in primary_joint["peaks"]:
        axis.scatter(
            peak["mode_pc1"],
            peak["mode_pc2"],
            marker="X",
            s=125,
            color=STRATUM_COLORS[0],
            edgecolor="white",
            linewidth=0.8,
            label=(
                "Scott KDE maximum"
                if peak["mode_index"] == 1
                else None
            ),
            zorder=5,
        )

    colorbar = fig.colorbar(scatter, ax=axis, pad=0.02)
    colorbar.set_label("HbA1c at baseline (%)")
    pc1 = component_metadata["PC1"]
    pc2 = component_metadata["PC2"]
    axis.set_xlabel(
        f"Clinical PC1: {pc1['name']} "
        f"({pc1['variance_explained_percent']:.1f}% variance)",
        fontsize=14,
    )
    axis.set_ylabel(
        "Clinical PC2: BMI/TG-HDL dissociation axis\n"
        f"({pc2['variance_explained_percent']:.1f}% variance; "
        "independent of C-peptide)",
        fontsize=14,
    )
    axis.set_title(
        "Pooled T2D clinical phenotype landscape",
        loc="left",
        fontweight="bold",
        y=1.075,
    )
    axis.text(
        0.0,
        1.010,
        (
            "Scott-rule 2D KDE with participant points colored by HbA1c"
        ),
        transform=axis.transAxes,
        fontsize=11,
        color="#444444",
        ha="left",
        va="bottom",
    )
    axis.text(
        0.0,
        -0.18,
        (
            "Validation and test pooled for description only: n=174 "
            "(91 validation, 83 test). No hidden state or downstream model."
        ),
        transform=axis.transAxes,
        fontsize=9.5,
        color="#444444",
        ha="left",
        va="top",
    )
    axis.legend(loc="best", frameon=True, fontsize=9)
    sns.despine(ax=axis)
    atomic_figure(fig, DENSITY_FIGURE_PATH)


def plot_marginals(
    frame: pd.DataFrame,
    component_metadata: dict[str, dict[str, object]],
    marginal_results: dict[str, dict[float, dict[str, object]]],
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    axis_specs = [
        (PC1_COLUMN, "PC1", STRATUM_COLORS[0]),
        (PC2_COLUMN, "PC2", STRATUM_COLORS[1]),
    ]
    for axis, (column, component, color) in zip(axes, axis_specs):
        values = frame[column].to_numpy(dtype=float)
        primary = marginal_results[column][KDE_PRIMARY_MULTIPLIER]
        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        axis.hist(
            values,
            bins=np.histogram_bin_edges(values, bins=HISTOGRAM_BIN_METHOD),
            density=True,
            color=color,
            alpha=0.22,
            edgecolor="white",
            linewidth=0.8,
            label="Pooled participants",
        )
        axis.plot(
            primary["grid"],
            primary["density"],
            color=color,
            linewidth=2.6,
            label="Scott KDE",
        )
        axis.fill_between(
            primary["grid"],
            0,
            primary["density"],
            color=color,
            alpha=0.12,
        )
        axis.axvline(
            q25,
            color=STRATUM_COLORS[3],
            linestyle="--",
            linewidth=1.8,
            label=f"Q1 = {q25:.2f}",
        )
        axis.axvline(
            q75,
            color=STRATUM_COLORS[2],
            linestyle="--",
            linewidth=1.8,
            label=f"Q3 = {q75:.2f}",
        )
        axis.axvline(
            median,
            color=STRATUM_COLORS[4],
            linestyle=":",
            linewidth=1.5,
            label=f"Median = {median:.2f}",
        )
        component_info = component_metadata[component]
        axis.set_title(
            (
                f"{component}: {component_info['name']}\n"
                f"Scott KDE local maxima: {primary['mode_count']}"
            ),
            loc="left",
            fontweight="bold",
            fontsize=14,
        )
        axis.set_xlabel(
            f"Clinical {component} score "
            f"({component_info['variance_explained_percent']:.1f}% variance)"
        )
        axis.set_ylabel("Density")
        axis.legend(frameon=True, fontsize=8.5)
        sns.despine(ax=axis)

    fig.suptitle(
        "Marginal clinical-axis distributions in pooled T2D participants",
        x=0.06,
        ha="left",
        fontweight="bold",
        fontsize=17,
    )
    fig.text(
        0.06,
        -0.015,
        (
            "n=174 pooled for standalone description. Quartile cutoffs are "
            "shown for the later extreme-group analysis, which has not been run."
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(top=0.78, bottom=0.20, wspace=0.22)
    atomic_figure(fig, MARGINAL_FIGURE_PATH)


def build_manifest(
    frame: pd.DataFrame,
    component_metadata: dict[str, dict[str, object]],
    joint_results: dict[float, dict[str, object]],
    marginal_results: dict[str, dict[float, dict[str, object]]],
    classification: str,
    sensitivity_note: str,
) -> dict[str, object]:
    primary_joint = joint_results[KDE_PRIMARY_MULTIPLIER]
    primary_modes = [
        {
            "clinical_pc1": mode["mode_pc1"],
            "clinical_pc2": mode["mode_pc2"],
            "relative_density": mode["relative_density"],
        }
        for mode in primary_joint["peaks"]
    ]
    marginal_summary = {}
    for column in (PC1_COLUMN, PC2_COLUMN):
        primary = marginal_results[column][KDE_PRIMARY_MULTIPLIER]
        marginal_summary[column] = {
            "mode_count": primary["mode_count"],
            "mode_locations": [
                mode["location"] for mode in primary["modes"]
            ],
            "q25": float(frame[column].quantile(0.25)),
            "median": float(frame[column].median()),
            "q75": float(frame[column].quantile(0.75)),
        }
    return {
        "analysis_name": "standalone_t2d_clinical_phenotype_landscape",
        "status": "AWAITING_CONFIRMATION_AFTER_STEPS_A_B",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed_steps": ["A_pooling", "B_density_landscape"],
        "paused_before_steps": [
            "C_quantile_extreme_profiles",
            "D_gmm_diagnostic",
            "E_interpretation",
        ],
        "scope": {
            "hidden_state_used": False,
            "prior_discrete_clustering_rerun": False,
            "exploratory_descriptive_only": True,
            "overrides_frozen_tier1_findings": False,
        },
        "cohort": {
            "population": "T2D complete-case validation and test participants",
            "validation_n": 91,
            "test_n": 83,
            "pooled_n": len(frame),
            "unique_participant_n": int(
                frame[PARTICIPANT_COLUMN].nunique()
            ),
            "hba1c_nonmissing_n": int(
                frame[HBA1C_COLUMN].notna().sum()
            ),
            "hba1c_missing_n": int(
                frame[HBA1C_COLUMN].isna().sum()
            ),
            "pooling_justification": POOLING_JUSTIFICATION,
        },
        "constants": {
            "KDE_BANDWIDTH_METHOD": KDE_BANDWIDTH_METHOD,
            "KDE_PRIMARY_MULTIPLIER": KDE_PRIMARY_MULTIPLIER,
            "KDE_SENSITIVITY_MULTIPLIERS": list(
                KDE_SENSITIVITY_MULTIPLIERS
            ),
            "JOINT_GRID_SIZE": JOINT_GRID_SIZE,
            "JOINT_GRID_PADDING_SD": JOINT_GRID_PADDING_SD,
            "JOINT_PEAK_WINDOW_GRID_POINTS": (
                JOINT_PEAK_WINDOW_GRID_POINTS
            ),
            "PEAK_MIN_RELATIVE_DENSITY": PEAK_MIN_RELATIVE_DENSITY,
            "MARGINAL_GRID_SIZE": MARGINAL_GRID_SIZE,
            "MARGINAL_PEAK_MIN_DISTANCE_GRID_POINTS": (
                MARGINAL_PEAK_MIN_DISTANCE_GRID_POINTS
            ),
            "MARGINAL_PEAK_MIN_PROMINENCE_FRACTION": (
                MARGINAL_PEAK_MIN_PROMINENCE_FRACTION
            ),
            "HISTOGRAM_BIN_METHOD": HISTOGRAM_BIN_METHOD,
            "STRATUM_COLORS": STRATUM_COLORS,
        },
        "component_metadata": component_metadata,
        "density_result": {
            "classification": classification,
            "primary_joint_mode_count": primary_joint["mode_count"],
            "primary_joint_mode_locations": primary_modes,
            "primary_effective_bandwidth_factor": primary_joint[
                "effective_bandwidth_factor"
            ],
            "joint_mode_count_sensitivity": {
                str(multiplier): int(result["mode_count"])
                for multiplier, result in joint_results.items()
            },
            "marginals": marginal_summary,
            "bandwidth_sensitivity_note": sensitivity_note,
            "mode_uncertainty_caveat": MODE_UNCERTAINTY_CAVEAT,
        },
        "caveats": {
            "nonfasting": NONFASTING_CAVEAT,
            "age": (
                "participants_age is age at study visit, not age at "
                "diabetes diagnosis."
            ),
            "hba1c_missing_display": (
                "The one participant without HbA1c is retained and shown "
                "in gray in the density figure."
            ),
            "subtype_claim": (
                "KDE modes and later quantile extremes are not validated "
                "clinical subtypes."
            ),
        },
        "inputs": [
            {
                "path": str(TARGETS_PATH),
                "sha256": sha256_file(TARGETS_PATH),
            },
            {
                "path": str(LOADINGS_PATH),
                "sha256": sha256_file(LOADINGS_PATH),
            },
            {
                "path": str(COVERAGE_AUDIT_PATH),
                "sha256": sha256_file(COVERAGE_AUDIT_PATH),
            },
            {
                "path": str(STATIC_PATH),
                "sha256": sha256_file(STATIC_PATH),
                "columns_read": [PARTICIPANT_COLUMN, HBA1C_COLUMN],
                "read_only": True,
            },
            {
                "path": str(PRIOR_NEGATIVE_RESULT_PATH),
                "sha256": sha256_file(PRIOR_NEGATIVE_RESULT_PATH),
                "role": "reference only; discrete clustering not rerun",
            },
        ],
        "outputs": [
            {
                "path": str(RESULTS_PATH),
                "sha256": sha256_file(RESULTS_PATH),
            },
            {
                "path": str(DENSITY_FIGURE_PATH),
                "sha256": sha256_file(DENSITY_FIGURE_PATH),
            },
            {
                "path": str(MARGINAL_FIGURE_PATH),
                "sha256": sha256_file(MARGINAL_FIGURE_PATH),
            },
            {
                "path": str(MANIFEST_PATH),
                "sha256": None,
                "note": "Self-hash intentionally omitted.",
            },
        ],
        "pending_outputs_not_created": [
            "extreme_group_profiles.csv",
            "extreme_group_overlap.csv",
            "gmm_diagnostic_results.csv",
            "fig_extreme_group_profiles.png",
            "fig_extreme_overlap_venn_or_crosstab.png",
            "fig_gmm_diagnostic.png",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
    }


def assert_no_em_dash(paths: list[Path]) -> None:
    for path in paths:
        if path.suffix not in {".csv", ".json"}:
            continue
        if NO_EM_DASH in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"Em dash found in output: {path}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame, component_metadata = load_inputs()
    joint_results = {
        multiplier: evaluate_joint_kde(frame, multiplier)
        for multiplier in KDE_SENSITIVITY_MULTIPLIERS
    }
    marginal_results = {
        axis: {
            multiplier: evaluate_marginal_kde(
                frame[axis].to_numpy(dtype=float),
                multiplier,
            )
            for multiplier in KDE_SENSITIVITY_MULTIPLIERS
        }
        for axis in (PC1_COLUMN, PC2_COLUMN)
    }
    classification, sensitivity_note = density_classification(
        joint_results,
        marginal_results,
    )
    results = build_results_table(
        frame,
        joint_results,
        marginal_results,
        classification,
        sensitivity_note,
    )
    atomic_csv(results, RESULTS_PATH)
    plot_density_landscape(
        frame,
        component_metadata,
        joint_results[KDE_PRIMARY_MULTIPLIER],
    )
    plot_marginals(frame, component_metadata, marginal_results)
    manifest = build_manifest(
        frame,
        component_metadata,
        joint_results,
        marginal_results,
        classification,
        sensitivity_note,
    )
    atomic_json(manifest, MANIFEST_PATH)
    assert_no_em_dash([RESULTS_PATH, MANIFEST_PATH])

    print(f"status={manifest['status']}")
    print(f"pooled_n={len(frame)}")
    print(f"classification={classification}")
    print(
        "primary_joint_modes="
        f"{joint_results[KDE_PRIMARY_MULTIPLIER]['mode_count']}"
    )
    print(
        "primary_joint_locations="
        f"{manifest['density_result']['primary_joint_mode_locations']}"
    )
    print(
        "marginal_mode_counts="
        f"{ {axis: marginal_results[axis][1.0]['mode_count'] for axis in (PC1_COLUMN, PC2_COLUMN)} }"
    )
    print(f"results={RESULTS_PATH}")
    print(f"density_figure={DENSITY_FIGURE_PATH}")
    print(f"marginal_figure={MARGINAL_FIGURE_PATH}")
    print(f"manifest={MANIFEST_PATH}")


if __name__ == "__main__":
    main()
