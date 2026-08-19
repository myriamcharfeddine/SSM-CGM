#!/usr/bin/env python3
"""Complete Steps C through E of the standalone T2D clinical landscape."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.mixture import GaussianMixture
import sklearn

import run_t2d_clinical_landscape as step_ab


OUTPUT_ROOT = step_ab.OUTPUT_ROOT
CANONICAL_MULTIMODAL_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "final_multimodal_dataset_20260515_184339.parquet"
)
SCRIPT_PATH = Path(__file__).resolve()
STEP_AB_SCRIPT_PATH = Path(step_ab.__file__).resolve()

EXTREME_PROFILES_PATH = OUTPUT_ROOT / "extreme_group_profiles.csv"
EXTREME_OVERLAP_PATH = OUTPUT_ROOT / "extreme_group_overlap.csv"
GMM_RESULTS_PATH = OUTPUT_ROOT / "gmm_diagnostic_results.csv"
EXTREME_FIGURE_PATH = OUTPUT_ROOT / "fig_extreme_group_profiles.png"
OVERLAP_FIGURE_PATH = (
    OUTPUT_ROOT / "fig_extreme_overlap_venn_or_crosstab.png"
)
GMM_FIGURE_PATH = OUTPUT_ROOT / "fig_gmm_diagnostic.png"
INTERPRETATION_PATH = OUTPUT_ROOT / "clinical_landscape_interpretation.md"

Q1_TOP_FRACTION = 0.25
STRICT_SENSITIVITY_FRACTION = 0.10
EXTREME_FRACTIONS = (Q1_TOP_FRACTION, STRICT_SENSITIVITY_FRACTION)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CI_LEVEL = 0.95
RANDOM_SEED = 42
GMM_COMPONENT_RANGE = (1, 2, 3, 4)
GMM_N_INIT = 50
GMM_COVARIANCE_TYPE = "full"
GMM_MAX_ITER = 2000
GMM_REG_COVAR = 1e-6
GMM_CONFIDENCE_THRESHOLD = 0.80
NO_EM_DASH = "\u2014"

PROFILE_METRICS = (
    "bmi",
    "log_tg_hdl",
    "log_c_peptide",
    "hba1c_percent_baseline",
    "mean_glucose",
    "participants_age",
)
FIGURE_METRICS = PROFILE_METRICS[:-1]
METRIC_METADATA = {
    "bmi": {"label": "BMI", "unit": "kg/m2", "decimals": 1},
    "log_tg_hdl": {
        "label": "Log TG/HDL",
        "unit": "natural log ratio",
        "decimals": 2,
    },
    "log_c_peptide": {
        "label": "Log C-peptide",
        "unit": "natural log ng/mL",
        "decimals": 2,
    },
    "hba1c_percent_baseline": {
        "label": "HbA1c",
        "unit": "%",
        "decimals": 1,
    },
    "mean_glucose": {
        "label": "Mean CGM glucose",
        "unit": "mg/dL",
        "decimals": 1,
    },
    "participants_age": {
        "label": "Age at study visit",
        "unit": "years",
        "decimals": 1,
    },
}
AXIS_METADATA = {
    "clinical_pc1": {
        "short": "PC1",
        "low_label": "PC1-low (insulin-deficiency-leaning)",
        "high_label": "PC1-high (insulin-resistance-leaning)",
    },
    "clinical_pc2": {
        "short": "PC2",
        "low_label": "PC2-low",
        "high_label": "PC2-high",
    },
}
GROUP_COLORS = {
    "PC1-low (insulin-deficiency-leaning)": step_ab.STRATUM_COLORS[2],
    "PC1-high (insulin-resistance-leaning)": step_ab.STRATUM_COLORS[0],
    "PC2-low": step_ab.STRATUM_COLORS[1],
    "PC2-high": step_ab.STRATUM_COLORS[3],
}
NONFASTING_CAVEAT = step_ab.NONFASTING_CAVEAT
AGE_CAVEAT = (
    "participants_age is age at study visit, not age at diabetes diagnosis."
)
EXTREME_CAVEAT = (
    "Extreme groups are quantile-based descriptions of this cohort and are "
    "not validated clinical subtypes."
)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return RANDOM_SEED + int.from_bytes(digest[:4], "little")


def load_complete_frame() -> tuple[pd.DataFrame, int]:
    frame, _ = step_ab.load_inputs()
    frame = frame.copy()
    frame[step_ab.PARTICIPANT_COLUMN] = frame[
        step_ab.PARTICIPANT_COLUMN
    ].astype(str)
    participant_ids = set(frame[step_ab.PARTICIPANT_COLUMN])
    cgm = pd.read_parquet(
        CANONICAL_MULTIMODAL_PATH,
        columns=[step_ab.PARTICIPANT_COLUMN, "cgm_glucose_mean"],
    )
    cgm[step_ab.PARTICIPANT_COLUMN] = cgm[
        step_ab.PARTICIPANT_COLUMN
    ].astype(str)
    cgm = cgm[cgm[step_ab.PARTICIPANT_COLUMN].isin(participant_ids)]
    valid_cgm_rows = int(cgm["cgm_glucose_mean"].notna().sum())
    participant_glucose = (
        cgm.groupby(step_ab.PARTICIPANT_COLUMN, as_index=False)[
            "cgm_glucose_mean"
        ]
        .mean()
        .rename(columns={"cgm_glucose_mean": "mean_glucose"})
    )
    frame = frame.merge(
        participant_glucose,
        on=step_ab.PARTICIPANT_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if frame["mean_glucose"].isna().any():
        missing = frame.loc[
            frame["mean_glucose"].isna(), step_ab.PARTICIPANT_COLUMN
        ].tolist()
        raise RuntimeError(f"Participants without valid CGM mean: {missing}")
    if len(frame) != step_ab.EXPECTED_POOLED_N:
        raise RuntimeError(f"Unexpected pooled cohort n={len(frame)}")
    return frame, valid_cgm_rows


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        clean.size,
        size=(BOOTSTRAP_REPLICATES, clean.size),
    )
    estimates = clean[indices].mean(axis=1)
    alpha = (1.0 - BOOTSTRAP_CI_LEVEL) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def bootstrap_difference_ci(
    high_values: np.ndarray,
    low_values: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    high = np.asarray(high_values, dtype=float)
    low = np.asarray(low_values, dtype=float)
    high = high[np.isfinite(high)]
    low = low[np.isfinite(low)]
    if high.size == 0 or low.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    high_indices = rng.integers(
        0,
        high.size,
        size=(BOOTSTRAP_REPLICATES, high.size),
    )
    low_indices = rng.integers(
        0,
        low.size,
        size=(BOOTSTRAP_REPLICATES, low.size),
    )
    estimates = high[high_indices].mean(axis=1) - low[low_indices].mean(
        axis=1
    )
    alpha = (1.0 - BOOTSTRAP_CI_LEVEL) / 2.0
    ci_low, ci_high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(ci_low), float(ci_high)


def assign_extreme_groups(
    scores: pd.Series,
    fraction: float,
) -> tuple[pd.Series, float, float]:
    low_cutoff = float(scores.quantile(fraction))
    high_cutoff = float(scores.quantile(1.0 - fraction))
    membership = pd.Series("middle", index=scores.index, dtype="object")
    membership.loc[scores <= low_cutoff] = "low"
    membership.loc[scores >= high_cutoff] = "high"
    return membership, low_cutoff, high_cutoff


def build_extreme_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fraction in EXTREME_FRACTIONS:
        for axis, axis_info in AXIS_METADATA.items():
            membership, low_cutoff, high_cutoff = assign_extreme_groups(
                frame[axis], fraction
            )
            groups = {
                "low": frame[membership == "low"],
                "high": frame[membership == "high"],
            }
            for group_direction, group_frame in groups.items():
                group_label = axis_info[f"{group_direction}_label"]
                for metric in PROFILE_METRICS:
                    values = group_frame[metric].to_numpy(dtype=float)
                    clean = values[np.isfinite(values)]
                    ci_low, ci_high = bootstrap_mean_ci(
                        clean,
                        stable_seed(
                            "mean",
                            fraction,
                            axis,
                            group_direction,
                            metric,
                        ),
                    )
                    rows.append(
                        {
                            "record_type": "group_summary",
                            "extreme_fraction": fraction,
                            "sensitivity_label": (
                                "primary_quartile"
                                if fraction == Q1_TOP_FRACTION
                                else "strict_decile_sensitivity"
                            ),
                            "axis": axis,
                            "axis_label": axis_info["short"],
                            "group_direction": group_direction,
                            "group_label": group_label,
                            "low_cutoff": low_cutoff,
                            "high_cutoff": high_cutoff,
                            "metric": metric,
                            "metric_label": METRIC_METADATA[metric]["label"],
                            "unit": METRIC_METADATA[metric]["unit"],
                            "group_n": len(group_frame),
                            "nonmissing_n": len(clean),
                            "mean": float(np.mean(clean)),
                            "sd": float(np.std(clean, ddof=1)),
                            "mean_ci_low": ci_low,
                            "mean_ci_high": ci_high,
                            "contrast_direction": None,
                            "difference_high_minus_low": None,
                            "difference_ci_low": None,
                            "difference_ci_high": None,
                            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                            "bootstrap_unit": "participant",
                            "age_definition": AGE_CAVEAT,
                            "nonfasting_caveat": NONFASTING_CAVEAT,
                            "extreme_group_caveat": EXTREME_CAVEAT,
                        }
                    )
            for metric in PROFILE_METRICS:
                high_values = groups["high"][metric].to_numpy(dtype=float)
                low_values = groups["low"][metric].to_numpy(dtype=float)
                high_clean = high_values[np.isfinite(high_values)]
                low_clean = low_values[np.isfinite(low_values)]
                difference = float(
                    np.mean(high_clean) - np.mean(low_clean)
                )
                ci_low, ci_high = bootstrap_difference_ci(
                    high_clean,
                    low_clean,
                    stable_seed("difference", fraction, axis, metric),
                )
                rows.append(
                    {
                        "record_type": "group_contrast",
                        "extreme_fraction": fraction,
                        "sensitivity_label": (
                            "primary_quartile"
                            if fraction == Q1_TOP_FRACTION
                            else "strict_decile_sensitivity"
                        ),
                        "axis": axis,
                        "axis_label": axis_info["short"],
                        "group_direction": None,
                        "group_label": None,
                        "low_cutoff": low_cutoff,
                        "high_cutoff": high_cutoff,
                        "metric": metric,
                        "metric_label": METRIC_METADATA[metric]["label"],
                        "unit": METRIC_METADATA[metric]["unit"],
                        "group_n": None,
                        "nonmissing_n": None,
                        "mean": None,
                        "sd": None,
                        "mean_ci_low": None,
                        "mean_ci_high": None,
                        "contrast_direction": "high_minus_low",
                        "difference_high_minus_low": difference,
                        "difference_ci_low": ci_low,
                        "difference_ci_high": ci_high,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "bootstrap_unit": "participant",
                        "age_definition": AGE_CAVEAT,
                        "nonfasting_caveat": NONFASTING_CAVEAT,
                        "extreme_group_caveat": EXTREME_CAVEAT,
                    }
                )
    return pd.DataFrame(rows)


def build_overlap_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered_groups = ["low", "middle", "high"]
    for fraction in EXTREME_FRACTIONS:
        pc1_group, pc1_low, pc1_high = assign_extreme_groups(
            frame["clinical_pc1"], fraction
        )
        pc2_group, pc2_low, pc2_high = assign_extreme_groups(
            frame["clinical_pc2"], fraction
        )
        table = pd.crosstab(pc1_group, pc2_group).reindex(
            index=ordered_groups,
            columns=ordered_groups,
            fill_value=0,
        )
        chi2, p_value, degrees_freedom, expected = chi2_contingency(
            table.to_numpy()
        )
        cramers_v = float(
            np.sqrt(
                chi2
                / (
                    len(frame)
                    * min(table.shape[0] - 1, table.shape[1] - 1)
                )
            )
        )
        for row_index, pc1_label in enumerate(ordered_groups):
            row_total = int(table.loc[pc1_label].sum())
            for column_index, pc2_label in enumerate(ordered_groups):
                count = int(table.loc[pc1_label, pc2_label])
                rows.append(
                    {
                        "extreme_fraction": fraction,
                        "sensitivity_label": (
                            "primary_quartile"
                            if fraction == Q1_TOP_FRACTION
                            else "strict_decile_sensitivity"
                        ),
                        "pc1_group": pc1_label,
                        "pc2_group": pc2_label,
                        "count": count,
                        "cohort_fraction": count / len(frame),
                        "row_fraction": count / row_total,
                        "expected_count_independence": float(
                            expected[row_index, column_index]
                        ),
                        "observed_minus_expected": float(
                            count - expected[row_index, column_index]
                        ),
                        "pc1_low_cutoff": pc1_low,
                        "pc1_high_cutoff": pc1_high,
                        "pc2_low_cutoff": pc2_low,
                        "pc2_high_cutoff": pc2_high,
                        "chi_square": float(chi2),
                        "chi_square_p_value": float(p_value),
                        "chi_square_degrees_freedom": int(degrees_freedom),
                        "cramers_v": cramers_v,
                        "n_pooled": len(frame),
                        "nonfasting_caveat": NONFASTING_CAVEAT,
                        "extreme_group_caveat": EXTREME_CAVEAT,
                    }
                )
    return pd.DataFrame(rows)


def fit_gmm_diagnostic(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, GaussianMixture, pd.DataFrame]:
    standardized_columns = ["z_bmi", "z_log_tg_hdl", "z_log_c_peptide"]
    matrix = frame[standardized_columns].to_numpy(dtype=float)
    models: dict[int, GaussianMixture] = {}
    model_rows: list[dict[str, object]] = []
    for components in GMM_COMPONENT_RANGE:
        model = GaussianMixture(
            n_components=components,
            covariance_type=GMM_COVARIANCE_TYPE,
            n_init=GMM_N_INIT,
            max_iter=GMM_MAX_ITER,
            reg_covar=GMM_REG_COVAR,
            random_state=RANDOM_SEED,
        )
        model.fit(matrix)
        models[components] = model
        model_rows.append(
            {
                "record_type": "model_summary",
                "n_components": components,
                "bic": float(model.bic(matrix)),
                "aic": float(model.aic(matrix)),
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
            }
        )
    selected_k = min(
        GMM_COMPONENT_RANGE,
        key=lambda components: models[components].bic(matrix),
    )
    selected = models[selected_k]
    probabilities = selected.predict_proba(matrix)
    assignments = selected.predict(matrix)
    max_posterior = probabilities.max(axis=1)
    sorted_probability = np.sort(probabilities, axis=1)
    second_posterior = (
        sorted_probability[:, -2]
        if selected_k > 1
        else np.zeros(len(frame), dtype=float)
    )
    if selected_k > 1:
        entropy = -np.sum(
            probabilities * np.log(np.clip(probabilities, 1e-15, 1.0)),
            axis=1,
        ) / np.log(selected_k)
    else:
        entropy = np.zeros(len(frame), dtype=float)
    overlap_fraction = float(
        np.mean(max_posterior < GMM_CONFIDENCE_THRESHOLD)
    )
    best_bic = float(selected.bic(matrix))
    for row in model_rows:
        row.update(
            {
                "selected_by_bic": row["n_components"] == selected_k,
                "delta_bic_from_best": float(row["bic"] - best_bic),
                "selected_k": selected_k,
                "mean_normalized_entropy": float(np.mean(entropy)),
                "median_normalized_entropy": float(np.median(entropy)),
                "p90_normalized_entropy": float(np.quantile(entropy, 0.90)),
                "mean_max_posterior": float(np.mean(max_posterior)),
                "median_max_posterior": float(np.median(max_posterior)),
                "minimum_max_posterior": float(np.min(max_posterior)),
                "component_overlap_threshold": GMM_CONFIDENCE_THRESHOLD,
                "component_overlap_fraction": overlap_fraction,
                "n_pooled": len(frame),
                "nonfasting_caveat": NONFASTING_CAVEAT,
                "diagnostic_caveat": EXTREME_CAVEAT,
            }
        )

    rows = list(model_rows)
    assignments_frame = frame[
        [
            step_ab.PARTICIPANT_COLUMN,
            step_ab.PC1_COLUMN,
            step_ab.PC2_COLUMN,
            "bmi",
            "log_tg_hdl",
            "log_c_peptide",
        ]
    ].copy()
    assignments_frame["assigned_component"] = assignments
    assignments_frame["max_posterior"] = max_posterior
    assignments_frame["second_posterior"] = second_posterior
    assignments_frame["normalized_entropy"] = entropy

    for component in range(selected_k):
        component_mask = assignments == component
        component_frame = frame.loc[component_mask]
        component_max = max_posterior[component_mask]
        component_entropy = entropy[component_mask]
        rows.append(
            {
                "record_type": "component_summary",
                "n_components": selected_k,
                "selected_by_bic": True,
                "selected_k": selected_k,
                "assigned_component": component,
                "component_n": int(component_mask.sum()),
                "component_fraction": float(component_mask.mean()),
                "mean_max_posterior": float(np.mean(component_max)),
                "median_max_posterior": float(np.median(component_max)),
                "mean_normalized_entropy": float(
                    np.mean(component_entropy)
                ),
                "component_mean_z_bmi": float(
                    component_frame["z_bmi"].mean()
                ),
                "component_mean_z_log_tg_hdl": float(
                    component_frame["z_log_tg_hdl"].mean()
                ),
                "component_mean_z_log_c_peptide": float(
                    component_frame["z_log_c_peptide"].mean()
                ),
                "component_mean_bmi": float(component_frame["bmi"].mean()),
                "component_mean_log_tg_hdl": float(
                    component_frame["log_tg_hdl"].mean()
                ),
                "component_mean_log_c_peptide": float(
                    component_frame["log_c_peptide"].mean()
                ),
                "component_overlap_threshold": GMM_CONFIDENCE_THRESHOLD,
                "component_overlap_fraction": overlap_fraction,
                "n_pooled": len(frame),
                "nonfasting_caveat": NONFASTING_CAVEAT,
                "diagnostic_caveat": EXTREME_CAVEAT,
            }
        )

    for row in assignments_frame.itertuples(index=False):
        rows.append(
            {
                "record_type": "participant_assignment",
                "n_components": selected_k,
                "selected_by_bic": True,
                "selected_k": selected_k,
                step_ab.PARTICIPANT_COLUMN: getattr(
                    row, step_ab.PARTICIPANT_COLUMN
                ),
                "assigned_component": int(row.assigned_component),
                "clinical_pc1": float(row.clinical_pc1),
                "clinical_pc2": float(row.clinical_pc2),
                "bmi": float(row.bmi),
                "log_tg_hdl": float(row.log_tg_hdl),
                "log_c_peptide": float(row.log_c_peptide),
                "max_posterior": float(row.max_posterior),
                "second_posterior": float(row.second_posterior),
                "normalized_entropy": float(row.normalized_entropy),
                "component_overlap_threshold": GMM_CONFIDENCE_THRESHOLD,
                "component_overlap_fraction": overlap_fraction,
                "n_pooled": len(frame),
                "nonfasting_caveat": NONFASTING_CAVEAT,
                "diagnostic_caveat": EXTREME_CAVEAT,
            }
        )
    return pd.DataFrame(rows), selected, assignments_frame


def plot_extreme_profiles(profiles: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    primary = profiles[
        (profiles["record_type"] == "group_summary")
        & (profiles["extreme_fraction"] == Q1_TOP_FRACTION)
        & (profiles["metric"].isin(FIGURE_METRICS))
    ].copy()
    group_order = [
        "PC1-low (insulin-deficiency-leaning)",
        "PC1-high (insulin-resistance-leaning)",
        "PC2-low",
        "PC2-high",
    ]
    short_labels = {
        group_order[0]: "PC1 low",
        group_order[1]: "PC1 high",
        group_order[2]: "PC2 low",
        group_order[3]: "PC2 high",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    axes_flat = axes.ravel()
    for axis, metric in zip(axes_flat, FIGURE_METRICS):
        current = primary[primary["metric"] == metric].set_index(
            "group_label"
        ).loc[group_order]
        x_positions = np.arange(len(group_order))
        values = current["mean"].to_numpy(dtype=float)
        lower = values - current["mean_ci_low"].to_numpy(dtype=float)
        upper = current["mean_ci_high"].to_numpy(dtype=float) - values
        bars = axis.bar(
            x_positions,
            values,
            color=[GROUP_COLORS[group] for group in group_order],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
        )
        axis.errorbar(
            x_positions,
            values,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#222222",
            capsize=3,
            linewidth=1.2,
            zorder=4,
        )
        decimals = int(METRIC_METADATA[metric]["decimals"])
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.{decimals}f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        axis.set_xticks(
            x_positions,
            [short_labels[group] for group in group_order],
            rotation=24,
            ha="right",
            fontsize=9,
        )
        axis.set_title(
            METRIC_METADATA[metric]["label"],
            loc="left",
            fontweight="bold",
            fontsize=13,
        )
        axis.set_ylabel(METRIC_METADATA[metric]["unit"], fontsize=10)
        sns.despine(ax=axis)

    note_axis = axes_flat[-1]
    note_axis.axis("off")
    note_axis.text(
        0.02,
        0.92,
        "Primary quartile extremes",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    note_axis.text(
        0.02,
        0.77,
        (
            "Bars show group means with 95% participant-bootstrap CIs.\n"
            "Each panel retains its native unit.\n\n"
            "PC1 high: higher BMI, TG/HDL, and C-peptide.\n"
            "PC2 high: lower BMI and higher TG/HDL, with little\n"
            "C-peptide separation by construction."
        ),
        ha="left",
        va="top",
        fontsize=10.5,
        linespacing=1.35,
    )
    fig.suptitle(
        "Clinical profiles of pooled T2D axis extremes",
        x=0.05,
        ha="left",
        fontweight="bold",
        fontsize=18,
    )
    fig.text(
        0.05,
        0.015,
        (
            "Quantile groups are descriptive, not validated subtypes. "
            "C-peptide and triglycerides were not confirmed fasting."
        ),
        ha="left",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(top=0.90, bottom=0.12, hspace=0.48, wspace=0.28)
    step_ab.atomic_figure(fig, EXTREME_FIGURE_PATH)


def plot_overlap(overlap: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.9))
    group_order = ["low", "middle", "high"]
    for axis, fraction in zip(axes, EXTREME_FRACTIONS):
        current = overlap[overlap["extreme_fraction"] == fraction]
        table = current.pivot(
            index="pc1_group",
            columns="pc2_group",
            values="count",
        ).reindex(index=group_order, columns=group_order)
        sns.heatmap(
            table,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=False,
            linewidths=1,
            linecolor="white",
            square=True,
            ax=axis,
            annot_kws={"fontsize": 12, "fontweight": "bold"},
        )
        label = "Quartile extremes" if fraction == 0.25 else "Decile extremes"
        cramers_v = float(current["cramers_v"].iloc[0])
        p_value = float(current["chi_square_p_value"].iloc[0])
        axis.set_title(
            f"{label}\nCramer's V = {cramers_v:.3f}, p = {p_value:.3f}",
            loc="left",
            fontweight="bold",
            fontsize=13,
        )
        axis.set_xlabel("PC2 membership", fontsize=11)
        axis.set_ylabel("PC1 membership", fontsize=11)
        axis.set_xticklabels(["Low", "Middle", "High"], rotation=0)
        axis.set_yticklabels(["Low", "Middle", "High"], rotation=0)
    fig.suptitle(
        "Overlap of PC1 and PC2 extreme-group membership",
        x=0.05,
        ha="left",
        fontweight="bold",
        fontsize=18,
    )
    fig.text(
        0.05,
        0.015,
        (
            "Cell labels are participant counts. Quantile groups are "
            "descriptive, not validated subtypes. C-peptide and "
            "triglycerides were not confirmed fasting."
        ),
        ha="left",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(top=0.82, bottom=0.15, wspace=0.28)
    step_ab.atomic_figure(fig, OVERLAP_FIGURE_PATH)


def plot_gmm_diagnostic(
    gmm_results: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    model_rows = gmm_results[gmm_results["record_type"] == "model_summary"]
    selected_k = int(model_rows["selected_k"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    bic_axis, scatter_axis = axes
    bic_axis.plot(
        model_rows["n_components"],
        model_rows["bic"],
        marker="o",
        color=step_ab.STRATUM_COLORS[1],
        linewidth=2.4,
        markersize=8,
    )
    selected_row = model_rows[model_rows["n_components"] == selected_k].iloc[0]
    bic_axis.scatter(
        [selected_k],
        [selected_row["bic"]],
        marker="X",
        s=150,
        color=step_ab.STRATUM_COLORS[0],
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        label=f"BIC-selected k = {selected_k}",
    )
    bic_axis.set_xticks(list(GMM_COMPONENT_RANGE))
    bic_axis.set_xlabel("Number of Gaussian components")
    bic_axis.set_ylabel("BIC, lower is better")
    bic_axis.set_title("Component-count diagnostic", loc="left", fontweight="bold")
    bic_axis.legend(frameon=True, fontsize=10)
    sns.despine(ax=bic_axis)

    if selected_k > 1:
        points = scatter_axis.scatter(
            assignments[step_ab.PC1_COLUMN],
            assignments[step_ab.PC2_COLUMN],
            c=assignments["normalized_entropy"],
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            s=43,
            alpha=0.86,
            edgecolor="white",
            linewidth=0.35,
        )
        colorbar = fig.colorbar(points, ax=scatter_axis, pad=0.02)
        colorbar.set_label("Normalized assignment entropy")
        component_counts = (
            assignments["assigned_component"].value_counts().sort_index()
        )
        scatter_axis.text(
            0.02,
            0.98,
            "Assigned component sizes: "
            + ", ".join(
                f"C{int(component)}={int(count)}"
                for component, count in component_counts.items()
            ),
            transform=scatter_axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "alpha": 0.9,
                "boxstyle": "round,pad=0.3",
            },
        )
        scatter_axis.set_xlabel("Clinical PC1: insulin-resistance axis")
        scatter_axis.set_ylabel("Clinical PC2: BMI/TG-HDL dissociation axis")
        scatter_axis.set_title(
            "Soft-assignment uncertainty",
            loc="left",
            fontweight="bold",
        )
        sns.despine(ax=scatter_axis)
    else:
        scatter_axis.axis("off")
        scatter_axis.text(
            0.5,
            0.55,
            "BIC favors one Gaussian component.\nNo entropy scatter is needed.",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
        )
    fig.suptitle(
        "Gaussian-mixture diagnostic for pooled T2D clinical markers",
        x=0.05,
        ha="left",
        fontweight="bold",
        fontsize=18,
    )
    fig.text(
        0.05,
        0.012,
        (
            "Diagnostic only. Components are not validated subtypes. "
            "C-peptide and triglycerides were not confirmed fasting."
        ),
        ha="left",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(top=0.84, bottom=0.15, wspace=0.30)
    step_ab.atomic_figure(fig, GMM_FIGURE_PATH)


def extract_group_value(
    profiles: pd.DataFrame,
    group_label: str,
    metric: str,
) -> float:
    row = profiles[
        (profiles["record_type"] == "group_summary")
        & (profiles["extreme_fraction"] == Q1_TOP_FRACTION)
        & (profiles["group_label"] == group_label)
        & (profiles["metric"] == metric)
    ]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected one profile row for {group_label}, {metric}"
        )
    return float(row["mean"].iloc[0])


def write_interpretation(
    profiles: pd.DataFrame,
    overlap: pd.DataFrame,
    gmm_results: pd.DataFrame,
) -> str:
    model_rows = gmm_results[gmm_results["record_type"] == "model_summary"]
    selected_k = int(model_rows["selected_k"].iloc[0])
    entropy = float(model_rows["mean_normalized_entropy"].iloc[0])
    overlap_fraction = float(model_rows["component_overlap_fraction"].iloc[0])
    components = gmm_results[
        gmm_results["record_type"] == "component_summary"
    ]
    smallest_component_n = int(components["component_n"].min())
    quartile_overlap = overlap[
        overlap["extreme_fraction"] == Q1_TOP_FRACTION
    ]
    cramers_v = float(quartile_overlap["cramers_v"].iloc[0])

    pc1_high_bmi = extract_group_value(
        profiles, AXIS_METADATA["clinical_pc1"]["high_label"], "bmi"
    )
    pc1_high_tg = extract_group_value(
        profiles,
        AXIS_METADATA["clinical_pc1"]["high_label"],
        "log_tg_hdl",
    )
    pc1_high_cp = extract_group_value(
        profiles,
        AXIS_METADATA["clinical_pc1"]["high_label"],
        "log_c_peptide",
    )
    pc1_low_bmi = extract_group_value(
        profiles, AXIS_METADATA["clinical_pc1"]["low_label"], "bmi"
    )
    pc1_low_cp = extract_group_value(
        profiles,
        AXIS_METADATA["clinical_pc1"]["low_label"],
        "log_c_peptide",
    )
    pc2_high_bmi = extract_group_value(
        profiles, AXIS_METADATA["clinical_pc2"]["high_label"], "bmi"
    )
    pc2_high_tg = extract_group_value(
        profiles,
        AXIS_METADATA["clinical_pc2"]["high_label"],
        "log_tg_hdl",
    )
    pc2_low_bmi = extract_group_value(
        profiles, AXIS_METADATA["clinical_pc2"]["low_label"], "bmi"
    )
    pc2_low_tg = extract_group_value(
        profiles,
        AXIS_METADATA["clinical_pc2"]["low_label"],
        "log_tg_hdl",
    )

    paragraph = (
        "The pooled 174-participant T2D cohort forms one dominant continuous "
        "mode on Clinical PC1 and PC2, with unimodal marginals and only "
        "low-density peak sensitivity under deliberate KDE undersmoothing. "
        f"The PC1-high quartile has higher BMI ({pc1_high_bmi:.1f}), log "
        f"TG/HDL ({pc1_high_tg:.2f}), and log C-peptide ({pc1_high_cp:.2f}), "
        "consistent with a compensated insulin-resistance-leaning profile, "
        f"whereas PC1-low is leaner (BMI {pc1_low_bmi:.1f}) with lower log "
        f"C-peptide ({pc1_low_cp:.2f}). PC2-high combines lower BMI "
        f"({pc2_high_bmi:.1f}) with higher log TG/HDL ({pc2_high_tg:.2f}), "
        f"while PC2-low combines higher BMI ({pc2_low_bmi:.1f}) with lower "
        f"log TG/HDL ({pc2_low_tg:.2f}); PC1 and PC2 quartile membership is "
        f"only weakly associated (Cramer's V {cramers_v:.3f}). The GMM check "
        f"complicates, rather than independently confirms, the continuous "
        f"picture: BIC selected {selected_k} components, but the smallest "
        f"contained only {smallest_component_n} participants, mean normalized "
        f"entropy was {entropy:.3f}, and {overlap_fraction * 100:.1f}% had "
        "maximum posterior below 0.80, indicating a sparse component and "
        "substantial assignment overlap. These extreme groups and mixture "
        "components are not validated clinical subtypes; they are exploratory "
        "descriptions of this cohort along data-driven axes. C-peptide and "
        "triglycerides were not confirmed fasting, and age is age at the study "
        "visit rather than age at diabetes diagnosis."
    )
    text = "# Standalone T2D clinical landscape interpretation\n\n" + paragraph + "\n"
    INTERPRETATION_PATH.write_text(text, encoding="utf-8")
    return paragraph


def finalize_manifest(
    frame: pd.DataFrame,
    valid_cgm_rows: int,
    profiles: pd.DataFrame,
    overlap: pd.DataFrame,
    gmm_results: pd.DataFrame,
    interpretation: str,
) -> None:
    manifest = json.loads(step_ab.MANIFEST_PATH.read_text(encoding="utf-8"))
    model_rows = gmm_results[gmm_results["record_type"] == "model_summary"]
    selected_k = int(model_rows["selected_k"].iloc[0])
    selected_components = gmm_results[
        gmm_results["record_type"] == "component_summary"
    ]
    quartile_overlap = overlap[
        overlap["extreme_fraction"] == Q1_TOP_FRACTION
    ]
    manifest.update(
        {
            "status": "QC_COMPLETE",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed_steps": [
                "A_pooling",
                "B_density_landscape",
                "C_quantile_extreme_profiles",
                "D_gmm_diagnostic",
                "E_bounded_interpretation",
            ],
            "paused_before_steps": [],
            "pending_outputs_not_created": [],
            "interpretation": interpretation,
            "extreme_group_result": {
                "primary_fraction": Q1_TOP_FRACTION,
                "sensitivity_fraction": STRICT_SENSITIVITY_FRACTION,
                "quartile_group_n_per_tail": 44,
                "decile_group_n_per_tail": 18,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "age_definition": AGE_CAVEAT,
                "nonfasting_caveat": NONFASTING_CAVEAT,
                "subtype_caveat": EXTREME_CAVEAT,
            },
            "overlap_result": {
                "quartile_cramers_v": float(
                    quartile_overlap["cramers_v"].iloc[0]
                ),
                "quartile_chi_square_p_value": float(
                    quartile_overlap["chi_square_p_value"].iloc[0]
                ),
                "interpretation": (
                    "PC1 and PC2 extreme memberships are largely independent "
                    "populations, with only weak quartile association."
                ),
            },
            "gmm_result": {
                "component_range": list(GMM_COMPONENT_RANGE),
                "selected_k_by_bic": selected_k,
                "bic_by_k": {
                    str(int(row.n_components)): float(row.bic)
                    for row in model_rows.itertuples(index=False)
                },
                "delta_bic_from_best_by_k": {
                    str(int(row.n_components)): float(
                        row.delta_bic_from_best
                    )
                    for row in model_rows.itertuples(index=False)
                },
                "component_sizes": {
                    str(int(row.assigned_component)): int(row.component_n)
                    for row in selected_components.itertuples(index=False)
                },
                "mean_normalized_entropy": float(
                    model_rows["mean_normalized_entropy"].iloc[0]
                ),
                "median_normalized_entropy": float(
                    model_rows["median_normalized_entropy"].iloc[0]
                ),
                "overlap_definition": (
                    "fraction with maximum posterior below 0.80"
                ),
                "component_overlap_fraction": float(
                    model_rows["component_overlap_fraction"].iloc[0]
                ),
                "interpretation": (
                    "BIC-selected mixture structure is diagnostic only and "
                    "is weakened by a tiny component and assignment overlap."
                ),
            },
            "mean_glucose_source": {
                "path": str(CANONICAL_MULTIMODAL_PATH),
                "participant_column": step_ab.PARTICIPANT_COLUMN,
                "glucose_column": "cgm_glucose_mean",
                "aggregation": "mean across non-missing participant CGM rows",
                "valid_cgm_rows": valid_cgm_rows,
                "participants_with_valid_mean": int(
                    frame["mean_glucose"].notna().sum()
                ),
                "read_only": True,
                "sha256": step_ab.sha256_file(CANONICAL_MULTIMODAL_PATH),
            },
            "implementation_scripts": [
                {
                    "path": str(STEP_AB_SCRIPT_PATH),
                    "sha256": step_ab.sha256_file(STEP_AB_SCRIPT_PATH),
                    "steps": ["A", "B"],
                },
                {
                    "path": str(SCRIPT_PATH),
                    "sha256": step_ab.sha256_file(SCRIPT_PATH),
                    "steps": ["C", "D", "E"],
                },
            ],
        }
    )
    manifest["constants"].update(
        {
            "Q1_TOP_FRACTION": Q1_TOP_FRACTION,
            "STRICT_SENSITIVITY_FRACTION": STRICT_SENSITIVITY_FRACTION,
            "BOOTSTRAP_REPLICATES": BOOTSTRAP_REPLICATES,
            "BOOTSTRAP_CI_LEVEL": BOOTSTRAP_CI_LEVEL,
            "RANDOM_SEED": RANDOM_SEED,
            "GMM_COMPONENT_RANGE": list(GMM_COMPONENT_RANGE),
            "GMM_N_INIT": GMM_N_INIT,
            "GMM_COVARIANCE_TYPE": GMM_COVARIANCE_TYPE,
            "GMM_MAX_ITER": GMM_MAX_ITER,
            "GMM_REG_COVAR": GMM_REG_COVAR,
            "GMM_CONFIDENCE_THRESHOLD": GMM_CONFIDENCE_THRESHOLD,
        }
    )
    manifest["scope"].update(
        {
            "hidden_state_used": False,
            "prior_discrete_clustering_rerun": False,
            "exploratory_descriptive_only": True,
        }
    )
    manifest["software"].update(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
            "scikit_learn": sklearn.__version__,
        }
    )
    output_paths = [
        step_ab.RESULTS_PATH,
        EXTREME_PROFILES_PATH,
        EXTREME_OVERLAP_PATH,
        GMM_RESULTS_PATH,
        step_ab.DENSITY_FIGURE_PATH,
        step_ab.MARGINAL_FIGURE_PATH,
        EXTREME_FIGURE_PATH,
        OVERLAP_FIGURE_PATH,
        GMM_FIGURE_PATH,
        INTERPRETATION_PATH,
    ]
    manifest["outputs"] = [
        {"path": str(path), "sha256": step_ab.sha256_file(path)}
        for path in output_paths
    ] + [
        {
            "path": str(step_ab.MANIFEST_PATH),
            "sha256": None,
            "note": "Self-hash intentionally omitted.",
        }
    ]
    step_ab.atomic_json(manifest, step_ab.MANIFEST_PATH)


def assert_no_em_dash(paths: list[Path]) -> None:
    for path in paths:
        if NO_EM_DASH in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"Em dash found in output: {path}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame, valid_cgm_rows = load_complete_frame()
    profiles = build_extreme_profiles(frame)
    overlap = build_overlap_table(frame)
    gmm_results, selected_gmm, assignments = fit_gmm_diagnostic(frame)

    step_ab.atomic_csv(profiles, EXTREME_PROFILES_PATH)
    step_ab.atomic_csv(overlap, EXTREME_OVERLAP_PATH)
    step_ab.atomic_csv(gmm_results, GMM_RESULTS_PATH)
    plot_extreme_profiles(profiles)
    plot_overlap(overlap)
    plot_gmm_diagnostic(gmm_results, assignments)
    interpretation = write_interpretation(profiles, overlap, gmm_results)
    finalize_manifest(
        frame,
        valid_cgm_rows,
        profiles,
        overlap,
        gmm_results,
        interpretation,
    )
    assert_no_em_dash(
        [
            EXTREME_PROFILES_PATH,
            EXTREME_OVERLAP_PATH,
            GMM_RESULTS_PATH,
            INTERPRETATION_PATH,
            step_ab.MANIFEST_PATH,
        ]
    )

    model_rows = gmm_results[gmm_results["record_type"] == "model_summary"]
    component_rows = gmm_results[
        gmm_results["record_type"] == "component_summary"
    ]
    print("status=QC_COMPLETE")
    print(f"pooled_n={len(frame)}")
    print(f"valid_cgm_rows={valid_cgm_rows}")
    print(f"selected_gmm_k={selected_gmm.n_components}")
    print(
        "gmm_component_sizes="
        f"{component_rows['component_n'].astype(int).tolist()}"
    )
    print(
        "gmm_mean_entropy="
        f"{float(model_rows['mean_normalized_entropy'].iloc[0]):.6f}"
    )
    print(
        "gmm_overlap_fraction="
        f"{float(model_rows['component_overlap_fraction'].iloc[0]):.6f}"
    )
    print(f"extreme_profiles={EXTREME_PROFILES_PATH}")
    print(f"extreme_overlap={EXTREME_OVERLAP_PATH}")
    print(f"gmm_results={GMM_RESULTS_PATH}")
    print(f"interpretation={INTERPRETATION_PATH}")
    print(f"manifest={step_ab.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
