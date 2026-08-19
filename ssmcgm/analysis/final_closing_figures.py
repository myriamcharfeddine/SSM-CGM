"""Step 7 final figure set using frozen and already exported artifacts."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


HOUSE_NAVY = "#003366"
HOUSE_CRIMSON = "#BA2828"
HOUSE_TEAL = "#5BBABA"
HOUSE_KEY_RED = "#FF0000"
HOUSE_GRAY = "#888888"
PLOT_BLACK = "#111111"
PLOT_WHITE = "#FFFFFF"
STRATUM_COLORS = {
    "Healthy": HOUSE_NAVY,
    "Prediabetes": HOUSE_TEAL,
    "Oral medication": HOUSE_CRIMSON,
    "Insulin": HOUSE_KEY_RED,
    "Unknown": HOUSE_GRAY,
}
STUDY_GROUP_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes_lifestyle_controlled": "Prediabetes",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled":
        "Oral medication",
    "insulin_dependent": "Insulin",
}
STUDY_GROUP_ORDER = (
    "Healthy",
    "Prediabetes",
    "Oral medication",
    "Insulin",
)
HIDDEN_COLUMNS = tuple(f"r_{index:03d}" for index in range(128))
POINT_SIZE = 28
POINT_ALPHA = 0.80
RANDOM_SEED = 42
NO_EM_DASH = "\u2014"
PROBE_LABELS = {
    "c_reactive_protein_i": "High-sensitivity CRP",
    "natriuretic_peptide_b_prohormon": "NT-proBNP",
    "bun_creatinine_ratio": "BUN/creatinine ratio",
}


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def frozen_projection(
    representations: pd.DataFrame,
    participant_ids: list[str],
    step3_directory: Path,
    representation_space: str,
    projection_space: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    projection_space = projection_space or representation_space
    selected = representations.loc[
        representations["representation_type"].eq(representation_space),
        ["participant_id", *HIDDEN_COLUMNS],
    ].copy()
    selected["participant_id"] = selected["participant_id"].astype(str)
    if selected.duplicated("participant_id").any():
        raise RuntimeError(
            f"Duplicate representation row for {representation_space}"
        )
    selected = selected.set_index("participant_id").reindex(participant_ids)
    values = selected.loc[:, HIDDEN_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(
            f"Nonfinite representation for {representation_space}"
        )
    frozen = (
        step3_directory / "frozen_validation_pipeline" / projection_space
    )
    feature_order = json.loads((frozen / "feature_order.json").read_text())
    if feature_order["source_dimensions"] != list(HIDDEN_COLUMNS):
        raise RuntimeError(
            f"Frozen feature order mismatch for {projection_space}"
        )
    kept = np.load(frozen / "kept_dimensions.npy")
    scaler = joblib.load(frozen / f"{projection_space}_scaler.joblib")
    pca = joblib.load(frozen / f"{projection_space}_pca.joblib")
    scores = pca.transform(scaler.transform(values[:, kept]))[:, :2]
    variance = np.asarray(pca.explained_variance_ratio_[:2], dtype=float)
    return scores, variance


def annotate_bars(
    axis: plt.Axes,
    digits: int,
    suffix: str = "",
) -> None:
    for patch in axis.patches:
        height = patch.get_height()
        axis.annotate(
            f"{height:.{digits}f}{suffix}",
            (patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def make_static_figure(
    output_path: Path,
    source_data_path: Path,
    step6_directory: Path,
) -> None:
    effects = pd.read_csv(
        step6_directory
        / "final_tables/table2_static_neutralization_effects.csv"
    )
    reliability = pd.read_csv(
        step6_directory
        / "final_tables/table3_representation_reliability.csv"
    )
    state_l2 = effects.loc[
        effects["metric"].eq("median full-neutral state L2"), "estimate"
    ].iloc[0]
    state_cosine = effects.loc[
        effects["metric"].eq("median full-neutral cosine"), "estimate"
    ].iloc[0]
    mean_forecast = effects.loc[
        effects["metric"].eq("mean absolute forecast difference"),
        "estimate",
    ].iloc[0]
    terminal_forecast = effects.loc[
        effects["metric"].eq("terminal forecast difference"), "estimate"
    ].iloc[0]
    source = pd.concat(
        [
            effects.assign(source_table="static_effects"),
            reliability.rename(columns={"split": "metric"}).assign(
                source_table="reliability"
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    source.to_csv(source_data_path, index=False)
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.5),
        gridspec_kw={"width_ratios": [0.75, 1.1, 2.2]},
    )
    axes[0].bar(
        ["Median\nfull-neutral L2"],
        [state_l2],
        color=HOUSE_NAVY,
        width=0.55,
    )
    axes[0].set_ylabel("State-space L2 distance")
    axes[0].set_title("A. Hidden-state effect")
    axes[0].text(
        0.5,
        0.91,
        f"Median cosine = {state_cosine:.3f}",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=9,
    )
    annotate_bars(axes[0], digits=2)
    axes[1].bar(
        [
            "Mean absolute\nforecast difference",
            "Terminal\nforecast difference",
        ],
        [mean_forecast, terminal_forecast],
        color=[HOUSE_NAVY, HOUSE_CRIMSON],
    )
    axes[1].set_ylabel("Forecast difference (mg/dL)")
    axes[1].set_title("B. Forecast effect")
    annotate_bars(axes[1], digits=2)
    metric_columns = (
        "odd_even_cosine",
        "top1",
        "top5",
        "median_icc",
    )
    metric_labels = (
        "Odd/even\ncosine",
        "Top-1\nretrieval",
        "Top-5\nretrieval",
        "Median\nICC",
    )
    validation = reliability.set_index("split").loc["validation"]
    test = reliability.set_index("split").loc["test"]
    positions = np.arange(len(metric_columns))
    axes[2].bar(
        positions - 0.18,
        [validation[column] for column in metric_columns],
        0.36,
        label="Validation (n=239)",
        color=HOUSE_NAVY,
    )
    axes[2].bar(
        positions + 0.18,
        [test[column] for column in metric_columns],
        0.36,
        label="Test (n=221)",
        color=HOUSE_CRIMSON,
    )
    axes[2].set_xticks(positions, metric_labels)
    axes[2].set_ylim(0, 1.08)
    axes[2].set_ylabel("Proportion or coefficient")
    axes[2].set_title("C. Static-neutral reliability")
    axes[2].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
    )
    annotate_bars(axes[2], digits=3)
    figure.suptitle(
        "Static conditioning changes states and forecasts; neutral representations remain reliable",
        fontsize=14,
        weight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def make_manifold_overlays(
    output_path: Path,
    source_data_path: Path,
    mapping_path: Path,
    representations: pd.DataFrame,
    features: pd.DataFrame,
    step3_directory: Path,
) -> None:
    features = features.copy()
    features["participant_id"] = features["participant_id"].astype(str)
    participant_ids = features["participant_id"].tolist()
    full_scores, full_variance = frozen_projection(
        representations,
        participant_ids,
        step3_directory,
        "full_all",
    )
    neutral_scores, neutral_variance = frozen_projection(
        representations,
        participant_ids,
        step3_directory,
        "neutral_all",
    )
    source_levels = sorted(
        features["study_group"].dropna().astype(str).unique().tolist()
    )
    unknown = sorted(set(source_levels) - set(STUDY_GROUP_LABELS))
    if unknown:
        raise RuntimeError(f"Unknown study-group levels: {unknown}")
    mapping = pd.DataFrame(
        [
            {
                "source_level": source_level,
                "display_label": STUDY_GROUP_LABELS[source_level],
                "display_order": STUDY_GROUP_ORDER.index(
                    STUDY_GROUP_LABELS[source_level]
                )
                + 1,
                "color": STRATUM_COLORS[
                    STUDY_GROUP_LABELS[source_level]
                ],
            }
            for source_level in source_levels
        ]
    )
    mapping.to_csv(mapping_path, index=False)
    source = features[
        [
            "participant_id",
            "mean_glucose",
            "hba1c",
            "study_group",
            "clinical_site",
        ]
    ].copy()
    source["study_group_label"] = source["study_group"].map(
        STUDY_GROUP_LABELS
    )
    source["full_pc1"] = full_scores[:, 0]
    source["full_pc2"] = full_scores[:, 1]
    source["neutral_pc1"] = neutral_scores[:, 0]
    source["neutral_pc2"] = neutral_scores[:, 1]
    source["full_pc1_validation_variance"] = full_variance[0]
    source["full_pc2_validation_variance"] = full_variance[1]
    source["neutral_pc1_validation_variance"] = neutral_variance[0]
    source["neutral_pc2_validation_variance"] = neutral_variance[1]
    source.to_csv(source_data_path, index=False)
    glucose = pd.to_numeric(source["mean_glucose"], errors="coerce").to_numpy()
    hba1c = pd.to_numeric(source["hba1c"], errors="coerce").to_numpy()
    glucose_observed = np.isfinite(glucose)
    hba1c_observed = np.isfinite(hba1c)
    glucose_normalization = Normalize(
        vmin=float(np.nanmin(glucose)),
        vmax=float(np.nanmax(glucose)),
    )
    hba1c_normalization = Normalize(
        vmin=float(np.nanmin(hba1c)),
        vmax=float(np.nanmax(hba1c)),
    )
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    glucose_scatter = axes[0, 0].scatter(
        neutral_scores[glucose_observed, 0],
        neutral_scores[glucose_observed, 1],
        c=glucose[glucose_observed],
        cmap="viridis",
        norm=glucose_normalization,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolor=PLOT_WHITE,
        linewidth=0.25,
    )
    axes[0, 0].set_title(
        f"A. Static-neutral test manifold by mean glucose\nn={glucose_observed.sum()}"
    )
    for label in STUDY_GROUP_ORDER:
        mask = source["study_group_label"].eq(label).to_numpy()
        axes[0, 1].scatter(
            neutral_scores[mask, 0],
            neutral_scores[mask, 1],
            color=STRATUM_COLORS[label],
            label=f"{label} (n={mask.sum()})",
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolor=PLOT_WHITE,
            linewidth=0.25,
        )
    axes[0, 1].set_title(
        "B. Static-neutral test manifold by study group\nn=221"
    )
    axes[0, 1].legend(
        frameon=False, fontsize=8.5, loc="best", title="Study group"
    )
    full_hba1c_scatter = axes[1, 0].scatter(
        full_scores[hba1c_observed, 0],
        full_scores[hba1c_observed, 1],
        c=hba1c[hba1c_observed],
        cmap="viridis",
        norm=hba1c_normalization,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolor=PLOT_WHITE,
        linewidth=0.25,
    )
    axes[1, 0].set_title(
        f"C. Full-profile test manifold by HbA1c\nn={hba1c_observed.sum()}"
    )
    axes[1, 1].scatter(
        neutral_scores[hba1c_observed, 0],
        neutral_scores[hba1c_observed, 1],
        c=hba1c[hba1c_observed],
        cmap="viridis",
        norm=hba1c_normalization,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolor=PLOT_WHITE,
        linewidth=0.25,
    )
    axes[1, 1].set_title(
        f"D. Static-neutral test manifold by HbA1c\nn={hba1c_observed.sum()}"
    )
    for row_index, column_index, variance in (
        (0, 0, neutral_variance),
        (0, 1, neutral_variance),
        (1, 0, full_variance),
        (1, 1, neutral_variance),
    ):
        axis = axes[row_index, column_index]
        axis.set_xlabel(
            f"PC1 ({100 * variance[0]:.1f}% validation variance)"
        )
        axis.set_ylabel(
            f"PC2 ({100 * variance[1]:.1f}% validation variance)"
        )
    glucose_colorbar = figure.colorbar(
        glucose_scatter, ax=axes[0, 0], fraction=0.045, pad=0.04
    )
    glucose_colorbar.set_label("Participant mean glucose (mg/dL)")
    hba1c_colorbar = figure.colorbar(
        full_hba1c_scatter,
        ax=[axes[1, 0], axes[1, 1]],
        fraction=0.025,
        pad=0.04,
    )
    hba1c_colorbar.set_label("HbA1c (%)")
    figure.suptitle(
        "Clinical overlays on matching frozen participant manifolds",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Full and neutral panels use their matching frozen validation PCA spaces; coordinates are not directly identical.",
        ha="center",
        color=HOUSE_GRAY,
        fontsize=9.5,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.91,
        bottom=0.08,
        top=0.90,
        wspace=0.28,
        hspace=0.30,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def make_context_figure(
    output_path: Path,
    source_data_path: Path,
    representations: pd.DataFrame,
    features: pd.DataFrame,
    step3_directory: Path,
    context_metrics: pd.Series,
) -> None:
    features = features.copy()
    features["participant_id"] = features["participant_id"].astype(str)
    participant_ids = features["participant_id"].tolist()
    contexts = (
        ("neutral_all", "All-recording", HOUSE_GRAY),
        ("neutral_night", "Nighttime", HOUSE_NAVY),
        ("neutral_day", "Daytime", HOUSE_TEAL),
    )
    projected = []
    source_rows = []
    for representation_space, label, color in contexts:
        scores, variance = frozen_projection(
            representations,
            participant_ids,
            step3_directory,
            representation_space,
            projection_space="neutral_all",
        )
        projected.append((label, color, scores, variance))
        source_rows.append(
            pd.DataFrame(
                {
                    "participant_id": participant_ids,
                    "context": label,
                    "pc1": scores[:, 0],
                    "pc2": scores[:, 1],
                    "pc1_validation_variance": variance[0],
                    "pc2_validation_variance": variance[1],
                }
            )
        )
    pd.concat(source_rows, ignore_index=True).to_csv(
        source_data_path, index=False
    )
    combined = np.vstack([item[2] for item in projected])
    x_limits = np.quantile(combined[:, 0], [0.005, 0.995])
    y_limits = np.quantile(combined[:, 1], [0.005, 0.995])
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(
        1, 3, figsize=(15.5, 5.3), sharex=True, sharey=True
    )
    for axis, (label, color, scores, variance) in zip(axes, projected):
        axis.scatter(
            scores[:, 0],
            scores[:, 1],
            s=POINT_SIZE,
            color=color,
            alpha=0.62,
            edgecolor=PLOT_WHITE,
            linewidth=0.25,
        )
        axis.set_title(f"{label} test representation\nn=221")
        axis.set_xlabel(
            f"Neutral PC1 ({100 * variance[0]:.1f}% validation variance)"
        )
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
    axes[0].set_ylabel(
        f"Neutral PC2 ({100 * projected[0][3][1]:.1f}% validation variance)"
    )
    figure.suptitle(
        "Night and day reorganize participant geometry on one frozen coordinate system",
        fontsize=14,
        weight="bold",
    )
    metric_text = (
        "Night versus day: distance correlation "
        f"{context_metrics['distance_spearman']:.3f} | "
        f"NN10 overlap {context_metrics['nn10_overlap']:.3f} | "
        f"median cosine {context_metrics['median_cosine']:.3f}"
    )
    figure.text(
        0.5, 0.035, metric_text, ha="center", fontsize=10.5, weight="bold"
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.18,
        top=0.80,
        wspace=0.16,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def make_k2_figure(
    output_path: Path,
    source_data_path: Path,
    assignments: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    assignments = assignments.copy()
    features = features.copy()
    assignments["participant_id"] = assignments["participant_id"].astype(str)
    features["participant_id"] = features["participant_id"].astype(str)
    source = features[
        ["participant_id", "tir_70_180", "mean_glucose", "glucose_cv"]
    ].merge(
        assignments[
            [
                "participant_id",
                "assigned_exploratory_group",
                "normalized_assignment_margin",
            ]
        ],
        on="participant_id",
        validate="one_to_one",
    )
    group_counts = source["assigned_exploratory_group"].value_counts().to_dict()
    if group_counts != {1: 196, 0: 25}:
        raise RuntimeError(f"Frozen k=2 counts changed: {group_counts}")
    source["group_label"] = source["assigned_exploratory_group"].map(
        {0: "Glycemic-tail group", 1: "Reference group"}
    )
    source["tir_percent"] = 100 * source["tir_70_180"]
    source["glucose_cv_percent"] = 100 * source["glucose_cv"]
    source.to_csv(source_data_path, index=False)
    order = ("Glycemic-tail group", "Reference group")
    colors = (HOUSE_CRIMSON, HOUSE_NAVY)
    panels = (
        (
            "tir_percent",
            "Time in range 70 to 180 (%)",
            "A. Time in range",
        ),
        ("mean_glucose", "Mean glucose (mg/dL)", "B. Mean glucose"),
        ("glucose_cv_percent", "Glucose CV (%)", "C. Glucose CV"),
        (
            "normalized_assignment_margin",
            "Normalized assignment margin",
            "D. Assignment confidence",
        ),
    )
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5.5),
        gridspec_kw={"width_ratios": [1.35, 1, 1, 1]},
    )
    rng = np.random.default_rng(RANDOM_SEED)
    for axis, (variable, y_label, title) in zip(axes, panels):
        values = [
            source.loc[source["group_label"].eq(label), variable]
            .dropna()
            .to_numpy(float)
            for label in order
        ]
        boxplot = axis.boxplot(
            values,
            labels=[
                f"Glycemic tail\n(n={len(values[0])})",
                f"Reference\n(n={len(values[1])})",
            ],
            patch_artist=True,
            widths=0.58,
            showfliers=False,
            medianprops={"color": PLOT_BLACK, "linewidth": 1.8},
        )
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        for position, data in enumerate(values, start=1):
            jitter = rng.normal(position, 0.045, len(data))
            axis.scatter(
                jitter,
                data,
                s=10,
                color=PLOT_BLACK,
                alpha=0.25,
            )
            axis.text(
                position,
                axis.get_ylim()[1],
                f"median {np.median(data):.1f}",
                ha="center",
                va="top",
                fontsize=8.3,
            )
        axis.set_ylabel(y_label)
        axis.set_title(title)
    figure.suptitle(
        "Exploratory near-threshold k=2 sensitivity analysis",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.02,
        "Frozen test groups represent an exploratory glycemic-tail stratification, not a confirmed subtype.",
        ha="center",
        color=HOUSE_GRAY,
        fontsize=9.5,
    )
    figure.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.18,
        top=0.80,
        wspace=0.30,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def make_probe_figure(
    output_path: Path,
    source_data_path: Path,
    transport: pd.DataFrame,
) -> None:
    order = (
        "c_reactive_protein_i",
        "natriuretic_peptide_b_prohormon",
        "bun_creatinine_ratio",
    )
    source = transport.set_index("target").reindex(order).reset_index()
    source["target_label"] = source["target"].map(PROBE_LABELS)
    source.to_csv(source_data_path, index=False)
    y_positions = np.arange(len(order))[::-1]
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    for offset, split, color, marker, label in (
        (
            0.11,
            "validation",
            HOUSE_NAVY,
            "o",
            "Validation nested CV",
        ),
        (
            -0.11,
            "test",
            HOUSE_CRIMSON,
            "s",
            "Secondary test transport",
        ),
    ):
        estimates = source[f"{split}_delta_r2"].to_numpy(float)
        lower = source[f"{split}_ci_low"].to_numpy(float)
        upper = source[f"{split}_ci_high"].to_numpy(float)
        axis.errorbar(
            estimates,
            y_positions + offset,
            xerr=np.vstack([estimates - lower, upper - estimates]),
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=4,
            markersize=7,
            label=label,
        )
        for x_value, y_value in zip(estimates, y_positions + offset):
            axis.annotate(
                f"{x_value:+.3f}",
                (x_value, y_value),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8.5,
                color=color,
            )
    axis.axvline(
        0.0, color=HOUSE_GRAY, linewidth=1.2, linestyle="--"
    )
    axis.set_yticks(
        y_positions, source["target_label"].tolist()
    )
    axis.set_xlabel(
        "Incremental $R^2$: glycemic and wearable baseline plus neutral state minus baseline"
    )
    axis.set_title(
        "External clinical probes: validation effects and test transport",
        fontsize=13,
        weight="bold",
    )
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    axis.text(
        0.01,
        -0.31,
        "95% participant-bootstrap intervals. The small hs-CRP validation increment exceeded the shuffled-state null, "
        "but its interval included zero and the effect reversed on test.",
        transform=axis.transAxes,
        fontsize=9.2,
        color=HOUSE_GRAY,
    )
    figure.subplots_adjust(
        left=0.25, right=0.97, bottom=0.42, top=0.88
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def scan_em_dash(paths: list[Path]) -> list[str]:
    affected: list[str] = []
    for root in paths:
        candidates = root.rglob("*") if root.is_dir() else [root]
        for path in candidates:
            if path.is_file() and path.suffix.lower() in {
                ".csv", ".json", ".log", ".md", ".py", ".tex"
            }:
                try:
                    if NO_EM_DASH in path.read_text():
                        affected.append(str(path))
                except UnicodeDecodeError:
                    continue
    return affected


def run_figure_stage(
    run_directory: Path,
    step3_directory: Path,
    step4_directory: Path,
    step5_directory: Path,
    step6_directory: Path,
) -> dict[str, Any]:
    output_directory = run_directory / "revised_figures"
    manifest_path = run_directory / "step7_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("neighbor_stage", {}).get("status") != "GATE2_COMPLETE":
        raise RuntimeError("Gate 2 is not complete")
    if manifest.get("hba1c_stage", {}).get("status") != "QC_COMPLETE":
        raise RuntimeError("HbA1c stage must complete before final figures")
    if manifest.get("figure_stage", {}).get("status") == "QC_COMPLETE":
        raise RuntimeError("Figure stage is already complete")
    if any(output_directory.iterdir()):
        raise RuntimeError(f"Revised figure directory is not empty: {output_directory}")
    representations = pd.read_parquet(
        step4_directory / "test_participant_representations.parquet"
    )
    features = pd.read_parquet(
        step4_directory / "test_glycemic_nuisance_features.parquet"
    )
    assignments = pd.read_parquet(
        step4_directory / "test_exploratory_k2_assignments.parquet"
    )
    context_metrics = pd.read_csv(
        step4_directory / "test_context_geometry_comparison.csv"
    ).set_index("comparison").loc["night_vs_day"]
    transport = pd.read_csv(
        step5_directory / "probe_transport_summary.csv"
    )
    output_paths = {
        "static": output_directory
        / "figure_static_conditioning_reliability.png",
        "manifold": output_directory
        / "figure_continuous_manifold_clinical_overlays.png",
        "context": output_directory / "figure_context_night_day.png",
        "k2": output_directory
        / "figure_exploratory_k2_sensitivity.png",
        "probes": output_directory
        / "figure_external_clinical_probes.png",
        "neighbors": output_directory
        / "figure_neighbor_clinical_sharing_full_vs_neutral.png",
    }
    source_paths = {
        "static": output_directory
        / "figure_static_conditioning_reliability_data.csv",
        "manifold": output_directory
        / "figure_continuous_manifold_clinical_overlays_data.csv",
        "context": output_directory / "figure_context_night_day_data.csv",
        "k2": output_directory
        / "figure_exploratory_k2_sensitivity_data.csv",
        "probes": output_directory
        / "figure_external_clinical_probes_data.csv",
    }
    mapping_path = output_directory / "study_group_label_mapping.csv"
    make_static_figure(
        output_paths["static"], source_paths["static"], step6_directory
    )
    make_manifold_overlays(
        output_paths["manifold"],
        source_paths["manifold"],
        mapping_path,
        representations,
        features,
        step3_directory,
    )
    make_context_figure(
        output_paths["context"],
        source_paths["context"],
        representations,
        features,
        step3_directory,
        context_metrics,
    )
    make_k2_figure(
        output_paths["k2"],
        source_paths["k2"],
        assignments,
        features,
    )
    make_probe_figure(
        output_paths["probes"], source_paths["probes"], transport
    )
    shutil.copy2(
        run_directory
        / "neighbor_sharing/figure_neighbor_clinical_sharing_full_vs_neutral.png",
        output_paths["neighbors"],
    )
    hba1c_directory = run_directory / "hba1c_positive_control"
    hba1c_incremental = (
        hba1c_directory / "figure_hba1c_incremental_value.png"
    )
    hba1c_manifold = (
        hba1c_directory / "figure_hba1c_full_vs_neutral_manifold.png"
    )
    figure_rows = [
        {
            "figure_id": "S7-F1",
            "figure_title": "Static conditioning and representation reliability",
            "output_path": str(output_paths["static"]),
            "source_figure": str(
                step6_directory
                / "final_figures/figure2_static_conditioning_reliability.png"
            ),
            "source_data": str(source_paths["static"]),
            "analysis_step": "Steps 1, 2, and 4",
            "participant_split": "validation and test",
            "participant_count": "12 pilot; 239 validation; 221 test",
            "main_message":
                "Static effects and reliability are shown on distinct scales.",
            "primary_or_exploratory": "primary",
            "palette_status": "house palette complete",
            "qc_status": "QC_COMPLETE",
            "notes": "State units and forecast mg/dL are separated.",
        },
        {
            "figure_id": "S7-F2",
            "figure_title": "Continuous manifold clinical overlays",
            "output_path": str(output_paths["manifold"]),
            "source_figure": str(
                step6_directory
                / "final_figures/figure3_continuous_participant_geometry.png"
            ),
            "source_data": str(source_paths["manifold"]),
            "analysis_step": "Step 7",
            "participant_split": "test",
            "participant_count": "221; HbA1c n=217",
            "main_message":
                "The continuous manifold tracks glucose, study group, and HbA1c.",
            "primary_or_exploratory": "primary",
            "palette_status": "house palette and continuous colorbars complete",
            "qc_status": "QC_COMPLETE",
            "notes": "Matching full and neutral frozen PCA spaces are labelled.",
        },
        {
            "figure_id": "S7-F3",
            "figure_title": "Night and day context geometry",
            "output_path": str(output_paths["context"]),
            "source_figure": str(
                step6_directory
                / "final_figures/figure4_context_dependence.png"
            ),
            "source_data": str(source_paths["context"]),
            "analysis_step": "Step 4",
            "participant_split": "test",
            "participant_count": "221",
            "main_message":
                "All-recording, night, and day representations differ on one frozen coordinate system.",
            "primary_or_exploratory": "primary",
            "palette_status": "house palette complete",
            "qc_status": "QC_COMPLETE",
            "notes": "Night is navy and day is teal.",
        },
        {
            "figure_id": "S7-F4",
            "figure_title": "Exploratory near-threshold k=2 sensitivity analysis",
            "output_path": str(output_paths["k2"]),
            "source_figure": str(
                step6_directory
                / "final_figures/figure5_exploratory_k2_glycemic_tail.png"
            ),
            "source_data": str(source_paths["k2"]),
            "analysis_step": "Steps 3B and 4",
            "participant_split": "test",
            "participant_count": "25 glycemic tail; 196 reference",
            "main_message":
                "The exploratory stratification captures a reproducible glycemic extreme.",
            "primary_or_exploratory": "exploratory",
            "palette_status": "house palette complete",
            "qc_status": "QC_COMPLETE",
            "notes": "TIR is primary; confidence is a separate panel.",
        },
        {
            "figure_id": "S7-F5",
            "figure_title": "External clinical probe transport",
            "output_path": str(output_paths["probes"]),
            "source_figure": str(
                step6_directory
                / "final_figures/figure6_incremental_clinical_probes.png"
            ),
            "source_data": str(source_paths["probes"]),
            "analysis_step": "Step 5",
            "participant_split": "validation and test",
            "participant_count": "235 validation; 217 test",
            "main_message":
                "External biomarker increments did not transport.",
            "primary_or_exploratory": "secondary predictive transport",
            "palette_status": "house palette complete",
            "qc_status": "QC_COMPLETE",
            "notes": "Validation is navy and test is crimson.",
        },
        {
            "figure_id": "S7-F6",
            "figure_title": "Nearest-neighbour clinical sharing",
            "output_path": str(output_paths["neighbors"]),
            "source_figure": str(
                run_directory
                / "neighbor_sharing/figure_neighbor_clinical_sharing_full_vs_neutral.png"
            ),
            "source_data": str(
                run_directory
                / "neighbor_sharing/neighbor_sharing_tier1_results.csv"
            ),
            "analysis_step": "Step 7",
            "participant_split": "test",
            "participant_count": "221; HbA1c and biomarkers n=217",
            "main_message":
                "Glycemic and study-group sharing remains after static neutralization.",
            "primary_or_exploratory": "primary",
            "palette_status": "house palette complete",
            "qc_status": "QC_COMPLETE",
            "notes": "Neutral FDR q-values are annotated.",
        },
        {
            "figure_id": "S7-F7",
            "figure_title": "HbA1c incremental predictive value",
            "output_path": str(hba1c_incremental),
            "source_figure": "",
            "source_data": str(
                hba1c_directory / "hba1c_incremental_value.csv"
            ),
            "analysis_step": "Step 7",
            "participant_split": "validation and targeted test transport",
            "participant_count": "235 validation; 217 test",
            "main_message":
                "Full and neutral HbA1c increments are compared with uncertainty.",
            "primary_or_exploratory": "targeted positive control",
            "palette_status": "house palette complete",
            "qc_status": "QC_COMPLETE",
            "notes": "Not an untouched confirmation.",
        },
        {
            "figure_id": "S7-F8",
            "figure_title": "HbA1c on full and neutral manifolds",
            "output_path": str(hba1c_manifold),
            "source_figure": "",
            "source_data": str(source_paths["manifold"]),
            "analysis_step": "Step 7",
            "participant_split": "test",
            "participant_count": "217 with HbA1c",
            "main_message":
                "HbA1c is displayed on matching frozen full and neutral PCA spaces.",
            "primary_or_exploratory": "targeted positive control",
            "palette_status": "continuous colorbar complete",
            "qc_status": "QC_COMPLETE",
            "notes": "Identical HbA1c color limits are used.",
        },
    ]
    figure_manifest = pd.DataFrame(figure_rows)
    figure_manifest_path = (
        output_directory / "revised_figure_manifest.csv"
    )
    figure_manifest.to_csv(figure_manifest_path, index=False)
    required_figures = [
        *output_paths.values(),
        hba1c_incremental,
        hba1c_manifold,
    ]
    if any(not path.exists() for path in required_figures):
        raise RuntimeError("One or more required revised figures are missing")
    em_dash_files = scan_em_dash(
        [Path(__file__), output_directory, hba1c_directory]
    )
    if em_dash_files:
        raise RuntimeError(
            "Forbidden Unicode U+2014 found: " + ", ".join(em_dash_files)
        )
    manifest["figure_stage"] = {
        "status": "QC_COMPLETE",
        "figure_count": len(figure_rows),
        "figure_manifest": str(figure_manifest_path),
        "study_group_mapping": str(mapping_path),
        "output_paths": [str(path) for path in required_figures],
        "original_step6_figures_overwritten": False,
        "palette_status": "QC_COMPLETE",
        "blockers": [],
    }
    write_json(manifest_path, manifest)
    with (run_directory / "step7_run.log").open("a") as handle:
        handle.write("STEP 7 revised figure stage completed\n")
        handle.write(f"Revised figure count: {len(figure_rows)}\n")
    return {
        "output_directory": str(output_directory),
        "figure_count": len(figure_rows),
        "figure_manifest": str(figure_manifest_path),
        "figure_paths": [str(path) for path in required_figures],
        "warnings": [],
        "blockers": [],
    }
