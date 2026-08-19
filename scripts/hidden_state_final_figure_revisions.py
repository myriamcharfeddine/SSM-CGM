#!/usr/bin/env python3
"""Submission-ready Step 6 figures and manual TIR verification.

All projections use validation-fitted scalers/PCA objects saved in Step 3.
No estimator is fitted here.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


HIDDEN = [f"r_{i:03d}" for i in range(128)]
TARGET_LABELS = {
    "c_reactive_protein_i": "High-sensitivity CRP",
    "natriuretic_peptide_b_prohormon": "NT-proBNP",
    "bun_creatinine_ratio": "BUN/creatinine ratio",
}


def _representation_matrix(
    representations: pd.DataFrame,
    participant_ids: list[str],
    space: str,
) -> np.ndarray:
    q = representations.loc[
        representations["representation_type"].eq(space),
        ["participant_id", *HIDDEN],
    ].copy()
    q["participant_id"] = q["participant_id"].astype(str)
    q = q.set_index("participant_id").reindex(participant_ids)
    x = q[HIDDEN].to_numpy(float)
    if x.shape != (len(participant_ids), 128) or not np.isfinite(x).all():
        raise RuntimeError(f"Invalid frozen representation matrix: {space} {x.shape}")
    return x


def frozen_projection(
    representations: pd.DataFrame,
    participant_ids: list[str],
    step3_dir: Path,
    representation_space: str,
    projection_space: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an already-fitted validation scaler/PCA; never fit a model."""
    projection_space = projection_space or representation_space
    x = _representation_matrix(representations, participant_ids, representation_space)
    frozen = step3_dir / "frozen_validation_pipeline" / projection_space
    keep = np.load(frozen / "kept_dimensions.npy")
    scaler = joblib.load(frozen / f"{projection_space}_scaler.joblib")
    pca = joblib.load(frozen / f"{projection_space}_pca.joblib")
    scores = pca.transform(scaler.transform(x[:, keep]))[:, :2]
    variance = np.asarray(pca.explained_variance_ratio_[:2], dtype=float)
    return scores, variance


def verify_test_tir(
    panel_path: Path,
    assignments: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Recompute TIR from valid raw CGM rows and audit all frozen group keys."""
    a = assignments.copy()
    f = features.copy()
    a["participant_id"] = a["participant_id"].astype(str)
    f["participant_id"] = f["participant_id"].astype(str)
    ids = a["participant_id"].tolist()
    panel = pd.read_parquet(
        panel_path,
        columns=["participant_id", "cgm_glucose_mean", "cgm_count"],
        filters=[("participant_id", "in", ids)],
    )
    panel["participant_id"] = panel["participant_id"].astype(str)
    rows = []
    for participant_id, group in panel.groupby("participant_id", sort=True):
        glucose = pd.to_numeric(group["cgm_glucose_mean"], errors="coerce")
        valid = group["cgm_count"].fillna(0).gt(0) & glucose.notna()
        values = glucose.loc[valid].to_numpy(float)
        rows.append({
            "participant_id": participant_id,
            "valid_cgm_rows": int(len(values)),
            "tir_recomputed_fraction": float(
                np.mean((values >= 70) & (values <= 180))
            ),
        })
    recomputed = pd.DataFrame(rows)
    merged = (
        recomputed
        .merge(
            a[["participant_id", "assigned_exploratory_group"]],
            on="participant_id",
            validate="one_to_one",
        )
        .merge(
            f[[
                "participant_id", "tir_70_180", "mean_glucose", "glucose_cv",
                "study_group", "exploratory_group",
            ]],
            on="participant_id",
            validate="one_to_one",
        )
    )
    merged["tir_abs_difference"] = (
        merged["tir_recomputed_fraction"] - merged["tir_70_180"]
    ).abs()
    counts = (
        merged["assigned_exploratory_group"].value_counts().sort_index().to_dict()
    )
    medians = merged.groupby("assigned_exploratory_group", observed=True).agg(
        tir_fraction=("tir_recomputed_fraction", "median"),
        mean_glucose_mgdl=("mean_glucose", "median"),
        glucose_cv_fraction=("glucose_cv", "median"),
        valid_cgm_rows=("valid_cgm_rows", "median"),
    )
    study_counts = pd.crosstab(
        merged["assigned_exploratory_group"], merged["study_group"]
    )
    checks = {
        "assignment_rows_221": len(a) == 221,
        "assignment_ids_unique": a["participant_id"].nunique() == 221,
        "feature_rows_221": len(f) == 221,
        "raw_recomputed_participants_221": len(recomputed) == 221,
        "merged_participants_221": len(merged) == 221,
        "group_counts_25_196": counts == {0: 25, 1: 196},
        "all_participants_have_valid_cgm": bool((merged["valid_cgm_rows"] > 0).all()),
        "saved_tir_matches_raw_recomputation": bool(
            merged["tir_abs_difference"].max() <= 1e-12
        ),
        "assignment_and_feature_group_labels_match": bool(
            (
                merged["assigned_exploratory_group"]
                == merged["exploratory_group"]
            ).all()
        ),
        "no_missing_study_group": int(merged["study_group"].isna().sum()) == 0,
        "all_four_study_groups_present": merged["study_group"].nunique() == 4,
        "tir_stored_as_fraction": bool(
            merged["tir_70_180"].between(0, 1, inclusive="both").all()
        ),
        "tir_figure_converts_fraction_to_percent": True,
    }
    payload = {
        "status": "QC_COMPLETE" if all(checks.values()) else "QC_FAILED",
        "purpose": (
            "Manual verification requested before final submission; this is a "
            "definition/key/coverage audit, not a new hypothesis test."
        ),
        "tir_definition": (
            "Fraction of rows with 70 <= cgm_glucose_mean <= 180 among rows "
            "where cgm_count > 0 and cgm_glucose_mean is nonmissing."
        ),
        "display_unit": "percent (100 × stored fraction)",
        "checks": checks,
        "group_counts": {str(k): int(v) for k, v in counts.items()},
        "group_labels": {
            "0": "glycemic-tail group",
            "1": "reference group",
        },
        "group_medians": {
            str(index): {
                "tir_fraction": float(row["tir_fraction"]),
                "tir_percent": float(100 * row["tir_fraction"]),
                "mean_glucose_mgdl": float(row["mean_glucose_mgdl"]),
                "glucose_cv_fraction": float(row["glucose_cv_fraction"]),
                "valid_cgm_rows": float(row["valid_cgm_rows"]),
            }
            for index, row in medians.iterrows()
        },
        "maximum_saved_vs_recomputed_tir_absolute_difference": float(
            merged["tir_abs_difference"].max()
        ),
        "study_group_counts_by_frozen_group": {
            str(index): {
                str(column): int(study_counts.loc[index, column])
                for column in study_counts.columns
            }
            for index in study_counts.index
        },
        "interpretation_caveat": (
            "No study group was excluded, but the 25-participant glycemic-tail "
            "group contains only insulin-dependent and medication-controlled "
            "participants; composition is therefore a major non-subtype caveat."
        ),
    }
    if payload["status"] != "QC_COMPLETE":
        raise RuntimeError(payload)
    return payload, merged


def _annotate_bars(ax: plt.Axes, digits: int = 3, suffix: str = "") -> None:
    for patch in ax.patches:
        value = patch.get_height()
        ax.annotate(
            f"{value:.{digits}f}{suffix}",
            (patch.get_x() + patch.get_width() / 2, value),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )


def make_static_reliability_figure(
    path: Path,
    validation: pd.Series,
    test: pd.Series,
    step1: dict,
) -> None:
    """Separate state units from forecast mg/dL and label reliability values."""
    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 5.5), gridspec_kw={"width_ratios": [0.75, 1.1, 2.2]}
    )
    axes[0].bar(
        ["Median\nfull–neutral L2"], [24.5532], color="#4C78A8", width=0.55
    )
    axes[0].set_ylabel("State-space L2 distance")
    axes[0].set_title("A. Hidden-state effect")
    axes[0].text(
        .5, .91, "Median cosine = 0.948",
        transform=axes[0].transAxes, ha="center", fontsize=9,
    )
    _annotate_bars(axes[0], digits=2)

    forecast = step1["descriptive_forecast_delta"]
    axes[1].bar(
        ["Mean absolute\nforecast difference", "Terminal\nforecast difference"],
        [
            forecast["mean_absolute_full_neutral_forecast_difference"],
            forecast["terminal_full_neutral_forecast_difference"],
        ],
        color=["#F58518", "#E45756"],
    )
    axes[1].set_ylabel("Forecast difference (mg/dL)")
    axes[1].set_title("B. Forecast effect")
    _annotate_bars(axes[1], digits=2)

    metrics = [
        "median_within_cosine", "top1_retrieval", "top5_retrieval", "median_icc"
    ]
    labels = ["Odd/even\ncosine", "Top-1\nretrieval", "Top-5\nretrieval", "Median\nICC"]
    x = np.arange(len(metrics))
    axes[2].bar(
        x - .18, [validation[m] for m in metrics], .36,
        label="Validation (n=239)", color="#4C78A8",
    )
    axes[2].bar(
        x + .18, [test[m] for m in metrics], .36,
        label="Test (n=221)", color="#54A24B",
    )
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 1.08)
    axes[2].set_ylabel("Proportion / coefficient")
    axes[2].set_title("C. Static-neutralized reliability")
    axes[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.20), ncol=2)
    _annotate_bars(axes[2], digits=3)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Static conditioning changes model behavior; neutral representations remain reliable",
        fontsize=14, weight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_continuous_manifold_figure(
    path: Path,
    representations: pd.DataFrame,
    features: pd.DataFrame,
    step3_dir: Path,
) -> None:
    """Full, neutral, and residualized frozen test PCA with glucose colorbar."""
    ids = features["participant_id"].astype(str).tolist()
    spaces = [
        ("full_all", "Full-profile test representation"),
        ("neutral_all", "Static-neutralized test representation"),
        ("neutral_glucose_residual", "Glucose-residualized neutral test representation"),
    ]
    projected = [
        (space, title, *frozen_projection(representations, ids, step3_dir, space))
        for space, title in spaces
    ]
    glucose = features.set_index(
        features["participant_id"].astype(str)
    ).reindex(ids)["mean_glucose"].to_numpy(float)
    norm = Normalize(vmin=float(np.nanmin(glucose)), vmax=float(np.nanmax(glucose)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    scatter = None
    for ax, (_, title, scores, variance) in zip(axes, projected):
        scatter = ax.scatter(
            scores[:, 0], scores[:, 1], c=glucose, cmap="viridis", norm=norm,
            s=25, alpha=.82, edgecolor="white", linewidth=.25,
        )
        ax.set_title(f"{title}\n(n=221)", fontsize=10.5)
        ax.set_xlabel(f"PC1 ({100 * variance[0]:.1f}% validation variance)")
        ax.set_ylabel(f"PC2 ({100 * variance[1]:.1f}% validation variance)")
        ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(scatter, ax=axes, fraction=.025, pad=.025)
    cbar.set_label("Participant mean glucose (mg/dL)")
    fig.suptitle(
        "Frozen-PCA test geometry: the color gradient identifies glycemic organization",
        fontsize=14, weight="bold",
    )
    fig.subplots_adjust(left=.06, right=.91, bottom=.13, top=.82, wspace=.28)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_context_figure(
    path: Path,
    representations: pd.DataFrame,
    features: pd.DataFrame,
    step3_dir: Path,
    context_metrics: pd.Series,
) -> None:
    """All/night/day test states on one frozen neutral-all PCA coordinate system."""
    ids = features["participant_id"].astype(str).tolist()
    contexts = [
        ("neutral_all", "All-recording", "#4C78A8"),
        ("neutral_night", "Nighttime", "#7A5195"),
        ("neutral_day", "Daytime", "#E07B39"),
    ]
    projected = [
        (
            space, title, color,
            *frozen_projection(
                representations, ids, step3_dir, space,
                projection_space="neutral_all",
            ),
        )
        for space, title, color in contexts
    ]
    all_scores = np.vstack([item[3] for item in projected])
    xlim = np.quantile(all_scores[:, 0], [.005, .995])
    ylim = np.quantile(all_scores[:, 1], [.005, .995])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), sharex=True, sharey=True)
    for ax, (_, title, color, scores, variance) in zip(axes, projected):
        ax.scatter(
            scores[:, 0], scores[:, 1], s=23, color=color, alpha=.62,
            edgecolor="white", linewidth=.25,
        )
        ax.set_title(f"{title} test state\n(n=221)")
        ax.set_xlabel(f"Neutral PC1 ({100 * variance[0]:.1f}% validation variance)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(
        f"Neutral PC2 ({100 * projected[0][4][1]:.1f}% validation variance)"
    )
    fig.suptitle(
        "Context-dependent reorganization on a common frozen coordinate system",
        fontsize=14, weight="bold",
    )
    metric_text = (
        "Night vs day (test):  pairwise-distance Spearman "
        f"{context_metrics['distance_spearman']:.3f}   |   "
        f"NN10 overlap {context_metrics['nn10_overlap']:.3f}   |   "
        f"median participant cosine {context_metrics['median_cosine']:.3f}"
    )
    fig.text(.5, .035, metric_text, ha="center", fontsize=10.5, weight="bold")
    fig.subplots_adjust(left=.07, right=.98, bottom=.18, top=.80, wspace=.16)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_k2_glycemic_figure(path: Path, verified: pd.DataFrame) -> None:
    """Show actual glycemic distributions, led by TIR, for frozen test groups."""
    q = verified.copy()
    q["group_label"] = q["assigned_exploratory_group"].map({
        0: "Glycemic-tail\n(n=25)",
        1: "Reference\n(n=196)",
    })
    q["tir_percent"] = 100 * q["tir_recomputed_fraction"]
    q["glucose_cv_percent"] = 100 * q["glucose_cv"]
    order = ["Glycemic-tail\n(n=25)", "Reference\n(n=196)"]
    panels = [
        ("tir_percent", "Time in range 70–180 (%)", "A. TIR (primary evidence)"),
        ("mean_glucose", "Mean glucose (mg/dL)", "B. Mean glucose"),
        ("glucose_cv_percent", "Glucose CV (%)", "C. Glucose variability"),
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.35, 1, 1]}
    )
    for ax, (variable, ylabel, title) in zip(axes, panels):
        values = [
            q.loc[q["group_label"].eq(label), variable].to_numpy(float)
            for label in order
        ]
        bp = ax.boxplot(
            values, labels=order, patch_artist=True, widths=.58,
            showfliers=False, medianprops={"color": "black", "linewidth": 1.8},
        )
        for patch, color in zip(bp["boxes"], ["#E45756", "#4C78A8"]):
            patch.set_facecolor(color)
            patch.set_alpha(.72)
        rng = np.random.default_rng(42)
        for index, data in enumerate(values, start=1):
            jitter = rng.normal(index, .045, len(data))
            ax.scatter(jitter, data, s=10, color="#222222", alpha=.25)
            median = float(np.median(data))
            ax.text(
                index, ax.get_ylim()[1], f"median {median:.1f}",
                ha="center", va="top", fontsize=8.5,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Exploratory k=2 glycemic-tail characterization (frozen test groups)",
        fontsize=14, weight="bold",
    )
    fig.text(
        .5, .02,
        "TIR independently recomputed from valid CGM rows; stored fractions displayed as percentages.",
        ha="center", fontsize=9.5, style="italic",
    )
    fig.subplots_adjust(left=.07, right=.98, bottom=.18, top=.79, wspace=.28)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_probe_forest_figure(path: Path, transport: pd.DataFrame) -> None:
    """Validation and test delta-R2 estimates with participant-bootstrap CIs."""
    order = [
        "c_reactive_protein_i",
        "natriuretic_peptide_b_prohormon",
        "bun_creatinine_ratio",
    ]
    q = transport.set_index("target").reindex(order)
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for offset, split, color, marker in [
        (.11, "validation", "#4C78A8", "o"),
        (-.11, "test", "#E45756", "s"),
    ]:
        estimate = q[f"{split}_delta_r2"].to_numpy(float)
        low = q[f"{split}_ci_low"].to_numpy(float)
        high = q[f"{split}_ci_high"].to_numpy(float)
        ax.errorbar(
            estimate, y + offset,
            xerr=np.vstack([estimate - low, high - estimate]),
            fmt=marker, color=color, ecolor=color, elinewidth=1.8,
            capsize=4, markersize=7, label=(
                "Validation nested CV" if split == "validation"
                else "Secondary test transport"
            ),
        )
        for x_value, y_value in zip(estimate, y + offset):
            ax.annotate(
                f"{x_value:+.3f}", (x_value, y_value),
                xytext=(5, 4), textcoords="offset points", fontsize=8.5,
            )
    ax.axvline(0, color="black", lw=1, linestyle="--")
    ax.set_yticks(y, [TARGET_LABELS[target] for target in order])
    ax.set_xlabel("Incremental $R^2$: baseline + neutral state minus baseline")
    ax.set_title(
        "Clinical probes: validation increments and secondary test transport",
        fontsize=13, weight="bold",
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2)
    ax.grid(axis="x", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        .01, -.30,
        "95% participant-bootstrap intervals. hs-CRP: validation permutation "
        "q=0.006, bootstrap CI includes zero; test direction reverses.",
        transform=ax.transAxes, fontsize=9.5,
    )
    fig.subplots_adjust(left=.25, right=.97, bottom=.42, top=.88)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
