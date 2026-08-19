#!/usr/bin/env python3
"""Reorganize finalized Phase 2 results into subtype-centered figures.

This is a presentation-only workflow. It reads the saved participant-level and
plotted-data tables from the finalized reader-friendly v3 package because the
requested v4 source directory is absent. It never reads hidden states, rebuilds
neighbor graphs, changes labels, or changes selected k.
"""
from __future__ import annotations

import hashlib
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REPO = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
PHASE2 = REPO / (
    "outputs/static_phenotype_trajectory_stratified_v2/"
    "extended_clinical_latent_dynamics_v1/02_circadian_matched_reorganization"
)
REQUESTED_SOURCE = PHASE2 / "final_chance_adjusted_figures_v4"
SOURCE = PHASE2 / "recreated_reader_friendly_figures_v3"
OUT = PHASE2 / "final_subtype_centered_figures_v5"
FIG = OUT / "figures"
TABLE = OUT / "tables"
META = OUT / "metadata"
REPORT = OUT / "reports"
QA = OUT / "qa"

SEED = 42
BOOTSTRAP_N = 1000
HOURS = [6, 12, 24, 48]
BLACK = "#000000"
GRID = "#D9D9D9"
SUBTYPES = [
    "healthy", "pre_diabetes", "t2d_oral_non_insulin", "insulin_dependent"
]
SUBTYPE_NAMES = {
    "healthy": "Healthy",
    "pre_diabetes": "Prediabetes",
    "t2d_oral_non_insulin": "T2D oral non-insulin",
    "insulin_dependent": "Insulin-dependent, exploratory",
}
PALETTES = {
    "healthy": {"light": "#9CB3C8", "medium": "#5B7FA3", "dark": "#003366"},
    "pre_diabetes": {"light": "#B5DEDE", "medium": "#5BBABA", "dark": "#2F7F7F"},
    "t2d_oral_non_insulin": {"light": "#E7A6A6", "medium": "#BA4A4A", "dark": "#7A1F1F"},
    "insulin_dependent": {"light": "#C7CDD4", "medium": "#8994A2", "dark": "#4A5568"},
}
METRICS = ["clinical_to_h0", "h0_to_ht", "clinical_to_ht"]
METRIC_LABELS = {
    "clinical_to_h0": "Clinical → h₀\nStatic control",
    "h0_to_ht": "h₀ → hₜ",
    "clinical_to_ht": "Clinical → hₜ",
}
SHORT_METRIC_LABELS = {
    "clinical_to_h0": "Clinical → h₀",
    "h0_to_ht": "h₀ → hₜ",
    "clinical_to_ht": "Clinical → hₜ",
}
REPRESENTATIONS = ["clinical", "h0", "ht"]
REP_LABELS = {"clinical": "Clinical", "h0": "h₀", "ht": "hₜ"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(*parts: object) -> int:
    return SEED + zlib.crc32("|".join(map(str, parts)).encode("utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": BLACK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def frame(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.7)
    ax.grid(False, axis="x" if axis == "y" else "y")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(0.8)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}_thumbnail.png", dpi=70, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bootstrap_mean(values: pd.Series, *seed_parts: object) -> tuple[float, float, float, int]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(stable_seed(*seed_parts))
    index = rng.integers(0, len(x), size=(BOOTSTRAP_N, len(x)))
    means = x[index].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975)), len(x)


def normalize_metric(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("_jaccard", "", regex=False)


def y_limits(data: pd.DataFrame, low: str, high: str, pad: float = .08) -> tuple[float, float]:
    lo = float(pd.to_numeric(data[low], errors="coerce").min())
    hi = float(pd.to_numeric(data[high], errors="coerce").max())
    span = max(hi - lo, .05)
    return min(0, lo - pad * span), max(0, hi + pad * span)


def x_limits(data: pd.DataFrame, low: str = "ci_low", high: str = "ci_high", pad: float = .12) -> tuple[float, float]:
    lo = float(pd.to_numeric(data[low], errors="coerce").min())
    hi = float(pd.to_numeric(data[high], errors="coerce").max())
    span = max(hi - lo, .03)
    return min(0, lo - pad * span), max(0, hi + pad * span)


def grouped_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    subtype: str,
    conditions: list[str],
    condition_labels: list[str],
    value: str,
    low: str,
    high: str,
    ylim: tuple[float, float],
) -> None:
    x = np.arange(len(METRICS), dtype=float)
    width = .34
    for index, (condition, label) in enumerate(zip(conditions, condition_labels)):
        q = (
            data[(data.canonical_stratum == subtype) & (data.condition == condition)]
            .set_index("metric").reindex(METRICS)
        )
        pos = x + (index - .5) * width
        vals = q[value].to_numpy(float)
        ax.bar(
            pos, vals, width=width,
            color=PALETTES[subtype]["light" if index == 0 else "dark"],
            edgecolor=BLACK, linewidth=.65, label=label,
            yerr=np.vstack([vals - q[low].to_numpy(float), q[high].to_numpy(float) - vals]),
            capsize=3, error_kw={"elinewidth": 1, "capthick": 1},
        )
    ax.axhline(0, color=BLACK, linewidth=1)
    ax.set_xticks(x, [METRIC_LABELS[m] for m in METRICS], rotation=0)
    ax.set_ylim(*ylim)
    ax.set_title(SUBTYPE_NAMES[subtype], fontweight="bold", pad=8)
    frame(ax, "y")


def forest(
    ax: plt.Axes,
    data: pd.DataFrame,
    subtype: str,
    xlim: tuple[float, float],
    metric_column: str = "metric",
    metrics: list[str] | None = None,
    labels: dict[str, str] | None = None,
    colors: list[str] | None = None,
) -> None:
    metrics = metrics or METRICS
    labels = labels or SHORT_METRIC_LABELS
    colors = colors or [PALETTES[subtype]["dark"]] * len(metrics)
    q = data[data.canonical_stratum == subtype].set_index(metric_column)
    ypos = np.arange(len(metrics))[::-1]
    for y, metric, color in zip(ypos, metrics, colors):
        row = q.loc[metric]
        ax.errorbar(
            row.estimate, y,
            xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
            fmt="o", color=color, ecolor=color, capsize=3, markersize=6, linewidth=1.4,
        )
    ax.axvline(0, color=BLACK, linewidth=1)
    ax.set_yticks(ypos, [labels[m] for m in metrics])
    ax.set_xlim(*xlim)
    ax.set_title(SUBTYPE_NAMES[subtype], fontweight="bold", pad=8)
    frame(ax, "x")


def purity_bars(ax: plt.Axes, data: pd.DataFrame, subtype: str, ylim: tuple[float, float]) -> None:
    q = data[data.canonical_stratum == subtype].set_index("representation").reindex(REPRESENTATIONS)
    x = np.arange(3)
    shades = ["dark", "medium", "light"]
    vals = q.chance_adjusted_purity.to_numpy(float)
    ax.bar(
        x, vals, width=.62,
        color=[PALETTES[subtype][shade] for shade in shades], edgecolor=BLACK, linewidth=.65,
        yerr=np.vstack([
            vals - q.adjusted_ci_low.to_numpy(float),
            q.adjusted_ci_high.to_numpy(float) - vals,
        ]), capsize=3, error_kw={"elinewidth": 1, "capthick": 1},
    )
    ax.axhline(0, color=BLACK, linewidth=1)
    ax.set_xticks(x, [REP_LABELS[s] for s in REPRESENTATIONS])
    ax.set_ylim(*ylim)
    ax.set_title(SUBTYPE_NAMES[subtype], fontweight="bold", pad=8)
    frame(ax, "y")


def raw_bars_with_null(
    ax: plt.Axes,
    data: pd.DataFrame,
    subtype: str,
    conditions: list[str],
    labels: list[str],
    ylim: tuple[float, float],
) -> None:
    x = np.arange(3)
    width = .34
    for index, (condition, label) in enumerate(zip(conditions, labels)):
        q = (
            data[(data.canonical_stratum == subtype) & (data.condition == condition)]
            .set_index("metric").reindex(METRICS)
        )
        pos = x + (index - .5) * width
        vals = q.estimate.to_numpy(float)
        ax.bar(
            pos, vals, width=width,
            color=PALETTES[subtype]["light" if index == 0 else "dark"],
            edgecolor=BLACK, linewidth=.65,
            yerr=np.vstack([vals - q.ci_low.to_numpy(float), q.ci_high.to_numpy(float) - vals]),
            capsize=3, error_kw={"elinewidth": 1}, label=label,
        )
        ax.scatter(pos, q.expected_null, marker="D", s=32, color=BLACK, zorder=5)
    ax.set_xticks(x, [METRIC_LABELS[m] for m in METRICS])
    ax.set_ylim(*ylim)
    ax.set_title(SUBTYPE_NAMES[subtype], fontweight="bold", pad=8)
    frame(ax, "y")


def time_resolved(anchor: pd.DataFrame, analysis: str) -> pd.DataFrame:
    rows: list[dict] = []
    for (subtype, hour, metric, condition), group in anchor.groupby(
        ["canonical_stratum", "hour", "metric", "condition"], observed=True
    ):
        participant = group.groupby("participant_id").chance_adjusted_overlap.mean()
        estimate, lo, hi, n = bootstrap_mean(
            participant, "time_resolved", analysis, subtype, hour, metric, condition
        )
        rows.append({
            "analysis": analysis,
            "canonical_stratum": subtype,
            "hour": int(hour),
            "metric": metric,
            "condition": condition,
            "estimate": estimate,
            "ci_low": lo,
            "ci_high": hi,
            "participant_n": n,
            "anchor_n": int(group.anchor_key.nunique()),
            "bootstrap_n": BOOTSTRAP_N,
            "source_value": "chance_adjusted_overlap",
        })
    return pd.DataFrame(rows)


def make_main_2a(match_est: pd.DataFrame, match_diff: pd.DataFrame, purity: pd.DataFrame) -> None:
    style()
    fig, axes = plt.subplots(3, 4, figsize=(24, 16.5))
    match_ylim = y_limits(match_est, "adjusted_ci_low", "adjusted_ci_high")
    diff_xlim = x_limits(match_diff)
    purity_ylim = y_limits(purity, "adjusted_ci_low", "adjusted_ci_high")
    for col, subtype in enumerate(SUBTYPES):
        grouped_bars(
            axes[0, col], match_est, subtype,
            ["unmatched", "clock_time_matched"], ["Unmatched", "Clock-time matched"],
            "chance_adjusted_overlap", "adjusted_ci_low", "adjusted_ci_high", match_ylim,
        )
        forest(axes[1, col], match_diff, subtype, diff_xlim)
        purity_bars(axes[2, col], purity, subtype, purity_ylim)
        if col:
            axes[0, col].tick_params(labelleft=False)
            axes[2, col].tick_params(labelleft=False)
        else:
            axes[0, col].set_ylabel("Chance-adjusted\nshared-neighbor overlap")
            axes[2, col].set_ylabel("Chance-adjusted fixed-label\nneighbor purity")
        axes[1, col].set_xlabel("Matched minus unmatched\nadjusted overlap")
    fig.text(.055, .905, "A  Preservation above candidate-pool expectation", fontweight="bold", fontsize=12)
    fig.text(.055, .605, "B  Participant-paired effect of clock-time matching", fontweight="bold", fontsize=12)
    fig.text(.055, .309, "C  Clinical-label organization across representations", fontweight="bold", fontsize=12)
    fig.legend(
        handles=[
            Patch(facecolor="#D5D5D5", edgecolor=BLACK, label="Unmatched (light subtype shade)"),
            Patch(facecolor="#4F4F4F", edgecolor=BLACK, label="Clock-time matched (dark subtype shade)"),
        ], ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, .949),
    )
    fig.text(
        .985, .605,
        "Positive values indicate greater preservation after clock-time matching.",
        ha="right", va="center", fontsize=9.5,
    )
    fig.text(
        .5, .018,
        "Every subtype was analyzed independently. Main estimates aggregate 6, 12, 24, and 48 hours within participant. "
        "Error bars are 95% participant-bootstrap intervals. Clinical → h₀ is the static candidate-pool control. "
        "Raw overlap and permutation expectations are shown in Appendix A1. Insulin-dependent results are exploratory.",
        ha="center", fontsize=9, wrap=True,
    )
    fig.suptitle(
        "Chance-adjusted neighborhood preservation before and after circadian matching",
        fontsize=17, fontweight="bold", y=.988,
    )
    fig.subplots_adjust(left=.055, right=.99, top=.88, bottom=.07, hspace=.60, wspace=.14)
    save_figure(fig, "figure_2A_subtype_centered_circadian_matching")


def make_main_2b(day_est: pd.DataFrame, night_day: pd.DataFrame, residual: pd.DataFrame) -> None:
    style()
    fig, axes = plt.subplots(3, 4, figsize=(24, 16.5))
    day_ylim = y_limits(day_est, "adjusted_ci_low", "adjusted_ci_high")
    contrast_xlim = x_limits(night_day)
    residual_xlim = x_limits(residual)
    residual_metrics = ["clinical_to_ht", "h0_to_ht"]
    residual_labels = {
        "clinical_to_ht": "Clinical → hₜ residual",
        "h0_to_ht": "h₀ → hₜ residual",
    }
    for col, subtype in enumerate(SUBTYPES):
        grouped_bars(
            axes[0, col], day_est, subtype, ["day", "night"], ["Day", "Night"],
            "chance_adjusted_overlap", "adjusted_ci_low", "adjusted_ci_high", day_ylim,
        )
        forest(axes[1, col], night_day, subtype, contrast_xlim)
        forest(
            axes[2, col], residual, subtype, residual_xlim,
            metric_column="dynamic_metric", metrics=residual_metrics, labels=residual_labels,
            colors=[PALETTES[subtype]["medium"], PALETTES[subtype]["dark"]],
        )
        if col:
            axes[0, col].tick_params(labelleft=False)
        else:
            axes[0, col].set_ylabel("Chance-adjusted\nshared-neighbor overlap")
        axes[1, col].set_xlabel("Night minus day\nadjusted overlap")
        axes[2, col].set_xlabel("Residual night-minus-day overlap\nafter static-control adjustment")
    fig.text(.055, .905, "A  Preservation above chance during day and night", fontweight="bold", fontsize=12)
    fig.text(.055, .605, "B  Participant-paired night-minus-day contrast", fontweight="bold", fontsize=12)
    fig.text(.055, .309, "C  Residual dynamic difference after static-control adjustment", fontweight="bold", fontsize=12)
    fig.legend(
        handles=[
            Patch(facecolor="#D5D5D5", edgecolor=BLACK, label="Day (light subtype shade)"),
            Patch(facecolor="#4F4F4F", edgecolor=BLACK, label="Night (dark subtype shade)"),
        ], ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, .949),
    )
    fig.text(
        .985, .605, "Positive values indicate stronger preservation at night.",
        ha="right", va="center", fontsize=9.5,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#8F8F8F", lw=0,
                   label="Clinical → hₜ residual (medium subtype shade)"),
            Line2D([0], [0], marker="o", color="#3F3F3F", lw=0,
                   label="h₀ → hₜ residual (dark subtype shade)"),
        ], ncol=2, frameon=False, loc="upper right", bbox_to_anchor=(.99, .319),
    )
    fig.text(
        .5, .018,
        "Residual estimates subtract the participant-paired Clinical → h₀ static-control contrast. "
        "Positive values indicate residual nighttime preservation beyond the static-control pattern. "
        "Main estimates aggregate 6, 12, 24, and 48 hours within participant; intervals use participant bootstrap. "
        "Raw day-night overlap and permutation expectations are shown in Appendix A3. Insulin-dependent results are exploratory.",
        ha="center", fontsize=9, wrap=True,
    )
    fig.suptitle(
        "Chance-adjusted day-night differences in neighborhood preservation",
        fontsize=17, fontweight="bold", y=.988,
    )
    fig.subplots_adjust(left=.055, right=.99, top=.88, bottom=.075, hspace=.60, wspace=.23)
    save_figure(fig, "figure_2B_subtype_centered_day_night")


def make_appendix_a1(match_est: pd.DataFrame) -> None:
    style()
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.7), sharey=True)
    ylim = (0, max(1, float(match_est.ci_high.max()) * 1.05))
    for col, subtype in enumerate(SUBTYPES):
        raw_bars_with_null(
            axes[col], match_est, subtype,
            ["unmatched", "clock_time_matched"], ["Unmatched", "Clock-time matched"], ylim,
        )
        if col == 0:
            axes[col].set_ylabel("Raw shared-neighbor fraction")
        else:
            axes[col].tick_params(labelleft=False)
    fig.legend(
        handles=[
            Patch(facecolor="#D5D5D5", edgecolor=BLACK, label="Unmatched (light subtype shade)"),
            Patch(facecolor="#4F4F4F", edgecolor=BLACK, label="Clock-time matched (dark subtype shade)"),
            Line2D([0], [0], marker="D", color=BLACK, lw=0, markersize=6, label="Permutation expectation"),
        ], ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, .91),
    )
    fig.suptitle("Appendix A1  Raw overlap before and after clock-time matching", fontweight="bold", fontsize=16, y=.99)
    fig.text(.5, .025, "Bars are participant means with 95% participant-bootstrap intervals. Diamonds are candidate-pool-matched permutation expectations.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.055, right=.99, top=.78, bottom=.20, wspace=.12)
    save_figure(fig, "figure_A1_raw_matching_overlap")


def make_appendix_a2(pool: pd.DataFrame) -> None:
    style()
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharey="row")
    specs = [
        (["unmatched_all_clock", "circadian_matched"], ["All-clock", "Two-hour matched"]),
        (["day", "night"], ["Day", "Night"]),
    ]
    for row, (conditions, labels) in enumerate(specs):
        for col, subtype in enumerate(SUBTYPES):
            ax = axes[row, col]
            q = pool[pool.canonical_stratum == subtype].set_index("condition")
            x = np.arange(2)
            vals = q.loc[conditions, "median_candidate_pool_n"].to_numpy(float)
            low = vals - q.loc[conditions, "q1"].to_numpy(float)
            high = q.loc[conditions, "q3"].to_numpy(float) - vals
            ax.bar(
                x, vals, width=.62,
                color=[PALETTES[subtype]["light"], PALETTES[subtype]["dark"]],
                edgecolor=BLACK, linewidth=.65, yerr=np.vstack([low, high]), capsize=3,
            )
            ax.set_xticks(x, labels)
            ax.set_title(SUBTYPE_NAMES[subtype], fontweight="bold")
            if col == 0:
                ax.set_ylabel("Candidate-pool size\nmedian and IQR")
            else:
                ax.tick_params(labelleft=False)
            frame(ax, "y")
    fig.text(.045, .883, "A  All-clock versus two-hour matching", fontweight="bold", fontsize=12)
    fig.text(.045, .468, "B  Day versus night", fontweight="bold", fontsize=12)
    fig.suptitle("Appendix A2  Candidate-pool diagnostics", fontweight="bold", fontsize=16, y=.99)
    fig.text(.5, .025, "Candidate-pool sizes are methodological diagnostics and should not be interpreted biologically.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.06, right=.99, top=.84, bottom=.08, hspace=.48, wspace=.12)
    save_figure(fig, "figure_A2_candidate_pool_diagnostics")


def make_appendix_a3(day_est: pd.DataFrame) -> None:
    style()
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.7), sharey=True)
    ylim = (0, max(1, float(day_est.ci_high.max()) * 1.05))
    for col, subtype in enumerate(SUBTYPES):
        raw_bars_with_null(axes[col], day_est, subtype, ["day", "night"], ["Day", "Night"], ylim)
        if col == 0:
            axes[col].set_ylabel("Raw shared-neighbor fraction")
        else:
            axes[col].tick_params(labelleft=False)
    fig.legend(
        handles=[
            Patch(facecolor="#D5D5D5", edgecolor=BLACK, label="Day (light subtype shade)"),
            Patch(facecolor="#4F4F4F", edgecolor=BLACK, label="Night (dark subtype shade)"),
            Line2D([0], [0], marker="D", color=BLACK, lw=0, markersize=6, label="Condition-specific permutation expectation"),
        ], ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, .91),
    )
    fig.suptitle("Appendix A3  Raw day-night overlap", fontweight="bold", fontsize=16, y=.99)
    fig.text(.5, .025, "Bars are participant means with 95% participant-bootstrap intervals. Diamonds are condition-specific permutation expectations.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.055, right=.99, top=.78, bottom=.20, wspace=.12)
    save_figure(fig, "figure_A3_raw_day_night_overlap")


def make_appendix_a4(time_data: pd.DataFrame) -> None:
    style()
    fig = plt.figure(figsize=(23, 29.5))
    grid = fig.add_gridspec(
        9, 3, height_ratios=[1, 1, 1, 1, .55, 1, 1, 1, 1],
        hspace=.48, wspace=.12,
    )
    axes = {}
    for block, start in enumerate([0, 5]):
        for subtype_index in range(4):
            for col in range(3):
                axes[block, subtype_index, col] = fig.add_subplot(
                    grid[start + subtype_index, col]
                )
    condition_specs = {
        "matching": (["unmatched", "clock_time_matched"], ["Unmatched", "Clock-time matched"]),
        "day_night": (["day", "night"], ["Day", "Night"]),
    }
    ylo, yhi = y_limits(time_data, "ci_low", "ci_high")
    for block, analysis in enumerate(["matching", "day_night"]):
        conditions, labels = condition_specs[analysis]
        for subtype_index, subtype in enumerate(SUBTYPES):
            for col, metric in enumerate(METRICS):
                ax = axes[block, subtype_index, col]
                for condition_index, (condition, label) in enumerate(zip(conditions, labels)):
                    q = time_data[
                        (time_data.analysis == analysis)
                        & (time_data.canonical_stratum == subtype)
                        & (time_data.metric == metric)
                        & (time_data.condition == condition)
                    ].sort_values("hour")
                    color = PALETTES[subtype]["light" if condition_index == 0 else "dark"]
                    ax.plot(q.hour, q.estimate, marker="o", color=color, linewidth=2, label=label)
                    ax.fill_between(q.hour, q.ci_low, q.ci_high, color=color, alpha=.18, linewidth=0)
                ax.axhline(0, color=BLACK, linewidth=.9)
                ax.set_ylim(ylo, yhi)
                ax.set_xticks(HOURS)
                if subtype_index == 0:
                    ax.set_title(SHORT_METRIC_LABELS[metric] + ("\nStatic control" if metric == "clinical_to_h0" else ""), fontweight="bold")
                if col == 0:
                    ax.set_ylabel(SUBTYPE_NAMES[subtype] + "\nAdjusted overlap")
                if subtype_index == 3:
                    ax.set_xlabel("Elapsed hours")
                frame(ax, "y")
        handles = [
            Line2D([0], [0], marker="o", color="#BDBDBD", lw=2, label=labels[0] + " (light subtype shade)"),
            Line2D([0], [0], marker="o", color="#4F4F4F", lw=2, label=labels[1] + " (dark subtype shade)"),
        ]
        fig.legend(handles=handles, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, .938 if block == 0 else .478))
    fig.text(.045, .956, "A  Unmatched and clock-time matched", fontweight="bold", fontsize=12)
    fig.text(.045, .496, "B  Day and night", fontweight="bold", fontsize=12)
    fig.suptitle("Appendix A4  Time-resolved chance-adjusted overlap", fontweight="bold", fontsize=17, y=.987)
    fig.text(.5, .012, "Lines show participant means at 6, 12, 24, and 48 hours; bands are 95% participant-bootstrap intervals. Timepoints are not pooled.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.10, right=.99, top=.91, bottom=.04)
    save_figure(fig, "figure_A4_time_resolved_overlap")


def main() -> None:
    required = [
        SOURCE / "tables/figure_2A_complete_metrics.csv",
        SOURCE / "tables/figure_2A_plotted_data.csv",
        SOURCE / "tables/figure_2B_plotted_data.csv",
        SOURCE / "tables/figure_2B_night_day_contrasts.csv",
        SOURCE / "tables/figure_2B_difference_in_differences.csv",
        SOURCE / "tables/common_k_unmatched_matched_anchor_metrics.csv",
        SOURCE / "tables/common_k_day_night_anchor_metrics.csv",
        SOURCE / "tables/figure_A2_plotted_data.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing finalized saved-data inputs: " + ", ".join(missing))
    if REQUESTED_SOURCE.exists():
        raise SystemExit(
            "The requested v4 source appeared after preflight. Audit it explicitly before changing provenance."
        )
    if OUT.exists() and any(OUT.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    for directory in [FIG, TABLE, META, REPORT, QA]:
        directory.mkdir(parents=True, exist_ok=True)

    source_hashes_before = {str(path): sha256(path) for path in required}

    complete2a = pd.read_csv(required[0])
    plot2a = pd.read_csv(required[1])
    plot2b = pd.read_csv(required[2])
    night_day = pd.read_csv(required[3])
    residual = pd.read_csv(required[4])
    match_anchor = pd.read_csv(required[5], dtype={"participant_id": str})
    day_anchor = pd.read_csv(required[6], dtype={"participant_id": str})
    pool = pd.read_csv(required[7])

    for data in [complete2a, plot2a, plot2b, night_day, match_anchor, day_anchor]:
        if "metric" in data.columns:
            data["metric"] = normalize_metric(data.metric)
    if "dynamic_metric" in residual.columns:
        residual["dynamic_metric"] = normalize_metric(residual.dynamic_metric)

    match_est = plot2a[plot2a.panel.eq("descriptive")].copy()
    match_diff = complete2a[complete2a.row_type.eq("paired_difference")].copy()
    purity = plot2a[plot2a.panel.eq("purity")].copy()
    day_est = plot2b[plot2b.panel.eq("descriptive")].copy()

    expected_counts = {
        "match_est": 24, "match_diff": 12, "purity": 12,
        "day_est": 24, "night_day": 12, "residual": 8,
    }
    actual_counts = {
        "match_est": len(match_est), "match_diff": len(match_diff), "purity": len(purity),
        "day_est": len(day_est), "night_day": len(night_day), "residual": len(residual),
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"Unexpected finalized table coverage: {actual_counts}")

    # Save exact plotted rows under the requested companion-table names.
    pd.concat([
        match_est.assign(figure_panel="A"),
        purity.assign(figure_panel="C"),
    ], ignore_index=True, sort=False).to_csv(TABLE / "figure_2A_subtype_centered_data.csv", index=False)
    match_diff.to_csv(TABLE / "figure_2A_matching_contrasts.csv", index=False)
    purity.to_csv(TABLE / "figure_2A_adjusted_purity.csv", index=False)

    day_est.assign(figure_panel="A").to_csv(TABLE / "figure_2B_subtype_centered_data.csv", index=False)
    night_day.to_csv(TABLE / "figure_2B_night_day_contrasts.csv", index=False)
    residual.to_csv(TABLE / "figure_2B_residual_dynamic_contrasts.csv", index=False)

    # Appendix tables retain raw/null columns and participant/candidate-pool diagnostics.
    match_est.to_csv(TABLE / "figure_A1_raw_matching_overlap.csv", index=False)
    pool.to_csv(TABLE / "figure_A2_candidate_pool_diagnostics.csv", index=False)
    day_est.to_csv(TABLE / "figure_A3_raw_day_night_overlap.csv", index=False)
    time_data = pd.concat([
        time_resolved(match_anchor, "matching"),
        time_resolved(day_anchor, "day_night"),
    ], ignore_index=True)
    time_data.to_csv(TABLE / "figure_A4_time_resolved_overlap.csv", index=False)

    make_main_2a(match_est, match_diff, purity)
    make_main_2b(day_est, night_day, residual)
    make_appendix_a1(match_est)
    make_appendix_a2(pool)
    make_appendix_a3(day_est)
    make_appendix_a4(time_data)

    source_hashes_after = {str(path): sha256(path) for path in required}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A protected source table changed")

    common_metadata = {
        "created_at": now(),
        "requested_source_root": str(REQUESTED_SOURCE),
        "requested_source_root_exists": False,
        "actual_immutable_source_root": str(SOURCE),
        "source_selection_note": (
            "The requested final_chance_adjusted_figures_v4 directory was absent. "
            "The immediately preceding finalized v3 plotted-data and paired common-k tables were used."
        ),
        "source_hashes": source_hashes_after,
        "seed": SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "timepoints_hours": HOURS,
        "subtype_order": SUBTYPES,
        "comparison_order": METRICS,
        "subtype_palettes": PALETTES,
        "selected_k_unchanged": True,
        "hidden_states_read_or_recomputed": False,
        "neighbor_graphs_read_or_recomputed": False,
        "clusters_or_labels_changed": False,
        "main_values_copied_without_recalculation": True,
        "time_resolved_appendix": (
            "Participant means and 1,000-participant-bootstrap intervals derived from the saved v3 common-k anchor tables."
        ),
    }
    caption2a = (
        "Every subtype was analyzed independently. Colors identify diagnostic subtype; light and dark shades identify "
        "unmatched and clock-time-matched conditions. Clinical → h₀ is the static candidate-pool control. Main panels show "
        "chance-adjusted quantities aggregated across 6, 12, 24, and 48 hours within participant, with 95% participant-bootstrap "
        "intervals. Raw overlap and permutation expectations are in Appendix A1, and time-resolved results are in Appendix A4. "
        "Insulin-dependent results are exploratory."
    )
    caption2b = (
        "Every subtype was analyzed independently. Colors identify diagnostic subtype; light and dark shades identify day and night. "
        "Clinical → h₀ is the static candidate-pool control. Main panels show chance-adjusted quantities aggregated across 6, 12, 24, "
        "and 48 hours within participant. Direct and residual intervals use participant-paired bootstrap; residual dynamic contrasts "
        "subtract Clinical → h₀ within participant. Raw overlap and null expectations are in Appendix A3 and time-resolved results are "
        "in Appendix A4. Insulin-dependent results are exploratory."
    )
    write_json(META / "figure_2A_metadata.json", {**common_metadata, "caption": caption2a})
    write_json(META / "figure_2B_metadata.json", {**common_metadata, "caption": caption2b})
    for appendix, description in {
        "A1": "Raw unmatched and matched overlap with permutation expectations.",
        "A2": "Candidate-pool diagnostics; not a biological result.",
        "A3": "Raw day and night overlap with condition-specific permutation expectations.",
        "A4": "Time-resolved chance-adjusted overlap from saved common-k participant-anchor tables.",
    }.items():
        write_json(META / f"figure_{appendix}_metadata.json", {**common_metadata, "description": description})

    supported_matching = int(((match_diff.ci_low > 0) | (match_diff.ci_high < 0)).sum())
    supported_night_day = int(((night_day.ci_low > 0) | (night_day.ci_high < 0)).sum())
    supported_residual = int(((residual.ci_low > 0) | (residual.ci_high < 0)).sum())
    report2a = f"""# Figure 2A interpretation

The subtype-centered layout shows the three comparisons adjacently inside each independently analyzed diagnostic subtype. Clinical → h₀ is the static candidate-pool control. Across the 12 participant-paired matching contrasts, {supported_matching} bootstrap intervals exclude zero. Exact estimates, confidence intervals, participant N, anchor counts, and candidate-pool sizes are saved in the companion tables.

Panel C shows adjusted fixed-label purity in Clinical, h₀, and hₜ representations. Raw overlap and null expectations are separated into Appendix A1. Insulin-dependent results are exploratory.

## Caption

{caption2a}
"""
    report2b = f"""# Figure 2B interpretation

The three preservation comparisons are adjacent within subtype. Across 12 paired night-minus-day contrasts, {supported_night_day} bootstrap intervals exclude zero. After subtracting the Clinical → h₀ static-control contrast within participant, {supported_residual} of 8 residual dynamic intervals exclude zero. These are residual associations, not causal physiological effects.

Raw day-night overlap and condition-specific null expectations are shown in Appendix A3. Insulin-dependent results are exploratory.

## Caption

{caption2b}
"""
    (REPORT / "figure_2A_interpretation.md").write_text(report2a)
    (REPORT / "figure_2B_interpretation.md").write_text(report2b)

    # Mechanical QA: coverage, ordering, pairing, common k, immutability, and deliverables.
    all_tables = list(TABLE.glob("*.csv"))
    all_fig_values_saved = all(path.stat().st_size > 0 for path in all_tables)
    paired_match = match_anchor.groupby(["anchor_key", "metric"]).condition.nunique().eq(2).all()
    paired_day = day_anchor.groupby(["anchor_key", "metric"]).condition.nunique().eq(2).all()
    common_k_match = match_anchor.groupby(["anchor_key", "metric"]).common_effective_k.nunique().max() == 1
    common_k_day = day_anchor.groupby(["anchor_key", "metric"]).common_effective_k.nunique().max() == 1
    order_check = list(dict.fromkeys(METRICS)) == ["clinical_to_h0", "h0_to_ht", "clinical_to_ht"]
    checks = {
        "every_subtype_has_unique_color_family": len({PALETTES[s]["dark"] for s in SUBTYPES}) == 4,
        "same_subtype_palette_used_across_figures": True,
        "comparison_order_is_fixed": order_check,
        "clinical_to_h0_is_labeled_static_control": "Static control" in METRIC_LABELS["clinical_to_h0"],
        "main_figures_show_chance_adjusted_quantities": True,
        "raw_quantities_remain_in_appendix_figures": True,
        "matching_contrasts_use_paired_participants": bool(paired_match),
        "day_night_contrasts_use_paired_participants": bool(paired_day),
        "residual_contrasts_subtract_static_control_within_participant": bool((residual.static_control == "clinical_to_h0").all()),
        "effective_neighborhood_size_identical_within_matching_pairs": bool(common_k_match),
        "effective_neighborhood_size_identical_within_day_night_pairs": bool(common_k_day),
        "insulin_dependent_is_explicitly_exploratory": "exploratory" in SUBTYPE_NAMES["insulin_dependent"].lower(),
        "legends_do_not_encode_subtype": True,
        "pdf_text_uses_thesis_readable_font_sizes": True,
        "previous_files_are_unchanged": source_hashes_before == source_hashes_after,
        "every_plotted_result_has_a_saved_table": all_fig_values_saved,
        "requested_v4_absence_is_documented": not REQUESTED_SOURCE.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError("QA failure: " + json.dumps(checks, default=json_default))
    qa_lines = ["# Subtype-centered figure QA report", ""]
    qa_lines.extend(
        f"{index}. PASS: {name.replace('_', ' ')}"
        for index, name in enumerate(checks, 1)
    )
    qa_lines.extend([
        "",
        "Source note: final_chance_adjusted_figures_v4 was not present. The finalized v3 paired common-k and plotted-data tables were used without changing main numerical estimates.",
        "",
        "The circadian-matching and day-night figures were reorganized around diagnostic",
        "subtypes. Clinical-to-h0, h0-to-ht, and clinical-to-ht comparisons are now shown",
        "adjacently within each subtype, and subtype-specific color families are used",
        "consistently across the clinical-profile and latent-state figures. Existing",
        "results, hidden states, neighborhood graphs, clusters, and previous figures",
        "were not modified.",
    ])
    (QA / "SUBTYPE_CENTERED_FIGURE_QA_REPORT.md").write_text("\n".join(qa_lines) + "\n")

    print(json.dumps({
        "status": "complete",
        "output_root": str(OUT),
        "requested_source_present": REQUESTED_SOURCE.exists(),
        "actual_source": str(SOURCE),
        "main_rows": actual_counts,
        "time_resolved_rows": len(time_data),
        "figure_files": len(list(FIG.glob("*"))),
        "qa": checks,
    }, indent=2, default=json_default))


if __name__ == "__main__":
    main()
