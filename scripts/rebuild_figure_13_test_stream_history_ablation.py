#!/usr/bin/env python3
"""Re-render figure_13_test_stream_history_ablation.png using the exact same
palette and style rules as figure_8_exercise_result_summary.png.

Standalone: only reads outputs/exercise_detector_model/history_ablation_full_stream/
test/metrics_by_distance.csv (already exported; no checkpoint needed).

The original script (scripts/generate_test_history_ablation_figure.py, no
longer present on disk, restored from git history for reference) used its
own three-color set (navy, a cyan-teal, a mustard gold: #003366 / #5BBABA /
#C58B00) that did not match the palette figure_8 uses (NAVY #003366 / TEAL
#1C9A91 / CRIMSON #C83E4D, drawn from scripts/figure_style_project.py's
STRAIN_COLORS + NAVY constants). This version swaps in that exact palette
and the matching whitegrid/four-spine axis style so the two figures read as
one consistent set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/exercise_detector_model"
METRICS = BASE / "history_ablation_full_stream/test/metrics_by_distance.csv"
FIG_DIR = BASE / "final_figures"

# Same palette as figure_8_exercise_result_summary.png
# (scripts/figure_style_project.py / generate_final_exercise_figures.py).
NAVY = "#003366"
TEAL = "#1C9A91"
CRIMSON = "#C83E4D"
GRAY = "#888888"
FRAME = "#000000"

LABELS = {
    "canonical": "Canonical",
    "detector_normal": "Detector\nnormal",
    "detector_recent_history_zero": "Detector\nhistory zero",
}
ORDER = ["canonical", "detector_normal", "detector_recent_history_zero"]
COLORS = [NAVY, TEAL, CRIMSON]
DISTANCE_ORDER = ["during_exercise", "0_2h_after_exercise", "far_from_episode"]
DISTANCE_LABELS = ["During\nexercise", "0-2 h after\nexercise", ">2 h / far"]


def apply_project_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": FRAME,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME)
        spine.set_linewidth(0.8)


def bars(ax, data, value, low, high, title, ylabel, ylim=None):
    data = data.set_index("condition").loc[ORDER].reset_index()
    x = np.arange(len(data))
    vals = data[value].to_numpy(float)
    lows = data[low].to_numpy(float)
    highs = data[high].to_numpy(float)
    ax.bar(x, vals, color=COLORS, width=0.58, edgecolor="white", zorder=3)
    ax.errorbar(
        x, vals, yerr=[vals - lows, highs - vals], fmt="none",
        ecolor="#111111", capsize=5, capthick=1.5, zorder=4,
    )
    for xi, val in zip(x, vals):
        ax.text(
            xi, val + (0.14 if value == "mae_mgdl" else 0.08), f"{val:.2f}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold",
        )
    ax.set_xticks(x, [LABELS[c] for c in ORDER])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    finish_axes(ax)


def grouped_lines(ax, data, value, low, high, title, ylabel):
    for condition, color in zip(ORDER, COLORS):
        subset = (
            data[data.condition.eq(condition)]
            .set_index("distance_from_exercise")
            .loc[DISTANCE_ORDER]
            .reset_index()
        )
        x = np.arange(len(subset))
        vals = subset[value].to_numpy(float)
        lows = subset[low].to_numpy(float)
        highs = subset[high].to_numpy(float)
        ax.errorbar(
            x, vals, yerr=[vals - lows, highs - vals], color=color,
            marker="o", lw=2, capsize=4, label=LABELS[condition].replace("\n", " "),
        )
    ax.set_xticks(np.arange(3), DISTANCE_LABELS)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7.8, frameon=True)
    finish_axes(ax)


def main() -> None:
    apply_project_style()
    metrics = pd.read_csv(METRICS)
    overall = metrics[metrics.distance_from_exercise.eq("all")]
    strata = metrics[~metrics.distance_from_exercise.eq("all")]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))
    bars(
        axes[0, 0], overall, "mae_mgdl", "mae_ci_lower", "mae_ci_upper",
        "A. Final test forecast accuracy", "MAE (mg/dL)", (0, 13.5),
    )
    bars(
        axes[0, 1], overall, "bias_mgdl_prediction_minus_observed",
        "bias_ci_lower", "bias_ci_upper", "B. Final test forecast bias",
        "Bias: q50 prediction - observed (mg/dL)", (-3.5, 0.3),
    )
    axes[0, 1].axhline(0, color="#111111", lw=1)
    grouped_lines(
        axes[1, 0], strata, "mae_mgdl", "mae_ci_lower", "mae_ci_upper",
        "C. MAE by distance from exercise", "MAE (mg/dL)",
    )
    grouped_lines(
        axes[1, 1], strata, "bias_mgdl_prediction_minus_observed",
        "bias_ci_lower", "bias_ci_upper", "D. Bias by distance from exercise",
        "Bias (mg/dL)",
    )
    axes[1, 1].axhline(0, color="#111111", lw=1)
    fig.suptitle(
        "Final test-stream detector comparison and recent-history ablation",
        fontsize=16, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.02,
        "221 held-out test participants; 1,861,356 forecast rows. Error bars are "
        "participant-cluster bootstrap 95% CIs. History-zero sets "
        "recent_exercise_30min/60min/120min to raw zero.",
        ha="center", va="top", fontsize=9, color="#666666",
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIG_DIR / f"figure_13_test_stream_history_ablation.{suffix}",
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)
    print(FIG_DIR / "figure_13_test_stream_history_ablation.png")


if __name__ == "__main__":
    main()
