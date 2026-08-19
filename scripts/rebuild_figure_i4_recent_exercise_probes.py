#!/usr/bin/env python3
"""Focused variant of figure_i4_hidden_state_probes.png: only the
recent_exercise_30/60/120min regression rows (the headline finding — the
canonical hidden state does not linearly encode recent exercise history,
the detector-informed one does), dropping classification panel A and the
current_glucose / recent_glucose_slope regression rows.

Standalone: reads the already-exported
outputs/exercise_detector_model/interpretability/hidden_state_probe_metrics.csv
(no model checkpoint needed).
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
OUT = ROOT / "outputs/exercise_detector_model/interpretability"
DATA = OUT / "hidden_state_probe_metrics.csv"

TEAL = "#1C9A91"
NAVY = "#003366"
FRAME = "#000000"

TARGETS = ["recent_exercise_30min", "recent_exercise_60min", "recent_exercise_120min"]
TARGET_LABELS = {
    "recent_exercise_30min": "Recent exercise\n(30 min)",
    "recent_exercise_60min": "Recent exercise\n(60 min)",
    "recent_exercise_120min": "Recent exercise\n(120 min)",
}
REP_ORDER = ["detector_informed_h_t", "canonical_h_t"]
REP_LABELS = {
    "detector_informed_h_t": "Detector-informed h_t",
    "canonical_h_t": "Canonical h_t",
}
REP_COLORS = {"detector_informed_h_t": TEAL, "canonical_h_t": NAVY}


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
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
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


def main() -> None:
    apply_project_style()
    data = pd.read_csv(DATA)
    data = data[data["target"].isin(TARGETS)]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    n_targets = len(TARGETS)
    bar_height = 0.36
    y_base = np.arange(n_targets)

    for offset, rep in zip((bar_height / 2, -bar_height / 2), REP_ORDER):
        values = [
            data[(data.target.eq(t)) & (data.representation.eq(rep))]["r2"].iloc[0]
            for t in TARGETS
        ]
        y = y_base + offset
        ax.barh(
            y, values, height=bar_height, color=REP_COLORS[rep],
            label=REP_LABELS[rep], edgecolor="white", zorder=3,
        )
        for yi, val in zip(y, values):
            ax.text(
                val + (0.015 if val >= 0 else -0.015),
                yi,
                f"{val:.2f}",
                ha="left" if val >= 0 else "right",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    ax.axvline(0, color="#888888", ls=":", lw=1, zorder=2)
    ax.set_yticks(y_base, [TARGET_LABELS[t] for t in TARGETS])
    ax.invert_yaxis()
    ax.set_xlim(-0.15, 0.85)
    ax.set_xlabel("Held-out R²")
    finish_axes(ax)
    fig.suptitle(
        "Recent exercise history is recoverable only from the\n"
        "detector-informed hidden state",
        fontweight="bold",
        fontsize=13,
        y=1.06,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
        ncol=2, frameon=True,
    )
    fig.text(
        0.5, -0.02,
        "Ridge regression probes of h_t on held-out test participants "
        "(n_train=1431, n_test=281). Canonical h_t never receives exercise "
        "labels and is at or below chance (R²≈0) at all three windows.",
        ha="center", va="top", fontsize=8.5, color="#666666",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT / f"figure_i4_recent_exercise_probes.{suffix}",
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)
    print(OUT / "figure_i4_recent_exercise_probes.png")


if __name__ == "__main__":
    main()
