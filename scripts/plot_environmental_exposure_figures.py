#!/usr/bin/env python3
"""Environmental exposure figure pair, redrawn to the project figure design
style guide (scripts/interpret_airedi_style_v2.py is that guide's executable
form -- this script imports it rather than restating colors/fonts).

Figure 1, "forecasting value": does adding bedroom humidity as a residual-head
input to the frozen SSMCGM-Stream base model clear the forecasting-value gate
at the 30-minute early-wake endpoint? Source numbers: outputs/env_residual_head_results.csv
(variant=base_plus_env / base_plus_env_shuffled_control, head_type=hgbt,
forecast_window=early_wake, horizon_minutes=30, split=test) and
outputs/env_forecasting_decision.json (0.3 mg/dL candidate-gate threshold).

Figure 2, "humidity association": the within-participant association between
bedroom humidity and overnight glucose range, plus its shuffled-night and
+24h-shifted negative controls. Source numbers: outputs/env_negative_controls.csv
and outputs/env_association_summary.md Section 6 (the one FDR-surviving pair,
hum_mean -> overnight_glucose_range).

Usage:
  python scripts/plot_environmental_exposure_figures.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.interpret_airedi_style_v2 import (
    ANNOTATION_COLOR, EVENT_COLOR, GRAY, NAVY, OBSERVED_COLOR, apply_whitegrid_style,
    sentence_case, set_title, signed_bar_colors,
)

OUT_DIR = ROOT / "outputs/environmental_exposure"

# -- figure 1: forecasting-value gate ----------------------------------------
FORECAST_ENDPOINT_LABEL = "Test MAE gain over frozen base (mg/dL), 30 min early-wake endpoint"
FORECAST_GATE_THRESHOLD_MGDL = 0.3
FORECAST_ROWS = [
    # label, mae_gain, ci_low, ci_high
    ("Frozen base", 0.0, 0.0, 0.0),
    ("Base + humidity", 0.04572908584410662, -0.061049422473107386, 0.15537585033233348),
    ("Base + shuffled humidity", 0.0664869773361021, -0.05109080512996904, 0.1791066038987501),
]

# -- figure 2: bedroom humidity association ----------------------------------
ASSOCIATION_X_LABEL = "Within-participant coefficient (mg/dL glucose range per 1 pct-point humidity)"
ASSOCIATION_ROWS = [
    # label, coefficient, ci_low, ci_high, is_primary
    ("Same night humidity", -0.36129516149439045, -0.5997702778544985, -0.11849169455952054, True),
    ("Shuffled night control", 0.2081087248492962, -0.016738491868710233, 0.4441859703453086, False),
    ("+24 h shifted control", -0.08616142282023484, -0.35387896586583584, 0.16426186155661485, False),
]


def _err_bars(rows, val_idx=1, lo_idx=2, hi_idx=3):
    vals = np.array([r[val_idx] for r in rows])
    los = np.array([r[lo_idx] for r in rows])
    his = np.array([r[hi_idx] for r in rows])
    return [vals - los, his - vals]


def _outside_legend_box(ax, text):
    """Small annotation box anchored just outside the axes, near the
    x/y-axis origin corner (bottom-left) -- never on top of plotted data."""
    ax.text(0.0, -0.30, text, transform=ax.transAxes, ha="left", va="top",
             fontsize=7.5, color=ANNOTATION_COLOR, clip_on=False,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                        edgecolor="black", linewidth=0.6))


def fig_forecasting_value():
    rows = FORECAST_ROWS
    labels = [sentence_case(r[0]) for r in rows]
    values = [r[1] for r in rows]
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(7.8, 3.4))

    bar_rows = rows[1:]
    bar_y = y[1:]
    bar_colors = signed_bar_colors([r[1] for r in bar_rows])
    ax.barh(bar_y, [r[1] for r in bar_rows], xerr=_err_bars(bar_rows),
             color=bar_colors, edgecolor="white", height=0.55, capsize=3, zorder=3)

    for (label, val, lo, hi), yi in zip(bar_rows, bar_y):
        ax.annotate(f"{val:+.3f} [{lo:+.3f}, {hi:+.3f}]", xy=(hi, yi),
                     xytext=(6, 0), textcoords="offset points",
                     color=ANNOTATION_COLOR, fontsize=7.5, ha="left", va="center")

    ref_y = y[0]
    ax.errorbar([rows[0][1]], [ref_y], xerr=[[0], [0]], fmt="D", color=NAVY,
                 markersize=6, zorder=4)
    ax.annotate("reference", xy=(rows[0][1], ref_y), xytext=(10, 0),
                 textcoords="offset points", color=ANNOTATION_COLOR, fontsize=7.5,
                 ha="left", va="center")

    ax.axvline(0, color="black", linewidth=1.0, zorder=2)
    ax.axvline(FORECAST_GATE_THRESHOLD_MGDL, color=EVENT_COLOR, linewidth=1.4,
                linestyle="--", zorder=2)
    ax.annotate(f"{FORECAST_GATE_THRESHOLD_MGDL:.1f} mg/dL meaningful threshold",
                 xy=(FORECAST_GATE_THRESHOLD_MGDL, len(rows) - 1.15),
                 xytext=(-6, 0), textcoords="offset points",
                 color=ANNOTATION_COLOR, fontsize=7.5, ha="right", va="center")

    apply_whitegrid_style(ax, axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(sentence_case(FORECAST_ENDPOINT_LABEL), fontsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    set_title(ax, "Humidity residual head does not clear the forecasting-value gate", fontsize=12)

    _outside_legend_box(
        ax,
        "Real humidity gain is small and its CI crosses zero;\n"
        "shuffled humidity is numerically larger on this endpoint.")

    fig.tight_layout()
    base = OUT_DIR / "env_forecasting_value"
    fig.savefig(str(base) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {base}.png")
    return str(base) + ".png"


def fig_humidity_association():
    rows = ASSOCIATION_ROWS
    labels = [sentence_case(r[0]) for r in rows]
    y = np.arange(len(rows))[::-1]
    colors = [OBSERVED_COLOR if r[4] else GRAY for r in rows]

    fig, ax = plt.subplots(figsize=(7.8, 3.6))

    for row, yi, c in zip(rows, y, colors):
        lo, hi = row[2], row[3]
        ax.errorbar([row[1]], [yi], xerr=[[row[1] - lo], [hi - row[1]]], fmt="o",
                     color=c, markersize=6, elinewidth=1.6, capsize=3, zorder=3)
        ax.annotate(f"{row[1]:+.3f} [{lo:+.3f}, {hi:+.3f}]", xy=(hi, yi),
                     xytext=(6, 0), textcoords="offset points",
                     color=ANNOTATION_COLOR, fontsize=7.5, ha="left", va="center")

    ax.axvline(0, color="black", linewidth=1.0, zorder=2)

    apply_whitegrid_style(ax, axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(sentence_case(ASSOCIATION_X_LABEL), fontsize=9)
    ax.set_xlim(-0.72, 0.72)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    set_title(ax, "Bedroom humidity association with overnight glucose range", fontsize=12)

    _outside_legend_box(
        ax,
        "Primary pair: FDR q=0.0466; participant-bootstrap CI\n"
        "excludes zero. Controls: shuffled-night and +24 h shifted\n"
        "CIs include zero.")

    fig.tight_layout()
    base = OUT_DIR / "env_humidity_association"
    fig.savefig(str(base) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {base}.png")
    return str(base) + ".png"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    forecasting_png = fig_forecasting_value()
    association_png = fig_humidity_association()

    # NOTE: outputs/environment_model_trained/fig_interp_environment_accuracy_comparison_v1.png
    # is a different figure (7-group vs 6-group forecast MAE/bias) and is left untouched.
    mirrors = {
        forecasting_png: [
            ROOT / "ReportMasterThesis-overleaf/figures/generated/env_forecasting_value.png",
        ],
        association_png: [
            ROOT / "ReportMasterThesis-overleaf/figures/generated/env_humidity_association.png",
        ],
    }
    for src, dests in mirrors.items():
        for dest in dests:
            shutil.copyfile(src, dest)
            print(f"copied -> {dest}")


if __name__ == "__main__":
    main()
