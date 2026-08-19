#!/usr/bin/env python3
"""AI-READI interpretability v2, environment variant: fig_interp_global_lomo_v2
recreated with a 12th "environmental exposure" group, using numbers from the
environment-trained checkpoint (scripts/interpret_airedi_global_v2_environment.py
--execute) instead of the original 11-group model.

Exact copy of fig_global_lomo() in scripts/interpret_airedi_global_figures_v2.py
(same axes, colors, error bars, legend, sort order), pointed at
outputs/interpretability_v2_environment/group_overall_lomo_v2.csv and titled
for 12 groups instead of 11.

Usage:
  python scripts/interpret_airedi_global_figures_v2_environment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.interpret_airedi_style_v2 import (
    GROUP_COLORS_V2, NEGATIVE_COLOR, POSITIVE_COLOR, apply_whitegrid_style, pretty_label, set_title,
    signed_bar_colors,
)

OUT_DIR = ROOT / "outputs/interpretability_v2_environment"


def _err_bars(df, lo_col, hi_col, val_col):
    return [df[val_col] - df[lo_col], df[hi_col] - df[val_col]]


def fig_global_lomo_environment():
    df = pd.read_csv(OUT_DIR / "group_overall_lomo_v2.csv").sort_values("raw_mean_abs_delta_mgdl")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    cols = [GROUP_COLORS_V2.get(g, "#888888") for g in df["group"]]
    y = np.arange(len(df))

    ax = axes[0]
    ax.barh(y, df["raw_mean_abs_delta_mgdl"],
           xerr=_err_bars(df, "raw_mean_abs_delta_ci_low", "raw_mean_abs_delta_ci_high", "raw_mean_abs_delta_mgdl"),
           color=cols, edgecolor="white", capsize=3, zorder=3)
    apply_whitegrid_style(ax, axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_label(g) for g in df["group"]], fontsize=8.5)
    ax.set_xlabel("Raw forecast sensitivity (mg/dL)")
    set_title(ax, "Output level reliance, all 12 groups (environment model)")

    ax = axes[1]
    bar_colors = signed_bar_colors(df["delta_mae_mgdl"])
    ax.barh(y, df["delta_mae_mgdl"], xerr=_err_bars(df, "delta_mae_ci_low", "delta_mae_ci_high", "delta_mae_mgdl"),
           color=bar_colors, edgecolor="white", capsize=3, zorder=3)
    ax.axvline(0, color="black", linewidth=1.0, zorder=4)
    apply_whitegrid_style(ax, axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels([], fontsize=1)
    ax.set_xlabel("Delta MAE when ablated (mg/dL)")
    set_title(ax, "Predictive usefulness, all 12 groups (environment model)")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=POSITIVE_COLOR, label="Worsens MAE (predictively useful)"),
               Patch(facecolor=NEGATIVE_COLOR, label="Does not worsen MAE")]
    ax.legend(handles=handles, fontsize=7, loc="lower right", frameon=True, facecolor="white",
             edgecolor="black", framealpha=1.0)

    fig.tight_layout()
    base = OUT_DIR / "fig_interp_global_lomo_v2_environment"
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {base}.pdf")


if __name__ == "__main__":
    fig_global_lomo_environment()
