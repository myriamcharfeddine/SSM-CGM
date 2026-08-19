#!/usr/bin/env python3
"""AI-READI interpretability v2, Phase 1 figures.

These four figures were specified in the original Phase 1 brief but never
generated (Phase 1 only produced corrected CSVs). Built here purely from
those existing CSVs, no GPU pass. Follows the project figure design style
guide (scripts/interpret_airedi_style_v2.py).

fig_interp_global_lomo_v2:      11-group LOMO, raw sensitivity + delta MAE
fig_interp_global_ht_v2:        6-group rigorous h_t joint norm
fig_interp_raw_vs_normalized_v2: post-hoc vs rigorous raw norm, the correct
                                 field for discussing overcounting, kept
                                 separate from the normalized-share ratio
fig_interp_state_vs_output_v2:  scatter, only the 6 groups common to both
                                 methods, rigorous h_t share (x) vs six-group
                                 renormalized LOMO share (y), axes explained
                                 as different concepts

Usage:
  python scripts/interpret_airedi_global_figures_v2.py
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
    ANNOTATION_COLOR, GROUP_COLORS_V2, NEGATIVE_COLOR, POSITIVE_COLOR, annotate_above_axes,
    apply_whitegrid_style, pretty_label, set_title, signed_bar_colors, style_legend,
)

OUT_DIR = ROOT / "outputs/interpretability_v2"


def _err_bars(df, lo_col, hi_col, val_col):
    return [df[val_col] - df[lo_col], df[hi_col] - df[val_col]]


def fig_global_lomo():
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
    set_title(ax, "Output level reliance, all 11 groups")

    ax = axes[1]
    bar_colors = signed_bar_colors(df["delta_mae_mgdl"])
    ax.barh(y, df["delta_mae_mgdl"], xerr=_err_bars(df, "delta_mae_ci_low", "delta_mae_ci_high", "delta_mae_mgdl"),
           color=bar_colors, edgecolor="white", capsize=3, zorder=3)
    ax.axvline(0, color="black", linewidth=1.0, zorder=4)
    apply_whitegrid_style(ax, axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels([], fontsize=1)
    ax.set_xlabel("Delta MAE when ablated (mg/dL)")
    set_title(ax, "Predictive usefulness, all 11 groups")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=POSITIVE_COLOR, label="Worsens MAE (predictively useful)"),
               Patch(facecolor=NEGATIVE_COLOR, label="Does not worsen MAE")]
    ax.legend(handles=handles, fontsize=7, loc="lower right", frameon=True, facecolor="white",
             edgecolor="black", framealpha=1.0)

    fig.tight_layout()
    base = OUT_DIR / "fig_interp_global_lomo_v2"
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {base}.pdf")


def fig_global_ht():
    df = pd.read_csv(OUT_DIR / "group_overall_rigorous_v2.csv").sort_values("normalized_share")
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    cols = [GROUP_COLORS_V2.get(g, "#888888") for g in df["group"]]
    y = np.arange(len(df))
    frac_lo = df["raw_joint_norm_ci_low"] / df["raw_joint_norm"] * df["normalized_share"]
    frac_hi = df["raw_joint_norm_ci_high"] / df["raw_joint_norm"] * df["normalized_share"]
    ax.barh(y, df["normalized_share"] * 100,
           xerr=[(df["normalized_share"] - frac_lo) * 100, (frac_hi - df["normalized_share"]) * 100],
           color=cols, edgecolor="white", capsize=3, zorder=3)
    apply_whitegrid_style(ax, axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_label(g) for g in df["group"]], fontsize=9)
    ax.set_xlabel("Rigorous h_t share (%), 6 h_t-reachable groups")
    set_title(ax, "State construction attribution, 6 groups")
    fig.tight_layout()
    base = OUT_DIR / "fig_interp_global_ht_v2"
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {base}.pdf")


def fig_raw_vs_normalized():
    df = pd.read_csv(OUT_DIR / "group_overall_compare_v2.csv").sort_values("raw_norm_ratio_posthoc_over_rigorous")
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    cols = [GROUP_COLORS_V2.get(g, "#888888") for g in df["group"]]
    y = np.arange(len(df))
    ax.barh(y, df["raw_norm_ratio_posthoc_over_rigorous"], color=cols, edgecolor="white", zorder=3)
    ax.axvline(1.0, color="black", linewidth=1.0, zorder=4)
    apply_whitegrid_style(ax, axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_label(g) for g in df["group"]], fontsize=9)
    ax.set_xlabel("Post hoc raw norm divided by rigorous raw norm")
    set_title(ax, "Collinearity overcount, raw norm ratio")
    ax.annotate("Ratio of 1 means no overcounting, this is the correct field for discussing overcounting",
               xy=(0.98, 0.04), xycoords="axes fraction", color=ANNOTATION_COLOR, fontsize=7, ha="right")
    fig.tight_layout()
    base = OUT_DIR / "fig_interp_raw_vs_normalized_v2"
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {base}.pdf")


def fig_state_vs_output():
    rig = pd.read_csv(OUT_DIR / "group_overall_rigorous_v2.csv").set_index("group")["normalized_share"]
    lomo_full = pd.read_csv(OUT_DIR / "group_overall_lomo_v2.csv").set_index("group")["raw_mean_abs_delta_mgdl"]
    six_groups = list(rig.index)
    lomo_six = lomo_full.reindex(six_groups)
    lomo_six_renorm = lomo_six / lomo_six.sum()

    # manual label offsets (points), tuned to this fixed 6-group layout to avoid collisions
    # in the low-share cluster (sleep, wearable activity / exercise, stress all sit close together)
    label_offsets = {
        "glucose history": (8, 6),
        "heart rate / physiology": (10, -4),
        "data quality / missingness": (-14, 12),
        "sleep": (10, 12),
        "wearable activity / exercise": (12, -14),
        "stress": (-70, -10),
    }

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for g in six_groups:
        x, y = rig[g] * 100, lomo_six_renorm[g] * 100
        ax.scatter(x, y, color=GROUP_COLORS_V2.get(g, "#888888"), s=110, edgecolor="black", linewidth=0.6, zorder=3)
        dx, dy = label_offsets.get(g, (6, 4))
        ax.annotate(pretty_label(g), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
                   arrowprops=dict(arrowstyle="-", color=ANNOTATION_COLOR, linewidth=0.6) if abs(dx) + abs(dy) > 16 else None)
    lims = [0, max(rig.max(), lomo_six_renorm.max()) * 100 * 1.25]
    ax.plot(lims, lims, color=ANNOTATION_COLOR, linewidth=1.0, linestyle=":", zorder=1)
    ax.annotate("Equal share reference", xy=(lims[1] * 0.6, lims[1] * 0.63), color=ANNOTATION_COLOR,
               fontsize=7, rotation=38)
    apply_whitegrid_style(ax)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Rigorous h_t share (%), state construction")
    ax.set_ylabel("LOMO output share (%), renormalized over the same 6 groups")
    set_title(ax, "Where state construction and output reliance diverge", pad=16)
    annotate_above_axes(
        ax, "Points below the line contribute more strongly to constructing h_t than they change the final output",
        fontsize=7,
    )
    fig.tight_layout()
    base = OUT_DIR / "fig_interp_state_vs_output_v2"
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {base}.pdf")


if __name__ == "__main__":
    fig_global_lomo()
    fig_global_ht()
    fig_raw_vs_normalized()
    fig_state_vs_output()
