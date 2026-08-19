"""F6: Summary schematic -- three converging evidence streams.

Summarizes the three independent convergent findings that support the
boundary conclusion. Each is colored by semantic role and annotated
with the key numeric value from the corresponding artifact.

This figure summarizes an evidence boundary, not a causal proof.
The three findings are presented as consistent, not as independent proofs.

Key values loaded from artifacts:
  - Partial correlation: block3_partial_information.csv
  - Glucose-defined delta_theta: wearable_cf_readiness/block3_60min_operating_point_comparison.csv
  - Wearable delta_theta (both doses): block3_gff_eval/ files
  - delta_obs reference: natural_experiment_effects.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from report_style import (
    C_EVENT, C_GRAY, C_LEAKY, C_NAVY, C_WEARABLE,
    apply_style, despine, save_fig,
)

apply_style()

PARTIAL = ROOT / "outputs/no_log_scenarios/meal_transfer/feature_evaluation/block3_partial_information.csv"
MEAL_OP = ROOT / "outputs/wearable_cf_readiness/block3_60min_operating_point_comparison.csv"
WEAR_OP = ROOT / "outputs/aireadi_stream_scenario_aware_wearable_prerise_score_v1/block3_gff_eval/block3_60min_operating_point_comparison.csv"
DOSE_OP = ROOT / "outputs/aireadi_stream_scenario_aware_wearable_prerise_score_v1/block3_gff_eval/dose_matched_6.25/block3_60min_operating_point_comparison.csv"
NE_PATH = ROOT / "outputs/aireadi_stream_scenario_aware_wearable_prerise_score_v1/block3_gff_eval/natural_experiment_effects.csv"

par = pd.read_csv(PARTIAL)
mo  = pd.read_csv(MEAL_OP)
wo  = pd.read_csv(WEAR_OP)
dos = pd.read_csv(DOSE_OP)
ne  = pd.read_csv(NE_PATH)

# load key values
pcorr_leaky   = float(par.loc[par["setup"] == "C_legacy_bidir_teacher", "partial_corr_with_q50_residual"].values[0])
pcorr_causal  = float(par.loc[par["setup"] == "Cprime_causal_teacher",  "partial_corr_with_q50_residual"].values[0])
meal_theta    = float(mo["delta_theta"].mean())
wear_theta_n  = float(wo["delta_theta"].mean())
wear_theta_d  = float(dos["delta_theta"].mean())
delta_obs     = float(ne["delta_obs"].mean())

print(f"[F6] partial corr leaky={pcorr_leaky:.3f} causal={pcorr_causal:.3f}")
print(f"[F6] glucose arm delta_theta={meal_theta:.2f}")
print(f"[F6] wearable native={wear_theta_n:.3f}  dose6.25={wear_theta_d:.3f}")
print(f"[F6] delta_obs={delta_obs:.2f}")

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 11)
ax.set_ylim(0, 7)
ax.axis("off")

# ---- helper to draw a labeled box ----
def draw_box(ax, cx, cy, w, h, text, color, fontsize=9, textcolor="white"):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.18",
        linewidth=1.5, edgecolor=color,
        facecolor=color, alpha=0.88, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4,
            multialignment="center")

def draw_arrow(ax, x0, y0, x1, y1, color, lw=2.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=2)

# ---- TITLE ----
ax.text(5.5, 6.7, "Study 1: Meal Counterfactual Audit -- Evidence Boundary",
        ha="center", va="center", fontsize=12, fontweight="bold", color="black")

# ---- THREE SOURCE BOXES ----
# 1. Orthogonalization (gray)
draw_box(ax, 1.5, 5.0, 3.0, 1.4,
         f"Orthogonalization\nLeaky partial corr: {pcorr_leaky:.3f}\nCausal partial corr: {pcorr_causal:.3f}\nDeployable features lack\nmeal-specific signal",
         C_GRAY, fontsize=8, textcolor="white")

# 2. Glucose-defined counterfactual (crimson)
draw_box(ax, 5.5, 5.0, 3.0, 1.4,
         f"Glucose-defined arm\ndelta_theta = {meal_theta:.1f} mg/dL\n(inverted vs +{delta_obs:.1f} observed)\nLabel is mean-reversion marker,\nnot causal meal driver",
         C_LEAKY, fontsize=8)

# 3. Wearable dose-scaled null (teal)
draw_box(ax, 9.5, 5.0, 2.8, 1.4,
         f"Wearable pre-rise arm\ndelta_theta = {wear_theta_n:.2f} (dose 1.0)\n            = {wear_theta_d:.2f} (dose 6.25)\nScales more negative at\nhigher dose; sign persists",
         C_WEARABLE, fontsize=8)

# ---- THREE RULING BOXES ----
draw_box(ax, 1.5, 3.0, 2.8, 0.9,
         "Ruling: not leakage headroom\nC-C' gap is fully unreachable",
         C_GRAY, fontsize=8)

draw_box(ax, 5.5, 3.0, 2.8, 0.9,
         "Ruling: not encoding artefact\nLead preservation confirmed (Step 2)",
         C_LEAKY, fontsize=8)

draw_box(ax, 9.5, 3.0, 2.8, 0.9,
         "Ruling: not dose artefact\nDose-matched sweep stays negative",
         C_WEARABLE, fontsize=8)

# ---- ARROWS from sources to rulings ----
draw_arrow(ax, 1.5, 4.30, 1.5, 3.46, C_GRAY)
draw_arrow(ax, 5.5, 4.30, 5.5, 3.46, C_LEAKY)
draw_arrow(ax, 9.5, 4.30, 9.5, 3.46, C_WEARABLE)

# ---- ARROWS from rulings to boundary box ----
draw_arrow(ax, 2.0, 2.55, 4.5, 1.75, C_GRAY)
draw_arrow(ax, 5.5, 2.55, 5.5, 1.75, C_LEAKY)
draw_arrow(ax, 9.0, 2.55, 6.5, 1.75, C_WEARABLE)

# ---- BOUNDARY CONCLUSION BOX ----
boundary_text = (
    "BOUNDARY CONCLUSION\n"
    "AI-READI cannot support a meal counterfactual in any constructible label.\n"
    f"Observed glucose rise: delta_obs = +{delta_obs:.1f} mg/dL at 60 min.\n"
    "Both constructible labels fail sign agreement (not under-dose, not dead channel, not encoding).\n"
    "Next required step: exogenous meal annotation (e.g., T1DEXI or CGMacros dataset)."
)
rect_b = FancyBboxPatch(
    (1.2, 0.5), 9.0, 1.4,
    boxstyle="round,pad=0.22",
    linewidth=2.5, edgecolor=C_NAVY,
    facecolor=C_NAVY, alpha=0.92, zorder=3,
)
ax.add_patch(rect_b)
ax.text(5.7, 1.2, boundary_text, ha="center", va="center",
        fontsize=8.5, fontweight="bold", color="white", zorder=4,
        multialignment="center")

fig.suptitle(
    "F6: Three convergent evidence streams support the same boundary conclusion\n"
    "(Each stream is independently consistent, not an independent causal proof)",
    fontweight="bold", fontsize=10, y=0.98,
)

out = Path(__file__).parent / "f6_summary_schematic.pdf"
save_fig(fig, str(out))
plt.close(fig)
print("[F6] done.")
