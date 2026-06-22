"""F4: Counterfactual A/B headline -- 2-panel.

Panel L: delta_theta at 60 min for glucose-defined arm and wearable arm
         at both operating points, with delta_obs (+22) reference band.
Panel R: glycemic-state gradient (hyper / in_range / hypo) for both arms.

Glucose arm values from: wearable_cf_readiness/block3_60min_operating_point_comparison.csv
                         wearable_cf_readiness/block3_60min_glycemic_state_breakdown_derived.csv
Wearable arm values from: block3_gff_eval/block3_60min_operating_point_comparison.csv
                          block3_gff_eval/block3_60min_glycemic_state_breakdown.csv
delta_obs reference: natural_experiment_effects.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from report_style import (
    C_EVENT, C_GRAY, C_LEAKY, C_NAVY, C_WEARABLE,
    apply_style, despine, save_fig,
)

apply_style()

MEAL_OP  = ROOT / "outputs/wearable_cf_readiness/block3_60min_operating_point_comparison.csv"
WEAR_OP  = ROOT / "outputs/aireadi_stream_scenario_aware_wearable_prerise_score_v1/block3_gff_eval/block3_60min_operating_point_comparison.csv"
MEAL_GS  = ROOT / "outputs/wearable_cf_readiness/block3_60min_glycemic_state_breakdown_derived.csv"
WEAR_GS  = ROOT / "outputs/aireadi_stream_scenario_aware_wearable_prerise_score_v1/block3_gff_eval/block3_60min_glycemic_state_breakdown.csv"
NE_PATH  = ROOT / "outputs/aireadi_stream_scenario_aware_wearable_prerise_score_v1/block3_gff_eval/natural_experiment_effects.csv"

mo = pd.read_csv(MEAL_OP)
wo = pd.read_csv(WEAR_OP)
mg = pd.read_csv(MEAL_GS)
wg = pd.read_csv(WEAR_GS)
ne = pd.read_csv(NE_PATH)

SETUPS = ["wearable_plus_shape", "wearable_only"]
SETUP_LABELS = {"wearable_plus_shape": "w+shape\n(11.5 FP/d)", "wearable_only": "w-only\n(4.9 FP/d)"}
STATES = ["hyper", "in_range", "hypo"]
STATE_LABELS = {"hyper": "Hyper\n(>180)", "in_range": "In-range\n(70-180)", "hypo": "Hypo\n(<70)"}

delta_obs_mean = float(ne["delta_obs"].mean())  # ~22.2

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.subplots_adjust(wspace=0.40)

# ---- Panel L: delta_theta at both operating points ----
x = np.arange(len(SETUPS))
width = 0.28

# glucose arm
meal_theta = [float(mo.loc[mo["setup"] == s, "delta_theta"].values[0]) for s in SETUPS]
meal_ci_lo = [float(mo.loc[mo["setup"] == s, "ci_lo_theta"].values[0]) for s in SETUPS]
meal_ci_hi = [float(mo.loc[mo["setup"] == s, "ci_hi_theta"].values[0]) for s in SETUPS]
meal_err_lo = [t - l for t, l in zip(meal_theta, meal_ci_lo)]
meal_err_hi = [h - t for t, h in zip(meal_theta, meal_ci_hi)]

# wearable arm
wear_theta = [float(wo.loc[wo["setup"] == s, "delta_theta"].values[0]) for s in SETUPS]
wear_ci_lo = [float(wo.loc[wo["setup"] == s, "ci_lo_theta"].values[0]) for s in SETUPS]
wear_ci_hi = [float(wo.loc[wo["setup"] == s, "ci_hi_theta"].values[0]) for s in SETUPS]
wear_err_lo = [t - l for t, l in zip(wear_theta, wear_ci_lo)]
wear_err_hi = [h - t for t, h in zip(wear_theta, wear_ci_hi)]

bars_m = axL.bar(x - width / 2, meal_theta, width,
                 color=C_LEAKY, edgecolor="white", linewidth=0.5, label="Glucose-defined arm")
bars_w = axL.bar(x + width / 2, wear_theta, width,
                 color=C_WEARABLE, edgecolor="white", linewidth=0.5, label="Wearable pre-rise arm")

axL.errorbar(x - width / 2, meal_theta,
             yerr=[meal_err_lo, meal_err_hi],
             fmt="none", ecolor="black", elinewidth=1.2, capsize=4, zorder=5)
axL.errorbar(x + width / 2, wear_theta,
             yerr=[wear_err_lo, wear_err_hi],
             fmt="none", ecolor="black", elinewidth=1.2, capsize=4, zorder=5)

# annotate values
for xi, (mv, wv) in enumerate(zip(meal_theta, wear_theta)):
    axL.text(xi - width / 2, mv - 1.2, f"{mv:.1f}", ha="center", va="top",
             fontweight="bold", fontsize=8, color="white")
    axL.text(xi + width / 2, wv - 0.008, f"{wv:.2f}", ha="center", va="top",
             fontweight="bold", fontsize=8, color=C_WEARABLE)

# delta_obs reference band
ne_lo = float(ne["ci_lo"].min())
ne_hi = float(ne["ci_hi"].max())
axL.axhspan(ne_lo, ne_hi, alpha=0.12, color=C_NAVY, zorder=0)
axL.axhline(delta_obs_mean, ls="--", lw=2, color=C_NAVY, zorder=2,
            label=f"Observed delta_obs = +{delta_obs_mean:.1f} mg/dL")
axL.text(len(SETUPS) - 0.3, delta_obs_mean + 0.5,
         f"+{delta_obs_mean:.1f}", color=C_NAVY, fontweight="bold", fontsize=9)

# sign-failure annotation
axL.axhline(0, ls="-", lw=0.8, color="black", alpha=0.3)
axL.text(0.5, 3, "SIGN FAILURE: both arms negative\nvs observed +22 (p=non-sig at any dose)",
         transform=axL.transAxes, ha="center", va="bottom", fontsize=8,
         color=C_EVENT, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_EVENT, alpha=0.8))

axL.set_xticks(x)
axL.set_xticklabels([SETUP_LABELS[s] for s in SETUPS])
axL.set_ylabel("delta_theta at 60 min (mg/dL)")
axL.set_title(
    "Both arms fail sign agreement vs +22 observed;\nglucose arm inverted (-35), wearable null (-0.21)",
    fontweight="bold",
)
axL.legend(fontsize=8, loc="lower right")
despine(axL)

# ---- Panel R: glycemic-state gradient for both arms ----
x2 = np.arange(len(STATES))
width2 = 0.28

# glucose arm (per setup, average of the two setups for display clarity)
for si, (setup, alpha_mult) in enumerate([("wearable_plus_shape", 1.0), ("wearable_only", 0.65)]):
    mg_sub = mg[mg["setup"] == setup]
    wg_sub = wg[wg["setup"] == setup]
    m_vals = [float(mg_sub.loc[mg_sub["glycemic_state"] == s, "delta_theta_mean"].values[0])
               for s in STATES]
    w_vals = [float(wg_sub.loc[wg_sub["glycemic_state"] == s, "delta_theta"].values[0])
               for s in STATES]
    lw = 2.0 if si == 0 else 1.0
    ls_m = "-o" if si == 0 else "--o"
    ls_w = "-s" if si == 0 else "--s"
    slabel = "w+shape" if si == 0 else "w-only"
    axR.plot(x2, m_vals, ls_m, color=C_LEAKY, linewidth=lw, markersize=6,
             alpha=alpha_mult, label=f"Glucose arm ({slabel})")
    axR.plot(x2, w_vals, ls_w, color=C_WEARABLE, linewidth=lw, markersize=6,
             alpha=alpha_mult, label=f"Wearable arm ({slabel})")

# delta_obs band
axR.axhspan(ne_lo, ne_hi, alpha=0.10, color=C_NAVY, zorder=0)
axR.axhline(delta_obs_mean, ls="--", lw=1.5, color=C_NAVY, zorder=2,
            label=f"delta_obs +{delta_obs_mean:.1f}")
axR.axhline(0, ls="-", lw=0.8, color="black", alpha=0.3)

# annotate the 30-50x magnitude difference
axR.annotate("30-50x\nmagnitude\ndifference", xy=(1, -0.20),
             xytext=(1.5, -12), fontsize=8, color="black",
             arrowprops=dict(arrowstyle="->", color="black", lw=1),
             fontweight="bold")

axR.set_xticks(x2)
axR.set_xticklabels([STATE_LABELS[s] for s in STATES])
axR.set_ylabel("Mean delta_theta at 60 min (mg/dL)")
axR.set_title(
    "Glycemic-state gradient preserved in both arms;\n30-50x magnitude difference (crimson vs teal scale)",
    fontweight="bold",
)
axR.legend(fontsize=7, loc="upper right", ncol=2)
despine(axR)

fig.suptitle(
    "F4: Counterfactual A/B -- both arms fail sign agreement with the +22 observed effect;\n"
    "glucose arm strongly inverted, wearable arm near-zero null",
    fontweight="bold", fontsize=10, y=1.02,
)

out = Path(__file__).parent / "f4_cf_ab_headline.pdf"
save_fig(fig, str(out))
plt.close(fig)
print("[F4] done.")
