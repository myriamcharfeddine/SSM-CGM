"""F5: Dose-sensitivity diagnostic -- 1-panel.

Plots wearable-arm delta_theta at model-space doses 1.0 (native) and 6.25 (matched
to glucose-defined arm) for both operating points.

IMPORTANT: This is explicitly a dose-sensitivity diagnostic, NOT a physiological
dose-response relationship. The approximately linear change between doses 1.0 and 6.25
does not establish a physiologically valid dose-response; dose 6.25 is outside the
[0,1] training distribution for the wearable label (binary, std=1 preprocessor).

Values from:
  native run: block3_gff_eval/block3_60min_operating_point_comparison.csv
  dose 6.25:  block3_gff_eval/dose_matched_6.25/block3_60min_operating_point_comparison.csv
  delta_obs:  block3_gff_eval/natural_experiment_effects.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from report_style import (
    C_EVENT, C_GRAY, C_LEAKY, C_NAVY, C_WEARABLE,
    apply_style, despine, save_fig,
)

apply_style()

NATIVE_OP = ROOT / "outputs/aireadi_stream_scenario_aware_wearable_prerise_score_v1/block3_gff_eval/block3_60min_operating_point_comparison.csv"
DOSE_OP   = ROOT / "outputs/aireadi_stream_scenario_aware_wearable_prerise_score_v1/block3_gff_eval/dose_matched_6.25/block3_60min_operating_point_comparison.csv"
NE_PATH   = ROOT / "outputs/aireadi_stream_scenario_aware_wearable_prerise_score_v1/block3_gff_eval/natural_experiment_effects.csv"

nat = pd.read_csv(NATIVE_OP)
dos = pd.read_csv(DOSE_OP)
ne  = pd.read_csv(NE_PATH)

SETUPS = ["wearable_plus_shape", "wearable_only"]
SETUP_LABELS = {"wearable_plus_shape": "Wearable + shape (11.5 FP/d)",
                "wearable_only":       "Wearable only (4.9 FP/d)"}
DOSES = [1.0, 6.25]
COLORS = {"wearable_plus_shape": C_WEARABLE, "wearable_only": "#72CECE"}

delta_obs_mean = float(ne["delta_obs"].mean())
ne_lo = float(ne["ci_lo"].min())
ne_hi = float(ne["ci_hi"].max())

fig, ax = plt.subplots(figsize=(7, 5))

for setup in SETUPS:
    theta_native = float(nat.loc[nat["setup"] == setup, "delta_theta"].values[0])
    theta_dose   = float(dos.loc[dos["setup"] == setup, "delta_theta"].values[0])
    thetas = [theta_native, theta_dose]
    color = COLORS[setup]
    ax.plot(DOSES, thetas, "-o", color=color, linewidth=2, markersize=8,
            label=SETUP_LABELS[setup], zorder=3)
    for dose, theta in zip(DOSES, thetas):
        ax.annotate(f"{theta:.2f}", xy=(dose, theta),
                    xytext=(8, 0), textcoords="offset points",
                    fontweight="bold", fontsize=9, color=color, va="center")

# delta_obs reference band
ax.axhspan(ne_lo, ne_hi, alpha=0.10, color=C_NAVY, zorder=0)
ax.axhline(delta_obs_mean, ls="--", lw=2, color=C_NAVY, zorder=2,
           label=f"Observed delta_obs = +{delta_obs_mean:.1f} mg/dL")
ax.text(6.28, delta_obs_mean + 0.3, f"+{delta_obs_mean:.1f}", color=C_NAVY,
        fontweight="bold", fontsize=9, va="bottom")

ax.axhline(0, ls="-", lw=0.8, color="black", alpha=0.3)

# annotate: dose does not rescue sign
ax.text(0.50, 0.35,
        "Dose 6.25 is 6x outside the training\ndistribution (wearable label is binary [0,1]).\n"
        "Linear scaling is a diagnostic check,\nNOT a physiological dose-response.\n"
        "Sign remains negative at all doses.",
        transform=ax.transAxes, ha="center", va="center", fontsize=8,
        color=C_EVENT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_EVENT, alpha=0.85))

ax.set_xlabel("Model-space meal-on dose\n(1.0 = native wearable; 6.25 = matched to glucose-defined arm)")
ax.set_ylabel("Mean delta_theta at 60 min (mg/dL)")
ax.set_title(
    "F5: Dose-sensitivity diagnostic -- wearable null scales linearly more negative;\n"
    "dose does not rescue sign agreement, ruling out under-dosing as explanation",
    fontweight="bold",
)
ax.set_xticks(DOSES)
ax.set_xticklabels(["1.0\n(native)", "6.25\n(matched dose)"])
ax.legend(fontsize=8, loc="lower left")
despine(ax)

out = Path(__file__).parent / "f5_dose_sensitivity.pdf"
save_fig(fig, str(out))
plt.close(fig)
print("[F5] done.")
