"""F2: pre-rise detector latency and operating-point tradeoff, 2-panel.

Panel L: median detection latency bars for all setups, 37-min Hochsmann wall.
Panel R: sensitivity vs FP/day scatter (operating-point tradeoff).

All values from codex_prerise_tuned_full_gate_meal_detection_latency.csv.
The 37-minute Hochsmann reference is a cited external benchmark value.
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

LATENCY_CSV = (ROOT / "outputs/no_log_scenarios/meal_transfer/feature_evaluation"
               / "codex_prerise_tuned_full_gate_meal_detection_latency.csv")

df = pd.read_csv(LATENCY_CSV)
print("[F2] latency CSV:")
print(df[["setup", "median_latency_min", "sensitivity", "fp_per_day"]].to_string(index=False))

HOCHSMANN_MIN = 37   # external reference (Hochsmann et al. 2022)

SETUP_META = {
    "wearable_plus_shape": ("Wearable + shape",  C_WEARABLE,  "o",  True),
    "wearable_only":        ("Wearable only",     "#72CECE",   "o",  True),
    "shape_only":           ("Shape only",        C_NAVY,      "s",  True),
    "D_student_prob":       ("D student prob",    C_GRAY,      "^",  False),
    "G_online_state":       ("G online state",    C_GRAY,      "^",  False),
    "H_online_full_state":  ("H online full",     C_GRAY,      "^",  False),
    "cgm_roc_2of3":         ("CGM ROC 2-of-3",   C_LEAKY,     "D",  False),
    "cgm_grid_style":       ("CGM grid-style",    "#CC6666",   "D",  False),
}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 4.8))
fig.subplots_adjust(wspace=0.42, right=0.86)

# ---- Panel L: latency bars ----
rows_for_bars = []
for _, row in df.iterrows():
    meta = SETUP_META.get(row["setup"])
    if meta is None:
        continue
    rows_for_bars.append({
        "label": meta[0],
        "color": meta[1],
        "latency": float(row["median_latency_min"]),
        "sensitivity": float(row["sensitivity"]),
        "fp_per_day": float(row["fp_per_day"]),
        "wearable": meta[3],
    })

# sort by latency
rows_for_bars.sort(key=lambda r: r["latency"])

bar_labels  = [r["label"] for r in rows_for_bars]
bar_colors  = [r["color"] for r in rows_for_bars]
bar_latency = [r["latency"] for r in rows_for_bars]

x = np.arange(len(bar_labels))
bars = axL.bar(x, bar_latency, color=bar_colors, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, bar_latency):
    axL.text(bar.get_x() + bar.get_width() / 2,
             max(bar.get_height(), 0) + 1.8,
             f"{val:.0f} min", ha="center", va="bottom", fontweight="bold", fontsize=8)
    if val == 0:
        # zero-latency bars are invisible otherwise -- mark the baseline point
        axL.scatter([bar.get_x() + bar.get_width() / 2], [0],
                    color=bar.get_facecolor(), edgecolor="black",
                    linewidths=0.8, s=45, zorder=4)

axL.axhline(HOCHSMANN_MIN, ls=":", lw=2, color=C_EVENT, zorder=3,
            label=f"Hochsmann wall {HOCHSMANN_MIN} min (CGM-only reference)")
axL.text(0.2, HOCHSMANN_MIN + 1.6, f"{HOCHSMANN_MIN} min (Hochsmann wall)",
         color=C_EVENT, fontsize=7.5, ha="left")
axL.set_xticks(x)
axL.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=7.5)
axL.set_ylabel("Median detection latency (min)\n(0 = fires at onset, negative = pre-rise)")
axL.set_title("Wearable+shape fires at 0 min median;\nbreaks the 37-min CGM-only Hochsmann wall",
              fontweight="bold")
axL.set_ylim(-2, max(bar_latency) + 8)
axL.legend(fontsize=7.5, loc="upper left")
despine(axL)

# ---- Panel R: sensitivity vs FP/day scatter ----
# Labels use a shared legend outside the plot rather than inline text: several
# points (e.g. G_online_state / H_online_full_state) share near-identical
# coordinates and inline labels collided badly on top of each other.
for _, row in df.iterrows():
    meta = SETUP_META.get(row["setup"])
    if meta is None:
        continue
    label, color, marker, _ = meta
    axR.scatter(float(row["fp_per_day"]), float(row["sensitivity"]),
                color=color, marker=marker, s=100, zorder=3,
                label=label, edgecolors="white", linewidths=0.8)

axR.set_xlabel("False positives per day")
axR.set_ylabel("Sensitivity at selected threshold")
axR.set_title("Sensitivity vs. FP/day tradeoff;\nearly detectors trade FP/day for lead time",
              fontweight="bold")
axR.margins(x=0.12, y=0.12)
axR.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5,
           frameon=True, borderaxespad=0.0, handletextpad=0.6)
despine(axR)

fig.suptitle(
    "F2: pre-rise wearable detector breaks the 37-min CGM-only wall "
    "at the cost of higher FP rate",
    fontweight="bold", fontsize=10, y=1.02,
)

out = Path(__file__).parent / "f2_prerise_detector.pdf"
save_fig(fig, str(out))
plt.close(fig)
print("[F2] done.")
