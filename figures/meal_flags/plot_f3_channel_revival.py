"""F3: Scenario-channel revival across two arms -- 2-panel.

Panel T (top): val_meal_G_ff by epoch for glucose-defined arm (crimson)
               and wearable arm (teal). Dead-channel baseline -0.0037 as gray dashed line.
Panel B (bottom): val_mae60_fo by epoch for both arms.
                  Hard MAE60 guard limit 14.568 as red dotted line.

All values from training_history.csv artifacts and block3_pass_fail.json.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from report_style import (
    C_EVENT, C_GRAY, C_LEAKY, C_NAVY, C_WEARABLE,
    apply_style, despine, save_fig,
)

apply_style()

MEAL_HIST  = ROOT / "outputs/aireadi_stream_scenario_aware_meal_v1/metrics/training_history.csv"
WEAR_HIST  = ROOT / "outputs/aireadi_stream_scenario_aware_wearable_prerise_score_v1/metrics/training_history.csv"
MEAL_PF    = ROOT / "outputs/aireadi_stream_scenario_aware_meal_v1/block3_pass_fail.json"
WEAR_PF    = ROOT / "outputs/aireadi_stream_scenario_aware_wearable_prerise_score_v1/block3_pass_fail.json"

mh = pd.read_csv(MEAL_HIST)
wh = pd.read_csv(WEAR_HIST)

with open(MEAL_PF) as f: mpf = json.load(f)
with open(WEAR_PF) as f: wpf = json.load(f)

DEAD_CHANNEL_GFF = float(mpf.get("gff_meal_proxy_event_baseline", -0.0037))
MAE60_LIMIT = float(wpf.get("mae60_hard_limit",
                              wpf.get("mae60_fo_all_baseline", 14.418) * 1.15))

print(f"[F3] dead-channel G_ff baseline: {DEAD_CHANNEL_GFF:.4f}")
print(f"[F3] MAE60 guard limit: {MAE60_LIMIT:.3f}")
print("[F3] meal arm epochs:", mh["epoch"].tolist())
print("[F3] wear arm epochs:", wh["epoch"].tolist())

fig, (axT, axB) = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
fig.subplots_adjust(hspace=0.4)

# ---- top panel: val_meal_G_ff ----
mh_valid = mh[mh["val_meal_G_ff"].notna() & np.isfinite(mh["val_meal_G_ff"])]
wh_valid = wh[wh["val_meal_G_ff"].notna() & np.isfinite(wh["val_meal_G_ff"])]

axT.plot(mh_valid["epoch"], mh_valid["val_meal_G_ff"], "-o", color=C_LEAKY,
         linewidth=2, markersize=5, label="Glucose-defined arm", zorder=3)
axT.plot(wh_valid["epoch"], wh_valid["val_meal_G_ff"], "-s", color=C_WEARABLE,
         linewidth=2, markersize=5, label="Wearable pre-rise arm", zorder=3)

axT.axhline(DEAD_CHANNEL_GFF, ls="--", lw=1.5, color=C_GRAY,
            label=f"Dead-channel baseline G_ff = {DEAD_CHANNEL_GFF:.4f}")
axT.axhline(0, ls="-", lw=0.8, color="black", alpha=0.3)

# annotate final revived values
m_final = float(mh_valid.iloc[-1]["val_meal_G_ff"])
w_final = float(wh_valid["val_meal_G_ff"].dropna().iloc[-1])
axT.annotate(f"+{m_final:.3f}", xy=(float(mh_valid.iloc[-1]["epoch"]), m_final),
             xytext=(5, 4), textcoords="offset points",
             fontweight="bold", fontsize=8, color=C_LEAKY)
axT.annotate(f"+{w_final:.3f}", xy=(float(wh_valid.iloc[-1]["epoch"]), w_final),
             xytext=(5, -12), textcoords="offset points",
             fontweight="bold", fontsize=8, color=C_WEARABLE)

axT.set_ylabel("Validation meal G_ff\n(factual-future gain on meal windows)")
axT.set_title(
    "Both arms revive G_ff from near-zero;\nglucose arm stronger (0.64 vs 0.42)",
    fontweight="bold",
)
axT.legend(fontsize=8)
despine(axT)

# ---- bottom panel: val_mae60_fo ----
mh_mae = mh[mh["val_mae60_fo"].notna() & np.isfinite(mh["val_mae60_fo"])]
wh_mae = wh[wh["val_mae60_fo"].notna() & np.isfinite(wh["val_mae60_fo"])]

axB.plot(mh_mae["epoch"], mh_mae["val_mae60_fo"], "-o", color=C_LEAKY,
         linewidth=2, markersize=5, label="Glucose-defined arm", zorder=3)
axB.plot(wh_mae["epoch"], wh_mae["val_mae60_fo"], "-s", color=C_WEARABLE,
         linewidth=2, markersize=5, label="Wearable pre-rise arm", zorder=3)

axB.axhline(MAE60_LIMIT, ls=":", lw=2, color=C_EVENT,
            label=f"Forecast guard limit {MAE60_LIMIT:.2f} mg/dL")
axB.text(mh_mae["epoch"].max(), MAE60_LIMIT + 0.03,
         f"guard limit {MAE60_LIMIT:.2f}", color=C_EVENT, fontsize=7.5, ha="right")

# mark guard warnings (epochs that exceed the limit)
for th, color, arm_h in [(mh_mae, C_LEAKY, "meal"), (wh_mae, C_WEARABLE, "wearable")]:
    warn = th[th["val_mae60_fo"] > MAE60_LIMIT]
    if not warn.empty:
        axB.scatter(warn["epoch"], warn["val_mae60_fo"], marker="x",
                    color=color, s=60, zorder=5, linewidths=2,
                    label=f"{arm_h} guard warn")

axB.set_xlabel("Training epoch (resuming from 5-epoch base)")
axB.set_ylabel("Validation MAE60 -- forecast only (mg/dL)")
axB.set_title(
    "MAE60 guard: wearable arm stays below limit;\nglucose arm degrades after epoch 11",
    fontweight="bold",
)
axB.legend(fontsize=8)
despine(axB)

fig.suptitle(
    "F3: Scenario-channel revival -- both arms recover G_ff from the dead-channel baseline;\n"
    "forecast guard separates checkpoint selection from overfitting",
    fontweight="bold", fontsize=10, y=1.01,
)

out = Path(__file__).parent / "f3_channel_revival.pdf"
save_fig(fig, str(out))
plt.close(fig)
print("[F3] done.")
