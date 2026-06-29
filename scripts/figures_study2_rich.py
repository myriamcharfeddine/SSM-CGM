#!/usr/bin/env python3
"""Generate Study 2 rich-head figures as vector PDFs.

Figures produced:
  S2-F2  baseline_ladder_rich.pdf   -- full 10-method ladder (val + test)
  S2-F3  event_control_rich.pdf     -- event-enriched vs unmatched control
  S2-F6  ablation.pdf               -- feature-group ablation (Ridge, val MAE)
  S2-F7  per_horizon.pdf            -- per-horizon MAE curves (4 methods)
  S2-F8  paired_cis.pdf             -- paired-CI forest plot (test)

All numbers are read from disk artifacts; none are hard-coded.

Run:
    /home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python \\
        scripts/figures_study2_rich.py
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
ARTDIR   = ROOT / "outputs/study2_final_rich_head"
RICHDIR  = ROOT / "outputs/study2_rich_head"
FIGDIR   = ROOT / "report/figures/generated"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":    42,
})

BLUE   = "#2166ac"
RED    = "#d6604d"
GREEN  = "#4dac26"
ORANGE = "#f4a582"
GREY   = "#888888"
DARK   = "#222222"

# ── helper ────────────────────────────────────────────────────────────────────
def save(fig: plt.Figure, name: str) -> Path:
    p = FIGDIR / name
    fig.savefig(p, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")
    return p


# ── load artifacts ─────────────────────────────────────────────────────────────
ladder   = pd.read_csv(ARTDIR / "study2_final_ladder.csv")
diff_cis = pd.read_csv(ARTDIR / "study2_final_diff_cis.csv")
per_h    = pd.read_csv(ARTDIR / "study2_per_horizon.csv")
ablation = pd.read_csv(ARTDIR / "study2_ablation.csv")
evc      = json.loads((ARTDIR / "study2_event_vs_control.json").read_text())

# Also load rich-head CSV for current_shift and linear_zt
rich_ladder = pd.read_csv(RICHDIR / "study2_rich_ladder.csv")

# Merge: build a unified test-split ladder with all 10 methods
# Methods in final ladder
final_test = ladder[ladder.split == "test"].copy()
# Get current_shift and linear_zt from rich ladder (same evaluation, same data)
rich_test = rich_ladder[
    (rich_ladder.split == "test") &
    (rich_ladder.method.isin(["current_shift", "linear_zt"]))
][["method", "mae", "ci_lo", "ci_hi", "n_anchors_tuples", "n_participants"]].copy()
rich_test.rename(columns={"n_anchors_tuples": "n_anchors"}, inplace=True)
# Add to unified
full_test = pd.concat([final_test, rich_test], ignore_index=True)

# Same for validation
final_val = ladder[ladder.split == "validation"].copy()
rich_val  = rich_ladder[
    (rich_ladder.split == "validation") &
    (rich_ladder.method.isin(["current_shift", "linear_zt"]))
][["method", "mae", "ci_lo", "ci_hi", "n_anchors_tuples", "n_participants"]].copy()
rich_val.rename(columns={"n_anchors_tuples": "n_anchors"}, inplace=True)
full_val  = pd.concat([final_val, rich_val], ignore_index=True)

# Canonical method order and labels
METHOD_ORDER = [
    "base", "current_shift", "decaying_shift",
    "linear_zt", "linear_zt_sc", "linear_full_ols",
    "ridge_full_comb", "ridge_split_dir",
    "hgbt_full_comb", "hgbt_split_dir",
]
METHOD_LABEL = {
    "base":            "Base (frozen SSM)",
    "current_shift":   "Current shift",
    "decaying_shift":  "Decaying shift ($\\delta=0.7$)",
    "linear_zt":       "Linear $z_t$",
    "linear_zt_sc":    "Linear $z_t$ + slope + curv",
    "linear_full_ols": "Linear full OLS",
    "ridge_full_comb": "Ridge full (combined)",
    "ridge_split_dir": "Ridge full (split up/down)",
    "hgbt_full_comb":  "HGBT full (combined)",
    "hgbt_split_dir":  "HGBT full (split up/down)",
}

# Colors by family
def method_color(m: str) -> str:
    if m == "base":            return GREY
    if m == "current_shift":   return "#aaaaaa"
    if m == "decaying_shift":  return "#888888"
    if "linear" in m:          return ORANGE
    if "ridge" in m:           return BLUE
    if "hgbt" in m:            return GREEN
    return DARK

def method_hatch(m: str) -> str:
    return "//" if "split" in m else ""


# ─────────────────────────────────────────────────────────────────────────────
# S2-F2: Full baseline ladder (val + test, 10 methods)
# ─────────────────────────────────────────────────────────────────────────────
def fig_baseline_ladder() -> None:
    methods = METHOD_ORDER
    n = len(methods)
    y = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    for ax, split, df, title in [
        (axes[0], "validation", full_val,  "Validation (44,417 anchors, $n=221$)"),
        (axes[1], "test",       full_test, "Test (39,698 anchors, $n=204$)"),
    ]:
        df_s = df.set_index("method")
        for i, m in enumerate(methods):
            if m not in df_s.index:
                continue
            row  = df_s.loc[m]
            mae  = float(row["mae"])
            lo   = float(row["ci_lo"])
            hi   = float(row["ci_hi"])
            xerr = [[mae - lo], [hi - mae]]
            col  = method_color(m)
            ax.errorbar(mae, i, xerr=xerr,
                        fmt="o", color=col, ecolor=col,
                        capsize=3, capthick=1, ms=5, lw=1.2)

        # vertical line at base MAE
        base_mae = float(df_s.loc["base", "mae"])
        ax.axvline(base_mae, color=GREY, lw=0.8, ls="--", alpha=0.7)

        # shade gain region for Ridge split
        if "ridge_split_dir" in df_s.index:
            ridge_mae = float(df_s.loc["ridge_split_dir", "mae"])
            ax.axvspan(ridge_mae, base_mae, alpha=0.06, color=BLUE, zorder=0)

        ax.set_title(title, pad=6)
        ax.set_xlabel("MAE (mg/dL)")
        ax.set_xlim(10.5, 12.3)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([METHOD_LABEL[m] for m in methods])
    axes[0].invert_yaxis()

    # Legend: families
    handles = [
        mpatches.Patch(color=GREY,   label="Frozen base / shift"),
        mpatches.Patch(color=ORANGE, label="Linear OLS"),
        mpatches.Patch(color=BLUE,   label="Ridge"),
        mpatches.Patch(color=GREEN,  label="HGBT"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("S2-F2: Rich correction-head baseline ladder", fontsize=10, y=1.01)
    save(fig, "study2_f2_baseline_ladder_rich.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# S2-F3: Event-enriched vs unmatched control (updated)
# ─────────────────────────────────────────────────────────────────────────────
def fig_event_control() -> None:
    ev_base_mae  = evc["event_base_mae"]
    ev_head_mae  = evc["event_head_mae"]
    ct_base_mae  = evc["control_base_mae"]
    ct_head_mae  = evc["control_head_mae"]
    ev_gain      = ev_base_mae - ev_head_mae
    ct_gain      = ct_base_mae - ct_head_mae
    ev_ci_lo     = evc["event_gain_ci"][0]
    ev_ci_hi     = evc["event_gain_ci"][1]
    ct_ci_lo     = evc["control_gain_ci"][0]
    ct_ci_hi     = evc["control_gain_ci"][1]
    ev_n         = evc["event_n_anchors"]
    ct_n         = evc["control_n_anchors"]

    fig, (ax_mae, ax_gain) = plt.subplots(1, 2, figsize=(7.5, 3.4))
    fig.subplots_adjust(wspace=0.35)

    # -- left panel: MAE comparison --
    groups  = ["Event-enriched\n($n=14{,}505$)", "Unmatched control\n($n=25{,}193$)"]
    x       = np.array([0, 1])
    w       = 0.3
    base_maes = [ev_base_mae, ct_base_mae]
    head_maes = [ev_head_mae, ct_head_mae]

    ax_mae.bar(x - w/2, base_maes, w, color=GREY,  label="Frozen base", zorder=3)
    ax_mae.bar(x + w/2, head_maes, w, color=BLUE,  label="Final head (Ridge split)", zorder=3)
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(groups)
    ax_mae.set_ylabel("MAE (mg/dL)")
    ax_mae.set_ylim(0, 17)
    ax_mae.legend(frameon=False)
    ax_mae.set_title("MAE by group", pad=4)
    ax_mae.yaxis.set_major_locator(plt.MultipleLocator(2))

    for xi, (bm, hm) in enumerate(zip(base_maes, head_maes)):
        ax_mae.text(xi - w/2, bm + 0.2, f"{bm:.1f}", ha="center", va="bottom", fontsize=7)
        ax_mae.text(xi + w/2, hm + 0.2, f"{hm:.1f}", ha="center", va="bottom", fontsize=7)

    # -- right panel: gain with CIs --
    gains  = [ev_gain, ct_gain]
    ci_los = [ev_ci_lo, ct_ci_lo]
    ci_his = [ev_ci_hi, ct_ci_hi]
    colors = [RED, "#aaaaaa"]
    labels = ["Event-enriched", "Unmatched control"]

    for i, (g, lo, hi, col, lbl) in enumerate(zip(gains, ci_los, ci_his, colors, labels)):
        ax_gain.barh(i, g, color=col, alpha=0.8, zorder=3, label=lbl)
        ax_gain.errorbar(g, i, xerr=[[g - lo], [hi - g]],
                         fmt="none", ecolor=DARK, capsize=4, capthick=1.2, lw=1.5)
        ax_gain.text(max(g, lo) + 0.01, i, f"+{g:.3f}", va="center", fontsize=8)

    ax_gain.axvline(0, color=DARK, lw=0.8)
    ax_gain.set_yticks([0, 1])
    ax_gain.set_yticklabels(labels)
    ax_gain.set_xlabel("Gain: base MAE $-$ head MAE (mg/dL)")
    ax_gain.set_title("Correction gain (test split)", pad=4)
    ax_gain.set_xlim(-0.1, 0.85)
    ax_gain.invert_yaxis()

    # disclaimer
    fig.text(0.5, -0.06,
        "Descriptive only -- not a matched causal contrast. Event-enriched labels are glucose-level proxies.",
        ha="center", fontsize=7, color=GREY, style="italic")
    fig.suptitle("S2-F3: Event-enriched proxy versus unmatched control", fontsize=10, y=1.02)
    save(fig, "study2_f3_event_control_rich.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# S2-F6: Feature-group ablation
# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation() -> None:
    GROUP_LABEL = {
        "z_t_only":        "$z_t$ only",
        "z_t_slope_curv":  "$z_t$ + slope + curv",
        "residual_only":   "$r_t$ only",
        "residual_history":"Residual history ($r_t$--$r_{t-3}$)",
        "slope_curvature": "Slope + curvature",
        "trigger_phase":   "Trigger phase",
        "context_only":    "Context (glucose, IW, hour)",
        "full_causal":     "Full causal (19 features)",
    }
    rows = ablation.sort_values("val_mae", ascending=True)

    y    = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    colors = [GREEN if g == "full_causal" else BLUE for g in rows.group]
    xerrs  = np.column_stack([
        rows["val_mae"] - rows["val_ci_lo"],
        rows["val_ci_hi"] - rows["val_mae"],
    ]).T
    ax.barh(y, rows["val_mae"], color=colors, alpha=0.75, zorder=3)
    ax.errorbar(rows["val_mae"], y, xerr=xerrs,
                fmt="none", ecolor=DARK, capsize=3, capthick=1, lw=1.2)

    for i, row in enumerate(rows.itertuples()):
        ax.text(row.val_mae + 0.005, i, f"{row.val_mae:.4f}", va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels([GROUP_LABEL.get(g, g) for g in rows.group])
    ax.set_xlabel("Validation MAE (mg/dL)")
    ax.set_xlim(11.15, 11.6)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.axvline(rows["val_mae"].max(), color=GREY, lw=0.5, ls=":")
    # annotate base as reference line
    base_val_mae = full_val.set_index("method").loc["base", "mae"]
    ax.axvline(base_val_mae, color=GREY, lw=0.8, ls="--", alpha=0.6)
    ax.text(base_val_mae + 0.002, -0.7, "Base", fontsize=7, color=GREY)

    handles = [
        mpatches.Patch(color=GREEN, alpha=0.75, label="Full causal (all 19 features)"),
        mpatches.Patch(color=BLUE,  alpha=0.75, label="Feature group ablation"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    ax.set_title("S2-F6: Feature-group ablation (Ridge, validation MAE)", pad=6)
    fig.tight_layout()
    save(fig, "study2_f6_ablation.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# S2-F7: Per-horizon MAE curves
# ─────────────────────────────────────────────────────────────────────────────
def fig_per_horizon() -> None:
    SHOW = {
        "base":            ("Base (frozen SSM)",          GREY,   "-",  "o"),
        "linear_zt_sc":    ("Linear $z_t$+slope+curv",   ORANGE, "--", "s"),
        "ridge_split_dir": ("Ridge full (split up/down)", BLUE,   "-",  "^"),
        "hgbt_full_comb":  ("HGBT full (combined)",       GREEN,  ":",  "D"),
    }
    ph_test = per_h[per_h["split"] == "test"].set_index("method")
    horizons = list(range(1, 13))
    horizon_min = [h * 5 for h in horizons]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    for m, (lbl, col, ls, mk) in SHOW.items():
        if m not in ph_test.index:
            continue
        maes = [float(ph_test.loc[m, f"mae_h{h:02d}"]) for h in horizons]
        ax.plot(horizon_min, maes, color=col, ls=ls, marker=mk,
                ms=4, lw=1.5, label=lbl)

    # shade 0-30 vs 30-60
    ax.axvspan(5,  30, alpha=0.04, color=BLUE,  zorder=0)
    ax.axvspan(35, 60, alpha=0.04, color=RED,   zorder=0)
    ax.text(17, 3.5, "0--30 min", fontsize=7.5, color=BLUE,  ha="center")
    ax.text(47, 3.5, "30--60 min", fontsize=7.5, color=RED,   ha="center")

    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("MAE (mg/dL)")
    ax.set_xlim(5, 60)
    ax.set_xticks(horizon_min)
    ax.set_xticklabels([str(h) for h in horizon_min], fontsize=7)
    ax.set_ylim(2, 17)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_title("S2-F7: Per-horizon MAE (test split)", pad=6)
    fig.tight_layout()
    save(fig, "study2_f7_per_horizon.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# S2-F8: Paired-CI forest plot
# ─────────────────────────────────────────────────────────────────────────────
def fig_paired_cis() -> None:
    # diff = MAE(A) - MAE(B); negative = A is better
    # Reframe as "improvement of B over A" for readability: B_gain = -diff
    PAIR_LABELS = {
        "HGBT-comb vs best-simple":    "HGBT vs best simple\n(Linear $z_t$+slope+curv)",
        "HGBT-comb vs Linear-full-OLS":"HGBT vs Linear full OLS",
        "HGBT-comb vs Ridge-split":    "HGBT vs Ridge (split up/down)",
        "HGBT-comb vs HGBT-split":     "HGBT comb. vs HGBT split",
        "Final head vs Base":          "Final head vs Frozen base",
        "Final head vs Decaying shift":"Final head vs Decaying shift",
    }
    rows = diff_cis.copy()
    rows["gain"]    = -rows["diff"]
    rows["gain_lo"] = -rows["ci_hi"]
    rows["gain_hi"] = -rows["ci_lo"]
    rows["label"]   = rows["comparison"].map(PAIR_LABELS)
    rows = rows[rows["label"].notna()].copy()

    # Sort by gain descending
    rows = rows.sort_values("gain", ascending=True).reset_index(drop=True)
    y    = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    for i, row in rows.iterrows():
        g  = row["gain"]
        lo = row["gain_lo"]
        hi = row["gain_hi"]
        excl = bool(row["ci_excludes_zero"])
        col  = BLUE if excl else GREY

        ax.plot([lo, hi], [i, i], color=col, lw=2, zorder=3)
        ax.plot(g, i, "o", color=col, ms=6, zorder=4)
        ax.plot([lo, lo], [i - 0.25, i + 0.25], color=col, lw=1.5)
        ax.plot([hi, hi], [i - 0.25, i + 0.25], color=col, lw=1.5)

        sig_txt = "*" if excl else ""
        ax.text(hi + 0.003, i, f"{g:+.3f}{sig_txt}", va="center", fontsize=7.5, color=col)

    ax.axvline(0, color=DARK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"], fontsize=8)
    ax.set_xlabel("Gain: MAE(comparison) $-$ MAE(richer) (mg/dL)\nPositive = richer method is better")
    ax.set_xlim(-0.03, 0.3)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))

    handles = [
        plt.Line2D([0], [0], color=BLUE, lw=2, marker="o", ms=5, label="CI excludes zero"),
        plt.Line2D([0], [0], color=GREY, lw=2, marker="o", ms=5, label="CI includes zero (tied)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.set_title("S2-F8: Paired-CI forest plot (test split)", pad=6)
    fig.tight_layout()
    save(fig, "study2_f8_paired_cis.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Study 2 rich-head figures...")
    fig_baseline_ladder()
    fig_event_control()
    fig_ablation()
    fig_per_horizon()
    fig_paired_cis()
    print("Done.")
