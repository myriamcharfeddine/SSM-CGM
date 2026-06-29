#!/usr/bin/env python3
"""Generate Study 2 figures S2-F1 through S2-F5.

All figures saved as vector PDFs to report/figures/generated/.

Figures:
  S2-F1  Trigger calibration panel (3 sub-panels)
  S2-F2  Baseline ladder forest plot
  S2-F3  Event-enriched proxy vs unmatched control
  S2-F4  Triggered-window examples (4 episodes)
  S2-F5  Deployment behavior (3 sub-panels)

Run with:
    python scripts/figures_study2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
ART     = ROOT / "outputs"
FIG_DIR = ROOT / "report/figures/generated"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette (Paul Tol muted, accessible) ───────────────────────────────
C = {
    "train":          "#44AA99",
    "validation":     "#332288",
    "test":           "#CC6677",
    "event":          "#CC3311",
    "control":        "#4477AA",
    "base":           "#BBBBBB",
    "current_shift":  "#DDDDDD",
    "decaying_shift": "#88CCEE",
    "linear":         "#44AA99",
    "head":           "#CC3311",
    "shadow":         "#EEEEEE",
}
METHOD_LABEL = {
    "base":           "Base (frozen SSM)",
    "current_shift":  "Current shift",
    "decaying_shift": "Decaying shift",
    "linear":         "Linear ($z_t$)",
    "head":           "Ridge head",
}
SPLIT_LABEL = {"train": "Train", "validation": "Validation", "test": "Test"}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# S2-F1: Trigger calibration (3-panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_s2_f1() -> Path:
    diag = json.loads(
        (ART / "study2_forecast_cache_5min/diagnostics/blocks_ab_diagnostics.json").read_text()
    )
    sweep = pd.read_csv(ART / "study2_forecast_cache_5min/study2_threshold_sweep.csv")
    thresholds = json.loads(
        (ART / "study2_forecast_cache_5min/study2_selected_thresholds.json").read_text()
    )
    tau_locked = thresholds["tau_up"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    fig.subplots_adjust(wspace=0.38)

    # Panel A: 80% PI coverage by split
    ax = axes[0]
    splits = ["train", "validation", "test"]
    coverage = [
        diag["3_one_step_80pct_coverage"][s]["empirical_coverage_80pct_interval"] * 100
        for s in splits
    ]
    bar_colors = [C[s] for s in splits]
    bars = ax.bar([SPLIT_LABEL[s] for s in splits], coverage,
                  color=bar_colors, width=0.5, zorder=3)
    ax.axhline(80, color="#CC3311", lw=1.2, ls="--", label="80\\% target")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Empirical coverage (\\%)")
    ax.set_title("(a) One-step 80\\% PI coverage")
    ax.legend(frameon=False, fontsize=7)
    for bar, val in zip(bars, coverage):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f"{val:.1f}\\%", ha="center", va="bottom", fontsize=7)

    # Panel B: trigger_up_rate vs tau by split
    ax = axes[1]
    for split in splits:
        sub = sweep[sweep["split"].eq(split)].sort_values("tau")
        ax.plot(sub["tau"], sub["trigger_up_rate"] * 100,
                color=C[split], lw=1.6, label=SPLIT_LABEL[split], marker="o",
                markersize=3)
    ax.axvline(tau_locked, color="#CC3311", lw=1.2, ls="--",
               label=f"Locked $\\tau={tau_locked}$")
    ax.set_xlabel("Threshold $\\tau$")
    ax.set_ylabel("Trigger-up rate (\\%)")
    ax.set_title("(b) Trigger rate vs.\\ threshold")
    ax.legend(frameon=False, fontsize=7)

    # Panel C: trigger_up and trigger_down rates at locked tau by split
    ax = axes[2]
    sweep_locked = sweep[sweep["tau"].eq(tau_locked)]
    x = np.arange(len(splits))
    w = 0.35
    for i, (col, label, hatch) in enumerate([
        ("trigger_up_rate",   "Trigger-up",   ""),
        ("trigger_down_rate", "Trigger-down", "//"),
    ]):
        vals = [
            float(sweep_locked[sweep_locked["split"].eq(s)][col].values[0]) * 100
            for s in splits
        ]
        ax.bar(x + (i - 0.5) * w, vals, width=w,
               color=C["event"] if i == 0 else C["control"],
               hatch=hatch, label=label, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABEL[s] for s in splits])
    ax.set_ylabel("Rate (\\%)")
    ax.set_title(f"(c) Rates at locked $\\tau={tau_locked}$")
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle("S2-F1: Trigger calibration", fontsize=9, y=1.01, fontweight="bold")
    out = FIG_DIR / "study2_f1_trigger_calibration.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  S2-F1 -> {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# S2-F2: Baseline ladder forest plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_s2_f2() -> Path:
    ladder = pd.read_csv(ART / "study2_blocks_c_to_h/study2_baseline_ladder.csv")
    diff   = pd.read_csv(ART / "study2_preblock_i/study2_extended_diff_cis.csv")
    ladder = ladder[ladder["subset"].eq("all")]

    methods = ["base", "current_shift", "decaying_shift", "linear", "head"]
    splits  = ["validation", "test"]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))

    y_step   = 1.2
    group_gap = 0.35
    n_splits = len(splits)
    positions: dict[tuple, float] = {}
    yticks, ylabels = [], []

    for mi, method in enumerate(methods):
        base_y = mi * (n_splits * y_step + group_gap)
        for si, split in enumerate(splits):
            row = ladder[(ladder["method"].eq(method)) & (ladder["split"].eq(split))]
            if row.empty:
                continue
            y   = base_y + si * y_step
            mae = float(row["mae"].values[0])
            lo  = float(row["ci_lo"].values[0])
            hi  = float(row["ci_hi"].values[0])
            col = C["validation"] if split == "validation" else C["test"]
            ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(mae, y, "o", color=col, ms=5, zorder=4)
            positions[(method, split)] = (y, mae)
        ytick_y = base_y + (n_splits - 1) * y_step / 2
        yticks.append(ytick_y)
        ylabels.append(METHOD_LABEL.get(method, method))

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (mg/dL)")
    ax.set_title("S2-F2: Baseline ladder (triggered non-insulin windows)")

    # Legend
    patch_val  = mpatches.Patch(color=C["validation"], label="Validation")
    patch_test = mpatches.Patch(color=C["test"],       label="Test")
    ax.legend(handles=[patch_val, patch_test], frameon=False, loc="lower right", fontsize=7)

    # Annotate linear-head difference (test)
    lh_row  = diff[(diff["split"].eq("test")) & (diff["comparison"].eq("linear_minus_head"))]
    lh_diff = float(lh_row["diff"].values[0])
    lh_lo   = float(lh_row["ci_lo"].values[0])
    lh_hi   = float(lh_row["ci_hi"].values[0])
    if ("linear", "test") in positions and ("head", "test") in positions:
        y_lin, x_lin = positions[("linear", "test")]
        y_hd,  x_hd  = positions[("head",   "test")]
        x_ann = max(x_lin, x_hd) + 0.22
        ax.annotate(
            f"$\\Delta$={lh_diff:.3f}\n[{lh_lo:.3f}, {lh_hi:.3f}]",
            xy=(x_ann, (y_lin + y_hd) / 2),
            fontsize=6.5, color="#666666", va="center",
        )

    # Annotate decaying_shift-head (test)
    dh_row  = diff[(diff["split"].eq("test")) & (diff["comparison"].eq("decaying_shift_minus_head"))]
    dh_diff = float(dh_row["diff"].values[0])
    dh_lo   = float(dh_row["ci_lo"].values[0])
    dh_hi   = float(dh_row["ci_hi"].values[0])
    if ("decaying_shift", "test") in positions and ("head", "test") in positions:
        y_dc, _ = positions[("decaying_shift", "test")]
        y_hd, _ = positions[("head", "test")]
        x_ann2  = 11.65
        ax.annotate(
            f"$\\Delta$={dh_diff:.3f}\n[{dh_lo:.3f}, {dh_hi:.3f}]",
            xy=(x_ann2, (y_dc + y_hd) / 2),
            fontsize=6.5, color="#333333", va="center",
        )

    out = FIG_DIR / "study2_f2_baseline_ladder.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  S2-F2 -> {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# S2-F3: Event-enriched proxy vs unmatched control
# ─────────────────────────────────────────────────────────────────────────────

def plot_s2_f3() -> Path:
    gap_df  = pd.read_csv(ART / "study2_blocks_c_to_h/study2_event_placebo_gap.csv")
    diff_df = pd.read_csv(ART / "study2_blocks_c_to_h/study2_difference_cis.csv")

    # Rename placebo column to control
    gap_df = gap_df.rename(columns={
        "gain_placebo_mae": "gain_control_mae",
        "gap_ev_minus_pl":  "gap_ev_minus_control",
    })

    methods_show  = ["decaying_shift", "linear", "head"]
    split_show    = "test"
    sub           = gap_df[gap_df["split"].eq(split_show)]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(methods_show))
    w = 0.32

    for i, (col, label, color, hatch) in enumerate([
        ("gain_event_mae",   "Event-enriched", C["event"],   ""),
        ("gain_control_mae", "Unmatched control", C["control"], "//"),
    ]):
        vals = []
        for m in methods_show:
            r = sub[sub["method"].eq(m)]
            vals.append(float(r[col].values[0]) if not r.empty else 0.0)
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w,
                      color=color, hatch=hatch, label=label, zorder=3)

    # Gap annotation for head (test), pull from difference_cis
    gap_ci_row = diff_df[(diff_df["split"].eq("test")) &
                          (diff_df["comparison"].eq("head_gain_event_minus_placebo"))]
    if not gap_ci_row.empty:
        gap_val = float(gap_ci_row["diff"].values[0])
        gap_lo  = float(gap_ci_row["ci_lo"].values[0])
        gap_hi  = float(gap_ci_row["ci_hi"].values[0])
        idx_head = methods_show.index("head")
        sub_head = sub[sub["method"].eq("head")]
        ev_gain  = float(sub_head["gain_event_mae"].values[0])
        ctrl_gain = float(sub_head["gain_control_mae"].values[0])
        top_y    = max(ev_gain, ctrl_gain) + 0.12
        ax.annotate(
            f"Gap={gap_val:+.3f}\n95\\% CI [{gap_lo:+.3f}, {gap_hi:+.3f}]",
            xy=(x[idx_head], top_y),
            ha="center", fontsize=7, color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaaaaa", lw=0.6),
        )

    ax.axhline(0, color="#888888", lw=0.8, ls="-")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods_show])
    ax.set_ylabel("Correction gain over base (mg/dL MAE)")
    ax.set_title("S2-F3: Event-enriched proxy versus unmatched control (test split)")
    ax.legend(frameon=False, fontsize=7)
    caption_text = (
        "This comparison is descriptive, not a matched causal contrast. "
        "Event-enriched labels are glucose-level proxies."
    )
    fig.text(0.5, -0.04, caption_text, ha="center", fontsize=6.5,
             color="#666666", style="italic")

    out = FIG_DIR / "study2_f3_event_control_gap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  S2-F3 -> {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# S2-F4: Triggered-window examples (2×2 grid)
# ─────────────────────────────────────────────────────────────────────────────

def _load_head():
    """Load Ridge head coefficients from JSON."""
    data = json.loads(
        (ART / "study2_blocks_c_to_h/study2_correction_head.json").read_text()
    )
    return data


def _head_correction(h_data: dict, z_t: float, residual: float,
                     glucose: float, trigger_up: bool,
                     scaler_mean: list, scaler_scale: list) -> float:
    """Compute Ridge correction delta for one horizon."""
    feats = np.array([z_t, residual, glucose, float(trigger_up)])
    feats_scaled = (feats - np.array(scaler_mean)) / np.array(scaler_scale)
    coef = np.array(h_data["coefs"])
    intercept = h_data["intercept"]
    return float(intercept + coef @ feats_scaled)


def plot_s2_f4() -> Path:
    episodes = json.loads(
        (ART / "study2_preblock_i/study2_s2f4_episodes.json").read_text()
    )
    head_data = _load_head()
    scaler_mean  = head_data["scaler_mean"]
    scaler_scale = head_data["scaler_scale"]

    PANEL_TITLES = {
        "trigger_up_event":      "Trigger-up, event-enriched\n(pid {pid}, glucose {g:.0f} mg/dL)",
        "trigger_up_placebo":    "Trigger-up, unmatched control\n(pid {pid}, glucose {g:.0f} mg/dL)",
        "trigger_down":          "Trigger-down, unmatched control\n(pid {pid}, glucose {g:.0f} mg/dL)",
        "trigger_up_event_extra":"Trigger-up, event-enriched\n(pid {pid}, glucose {g:.0f} mg/dL)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    fig.subplots_adjust(hspace=0.44, wspace=0.32)

    for ep, ax in zip(episodes, axes.ravel()):
        ctx = ep["context_anchors"]
        fc  = ep["forecast_at_trigger"]
        pid = ep["participant_id"]
        g   = ep["current_glucose_at_trigger"] or 0.0
        cat = ep["category"]
        trig_ds   = ep["trigger_anchor_ds"]
        trigger_up  = ep["trigger_up"]

        # Context: glucose observations before trigger
        ctx_ds  = [r["anchor_ds"] for r in ctx]
        ctx_g   = [r["current_glucose"] or np.nan for r in ctx]
        # Relative time (steps from trigger, each step=5 min)
        ctx_t   = [(d - trig_ds) * 5 for d in ctx_ds]

        # Forecast at trigger: base q10/q50/q90
        fc_sorted = sorted(fc, key=lambda r: r["horizon_step"])
        fc_h   = [r["horizon_step"] for r in fc_sorted]
        fc_t   = [h * 5 for h in fc_h]  # minutes after trigger
        fc_q50 = [r["q50"] for r in fc_sorted]
        fc_q10 = [r["q10"] for r in fc_sorted]
        fc_q90 = [r["q90"] for r in fc_sorted]
        fc_tgt = [r["target"] for r in fc_sorted]

        # Head correction (using anchor features from last context row = trigger)
        last_ctx = ctx[-1]
        z_t_val  = last_ctx.get("z_t", 0.0) or 0.0
        res_val  = last_ctx.get("one_step_residual", 0.0) or 0.0
        glc_val  = last_ctx.get("current_glucose", g) or g
        fc_head  = []
        for r in fc_sorted:
            h = r["horizon_step"]
            h_key = str(h)
            if h_key in head_data["horizons"]:
                delta = _head_correction(
                    head_data["horizons"][h_key],
                    z_t_val, res_val, glc_val, trigger_up,
                    scaler_mean, scaler_scale,
                )
                fc_head.append(r["q50"] + delta)
            else:
                fc_head.append(r["q50"])

        # Plot
        # Context observations
        ax.plot(ctx_t, ctx_g, "k-", lw=1.4, label="Observed glucose", zorder=4)
        ax.plot(ctx_t, ctx_g, "ko", ms=3, zorder=5)

        # Base forecast: shaded band + median
        ax.fill_between(fc_t, fc_q10, fc_q90,
                        color=C["base"], alpha=0.35, label="Base 80\\% PI")
        ax.plot(fc_t, fc_q50, color="#888888", lw=1.2, ls="--", label="Base median")

        # Head median
        ax.plot(fc_t, fc_head, color=C["head"], lw=1.4, ls="-", label="Head median")

        # Realized glucose
        ax.plot(fc_t, fc_tgt, "k.", ms=5, label="Realized", zorder=5)

        # Trigger marker
        ax.axvline(0, color="#CC3311", lw=0.9, ls=":", label="Trigger")

        label_str = (PANEL_TITLES.get(cat, cat)
                     .format(pid=pid, g=g))
        ax.set_title(label_str, fontsize=7.5)
        ax.set_xlabel("Minutes from trigger", fontsize=7)
        ax.set_ylabel("Glucose (mg/dL)", fontsize=7)
        ax.tick_params(labelsize=7)

    # Shared legend from first axis
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("S2-F4: Triggered-window examples (test split)", fontsize=9,
                 fontweight="bold")

    out = FIG_DIR / "study2_f4_triggered_episodes.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  S2-F4 -> {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# S2-F5: Deployment behavior (3-panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_s2_f5() -> Path:
    deploy  = pd.read_csv(ART / "study2_blocks_c_to_h/study2_deployment_metrics.csv")
    abstain = pd.read_csv(ART / "study2_blocks_c_to_h/study2_abstention.csv")

    splits = ["train", "validation", "test"]
    bar_colors = [C[s] for s in splits]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    fig.subplots_adjust(wspace=0.38)

    # Panel A: trigger rate
    ax = axes[0]
    rates = [float(deploy[deploy["split"].eq(s)]["trigger_rate"].values[0]) * 100
             for s in splits]
    bars = ax.bar([SPLIT_LABEL[s] for s in splits], rates,
                  color=bar_colors, width=0.5, zorder=3)
    ax.set_ylabel("Trigger rate (\\%)")
    ax.set_title("(a) Trigger rate")
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                f"{val:.1f}\\%", ha="center", va="bottom", fontsize=7)

    # Panel B: episode duration (median + p90)
    ax = axes[1]
    meds = [float(deploy[deploy["split"].eq(s)]["trigger_duration_minutes_median"].values[0])
            for s in splits]
    p90s = [float(deploy[deploy["split"].eq(s)]["trigger_duration_minutes_p90"].values[0])
            for s in splits]
    x = np.arange(len(splits))
    w = 0.32
    ax.bar(x - w / 2, meds, width=w, color=bar_colors, label="Median", zorder=3)
    ax.bar(x + w / 2, p90s, width=w, color=bar_colors, alpha=0.45, label="P90",
           hatch="//", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABEL[s] for s in splits])
    ax.set_ylabel("Episode duration (min)")
    ax.set_title("(b) Episode duration")
    ax.legend(frameon=False, fontsize=7)

    # Panel C: abstention breakdown (stacked bar)
    ax = axes[2]
    abstain_rates = []
    reason_labels = ["first_anchor", "missing_interval"]
    for s in splits:
        row = abstain[abstain["split"].eq(s)]
        rates_r = []
        for r in reason_labels:
            col = f"rate_{r}"
            if col in row.columns and pd.notna(row[col].values[0]):
                rates_r.append(float(row[col].values[0]) * 100)
            else:
                rates_r.append(0.0)
        abstain_rates.append(rates_r)

    abstain_arr = np.array(abstain_rates)  # (n_splits, n_reasons)
    x = np.arange(len(splits))
    reason_colors = [C["event"], C["decaying_shift"]]
    reason_display = ["First anchor", "Missing interval"]
    bottom = np.zeros(len(splits))
    for ri, (rlab, rcol) in enumerate(zip(reason_display, reason_colors)):
        ax.bar(x, abstain_arr[:, ri], width=0.5, bottom=bottom,
               color=rcol, label=rlab, zorder=3)
        bottom += abstain_arr[:, ri]

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABEL[s] for s in splits])
    ax.set_ylabel("Abstention rate (\\%)")
    ax.set_title("(c) Abstention breakdown")
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle("S2-F5: Deployment behavior (non-insulin participants)",
                 fontsize=9, y=1.01, fontweight="bold")

    out = FIG_DIR / "study2_f5_deployment.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  S2-F5 -> {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating Study 2 figures...")
    plot_s2_f1()
    plot_s2_f2()
    plot_s2_f3()
    plot_s2_f4()
    plot_s2_f5()
    print("Done. All figures in:", FIG_DIR)


if __name__ == "__main__":
    main()
