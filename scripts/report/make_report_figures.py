#!/usr/bin/env python3
"""
make_report_figures.py
Generate all report figures from outputs/<run>/ data.

Priority order:
  1. Copy pre-built PNGs from outputs/<run>/figures/ (fast, always done)
  2. Generate additional figures from metrics CSVs/JSONs (needs matplotlib)

Figures produced in report/figures/generated/:
  horizon_mae.png           — copied from run, enhanced if matplotlib available
  personalization_warmup.png — copied from run, enhanced if matplotlib available
  subgroup_study_group.png  — MAE + coverage + bias by study group
  subgroup_hba1c.png        — MAE + TIR by HbA1c quartile
  subgroup_bmi.png          — MAE by BMI quartile
  subgroup_site.png         — MAE + bias by clinical site
  subgroup_medication.png   — MAE by medication flag
  subgroup_all_panel.png    — combined 5-panel subgroup overview
  bias_diagnostic.png       — 4 personalization modes
  scenario_comparison.png   — 5 scenario modes
  clinical_safety.png       — hypo/hyper precision + recall
  tir_by_group.png          — true vs predicted TIR by study group
  training_curves.png       — train/val pinball + throughput by epoch
  summary_panel.png         — 6-panel overview combining key results

Usage (from repo root):
    python scripts/report/make_report_figures.py
    python scripts/report/make_report_figures.py \\
        --collected outputs/_report_collected.json \\
        --out-dir report/figures/generated
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

# ── Copy map: run/figures/<src> → generated/<dst> ───────────────────────────
COPY_MAP = {
    "mae_by_horizon.png": "horizon_mae.png",
    "personalization_sweep.png": "personalization_warmup.png",
}

# ── Colour palette ───────────────────────────────────────────────────────────
COLORS = {
    "healthy": "#4c72b0",
    "pre_diabetes": "#55a868",
    "oral_med": "#c44e52",
    "insulin_dep": "#8172b2",
    "default": ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"],
    "modes": ["#2d7dd2", "#97cc04", "#f45d01", "#e4b429", "#474647"],
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _require_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        print("  matplotlib not installed — skipping generated figures", file=sys.stderr)
        print("  Install with:  pip install matplotlib", file=sys.stderr)
        return False


def _save(fig, out_dir: Path, name: str):
    import matplotlib.pyplot as plt
    p = out_dir / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {name}")


def _fmt(val, dec=2):
    try:
        return round(float(val), dec)
    except (TypeError, ValueError):
        return 0.0


def _load_subgroup_rows(data: dict, key: str) -> list[dict]:
    return [r for r in data.get("subgroup_metrics", []) if r.get("subgroup") == key]


# ── Step 1: copy pre-built figures ───────────────────────────────────────────

def copy_run_figures(run_path: Path, out_dir: Path) -> set[str]:
    copied: set[str] = set()
    fig_dir = run_path / "figures"
    if not fig_dir.exists():
        print(f"  No figures/ dir in {run_path.name}")
        return copied
    for src_name, dst_name in COPY_MAP.items():
        src = fig_dir / src_name
        if src.exists():
            dst = out_dir / dst_name
            shutil.copy2(src, dst)
            print(f"  Copied  {src_name} → {dst_name}")
            copied.add(dst_name)
        else:
            print(f"  Missing {src_name} in run (will regenerate)")
    return copied


# ── Step 2: individual generated figures ─────────────────────────────────────

def fig_horizon_mae(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    rows = data.get("horizon_metrics", [])
    if not rows:
        return
    minutes = [_fmt(r["horizon_minutes"], 0) for r in rows]
    maes    = [_fmt(r["mae"]) for r in rows]
    rmses   = [_fmt(r["rmse"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(minutes, maes,  "o-", color="#c44e52", lw=1.8, ms=5, label="MAE")
    ax.plot(minutes, rmses, "s--", color="#4c72b0", lw=1.4, ms=4, label="RMSE")
    ax.set_xlabel("Forecast horizon (min)", fontsize=10)
    ax.set_ylabel("Error (mg/dL)", fontsize=10)
    ax.set_title("Forecast accuracy vs. horizon — test split (forecast-only)", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "horizon_mae.png")


def fig_personalization_warmup(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    rows = data.get("personalization_sweep", [])
    if not rows:
        return
    wh   = [_fmt(r["warmup_hours"], 0) for r in rows]
    maes = [_fmt(r["mae"]) for r in rows]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(wh, maes, "o-", color="#4c72b0", lw=1.8, ms=6)
    ax.axhline(maes[0], color="gray", lw=1, ls="--", label="cold start")
    ax.fill_between(wh, maes[0], maes, alpha=0.12, color="#4c72b0")
    ax.set_xlabel("Warm-up hours streamed per participant", fontsize=10)
    ax.set_ylabel("MAE (mg/dL)", fontsize=10)
    ax.set_title("Personalization warm-start (offset adapter)", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "personalization_warmup.png")


def fig_subgroup_study_group(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    rows = _load_subgroup_rows(data, "participants_study_group")
    if not rows:
        return

    label_map = {
        "healthy": "Healthy",
        "pre_diabetes_lifestyle_controlled": "Pre-DM\n(lifestyle)",
        "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral/\nInject.",
        "insulin_dependent": "Insulin\nDependent",
    }
    labels   = [label_map.get(r["level"], r["level"]) for r in rows]
    maes     = [_fmt(r["mae"]) for r in rows]
    coverages= [_fmt(r["coverage80"], 3) * 100 for r in rows]
    biases   = [_fmt(r["bias"], 3) for r in rows]
    tir_true = [_fmt(r["tir_true"], 3) * 100 for r in rows]
    tir_pred = [_fmt(r["tir_predicted"], 3) * 100 for r in rows]

    x  = np.arange(len(labels))
    cs = COLORS["default"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel A: MAE
    bars = axes[0].bar(x, maes, color=cs[:len(labels)], width=0.6, edgecolor="k", lw=0.6)
    for b, v in zip(bars, maes):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("MAE (mg/dL)"); axes[0].set_title("(A) MAE by study group")

    # Panel B: TIR true vs predicted
    w = 0.35
    bars1 = axes[1].bar(x - w/2, tir_true, w, label="True TIR",
                        color=[c + "bb" for c in ["#4c72b0","#55a868","#c44e52","#8172b2"][:len(labels)]],
                        edgecolor="k", lw=0.5)
    bars2 = axes[1].bar(x + w/2, tir_pred, w, label="Predicted TIR",
                        color=cs[:len(labels)], edgecolor="k", lw=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("TIR 70–180 mg/dL (%)"); axes[1].set_title("(B) TIR by study group")
    axes[1].legend(fontsize=8)

    # Panel C: bias
    bar_colors = ["#c44e52" if b < 0 else "#55a868" for b in biases]
    axes[2].bar(x, biases, color=bar_colors, width=0.6, edgecolor="k", lw=0.6)
    axes[2].axhline(0, color="k", lw=0.8)
    for i, v in enumerate(biases):
        axes[2].text(i, v + (0.05 if v >= 0 else -0.15),
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("Bias (mg/dL)"); axes[2].set_title("(C) Forecast bias by study group")

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Study group subgroup analysis (forecast-only, test split)", fontsize=11, y=1.01)
    plt.tight_layout()
    _save(fig, out_dir, "subgroup_study_group.png")


def fig_subgroup_hba1c(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    rows = [r for r in data.get("subgroup_metrics", [])
            if r.get("subgroup") == "HbA1c_quartile" and r.get("level")]
    if not rows:
        return
    # sort by HbA1c level string
    rows = sorted(rows, key=lambda r: r.get("level", ""))
    labels   = [r["level"] for r in rows]
    maes     = [_fmt(r["mae"]) for r in rows]
    tir_true = [_fmt(r["tir_true"], 3) * 100 for r in rows]
    tir_gap  = [_fmt(r["tir_gap"], 4) * 100 for r in rows]

    x = np.arange(len(labels)); cs = COLORS["default"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    bars = axes[0].bar(x, maes, color=cs[:len(labels)], width=0.6, edgecolor="k", lw=0.5)
    for b, v in zip(bars, maes):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    axes[0].set_ylabel("MAE (mg/dL)"); axes[0].set_title("(A) MAE by HbA1c quartile")

    bar_colors = ["#c44e52" if g > 0 else "#55a868" for g in tir_gap]
    axes[1].bar(x, tir_gap, color=bar_colors, width=0.6, edgecolor="k", lw=0.5)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    axes[1].set_ylabel("TIR gap (predicted − true, pp)"); axes[1].set_title("(B) TIR gap by HbA1c")

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("HbA1c quartile subgroup analysis (forecast-only, test split)", fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, "subgroup_hba1c.png")


def fig_subgroup_site(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    rows = _load_subgroup_rows(data, "participants_clinical_site")
    if not rows:
        return
    labels = [r["level"] for r in rows]
    maes   = [_fmt(r["mae"]) for r in rows]
    biases = [_fmt(r["bias"], 3) for r in rows]
    x = np.arange(len(labels)); cs = COLORS["default"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    bars = axes[0].bar(x, maes, color=cs[:len(labels)], width=0.5, edgecolor="k", lw=0.5)
    for b, v in zip(bars, maes):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("MAE (mg/dL)"); axes[0].set_title("(A) MAE by clinical site")

    bar_colors = ["#c44e52" if b < 0 else "#55a868" for b in biases]
    axes[1].bar(x, biases, color=bar_colors, width=0.5, edgecolor="k", lw=0.5)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Bias (mg/dL)"); axes[1].set_title("(B) Bias by clinical site")

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Clinical site subgroup analysis (forecast-only, test split)", fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, "subgroup_site.png")


def fig_subgroup_medication(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    rows_ins = _load_subgroup_rows(data, "med_insulin")
    rows_any = _load_subgroup_rows(data, "med_any_diabetes_drug")
    if not rows_ins and not rows_any:
        return

    def extract(rows, label_map):
        labels = [label_map.get(r["level"], str(r["level"])) for r in rows]
        maes   = [_fmt(r["mae"]) for r in rows]
        return labels, maes

    ins_labels, ins_maes = extract(rows_ins, {"0": "No insulin", "1": "Insulin user"})
    any_labels, any_maes = extract(rows_any, {"0": "No diabetes drug", "1": "Any diabetes drug"})
    cs = COLORS["default"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, labels, maes, title in [
        (axes[0], ins_labels, ins_maes, "MAE by insulin use (med_insulin)"),
        (axes[1], any_labels, any_maes, "MAE by any diabetes drug"),
    ]:
        x = np.arange(len(labels))
        bars = ax.bar(x, maes, color=cs[:len(labels)], width=0.4, edgecolor="k", lw=0.5)
        for b, v in zip(bars, maes):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("MAE (mg/dL)"); ax.set_title(title)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Medication subgroup analysis (forecast-only, test split)", fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, "subgroup_medication.png")


def fig_bias_diagnostic(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    rows = data.get("bias_diagnostic", [])
    if not rows:
        return
    mode_labels = {
        "raw": "Raw\n(no personalization)",
        "offset-corrected": "Offset\ncorrected",
        "personalized": "Personalized",
        "personalized+offset-corrected": "Pers. +\noffset",
    }
    labels  = [mode_labels.get(r["bias_mode"], r["bias_mode"]) for r in rows]
    maes    = [_fmt(r["mae"]) for r in rows]
    biases  = [_fmt(r["bias"], 3) for r in rows]
    covs    = [_fmt(r["coverage80"], 3) * 100 for r in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    cs = COLORS["default"]

    bars = axes[0].bar(x, maes, color=cs[:len(labels)], width=0.55, edgecolor="k", lw=0.5)
    for b, v in zip(bars, maes):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("MAE (mg/dL)"); axes[0].set_title("(A) MAE by personalization mode")

    bar_colors = ["#c44e52" if b < 0 else "#55a868" for b in biases]
    axes[1].bar(x, biases, color=bar_colors, width=0.55, edgecolor="k", lw=0.5)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Bias (mg/dL)"); axes[1].set_title("(B) Bias by personalization mode")

    axes[2].bar(x, covs, color=cs[:len(labels)], width=0.55, edgecolor="k", lw=0.5)
    axes[2].axhline(80, color="k", lw=0.8, ls="--", label="Nominal 80%")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("Coverage (%)"); axes[2].set_title("(C) 80% interval coverage")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Personalization modes comparison (test split)", fontsize=11, y=1.01)
    plt.tight_layout()
    _save(fig, out_dir, "bias_diagnostic.png")


def fig_scenario_comparison(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    rows = data.get("scenario_metrics", [])
    if not rows:
        return
    mode_labels = {
        "forecast_only":     "Forecast-only\n(deployable)",
        "factual_future":    "Factual future\n(oracle)",
        "meal_proxy":        "Meal proxy",
        "activity_proxy":    "Activity proxy",
        "sleep_rest_proxy":  "Sleep/rest\nproxy",
    }
    labels = [mode_labels.get(r["scenario_mode"], r["scenario_mode"]) for r in rows]
    maes   = [_fmt(r["mae"]) for r in rows]
    covs   = [_fmt(r["coverage80"], 3) * 100 for r in rows]
    x = np.arange(len(labels))
    cs = COLORS["modes"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bars = axes[0].bar(x, maes, color=cs[:len(labels)], width=0.6, edgecolor="k", lw=0.5)
    for b, v in zip(bars, maes):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("MAE (mg/dL)"); axes[0].set_title("(A) MAE by evaluation mode")

    # zoom in to show the difference
    mae_min = min(maes) - 0.005
    mae_max = max(maes) + 0.010
    axes[0].set_ylim(mae_min, mae_max)
    axes[0].annotate("Differences are < 0.001 mg/dL:\nscenario mask has minimal\nimpact at 60-min horizon",
                     xy=(0.5, 0.5), xycoords="axes fraction",
                     fontsize=7, ha="center", color="gray",
                     bbox=dict(fc="white", ec="gray", alpha=0.7))

    axes[1].bar(x, covs, color=cs[:len(labels)], width=0.6, edgecolor="k", lw=0.5)
    axes[1].axhline(80, color="k", lw=0.8, ls="--", label="Nominal 80%")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("80% Coverage (%)"); axes[1].set_title("(B) Coverage by evaluation mode")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Scenario / evaluation mode comparison (test split)\n"
                 "[proxy modes are not validated causal interventions]",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, "scenario_comparison.png")


def fig_clinical_safety(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    cs_data = data.get("clinical_safety", {})
    if not cs_data:
        return
    hypo  = cs_data.get("hypoglycemia_detection", {})
    hyper = cs_data.get("hyperglycemia_detection", {})

    metrics   = ["Precision", "Recall", "F1"]
    def f1(p, r): return 2*p*r/(p+r) if (p+r) > 0 else 0
    hypo_vals  = [_fmt(hypo.get("precision",0),3),
                  _fmt(hypo.get("recall",0),3),
                  round(f1(_fmt(hypo.get("precision",0),3), _fmt(hypo.get("recall",0),3)), 3)]
    hyper_vals = [_fmt(hyper.get("precision",0),3),
                  _fmt(hyper.get("recall",0),3),
                  round(f1(_fmt(hyper.get("precision",0),3), _fmt(hyper.get("recall",0),3)), 3)]

    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, hypo_vals,  w, label="Hypo <70",  color="#c44e52", edgecolor="k", lw=0.5)
    b2 = ax.bar(x + w/2, hyper_vals, w, label="Hyper >180", color="#4c72b0", edgecolor="k", lw=0.5)
    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Clinical event detection (median forecast rule, 60-min horizon)", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "clinical_safety.png")


def fig_tir_by_group(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    rows = _load_subgroup_rows(data, "participants_study_group")
    if not rows:
        return
    label_map = {
        "healthy": "Healthy",
        "pre_diabetes_lifestyle_controlled": "Pre-DM",
        "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral/Inj.",
        "insulin_dependent": "Insulin",
    }
    labels   = [label_map.get(r["level"], r["level"]) for r in rows]
    tir_true = [_fmt(r["tir_true"],3)*100 for r in rows]
    tir_pred = [_fmt(r["tir_predicted"],3)*100 for r in rows]
    tir_gap  = [_fmt(r["tir_gap"],4)*100 for r in rows]

    x = np.arange(len(labels)); w = 0.35
    cs = COLORS["default"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    b1 = axes[0].bar(x - w/2, tir_true, w, label="True TIR",
                     color=[c + "99" for c in cs[:len(labels)]], edgecolor="k", lw=0.5)
    b2 = axes[0].bar(x + w/2, tir_pred, w, label="Predicted TIR",
                     color=cs[:len(labels)], edgecolor="k", lw=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("TIR 70–180 mg/dL (%)"); axes[0].set_title("(A) True vs. predicted TIR")
    axes[0].legend(fontsize=9)

    bar_colors = ["#c44e52" if g > 0 else "#55a868" for g in tir_gap]
    axes[1].bar(x, tir_gap, color=bar_colors, width=0.5, edgecolor="k", lw=0.5)
    axes[1].axhline(0, color="k", lw=0.8)
    for i, v in enumerate(tir_gap):
        axes[1].text(i, v + (0.05 if v >= 0 else -0.3),
                     f"{v:.2f}", ha="center", fontsize=9)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("TIR gap (predicted − true, pp)")
    axes[1].set_title("(B) TIR over-prediction gap")

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Time-in-range calibration by study group (test split)", fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, "tir_by_group.png")


def fig_training_curves(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    ts = data.get("training_summary", {})
    hist = ts.get("history", [])
    if len(hist) < 2:
        print("  SKIP training_curves: fewer than 2 epochs")
        return
    epochs      = [e["epoch"] for e in hist]
    train_loss  = [_fmt(e["train_pinball_mgdl"], 3) for e in hist]
    val_loss    = [_fmt(e["val_pinball_mgdl"], 3) for e in hist]
    throughput  = [_fmt(e.get("anchors_per_s", 0), 0) for e in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, train_loss, "o-", color="#c44e52", lw=1.8, ms=5, label="Train pinball")
    ax1.plot(epochs, val_loss,   "s--", color="#4c72b0", lw=1.8, ms=5, label="Val pinball")
    best_ep = epochs[val_loss.index(min(val_loss))]
    ax1.axvline(best_ep, color="gray", lw=1, ls=":", label=f"Best val (epoch {best_ep})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Pinball loss (mg/dL)")
    ax1.set_title("(A) Training and validation loss")
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2.bar(epochs, throughput, color="#55a868", edgecolor="k", lw=0.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Anchors / second")
    ax2.set_title("(B) Training throughput per epoch")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir, "training_curves.png")


def fig_summary_panel(data: dict, out_dir: Path):
    """Six-panel overview combining the key results."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    cs = COLORS["default"]

    # ── Panel 1: MAE by study group ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    rows = _load_subgroup_rows(data, "participants_study_group")
    if rows:
        lmap = {"healthy":"Healthy","pre_diabetes_lifestyle_controlled":"Pre-DM",
                "oral_medication_and_or_non_insulin_injectable_medication_controlled":"Oral/Inj.",
                "insulin_dependent":"Insulin"}
        labels = [lmap.get(r["level"], r["level"]) for r in rows]
        maes   = [_fmt(r["mae"]) for r in rows]
        x = np.arange(len(labels))
        bars = ax1.bar(x, maes, color=cs[:len(labels)], width=0.6, edgecolor="k", lw=0.5)
        for b, v in zip(bars, maes):
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
        ax1.set_ylabel("MAE (mg/dL)", fontsize=9)
    ax1.set_title("MAE by study group", fontsize=10, fontweight="bold")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # ── Panel 2: Horizon MAE ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    hrows = data.get("horizon_metrics", [])
    if hrows:
        minutes = [_fmt(r["horizon_minutes"], 0) for r in hrows]
        maes    = [_fmt(r["mae"]) for r in hrows]
        ax2.plot(minutes, maes, "o-", color="#c44e52", lw=1.8, ms=4)
        ax2.fill_between(minutes, maes, alpha=0.12, color="#c44e52")
        ax2.set_xlabel("Horizon (min)", fontsize=9)
        ax2.set_ylabel("MAE (mg/dL)", fontsize=9)
    ax2.set_title("MAE vs. forecast horizon", fontsize=10, fontweight="bold")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # ── Panel 3: Training curves ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ts   = data.get("training_summary", {})
    hist = ts.get("history", [])
    if len(hist) >= 2:
        epochs     = [e["epoch"] for e in hist]
        train_loss = [_fmt(e["train_pinball_mgdl"], 3) for e in hist]
        val_loss   = [_fmt(e["val_pinball_mgdl"], 3) for e in hist]
        ax3.plot(epochs, train_loss, "o-", color="#c44e52", lw=1.6, ms=4, label="Train")
        ax3.plot(epochs, val_loss,   "s--", color="#4c72b0", lw=1.6, ms=4, label="Val")
        ax3.set_xlabel("Epoch", fontsize=9); ax3.set_ylabel("Pinball (mg/dL)", fontsize=9)
        ax3.legend(fontsize=8)
    ax3.set_title("Training / validation loss", fontsize=10, fontweight="bold")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # ── Panel 4: Bias diagnostic ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    brows = data.get("bias_diagnostic", [])
    if brows:
        bmode_map = {"raw":"Raw","offset-corrected":"Offset\ncorr.",
                     "personalized":"Pers.","personalized+offset-corrected":"Pers.+\noffset"}
        blabels = [bmode_map.get(r["bias_mode"], r["bias_mode"]) for r in brows]
        bvals   = [_fmt(r["bias"], 3) for r in brows]
        x = np.arange(len(blabels))
        bar_colors = ["#c44e52" if v < 0 else "#55a868" for v in bvals]
        ax4.bar(x, bvals, color=bar_colors, width=0.55, edgecolor="k", lw=0.5)
        ax4.axhline(0, color="k", lw=0.8)
        ax4.set_xticks(x); ax4.set_xticklabels(blabels, fontsize=8)
        ax4.set_ylabel("Bias (mg/dL)", fontsize=9)
    ax4.set_title("Bias by personalization mode", fontsize=10, fontweight="bold")
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    # ── Panel 5: TIR gap by study group ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if rows:
        tir_gap = [_fmt(r["tir_gap"], 4)*100 for r in rows]
        x       = np.arange(len(labels))
        bar_colors2 = ["#c44e52" if g > 0 else "#55a868" for g in tir_gap]
        ax5.bar(x, tir_gap, color=bar_colors2, width=0.6, edgecolor="k", lw=0.5)
        ax5.axhline(0, color="k", lw=0.8)
        ax5.set_xticks(x); ax5.set_xticklabels(labels, fontsize=8)
        ax5.set_ylabel("TIR gap (pp)", fontsize=9)
    ax5.set_title("TIR over-prediction gap", fontsize=10, fontweight="bold")
    ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

    # ── Panel 6: Clinical safety ──────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    cs_data = data.get("clinical_safety", {})
    hypo    = cs_data.get("hypoglycemia_detection", {})
    hyper   = cs_data.get("hyperglycemia_detection", {})
    if hypo and hyper:
        def f1(p, r): return 2*p*r/(p+r) if (p+r) > 0 else 0
        metrics   = ["Precision", "Recall", "F1"]
        hypo_v  = [_fmt(hypo.get("precision",0),3), _fmt(hypo.get("recall",0),3),
                   round(f1(_fmt(hypo.get("precision",0),3), _fmt(hypo.get("recall",0),3)),3)]
        hyper_v = [_fmt(hyper.get("precision",0),3), _fmt(hyper.get("recall",0),3),
                   round(f1(_fmt(hyper.get("precision",0),3), _fmt(hyper.get("recall",0),3)),3)]
        x = np.arange(3); w = 0.35
        ax6.bar(x - w/2, hypo_v,  w, label="Hypo <70",   color="#c44e52", edgecolor="k", lw=0.4)
        ax6.bar(x + w/2, hyper_v, w, label="Hyper >180", color="#4c72b0", edgecolor="k", lw=0.4)
        ax6.set_xticks(x); ax6.set_xticklabels(metrics, fontsize=9)
        ax6.set_ylim(0, 1.05)
        ax6.legend(fontsize=8)
    ax6.set_title("Clinical event detection", fontsize=10, fontweight="bold")
    ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)

    fig.suptitle("AI-READI SSMCGM-Stream — Summary Panel (test split, forecast-only)",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, out_dir, "summary_panel.png")


# ── Wire all generated figures into LaTeX ────────────────────────────────────

def list_generated(out_dir: Path):
    figs = sorted(p.name for p in out_dir.iterdir() if p.is_file()
                  and p.suffix in {".png", ".pdf"})
    print(f"\nAll figures in {out_dir}/:")
    for f in figs:
        print(f"  {f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected", default="outputs/_report_collected.json")
    parser.add_argument("--out-dir",   default="report/figures/generated")
    args = parser.parse_args()

    collected_path = Path(args.collected)
    if not collected_path.exists():
        print(f"ERROR: {collected_path} not found.  Run collect_latest_results.py first.",
              file=sys.stderr)
        sys.exit(1)

    with open(collected_path) as f:
        data = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {out_dir}/\n")

    # Step 1 — copy pre-built PNGs from the run
    run_path = Path(data.get("run_path", "."))
    print("── Step 1: copy run figures ──")
    copied = copy_run_figures(run_path, out_dir)

    # Step 2 — generate the rest with matplotlib
    print("\n── Step 2: generate figures from metrics ──")
    if not _require_matplotlib():
        list_generated(out_dir)
        return

    # Replace the basic copies with enhanced versions
    fig_horizon_mae(data, out_dir)
    fig_personalization_warmup(data, out_dir)

    # Additional figures not produced by the run
    fig_subgroup_study_group(data, out_dir)
    fig_subgroup_hba1c(data, out_dir)
    fig_subgroup_site(data, out_dir)
    fig_subgroup_medication(data, out_dir)
    fig_bias_diagnostic(data, out_dir)
    fig_scenario_comparison(data, out_dir)
    fig_clinical_safety(data, out_dir)
    fig_tir_by_group(data, out_dir)
    fig_training_curves(data, out_dir)
    fig_summary_panel(data, out_dir)   # combined overview last

    list_generated(out_dir)


if __name__ == "__main__":
    main()
