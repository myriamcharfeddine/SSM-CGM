#!/usr/bin/env python3
"""Block 6 - regenerate meal-flag tables and figures from the new feature
evaluation (Blocks 1-4).

Consumes the unified harness outputs in
``outputs/no_log_scenarios/meal_transfer/feature_evaluation/`` (tag ``final``)
and overwrites the LaTeX tables and the redesigned figures used by
``report/sections/meal_flag_experiments.tex``:

  tables : meal_main_results, meal_causal_results, meal_negative_controls,
           meal_subgroup_results, meal_feature_provenance,
           meal_selection_decision, meal_partial_information
  figures: retrospective_causal_gap (peak error + hyper AUPRC main, MAE60 inset),
           negative_controls (headline), insulin_sensitivity (insulin split)

All numbers are reported on the non-insulin primary cohort with A' as the
baseline and C' as the reachable ceiling; the bidirectional teacher and the
predmeal_flag are shown only as flagged leakage diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
FEAT = REPO / "outputs/no_log_scenarios/meal_transfer/feature_evaluation"
TABLE_DIR = REPO / "report/tables/meal_flags"
FIG_DIR = REPO / "report/figures/meal_flags"

LABELS = {
    "BASELINE_q50_uncorrected": "q50",
    "Aprime": "A': slope+level+clock",
    "Cprime_causal_teacher": "C': causal teacher (ceiling)",
    "C_legacy_bidir_teacher": "C: bidir teacher (leakage)",
    "B_old_predmeal_flag": "B: predmeal flag (leakage)",
    "D_student_prob": "D: student prob.",
    "G_online_state": "G: online state",
    "H_online_full_state": "H: online full state",
}
GATE = ["MAE_eval_meal_window", "peak_error_1h", "time_to_peak_error_min",
        "hyper_AUPRC_max1h", "hyper_recall_at_p80_1h"]


def fmt(v, nd=3):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "--"
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def esc(s):
    return (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("'", r"$'$"))


def write_table(path: Path, headers, rows, align=None, resize=True):
    align = align or ("l" + "r" * (len(headers) - 1))
    L = []
    if resize:
        L.append(r"\resizebox{\textwidth}{!}{%")
    L += [rf"\begin{{tabular}}{{{align}}}", r"\toprule",
          " & ".join(headers) + r" \\", r"\midrule"]
    for r in rows:
        L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if resize:
        L.append(r"}")
    path.write_text("\n".join(L) + "\n")


def _ni(tag):
    ep = pd.read_csv(FEAT / f"{tag}_meal_feature_endpoints.csv")
    return ep[ep.cohort == "non_insulin"].set_index("setup")


def table_main_results(tag):
    ni = _ni(tag)
    order = ["BASELINE_q50_uncorrected", "Aprime", "Cprime_causal_teacher",
             "C_legacy_bidir_teacher", "B_old_predmeal_flag"]
    rows = []
    for s in order:
        if s not in ni.index:
            continue
        rows.append([esc(LABELS.get(s, s)), fmt(ni.loc[s, "MAE_eval_meal_window"]),
                     fmt(ni.loc[s, "peak_error_1h"]), fmt(ni.loc[s, "time_to_peak_error_min"], 1),
                     fmt(ni.loc[s, "hyper_AUPRC_max1h"]), fmt(ni.loc[s, "hyper_recall_at_p80_1h"]),
                     fmt(ni.loc[s, "MAE_60min"])])
    write_table(TABLE_DIR / "meal_main_results.tex",
                ["Setup", "Meal MAE", "Peak err.", "TTP err.", "Hyper AUPRC", "Recall@.80", "MAE60"],
                rows, align="lrrrrrr")


def table_causal_results(tag):
    ni = _ni(tag)
    order = ["Aprime", "Cprime_causal_teacher", "D_student_prob", "G_online_state", "H_online_full_state"]
    rows = []
    for s in order:
        if s not in ni.index:
            continue
        rows.append([esc(LABELS.get(s, s)), fmt(ni.loc[s, "MAE_eval_meal_window"]),
                     fmt(ni.loc[s, "peak_error_1h"]), fmt(ni.loc[s, "time_to_peak_error_min"], 1),
                     fmt(ni.loc[s, "hyper_AUPRC_max1h"]), fmt(ni.loc[s, "hyper_recall_at_p80_1h"]),
                     fmt(ni.loc[s, "MAE_60min"])])
    write_table(TABLE_DIR / "meal_causal_results.tex",
                ["Setup", "Meal MAE", "Peak err.", "TTP err.", "Hyper AUPRC", "Recall@.80", "MAE60"],
                rows, align="lrrrrrr")


def table_negative_controls(tag):
    df = pd.read_csv(FEAT / f"{tag}_meal_feature_negative_controls.csv")
    keep = ["Cprime_causal_teacher", "D_student_prob", "H_online_full_state", "B_old_predmeal_flag"]
    rows = []
    for s in keep:
        for ctrl in ["real", "shuffle", "time_shift", "block_shuffle"]:
            r = df[(df.setup == s) & (df.control == ctrl)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([esc(LABELS.get(s, s)), esc(ctrl), fmt(r["MAE_eval_meal_window"]),
                         fmt(r["peak_error_1h"]), fmt(r["hyper_AUPRC_max1h"])])
    write_table(TABLE_DIR / "meal_negative_controls.tex",
                ["Setup", "Control", "Meal MAE", "Peak err.", "Hyper AUPRC"], rows, align="llrrr")


def table_subgroups(tag):
    df = pd.read_csv(FEAT / f"{tag}_meal_feature_subgroups.csv")
    sub = df[(df.subgroup == "med_insulin") &
             (df.setup.isin(["BASELINE_q50_uncorrected", "Aprime", "H_online_full_state", "D_student_prob"]))]
    rows = []
    for _, r in sub.iterrows():
        rows.append([esc(r["value"]), esc(LABELS.get(r["setup"], r["setup"])),
                     f"{int(r['n_participants'])}", fmt(r["MAE_eval_meal_window"]),
                     fmt(r["peak_error_1h"]), fmt(r["MAE_60min"])])
    write_table(TABLE_DIR / "meal_subgroup_results.tex",
                ["Insulin", "Setup", "$n$", "Meal MAE", "Peak err.", "MAE60"], rows, align="llrrrr")


def table_provenance(tag):
    df = pd.read_csv(FEAT / f"{tag}_teacher_provenance.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append([esc(r["feature"]), str(int(r["max_input_offset_steps_vs_anchor"])),
                     str(r["uses_future_glucose"]), str(r["leakage_free"])])
    write_table(TABLE_DIR / "meal_feature_provenance.tex",
                ["Feature", "Max input offset (steps)", "Uses future glucose", "Leakage free"],
                rows, align="lrll")


def table_selection(tag):
    s = pd.read_csv(FEAT / f"{tag}_meal_selected_summary.csv")
    rows = []
    for _, r in s.iterrows():
        rows.append([esc(LABELS.get(r["setup"], r["setup"])),
                     f"{int(r['n_passing_endpoints'])}/{int(r['min_required'])}",
                     str(bool(r["leakage_free"])), r"\textbf{%s}" % str(bool(r["selected"]))])
    write_table(TABLE_DIR / "meal_selection_decision.tex",
                ["Setup", "Passing endpoints", "Leakage free", "Selected"], rows,
                align="lrll", resize=False)


def table_partial_information(tag):
    pi = pd.read_csv(FEAT / f"{tag}_partial_information.csv")
    rows = []
    for _, r in pi.iterrows():
        rows.append([esc(LABELS.get(r["setup"], r["setup"])), esc(r["feature"]),
                     fmt(r["partial_corr_with_q50_residual"], 3), fmt(r["delta_meal_window_mae"], 3),
                     fmt(r["delta_peak_error"], 3)])
    write_table(TABLE_DIR / "meal_partial_information.tex",
                ["Setup", "Feature", "Partial corr.", r"$\Delta$ Meal MAE", r"$\Delta$ Peak"],
                rows, align="llrrr")


def fig_endpoint_gap(tag):
    """Replace retrospective_causal_gap: peak error + hyper AUPRC main, MAE60 inset."""
    ni = _ni(tag)
    order = ["BASELINE_q50_uncorrected", "Aprime", "Cprime_causal_teacher",
             "D_student_prob", "H_online_full_state", "C_legacy_bidir_teacher"]
    order = [o for o in order if o in ni.index]
    short = {"BASELINE_q50_uncorrected": "q50", "Aprime": "A'", "Cprime_causal_teacher": "C'",
             "D_student_prob": "D", "H_online_full_state": "H", "C_legacy_bidir_teacher": "C (leak)"}
    colors = ["#707070", "#3a7d44", "#2f5d7c", "#7aa36f", "#9bbf85", "#b03a2e"]
    cats = [short[o] for o in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(cats, [ni.loc[o, "peak_error_1h"] for o in order], color=colors)
    axes[0].set_title("1 h peak error (primary endpoint)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("mg/dL")
    axes[1].bar(cats, [ni.loc[o, "hyper_AUPRC_max1h"] for o in order], color=colors)
    axes[1].set_title("Hyperglycemia AUPRC (primary endpoint)", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0.8, max(0.95, float(ni["hyper_AUPRC_max1h"].max()) + 0.02))
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    # MAE60 demoted to a small inset on the first axis
    inset = axes[0].inset_axes([0.58, 0.6, 0.4, 0.36])
    inset.bar(cats, [ni.loc[o, "MAE_60min"] for o in order], color=colors)
    inset.set_title("MAE60 (report only)", fontsize=7)
    inset.tick_params(axis="x", labelsize=5, rotation=45)
    inset.tick_params(axis="y", labelsize=6)
    fig.suptitle("The reachable causal ceiling C' sits at A'; the C leakage bar is unreachable headroom",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "retrospective_causal_gap.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "retrospective_causal_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_negative_controls(tag):
    df = pd.read_csv(FEAT / f"{tag}_meal_feature_negative_controls.csv")
    setups = [s for s in ["D_student_prob", "H_online_full_state", "B_old_predmeal_flag"]
              if s in df.setup.unique()]
    ctrls = ["real", "shuffle", "time_shift", "block_shuffle"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, metric in enumerate(["peak_error_1h", "hyper_AUPRC_max1h"]):
        piv = df[df.setup.isin(setups)].pivot(index="control", columns="setup", values=metric).reindex(ctrls)
        piv.columns = [LABELS.get(c, c).split(":")[0] for c in piv.columns]
        piv.plot(kind="bar", ax=axes[j], width=0.8)
        axes[j].set_title(metric.replace("_", " "), fontsize=10, fontweight="bold")
        axes[j].tick_params(axis="x", labelrotation=15, labelsize=8)
        axes[j].legend(fontsize=7, frameon=False)
        axes[j].grid(axis="y", alpha=0.25)
    fig.suptitle("Negative controls: real timing vs shuffled / shifted / block-shuffled (headline evidence)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "negative_controls.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "negative_controls.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_insulin_split(tag):
    df = pd.read_csv(FEAT / f"{tag}_meal_feature_subgroups.csv")
    sub = df[(df.subgroup == "med_insulin") &
             (df.setup.isin(["Aprime", "H_online_full_state"]))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, metric in enumerate(["peak_error_1h", "MAE_60min"]):
        piv = sub.pivot(index="value", columns="setup", values=metric)
        piv.columns = [LABELS.get(c, c).split(":")[0] for c in piv.columns]
        piv.plot(kind="bar", ax=axes[j], width=0.7)
        axes[j].set_title(metric.replace("_", " "), fontsize=10, fontweight="bold")
        axes[j].tick_params(axis="x", labelrotation=0, labelsize=9)
        axes[j].legend(fontsize=8, frameon=False)
        axes[j].grid(axis="y", alpha=0.25)
    fig.suptitle("Insulin vs non-insulin split: uncertainty expands for insulin users",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "insulin_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "insulin_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table_main_results(args.tag)
    table_causal_results(args.tag)
    table_negative_controls(args.tag)
    table_subgroups(args.tag)
    table_provenance(args.tag)
    table_selection(args.tag)
    table_partial_information(args.tag)
    fig_endpoint_gap(args.tag)
    fig_negative_controls(args.tag)
    fig_insulin_split(args.tag)
    print("regenerated meal-flag tables and figures from feature_evaluation tag:", args.tag)


if __name__ == "__main__":
    main()
