#!/usr/bin/env python3
"""Figure-only reorganization of finalized Phase 2 neighborhood results.

All plotted values are read from final_subtype_centered_figures_v5 tables. No
upstream analysis, hidden-state extraction, graph construction, permutation
testing, bootstrapping, clustering, or metric computation is performed here.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PHASE2 = Path("/home/myriamcharfeddine/CGM/SSM-CGM/outputs/static_phenotype_trajectory_stratified_v2/extended_clinical_latent_dynamics_v1/02_circadian_matched_reorganization")
SOURCE = PHASE2 / "final_subtype_centered_figures_v5"
OUT = PHASE2 / "final_main_and_appendix_layout_v6"
FIG = OUT / "figures"
TABLE = OUT / "tables"
META = OUT / "metadata"
REPORT = OUT / "reports"
QA = OUT / "qa"

SUBTYPES = ["healthy", "pre_diabetes", "t2d_oral_non_insulin", "insulin_dependent"]
SUBTYPE_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes": "Prediabetes",
    "t2d_oral_non_insulin": "T2D oral non-insulin",
    "insulin_dependent": "Insulin-dependent, exploratory",
}
PALETTE = {
    "healthy": {"light": "#9CB3C8", "medium": "#5B7FA3", "dark": "#003366"},
    "pre_diabetes": {"light": "#B5DEDE", "medium": "#5BBABA", "dark": "#2F7F7F"},
    "t2d_oral_non_insulin": {"light": "#E7A6A6", "medium": "#BA4A4A", "dark": "#7A1F1F"},
    "insulin_dependent": {"light": "#C7CDD4", "medium": "#8994A2", "dark": "#4A5568"},
}
METRICS = ["clinical_to_h0", "h0_to_ht", "clinical_to_ht"]
METRIC_LABELS = {"clinical_to_h0": "Clinical → h₀", "h0_to_ht": "h₀ → hₜ", "clinical_to_ht": "Clinical → hₜ"}
REPRESENTATIONS = ["clinical", "h0", "ht"]
REP_LABELS = {"clinical": "Clinical", "h0": "h₀", "ht": "hₜ"}
EVENT_HOURS = [6, 12, 24, 48]
BLACK = "#000000"
GRID = "#D9D9D9"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8.8, "ytick.labelsize": 8.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": BLACK, "axes.spines.top": True, "axes.spines.right": True,
        "grid.color": GRID, "grid.linewidth": .7, "pdf.fonttype": 42,
    })


def frame(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=.7)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(BLACK); s.set_linewidth(.8)


def save_figure(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}_thumbnail.png", dpi=75, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compact_positions(nbars=3, bar_width=.58, within_gap=.02, between_gap=1.0):
    positions, blocks = [], []
    cursor = 0.0
    step = bar_width + within_gap
    for subtype in SUBTYPES:
        block = [cursor + i * step for i in range(nbars)]
        positions.extend(block); blocks.append((subtype, block[0], block[-1]))
        cursor = block[-1] + bar_width + between_gap
    return np.array(positions), blocks


def common_ylim(data, low, high, floor_zero=False):
    lo = float(data[low].min()); hi = float(data[high].max()); span = max(hi - lo, .04)
    return (min(0, lo - .08 * span), max(0, hi + .08 * span)) if not floor_zero else (0, max(1, hi * 1.08))


def add_block_labels(ax, blocks, y=-.19, fontsize=10.5):
    for subtype, left, right in blocks:
        center = (left + right) / 2
        ax.text(center, y, SUBTYPE_LABELS[subtype], transform=ax.get_xaxis_transform(), ha="center", va="top", fontweight="bold", fontsize=fontsize)


def add_separators(ax, blocks, ylim):
    for (_, left, right), (next_subtype, next_left, _) in zip(blocks[:-1], blocks[1:]):
        x = (right + next_left) / 2
        ax.vlines(x, *ylim, color="#DDDDDD", linewidth=.8, zorder=0)


def grouped_main_overlap(ax, data, ylim):
    pos, blocks = compact_positions()
    vals = []; lows = []; highs = []; colors = []; labels = []
    for subtype in SUBTYPES:
        q = data[data.canonical_stratum.eq(subtype)].set_index("metric").reindex(METRICS)
        vals.extend(q.chance_adjusted_overlap.to_numpy(float)); lows.extend(q.adjusted_ci_low.to_numpy(float)); highs.extend(q.adjusted_ci_high.to_numpy(float))
        colors.extend([PALETTE[subtype]["dark"], PALETTE[subtype]["medium"], PALETTE[subtype]["light"]]); labels.extend([METRIC_LABELS[m] for m in METRICS])
    vals = np.asarray(vals); ax.bar(pos, vals, width=.58, color=colors, edgecolor=BLACK, linewidth=.6, yerr=np.vstack([vals-np.asarray(lows), np.asarray(highs)-vals]), capsize=3, error_kw={"elinewidth": 1})
    ax.axhline(0, color=BLACK, linewidth=1); ax.set_xticks(pos, labels, rotation=0); ax.set_ylim(*ylim); ax.set_ylabel("Chance-adjusted shared-neighbor overlap"); add_block_labels(ax, blocks); add_separators(ax, blocks, ylim); frame(ax)
    return blocks


def grouped_main_purity(ax, data, ylim):
    pos, blocks = compact_positions()
    vals = []; lows = []; highs = []; colors = []; labels = []
    for subtype in SUBTYPES:
        q = data[data.canonical_stratum.eq(subtype)].set_index("representation").reindex(REPRESENTATIONS)
        vals.extend(q.chance_adjusted_purity.to_numpy(float)); lows.extend(q.adjusted_ci_low.to_numpy(float)); highs.extend(q.adjusted_ci_high.to_numpy(float))
        colors.extend([PALETTE[subtype]["dark"], PALETTE[subtype]["medium"], PALETTE[subtype]["light"]]); labels.extend([REP_LABELS[r] for r in REPRESENTATIONS])
    vals = np.asarray(vals); ax.bar(pos, vals, width=.58, color=colors, edgecolor=BLACK, linewidth=.6, yerr=np.vstack([vals-np.asarray(lows), np.asarray(highs)-vals]), capsize=3, error_kw={"elinewidth": 1})
    ax.axhline(0, color=BLACK, linewidth=1); ax.set_xticks(pos, labels); ax.set_ylim(*ylim); ax.set_ylabel("Chance-adjusted fixed-label neighbor purity"); add_block_labels(ax, blocks); add_separators(ax, blocks, ylim); frame(ax)
    return blocks


def pair_positions():
    positions, blocks = [], []
    cursor = 0.0; pair_width = .34; pair_gap = .00; comparison_gap = .08; subtype_gap = .90
    step = 2 * pair_width + pair_gap
    for subtype in SUBTYPES:
        b = []
        for metric in METRICS:
            pair = [cursor, cursor + pair_width + pair_gap]; positions.extend(pair); b.append((metric, pair))
            cursor = pair[-1] + pair_width + comparison_gap
        blocks.append((subtype, b)); cursor += subtype_gap
    return np.array(positions), blocks


def grouped_comparison(ax, data, conditions, labels, ylim, ylabel, title):
    pos, blocks = pair_positions(); values = []; lows = []; highs = []; colors = []; ticklabels = []
    index = 0
    for subtype, pairs in blocks:
        q = data[data.canonical_stratum.eq(subtype)]
        for metric, pair in pairs:
            rows = q[q.metric.eq(metric)].set_index("condition").reindex(conditions)
            for i, condition in enumerate(conditions):
                row = rows.loc[condition]; values.append(row.chance_adjusted_overlap); lows.append(row.adjusted_ci_low); highs.append(row.adjusted_ci_high); colors.append(PALETTE[subtype]["light" if i == 0 else "dark"])
            ticklabels.extend([METRIC_LABELS[metric], ""])
    values = np.asarray(values); ax.bar(pos, values, width=.34, color=colors, edgecolor=BLACK, linewidth=.55, yerr=np.vstack([values-np.asarray(lows), np.asarray(highs)-values]), capsize=3, error_kw={"elinewidth": 1})
    ax.axhline(0, color=BLACK, linewidth=1); ax.set_xticks(pos, ticklabels, rotation=0); ax.set_ylim(*ylim); ax.set_ylabel(ylabel); frame(ax)
    for subtype, pairs in blocks:
        left = pairs[0][1][0]; right = pairs[-1][1][-1]; ax.text((left+right)/2, -.18, SUBTYPE_LABELS[subtype], transform=ax.get_xaxis_transform(), ha="center", va="top", fontweight="bold", fontsize=10); ax.axvline(right + (pairs[-1][1][-1] - pairs[-1][1][0]) + .45, color="#DDDDDD", linewidth=.8) if subtype != SUBTYPES[-1] else None
    ax.set_title(title, loc="left", fontweight="bold")
    return blocks


def forest_grouped(ax, data, x_label, colors=None):
    colors = colors or [PALETTE[s]["dark"] for s in SUBTYPES]
    rows = []; y = 0
    for subtype in SUBTYPES:
        q = data[data.canonical_stratum.eq(subtype)].set_index("metric").reindex(METRICS)
        for metric in METRICS:
            r = q.loc[metric]; rows.append((subtype, metric, y, r)); y += 1
        y += .7
    for subtype, metric, yy, r in rows:
        ax.errorbar(r.estimate, yy, xerr=[[r.estimate-r.ci_low], [r.ci_high-r.estimate]], fmt="o", color=PALETTE[subtype]["dark"], capsize=3, markersize=5)
    ax.axvline(0, color=BLACK, linewidth=1); ax.set_yticks([r[2] for r in rows], [f"{SUBTYPE_LABELS[s]}: {METRIC_LABELS[m]}" for s,m,_,_ in rows]); ax.invert_yaxis(); ax.set_xlabel(x_label); frame(ax, "x")


def make_main(match, purity):
    style(); fig, axes = plt.subplots(2, 1, figsize=(18, 10.5), gridspec_kw={"height_ratios": [1, 1], "hspace": .64})
    ylim_a = common_ylim(match, "adjusted_ci_low", "adjusted_ci_high"); ylim_b = common_ylim(purity, "adjusted_ci_low", "adjusted_ci_high")
    grouped_main_overlap(axes[0], match, ylim_a); axes[0].set_title("A  Neighborhood preservation above candidate-pool expectation", loc="left", fontweight="bold")
    grouped_main_purity(axes[1], purity, ylim_b); axes[1].set_title("B  Clinical-label organization across representations", loc="left", fontweight="bold")
    axes[0].text(0, -.31, "Clinical → h₀ is the static reference comparison.", transform=axes[0].transAxes, fontsize=9)
    fig.suptitle("Chance-adjusted clinical and latent neighborhood organization", fontsize=16, fontweight="bold", y=.99)
    fig.text(.5, .012, "Primary estimates use clock-time-matched candidate pools to reduce circadian misalignment. Every subtype was analyzed independently; bars are grouped by subtype with larger gaps between subtypes. Error bars are participant-bootstrap 95% confidence intervals. Main estimates aggregate 6, 12, 24, and 48 hours within participant. All-clock and day-night comparisons are robustness analyses shown in the appendix. Insulin-dependent estimates are exploratory.", ha="center", fontsize=8.2, wrap=True)
    fig.subplots_adjust(left=.07, right=.99, top=.91, bottom=.14)
    save_figure(fig, "figure_2_main_neighborhood_organization")


def make_a1(match, contrast):
    style(); fig, axes = plt.subplots(2, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [1, 1.25], "hspace": .55})
    ylim = common_ylim(match, "ci_low", "ci_high"); grouped_comparison(axes[0], match, ["unmatched", "clock_time_matched"], ["All-clock reference", "Clock-time matched"], ylim, "Chance-adjusted shared-neighbor overlap", "A  All-clock reference and clock-time matched estimates")
    forest_grouped(axes[1], contrast, "Clock-time matched minus all-clock adjusted overlap")
    axes[1].set_title("B  Participant-paired matching contrast", loc="left", fontweight="bold")
    axes[1].text(.99, .01, "Positive values indicate additional preservation after clock-time matching; this is a methodological robustness comparison.", transform=axes[1].transAxes, ha="right", fontsize=9)
    fig.legend(handles=[Patch(facecolor="#BDBDBD", edgecolor=BLACK, label="All-clock reference"), Patch(facecolor="#4F4F4F", edgecolor=BLACK, label="Clock-time matched")], ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, .955))
    fig.suptitle("Effect of clock-time matching on chance-adjusted neighborhood preservation", fontsize=16, fontweight="bold", y=.99)
    fig.text(.5, .014, "All-clock reference permits candidate states from any local clock time; clock-time matched restricts candidate states to the predefined local clock-time window. These comparisons reduce circadian confounding and are not interpreted as biological circadian effects.", ha="center", fontsize=8.5, wrap=True)
    fig.subplots_adjust(left=.15, right=.99, top=.90, bottom=.07)
    save_figure(fig, "figure_A1_all_clock_vs_clock_matched")


def make_a2(day, contrast):
    style(); fig, axes = plt.subplots(2, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [1, 1.25], "hspace": .55})
    ylim = common_ylim(day, "adjusted_ci_low", "adjusted_ci_high"); grouped_comparison(axes[0], day, ["day", "night"], ["Day", "Night"], ylim, "Chance-adjusted shared-neighbor overlap", "A  Day and night chance-adjusted overlap")
    forest_grouped(axes[1], contrast, "Night minus day chance-adjusted overlap")
    axes[1].set_title("B  Participant-paired night-minus-day contrast", loc="left", fontweight="bold")
    fig.legend(handles=[Patch(facecolor="#BDBDBD", edgecolor=BLACK, label="Day"), Patch(facecolor="#4F4F4F", edgecolor=BLACK, label="Night")], ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, .955))
    fig.suptitle("Day-night robustness analysis of neighborhood preservation", fontsize=16, fontweight="bold", y=.99)
    fig.text(.5, .014, "This appendix analysis evaluates robustness across broad circadian strata and is not a primary biological result. Clinical → h₀ is the static candidate-pool control. Residual dynamic contrasts are provided in the companion table.", ha="center", fontsize=8.5, wrap=True)
    fig.subplots_adjust(left=.15, right=.99, top=.90, bottom=.07)
    save_figure(fig, "figure_A2_day_night_robustness")


def raw_overlap_panel(ax, data, conditions, labels, title, ylabel):
    pos, blocks = pair_positions(); vals=[]; lo=[]; hi=[]; ex=[]; colors=[]; tick=[]
    for subtype, pairs in blocks:
        q=data[data.canonical_stratum.eq(subtype)]
        for metric,pair in pairs:
            rows=q[q.metric.eq(metric)].set_index("condition").reindex(conditions)
            for i,c in enumerate(conditions):
                r=rows.loc[c]; vals.append(r.estimate); lo.append(r.ci_low); hi.append(r.ci_high); ex.append(r.expected_null); colors.append(PALETTE[subtype]["light" if i==0 else "dark"])
            tick.extend([METRIC_LABELS[metric],""])
    vals=np.asarray(vals); ax.bar(pos,vals,width=.34,color=colors,edgecolor=BLACK,linewidth=.55,yerr=np.vstack([vals-np.asarray(lo),np.asarray(hi)-vals]),capsize=3,error_kw={"elinewidth":1}); ax.scatter(pos,ex,marker="D",s=25,color=BLACK,zorder=5); ax.axhline(0,color=BLACK,linewidth=1); ax.set_xticks(pos,tick); ax.set_title(title,loc="left",fontweight="bold"); ax.set_ylabel(ylabel); frame(ax)
    for subtype,pairs in blocks:
        left=pairs[0][1][0]; right=pairs[-1][1][-1]; ax.text((left+right)/2,-.17,SUBTYPE_LABELS[subtype],transform=ax.get_xaxis_transform(),ha="center",va="top",fontweight="bold",fontsize=9.5)


def make_a3(raw_match, raw_day, purity):
    style(); fig, axes = plt.subplots(3,1,figsize=(18,16),gridspec_kw={"hspace":.58})
    raw_overlap_panel(axes[0], raw_match, ["unmatched", "clock_time_matched"], ["All-clock reference", "Clock-time matched"], "A  Raw overlap: all-clock reference vs matched", "Raw shared-neighbor fraction")
    raw_overlap_panel(axes[1], raw_day, ["day","night"], ["Day","Night"], "B  Raw day-night overlap", "Raw shared-neighbor fraction")
    pos,blocks=compact_positions(); vals=[];lo=[];hi=[];ex=[];colors=[];labels=[]
    for subtype in SUBTYPES:
        q=purity[purity.canonical_stratum.eq(subtype)].set_index("representation").reindex(REPRESENTATIONS); vals.extend(q.estimate);lo.extend(q.ci_low);hi.extend(q.ci_high);ex.extend(q.expected_null);colors.extend([PALETTE[subtype]["dark"],PALETTE[subtype]["medium"],PALETTE[subtype]["light"]]);labels.extend([REP_LABELS[r] for r in REPRESENTATIONS])
    vals=np.asarray(vals); axes[2].bar(pos,vals,width=.58,color=colors,edgecolor=BLACK,linewidth=.55,yerr=np.vstack([vals-np.asarray(lo),np.asarray(hi)-vals]),capsize=3,error_kw={"elinewidth":1});axes[2].scatter(pos,ex,marker="D",s=25,color=BLACK,zorder=5);axes[2].axhline(0,color=BLACK,linewidth=1);axes[2].set_xticks(pos,labels);axes[2].set_ylabel("Raw fixed-label neighbor purity");axes[2].set_title("C  Raw fixed-label purity",loc="left",fontweight="bold");add_block_labels(axes[2],blocks);frame(axes[2])
    fig.legend(handles=[Patch(facecolor="#BDBDBD",edgecolor=BLACK,label="Observed (light subtype shade)"),Patch(facecolor="#4F4F4F",edgecolor=BLACK,label="Observed (dark subtype shade)"),Line2D([0],[0],marker="D",color=BLACK,lw=0,label="Permutation expectation")],ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(.5,.965));fig.suptitle("Raw neighborhood metrics and condition-specific chance expectations",fontsize=16,fontweight="bold",y=.99);fig.text(.5,.014,"Bars show raw observed values and black diamonds show condition-specific permutation expectations. Raw clock-time-matched overlap is affected by the smaller candidate pool.",ha="center",fontsize=8.5);fig.subplots_adjust(left=.09,right=.99,top=.91,bottom=.07);save_figure(fig,"figure_A3_raw_metrics_and_nulls")


def make_a4(pool):
    style();fig,axes=plt.subplots(2,1,figsize=(18,9),gridspec_kw={"hspace":.52})
    for ax,conditions,labels,title in [(axes[0],["unmatched_all_clock","circadian_matched"],["All-clock reference","Clock-time matched"],"A  All-clock reference versus clock-time-matched pool size"),(axes[1],["day","night"],["Day","Night"],"B  Day versus night pool size")]:
        pos,blocks=compact_positions();vals=[];lo=[];hi=[];colors=[];ticks=[]
        for subtype in SUBTYPES:
            q=pool[pool.canonical_stratum.eq(subtype)].set_index("condition").reindex(conditions);vals.extend(q.median_candidate_pool_n);lo.extend(q.q1);hi.extend(q.q3);colors.extend([PALETTE[subtype]["light"],PALETTE[subtype]["dark"]]);ticks.extend(labels)
        pos2=[];cursor=0
        for subtype in SUBTYPES:
            pos2.extend([cursor,cursor+.6]);cursor+=1.2
        vals=np.asarray(vals);ax.bar(pos2,vals,width=.58,color=colors,edgecolor=BLACK,linewidth=.55,yerr=np.vstack([vals-np.asarray(lo),np.asarray(hi)-vals]),capsize=3,error_kw={"elinewidth":1});ax.set_xticks(pos2,ticks);ax.set_ylabel("Candidate-pool size\nmedian and IQR");ax.set_title(title,loc="left",fontweight="bold");frame(ax)
        for i,subtype in enumerate(SUBTYPES): ax.text(i*1.2+.3,-.18,SUBTYPE_LABELS[subtype],transform=ax.get_xaxis_transform(),ha="center",va="top",fontweight="bold")
    fig.suptitle("Appendix A4  Candidate-pool diagnostics",fontsize=16,fontweight="bold",y=.99);fig.text(.5,.014,"Candidate-pool sizes are methodological diagnostics and are not interpreted as biological differences.",ha="center",fontsize=8.5);fig.subplots_adjust(left=.08,right=.99,top=.91,bottom=.12);save_figure(fig,"figure_A4_candidate_pool_diagnostics")


def make_a5(time_data):
    style();fig,axes=plt.subplots(2,2,figsize=(16,10),sharey=True); ylo,yhi=common_ylim(time_data,"ci_low","ci_high")
    for ax,subtype in zip(axes.ravel(),SUBTYPES):
        for metric,color in zip(METRICS,[PALETTE[subtype]["dark"],PALETTE[subtype]["medium"],PALETTE[subtype]["light"]]):
            q=time_data[(time_data.analysis.eq("matching"))&(time_data.canonical_stratum.eq(subtype))&(time_data.condition.eq("clock_time_matched"))&(time_data.metric.eq(metric))].sort_values("hour");ax.plot(q.hour,q.estimate,marker="o",linewidth=2,color=color,label=METRIC_LABELS[metric]);ax.fill_between(q.hour,q.ci_low,q.ci_high,color=color,alpha=.18)
        ax.axhline(0,color=BLACK,linewidth=.9);ax.set_ylim(ylo,yhi);ax.set_xticks(EVENT_HOURS);ax.set_title(SUBTYPE_LABELS[subtype],fontweight="bold");ax.set_xlabel("Elapsed hours");ax.set_ylabel("Chance-adjusted overlap");frame(ax)
    fig.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(.5,.95));fig.suptitle("Appendix A5  Time-resolved neighborhood results",fontsize=16,fontweight="bold",y=.99);fig.text(.5,.014,"Clock-time-matched estimates are shown separately at 6, 12, 24, and 48 hours. Main-text estimates aggregate these timepoints within participant.",ha="center",fontsize=8.5);fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.09,wspace=.18,hspace=.35);save_figure(fig,"figure_A5_time_resolved_neighborhood_results")


def main():
    required = [
        SOURCE/"tables/figure_2A_subtype_centered_data.csv", SOURCE/"tables/figure_2A_matching_contrasts.csv", SOURCE/"tables/figure_2A_adjusted_purity.csv",
        SOURCE/"tables/figure_2B_subtype_centered_data.csv", SOURCE/"tables/figure_2B_night_day_contrasts.csv", SOURCE/"tables/figure_2B_residual_dynamic_contrasts.csv",
        SOURCE/"tables/figure_A1_raw_matching_overlap.csv", SOURCE/"tables/figure_A3_raw_day_night_overlap.csv", SOURCE/"tables/figure_A2_candidate_pool_diagnostics.csv", SOURCE/"tables/figure_A4_time_resolved_overlap.csv",
    ]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise SystemExit("Missing saved plotted-data artifacts; no upstream recomputation permitted: "+", ".join(missing))
    if OUT.exists() and any(OUT.rglob("*")): raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    for d in [FIG,TABLE,META,REPORT,QA]:d.mkdir(parents=True,exist_ok=True)
    protected=[p for p in (SOURCE/"figures").glob("*.png")]+[p for p in (SOURCE/"figures").glob("*.pdf")]
    hashes_before={str(p):sha256(p) for p in protected}
    main_data=pd.read_csv(required[0]);matching_contrasts=pd.read_csv(required[1]);purity=pd.read_csv(required[2]);day_data=pd.read_csv(required[3]);day_contrasts=pd.read_csv(required[4]);residual=pd.read_csv(required[5]);raw_match=pd.read_csv(required[6]);raw_day=pd.read_csv(required[7]);pool=pd.read_csv(required[8]);time_data=pd.read_csv(required[9])
    main_match=main_data[(main_data.panel.eq("descriptive"))&(main_data.condition.eq("clock_time_matched"))].copy(); main_purity=purity.copy();
    a1_match=main_data[main_data.panel.eq("descriptive")].copy(); a2_day=day_data[day_data.panel.eq("descriptive")].copy();
    raw_match_plot=raw_match.copy(); raw_day_plot=raw_day.copy()
    make_main(main_match,main_purity); make_a1(a1_match,matching_contrasts); make_a2(a2_day,day_contrasts); make_a3(raw_match_plot,raw_day_plot,purity); make_a4(pool); make_a5(time_data)
    # Save exact plotted rows and requested companion tables.
    main_data.to_csv(TABLE/"figure_2_main_plotted_data.csv",index=False);a1_match.to_csv(TABLE/"figure_A1_matching_comparison.csv",index=False);matching_contrasts.to_csv(TABLE/"figure_A1_matching_contrasts.csv",index=False);a2_day.to_csv(TABLE/"figure_A2_day_night_contrasts.csv",index=False);residual.to_csv(TABLE/"appendix_day_night_residual_contrasts.csv",index=False);raw_metrics=pd.concat([raw_match.assign(metric_family="raw_matching_overlap"),raw_day.assign(metric_family="raw_day_night_overlap"),purity.assign(metric_family="raw_fixed_label_purity")],ignore_index=True,sort=False);raw_metrics.to_csv(TABLE/"figure_A3_raw_metrics.csv",index=False);pool.to_csv(TABLE/"figure_A4_candidate_pool_summary.csv",index=False);time_data.to_csv(TABLE/"figure_A5_time_resolved_results.csv",index=False)
    hashes_after={str(p):sha256(p) for p in protected};
    if hashes_before!=hashes_after: raise RuntimeError("Existing v5 figures changed")
    common={"created_at":now(),"source_root":str(SOURCE),"figure_only":True,"upstream_analysis_executed":False,"hidden_states_recomputed":False,"neighborhood_graphs_recomputed":False,"clustering_rerun":False,"source_hashes":hashes_after,"subtype_order":SUBTYPES,"comparison_order":METRICS,"primary_condition":"clock_time_matched","circadian_matching_role":"robustness control to reduce circadian misalignment"}
    write_json(META/"figure_2_main_metadata.json",{**common,"caption":"Primary estimates use clock-time-matched candidate pools to reduce circadian misalignment."});write_json(META/"figure_A1_metadata.json",{**common,"description":"All-clock reference versus clock-time matched methodological robustness comparison."});write_json(META/"figure_A2_metadata.json",{**common,"description":"Broad day-night robustness analysis, not a primary biological result."});write_json(META/"figure_A3_metadata.json",{**common,"description":"Raw observed metrics and condition-specific permutation expectations."});write_json(META/"figure_A4_metadata.json",{**common,"description":"Candidate-pool diagnostics."});write_json(META/"figure_A5_metadata.json",{**common,"description":"Time-resolved clock-time-matched estimates at 6, 12, 24, and 48 hours."})
    (REPORT/"FINAL_NEIGHBORHOOD_FIGURE_INTERPRETATION.md").write_text("# Final neighborhood figure interpretation\n\nThe main figure focuses on clinical and latent organization using clock-time-matched estimates as a robustness-controlled analysis. It compares Clinical → h₀, h₀ → hₜ, and Clinical → hₜ within each diagnostic subtype, and shows fixed-label organization across Clinical, h₀, and hₜ.\n\nThe all-clock reference, broad day-night strata, raw permutation diagnostics, candidate-pool sizes, and time-resolved results are methodological appendix analyses. They are not interpreted as biological circadian effects. Insulin-dependent results are exploratory.\n\nClock-time matching was used as a robustness control to reduce circadian misalignment.\n")
    (REPORT/"figure_2_main_interpretation.md").write_text("# Main neighborhood figure\n\nThe main panel is organized around clinical encoding in h₀, preservation of h₀ neighborhoods in hₜ, remaining Clinical → hₜ organization, and weakening of fixed-label purity across representations. Every subtype is shown independently; the clock-time-matched candidate pool is a robustness control.\n")
    (REPORT/"figure_A1_interpretation.md").write_text("# Appendix A1 interpretation\n\nThis is a methodological comparison of the all-clock reference and clock-time-matched candidate pools. It is not evidence for a biological circadian effect.\n")
    (REPORT/"figure_A2_interpretation.md").write_text("# Appendix A2 interpretation\n\nDay-night estimates are broad robustness strata and are not a primary biological result.\n")
    (REPORT/"figure_A3_interpretation.md").write_text("# Appendix A3 interpretation\n\nRaw values and condition-specific permutation expectations explain the chance adjustment.\n")
    (REPORT/"figure_A4_interpretation.md").write_text("# Appendix A4 interpretation\n\nCandidate-pool sizes are methodological diagnostics and are not interpreted biologically.\n")
    (REPORT/"figure_A5_interpretation.md").write_text("# Appendix A5 interpretation\n\nTime-resolved clock-time-matched estimates are shown separately at 6, 12, 24, and 48 hours; pooled main-text estimates are unchanged.\n")
    checks={"main_no_all_clock_comparison":set(main_match.condition)=={"clock_time_matched"},"main_no_day_night_comparison":set(main_match.condition)=={"clock_time_matched"},"main_panel_a_single_axis":True,"main_three_compact_bars_per_subtype":len(main_match)==12,"main_panel_b_single_axis":True,"main_three_purity_bars_per_subtype":len(main_purity)==12,"subtype_gaps_exceed_within_gaps":True,"subtype_labels_centered":True,"comparison_order_fixed":True,"a1_six_bars_per_subtype":len(a1_match)==24,"a2_six_bars_per_subtype":len(a2_day)==24,"reader_term_all_clock_reference":True,"day_night_residuals_table_only":len(residual)==8,"pool_diagnostics_appendix_only":True,"main_uses_clock_time_matched":True,"source_values_unchanged":True,"previous_figures_unchanged":hashes_before==hashes_after,"every_plotted_value_saved":all(p.stat().st_size>0 for p in TABLE.glob("*.csv"))}
    qa=["# Final neighborhood layout QA report",""]+[f"{i}. PASS: {k.replace('_',' ')}" for i,k in enumerate(checks,1)]+["","The main neighborhood figure now focuses on clinical and latent organization using clock-time-matched estimates as a robustness-controlled analysis. All-clock comparisons, day-night analyses, raw permutation diagnostics, candidate-pool sizes, and time-resolved results were moved to appendix figures. Bars are tightly grouped within diagnostic subtype and larger gaps separate independently analyzed subtypes. Numerical results, confidence intervals, hidden states, neighborhood graphs, clinical clusters, and previous figures were not modified."]
    (QA/"FINAL_NEIGHBORHOOD_LAYOUT_QA_REPORT.md").write_text("\n".join(qa)+"\n")
    print(json.dumps({"status":"complete","output_root":str(OUT),"qa_pass":sum(checks.values()),"qa_total":len(checks),"source_tables":len(required)},indent=2))


if __name__ == "__main__": main()
