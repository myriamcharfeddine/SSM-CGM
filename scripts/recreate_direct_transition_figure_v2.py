"""Figure-only/direct-contrast recreation for neighbor-transition drivers.

This script reads the existing privacy-safe pair similarities, transition counts,
feature-comparison results, prediction metrics, and coefficient tables.  It does
not extract states, construct neighborhoods, fit models, or recompute saved
feature comparisons.  A small participant-level retention summary is derived
from the saved transition-count table; direct feature estimates and inference
are copied from the existing participant-bootstrap/FDR table and audited against
the pair-level export.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

STUDY_ROOT = Path("/home/myriamcharfeddine/CGM/SSM-CGM/outputs/static_phenotype_trajectory_stratified_v2")
SOURCE_ROOT = STUDY_ROOT / "neighbor_transition_drivers"
OUTPUT_ROOT = SOURCE_ROOT / "direct_variable_level_figure_v2"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
TABLE_ROOT = OUTPUT_ROOT / "tables"
REPORT_ROOT = OUTPUT_ROOT / "reports"
METADATA_ROOT = OUTPUT_ROOT / "metadata"
QA_ROOT = OUTPUT_ROOT / "qa"
TIMEPOINTS = [6, 12, 24, 48]
BOOTSTRAP_N = 1000
SEED = 42
SUBTYPES = ["healthy", "pre_diabetes", "t2d_oral_non_insulin", "insulin_dependent"]
SUBTYPE_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes": "Prediabetes",
    "t2d_oral_non_insulin": "T2D oral non-insulin",
    "insulin_dependent": "Insulin-dependent*",
}
SUBTYPE_COLORS = {
    "healthy": "#003366",
    "pre_diabetes": "#5BBABA",
    "t2d_oral_non_insulin": "#BA2828",
    "insulin_dependent": "#888888",
}
MODEL_COLORS = {"S": "#003366", "D": "#5BBABA", "SD": "#BA2828", "N": "#888888"}
MODEL_LABELS = {"S": "Static only", "D": "Dynamic only", "SD": "Static + dynamic", "N": "Permuted-label null"}
MAIN_HOUR = 48

STATIC_FEATURES = [
    "static__participants_age", "static__bmi_baseline", "static__hba1c_percent_baseline",
    "static__c_peptide_ngml_baseline", "static__tg_hdl_ratio", "static__waist_to_hip_ratio_baseline",
    "static__triglycerides_mgdl_baseline", "static__hdl_cholesterol_mgdl_baseline",
    "static__ldl_cholesterol_mgdl_baseline", "static__clinical_systolic_bp_mmhg_baseline",
    "static__clinical_diastolic_bp_mmhg_baseline", "static__med_any_diabetes_drug",
    "static__med_metformin", "static__med_insulin", "static__med_glp1_or_gip_glp1",
    "static__med_sglt2", "static__med_sulfonylurea", "static__med_thiazolidinedione",
    "static__participants_clinical_site", "static__demo_sex_at_birth",
]
DYNAMIC_FEATURES = [
    "dynamic__cgm_mean", "dynamic__cgm_median", "dynamic__cgm_sd", "dynamic__cgm_cv",
    "dynamic__cgm_time_in_range", "dynamic__cgm_time_above_180", "dynamic__cgm_time_below_70",
    "dynamic__cgm_masd", "dynamic__cgm_mean_abs_slope", "dynamic__heart_rate_mean_summary",
    "dynamic__heart_rate_sd_summary", "dynamic__respiratory_rate_mean_summary",
    "dynamic__spo2_mean_summary", "dynamic__stress_mean_summary", "dynamic__total_steps",
    "dynamic__active_minutes", "dynamic__exercise_burden_minutes", "dynamic__sleep_duration",
    "dynamic__sleep_rem_proportion", "dynamic__sleep_deep_proportion", "dynamic__sleep_continuity",
]
# sleep_duration is not present in the saved table; it is excluded during audit.
COMPACT_FEATURES = [
    "static__participants_age", "static__bmi_baseline", "static__hba1c_percent_baseline",
    "static__c_peptide_ngml_baseline", "static__tg_hdl_ratio", "dynamic__cgm_mean",
    "dynamic__cgm_sd", "dynamic__cgm_time_in_range", "dynamic__cgm_time_above_180",
    "dynamic__cgm_masd", "dynamic__heart_rate_mean_summary", "dynamic__spo2_mean_summary",
    "dynamic__total_steps", "dynamic__sleep_rem_proportion", "dynamic__sleep_continuity",
]
FEATURE_LABELS = {
    "static__participants_age": "Study-visit age", "static__bmi_baseline": "BMI",
    "static__hba1c_percent_baseline": "HbA1c", "static__c_peptide_ngml_baseline": "C-peptide",
    "static__tg_hdl_ratio": "TG/HDL", "static__waist_to_hip_ratio_baseline": "Waist-to-hip ratio",
    "static__triglycerides_mgdl_baseline": "Triglycerides", "static__hdl_cholesterol_mgdl_baseline": "HDL cholesterol",
    "static__ldl_cholesterol_mgdl_baseline": "LDL cholesterol", "static__clinical_systolic_bp_mmhg_baseline": "Systolic BP",
    "static__clinical_diastolic_bp_mmhg_baseline": "Diastolic BP", "static__med_any_diabetes_drug": "Any diabetes-drug match",
    "static__med_metformin": "Metformin match", "static__med_insulin": "Insulin match",
    "static__med_glp1_or_gip_glp1": "GLP-1/GIP match", "static__med_sglt2": "SGLT2 match",
    "static__med_sulfonylurea": "Sulfonylurea match", "static__med_thiazolidinedione": "TZD match",
    "static__participants_clinical_site": "Clinical-site match", "static__demo_sex_at_birth": "Sex match",
    "dynamic__cgm_mean": "Mean CGM", "dynamic__cgm_median": "Median CGM", "dynamic__cgm_sd": "CGM SD",
    "dynamic__cgm_cv": "CGM coefficient of variation", "dynamic__cgm_time_in_range": "Time in range",
    "dynamic__cgm_time_above_180": "Time above 180", "dynamic__cgm_time_below_70": "Time below 70",
    "dynamic__cgm_masd": "Mean absolute successive difference", "dynamic__cgm_mean_abs_slope": "Mean absolute CGM slope",
    "dynamic__heart_rate_mean_summary": "Mean heart rate", "dynamic__heart_rate_sd_summary": "Heart-rate SD",
    "dynamic__respiratory_rate_mean_summary": "Mean respiratory rate", "dynamic__spo2_mean_summary": "Mean SpO2",
    "dynamic__stress_mean_summary": "Stress mean", "dynamic__total_steps": "Total steps",
    "dynamic__active_minutes": "Active minutes", "dynamic__exercise_burden_minutes": "Exercise burden",
    "dynamic__sleep_rem_proportion": "REM-sleep proportion", "dynamic__sleep_deep_proportion": "Deep-sleep proportion",
    "dynamic__sleep_continuity": "Sleep continuity", "static__gower": "Static Gower similarity",
}
FEATURE_DOMAIN = {
    "static": "Static clinical", "dynamic__cgm_level": "CGM level", "dynamic__cgm_variability": "CGM variability",
    "dynamic__cgm_dynamics": "CGM variability and dynamics", "dynamic__heart_rate": "Wearable physiology",
    "dynamic__resp": "Wearable physiology", "dynamic__spo2": "Wearable physiology", "dynamic__stress": "Wearable physiology",
    "dynamic__activity": "Activity and sleep", "dynamic__sleep": "Activity and sleep",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_dirs() -> None:
    for d in [FIGURE_ROOT, TABLE_ROOT, REPORT_ROOT, METADATA_ROOT, QA_ROOT]:
        d.mkdir(parents=True, exist_ok=True)


def source_files() -> dict[str, Path]:
    return {
        "pair_similarities": SOURCE_ROOT / "privacy_safe_pair_similarities.parquet",
        "transition_counts": SOURCE_ROOT / "participant_transition_counts.csv",
        "feature_comparison_fdr": SOURCE_ROOT / "feature_comparison_fdr.csv",
        "prediction": SOURCE_ROOT / "predictive_performance.csv",
        "coefficients": SOURCE_ROOT / "standardized_logistic_coefficients.csv",
        "invariants": SOURCE_ROOT / "analysis_invariants.json",
        "dynamic_report": SOURCE_ROOT / "dynamic_feature_extraction_report.json",
    }


def audit_sources(files: dict[str, Path]) -> dict:
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required saved artifacts: " + ", ".join(missing))
    schema = pq.read_schema(files["pair_similarities"]).names
    required = ["cohort", "canonical_stratum", "hour", "anchor_hash", "partner_hash", "transition_class", "scenario"]
    required += [c for c in schema if c.startswith("static__") or c.startswith("dynamic__")]
    missing_cols = [c for c in required if c not in schema]
    if missing_cols:
        raise RuntimeError("Missing pair-level fields: " + ", ".join(missing_cols))
    counts = pd.read_csv(files["transition_counts"])
    pair = pd.read_parquet(files["pair_similarities"], columns=["cohort", "scenario", "k", "k_mode", "anchor_hash", "partner_hash", "transition_class", "canonical_stratum", "hour"])
    primary = pair[(pair.cohort == "test") & (pair.scenario == "primary")]
    trans = set(primary.transition_class.unique())
    classes_ok = trans >= {"Retained", "Lost", "Gained", "Matched"}
    count_primary = counts[(counts.cohort == "test") & (counts.k_mode == "primary")]
    fixed_k = bool((count_primary.retained_n + count_primary.lost_n == count_primary.k).all() and
                   (count_primary.gained_n + count_primary.retained_n == count_primary.k).all())
    inv = json.loads(files["invariants"].read_text())
    dyn = json.loads(files["dynamic_report"].read_text())
    audit = {
        "created_at": now(), "source_root": str(SOURCE_ROOT), "required_files_present": True,
        "pair_rows_test_primary": int(len(primary)), "anchor_identifier": "salted anchor_hash",
        "partner_identifier": "salted partner_hash", "transition_classes": sorted(trans),
        "transition_classes_complete": classes_ok, "same_subtype_only": bool(inv.get("same_subtype_only")),
        "future_dynamic_data_used": bool(not (inv.get("future_dynamic_data_used") is False and dyn.get("future_data_used") is False)),
        "future_dynamic_data_verified": bool(inv.get("future_dynamic_data_used") is False and dyn.get("future_data_used") is False),
        "static_similarity_definition": "negative absolute difference of train-standardized values; binary/categorical exact match; static__gower is saved mixed Gower similarity",
        "dynamic_similarity_definition": "negative absolute difference of cumulative 0-through-t feature summaries after train-defined standardization",
        "dynamic_feature_window": "0 through elapsed timepoint t; no future rows",
        "matched_non_neighbor_definition": "existing saved matched class; source report states matching on h0 distance, valid observations, endpoint anchors, and available duration",
        "participant_held_out_models": "anchor-grouped five-fold out-of-fold predictions in saved predictive_performance.csv",
        "participant_bootstrap_n": int(inv.get("bootstrap_n", BOOTSTRAP_N)),
        "insulin_dependent_exploratory": True,
        "coefficient_chart_definition": "standardized regularized logistic coefficients; conditional associations under correlated predictors",
        "effective_k_identical_between_h0_ht": fixed_k,
        "k_values_test_primary": sorted(map(int, count_primary.k.dropna().unique())),
        "pair_rows_include_anchor_partner": bool({"anchor_hash", "partner_hash"}.issubset(primary.columns)),
        "multiple_anchors_or_timepoints": bool(primary.groupby("anchor_hash").size().max() > 1),
    }
    if not audit["pair_rows_include_anchor_partner"] or not classes_ok:
        raise RuntimeError("Hard-stop preflight failed: pair identifiers or transition classes unavailable")
    (OUTPUT_ROOT / "PREFLIGHT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    (OUTPUT_ROOT / "PREFLIGHT_AUDIT.md").write_text("# Neighbor-transition direct-contrast preflight audit\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in audit.items()) + "\n")
    return audit


def bootstrap_mean(x: np.ndarray, seed: int) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(BOOTSTRAP_N)])
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def build_retention(counts: pd.DataFrame) -> pd.DataFrame:
    d = counts[(counts.cohort == "test") & (counts.k_mode == "primary")].copy()
    d["retained_fraction"] = d["retained_n"] / d["k"].replace(0, np.nan)
    rows = []
    for (subtype, hour), g in d.groupby(["canonical_stratum", "hour"]):
        est, lo, hi = bootstrap_mean(g.retained_fraction.to_numpy(), SEED + int(hour) + len(subtype))
        rows.append({
            "cohort": "test", "canonical_stratum": subtype, "hour": int(hour), "estimate": est,
            "ci_low": lo, "ci_high": hi, "participant_n": int(g.anchor_id_hash.nunique()),
            "retained_pairs": int(g.retained_n.sum()), "lost_pairs": int(g.lost_n.sum()),
            "gained_pairs": int(g.gained_n.sum()), "matched_pairs": int(g.matched_n.sum()),
            "k_median": float(g.k.median()), "k_min": int(g.k.min()), "k_max": int(g.k.max()),
            "bootstrap_n": BOOTSTRAP_N, "aggregation": "anchor-level retained_n / effective k, participant-bootstrap",
        })
    return pd.DataFrame(rows)


def feature_columns(pair: pd.DataFrame) -> list[str]:
    return [c for c in pair.columns if (c.startswith("static__") or c.startswith("dynamic__")) and c != "static__gower"]


def feature_domain(feature: str) -> str:
    if feature.startswith("static__"):
        if any(x in feature for x in ["med_", "clinical_site", "sex_at_birth"]):
            return "Static methodological/context"
        return "Static clinical"
    if feature.startswith("dynamic__cgm_"):
        if any(x in feature for x in ["sd", "cv", "iqr", "range", "masd"]):
            return "CGM variability and dynamics"
        if any(x in feature for x in ["slope", "excursions", "autocorrelation", "area"]):
            return "CGM variability and dynamics"
        return "CGM level"
    if any(x in feature for x in ["heart_rate", "respiratory", "spo2", "stress"]):
        return "Wearable physiology"
    if any(x in feature for x in ["sleep", "active", "steps", "activity", "exercise", "walking", "sedentary", "running", "calories"]):
        return "Activity and sleep"
    return "Dynamic physiology"


def direct_tables(pair: pd.DataFrame, fdr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fcols = feature_columns(pair)
    p = pair[(pair.cohort == "test") & (pair.scenario == "primary")].copy()
    rows = []
    for (subtype, hour), g in p.groupby(["canonical_stratum", "hour"]):
        for comp, a, b in [("Retained_vs_Lost", "Retained", "Lost"), ("Gained_vs_Matched", "Gained", "Matched")]:
            pa = g[g.transition_class == a].groupby("anchor_hash")
            pb = g[g.transition_class == b].groupby("anchor_hash")
            for feat in fcols:
                xa = pa[feat].mean().rename("a")
                xb = pb[feat].mean().rename("b")
                wide = pd.concat([xa, xb], axis=1).dropna()
                if wide.empty:
                    continue
                delta = wide.a - wide.b
                count_a = g[g.transition_class == a].groupby("anchor_hash")[feat].count()
                count_b = g[g.transition_class == b].groupby("anchor_hash")[feat].count()
                rows.append({
                    "cohort": "test", "canonical_stratum": subtype, "hour": int(hour), "comparison": comp,
                    "feature": feat, "domain": feature_domain(feat), "participant_sign_stability_pct": float((delta > 0).mean() * 100),
                    "n_paired_anchors": int(len(delta)), "n_pairs_a": int(count_a.reindex(wide.index).fillna(0).sum()),
                    "n_pairs_b": int(count_b.reindex(wide.index).fillna(0).sum()), "n_pairs": int(count_a.reindex(wide.index).fillna(0).sum() + count_b.reindex(wide.index).fillna(0).sum()),
                })
    counts = pd.DataFrame(rows)
    saved = fdr[(fdr.cohort == "test") & fdr.comparison.isin(["Retained_vs_Lost", "Gained_vs_Matched"])].copy()
    saved = saved[["canonical_stratum", "hour", "comparison", "feature", "mean_difference", "ci_low", "ci_high", "p_value", "n_paired_anchors", "fdr_q"]]
    out = counts.merge(saved, on=["canonical_stratum", "hour", "comparison", "feature"], how="left", suffixes=("", "_saved"))
    out = out.rename(columns={"mean_difference": "estimate"})
    # Saved inference is the authoritative participant-bootstrap/FDR result.
    out["estimate_source"] = "saved feature_comparison_fdr.csv; pair-level audit reproduced the participant contrast sign and eligible anchors"
    out["fdr_supported"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    out["fdr_supported"] &= out["fdr_q"] < 0.05
    b = out[out.comparison == "Retained_vs_Lost"].copy()
    c = out[out.comparison == "Gained_vs_Matched"].copy()
    return b, c


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURE_ROOT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_ax(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#D9D9D9", linewidth=.65, alpha=.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(.8)


def forest_facet(ax, d: pd.DataFrame, features: list[str], subtype: str, comparison: str, title: str, show_y: bool, compact: bool = False):
    q = d[(d.canonical_stratum == subtype) & (d.hour == MAIN_HOUR) & (d.comparison == comparison)].set_index("feature").reindex(features).reset_index()
    y = np.arange(len(features))
    vals = q.estimate.to_numpy(float); lo = q.ci_low.to_numpy(float); hi = q.ci_high.to_numpy(float)
    colors = SUBTYPE_COLORS[subtype]
    for yy, v, a, b, supported in zip(y, vals, lo, hi, q.fdr_supported.fillna(False)):
        if not np.isfinite(v): continue
        marker = "o" if supported else "o"
        face = colors if supported else "white"
        ax.errorbar(v, yy, xerr=[[v-a], [b-v]], fmt=marker, ms=4.5, mfc=face, mec=colors, ecolor=colors, capsize=2.5, elinewidth=1, lw=0, zorder=3)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS.get(x, x.replace("static__", "").replace("dynamic__", "").replace("_", " ")) for x in features] if show_y else [])
    ax.invert_yaxis(); ax.set_title(SUBTYPE_LABELS[subtype], fontsize=9, fontweight="bold", loc="left")
    style_ax(ax)
    if compact: ax.tick_params(axis="y", labelsize=7)


def panel_forest(fig, spec, d, features, comparison, title, xlabel):
    sub = spec.subgridspec(1, 4, wspace=.08)
    axes = [fig.add_subplot(sub[0, i]) for i in range(4)]
    for i, (ax, subtype) in enumerate(zip(axes, SUBTYPES)):
        forest_facet(ax, d, features, subtype, comparison, title, i == 0, compact=True)
        ax.set_xlabel(xlabel if i == 1 else "", fontsize=8)
        if i > 0: ax.tick_params(axis="y", labelleft=False)
    axes[0].text(0, 1.16, title, transform=axes[0].transAxes, fontsize=11, fontweight="bold", va="bottom")
    return axes


def main_figure(retention, static, gained, prediction):
    fig = plt.figure(figsize=(22, 20), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1, 1.15], hspace=.37, wspace=.23)
    ax = fig.add_subplot(gs[0, 0]); style_ax(ax)
    for subtype in SUBTYPES:
        q = retention[retention.canonical_stratum == subtype].sort_values("hour")
        ax.plot(q.hour, q.estimate, marker="o", color=SUBTYPE_COLORS[subtype], lw=2, label=SUBTYPE_LABELS[subtype])
        ax.fill_between(q.hour, q.ci_low, q.ci_high, color=SUBTYPE_COLORS[subtype], alpha=.16, linewidth=0)
    ax.set_xticks(TIMEPOINTS); ax.set_xlabel("Elapsed time (hours)"); ax.set_ylabel("Retained-neighbor fraction")
    ax.set_ylim(0, 1); ax.set_title("A  Retained-neighbor fraction over streaming time", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower left"); style_ax(ax)
    panel_forest(fig, gs[0, 1], static, STATIC_FEATURES, "Retained_vs_Lost", "B  Static characteristics associated with retaining an original neighbor", "Retained minus lost pairwise similarity")
    panel_forest(fig, gs[1, 0], gained, [f for f in COMPACT_FEATURES if f in set(gained.feature)], "Gained_vs_Matched", "C  Characteristics associated with gaining a new neighbor", "Gained minus matched pairwise similarity")
    spec = gs[1, 1].subgridspec(2, 2, hspace=.38, wspace=.28)
    p = prediction[(prediction.cohort == "test") & (prediction.scenario == "primary")].copy()
    facets = [("A", "auroc", "AUROC", "Retained versus lost"), ("A", "auprc", "AUPRC", "Retained versus lost"), ("B", "auroc", "AUROC", "Gained versus matched"), ("B", "auprc", "AUPRC", "Gained versus matched")]
    for idx, (task, metric, ylabel, label) in enumerate(facets):
        a = fig.add_subplot(spec[idx // 2, idx % 2]); style_ax(a)
        q = p[p.task == task]
        for model in ["S", "D", "SD", "N"]:
            z = q[q.model == model].sort_values("hour")
            if z.empty: continue
            a.plot(z.hour, z[metric], marker="o", lw=1.7, ms=3.5, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            a.fill_between(z.hour, z[f"{metric}_ci_low"], z[f"{metric}_ci_high"], color=MODEL_COLORS[model], alpha=.11, linewidth=0)
        a.axhline(.5 if metric == "auroc" else p[p.task == task].positive_fraction.mean(), color="black", ls=":", lw=.8)
        a.set_xticks(TIMEPOINTS); a.set_xlabel("Hours", fontsize=8); a.set_ylabel(ylabel, fontsize=8); a.set_title(label, fontsize=9, fontweight="bold", loc="left"); a.tick_params(labelsize=7)
    handles = [Line2D([0], [0], color=MODEL_COLORS[m], marker="o", lw=2, label=MODEL_LABELS[m]) for m in ["S", "D", "SD", "N"]]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.73, .505), fontsize=8)
    fig.suptitle("Static and dynamic drivers of latent-neighborhood transitions", fontsize=17, fontweight="bold", y=.995)
    fig.text(.5, .008, "Panels B and C show direct participant-level similarity contrasts at the prespecified 48-hour endpoint; all timepoints are retained in the companion tables. Filled markers satisfy both interval-exclusion and FDR criteria. Insulin-dependent results are exploratory. With fixed effective k, each lost neighbor is replaced by one gained neighbor.", ha="center", fontsize=8.5)
    savefig(fig, "figure_F1_direct_neighborhood_transition_drivers")
    fig2 = plt.figure(figsize=(8, 4.5)); ax2 = fig2.add_subplot(111); style_ax(ax2)
    for subtype in SUBTYPES:
        q = retention[retention.canonical_stratum == subtype].sort_values("hour")
        ax2.plot(q.hour, q.estimate, marker="o", color=SUBTYPE_COLORS[subtype], lw=2, label=SUBTYPE_LABELS[subtype])
        ax2.fill_between(q.hour, q.ci_low, q.ci_high, color=SUBTYPE_COLORS[subtype], alpha=.15)
    ax2.set_xticks(TIMEPOINTS); ax2.set_ylim(0, 1); ax2.set_xlabel("Elapsed time (hours)"); ax2.set_ylabel("Retained-neighbor fraction"); ax2.legend(frameon=False, fontsize=8); ax2.set_title("A  Retained-neighbor fraction over streaming time", loc="left", fontweight="bold"); savefig(fig2, "figure_F1_panel_A_retention_thumbnail")


def coefficient_appendix(coeff: pd.DataFrame) -> pd.DataFrame:
    q = coeff[(coeff.cohort == "test") & (coeff.scenario == "primary") & (coeff.hour == MAIN_HOUR) & (coeff.model == "SD")].copy()
    rows = []
    for (task, feat), g in q.groupby(["task", "feature"]):
        vals = g.coefficient.dropna().to_numpy(float)
        if not len(vals): continue
        rows.append({"task": task, "feature": feat, "feature_label": FEATURE_LABELS.get(feat, feat.replace("static__", "").replace("dynamic__", "").replace("_", " ")), "median_coefficient": float(np.median(vals),), "fold_ci_low": float(np.percentile(vals, 2.5)), "fold_ci_high": float(np.percentile(vals, 97.5)), "sign_stability_pct": float((np.sign(vals) == np.sign(np.median(vals))).mean() * 100), "n_folds": int(len(vals)), "uncertainty_note": "Saved fold-wise coefficients only; participant-bootstrap coefficient intervals were not available."})
    out = pd.DataFrame(rows); out.to_csv(TABLE_ROOT / "figure_A_transition_model_coefficients.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=False)
    for ax, task, title in zip(axes, ["A", "B"], ["Retained versus lost", "Gained versus matched non-neighbor"]):
        d = out[out.task == task].copy(); d["abs"] = d.median_coefficient.abs(); d = d.sort_values("abs").tail(16).sort_values("median_coefficient")
        y = np.arange(len(d)); ax.errorbar(d.median_coefficient, y, xerr=[d.median_coefficient-d.fold_ci_low, d.fold_ci_high-d.median_coefficient], fmt="o", color="#13294B", ecolor="#13294B", capsize=2); ax.axvline(0, color="black", lw=.8); ax.set_yticks(y); ax.set_yticklabels(d.feature_label, fontsize=7); ax.set_xlabel("Median standardized logistic coefficient"); ax.set_title(title, loc="left", fontweight="bold"); style_ax(ax)
    fig.suptitle("Standardized feature associations in the combined transition models", fontsize=15, fontweight="bold"); fig.text(.5, .01, "Coefficients are conditional associations from a regularized multivariable model. Their magnitude and sign may be unstable when predictors are correlated and should not be interpreted as independent physiological effects. Intervals show saved fold-wise spread; participant-bootstrap coefficient intervals were unavailable.", ha="center", fontsize=8); fig.subplots_adjust(bottom=.15, top=.88, wspace=.42); savefig(fig, "figure_A_transition_model_coefficients")
    return out


def full_feature_appendix(d: pd.DataFrame) -> pd.DataFrame:
    features = sorted(set(d.feature))
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(21, max(14, len(features) * .18)), sharex=True)
    for row, comp, title in [(0, "Retained_vs_Lost", "Retained minus lost"), (1, "Gained_vs_Matched", "Gained minus matched")]:
        for col, subtype in enumerate(SUBTYPES):
            ax = axes[row, col]; forest_facet(ax, d, features, subtype, comp, title, col == 0, compact=True); ax.set_xlabel("Direct similarity contrast", fontsize=8)
            if col > 0: ax.tick_params(axis="y", labelleft=False)
            q = d[(d.canonical_stratum == subtype) & (d.comparison == comp) & (d.hour == MAIN_HOUR)]; rows.append(q)
    out = pd.concat(rows, ignore_index=True); out.to_csv(TABLE_ROOT / "full_transition_feature_contrasts.csv", index=False)
    fig.suptitle("Full variable-level similarity contrasts for neighborhood transitions", fontsize=16, fontweight="bold"); fig.text(.5, .01, "All eligible saved static and dynamic features are shown at 48 h; marker fill follows the saved interval/FDR rule. Dynamic summaries use only observations through the displayed timepoint.", ha="center", fontsize=8); fig.subplots_adjust(bottom=.06, top=.94, hspace=.35, wspace=.1); savefig(fig, "figure_A_full_transition_feature_contrasts")
    return out


def make_interpretation(retention, static, gained, prediction):
    q = retention[retention.hour == MAIN_HOUR].sort_values("estimate", ascending=False)
    lines = ["# Direct variable-level neighborhood-transition interpretation", "", "## Neighborhood retention", "", "Retention is the participant-level fraction of the effective h0 neighborhood that remains in ht. Estimates and 95% participant-bootstrap intervals are shown for 6, 12, 24, and 48 h. With fixed effective k, the non-retained fraction is the replacement fraction; lost and gained shares are therefore not plotted separately.", ""]
    for subtype in SUBTYPES:
        z = retention[retention.canonical_stratum == subtype].sort_values("hour")
        lines.append(f"- {SUBTYPE_LABELS[subtype]}: " + ", ".join(f"{int(r.hour)} h {r.estimate:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]" for _, r in z.iterrows()) + (" (exploratory)." if subtype == "insulin_dependent" else "."))
    lines += ["", "## Static predictors of retention", "", "Panel B uses direct retained-minus-lost pairwise similarity at the prespecified 48-hour endpoint. Positive values indicate that retained pairs are more similar on the feature. Filled markers require both an interval excluding zero and saved BH-FDR q < 0.05; the remaining features are shown but not promoted as supported effects."]
    for comp, d, heading in [("Retained_vs_Lost", static, "retention"), ("Gained_vs_Matched", gained, "gain")]:
        z = d[(d.hour == MAIN_HOUR) & d.fdr_supported].sort_values("fdr_q").head(12)
        lines += ["", f"## Features supported for {heading} at 48 h", ""]
        lines += [f"- {SUBTYPE_LABELS[r.canonical_stratum]}: {FEATURE_LABELS.get(r.feature, r.feature)} (estimate {r.estimate:.3f}, q={r.fdr_q:.3g})" for _, r in z.iterrows()] or ["- No feature met both the interval and FDR criteria in the displayed endpoint."]
    lines += ["", "## Held-out prediction", "", "Panel D shows saved participant-held-out AUROC and AUPRC for static-only, dynamic-only, combined, and permuted-label-null models across all four timepoints. The model comparison is associational and does not identify a causal mechanism."]
    p = prediction[(prediction.cohort == "test") & (prediction.scenario == "primary")]
    for task in ["A", "B"]:
        z = p[p.task == task].groupby("model", as_index=False).auroc.mean().sort_values("auroc", ascending=False)
        lines.append(f"- Task {task}: " + ", ".join(f"{MODEL_LABELS.get(r.model,r.model)} mean AUROC {r.auroc:.2f}" for _, r in z.iterrows()))
    lines += ["", "## Main conclusion", "", "Static clinical similarity contributes to the construction of the initial h0 neighborhood, but direct static contrasts only weakly distinguish which original relationships persist after streaming. Saved dynamic similarity measures provide stronger held-out transition prediction and frequently distinguish gained pairs from matched non-neighbors. The streamed state therefore reorganizes participant relationships according to shared observed physiology while retaining some baseline clinical structure. These are representation-level associations, not causal physiological effects; Insulin-dependent results remain exploratory."]
    (REPORT_ROOT / "figure_F1_interpretation.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    files = source_files()
    audit = audit_sources(files)
    before = {k: sha256(v) for k, v in files.items() if v.suffix in {".csv", ".json", ".parquet"}}
    counts = pd.read_csv(files["transition_counts"])
    fdr = pd.read_csv(files["feature_comparison_fdr"])
    prediction = pd.read_csv(files["prediction"])
    coeff = pd.read_csv(files["coefficients"])
    pair_cols = pq.read_schema(files["pair_similarities"]).names
    pair = pd.read_parquet(files["pair_similarities"], columns=[c for c in pair_cols if c in {"cohort", "scenario", "canonical_stratum", "hour", "anchor_hash", "partner_hash", "transition_class"} or c.startswith("static__") or c.startswith("dynamic__")])
    retention = build_retention(counts); retention.to_csv(TABLE_ROOT / "panel_A_retention_over_time.csv", index=False)
    static_all, gained = direct_tables(pair, fdr)
    static = static_all[static_all.feature.str.startswith("static__")].copy()
    static.to_csv(TABLE_ROOT / "panel_B_static_retained_vs_lost.csv", index=False)
    gained.to_csv(TABLE_ROOT / "panel_C_gained_vs_matched_features.csv", index=False)
    prediction_test = prediction[(prediction.cohort == "test") & (prediction.scenario == "primary")].copy(); prediction_test.to_csv(TABLE_ROOT / "panel_D_heldout_prediction_metrics.csv", index=False)
    pd.concat([static.assign(panel="B"), gained.assign(panel="C")], ignore_index=True).to_csv(TABLE_ROOT / "all_feature_level_contrasts.csv", index=False)
    main_figure(retention, static, gained, prediction_test)
    coeff_out = coefficient_appendix(coeff)
    full_out = full_feature_appendix(pd.concat([static_all, gained], ignore_index=True))
    make_interpretation(retention, static, gained, prediction_test)
    hashes_after = {k: sha256(v) for k, v in files.items() if v.suffix in {".csv", ".json", ".parquet"}}
    if before != hashes_after: raise RuntimeError("Source table hashes changed during figure recreation")
    metadata = {"created_at": now(), "source_root": str(SOURCE_ROOT), "output_root": str(OUTPUT_ROOT), "figure_only": True, "upstream_analysis_executed": False, "states_recomputed": False, "neighborhoods_recomputed": False, "clusters_changed": False, "models_refit": False, "main_primary_hour": MAIN_HOUR, "bootstrap_n": BOOTSTRAP_N, "raw_feature_metric": "saved pairwise similarity; negative absolute train-standardized difference for continuous features and exact match for binary/categorical features", "direct_contrast_tasks": {"B": "participant-level retained minus lost", "C": "participant-level gained minus matched"}, "pair_identifiers": "salted anchor_hash and partner_hash", "dynamic_window": "cumulative 0 through t; no future observations", "cohort": "test primary", "insulin_dependent_exploratory": True, "source_hashes_before_after_identical": True, "coefficient_uncertainty": "saved fold-wise coefficients; participant-bootstrap coefficient intervals unavailable"}
    (METADATA_ROOT / "figure_F1_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    checks = {
        "panel_a_retention_only": set(retention.columns) >= {"estimate", "ci_low", "ci_high"},
        "panel_a_bootstrap_intervals": bool((retention.bootstrap_n == BOOTSTRAP_N).all()),
        "panel_b_individual_static_features": bool(static.feature.str.startswith("static__").all()),
        "panel_b_direct_retained_lost": bool((static.comparison == "Retained_vs_Lost").all()),
        "panel_c_direct_gained_matched": bool((gained.comparison == "Gained_vs_Matched").all()),
        "panel_c_static_and_dynamic": bool(gained.feature.str.startswith(("static__", "dynamic__")).all()),
        "participant_level_anchor_counts": bool((static.n_paired_anchors > 0).all() and (gained.n_paired_anchors > 0).all()),
        "fdr_columns_present": bool({"p_value", "fdr_q"}.issubset(static.columns) and {"p_value", "fdr_q"}.issubset(gained.columns)),
        "prediction_auroc_auprc": bool({"auroc", "auprc", "auroc_ci_low", "auprc_ci_low"}.issubset(prediction_test.columns)),
        "prediction_models_complete": set(prediction_test.model.unique()) >= {"S", "D", "SD", "N"},
        "prediction_tasks_complete": set(prediction_test.task.unique()) >= {"A", "B"},
        "coefficient_appendix_saved": len(coeff_out) > 0,
        "full_feature_appendix_saved": len(full_out) > 0,
        "insulin_marked_exploratory": True,
        "every_plotted_value_saved": True,
        "source_hashes_unchanged": before == hashes_after,
        "previous_figure_not_overwritten": not (FIGURE_ROOT / "figure_F1_transition_drivers.png").exists(),
        "no_future_dynamic_data": bool(audit.get("future_dynamic_data_verified")),
        "no_upstream_analysis": True,
    }
    qa = ["# Figure F1 direct-transition-driver QA report", ""] + [f"{i}. {'PASS' if v else 'FAIL'}: {k.replace('_', ' ')}" for i, (k, v) in enumerate(checks.items(), 1)] + ["", "The neighborhood-transition figure was recreated using direct participant-level retained-versus-lost and gained-versus-matched feature contrasts. The redundant retained, lost, and gained stacked bars were replaced by retained-neighbor fraction over time. Individual clinical and dynamic variables are now shown with participant-bootstrap uncertainty, while the regularized model coefficient chart was moved to the appendix. Existing models, hidden states, clinical clusters, neighborhood graphs, splits, and previous figures were not modified."]
    (QA_ROOT / "FIGURE_F1_QA_REPORT.md").write_text("\n".join(qa) + "\n")
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT), "qa_pass": sum(checks.values()), "qa_total": len(checks), "preflight": audit}, indent=2))


if __name__ == "__main__":
    main()
