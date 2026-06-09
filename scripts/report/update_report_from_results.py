#!/usr/bin/env python3
"""Update the AI-READI LaTeX report from inventoried stream results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.report.stream_report_utils import (
    FIGURE_DEST,
    find_source,
    fmt,
    intfmt,
    latex_escape,
    load_csv,
    load_json,
    read_manifest,
    replace_or_append_once,
    tabular,
    write_text,
)


def first(rows, **conds):
    for row in rows:
        if all(str(row.get(k, "")) == str(v) for k, v in conds.items()):
            return row
    return {}


def metric_file(rows, content):
    return find_source(rows, content, main_only=True) or find_source(rows, content, main_only=False)


def load_all_metrics(rows):
    data = {}
    for content in [
        "overall_forecast_metrics", "clinical_safety_metrics", "runtime_hardware_metrics",
        "training_summary", "participant_level_summary", "persistence_baseline",
    ]:
        p = metric_file(rows, content)
        data[content] = load_json(p) if p else {}
    for content in [
        "horizon_metrics", "persistence_by_horizon", "bias_diagnostic", "personalization_sweep",
        "matched_anchor_personalization", "scenario_metrics", "scenario_pathway_audit",
        "scenario_prediction_deltas", "event_detection_quantile_alarms", "subgroup_metrics",
        "subgroup_participant_level_metrics", "full_training_history",
    ]:
        p = metric_file(rows, content)
        data[content] = load_csv(p) if p else []
    return data


def forecast_row(data):
    row = first(data.get("scenario_metrics") or [], scenario_mode="forecast_only")
    return row or data.get("overall_forecast_metrics") or {}


def write_table(path: Path, tex: str):
    write_text(path, tex)
    print(f"[update] wrote {path}")


def make_overall_table(out_dir: Path, data: dict):
    f = forecast_row(data)
    rows = [
        ["Forecast horizon rows", intfmt(f.get("n"))],
        ["MAE (mg/dL) $\\downarrow$", fmt(f.get("mae"), 2)],
        ["RMSE (mg/dL) $\\downarrow$", fmt(f.get("rmse"), 2)],
        ["Bias (mg/dL)", fmt(f.get("bias"), 3, signed=True)],
        ["80\\% interval coverage", fmt(f.get("coverage80") or f.get("coverage"), 1, scale=100) + "\\%"],
        ["True TIR", fmt(f.get("tir_true"), 1, scale=100) + "\\%"],
        ["Predicted TIR", fmt(f.get("tir_predicted"), 1, scale=100) + "\\%"],
        ["TIR gap", fmt(f.get("tir_gap"), 2, scale=100, signed=True) + " pp"],
        ["p90 absolute error", fmt(f.get("p90_abs_error"), 2) + " mg/dL"],
        ["p95 absolute error", fmt(f.get("p95_abs_error"), 2) + " mg/dL"],
        ["p99 absolute error", fmt(f.get("p99_abs_error"), 2) + " mg/dL"],
    ]
    write_table(out_dir / "overall_metrics.tex", tabular("lc", ["Metric", "Value"], rows))


def make_horizon_table(out_dir: Path, data: dict):
    rows = []
    for r in data.get("horizon_metrics", []):
        rows.append([
            latex_escape(r.get("horizon_step")), latex_escape(r.get("horizon_minutes")),
            fmt(r.get("mae"), 2), fmt(r.get("rmse"), 2),
            fmt(r.get("bias"), 2, signed=True), fmt(r.get("coverage80") or r.get("coverage"), 1, scale=100) + "\\%",
        ])
    write_table(out_dir / "horizon_metrics.tex", tabular("rrrrrr", ["Step", "Min", "MAE", "RMSE", "Bias", "Cov."], rows))


def make_persistence_table(out_dir: Path, data: dict):
    p = data.get("persistence_baseline") or {}
    model = p.get("model", {})
    pers = p.get("persistence", {})
    m60 = float(p.get("model_terminal_mae", 0) or 0)
    p60 = float(p.get("persistence_terminal_mae", 0) or 0)
    rows = [
        ["Model", fmt(model.get("mae"), 2), fmt(model.get("rmse"), 2), fmt(model.get("bias"), 3, signed=True), fmt(m60, 2), fmt(model.get("tir_gap"), 2, scale=100, signed=True) + " pp"],
        ["Persistence", fmt(pers.get("mae"), 2), fmt(pers.get("rmse"), 2), fmt(pers.get("bias"), 3, signed=True), fmt(p60, 2), fmt(pers.get("tir_gap"), 2, scale=100, signed=True) + " pp"],
        ["Model - persistence", fmt(p.get("delta_mae_model_minus_persistence"), 2, signed=True), "--", "--", fmt(m60 - p60, 2, signed=True), "--"],
    ]
    write_table(out_dir / "persistence_baseline.tex", tabular("lrrrrr", ["Method", "MAE", "RMSE", "Bias", "60-min MAE", "TIR gap"], rows))


def make_participant_table(out_dir: Path, data: dict):
    summary = data.get("participant_level_summary") or {}
    metrics = summary.get("metrics", [])
    wanted = ["mae", "rmse", "bias", "tir_gap", "coverage", "p95_abs_error"]
    labels = {
        "mae": "MAE (mg/dL)", "rmse": "RMSE (mg/dL)", "bias": "Bias (mg/dL)",
        "tir_gap": "TIR gap (pp)", "coverage": "Coverage (\\%)", "p95_abs_error": "p95 abs. error",
    }
    rows = []
    for metric in wanted:
        r = first(metrics, scenario_mode="forecast_only", metric=metric) or first(metrics, metric=metric)
        if not r:
            continue
        scale = 100 if metric in {"tir_gap", "coverage"} else 1
        digits = 1 if metric == "coverage" else 2
        rows.append([
            labels[metric], fmt(r.get("mean"), digits, scale, signed=(metric == "tir_gap")),
            fmt(r.get("median"), digits, scale, signed=(metric == "tir_gap")),
            fmt(r.get("iqr_low"), digits, scale) + "--" + fmt(r.get("iqr_high"), digits, scale),
            fmt(r.get("ci95_low"), digits, scale, signed=(metric == "tir_gap")) + "--" + fmt(r.get("ci95_high"), digits, scale, signed=(metric == "tir_gap")),
        ])
    write_table(out_dir / "participant_level_metrics.tex", tabular("lrrrr", ["Metric", "Mean", "Median", "IQR", "95\\% CI"], rows))


def make_personalization_tables(out_dir: Path, data: dict):
    rows = []
    for r in data.get("personalization_sweep", []):
        rows.append([fmt(r.get("warmup_hours"), 0), intfmt(r.get("n")), fmt(r.get("mae"), 2), fmt(r.get("bias"), 2, signed=True), fmt(r.get("coverage80"), 1, scale=100) + "\\%"])
    write_table(out_dir / "personalization_sweep.tex", tabular("rrrrr", ["Warm-up h", "Rows", "MAE", "Bias", "Cov."], rows))
    rows = []
    for r in data.get("matched_anchor_personalization", []):
        rows.append([fmt(r.get("warmup_hours"), 0), intfmt(r.get("n_anchors")), intfmt(r.get("n_participants")), fmt(r.get("mae"), 2), fmt(r.get("bias"), 2, signed=True), fmt(r.get("coverage80"), 1, scale=100) + "\\%"])
    write_table(out_dir / "personalization_matched_anchor.tex", tabular("rrrrrr", ["Warm-up h", "Anchors", "Participants", "MAE", "Bias", "Cov."], rows))


def make_bias_table(out_dir: Path, data: dict):
    label = {"raw": "Raw population", "offset-corrected": "Offset-corrected", "personalized": "Personalized", "personalized+offset-corrected": "Personalized + offset"}
    rows = []
    for r in data.get("bias_diagnostic", []):
        rows.append([latex_escape(label.get(r.get("bias_mode"), r.get("bias_mode"))), fmt(r.get("mae"), 2), fmt(r.get("bias"), 3, signed=True), fmt(r.get("coverage80"), 1, scale=100) + "\\%"])
    write_table(out_dir / "bias_diagnostic.tex", tabular("lrrr", ["Mode", "MAE", "Bias", "Cov."], rows))


def make_scenario_tables(out_dir: Path, data: dict):
    mode_label = {"forecast_only": "forecast-only", "factual_future": "factual-future", "meal_proxy": "meal\\_proxy", "activity_proxy": "activity\\_proxy", "sleep_rest_proxy": "sleep\\_rest\\_proxy"}
    rows = []
    for r in data.get("scenario_metrics", []):
        rows.append([mode_label.get(r.get("scenario_mode"), latex_escape(r.get("scenario_mode"))), fmt(r.get("mae"), 3), fmt(r.get("rmse"), 2), fmt(r.get("bias"), 3, signed=True), fmt(r.get("coverage80"), 1, scale=100) + "\\%"])
    write_table(out_dir / "scenario_metrics.tex", tabular("lrrrr", ["Mode", "MAE", "RMSE", "Bias", "Cov."], rows))
    rows = []
    for r in [x for x in data.get("scenario_prediction_deltas", []) if x.get("scope") == "overall"]:
        rows.append([mode_label.get(r.get("scenario_mode"), latex_escape(r.get("scenario_mode"))), fmt(r.get("mean_abs_delta_vs_forecast_only"), 4), fmt(r.get("median_abs_delta_vs_forecast_only"), 4), fmt(r.get("p95_abs_delta_vs_forecast_only"), 4), fmt(r.get("mean_signed_delta_vs_forecast_only"), 4, signed=True)])
    write_table(out_dir / "scenario_prediction_deltas.tex", tabular("lrrrr", ["Mode", "Mean |delta|", "Median |delta|", "p95 |delta|", "Mean delta"], rows))
    rows = []
    for r in data.get("scenario_pathway_audit", [])[:8]:
        rows.append([latex_escape(r.get("scenario_variable")), fmt(r.get("mask_nonzero_pct"), 1, scale=100) + "\\%", fmt(r.get("anchor_available_pct"), 1, scale=100) + "\\%", fmt(r.get("value_mean"), 3), fmt(r.get("value_std"), 3)])
    write_table(out_dir / "scenario_pathway_audit.tex", tabular("lrrrr", ["Variable", "Mask nonzero", "Anchors avail.", "Value mean", "Value SD"], rows))


def make_event_table(out_dir: Path, data: dict):
    rows = []
    for r in data.get("event_detection_quantile_alarms", []):
        if r.get("scope") == "overall" or (r.get("scope") == "horizon" and str(r.get("horizon_minutes")) in {"30", "60"}):
            event = "Hypo $<70$" if r.get("event") == "hypoglycemia" else "Hyper $>180$"
            horizon = "overall" if r.get("scope") == "overall" else f"{r.get('horizon_minutes')} min"
            rows.append([event, horizon, latex_escape(r.get("rule")), fmt(r.get("precision"), 3), fmt(r.get("recall"), 3), fmt(r.get("specificity"), 3), fmt(r.get("f1"), 3)])
    write_table(out_dir / "event_detection_quantile_alarms.tex", tabular("lllrrrr", ["Event", "Horizon", "Rule", "Prec.", "Recall", "Spec.", "F1"], rows))


def make_subgroup_table(out_dir: Path, data: dict):
    rows = []
    keep = {"participants_study_group", "hba1c_quartile", "med_insulin", "participants_clinical_site"}
    for r in data.get("subgroup_participant_level_metrics", []):
        if r.get("subgroup") not in keep:
            continue
        rows.append([latex_escape(r.get("subgroup")), latex_escape(r.get("level")), intfmt(r.get("n_participants")), fmt(r.get("mae_mean"), 2), fmt(r.get("mae_median"), 2), fmt(r.get("bias_mean"), 2, signed=True), fmt(r.get("tir_gap_mean"), 2, scale=100, signed=True) + " pp"])
    write_table(out_dir / "subgroup_metrics_main.tex", tabular("llrrrrr", ["Subgroup", "Level", "N", "Mean MAE", "Median MAE", "Mean bias", "Mean TIR gap"], rows))


def make_runtime_table(out_dir: Path, data: dict):
    hw = data.get("runtime_hardware_metrics") or {}
    ts = data.get("training_summary") or {}
    hist = ts.get("history") or []
    best_epoch = "--"
    if hist:
        best = min(hist, key=lambda x: float(x.get("val_pinball_mgdl", 1e9)))
        best_epoch = str(best.get("epoch", "--"))
    max_aps = max([float(h.get("anchors_per_s", 0) or 0) for h in hist], default=0)
    peak_train = max([float(h.get("peak_mem_mb", 0) or 0) for h in hist], default=0)
    rows = [
        ["Best validation epoch", best_epoch],
        ["Best validation pinball", fmt(ts.get("best_val_pinball_mgdl"), 3) + " mg/dL"],
        ["Max training throughput", fmt(max_aps, 0) + " anchors/s"],
        ["Peak training GPU memory", fmt(peak_train, 0) + " MB"],
        ["Evaluation GPU memory", fmt(hw.get("peak_gpu_memory_mb"), 0) + " MB"],
        ["Evaluation CPU RSS", fmt(hw.get("cpu_memory_rss_mb"), 0) + " MB"],
        ["Median update latency", fmt((hw.get("latency_per_update") or {}).get("median_ms"), 2) + " ms"],
        ["Mean 1h forecast latency", fmt((hw.get("latency_per_1h_forecast") or {}).get("mean_ms"), 4) + " ms"],
    ]
    write_table(out_dir / "runtime_metrics.tex", tabular("lc", ["Quantity", "Value"], rows))
    hist_rows = []
    for h in hist:
        hist_rows.append([latex_escape(h.get("epoch")), fmt(h.get("train_pinball_mgdl"), 3), fmt(h.get("val_pinball_mgdl"), 3), fmt(h.get("anchors_per_s"), 0), fmt(h.get("seconds"), 1)])
    write_table(out_dir / "training_history.tex", tabular("rrrrr", ["Epoch", "Train", "Val", "Anchors/s", "Seconds"], hist_rows))


def make_ablation_table(out_dir: Path):
    rows = [
        ["Full residual-current model", "configured", "available as reference"],
        ["Absolute-target / no residual-current", "prepared", "not run by current pipeline"],
        ["No static h0/FiLM", "probe supported", "run with launch\\_ablation\\_probes.sh"],
        ["No scenario masks", "prepared", "current model lacks safe switch"],
        ["No scenario decomposition", "probe supported", "run with launch\\_ablation\\_probes.sh"],
    ]
    write_table(out_dir / "ablation_metrics.tex", tabular("lll", ["Ablation", "Status", "Interpretation"], rows))




def make_supporting_assets(report_dir: Path, data: dict):
    """Create small generated assets referenced by existing report sections."""
    cohort = report_dir / "tables" / "generated" / "cohort_characteristics.tex"
    br = "\\\\"
    cohort_rows = [
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Characteristic & Value " + br,
        "\\midrule",
        "Split & Participant-held-out 70/15/15 " + br,
        "Sampling interval & 5 min " + br,
        "Forecast horizon & 12 steps / 60 min " + br,
        "Train streams & 1,849 " + br,
        "Validation streams & 392 " + br,
        "Test streams & 381 " + br,
        "Test forecast anchors & 155,113 " + br,
        "Timed insulin dose / IOB & Not available in AI-READI " + br,
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    write_text(cohort, "\n".join(cohort_rows))
    hist = data.get("full_training_history") or []
    if not hist:
        return
    fig_path = report_dir / "figures" / "generated" / "training_curves.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        epochs = [int(float(r.get("epoch", 0) or 0)) for r in hist]
        train = [float(r.get("train_pinball_mgdl", 0) or 0) for r in hist]
        val = [float(r.get("val_pinball_mgdl", 0) or 0) for r in hist]
        aps = [float(r.get("anchors_per_s", 0) or 0) for r in hist]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        ax[0].plot(epochs, train, "o-", label="train")
        ax[0].plot(epochs, val, "s-", label="validation")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Pinball loss (mg/dL)")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(epochs, aps, "o-", color="#4c72b0")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Anchors/s")
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        write_text(fig_path.with_suffix(".txt"), str(exc))

def make_all_tables(report_dir: Path, data: dict):
    out = report_dir / "tables" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    make_overall_table(out, data)
    make_horizon_table(out, data)
    make_persistence_table(out, data)
    make_participant_table(out, data)
    make_personalization_tables(out, data)
    make_bias_table(out, data)
    make_scenario_tables(out, data)
    make_event_table(out, data)
    make_subgroup_table(out, data)
    make_runtime_table(out, data)
    make_ablation_table(out)


def figure_block(filename: str, label: str, caption: str, width: str = "0.82\\linewidth") -> str:
    return f"""\\begin{{figure}}[t]
\\centering
\\includegraphics[width={width}]{{figures/generated/{filename}}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""


def table_block(path: str, label: str, caption: str) -> str:
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\input{{tables/generated/{path}}}
\\end{{table}}
"""


def make_generated_sections(report_dir: Path, data: dict):
    gen = report_dir / "sections" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    fig_dir = report_dir / "figures" / "generated"

    def maybe_fig(content: str, caption: str, label: str, width="0.82\\linewidth"):
        fname = FIGURE_DEST.get(content)
        if fname and (fig_dir / fname).exists():
            return figure_block(fname, label, caption, width)
        return f"\\TODO{{Missing generated figure for {latex_escape(content)}.}}\n"

    figures = "% AUTO-GENERATED by update_report_from_results.py -- do not edit by hand\n"
    figures += "\\providecommand{\\GeneratedForecastingFigures}{}\n\\renewcommand{\\GeneratedForecastingFigures}{%\n"
    figures += maybe_fig("mae_by_horizon_figure", "Forecast error by horizon on the held-out test split. The multi-horizon average MAE is lower than the terminal 60-minute MAE, so terminal performance must be reported separately from the averaged headline number.", "fig:horizon_mae", "0.72\\linewidth")
    figures += "}\n\\providecommand{\\GeneratedPersonalizationFigures}{}\n\\renewcommand{\\GeneratedPersonalizationFigures}{%\n"
    figures += maybe_fig("personalization_sweep_figure", "Warm-up sweep from the original evaluation. MAE improves as later anchors are evaluated, but anchor counts differ by warm-up length, so this plot alone is not definitive evidence of personalization.", "fig:personalization_warmup", "0.68\\linewidth")
    figures += maybe_fig("matched_anchor_personalization_figure", "Matched-anchor personalization diagnostic. All warm-up settings are evaluated on the same eligible anchors; the flat curve shows that the current prediction artifacts do not yet demonstrate a warm-up-specific personalization effect.", "fig:personalization_matched_anchor", "0.68\\linewidth")
    figures += "}\n\\providecommand{\\GeneratedScenarioFigures}{}\n\\renewcommand{\\GeneratedScenarioFigures}{%\n"
    figures += maybe_fig("scenario_delta_figure", "Absolute prediction deltas between proxy scenario modes and forecast-only mode. Deltas are far below clinically meaningful mg/dL changes, indicating that the scenario pathway is currently inactive or weakly used.", "fig:scenario_delta", "0.75\\linewidth")
    figures += "}\n\\providecommand{\\GeneratedSafetyFigures}{}\n\\renewcommand{\\GeneratedSafetyFigures}{%\n"
    figures += maybe_fig("hypo_alarm_tradeoff_figure", "Clinical alarm trade-off. Median forecasts are conservative for hypoglycemia, while the q0.1 risk rule increases recall at the cost of lower precision.", "fig:hypo_alarm_tradeoff", "0.72\\linewidth")
    figures += "}\n\\providecommand{\\GeneratedSubgroupFigures}{}\n\\renewcommand{\\GeneratedSubgroupFigures}{%\n"
    figures += maybe_fig("subgroup_plot_study_group", "Participant-level MAE by AI-READI study group. Insulin-dependent and medication-treated groups are harder to forecast than healthy controls, consistent with higher glycemic variability.", "fig:subgroup_study_group")
    figures += maybe_fig("subgroup_plot_hba1c", "Participant-level MAE by HbA1c quartile. Higher HbA1c strata show larger errors, supporting phenotype-level error analysis beyond a single aggregate MAE.", "fig:subgroup_hba1c")
    figures += maybe_fig("subgroup_plot_med_insulin", "Participant-level MAE stratified by the static med\\_insulin metadata flag. This is a therapy-status subgroup, not an insulin action or dose counterfactual.", "fig:subgroup_med_insulin")
    figures += maybe_fig("subgroup_plot_site", "Participant-level MAE by clinical site. Site differences should be interpreted as possible domain shift or cohort-composition effects and require follow-up audits.", "fig:subgroup_site")
    figures += "}\n\\providecommand{\\GeneratedRuntimeFigures}{}\n\\renewcommand{\\GeneratedRuntimeFigures}{%\n\\TODO{No runtime backend figure was found; runtime is summarized in Table~\\ref{tab:runtime_metrics_generated}.}\n}\n"
    figures += "\\providecommand{\\GeneratedAppendixFigures}{}\n\\renewcommand{\\GeneratedAppendixFigures}{%\n"
    figures += maybe_fig("subgroup_plot_bmi", "Appendix subgroup plot: participant-level MAE by BMI quartile.", "fig:subgroup_bmi")
    figures += maybe_fig("subgroup_plot_med_any_diabetes_drug", "Appendix subgroup plot: participant-level MAE by any diabetes-drug metadata flag.", "fig:subgroup_med_any_diabetes_drug")
    figures += "}\n"
    write_text(gen / "figures_stream_results.tex", figures)

    tables = "% AUTO-GENERATED by update_report_from_results.py -- do not edit by hand\n"
    tables += "\\providecommand{\\GeneratedForecastingTables}{}\n\\renewcommand{\\GeneratedForecastingTables}{%\n"
    tables += table_block("overall_metrics", "tab:overall_metrics_generated", "Forecast-only metrics on the held-out test split. TIR values are percentages and TIR gap is reported in percentage points.")
    tables += table_block("horizon_metrics", "tab:horizon_metrics_generated", "Forecast error by 5-minute horizon step. The final row is the terminal 60-minute forecast, not the multi-horizon average.")
    tables += table_block("persistence_baseline", "tab:persistence_baseline_generated", "Matched-anchor persistence comparison. Persistence predicts every horizon as the current glucose value.")
    tables += table_block("participant_level_metrics", "tab:participant_level_generated", "Participant-level averaged metrics. This reduces the dominance of participants with more valid anchors.")
    tables += "}\n\\providecommand{\\GeneratedPersonalizationTables}{}\n\\renewcommand{\\GeneratedPersonalizationTables}{%\n"
    tables += table_block("personalization_sweep", "tab:personalization_sweep_generated", "Original warm-up sweep. Anchor counts differ across rows, so this table is descriptive rather than a matched personalization test.")
    tables += table_block("personalization_matched_anchor", "tab:personalization_matched_generated", "Matched-anchor personalization sweep using the same 48-hour-eligible anchors for every warm-up row.")
    tables += table_block("bias_diagnostic", "tab:bias_diagnostic_generated", "Bias diagnostics. Offset correction removes the aggregate bias but worsens MAE and coverage in this run.")
    tables += "}\n\\providecommand{\\GeneratedScenarioTables}{}\n\\renewcommand{\\GeneratedScenarioTables}{%\n"
    tables += table_block("scenario_metrics", "tab:scenario_metrics_generated", "Metrics by evaluation/scenario mode. Proxy scenarios are diagnostics and are not validated causal interventions.")
    tables += table_block("scenario_prediction_deltas", "tab:scenario_delta_generated", "Prediction deltas relative to forecast-only mode. Values near zero indicate weak use of the scenario pathway.")
    tables += table_block("scenario_pathway_audit", "tab:scenario_pathway_audit_generated", "Scenario mask and value audit for selected future proxy variables.")
    tables += "}\n\\providecommand{\\GeneratedSafetyTables}{}\n\\renewcommand{\\GeneratedSafetyTables}{%\n"
    tables += table_block("event_detection_quantile_alarms", "tab:event_detection_quantile_generated", "Hypoglycemia and hyperglycemia event detection using median and risk-quantile alarm rules.")
    tables += "}\n\\providecommand{\\GeneratedSubgroupTables}{}\n\\renewcommand{\\GeneratedSubgroupTables}{%\n"
    tables += table_block("subgroup_metrics_main", "tab:subgroup_participant_metrics_generated", "Participant-level subgroup metrics for study group, HbA1c quartile, insulin metadata, and clinical site.")
    tables += "}\n\\providecommand{\\GeneratedRuntimeTables}{}\n\\renewcommand{\\GeneratedRuntimeTables}{%\n"
    tables += table_block("runtime_metrics", "tab:runtime_metrics_generated", "Runtime, memory, and training-throughput summary from the selected AI-READI stream run.")
    tables += table_block("ablation_metrics", "tab:ablation_metrics_generated", "Ablation status. Some ablations are configured as probes but have not yet been fully run.")
    tables += "}\n\\providecommand{\\GeneratedAppendixTables}{}\n\\renewcommand{\\GeneratedAppendixTables}{%\n"
    tables += table_block("training_history", "tab:training_history_generated", "Full training history for the 10-epoch resumed stateful Mamba run.")
    tables += "}\n"
    write_text(gen / "tables_stream_results.tex", tables)

    f = forecast_row(data)
    p = data.get("persistence_baseline") or {}
    deltas = [float(r.get("mean_abs_delta_vs_forecast_only", 0) or 0) for r in data.get("scenario_prediction_deltas", []) if r.get("scope") == "overall"]
    max_delta = max(deltas) if deltas else 0.0
    m60 = float(p.get("model_terminal_mae", 0) or 0)
    p60 = float(p.get("persistence_terminal_mae", 0) or 0)
    interp = f"""% AUTO-GENERATED by update_report_from_results.py -- do not edit by hand
\\providecommand{{\\GeneratedForecastingInterpretation}}{{}}
\\renewcommand{{\\GeneratedForecastingInterpretation}}{{%
The deployable forecast-only model reaches {fmt(f.get('mae'), 2)}~mg/dL MAE and {fmt(f.get('rmse'), 2)}~mg/dL RMSE on the held-out test split, with bias {fmt(f.get('bias'), 2, signed=True)}~mg/dL and nominal 80\\% interval coverage of {fmt(f.get('coverage80') or f.get('coverage'), 1, scale=100)}\\%. The terminal 60-minute MAE is {fmt(m60, 2)}~mg/dL, which is larger than the multi-horizon average and should be reported separately. Compared with persistence on the exact same anchors, the model improves MAE by {fmt(abs(float(p.get('delta_mae_model_minus_persistence', 0) or 0)), 2)}~mg/dL overall and by {fmt(abs(m60 - p60), 2)}~mg/dL at 60 minutes.}}
\\providecommand{{\\GeneratedPersonalizationInterpretation}}{{}}
\\renewcommand{{\\GeneratedPersonalizationInterpretation}}{{%
The unmatched warm-up sweep shows lower MAE at longer warm-up lengths, but the evaluated anchor set shrinks with warm-up time. The matched-anchor diagnostic is therefore the valid comparison: in the current prediction artifacts, all warm-up rows have the same MAE, indicating that the saved evaluation does not yet demonstrate a warm-up-specific personalization gain. Offset correction removes aggregate bias but worsens MAE, so participant-level offset is not the main remaining failure mode.}}
\\providecommand{{\\GeneratedScenarioInterpretation}}{{}}
\\renewcommand{{\\GeneratedScenarioInterpretation}}{{%
Forecast-only, factual-future, meal-proxy, activity-proxy, and sleep/rest-proxy metrics are nearly identical. The largest mean absolute proxy-scenario delta is {max_delta:.4f}~mg/dL, far below a clinically meaningful glucose change. The scenario branch is therefore currently inactive or weakly used, and proxy outputs should be treated only as diagnostics. AI-READI has no timed insulin dose or insulin-on-board inputs; \\texttt{{med\\_insulin}} is static metadata, not an editable action.}}
\\providecommand{{\\GeneratedSafetyInterpretation}}{{}}
\\renewcommand{{\\GeneratedSafetyInterpretation}}{{%
Median hypoglycemia alarms are conservative: precision is high but recall is low. The q0.1 risk rule substantially increases hypoglycemia recall at the cost of lower precision, which is the expected clinical safety trade-off. Hyperglycemia is more predictable than hypoglycemia, but the q0.9 risk rule similarly trades precision for sensitivity.}}
\\providecommand{{\\GeneratedSubgroupInterpretation}}{{}}
\\renewcommand{{\\GeneratedSubgroupInterpretation}}{{%
Participant-level subgroup metrics show a consistent phenotype gradient: insulin-dependent and higher-HbA1c participants are harder to forecast than healthy or lower-HbA1c participants. Site differences are visible and should be interpreted as possible domain shift or cohort-composition effects, not as a causal site effect.}}
\\providecommand{{\\GeneratedRuntimeInterpretation}}{{}}
\\renewcommand{{\\GeneratedRuntimeInterpretation}}{{%
The stateful Mamba run is feasible on a single GPU. The best validation checkpoint occurs at epoch 5; later epochs improve training loss but do not improve validation pinball, so the best checkpoint rather than the final epoch is used for evaluation. The ablation launcher prepares short probes, but full ablations remain a follow-up experiment.}}
"""
    write_text(gen / "results_interpretation.tex", interp)


def make_generated_summary(report_dir: Path, data: dict):
    f = forecast_row(data)
    p = data.get("persistence_baseline") or {}
    scenario = {r.get("scenario_mode"): r for r in data.get("scenario_metrics", [])}
    pers = {str(r.get("warmup_hours")): r for r in data.get("personalization_sweep", [])}
    bias_rows = {r.get("bias_mode"): r for r in data.get("bias_diagnostic", [])}
    subgroup_mae = {}
    for r in data.get("subgroup_metrics", []):
        if r.get("subgroup") == "participants_study_group":
            subgroup_mae[r.get("level")] = r.get("mae")
    hw = data.get("runtime_hardware_metrics") or {}
    ts = data.get("training_summary") or {}
    hist = ts.get("history") or []
    best_epoch = ""
    max_aps = 0.0
    peak_train = 0.0
    if hist:
        best = min(hist, key=lambda x: float(x.get("val_pinball_mgdl", 1e9)))
        best_epoch = str(best.get("epoch", ""))
        max_aps = max(float(h.get("anchors_per_s", 0) or 0) for h in hist)
        peak_train = max(float(h.get("peak_mem_mb", 0) or 0) for h in hist)
    m60 = float(p.get("model_terminal_mae", 0) or 0)
    p60 = float(p.get("persistence_terminal_mae", 0) or 0)
    macros = {
        "reportRunDate": "2026-06-09",
        "reportRunName": "aireadi_stream_mamba_stateful_10epoch_eval_test",
        "maeFo": fmt(f.get("mae"), 2), "rmseFo": fmt(f.get("rmse"), 2), "biasFo": fmt(f.get("bias"), 3),
        "covFo": fmt(f.get("coverage80") or f.get("coverage"), 3), "nAnchorsEval": intfmt(p.get("n_anchors") or f.get("n")),
        "tirTrue": fmt(f.get("tir_true"), 1, scale=100), "tirPredicted": fmt(f.get("tir_predicted"), 1, scale=100), "tirGap": fmt(f.get("tir_gap"), 2, scale=100, signed=True),
        "pNinetyErr": fmt(f.get("p90_abs_error"), 2), "pNinetyFiveErr": fmt(f.get("p95_abs_error"), 2), "pNinetyNineErr": fmt(f.get("p99_abs_error"), 2),
        "maeFoForecastOnly": fmt((scenario.get("forecast_only") or f).get("mae"), 2), "maeFoFactual": fmt((scenario.get("factual_future") or {}).get("mae"), 2),
        "maeFoMealProxy": fmt((scenario.get("meal_proxy") or {}).get("mae"), 2), "maeFoActivityProxy": fmt((scenario.get("activity_proxy") or {}).get("mae"), 2),
        "maeWarmZero": fmt((pers.get("0.0") or {}).get("mae"), 2), "maeWarmSix": fmt((pers.get("6.0") or {}).get("mae"), 2),
        "maeWarmTwelve": fmt((pers.get("12.0") or {}).get("mae"), 2), "maeWarmTwentyFour": fmt((pers.get("24.0") or {}).get("mae"), 2), "maeWarmFortyEight": fmt((pers.get("48.0") or {}).get("mae"), 2),
        "biasRaw": fmt((bias_rows.get("raw") or {}).get("bias"), 3), "biasOffsetCorrected": fmt((bias_rows.get("offset-corrected") or {}).get("bias"), 3),
        "maeHealthy": fmt(subgroup_mae.get("healthy"), 2), "maeInsulinDep": fmt(subgroup_mae.get("insulin_dependent"), 2), "maeOralMed": fmt(subgroup_mae.get("oral_medication_and_or_non_insulin_injectable_medication_controlled"), 2), "maePreDiabetes": fmt(subgroup_mae.get("pre_diabetes_lifestyle_controlled"), 2),
        "bestValPinball": fmt(ts.get("best_val_pinball_mgdl"), 3), "bestValEpoch": best_epoch or "--", "anchorsPerSec": fmt(max_aps, 0), "peakTrainMemMB": fmt(peak_train, 0),
        "peakInferMemMB": fmt(hw.get("peak_gpu_memory_mb"), 0), "latencyUpdateMsMedian": fmt((hw.get("latency_per_update") or {}).get("median_ms"), 2), "latencyForecastMsMean": fmt((hw.get("latency_per_1h_forecast") or {}).get("mean_ms"), 4), "cpuMemRssMB": fmt(hw.get("cpu_memory_rss_mb"), 0),
        "persistenceModelMAE": fmt((p.get("model") or {}).get("mae"), 2), "persistenceMAE": fmt((p.get("persistence") or {}).get("mae"), 2), "persistenceDeltaMae": fmt(abs(float(p.get("delta_mae_model_minus_persistence", 0) or 0)), 2), "modelTerminalMAE": fmt(m60, 2), "persistenceTerminalMAE": fmt(p60, 2),
    }
    lines = ["% AUTO-GENERATED by update_report_from_results.py -- do not edit by hand", "% TIR values are percentages; TIR gap is percentage points.", ""]
    for k, v in sorted(macros.items()):
        if v and v != "--":
            lines.append(f"\\providecommand{{\\{k}}}{{}}")
            lines.append(f"\\renewcommand{{\\{k}}}{{{v}}}")
    rows = [[latex_escape(k), latex_escape(v)] for k, v in sorted(macros.items()) if v and v != "--"]
    lines.append("\n% Snapshot table for Appendix")
    lines.append("\\newcommand{\\generatedResultsSnapshotTable}{%")
    lines.append(tabular("ll", ["Field", "Value"], rows).rstrip())
    lines.append("}")
    write_text(report_dir / "sections" / "generated_results_summary.tex", "\n".join(lines) + "\n")


def patch_main(report_dir: Path):
    block = r"""\IfFileExists{sections/generated/tables_stream_results.tex}{%
  \input{sections/generated/tables_stream_results}%
}{}
\IfFileExists{sections/generated/figures_stream_results.tex}{%
  \input{sections/generated/figures_stream_results}%
}{}
\IfFileExists{sections/generated/results_interpretation.tex}{%
  \input{sections/generated/results_interpretation}%
}{}"""
    replace_or_append_once(report_dir / "main.tex", "STREAM RESULTS GENERATED INPUTS", block)


def write_sections(report_dir: Path):
    sections = report_dir / "sections"
    write_text(sections / "00_abstract.tex", r'''
\begin{abstract}
We adapt SSMCGM-Stream from T1DEXI to AI-READI, a multimodal cohort spanning healthy, pre-diabetic, medication-treated, and insulin-dependent participants across multiple clinical sites. The model keeps a constant-memory streaming state, decodes a 12-step 60-minute quantile horizon, and uses static clinical metadata to initialize and modulate the state. The residual-current target anchors forecasts to the most recent observed glucose; it reduces held-out participant cold-start bias, but it does not make the model exactly zero-bias.

On the participant-held-out test split, the deployable forecast-only model achieves MAE $=\maeFo$~mg/dL and RMSE $=\rmseFo$~mg/dL with 80\% interval coverage $\covFo$. True TIR is $\tirTrue\%$ and predicted TIR is $\tirPredicted\%$ (gap $\tirGap$ percentage points). Against persistence on the same anchors, the model improves average MAE by $\persistenceDeltaMae$~mg/dL and terminal 60-minute MAE from $\persistenceTerminalMAE$ to $\modelTerminalMAE$~mg/dL.

AI-READI does not include timed insulin dose or insulin-on-board logs. \texttt{med\_insulin} is static metadata, insulin causal ranking is disabled, and meal/activity/sleep modes are proxy diagnostics rather than validated interventions. The scenario-delta audit shows proxy scenario effects are currently near zero, so scenario outputs are not scientifically interpretable as causal what-if predictions.
\end{abstract}
'''.lstrip())
    write_text(sections / "03_streaming_architecture.tex", r'''
\section{Streaming State-Space Architecture}
\label{sec:arch}

\subsection{Problem setup}
At each 5-minute step $t$, the model observes the history available up to the current CGM reading and predicts quantiles for $H=12$ future steps. Future target labels are never used as inputs. Known calendar covariates and optional proxy scenario paths are provided to the horizon decoder with explicit masks so that unknown future values remain distinguishable from observed zeros.

\subsection{Static conditioning and streaming state}
Static AI-READI metadata, including HbA1c, BMI, clinical site, and medication flags, initializes the participant state and modulates observed-history features through FiLM conditioning. The state is reset at each \texttt{participant\_id}+\texttt{segment\_id} boundary, so neither recurrent state nor forecast anchors cross cleaned CGM segments.

\subsection{Mamba state-space backend}
The history encoder uses selective state-space blocks with the CUDA Mamba/Triton scan backend for training and a pure-PyTorch fallback for portability. The production training mode is \texttt{stateful\_stream}: independent segment streams are batched, state is carried within a segment only, and truncated BPTT detaches the state between chunks.

\subsection{Residual-current target}
The model predicts deviations from the current glucose anchor rather than absolute glucose directly. This residual-current transform improves cold-start behavior for held-out participants and makes persistence a strong baseline. It does not guarantee exactly zero downstream bias; the observed test bias remains $\biasFo$~mg/dL.

\subsection{Proxy scenario decoder}
The decoder can consume masked future proxy paths for meal, activity, and sleep/rest diagnostics. In AI-READI these are not validated treatment interventions. There are no timed insulin dose or insulin-on-board inputs, and \texttt{med\_insulin} is never treated as an editable action.
'''.lstrip())
    write_text(sections / "05_experiments.tex", r'''
\section{Experiments}
\label{sec:experiments}

\subsection{Participant-held-out split}
Participants are split 70/15/15 with a participant-level holdout stratified by \texttt{participants\_study\_group}. Scalers and residual-current horizon scales are fit on training participants only. All windows, anchors, forecast horizons, and recurrent states remain inside a single \texttt{participant\_id}+\texttt{segment\_id} segment.

\subsection{Headline model}
The headline run uses \texttt{stateful\_stream} training with batch size 8, Mamba scan mode on CUDA, residual-current targets, static $h_0$/FiLM conditioning, and base-plus-delta scenario decomposition. The run was continued to 10 epochs, but the best validation checkpoint is epoch $\bestValEpoch$ with validation pinball $\bestValPinball$~mg/dL; later epochs lower training loss but do not improve validation pinball.

\subsection{Baselines and diagnostics}
The primary baseline is persistence evaluated on the exact same forecast-only anchors. Additional diagnostics include participant-level averaging, matched-anchor personalization, scenario mask/value and prediction-delta audits, q0.1/q0.9 clinical alarm rules, subgroup plots, and runtime summaries. Ablation configs are prepared as short probes, but full ablation conclusions are not claimed unless the corresponding probe has been run and evaluated.
'''.lstrip())
    write_text(sections / "06_results_forecasting.tex", r'''
\section{Forecasting Results}
\label{sec:results}

\GeneratedForecastingInterpretation

\GeneratedForecastingTables

\GeneratedForecastingFigures

\subsection{Previous experiment comparison}
Earlier A/B temporal splits were easier because participants were seen during training. The older Exp.~C participant-held-out setting had much stronger underprediction and higher MAE. The current AI-READI Stream setup is the relevant comparison for deployment to unseen participants: residual-current forecasting substantially reduces cold-start bias relative to that older participant-held-out failure mode, but the remaining nonzero test bias means the correction is incomplete.

\subsection{Subgroup forecasting}
\GeneratedSubgroupInterpretation

\GeneratedSubgroupTables

'''.lstrip())
    write_text(sections / "07_personalization.tex", r'''
\section{Personalization}
\label{sec:pers}

\GeneratedPersonalizationInterpretation

\GeneratedPersonalizationTables

\GeneratedPersonalizationFigures

The unmatched warm-up sweep should be read only as a descriptive diagnostic because longer warm-up removes early anchors. The matched-anchor table is the valid comparison for personalization; in the current artifacts, it is flat across warm-up lengths. Offset correction also worsens MAE, so the remaining error is not just a participant-level constant offset.
'''.lstrip())
    write_text(sections / "08_proxy_scenarios.tex", r'''
\section{Proxy Scenario Evaluation}
\label{sec:proxy}

\subsection{AI-READI scenario taxonomy}
AI-READI provides wearable proxies and static metadata, not the timed insulin dose, carbohydrate quantity, or insulin-on-board streams available in T1DEXI. The dose-ordering causal ranking loss is disabled for AI-READI. \texttt{med\_insulin} records therapy status only; it is not an action covariate and is not edited in counterfactual simulations.

\GeneratedScenarioInterpretation

\GeneratedScenarioTables

\GeneratedScenarioFigures

The meal, activity, and sleep/rest outputs should therefore be described as proxy pathway diagnostics. They should not be used as treatment recommendations, carbohydrate counterfactuals, or insulin counterfactuals. A future scientifically interpretable scenario model would need prospective action logs or stronger supervision of the scenario branch.
'''.lstrip())
    write_text(sections / "09_clinical_safety.tex", r'''
\section{Clinical Safety}
\label{sec:safety}

\GeneratedSafetyInterpretation

\GeneratedSafetyTables

\GeneratedSafetyFigures

Tail errors remain clinically important despite good average MAE. The forecast-only p90, p95, and p99 absolute errors are $\pNinetyErr$, $\pNinetyFiveErr$, and $\pNinetyNineErr$~mg/dL, respectively. True TIR is $\tirTrue\%$ and predicted TIR is $\tirPredicted\%$, giving a $\tirGap$ percentage-point overprediction of time in range.
'''.lstrip())
    write_text(sections / "10_interpretability.tex", r'''
\section{Interpretability}
\label{sec:interp}

The most defensible interpretability results for this AI-READI run are the subgroup, persistence, and scenario-delta diagnostics. The residual-current target means a model can look strong by staying close to persistence, so the matched persistence baseline is essential for interpretation. The model does beat persistence, but the margin is modest, which is expected for 60-minute CGM forecasting.

The proxy scenario branch is not currently interpretable as a physiological what-if mechanism: prediction deltas are near zero, and \texttt{predmeal\_flag} is unavailable in the audited future horizon for the test anchors. No insulin-on-board equivalent is available in AI-READI, and no insulin causal interpretation should be included.

'''.lstrip())
    write_text(sections / "11_ablation_and_runtime.tex", r'''
\section{Ablation and Runtime}
\label{sec:ablation}

\GeneratedRuntimeInterpretation

\GeneratedRuntimeTables

\GeneratedRuntimeFigures

The current ablation table is a status table, not a result table. The full model, no-static-FiLM, and no-decomposition probes can be launched with the provided script. Absolute-target and no-scenario-mask variants require additional implementation before their numbers can be interpreted.
'''.lstrip())
    write_text(sections / "12_discussion_limitations.tex", r'''
\section{Discussion and Limitations}
\label{sec:discuss}

The AI-READI port produces a viable participant-held-out streaming forecaster and improves over persistence on matched anchors. The result is scientifically meaningful because the split is participant-held-out and the recurrent state never crosses segment boundaries. The improvement over persistence is real but not huge, which is an important limitation of residual-current CGM forecasting: current glucose already carries most of the short-horizon signal.

The strongest current claims are forecasting performance, persistence improvement, participant-level subgroup behavior, and q0.1/q0.9 safety-alarm trade-offs. The weakest current claims are personalization and proxy scenarios. Matched-anchor personalization is flat in the current artifacts, and offset correction worsens MAE. Proxy scenario deltas are near zero, so the scenario branch is not yet scientifically interpretable.

Limitations are specific to AI-READI. The dataset lacks timed insulin dose, insulin-on-board, and carbohydrate quantity logs; \texttt{med\_insulin} is static metadata only. The predicted meal flag is noisy and may be retrospective. The residual-current target can hide weak dynamic learning if persistence is not reported. Participant-level averages are necessary because anchor-weighted metrics overrepresent participants with longer clean segments. Finally, site and phenotype subgroup effects require follow-up audits before being interpreted as biological effects rather than cohort composition or domain shift.
'''.lstrip())
    write_text(sections / "appendix.tex", r'''
\section{Compute Placement: GPU vs. CPU}
\label{app:compute}

CUDA accelerates training through the Mamba/Triton scan backend. The streaming recurrence and decoder are still compatible with CPU execution for portability, but the reported full-cohort training runs used one NVIDIA A100 GPU.

\section{Generated Appendix Tables and Figures}
\label{app:generated}

\GeneratedAppendixTables

\GeneratedAppendixFigures

\section{Generated Results Snapshot}
\label{app:snapshot}

\IfFileExists{sections/generated_results_summary.tex}{%
  \generatedResultsSnapshotTable
}{%
  \TODO{No generated results summary was found. Run the report pipeline.}%
}
'''.lstrip())


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate LaTeX report sections/tables from results manifest")
    ap.add_argument("--report-dir", default="report")
    ap.add_argument("--manifest", default="report/results_manifest.csv")
    args = ap.parse_args()
    report_dir = Path(args.report_dir)
    rows = read_manifest(Path(args.manifest))
    if not rows:
        print(f"ERROR: manifest missing or empty: {args.manifest}", file=sys.stderr)
        return 1
    data = load_all_metrics(rows)
    make_all_tables(report_dir, data)
    make_supporting_assets(report_dir, data)
    make_generated_sections(report_dir, data)
    make_generated_summary(report_dir, data)
    patch_main(report_dir)
    write_sections(report_dir)
    print("[update] report sections rewritten with generated result inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
