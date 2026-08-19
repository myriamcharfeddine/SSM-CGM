#!/usr/bin/env python3
"""
make_report_tables.py
Generate LaTeX tables from collected metrics and write them to
report/tables/generated/.

Usage:
    python scripts/report/make_report_tables.py \
        --collected outputs/_report_collected.json \
        --out-dir report/tables/generated
"""
import argparse
import json
import sys
from pathlib import Path


def fmt(val, decimals=2, fallback="---"):
    """Format a numeric value or return fallback."""
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return fallback


def write_table(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Wrote {path}")


def make_overall_metrics(data: dict, out_dir: Path):
    m = data.get("overall_metrics", {})
    if not m:
        print("  SKIP overall_metrics: no data")
        return
    rows = [
        ("$n$ (5-min anchors)", f"{int(m.get('n', 0)):,}"),
        ("MAE (mg/dL) $\\downarrow$", fmt(m.get("mae"))),
        ("RMSE (mg/dL) $\\downarrow$", fmt(m.get("rmse"))),
        ("Bias (mg/dL)", fmt(m.get("bias"), decimals=3)),
        ("80\\% coverage $\\uparrow$", fmt(m.get("coverage80"), decimals=3)),
        ("TIR true (\\%)", fmt(float(m.get("tir_true", 0)) * 100, decimals=1)),
        ("TIR predicted (\\%)", fmt(float(m.get("tir_predicted", 0)) * 100, decimals=1)),
        ("TIR gap (pp)", fmt(float(m.get("tir_gap", 0)) * 100, decimals=2)),
        ("p90 abs.~error (mg/dL)", fmt(m.get("p90_abs_error"))),
        ("p95 abs.~error (mg/dL)", fmt(m.get("p95_abs_error"))),
        ("p99 abs.~error (mg/dL)", fmt(m.get("p99_abs_error"))),
    ]
    lines = ["\\begin{tabular}{lc}", "\\toprule",
             "Metric & Value \\\\", "\\midrule"]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out_dir / "overall_metrics.tex", "\n".join(lines) + "\n")


def make_scenario_metrics(data: dict, out_dir: Path):
    rows = data.get("scenario_metrics", [])
    if not rows:
        print("  SKIP scenario_metrics: no data")
        return
    mode_labels = {
        "forecast_only": "forecast-only (deployable)",
        "factual_future": "factual-future (oracle)",
        "meal_proxy": "meal\\_proxy",
        "activity_proxy": "activity\\_proxy",
        "sleep_rest_proxy": "sleep\\_rest\\_proxy",
    }
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Mode & MAE (mg/dL) $\\downarrow$ & RMSE & Bias & Cov.~80\\% \\\\",
             "\\midrule"]
    for row in rows:
        mode = mode_labels.get(row.get("scenario_mode", ""), row.get("scenario_mode", ""))
        lines.append(
            f"{mode} & {fmt(row.get('mae'))} & {fmt(row.get('rmse'))} "
            f"& {fmt(row.get('bias'), decimals=3)} & {fmt(row.get('coverage80'), decimals=3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out_dir / "scenario_metrics.tex", "\n".join(lines) + "\n")


def make_subgroup_metrics(data: dict, out_dir: Path):
    rows = data.get("subgroup_metrics", [])
    if not rows:
        print("  SKIP subgroup_metrics: no data")
        return
    study_rows = [r for r in rows if r.get("subgroup") == "participants_study_group"]
    if not study_rows:
        print("  SKIP subgroup_metrics: no study_group rows")
        return
    group_labels = {
        "healthy": "Healthy",
        "pre_diabetes_lifestyle_controlled": "Pre-diabetes (lifestyle)",
        "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral/injectable med",
        "insulin_dependent": "Insulin-dependent",
    }
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Study group & MAE (mg/dL) $\\downarrow$ & Bias \\\\",
             "\\midrule"]
    for row in study_rows:
        label = group_labels.get(row.get("level", ""), row.get("level", ""))
        lines.append(
            f"{label} & {fmt(row.get('mae'))} & {fmt(row.get('bias'), decimals=3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out_dir / "subgroup_metrics.tex", "\n".join(lines) + "\n")


def make_personalization_warmup(data: dict, out_dir: Path):
    rows = data.get("personalization_sweep", [])
    if not rows:
        print("  SKIP personalization_warmup: no data")
        return
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Warm-up (h) & MAE (mg/dL) $\\downarrow$ & Bias \\\\",
             "\\midrule"]
    for row in rows:
        wh = fmt(float(row.get("warmup_hours", 0)), decimals=0)
        lines.append(
            f"{wh} & {fmt(row.get('mae'))} & {fmt(row.get('bias'), decimals=3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out_dir / "personalization_warmup.tex", "\n".join(lines) + "\n")


def make_bias_diagnostic(data: dict, out_dir: Path):
    rows = data.get("bias_diagnostic", [])
    if not rows:
        print("  SKIP bias_diagnostic: no data")
        return
    mode_labels = {
        "raw": "Raw (no personalization)",
        "offset-corrected": "Offset-corrected",
        "personalized": "Personalized",
        "personalized+offset-corrected": "Personalized + offset-corrected",
    }
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Mode & MAE (mg/dL) $\\downarrow$ & Bias (mg/dL) \\\\",
             "\\midrule"]
    for row in rows:
        label = mode_labels.get(row.get("bias_mode", ""), row.get("bias_mode", ""))
        lines.append(
            f"{label} & {fmt(row.get('mae'))} & {fmt(row.get('bias'), decimals=3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out_dir / "bias_diagnostic.tex", "\n".join(lines) + "\n")


def make_clinical_safety(data: dict, out_dir: Path):
    cs = data.get("clinical_safety", {})
    if not cs:
        print("  SKIP clinical_safety: no data")
        return
    hypo = cs.get("hypoglycemia_detection", {})
    hyper = cs.get("hyperglycemia_detection", {})
    lines = ["\\begin{tabular}{llcc}", "\\toprule",
             "Event & Rule & Precision $\\uparrow$ & Recall $\\uparrow$ \\\\",
             "\\midrule"]
    if hypo:
        lines.append(
            f"Hypo $<70$ & median & "
            f"{fmt(hypo.get('precision'), decimals=3)} & "
            f"{fmt(hypo.get('recall'), decimals=3)} \\\\"
        )
    if hyper:
        lines.append(
            f"Hyper $>180$ & median & "
            f"{fmt(hyper.get('precision'), decimals=3)} & "
            f"{fmt(hyper.get('recall'), decimals=3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out_dir / "clinical_safety.tex", "\n".join(lines) + "\n")


def make_horizon_metrics(data: dict, out_dir: Path):
    rows = data.get("horizon_metrics", [])
    if not rows:
        print("  SKIP horizon_metrics: no data")
        return
    lines = ["\\begin{tabular}{cccc}", "\\toprule",
             "Step & Minutes & MAE (mg/dL) $\\downarrow$ & RMSE \\\\",
             "\\midrule"]
    for row in rows:
        step = row.get("horizon_step", "")
        mins = row.get("horizon_minutes", "")
        lines.append(
            f"{step} & {mins} & {fmt(row.get('mae'))} & {fmt(row.get('rmse'))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out_dir / "horizon_metrics.tex", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected", default="outputs/_report_collected.json")
    parser.add_argument("--out-dir", default="report/tables/generated")
    args = parser.parse_args()

    collected_path = Path(args.collected)
    if not collected_path.exists():
        print(f"ERROR: {collected_path} not found.  Run collect_latest_results.py first.",
              file=sys.stderr)
        sys.exit(1)

    with open(collected_path) as f:
        data = json.load(f)

    out_dir = Path(args.out_dir)
    print(f"Writing tables to {out_dir}/")

    make_overall_metrics(data, out_dir)
    make_scenario_metrics(data, out_dir)
    make_subgroup_metrics(data, out_dir)
    make_personalization_warmup(data, out_dir)
    make_bias_diagnostic(data, out_dir)
    make_clinical_safety(data, out_dir)
    make_horizon_metrics(data, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
