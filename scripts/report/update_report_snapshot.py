#!/usr/bin/env python3
"""
update_report_snapshot.py
Pull the latest metrics from outputs/ and write
report/sections/generated_results_summary.tex with up-to-date providecommand
overrides that macros.tex uses as fallbacks.

Usage:
    python scripts/report/update_report_snapshot.py --outputs-root outputs
    python scripts/report/update_report_snapshot.py \
        --collected outputs/_report_collected.json
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path


def find_latest_eval_run(outputs_root: Path) -> Path | None:
    """Return the most recently modified eval run directory."""
    candidates = sorted(
        [p for p in outputs_root.iterdir()
         if p.is_dir() and ("eval_test" in p.name or "eval_validation" in p.name or
                            p.name.startswith("aireadi_stream_full"))],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_csv_first_row(path: Path) -> dict:
    import csv
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            return next(reader, {})
    except Exception:
        return {}


def load_csv_all_rows(path: Path) -> list[dict]:
    import csv
    rows = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception:
        pass
    return rows


def safe_float(val, decimals=2, scale=1.0, fallback=r"\PENDING"):
    try:
        return f"{float(val) * scale:.{decimals}f}"
    except (TypeError, ValueError):
        return fallback


def make_macros(run_dir: Path, run_date: str) -> dict[str, str]:
    """Return {macro_name: value} dict from available metrics files."""
    macros: dict[str, str] = {}
    P = r"\PENDING"

    macros["reportRunName"] = run_dir.name
    macros["reportRunDate"] = run_date

    # overall metrics
    m = load_json(run_dir / "metrics" / "overall_metrics.json")
    macros["maeFo"] = safe_float(m.get("mae"))
    macros["rmseFo"] = safe_float(m.get("rmse"))
    macros["biasFo"] = safe_float(m.get("bias"), decimals=3)
    macros["pinFo"] = safe_float(m.get("pinball") or m.get("pinball_loss"), fallback=P)
    macros["covFo"] = safe_float(m.get("coverage80"), decimals=3)
    macros["nAnchorsEval"] = (
        f"{int(m.get('n', 0)):,}" if m.get("n") else P
    )
    macros["tirTrue"] = safe_float(m.get("tir_true"), decimals=3)
    macros["tirPredicted"] = safe_float(m.get("tir_predicted"), decimals=3)
    macros["tirGap"] = safe_float(m.get("tir_gap"), decimals=4)
    macros["pNinetyErr"] = safe_float(m.get("p90_abs_error"))
    macros["pNinetyFiveErr"] = safe_float(m.get("p95_abs_error"))
    macros["pNinetyNineErr"] = safe_float(m.get("p99_abs_error"))

    # clinical safety (stored inside overall_metrics.json for eval runs)
    cs_hypo = m.get("hypoglycemia_detection", {})
    cs_hyper = m.get("hyperglycemia_detection", {})
    macros["hypoPrec"] = safe_float(cs_hypo.get("precision"), decimals=3)
    macros["hypoRecall"] = safe_float(cs_hypo.get("recall"), decimals=3)
    macros["hyperPrec"] = safe_float(cs_hyper.get("precision"), decimals=3)
    macros["hyperRecall"] = safe_float(cs_hyper.get("recall"), decimals=3)

    # scenario metrics
    sc_rows = load_csv_all_rows(run_dir / "metrics" / "scenario_metrics.csv")
    for row in sc_rows:
        mode = row.get("scenario_mode", "")
        mae_val = safe_float(row.get("mae"))
        if mode == "forecast_only":
            macros["maeFoForecastOnly"] = mae_val
        elif mode == "factual_future":
            macros["maeFoFactual"] = mae_val
        elif mode == "meal_proxy":
            macros["maeFoMealProxy"] = mae_val
        elif mode == "activity_proxy":
            macros["maeFoActivityProxy"] = mae_val

    # personalization sweep
    pers_rows = load_csv_all_rows(run_dir / "metrics" / "personalization_sweep.csv")
    for row in pers_rows:
        try:
            wh = float(row.get("warmup_hours", -1))
        except (TypeError, ValueError):
            continue
        mae_val = safe_float(row.get("mae"))
        if wh == 0.0:
            macros["maeWarmZero"] = mae_val
        elif wh == 6.0:
            macros["maeWarmSix"] = mae_val
        elif wh == 12.0:
            macros["maeWarmTwelve"] = mae_val
        elif wh == 24.0:
            macros["maeWarmTwentyFour"] = mae_val
        elif wh == 48.0:
            macros["maeWarmFortyEight"] = mae_val

    # bias diagnostic
    bias_rows = load_csv_all_rows(run_dir / "metrics" / "bias_diagnostic.csv")
    for row in bias_rows:
        mode = row.get("bias_mode", "")
        if mode == "raw":
            macros["biasRaw"] = safe_float(row.get("bias"), decimals=3)
        elif mode == "offset-corrected":
            macros["biasOffsetCorrected"] = safe_float(row.get("bias"), decimals=3)

    # subgroup metrics
    sg_rows = load_csv_all_rows(run_dir / "metrics" / "subgroup_metrics.csv")
    for row in sg_rows:
        if row.get("subgroup") != "participants_study_group":
            continue
        level = row.get("level", "")
        mae_val = safe_float(row.get("mae"))
        if level == "healthy":
            macros["maeHealthy"] = mae_val
        elif level == "insulin_dependent":
            macros["maeInsulinDep"] = mae_val
        elif "oral_medication" in level:
            macros["maeOralMed"] = mae_val
        elif "pre_diabetes" in level:
            macros["maePreDiabetes"] = mae_val

    # hardware metrics
    hw = load_json(run_dir / "hardware" / "hardware_metrics.json")
    lu = hw.get("latency_per_update", {})
    lf = hw.get("latency_per_1h_forecast", {})
    macros["latencyUpdateMsMedian"] = safe_float(lu.get("median_ms"))
    macros["latencyForecastMsMean"] = safe_float(lf.get("mean_ms"), decimals=4)
    macros["peakInferMemMB"] = safe_float(hw.get("peak_gpu_memory_mb"))
    macros["cpuMemRssMB"] = safe_float(hw.get("cpu_memory_rss_mb"), decimals=0)

    # training summary (from the training run, not eval run)
    # look in parent or sibling directories
    train_dir_name = run_dir.name.replace("_eval_test", "").replace("_eval_validation", "")
    train_dir = run_dir.parent / train_dir_name
    ts = load_json(train_dir / "metrics" / "training_summary.json") if train_dir.exists() else {}
    macros["bestValPinball"] = safe_float(ts.get("best_val_pinball_mgdl"))
    if ts.get("history"):
        hist = ts["history"]
        val_losses = [e.get("val_pinball_mgdl", float("inf")) for e in hist]
        best_ep = hist[val_losses.index(min(val_losses))].get("epoch", "?")
        macros["bestValEpoch"] = str(best_ep)
        last_epoch = hist[-1]
        macros["peakTrainMemMB"] = safe_float(last_epoch.get("peak_mem_mb"), decimals=0)
        # use max anchors/s across epochs
        throughputs = [e.get("anchors_per_s", 0) for e in ts["history"]]
        if throughputs:
            macros["anchorsPerSec"] = f"{max(throughputs):.0f}"

    return macros


def render_tex(macros: dict[str, str], run_dir: Path, run_date: str) -> str:
    lines = [
        "% AUTO-GENERATED by update_report_snapshot.py — do not edit by hand",
        f"% Run: {run_dir.name}",
        f"% Date: {run_date}",
        "%",
        "% These commands override the \\PENDING fallbacks in macros.tex.",
        "",
    ]
    for name, val in sorted(macros.items()):
        # Escape backslashes already present (\\PENDING passthrough)
        if val == r"\PENDING":
            continue  # let macros.tex fallback handle it
        lines.append(f"\\renewcommand{{\\{name}}}{{{val}}}")

    # Snapshot table for appendix
    lines += [
        "",
        "% Snapshot table for Appendix",
        "\\newcommand{\\generatedResultsSnapshotTable}{%",
        "  \\begin{tabular}{ll}",
        "  \\toprule",
        "  Field & Value \\\\",
        "  \\midrule",
        f"  Run name & \\texttt{{{run_dir.name}}} \\\\",
        f"  Report date & {run_date} \\\\",
    ]
    for name, val in sorted(macros.items()):
        if name in ("reportRunName", "reportRunDate"):
            continue
        if val == r"\PENDING":
            continue
        label = name.replace("_", "\\_")
        lines.append(f"  {label} & {val} \\\\")
    lines += ["  \\bottomrule", "  \\end{tabular}", "}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--collected", default=None,
                        help="Use pre-collected JSON instead of scanning outputs/.")
    parser.add_argument("--out", default="report/sections/generated_results_summary.tex")
    args = parser.parse_args()

    run_date = datetime.date.today().isoformat()

    if args.collected:
        cpath = Path(args.collected)
        if not cpath.exists():
            print(f"ERROR: {cpath} not found", file=sys.stderr)
            sys.exit(1)
        with open(cpath) as f:
            data = json.load(f)
        run_dir = Path(data.get("run_path", "."))
    else:
        outputs_root = Path(args.outputs_root)
        run_dir = find_latest_eval_run(outputs_root)
        if run_dir is None:
            print("No eval run found; writing empty snapshot.", file=sys.stderr)
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                "% AUTO-GENERATED — no results found yet\n"
                f"% Date: {run_date}\n"
            )
            print(f"Wrote empty snapshot to {out}")
            return

    print(f"Using run: {run_dir}")
    macros = make_macros(run_dir, run_date)
    tex = render_tex(macros, run_dir, run_date)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"Wrote snapshot to {out}")
    print(f"  {len([v for v in macros.values() if v != r'PENDING'])} macros populated")


if __name__ == "__main__":
    main()
