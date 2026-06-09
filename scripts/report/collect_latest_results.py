#!/usr/bin/env python3
"""
collect_latest_results.py
Search outputs/aireadi_stream_full*/ for metrics files and collect them into
a single JSON summary for the report pipeline.

Usage:
    python scripts/report/collect_latest_results.py --outputs-root outputs
    python scripts/report/collect_latest_results.py --outputs-root outputs \
        --run-name aireadi_stream_mamba_stateful_5epoch_eval_test
"""
import argparse
import json
import sys
from pathlib import Path


def find_runs(outputs_root: Path) -> list[Path]:
    """Return all aireadi_stream_full* and eval run directories, newest first."""
    candidates = sorted(
        [p for p in outputs_root.iterdir()
         if p.is_dir() and (p.name.startswith("aireadi_stream_full") or
                            "eval" in p.name)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates


def load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARN: could not load {path}: {e}", file=sys.stderr)
        return {}


def load_csv_rows(path: Path) -> list[dict]:
    import csv
    rows = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception as e:
        print(f"  WARN: could not load {path}: {e}", file=sys.stderr)
    return rows


def collect_run(run_dir: Path) -> dict:
    result = {"run_name": run_dir.name, "run_path": str(run_dir)}

    # config
    cfg_path = run_dir / "config_resolved.yaml"
    if cfg_path.exists():
        result["config_path"] = str(cfg_path)
        try:
            import yaml  # type: ignore
            with open(cfg_path) as f:
                result["config"] = yaml.safe_load(f)
        except ImportError:
            result["config"] = "(yaml not available)"
        except Exception as e:
            result["config"] = f"(error: {e})"

    metrics_dir = run_dir / "metrics"
    hw_dir = run_dir / "hardware"

    # overall metrics
    for fname in ["overall_metrics.json", "clinical_safety.json"]:
        p = metrics_dir / fname
        if p.exists():
            result[fname.replace(".json", "")] = load_json(p)

    # CSV metrics
    for fname in ["personalization_sweep.csv", "subgroup_metrics.csv",
                  "scenario_metrics.csv", "horizon_metrics.csv",
                  "bias_diagnostic.csv"]:
        p = metrics_dir / fname
        if p.exists():
            result[fname.replace(".csv", "")] = load_csv_rows(p)

    # hardware metrics
    for fname in ["hardware_metrics.json"]:
        p = hw_dir / fname
        if p.exists():
            result[fname.replace(".json", "")] = load_json(p)

    # training summary
    for fname in ["training_summary.json"]:
        p = metrics_dir / fname
        if p.exists():
            result["training_summary"] = load_json(p)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--run-name", default=None,
                        help="Use a specific run name instead of auto-detecting.")
    parser.add_argument("--out", default="outputs/_report_collected.json")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    if not outputs_root.exists():
        print(f"ERROR: outputs root {outputs_root} not found", file=sys.stderr)
        sys.exit(1)

    if args.run_name:
        run_dir = outputs_root / args.run_name
        if not run_dir.exists():
            print(f"ERROR: run {run_dir} not found", file=sys.stderr)
            sys.exit(1)
        runs = [run_dir]
    else:
        runs = find_runs(outputs_root)
        if not runs:
            print("No matching output runs found.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(runs)} run(s). Using latest: {runs[0].name}")
        runs = runs[:1]

    collected = collect_run(runs[0])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(collected, f, indent=2, default=str)
    print(f"Saved collected results to {out_path}")


if __name__ == "__main__":
    main()
