#!/usr/bin/env python3
"""Inventory generated AI-READI stream outputs for the LaTeX report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.report.stream_report_utils import (
    add_missing_required,
    classify,
    discover_outputs,
    infer_training_run_for_eval,
    normalize_run,
    write_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Recursively inventory outputs/ for report-safe results")
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--report-dir", default="report")
    ap.add_argument("--run", default=None, help="Selected eval run name or path; default latest eval_test")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    report_dir = Path(args.report_dir)
    if not outputs_root.exists():
        print(f"ERROR: outputs root not found: {outputs_root}", file=sys.stderr)
        return 1
    selected_run = normalize_run(outputs_root, args.run)
    training_run = infer_training_run_for_eval(selected_run, outputs_root)

    paths = discover_outputs(outputs_root)
    rows = [classify(path, outputs_root, selected_run, training_run) for path in paths]
    rows = add_missing_required(rows, selected_run)
    out = report_dir / "results_manifest.csv"
    write_manifest(out, rows)

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"[inventory] discovered {len(paths)} files under {outputs_root}")
    print(f"[inventory] selected run: {selected_run if selected_run else 'none'}")
    print(f"[inventory] training run: {training_run if training_run else 'none'}")
    print(f"[inventory] wrote {out}")
    print("[inventory] status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
