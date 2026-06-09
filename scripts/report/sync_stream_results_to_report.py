#!/usr/bin/env python3
"""Copy selected report-safe stream outputs into report/generated folders."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.report.stream_report_utils import (
    add_missing_required,
    classify,
    copy_selected,
    discover_outputs,
    infer_training_run_for_eval,
    normalize_run,
    read_manifest,
    write_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync selected AI-READI stream results into report/generated folders")
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--report-dir", default="report")
    ap.add_argument("--run", default=None, help="Selected eval run name or path; default latest eval_test")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    report_dir = Path(args.report_dir)
    manifest = Path(args.manifest) if args.manifest else report_dir / "results_manifest.csv"
    selected_run = normalize_run(outputs_root, args.run)
    training_run = infer_training_run_for_eval(selected_run, outputs_root)

    if manifest.exists():
        # Reclassify to honor the selected run passed to sync.
        source_paths = [Path(r["source_path"]) for r in read_manifest(manifest) if r.get("source_path") and Path(r["source_path"]).exists()]
    else:
        source_paths = discover_outputs(outputs_root)
    rows = [classify(path, outputs_root, selected_run, training_run) for path in sorted(set(source_paths))]
    rows = add_missing_required(rows, selected_run)
    rows = copy_selected(rows, report_dir, selected_run)
    write_manifest(manifest, rows)

    copied = sum(1 for r in rows if r.get("destination_path"))
    main = sum(1 for r in rows if r.get("status") == "main_report")
    appendix = sum(1 for r in rows if r.get("status") == "appendix")
    missing = sum(1 for r in rows if r.get("status") == "missing")
    print(f"[sync] selected run: {selected_run if selected_run else 'none'}")
    print(f"[sync] copied {copied} safe files into {report_dir}/figures/generated and {report_dir}/tables/generated")
    print(f"[sync] main_report={main} appendix={appendix} missing={missing}")
    print(f"[sync] updated {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
