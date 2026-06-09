#!/usr/bin/env python3
"""Run AI-READI stream evaluation diagnostics from existing prediction outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.evaluate_stream_aireadi import build_eval_streams, load_config, load_model_from_checkpoint, resolve_device
from ssmcgm.evaluation.baselines import persistence_baseline
from ssmcgm.evaluation.diagnostics import (
    ensure_output_dirs,
    load_predictions,
    maybe_copy_report_outputs,
    write_summary,
)
from ssmcgm.evaluation.event_detection import event_detection_quantile_alarms
from ssmcgm.evaluation.participant_metrics import participant_level_metrics
from ssmcgm.evaluation.plotting import matched_personalization, subgroup_outputs
from ssmcgm.evaluation.scenario_audit import scenario_pathway_audit, scenario_prediction_deltas

TASKS = [
    "persistence",
    "participant_metrics",
    "matched_personalization",
    "scenario_audit",
    "q01_hypo",
    "subgroup_plots",
]
STREAM_TASKS = {"persistence", "scenario_audit"}


def parse_args():
    ap = argparse.ArgumentParser(description="Run AI-READI stream diagnostics from evaluation outputs")
    ap.add_argument("--config", default=str(ROOT / "configs/aireadi_stream_full.yaml"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-dir", required=True, help="Existing evaluate_stream_aireadi output directory")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--max-participants", type=int, default=None, help="Optional debug cap when rebuilding streams")
    ap.add_argument("--smoke", action="store_true", help="Use smoke participant caps while rebuilding streams")
    ap.add_argument("--warmup-hours", type=float, nargs="+", default=[0, 6, 12, 24, 48])
    return ap.parse_args()


def infer_split(predictions: pd.DataFrame) -> str:
    if "split" not in predictions.columns:
        raise ValueError("Prediction parquet has no split column; pass an eval output produced by evaluate_stream_aireadi.py")
    vals = sorted(str(v) for v in predictions["split"].dropna().unique())
    if len(vals) != 1:
        raise ValueError(f"Expected exactly one split in prediction parquet, found {vals}")
    return vals[0]


def maybe_build_streams(args, cfg, split_name: str, tasks: Sequence[str]):
    if not (set(tasks) & STREAM_TASKS):
        return None, None, None, None
    print("[diagnostics] rebuilding eval streams for tasks needing anchor-current/scenario-mask data")
    # Stream rebuilding is CPU work; checkpoint metadata is enough, so load on CPU even if diagnostics later run on CUDA.
    model, spec, pre, ckpt = load_model_from_checkpoint(args.ckpt, "cpu")
    streams = build_eval_streams(
        cfg, spec, pre, split_name, smoke=bool(args.smoke), max_participants=args.max_participants
    )
    print(f"[diagnostics] rebuilt {len(streams)} {split_name} streams")
    return model, spec, pre, streams


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    dirs = ensure_output_dirs(args.output_dir)
    generated: List[str] = []
    warnings: List[str] = []

    pred = load_predictions(args.eval_dir)
    split_name = infer_split(pred)
    print(f"[diagnostics] loaded predictions: rows={len(pred):,} split={split_name} eval_dir={args.eval_dir}")
    model, spec, pre, streams = maybe_build_streams(args, cfg, split_name, args.tasks)

    participant_df = None
    if "persistence" in args.tasks:
        if streams is None:
            raise RuntimeError("persistence task requires rebuilt streams")
        print("[diagnostics] task=persistence")
        generated.extend(persistence_baseline(pred, streams, dirs["metrics"]))

    if "participant_metrics" in args.tasks or "subgroup_plots" in args.tasks:
        print("[diagnostics] task=participant_metrics")
        participant_df, files = participant_level_metrics(pred, dirs["metrics"])
        generated.extend(files)

    if "matched_personalization" in args.tasks:
        print("[diagnostics] task=matched_personalization")
        print("[diagnostics] matched personalization uses the same eligible anchors for every warm-up row")
        generated.extend(matched_personalization(pred, dirs["metrics"], dirs["figures"], warmup_hours=args.warmup_hours))

    if "scenario_audit" in args.tasks:
        if streams is None or spec is None:
            raise RuntimeError("scenario_audit task requires rebuilt streams")
        print("[diagnostics] task=scenario_audit")
        files, task_warnings = scenario_prediction_deltas(pred, dirs["metrics"], dirs["figures"])
        generated.extend(files)
        warnings.extend(task_warnings)
        generated.extend(scenario_pathway_audit(pred, streams, spec, dirs["metrics"]))

    if "q01_hypo" in args.tasks:
        print("[diagnostics] task=q01_hypo")
        generated.extend(event_detection_quantile_alarms(pred, dirs["metrics"], dirs["figures"]))

    if "subgroup_plots" in args.tasks:
        print("[diagnostics] task=subgroup_plots")
        if participant_df is None:
            participant_df, files = participant_level_metrics(pred, dirs["metrics"])
            generated.extend(files)
        generated.extend(subgroup_outputs(participant_df, dirs["metrics"], dirs["figures"]))

    copied = maybe_copy_report_outputs(args.output_dir, generated)
    summary = write_summary(
        args.output_dir,
        tasks=args.tasks,
        generated_files=generated,
        warnings=warnings,
        copied_files=copied,
        extra={
            "config": str(args.config),
            "checkpoint": str(args.ckpt),
            "eval_dir": str(args.eval_dir),
            "split": split_name,
            "device_requested": args.device,
            "device_resolved": device,
            "n_prediction_rows": int(len(pred)),
        },
    )
    print(f"[diagnostics] wrote summary: {summary}")
    if warnings:
        for warning in warnings:
            print(f"[diagnostics] WARNING: {warning}")


if __name__ == "__main__":
    main()
