#!/usr/bin/env python3
"""Measure AI-READI stream update/forecast latency and state memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from scripts.evaluate_stream_aireadi import build_eval_streams, load_model_from_checkpoint, resolve_device
from ssmcgm.evaluation.aireadi_streaming import evaluate_aireadi_streams


def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark AI-READI stream checkpoint")
    ap.add_argument("--config", default=str(ROOT / "configs/aireadi_stream_full.yaml"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-participants", type=int, default=2,
                    help="number of built eval streams to benchmark after smoke/full split construction")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = resolve_device(args.device)
    base_out = Path(cfg["output"]["base_dir"])
    ckpt = Path(args.ckpt or (base_out / "checkpoints" / "best_model_checkpoint.pt"))
    model, spec, pre, _ = load_model_from_checkpoint(ckpt, device)
    streams = build_eval_streams(cfg, spec, pre, "test", smoke=args.smoke, max_participants=None)
    if args.max_participants:
        streams = streams[:args.max_participants]
    if not streams:
        raise RuntimeError("No test streams were built for benchmarking. Check split/data config.")
    res = evaluate_aireadi_streams(
        model, streams, scenario_mode="forecast_only",
        anchor_stride_steps=max(1, round(cfg.get("evaluation", {}).get("anchor_stride_minutes", 15) / spec.bin_minutes)),
        max_anchors_per_stream=cfg.get("smoke", {}).get("max_anchors_per_stream") if args.smoke else 24,
        device=device, record_latency=True,
    )
    out_dir = base_out / "hardware"
    out_dir.mkdir(parents=True, exist_ok=True)
    res["memory"].to_csv(out_dir / "benchmark_stream_state_memory.csv", index=False)
    with (out_dir / "benchmark_hardware_metrics.json").open("w") as f:
        json.dump(res["hardware"], f, indent=2)
    print(json.dumps(res["hardware"], indent=2))


if __name__ == "__main__":
    main()
