#!/usr/bin/env python3
"""Evaluate one AI-READI stream checkpoint across forecast/proxy modes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import yaml

from ssmcgm.data.aireadi import (
    AireadiFeatureSpec,
    AireadiPreprocessor,
    build_stream_feature_spec,
    infer_or_validate_schema,
    load_aireadi_panel,
    make_aireadi_stream_splits,
    make_participant_streams,
    prepare_aireadi_panel,
)
from ssmcgm.evaluation.aireadi_streaming import evaluate_aireadi_streams, write_evaluation_bundle
from ssmcgm.models.aireadi_stream import AireadiStreamModel, AireadiStreamModelConfig


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate AI-READI stream checkpoint")
    ap.add_argument("--config", default=str(ROOT / "configs/aireadi_stream_full.yaml"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-participants", type=int, default=None)
    ap.add_argument("--max-anchors-per-stream", type=int, default=None)
    return ap.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return requested


def load_model_from_checkpoint(ckpt_path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    md = ckpt["metadata"]
    spec = AireadiFeatureSpec(**md["feature_spec"])
    pre = AireadiPreprocessor.from_jsonable(md["preprocessor"])
    mcfg = AireadiStreamModelConfig(**md["model_config"])
    model = AireadiStreamModel(spec, pre, mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, spec, pre, ckpt


def build_eval_streams(cfg, spec, pre, split_name: str, *, smoke: bool, max_participants=None):
    dc = cfg["data"]
    smoke_participants = max_participants or (cfg.get("smoke", {}).get("participants") if smoke else None)
    df = load_aireadi_panel(
        dc["panel_path"],
        static_path=dc.get("static_path"),
        cohort_path=dc.get("cohort_path"),
        smoke_participants=smoke_participants,
    )
    schema = infer_or_validate_schema(df, dc.get("schema"))
    df = prepare_aireadi_panel(
        df, schema,
        bin_minutes=cfg["dataset"].get("bin_minutes", 5),
        clean_min_segment_hours=cfg["dataset"].get("clean_min_segment_hours", 49),
    )
    split_cfg = cfg.get("split", {})
    split = make_aireadi_stream_splits(
        df,
        split_mode=split_cfg.get("mode", "participant_heldout"),
        train=split_cfg.get("train", 0.70),
        val=split_cfg.get("val", 0.15),
        test=split_cfg.get("test", 0.15),
        seed=split_cfg.get("seed", 42),
        stratify_col=split_cfg.get("stratify_col"),
        existing_split_path=split_cfg.get("existing_split_path"),
    )
    # Use the checkpoint feature spec/preprocessor, but validate that current panel
    # still contains the referenced columns.
    build_stream_feature_spec(df, schema, horizon_steps=spec.horizon_steps, bin_minutes=spec.bin_minutes)
    return make_participant_streams(
        df, split, schema, feature_spec=spec, preprocessor=pre,
        splits=[split_name], min_steps=spec.horizon_steps + 2,
        max_participants=(cfg.get("smoke", {}).get("eval_participants") if smoke else max_participants),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    base_out = Path(args.output_dir or cfg["output"]["base_dir"])
    ckpt_path = Path(args.ckpt or (base_out / "checkpoints" / "best_model_checkpoint.pt"))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model, spec, pre, ckpt = load_model_from_checkpoint(ckpt_path, device)
    streams = build_eval_streams(cfg, spec, pre, args.split, smoke=args.smoke, max_participants=args.max_participants)
    if not streams:
        raise RuntimeError(f"No {args.split} streams were built for evaluation.")
    eval_cfg = cfg.get("evaluation", {})
    stride_steps = max(1, round(eval_cfg.get("anchor_stride_minutes", 15) / spec.bin_minutes))
    max_anchors = args.max_anchors_per_stream or (cfg.get("smoke", {}).get("max_anchors_per_stream") if args.smoke else None)
    modes = ["forecast_only", "factual_future", "meal_proxy", "activity_proxy", "sleep_rest_proxy"]
    frames = []
    memory_frames = []
    hardware = {}
    for mode in modes:
        print(f"[evaluate_stream_aireadi] mode={mode} streams={len(streams)}")
        res = evaluate_aireadi_streams(
            model, streams, scenario_mode=mode, anchor_stride_steps=stride_steps,
            warmup_steps=0, max_anchors_per_stream=max_anchors, device=device,
            record_latency=(mode == "forecast_only"),
        )
        frames.append(res["predictions"])
        memory_frames.append(res["memory"].assign(scenario_mode=mode))
        if mode == "forecast_only":
            hardware = res["hardware"]
    combined = {
        "predictions": pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        "memory": pd.concat(memory_frames, ignore_index=True) if memory_frames else pd.DataFrame(),
        "hardware": hardware,
        "n_participants": len({s.participant_id for s in streams}),
    }
    out = write_evaluation_bundle(
        combined,
        base_out,
        warmup_hours=eval_cfg.get("personalization_warmup_hours", [0, 6, 12, 24, 48]),
        config={**cfg, "runtime": {"device": device, "smoke": bool(args.smoke), "ckpt": str(ckpt_path)}},
    )
    print(f"[evaluate_stream_aireadi] wrote {out['n_rows']:,} rows -> {out['prediction_file']}")
    print(f"[evaluate_stream_aireadi] MAE={out['overall']['mae']:.3f} RMSE={out['overall']['rmse']:.3f}")


if __name__ == "__main__":
    main()
