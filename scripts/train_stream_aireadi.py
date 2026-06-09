#!/usr/bin/env python3
"""Train AI-READI-Stream-Full on participant+segment streams."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from ssmcgm.data.aireadi import (
    AireadiPreprocessor,
    build_stream_feature_spec,
    infer_or_validate_schema,
    load_aireadi_panel,
    make_aireadi_stream_splits,
    make_participant_streams,
    prepare_aireadi_panel,
    write_schema_mapping,
)
from ssmcgm.models.aireadi_stream import AireadiStreamModel, AireadiStreamModelConfig
from ssmcgm.training.aireadi_stream_trainer import AireadiTrainConfig, train_stateful_stream

TRAIN_MODES = ("windowed_debug", "stateful_stream")


def parse_args():
    ap = argparse.ArgumentParser(description="Train AI-READI stream model")
    ap.add_argument("--config", default=str(ROOT / "configs/aireadi_stream_full.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-participants", type=int, default=None)
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint to continue from; --epochs means additional epochs")
    ap.add_argument("--train-mode", choices=TRAIN_MODES, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--context-steps", type=int, default=None,
                    help="windowed_debug context length in 5-minute steps; default comes from config")
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


def require_requested_backend(model_cfg: dict, device: str) -> None:
    scan_mode = str(model_cfg.get("scan_mode", "chunked"))
    if device == "cuda" and scan_mode == "mamba":
        try:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on CUDA extension install
            raise RuntimeError(
                "model.scan_mode='mamba' with --device cuda requires "
                "mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined. "
                "Install/fix mamba-ssm or set model.scan_mode='chunked' for fallback."
            ) from exc


def _steps(minutes: int, bin_minutes: int) -> int:
    return max(1, round(float(minutes) / float(bin_minutes)))


def load_resume_state(resume_path, out_dir: Path):
    if not resume_path:
        return None, 0, [], None, None, None
    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_epoch = int(ckpt.get("epoch", -1))
    summary_path = out_dir / "metrics" / "training_summary.json"
    previous_history = []
    previous_best = None
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)
        previous_history = list(summary.get("history", []))
        previous_best = summary.get("best_val_pinball_mgdl")
    elif ckpt.get("metrics"):
        previous_history = [dict(ckpt["metrics"])]
        previous_best = ckpt["metrics"].get("val_pinball_mgdl")
    if previous_history:
        start_epoch = max(int(row.get("epoch", ckpt_epoch)) for row in previous_history) + 1
    else:
        start_epoch = ckpt_epoch + 1
    return (
        ckpt,
        start_epoch,
        previous_history,
        previous_best,
        ckpt.get("optimizer_state_dict"),
        ckpt.get("scaler_state_dict"),
    )


def build_data(cfg: dict, *, smoke: bool, max_participants=None):
    dc = cfg["data"]
    smoke_participants = (
        max_participants
        or (cfg.get("smoke", {}).get("participants") if smoke else None)
    )
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
    spec = build_stream_feature_spec(
        df,
        schema,
        horizon_steps=cfg["dataset"].get("horizon_steps", 12),
        bin_minutes=cfg["dataset"].get("bin_minutes", 5),
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
    pre = AireadiPreprocessor.fit(df, spec, split.participants("train"))
    min_steps = spec.horizon_steps + 2
    train_streams = make_participant_streams(
        df, split, schema, feature_spec=spec, preprocessor=pre,
        splits=["train"], min_steps=min_steps,
        max_participants=(cfg.get("smoke", {}).get("train_participants") if smoke else None),
    )
    val_streams = make_participant_streams(
        df, split, schema, feature_spec=spec, preprocessor=pre,
        splits=["validation"], min_steps=min_steps,
        max_participants=(cfg.get("smoke", {}).get("val_participants") if smoke else None),
    )
    return df, schema, spec, split, pre, train_streams, val_streams


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    train_cfg_raw = dict(cfg["training"])
    train_mode = args.train_mode or train_cfg_raw.get("train_mode", "stateful_stream")
    if train_mode not in TRAIN_MODES:
        raise ValueError(f"unknown train mode {train_mode!r}; expected one of {TRAIN_MODES}")
    require_requested_backend(cfg.get("model", {}), device)

    out_dir = Path(args.output_dir or cfg["output"]["base_dir"])
    for sub in ["checkpoints", "predictions", "metrics", "figures", "hardware"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    resume_ckpt, resume_start_epoch, resume_history, resume_best_val, resume_optimizer_state, resume_scaler_state = load_resume_state(
        args.resume_from, out_dir
    )

    df, schema, spec, split, pre, train_streams, val_streams = build_data(
        cfg, smoke=args.smoke, max_participants=args.max_participants
    )
    if not train_streams:
        raise RuntimeError("No AI-READI train streams were built. Check split/data paths.")
    if not val_streams:
        raise RuntimeError("No AI-READI validation streams were built. Check split/data paths.")

    model_cfg = AireadiStreamModelConfig(**cfg["model"])
    model = AireadiStreamModel(spec, pre, model_cfg)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        print(
            f"[train_stream_aireadi] resume: loaded {args.resume_from} "
            f"from checkpoint epoch={resume_ckpt.get('epoch', -1)}; next_epoch={resume_start_epoch}"
        )
        if resume_optimizer_state is None:
            print("[train_stream_aireadi] resume: optimizer state not found; optimizer will restart.")
    bin_minutes = spec.bin_minutes
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif train_mode == "windowed_debug":
        batch_size = train_cfg_raw.get("windowed_debug_batch_size", 32)
    else:
        batch_size = train_cfg_raw.get("batch_size", 8)
    context_steps = args.context_steps or train_cfg_raw.get("windowed_context_steps", 288)
    train_cfg = AireadiTrainConfig(
        train_mode=train_mode,
        chunk_steps=round(train_cfg_raw.get("chunk_length_hours", 6) * 60 / bin_minutes),
        detach_state_every_chunk=train_cfg_raw.get("detach_state_every_chunk", True),
        training_anchor_stride_steps=_steps(train_cfg_raw.get("training_anchor_stride_minutes", 15), bin_minutes),
        val_anchor_stride_steps=_steps(train_cfg_raw.get("val_anchor_stride_minutes", 15), bin_minutes),
        batch_size=int(batch_size),
        windowed_context_steps=int(context_steps),
        max_epochs=args.epochs or (cfg.get("smoke", {}).get("epochs", 1) if args.smoke else train_cfg_raw.get("max_epochs", 20)),
        lr=train_cfg_raw.get("lr", 1e-3),
        weight_decay=train_cfg_raw.get("weight_decay", 1e-4),
        gradient_clip_val=train_cfg_raw.get("gradient_clip_val", 1.0),
        mixed_precision=bool(train_cfg_raw.get("mixed_precision", True) and device == "cuda"),
        patience=train_cfg_raw.get("patience", 5),
        max_train_streams=(cfg.get("smoke", {}).get("max_train_streams") if args.smoke else None),
        max_val_streams=(cfg.get("smoke", {}).get("max_val_streams") if args.smoke else None),
        max_anchors_per_stream=(cfg.get("smoke", {}).get("max_anchors_per_stream") if args.smoke else None),
        seed=cfg.get("split", {}).get("seed", 42),
    )
    resolved = copy.deepcopy(cfg)
    resolved.setdefault("training", {})["train_mode"] = train_cfg.train_mode
    resolved["training"]["batch_size"] = train_cfg.batch_size
    resolved["training"]["windowed_context_steps"] = train_cfg.windowed_context_steps
    resolved["runtime"] = {
        "device": device,
        "smoke": bool(args.smoke),
        "train_mode": train_cfg.train_mode,
        "batch_size": train_cfg.batch_size,
        "windowed_context_steps": train_cfg.windowed_context_steps,
        "scan_mode": model_cfg.scan_mode,
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "initial_epoch": int(resume_start_epoch),
        "additional_epochs": int(train_cfg.max_epochs),
        "n_train_streams": len(train_streams),
        "n_val_streams": len(val_streams),
        "n_panel_rows_after_cleaning": int(len(df)),
        "split_source": split.source,
    }
    write_schema_mapping(schema, spec, out_dir / "schema_mapping.json")
    with (out_dir / "preprocessor.json").open("w") as f:
        json.dump(pre.to_jsonable(), f, indent=2)
    with (out_dir / "config_resolved.yaml").open("w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    print(f"[train_stream_aireadi] model params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")
    print(
        f"[train_stream_aireadi] streams train={len(train_streams)} validation={len(val_streams)} "
        f"device={device} mode={train_cfg.train_mode} batch_size={train_cfg.batch_size} scan={model_cfg.scan_mode}"
    )
    print("[train_stream_aireadi] output:", out_dir)
    train_stateful_stream(
        model, train_streams, val_streams, train_cfg,
        device=device, output_dir=out_dir, resolved_config=resolved,
        initial_epoch=resume_start_epoch, previous_history=resume_history,
        previous_best_val=resume_best_val,
        resume_optimizer_state=resume_optimizer_state, resume_scaler_state=resume_scaler_state,
    )


if __name__ == "__main__":
    main()
