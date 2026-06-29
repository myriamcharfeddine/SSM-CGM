#!/usr/bin/env python3
"""Prepare or train the Path B imposed exercise sensitivity head.

The magnitude is imposed from the registered descriptive-in-cohort prior.
AI-READI supplies timing, shape, and glucose-state gating. The output is a
planning response, not a causal estimate.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssmcgm.data.aireadi import (
    AireadiFeatureSpec,
    AireadiPreprocessor,
    infer_or_validate_schema,
    load_aireadi_panel,
    make_aireadi_stream_splits,
    make_participant_streams,
    prepare_aireadi_panel,
)
from ssmcgm.models.aireadi_stream import (
    AireadiStreamModel,
    AireadiStreamModelConfig,
)
from ssmcgm.training.exercise_head_trainer import (
    ExerciseHeadTrainConfig,
    train_exercise_head,
)


DEFAULT_CONFIG = ROOT / "configs/study2_exercise_path_b.yaml"
PYTHON = "/home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python"
HR_BASELINE_PERCENTILE = 0.05
MIN_BASELINE_ROWS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate configuration and checkpoint wiring without loading panel data or training.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def initialize_distributed(device: str) -> tuple[int, int, str]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if device == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend)
    if device == "cuda" and world_size > 1:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    return rank, world_size, device


def stable_hr_floors(panel: pd.DataFrame) -> dict[str, float]:
    work = panel.copy()
    sleep_columns = [
        column
        for column in (
            "sleep_stage_deep",
            "sleep_stage_light",
            "sleep_stage_rem",
            "sleep_stage_awake",
            "sleep_stage_unknown",
        )
        if column in work
    ]
    sleep_sum = work[sleep_columns].fillna(0).sum(axis=1)
    awake_value = (
        work["sleep_stage_awake"].fillna(0)
        if "sleep_stage_awake" in work
        else pd.Series(0.0, index=work.index)
    )
    asleep = (sleep_sum > 0) & (awake_value < 0.5)
    work["_awake_for_exercise"] = ~asleep

    valid_awake = work[
        work["_awake_for_exercise"]
        & work["heart_rate_mean"].notna()
    ]
    cohort_floor = float(
        valid_awake["heart_rate_mean"].quantile(HR_BASELINE_PERCENTILE)
    )
    floors: dict[str, float] = {}
    for participant_id, group in work.groupby("participant_id", sort=False):
        awake = group[group["_awake_for_exercise"]]
        if "activity_stage_sedentary" in awake:
            selected = awake[
                awake["activity_stage_sedentary"].fillna(0) > 0.5
            ]
        else:
            selected = awake.iloc[:0]
        if len(selected) < MIN_BASELINE_ROWS:
            selected = awake[
                awake["activity_steps_per_min"].fillna(999) <= 5
            ]
        if len(selected) < MIN_BASELINE_ROWS:
            selected = awake
        hr_values = selected["heart_rate_mean"].dropna()
        floor = (
            float(hr_values.quantile(HR_BASELINE_PERCENTILE))
            if len(hr_values) >= 5
            else cohort_floor
        )
        floors[str(participant_id)] = floor
    return floors


def build_model(config: dict, device: str):
    checkpoint_path = resolve_path(config["base_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint["metadata"]
    feature_spec = AireadiFeatureSpec(**metadata["feature_spec"])
    preprocessor = AireadiPreprocessor.from_jsonable(metadata["preprocessor"])
    base_config = AireadiStreamModelConfig(**metadata["model_config"])
    head = config["exercise_head"]
    model_config = replace(
        base_config,
        hr_exercise=bool(head["enabled"]),
        hr_exercise_gain_target=float(head["exercise_g_target"]),
        hr_exercise_deadzone_bpm=float(head["hr_deadzone_bpm"]),
        hr_exercise_lag_support_min=int(head["hr_lag_support_min"]),
        hr_exercise_g_floor_mgdl=float(head["g_floor_mgdl"]),
        hr_exercise_bout_median_min=int(head["bout_median_min"]),
        hr_exercise_rise_to_peak_min=int(head["rise_to_peak_min"]),
        hr_exercise_decay_min=int(head["decay_min"]),
        route_future_hr_via_exercise_head=bool(
            head["route_future_hr_via_head"]
        ),
    )
    model = AireadiStreamModel(feature_spec, preprocessor, model_config)
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if sorted(incompatible.missing_keys) != ["exercise_head.a_dir"]:
        raise RuntimeError(
            f"Unexpected missing checkpoint keys: {incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {incompatible.unexpected_keys}"
        )
    trainable = model.configure_exercise_head_training()
    if trainable != ["exercise_head.a_dir"]:
        raise RuntimeError(f"Unexpected trainable parameters: {trainable}")
    return model, feature_spec, preprocessor, checkpoint_path


def build_data(config: dict, feature_spec, preprocessor):
    data_config = config["data"]
    panel = load_aireadi_panel(
        data_config["panel_path"],
        static_path=data_config.get("static_path"),
        cohort_path=data_config.get("cohort_path"),
    )
    panel["participant_id"] = panel["participant_id"].astype(str)
    floors = stable_hr_floors(panel)
    schema = infer_or_validate_schema(panel, data_config.get("schema"))
    prepared = prepare_aireadi_panel(
        panel,
        schema,
        bin_minutes=int(config["dataset"]["bin_minutes"]),
        clean_min_segment_hours=float(
            config["dataset"]["clean_min_segment_hours"]
        ),
    )
    split_config = config["split"]
    split = make_aireadi_stream_splits(
        prepared,
        split_mode=split_config["mode"],
        train=float(split_config["train"]),
        val=float(split_config["val"]),
        test=float(split_config["test"]),
        seed=int(split_config["seed"]),
        stratify_col=split_config.get("stratify_col"),
        existing_split_path=split_config.get("existing_split_path"),
    )
    minimum_steps = feature_spec.horizon_steps + 2
    train_streams = make_participant_streams(
        prepared,
        split,
        schema,
        feature_spec=feature_spec,
        preprocessor=preprocessor,
        splits=["train"],
        min_steps=minimum_steps,
    )
    val_streams = make_participant_streams(
        prepared,
        split,
        schema,
        feature_spec=feature_spec,
        preprocessor=preprocessor,
        splits=["validation"],
        min_steps=minimum_steps,
    )
    return prepared, split, floors, train_streams, val_streams


def build_train_config(config: dict, smoke: bool) -> ExerciseHeadTrainConfig:
    training = config["training"]
    smoke_config = config.get("smoke", {})
    bin_minutes = int(config["dataset"]["bin_minutes"])
    return ExerciseHeadTrainConfig(
        batch_size=int(training["batch_size"]),
        chunk_steps=round(
            float(training["chunk_length_hours"]) * 60 / bin_minutes
        ),
        training_anchor_stride_steps=round(
            float(training["training_anchor_stride_minutes"]) / bin_minutes
        ),
        val_anchor_stride_steps=round(
            float(training["val_anchor_stride_minutes"]) / bin_minutes
        ),
        max_epochs=int(
            smoke_config["max_epochs"]
            if smoke
            else training["max_epochs"]
        ),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_val=float(training["gradient_clip_val"]),
        scenario_decomposition_weight=float(
            training["scenario_decomposition_weight"]
        ),
        lambda_ex_prior=float(training["lambda_ex_prior"]),
        exercise_g_target=float(
            config["exercise_head"]["exercise_g_target"]
        ),
        patience=int(training["patience"]),
        max_train_streams=(
            int(smoke_config["max_train_streams"]) if smoke else None
        ),
        max_val_streams=(
            int(smoke_config["max_val_streams"]) if smoke else None
        ),
        max_anchors_per_stream=(
            int(smoke_config["max_anchors_per_stream"])
            if smoke
            else None
        ),
        seed=int(training["seed"]),
    )


def print_launch(config_path: Path) -> None:
    command = (
        f"cd {ROOT}\n"
        f"{PYTHON} scripts/train_study2_exercise_path_b.py "
        f"--config {config_path} --device cuda"
    )
    print("Launch command:")
    print(command)
    print()
    print("PAUSE: confirm training launch before consuming compute.")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    step0_manifest = json.loads(
        resolve_path(config["step0_manifest"]).read_text()
    )
    required_caveats = (
        "anchor_validation_role",
        "r_reference_assumption",
        "s_time_reference",
    )
    missing_caveats = [
        key for key in required_caveats if key not in step0_manifest
    ]
    if missing_caveats:
        raise RuntimeError(
            f"Step 0 manifest lacks required caveats: {missing_caveats}"
        )

    requested_device = resolve_device(args.device)
    rank, world_size, device = initialize_distributed(requested_device)
    model, feature_spec, preprocessor, checkpoint_path = build_model(
        config, device
    )
    output_dir = resolve_path(
        args.output_dir or config["output"]["base_dir"]
    )

    print("=" * 72)
    print("PATH B EXERCISE HEAD TRAINING PREPARATION")
    print("=" * 72)
    print("Magnitude status: IMPOSED, not estimated from AI-READI.")
    print("AI-READI contribution: timing, shape, and glucose-state gating.")
    print("Output role: structurally imposed planning response, not a causal estimate.")
    print(f"Frozen base checkpoint: {checkpoint_path}")
    print("Trainable parameters: ['exercise_head.a_dir']")
    print("Future inputs: HR only; all other future scenario masks are zero.")
    print(f"Rank-safe histories: training_history.rank{{rank}}.csv, world_size={world_size}")
    print("Rank 0 merges rank histories after all ranks finish.")
    print(f"Forecast-only validation MAE will be logged every epoch.")
    print(f"Output directory: {output_dir}")

    if args.prepare_only:
        print_launch(config_path)
        return

    prepared, split, floors, train_streams, val_streams = build_data(
        config, feature_spec, preprocessor
    )
    if not train_streams or not val_streams:
        raise RuntimeError("Training or validation streams are empty")
    train_config = build_train_config(config, args.smoke)
    resolved = copy.deepcopy(config)
    resolved["runtime"] = {
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "smoke": bool(args.smoke),
        "n_prepared_rows": len(prepared),
        "n_train_streams": len(train_streams),
        "n_val_streams": len(val_streams),
        "split_source": split.source,
        "base_checkpoint": str(checkpoint_path),
    }
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config_resolved.yaml").open("w") as handle:
            yaml.safe_dump(resolved, handle, sort_keys=False)

    train_exercise_head(
        model,
        train_streams,
        val_streams,
        floors,
        train_config,
        device=device,
        output_dir=output_dir,
        resolved_config=resolved,
    )


if __name__ == "__main__":
    main()
