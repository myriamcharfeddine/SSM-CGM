"""Trainer for the imposed cohort exercise sensitivity head.

The frozen model supplies forecast residuals. Only exercise_head.a_dir is updated.
Magnitude remains imposed from the registered prior and is not identified from
AI-READI.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch

from ..data.aireadi import AireadiParticipantStream
from ..models.aireadi_stream import AireadiStreamModel
from .aireadi_stream_trainer import (
    _batch_static_context,
    _make_stateful_batches,
    quantile_loss_mgdl,
    save_aireadi_checkpoint,
)


@dataclass
class ExerciseHeadTrainConfig:
    batch_size: int = 8
    chunk_steps: int = 72
    training_anchor_stride_steps: int = 3
    val_anchor_stride_steps: int = 3
    max_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_val: float = 1.0
    scenario_decomposition_weight: float = 1.0
    lambda_ex_prior: float = 0.2
    exercise_g_target: float = 0.30363636363636365
    patience: int = 20
    max_train_streams: int | None = None
    max_val_streams: int | None = None
    max_anchors_per_stream: int | None = None
    seed: int = 0


def _rank_info() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _anchor_tensors(model, batch, out, chunk_start: int = 0):
    horizon = model.feature_spec.horizon_steps
    anchors = [
        (batch_index, anchor)
        for batch_index in range(len(batch.anchors))
        for anchor in batch.anchors[batch_index]
        if anchor + horizon < batch.real_len[batch_index]
    ]
    if not anchors:
        return None
    device = out.device
    batch_index = torch.tensor(
        [item[0] for item in anchors], dtype=torch.long, device=device
    )
    position = torch.tensor(
        [item[1] for item in anchors], dtype=torch.long, device=device
    )
    relative = position - int(chunk_start)
    future = (
        position[:, None]
        + 1
        + torch.arange(horizon, device=device)[None, :]
    )
    return anchors, batch_index, position, relative, future


def _resting_hr_tensor(
    batch,
    batch_index: torch.Tensor,
    resting_hr_by_pid: Dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    values = []
    for index in batch_index.detach().cpu().tolist():
        participant_id = str(batch.stream_keys[index][0])
        if participant_id not in resting_hr_by_pid:
            raise KeyError(f"Missing stable HR floor for participant {participant_id}")
        values.append(float(resting_hr_by_pid[participant_id]))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _decode_batch(
    model: AireadiStreamModel,
    batch,
    static_context,
    out: torch.Tensor,
    resting_hr_by_pid: Dict[str, float],
):
    packed = _anchor_tensors(model, batch, out)
    if packed is None:
        return None
    _anchors, batch_index, position, relative, future = packed
    anchor_context = _batch_static_context(static_context, batch, batch_index)
    hidden = out[batch_index, relative]
    future_time = batch.time_features[batch_index[:, None], future]
    future_scenario = batch.scenario_values[batch_index[:, None], future]
    target = batch.target[batch_index[:, None], future]
    current = batch.target[batch_index, position]
    resting = _resting_hr_tensor(
        batch, batch_index, resting_hr_by_pid, out.device
    )

    no_action_mask = torch.zeros_like(
        batch.scenario_mask[batch_index[:, None], future]
    )
    no_action_hr_delta = torch.zeros(
        len(batch_index),
        model.feature_spec.horizon_steps,
        dtype=out.dtype,
        device=out.device,
    )
    forecast_raw = model.decode_horizon(
        hidden,
        anchor_context,
        future_time,
        future_scenario,
        no_action_mask,
        current_glucose_mgdl=current,
        future_hr_delta_bpm=no_action_hr_delta,
    )

    exercise_mask = torch.zeros_like(no_action_mask)
    hr_index = model.feature_spec.scenario_reals.index("heart_rate_mean")
    factual_mask = batch.scenario_mask[batch_index[:, None], future]
    exercise_mask[..., hr_index] = factual_mask[..., hr_index]
    scenario_raw = model.decode_horizon(
        hidden,
        anchor_context,
        future_time,
        future_scenario,
        exercise_mask,
        current_glucose_mgdl=current,
        resting_hr_bpm=resting,
    )

    current_column = current.view(-1, 1, 1)
    return (
        current_column + forecast_raw,
        current_column + scenario_raw,
        target,
    )


def _epoch_metrics(
    model: AireadiStreamModel,
    streams: Sequence[AireadiParticipantStream],
    config: ExerciseHeadTrainConfig,
    resting_hr_by_pid: Dict[str, float],
    device: torch.device,
    *,
    training: bool,
    optimizer=None,
) -> dict:
    model.eval()
    stride = (
        config.training_anchor_stride_steps
        if training
        else config.val_anchor_stride_steps
    )
    selected = list(
        streams[
            : config.max_train_streams
            if training
            else config.max_val_streams
        ]
    ) if (
        (training and config.max_train_streams)
        or ((not training) and config.max_val_streams)
    ) else list(streams)
    rng = random.Random(config.seed)
    batches = _make_stateful_batches(
        selected,
        config,
        device,
        horizon=model.feature_spec.horizon_steps,
        stride=stride,
        shuffle=training,
        rng=rng,
    )

    totals = {
        "forecast_pinball": 0.0,
        "scenario_decomposition_pinball": 0.0,
        "prior": 0.0,
        "total": 0.0,
        "forecast_abs_error": 0.0,
        "n_anchors": 0,
        "n_values": 0,
    }
    median_index = model.exercise_head.median_index

    for batch in batches:
        static_context = model.encode_static(batch.static_cat, batch.static_cont)
        state = model.init_stream(static_context)
        with torch.no_grad():
            state, out = model.scan_chunk(batch.dynamic, static_context, state)
        decoded = _decode_batch(
            model, batch, static_context, out.detach(), resting_hr_by_pid
        )
        if decoded is None:
            continue
        forecast_prediction, scenario_prediction, target = decoded
        forecast_loss = quantile_loss_mgdl(
            forecast_prediction, target, model.quantiles
        )
        scenario_loss = quantile_loss_mgdl(
            scenario_prediction, target, model.quantiles
        )
        sensitivity = model.exercise_head.sensitivity
        prior_loss = config.lambda_ex_prior * (
            sensitivity - config.exercise_g_target
        ).pow(2)
        total_loss = (
            forecast_loss
            + config.scenario_decomposition_weight * scenario_loss
            + prior_loss
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            rank, world_size = _rank_info()
            if world_size > 1 and torch.distributed.is_initialized():
                for parameter in model.exercise_head.parameters():
                    if parameter.grad is not None:
                        torch.distributed.all_reduce(
                            parameter.grad,
                            op=torch.distributed.ReduceOp.SUM,
                        )
                        parameter.grad.div_(float(world_size))
            torch.nn.utils.clip_grad_norm_(
                model.exercise_head.parameters(),
                config.gradient_clip_val,
            )
            optimizer.step()

        n_anchors = int(target.shape[0])
        n_values = int(target.numel())
        totals["forecast_pinball"] += float(forecast_loss.detach()) * n_anchors
        totals["scenario_decomposition_pinball"] += (
            float(scenario_loss.detach()) * n_anchors
        )
        totals["prior"] += float(prior_loss.detach()) * n_anchors
        totals["total"] += float(total_loss.detach()) * n_anchors
        totals["forecast_abs_error"] += float(
            (
                forecast_prediction[..., median_index]
                - target
            ).abs().sum().detach()
        )
        totals["n_anchors"] += n_anchors
        totals["n_values"] += n_values

    n_anchors = max(1, totals["n_anchors"])
    n_values = max(1, totals["n_values"])
    prefix = "train" if training else "val"
    return {
        f"{prefix}_forecast_pinball_mgdl": totals["forecast_pinball"] / n_anchors,
        f"{prefix}_scenario_decomposition_pinball_mgdl": (
            totals["scenario_decomposition_pinball"] / n_anchors
        ),
        f"{prefix}_prior_loss": totals["prior"] / n_anchors,
        f"{prefix}_total_loss": totals["total"] / n_anchors,
        f"{prefix}_forecast_only_mae_mgdl": (
            totals["forecast_abs_error"] / n_values
        ),
        f"n_{prefix}_anchors": totals["n_anchors"],
    }


def _write_rank_history(path: Path, rows: List[dict], rank: int) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    rank_path = path / f"training_history.rank{rank}.csv"
    if rows:
        with rank_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rank_path


def _merge_rank_histories(path: Path, world_size: int) -> Path:
    frames = []
    for rank in range(world_size):
        rank_path = path / f"training_history.rank{rank}.csv"
        if rank_path.exists():
            frame = pd.read_csv(rank_path)
            frame["rank"] = rank
            frames.append(frame)
    if not frames:
        raise RuntimeError("No rank-suffixed training histories were written")
    merged = pd.concat(frames, ignore_index=True)
    output = path / "training_history.csv"
    merged.to_csv(output, index=False)
    return output


def train_exercise_head(
    model: AireadiStreamModel,
    train_streams: Sequence[AireadiParticipantStream],
    val_streams: Sequence[AireadiParticipantStream],
    resting_hr_by_pid: Dict[str, float],
    config: ExerciseHeadTrainConfig,
    *,
    device: str,
    output_dir,
    resolved_config: dict,
) -> dict:
    rank, world_size = _rank_info()
    device_object = torch.device(device)
    model.to(device_object)
    trainable = model.configure_exercise_head_training()
    if trainable != ["exercise_head.a_dir"]:
        raise RuntimeError(f"Unexpected trainable parameters: {trainable}")

    optimizer = torch.optim.AdamW(
        model.exercise_head.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    output = Path(output_dir)
    metrics_dir = output / "metrics"
    checkpoint_dir = output / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: List[dict] = []
    best_value = float("inf")
    bad_epochs = 0
    for epoch in range(config.max_epochs):
        train_metrics = _epoch_metrics(
            model,
            train_streams,
            config,
            resting_hr_by_pid,
            device_object,
            training=True,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics = _epoch_metrics(
                model,
                val_streams,
                config,
                resting_hr_by_pid,
                device_object,
                training=False,
            )
        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "exercise_sensitivity": float(
                model.exercise_head.sensitivity.detach().cpu()
            ),
            "rank": rank,
        }
        history.append(row)
        print(
            f"[exercise-head][rank={rank}] epoch={epoch} "
            f"val_forecast_mae={row['val_forecast_only_mae_mgdl']:.4f} "
            f"val_scenario_pinball={row['val_scenario_decomposition_pinball_mgdl']:.4f} "
            f"S={row['exercise_sensitivity']:.6f}"
        )

        selection_value = row["val_scenario_decomposition_pinball_mgdl"]
        if rank == 0 and selection_value < best_value - 1e-6:
            best_value = selection_value
            bad_epochs = 0
            save_aireadi_checkpoint(
                model,
                checkpoint_dir / "best_model_checkpoint.pt",
                config=resolved_config,
                train_config=config,
                epoch=epoch,
                metrics=row,
                optimizer=optimizer,
            )
        elif rank == 0:
            bad_epochs += 1
        if world_size > 1 and torch.distributed.is_initialized():
            stop = torch.tensor(
                int(bad_epochs >= config.patience),
                device=device_object,
            )
            torch.distributed.broadcast(stop, src=0)
            should_stop = bool(stop.item())
        else:
            should_stop = bad_epochs >= config.patience
        if should_stop:
            break

    rank_history = _write_rank_history(metrics_dir, history, rank)
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
    merged_history = None
    if rank == 0:
        merged_history = _merge_rank_histories(metrics_dir, world_size)
        save_aireadi_checkpoint(
            model,
            checkpoint_dir / "final_model_checkpoint.pt",
            config=resolved_config,
            train_config=config,
            epoch=history[-1]["epoch"],
            metrics=history[-1],
            optimizer=optimizer,
        )
        summary = {
            "magnitude_status": "imposed, not estimated from AI-READI",
            "best_val_scenario_decomposition_pinball_mgdl": best_value,
            "final_exercise_sensitivity": history[-1]["exercise_sensitivity"],
            "rank_history": str(rank_history),
            "merged_history": str(merged_history),
            "train_config": asdict(config),
        }
        (metrics_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    return {
        "history": history,
        "rank_history": rank_history,
        "merged_history": merged_history,
    }
