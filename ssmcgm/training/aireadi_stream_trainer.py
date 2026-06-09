"""Training loops for AI-READI stream and windowed-debug modes."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..data.aireadi import AireadiParticipantStream
from ..models.aireadi_stream import AireadiStreamModel
from ..stream.state import StaticContext, StreamState


@dataclass
class AireadiTrainConfig:
    train_mode: str = "stateful_stream"
    chunk_steps: int = 72
    detach_state_every_chunk: bool = True
    training_anchor_stride_steps: int = 3
    val_anchor_stride_steps: int = 3
    batch_size: int = 8
    windowed_context_steps: int = 288
    max_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    mixed_precision: bool = True
    patience: int = 5
    max_train_streams: Optional[int] = None
    max_val_streams: Optional[int] = None
    max_anchors_per_stream: Optional[int] = None
    log_every: int = 25
    seed: int = 42


@dataclass
class _AireadiStreamBatch:
    dynamic: torch.Tensor
    time_features: torch.Tensor
    scenario_values: torch.Tensor
    scenario_mask: torch.Tensor
    target: torch.Tensor
    observed: torch.Tensor
    static_cont: torch.Tensor
    static_cat: torch.Tensor
    real_len: List[int]
    anchors: List[List[int]] = field(default_factory=list)
    stream_keys: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class _AireadiWindowBatch:
    dynamic_context: torch.Tensor
    future_time_features: torch.Tensor
    future_scenario_values: torch.Tensor
    future_scenario_mask: torch.Tensor
    target_current: torch.Tensor
    target_future: torch.Tensor
    static_cont: torch.Tensor
    static_cat: torch.Tensor
    stream_keys: List[Tuple[str, int]] = field(default_factory=list)
    anchors: List[int] = field(default_factory=list)


def quantile_loss_mgdl(pred_mgdl: torch.Tensor, target_mgdl: torch.Tensor,
                       quantiles: Sequence[float]) -> torch.Tensor:
    """Mean pinball loss over ``(anchors, horizon, quantiles)``."""
    target = target_mgdl.unsqueeze(-1)
    losses = []
    for qi, q in enumerate(quantiles):
        err = target[..., 0] - pred_mgdl[..., qi]
        losses.append(torch.maximum((q - 1.0) * err, q * err))
    return torch.stack(losses, dim=-1).mean()


def _valid_anchors(stream: AireadiParticipantStream, horizon: int, stride: int,
                   max_anchors: Optional[int] = None) -> List[int]:
    anchors: List[int] = []
    last = -10 ** 9
    obs = stream.observed
    for t in range(0, stream.n_steps - horizon):
        if (t - last) < stride:
            continue
        if bool(obs[t]) and bool(obs[t + 1:t + 1 + horizon].all()):
            anchors.append(t)
            last = t
            if max_anchors is not None and len(anchors) >= max_anchors:
                break
    return anchors


def _detach_state(state: StreamState) -> StreamState:
    return StreamState(
        layer_states=[s.detach() for s in state.layer_states],
        conv_states=None if state.conv_states is None else [None if c is None else c.detach() for c in state.conv_states],
        last_output=None if state.last_output is None else state.last_output.detach(),
        static_context=None,
        step=state.step,
    )


def _batch_static_context(sctx: StaticContext, batch: _AireadiStreamBatch,
                          b_idx: torch.Tensor) -> StaticContext:
    return StaticContext(
        embedding=sctx.embedding[b_idx],
        raw_static_cat=batch.static_cat[b_idx],
        raw_static_cont=batch.static_cont[b_idx],
    )


def _decode_batch_anchor_loss(model: AireadiStreamModel, batch: _AireadiStreamBatch,
                              sctx: StaticContext, out: torch.Tensor,
                              chunk_start: int, anchors_bt: List[Tuple[int, int]]) -> torch.Tensor:
    H = model.feature_spec.horizon_steps
    device = out.device
    b_idx = torch.tensor([b for b, _ in anchors_bt], dtype=torch.long, device=device)
    pos = torch.tensor([t for _, t in anchors_bt], dtype=torch.long, device=device)
    rel = pos - int(chunk_start)
    fut = pos[:, None] + 1 + torch.arange(H, device=device)[None, :]
    h_t = out[b_idx, rel]
    anchor_sctx = _batch_static_context(sctx, batch, b_idx)
    future_time = batch.time_features[b_idx[:, None], fut]
    future_scenario = batch.scenario_values[b_idx[:, None], fut]
    future_mask = torch.zeros_like(batch.scenario_mask[b_idx[:, None], fut])
    raw = model.decode_horizon(h_t, anchor_sctx, future_time, future_scenario, future_mask)
    current = batch.target[b_idx, pos]
    pred = current.view(-1, 1, 1) + raw
    target = batch.target[b_idx[:, None], fut]
    return quantile_loss_mgdl(pred, target, model.quantiles)


def _make_stateful_batches(streams: Sequence[AireadiParticipantStream], cfg: AireadiTrainConfig,
                           device: torch.device, *, horizon: int, stride: int,
                           shuffle: bool, rng: random.Random) -> List[_AireadiStreamBatch]:
    if not streams:
        return []
    batch_size = max(1, int(cfg.batch_size))
    order = sorted(range(len(streams)), key=lambda i: streams[i].n_steps)
    batches: List[_AireadiStreamBatch] = []
    for start in range(0, len(order), batch_size):
        group = [streams[i] for i in order[start:start + batch_size]]
        L = max(s.n_steps for s in group)
        B = len(group)
        n_dyn = int(group[0].dynamic.shape[1])
        n_time = int(group[0].time_features.shape[1])
        n_scenario = int(group[0].scenario_values.shape[1])
        dynamic = torch.zeros(B, L, n_dyn, dtype=torch.float32)
        time_features = torch.zeros(B, L, n_time, dtype=torch.float32)
        scenario_values = torch.zeros(B, L, n_scenario, dtype=torch.float32)
        scenario_mask = torch.zeros(B, L, n_scenario, dtype=torch.float32)
        target = torch.full((B, L), float("nan"), dtype=torch.float32)
        observed = torch.zeros(B, L, dtype=torch.bool)
        static_cont = torch.stack([s.static_cont.float() for s in group], dim=0)
        static_cat = torch.stack([s.static_cat.long() for s in group], dim=0)
        real_len: List[int] = []
        anchors: List[List[int]] = []
        stream_keys: List[Tuple[str, int]] = []
        for b, s in enumerate(group):
            n = s.n_steps
            real_len.append(n)
            stream_keys.append((s.participant_id, int(s.segment_id)))
            dynamic[b, :n] = s.dynamic.float()
            time_features[b, :n] = s.time_features.float()
            scenario_values[b, :n] = s.scenario_values.float()
            scenario_mask[b, :n] = s.scenario_mask.float()
            target[b, :n] = s.target.float()
            observed[b, :n] = s.observed.bool()
            if n < L:
                dynamic[b, n:] = s.dynamic[-1].float()
                time_features[b, n:] = s.time_features[-1].float()
                scenario_values[b, n:] = s.scenario_values[-1].float()
                scenario_mask[b, n:] = 0.0
            anchors.append(_valid_anchors(s, horizon, stride, cfg.max_anchors_per_stream))
        batches.append(_AireadiStreamBatch(
            dynamic=dynamic.to(device),
            time_features=time_features.to(device),
            scenario_values=scenario_values.to(device),
            scenario_mask=scenario_mask.to(device),
            target=target.to(device),
            observed=observed.to(device),
            static_cont=static_cont.to(device),
            static_cat=static_cat.to(device),
            real_len=real_len,
            anchors=anchors,
            stream_keys=stream_keys,
        ))
    if shuffle:
        rng.shuffle(batches)
    return batches


def _run_stateful_batch(model: AireadiStreamModel, batch: _AireadiStreamBatch,
                        cfg: AireadiTrainConfig, optimizer, scaler,
                        device: torch.device) -> Tuple[float, int]:
    use_amp = bool(cfg.mixed_precision and device.type == "cuda")
    B, L = int(batch.dynamic.shape[0]), int(batch.dynamic.shape[1])
    sctx0 = model.encode_static(batch.static_cat, batch.static_cont)
    state = model.init_stream(sctx0)
    total_loss, total_terms = 0.0, 0
    for start in range(0, L, cfg.chunk_steps):
        end = min(start + cfg.chunk_steps, L)
        anchors_bt = [
            (b, t)
            for b in range(B)
            for t in batch.anchors[b]
            if start <= t < end and t + model.feature_spec.horizon_steps < batch.real_len[b]
        ]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            sctx = model.encode_static(batch.static_cat, batch.static_cont)
            state, out = model.scan_chunk(batch.dynamic[:, start:end], sctx, state)
            loss = _decode_batch_anchor_loss(model, batch, sctx, out, start, anchors_bt) if anchors_bt else None
        if loss is not None:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
                optimizer.step()
            total_loss += float(loss.detach()) * len(anchors_bt)
            total_terms += len(anchors_bt)
        if cfg.detach_state_every_chunk:
            state = _detach_state(state)
    return total_loss / max(total_terms, 1), total_terms


def _stream_train_epoch(model: AireadiStreamModel, streams: Sequence[AireadiParticipantStream],
                        cfg: AireadiTrainConfig, optimizer, scaler,
                        device: torch.device, rng: random.Random) -> Dict[str, float]:
    model.train()
    batches = _make_stateful_batches(
        streams, cfg, device, horizon=model.feature_spec.horizon_steps,
        stride=cfg.training_anchor_stride_steps, shuffle=True, rng=rng
    )
    total_loss, total_terms = 0.0, 0
    for bi, batch in enumerate(batches):
        loss, n_terms = _run_stateful_batch(model, batch, cfg, optimizer, scaler, device)
        total_loss += loss * n_terms
        total_terms += n_terms
        if cfg.log_every and (bi + 1) % cfg.log_every == 0:
            print(f"[aireadi-train] batch {bi + 1}/{len(batches)} loss={total_loss / max(total_terms, 1):.3f}")
    return {"train_pinball_mgdl": total_loss / max(total_terms, 1), "n_train_anchors": total_terms}


@torch.no_grad()
def validate_stateful_stream(model: AireadiStreamModel, streams: Sequence[AireadiParticipantStream],
                             cfg: AireadiTrainConfig, device: torch.device) -> Dict[str, float]:
    model.eval()
    streams = list(streams[:cfg.max_val_streams] if cfg.max_val_streams else streams)
    rng = random.Random(cfg.seed)
    batches = _make_stateful_batches(
        streams, cfg, device, horizon=model.feature_spec.horizon_steps,
        stride=cfg.val_anchor_stride_steps, shuffle=False, rng=rng
    )
    total_loss, total_terms = 0.0, 0
    for batch in batches:
        anchors_bt = [(b, t) for b in range(len(batch.anchors)) for t in batch.anchors[b]]
        if not anchors_bt:
            continue
        sctx = model.encode_static(batch.static_cat, batch.static_cont)
        state = model.init_stream(sctx)
        state, out = model.scan_chunk(batch.dynamic, sctx, state)
        loss = _decode_batch_anchor_loss(model, batch, sctx, out, 0, anchors_bt)
        total_loss += float(loss.detach()) * len(anchors_bt)
        total_terms += len(anchors_bt)
    return {"val_pinball_mgdl": total_loss / max(total_terms, 1), "n_val_anchors": total_terms}


def _valid_window_anchors(stream: AireadiParticipantStream, horizon: int, stride: int,
                          context_steps: int, max_anchors: Optional[int]) -> List[int]:
    anchors = [
        t for t in _valid_anchors(stream, horizon, stride, None)
        if t + 1 >= context_steps and t + horizon < stream.n_steps
    ]
    if max_anchors is not None:
        anchors = anchors[:max_anchors]
    return anchors


def _make_window_index(streams: Sequence[AireadiParticipantStream], cfg: AireadiTrainConfig,
                       *, stride: int, horizon: int) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    for si, stream in enumerate(streams):
        for anchor in _valid_window_anchors(stream, horizon, stride, cfg.windowed_context_steps, cfg.max_anchors_per_stream):
            start = anchor + 1 - cfg.windowed_context_steps
            if start < 0 or anchor + horizon >= stream.n_steps:
                raise AssertionError("windowed_debug generated a window crossing stream bounds")
            windows.append((si, anchor))
    return windows


def _make_window_batch(streams: Sequence[AireadiParticipantStream], items: Sequence[Tuple[int, int]],
                       cfg: AireadiTrainConfig, device: torch.device, *, horizon: int) -> _AireadiWindowBatch:
    C = int(cfg.windowed_context_steps)
    dynamic, future_time, future_scenario, future_mask = [], [], [], []
    current, target_future, static_cont, static_cat = [], [], [], []
    keys: List[Tuple[str, int]] = []
    anchors: List[int] = []
    for si, anchor in items:
        s = streams[si]
        start = anchor + 1 - C
        fut = slice(anchor + 1, anchor + 1 + horizon)
        if start < 0:
            raise AssertionError("windowed_debug context starts before segment")
        if anchor + horizon >= s.n_steps:
            raise AssertionError("windowed_debug future crosses segment")
        context = s.dynamic[start:anchor + 1].float()
        if int(context.shape[0]) != C:
            raise AssertionError("windowed_debug context length mismatch")
        dynamic.append(context)
        future_time.append(s.time_features[fut].float())
        future_scenario.append(s.scenario_values[fut].float())
        future_mask.append(s.scenario_mask[fut].float())
        target_future.append(s.target[fut].float())
        static_cont.append(s.static_cont.float())
        static_cat.append(s.static_cat.long())
        current.append(s.target[anchor].float())
        keys.append((s.participant_id, int(s.segment_id)))
        anchors.append(int(anchor))
    return _AireadiWindowBatch(
        dynamic_context=torch.stack(dynamic, dim=0).to(device),
        future_time_features=torch.stack(future_time, dim=0).to(device),
        future_scenario_values=torch.stack(future_scenario, dim=0).to(device),
        future_scenario_mask=torch.stack(future_mask, dim=0).to(device),
        target_current=torch.stack(current, dim=0).to(device),
        target_future=torch.stack(target_future, dim=0).to(device),
        static_cont=torch.stack(static_cont, dim=0).to(device),
        static_cat=torch.stack(static_cat, dim=0).to(device),
        stream_keys=keys,
        anchors=anchors,
    )


def _iter_window_batches(streams: Sequence[AireadiParticipantStream], windows: List[Tuple[int, int]],
                         cfg: AireadiTrainConfig, device: torch.device, *, horizon: int,
                         shuffle: bool, rng: random.Random):
    order = list(range(len(windows)))
    if shuffle:
        rng.shuffle(order)
    batch_size = max(1, int(cfg.batch_size))
    for start in range(0, len(order), batch_size):
        items = [windows[i] for i in order[start:start + batch_size]]
        yield _make_window_batch(streams, items, cfg, device, horizon=horizon)


def _windowed_batch_loss(model: AireadiStreamModel, batch: _AireadiWindowBatch) -> torch.Tensor:
    sctx = model.encode_static(batch.static_cat, batch.static_cont)
    state = model.init_stream(sctx)
    state, out = model.scan_chunk(batch.dynamic_context, sctx, state)
    raw = model.decode_horizon(
        out[:, -1],
        sctx,
        batch.future_time_features,
        batch.future_scenario_values,
        torch.zeros_like(batch.future_scenario_mask),
    )
    pred = batch.target_current.view(-1, 1, 1) + raw
    return quantile_loss_mgdl(pred, batch.target_future, model.quantiles)


def _windowed_train_epoch(model: AireadiStreamModel, streams: Sequence[AireadiParticipantStream],
                          windows: List[Tuple[int, int]], cfg: AireadiTrainConfig,
                          optimizer, scaler, device: torch.device,
                          rng: random.Random) -> Dict[str, float]:
    model.train()
    use_amp = bool(cfg.mixed_precision and device.type == "cuda")
    total_loss, total_terms = 0.0, 0
    n_batches = math.ceil(len(windows) / max(1, cfg.batch_size)) if windows else 0
    for bi, batch in enumerate(_iter_window_batches(
        streams, windows, cfg, device, horizon=model.feature_spec.horizon_steps, shuffle=True, rng=rng
    )):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            loss = _windowed_batch_loss(model, batch)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
            optimizer.step()
        n_terms = int(batch.dynamic_context.shape[0])
        total_loss += float(loss.detach()) * n_terms
        total_terms += n_terms
        if cfg.log_every and (bi + 1) % cfg.log_every == 0:
            print(f"[aireadi-train] window batch {bi + 1}/{n_batches} loss={total_loss / max(total_terms, 1):.3f}")
    return {"train_pinball_mgdl": total_loss / max(total_terms, 1), "n_train_anchors": total_terms}


@torch.no_grad()
def validate_windowed_debug(model: AireadiStreamModel, streams: Sequence[AireadiParticipantStream],
                            windows: List[Tuple[int, int]], cfg: AireadiTrainConfig,
                            device: torch.device) -> Dict[str, float]:
    model.eval()
    rng = random.Random(cfg.seed)
    total_loss, total_terms = 0.0, 0
    for batch in _iter_window_batches(
        streams, windows, cfg, device, horizon=model.feature_spec.horizon_steps, shuffle=False, rng=rng
    ):
        loss = _windowed_batch_loss(model, batch)
        n_terms = int(batch.dynamic_context.shape[0])
        total_loss += float(loss.detach()) * n_terms
        total_terms += n_terms
    return {"val_pinball_mgdl": total_loss / max(total_terms, 1), "n_val_anchors": total_terms}


def _write_history(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_aireadi_checkpoint(model: AireadiStreamModel, path, *, config: dict,
                            train_config: AireadiTrainConfig, epoch: int, metrics: dict,
                            optimizer=None, scaler=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": model.checkpoint_metadata(),
        "config": config,
        "train_config": asdict(train_config),
        "epoch": epoch,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def train_stateful_stream(
    model: AireadiStreamModel,
    train_streams: Sequence[AireadiParticipantStream],
    val_streams: Sequence[AireadiParticipantStream],
    cfg: AireadiTrainConfig,
    *,
    device: str = "cpu",
    output_dir,
    resolved_config: Optional[dict] = None,
    initial_epoch: int = 0,
    previous_history: Optional[List[dict]] = None,
    previous_best_val: Optional[float] = None,
    resume_optimizer_state: Optional[dict] = None,
    resume_scaler_state: Optional[dict] = None,
) -> Dict[str, object]:
    if cfg.train_mode not in {"stateful_stream", "windowed_debug"}:
        raise ValueError("AI-READI trainer supports train_mode='stateful_stream' or 'windowed_debug'.")
    print("[aireadi-train] insulin causal ranking loss is disabled for AI-READI; med_insulin is static metadata.")
    device_obj = torch.device(device)
    model.to(device_obj)
    train_streams = list(train_streams[:cfg.max_train_streams] if cfg.max_train_streams else train_streams)
    val_streams = list(val_streams[:cfg.max_val_streams] if cfg.max_val_streams else val_streams)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.mixed_precision and device_obj.type == "cuda"))
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            print("[aireadi-train] resume: optimizer state restored.")
        except Exception as exc:
            print(f"[aireadi-train] resume: optimizer state ignored ({type(exc).__name__}: {exc}).")
    if resume_scaler_state is not None and scaler.is_enabled():
        try:
            scaler.load_state_dict(resume_scaler_state)
            print("[aireadi-train] resume: AMP scaler state restored.")
        except Exception as exc:
            print(f"[aireadi-train] resume: AMP scaler state ignored ({type(exc).__name__}: {exc}).")
    out = Path(output_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_windows: List[Tuple[int, int]] = []
    val_windows: List[Tuple[int, int]] = []
    if cfg.train_mode == "windowed_debug":
        train_windows = _make_window_index(
            train_streams, cfg, stride=cfg.training_anchor_stride_steps, horizon=model.feature_spec.horizon_steps
        )
        val_windows = _make_window_index(
            val_streams, cfg, stride=cfg.val_anchor_stride_steps, horizon=model.feature_spec.horizon_steps
        )
        print(
            f"[aireadi-train] windowed_debug context_steps={cfg.windowed_context_steps} "
            f"batch_size={cfg.batch_size} windows={len(train_windows)}/{len(val_windows)}"
        )
        if not train_windows:
            raise RuntimeError("windowed_debug built no train windows; lower windowed_context_steps or check data.")
    else:
        print(f"[aireadi-train] stateful_stream batch_size={cfg.batch_size} chunk_steps={cfg.chunk_steps}")

    best_val = float(previous_best_val) if previous_best_val is not None else float("inf")
    bad_epochs = 0
    history: List[dict] = list(previous_history or [])
    rng = random.Random(cfg.seed + int(initial_epoch))
    for epoch in range(int(initial_epoch), int(initial_epoch) + int(cfg.max_epochs)):
        t0 = time.perf_counter()
        if device_obj.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device_obj)
        if cfg.train_mode == "windowed_debug":
            train_metrics = _windowed_train_epoch(
                model, train_streams, train_windows, cfg, optimizer, scaler, device_obj, rng
            )
            val_metrics = validate_windowed_debug(model, val_streams, val_windows, cfg, device_obj)
        else:
            train_metrics = _stream_train_epoch(model, train_streams, cfg, optimizer, scaler, device_obj, rng)
            val_metrics = validate_stateful_stream(model, val_streams, cfg, device_obj)
        elapsed = time.perf_counter() - t0
        train_anchors = int(train_metrics.get("n_train_anchors", 0))
        peak_mem_mb = torch.cuda.max_memory_allocated(device_obj) / 1e6 if device_obj.type == "cuda" else float("nan")
        rec = {
            "epoch": epoch,
            "seconds": elapsed,
            "anchors_per_s": train_anchors / max(elapsed, 1e-9),
            "peak_mem_mb": peak_mem_mb,
            **train_metrics,
            **val_metrics,
        }
        history.append(rec)
        print(
            f"[aireadi-train] epoch={epoch} train_pinball={rec['train_pinball_mgdl']:.3f} "
            f"val_pinball={rec['val_pinball_mgdl']:.3f} anchors={rec['n_train_anchors']}/{rec['n_val_anchors']} "
            f"({rec['anchors_per_s']:.0f} anc/s, {rec['peak_mem_mb']:.0f}MB peak)"
        )
        current_val = rec["val_pinball_mgdl"]
        if current_val < best_val - 1e-6 or math.isnan(best_val):
            best_val = current_val
            bad_epochs = 0
            save_aireadi_checkpoint(
                model,
                ckpt_dir / "best_model_checkpoint.pt",
                config=resolved_config or {},
                train_config=cfg,
                epoch=epoch,
                metrics=rec,
                optimizer=optimizer,
                scaler=scaler,
            )
        else:
            bad_epochs += 1
        if bad_epochs >= cfg.patience:
            print(f"[aireadi-train] early stop at epoch {epoch}; best_val={best_val:.3f}")
            break

    save_aireadi_checkpoint(
        model,
        ckpt_dir / "final_model_checkpoint.pt",
        config=resolved_config or {},
        train_config=cfg,
        epoch=history[-1]["epoch"] if history else -1,
        metrics=history[-1] if history else {},
        optimizer=optimizer,
        scaler=scaler,
    )
    _write_history(out / "metrics" / "training_history.csv", history)
    with (out / "metrics" / "training_summary.json").open("w") as f:
        json.dump({"best_val_pinball_mgdl": best_val, "history": history}, f, indent=2)
    return {"history": history, "best_val_pinball_mgdl": best_val}
