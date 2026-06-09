"""Chronological per-participant streaming evaluation (spec §9, §10, §21).

The production simulation, for each held-out participant:

    sctx  = model.encode_static(static_features)     # once
    state = model.init_stream(sctx)                  # patient-specific h0
    for t in chronological order:
        h_t = advance state to include observation t          # observed-only
        if t is a valid forecast anchor:
            pred = model.decode_horizon(h_t, sctx, T[t+1:t+H], A[t+1:t+H], M[t+1:t+H])
            score pred vs target[t+1:t+H]

Key invariants (spec §3, §6): the state is initialized **once** and advanced
forward (never a 24/48h window rebuilt at an anchor); only data observed up to ``t``
reaches the state; future planned/scenario values enter **only** the horizon decoder;
the raw-CGM target is never a decoder covariate, so no future glucose leaks.

Two equivalent state-advance methods (``scan`` == repeated ``step`` to ~1e-10):

* ``method="step"`` — the literal deployment path, one ``update_stream`` per 5-min
  step. Exercised by the production tests and used to sample per-call latency.
* ``method="scan"`` (default for bulk eval) — vectorize the recurrence with one
  ``temporal.scan`` over the whole record (feature-building hoisted out of the loop),
  then batch-decode all anchors. Identical forecasts, ~100x faster on long records.

Warm-up (spec §11): ``warmup_steps`` gates *scoring* only — the state is still
advanced during warm-up; anchors before the warm-up boundary emit no rows.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..data.streaming import ParticipantStream, StreamFeatureSpec
from .metrics import DEFAULT_QUANTILES, qcol, stream_state_bytes
from .target_transform import TargetTransform

SCENARIO_MODES = ("forecast_only", "factual", "planned")
STATE_METHODS = ("scan", "step")


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def _quantile_levels(model) -> List[float]:
    qs = getattr(getattr(model, "loss", None), "quantiles", None)
    return list(qs) if qs else list(DEFAULT_QUANTILES)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _scenario_inputs(model, dec_cat, dec_cont, mode: str, planned_spec):
    """Scenario values ``A`` and availability mask ``M`` for the decoder (any batch).

    ``forecast_only`` → all-zero mask (deployable baseline); ``factual`` → all-ones
    mask over the *observed* future actions (retrospective upper bound); ``planned`` →
    mask 1 only for the named variables, with their values overridden (scaled space).
    """
    A = model._scenario_values(dec_cat, dec_cont)  # (B, H, n_scenario)
    if mode == "forecast_only":
        return A, torch.zeros_like(A)
    if mode == "factual":
        return A, torch.ones_like(A)
    if mode == "planned":
        A, M = A.clone(), torch.zeros_like(A)
        for name, val in (planned_spec or {}).items():
            if name in model.scenario_vars:
                j = model.scenario_vars.index(name)
                A[..., j] = float(val)
                M[..., j] = 1.0
        return A, M
    raise ValueError(f"scenario mode must be one of {SCENARIO_MODES}, got {mode!r}")


def _baked_cont(stream: ParticipantStream, spec: StreamFeatureSpec) -> torch.Tensor:
    """Encoder cont with ``relative_time_idx`` set to the canonical "now" value at
    every step (every observed step is the present; spec §4 ``rel_policy="anchor_now"``)."""
    cont = stream.full_cont
    if spec.rel_time_idx_col is None or stream.rel_now is None:
        return cont
    cont = cont.clone()
    cont[:, spec.rel_time_idx_col] = stream.rel_now
    return cont


def _collect_anchors(stream: ParticipantStream, spec: StreamFeatureSpec, *,
                     warmup_steps: int, anchor_stride: int, eval_split: Optional[str],
                     max_anchors: Optional[int]) -> List[int]:
    """Valid forecast-anchor positions ``t`` (full future exists, past warm-up, on
    stride, in the requested split). Pure python over the timeline — cheap."""
    n, H = stream.n_steps, spec.horizon
    anchors: List[int] = []
    last = -10 ** 9
    for t in range(n):
        if t > n - 1 - H:
            break
        if t + 1 <= warmup_steps:
            continue
        if eval_split is not None and stream.split_at(t) != eval_split:
            continue
        if (t - last) < anchor_stride:
            continue
        last = t
        anchors.append(t)
        if max_anchors is not None and len(anchors) >= max_anchors:
            break
    return anchors


@torch.no_grad()
def _decode_anchors(model, sctx, stream: ParticipantStream, spec: StreamFeatureSpec,
                    anchors: List[int], h_t: torch.Tensor, *, scenario_mode: str,
                    planned_spec, qlevels: List[float], device, transform=None) -> List[dict]:
    """Batch-decode every anchor from its ``h_t`` and emit per-(anchor, horizon) rows.

    ``h_t`` is ``(A, d_model)`` — the state output at each anchor, obtained identically
    by ``step`` or ``scan``. The future known/scenario inputs are sliced from the full
    stream tensors (with the canonical ``[1..H]`` ``relative_time_idx`` ramp); the raw
    CGM target is *not* among them, so nothing future leaks into the forecast.
    """
    if not anchors:
        return []
    A, H = len(anchors), spec.horizon
    pos = torch.tensor(anchors, device=device)
    idx = (pos[:, None] + 1) + torch.arange(H, device=device)[None, :]   # (A, H) future rows
    dec_cont = stream.full_cont[idx].clone()                              # (A, H, n_cont)
    dec_cat = stream.full_cat[idx]                                        # (A, H, n_cat)
    if spec.rel_time_idx_col is not None and stream.rel_decoder is not None:
        dec_cont[:, :, spec.rel_time_idx_col] = stream.rel_decoder       # canonical [1..H]

    if model.decoder_time_fusion is not None:
        tfeats = model._features(dec_cat, dec_cont, model._time_decoder_vars)
        Tfeat, _ = model.decoder_time_fusion(tfeats, return_contributions=False)
    else:
        Tfeat = dec_cont.new_zeros(A, H, 0)
    Avals, M = _scenario_inputs(model, dec_cat, dec_cont, scenario_mode, planned_spec)
    e_s = sctx.embedding.expand(A, -1)
    pred = model.decoder(h_t, e_s, Tfeat, Avals, M)                      # (A, H, Q) raw output
    ts = stream.target_scale.unsqueeze(0).expand(A, -1)
    transform = transform or TargetTransform(mode="group")
    anchor_mgdl = (stream.anchor[pos] if (transform.needs_anchor and stream.anchor is not None)
                   else torch.zeros(A, device=device))
    pred = transform.predict_mgdl(model, pred, anchor_mgdl, ts).cpu()    # -> mg/dL

    rows: List[dict] = []
    target = stream.target.cpu()
    observed = stream.target_observed.cpu()
    time_idx = stream.time_idx.cpu()
    seg = stream.segment_id.cpu() if stream.segment_id is not None else None
    for a, t in enumerate(anchors):
        steps_since_start = t + 1
        hours_since_start = steps_since_start * spec.bin_minutes / 60.0
        split = stream.split_at(t)
        anchor_seg = int(seg[t]) if seg is not None else -1
        for h in range(H):
            ft = t + 1 + h
            tgt = float(target[ft])
            row = {
                "participant_id": stream.participant_id, "split": split,
                "anchor_time_idx": int(time_idx[t]),
                "steps_since_start": steps_since_start, "hours_since_start": hours_since_start,
                "horizon_step": h + 1, "horizon_minutes": spec.bin_minutes * (h + 1),
                "target": tgt, "observed": bool(observed[ft]) and np.isfinite(tgt),
                "scenario_mode": scenario_mode, "segment_id": anchor_seg,
            }
            for qi, level in enumerate(qlevels):
                row[qcol(level)] = float(pred[a, h, qi])
            rows.append(row)
    return rows


@torch.no_grad()
def run_participant_stream(
    model,
    stream: ParticipantStream,
    spec: StreamFeatureSpec,
    *,
    scenario_mode: str = "forecast_only",
    planned_spec: Optional[Dict[str, float]] = None,
    warmup_steps: int = 0,
    anchor_stride: int = 1,
    eval_split: Optional[str] = None,
    max_anchors: Optional[int] = None,
    method: str = "step",
    transform: Optional["TargetTransform"] = None,
    quantiles: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Stream one participant chronologically and score every valid forecast anchor.

    ``method="step"`` advances the state with one ``update_stream`` per timestep (the
    deployment path; ``n_update_calls == n_steps``). ``method="scan"`` advances it with
    a single vectorized ``temporal.scan`` (identical h_t, much faster). Returns
    ``{"rows", "state_bytes", "n_steps", "n_anchors", "n_update_calls", ...}``.
    """
    if scenario_mode not in SCENARIO_MODES:
        raise ValueError(f"scenario_mode must be one of {SCENARIO_MODES}")
    if method not in STATE_METHODS:
        raise ValueError(f"method must be one of {STATE_METHODS}")
    device = _model_device(model)
    model.eval()
    stream = stream.to(device)
    qlevels = list(quantiles) if quantiles is not None else _quantile_levels(model)
    n, H = stream.n_steps, spec.horizon
    enc_vars = model.encoder_variables
    cont_baked = _baked_cont(stream, spec)

    x0 = {"encoder_cat": stream.full_cat[:1].unsqueeze(0),
          "encoder_cont": stream.full_cont[:1].unsqueeze(0)}
    sctx = model.encode_static(x0)
    state = model.init_stream(sctx)
    state_bytes = stream_state_bytes(state)

    anchors = _collect_anchors(stream, spec, warmup_steps=warmup_steps,
                               anchor_stride=anchor_stride, eval_split=eval_split,
                               max_anchors=max_anchors)
    anchor_set = set(anchors)
    n_update = 0

    if method == "step":
        h_list = []
        for t in range(n):
            cat_t = stream.full_cat[t].view(1, 1, -1)
            feats = model._features(cat_t, cont_baked[t].view(1, 1, -1), enc_vars)
            obs_t = {nm: feats[nm][:, 0] for nm in enc_vars}
            state = model.update_stream(state, obs_t)
            n_update += 1
            if t in anchor_set:
                h_list.append(state.last_output)            # (1, d)
        h_t = torch.cat(h_list, dim=0) if h_list else stream.full_cont.new_zeros(0, model.hparams.hidden_size)
    else:  # scan — one vectorized pass over the whole record
        enc_feats = model._features(stream.full_cat.unsqueeze(0), cont_baked.unsqueeze(0), enc_vars)
        u, _ = model._fuse_history(enc_feats, sctx.embedding)            # (1, n, d)
        out, *_ = model.temporal.scan(u, state.layer_states, state.conv_states,
                                      static_embedding=sctx.embedding)
        h_t = out[0, torch.tensor(anchors, device=device)] if anchors else out[0, :0]

    rows = _decode_anchors(model, sctx, stream, spec, anchors, h_t, scenario_mode=scenario_mode,
                           planned_spec=planned_spec, qlevels=qlevels, device=device, transform=transform)
    return {"rows": rows, "state_bytes": int(state_bytes), "n_steps": n,
            "n_anchors": len(anchors), "n_update_calls": n_update,
            "n_decode_calls": len(anchors), "participant_id": stream.participant_id}


@torch.no_grad()
def sample_deployment_latency(model, stream: ParticipantStream, spec: StreamFeatureSpec,
                              *, reps: int = 50, warmup: int = 5) -> Dict[str, List[float]]:
    """Per-call latency (ms) of the single-step deployment API (spec §13.6).

    Times ``update_stream`` and ``decode_horizon`` on one participant's state — the
    real online cost regardless of which bulk method scored the cohort."""
    device = _model_device(model)
    model.eval()
    stream = stream.to(device)
    cont_baked = _baked_cont(stream, spec)
    enc_vars = model.encoder_variables
    H = spec.horizon
    x0 = {"encoder_cat": stream.full_cat[:1].unsqueeze(0),
          "encoder_cont": stream.full_cont[:1].unsqueeze(0)}
    sctx = model.encode_static(x0)
    state = model.init_stream(sctx)
    feats = model._features(stream.full_cat[0].view(1, 1, -1), cont_baked[0].view(1, 1, -1), enc_vars)
    obs0 = {nm: feats[nm][:, 0] for nm in enc_vars}
    t0idx = min(spec.horizon, stream.n_steps - H - 1)
    dec_cont = stream.full_cont[t0idx + 1:t0idx + 1 + H].unsqueeze(0)
    dec_cat = stream.full_cat[t0idx + 1:t0idx + 1 + H].unsqueeze(0)
    if model.decoder_time_fusion is not None:
        Tfeat, _ = model.decoder_time_fusion(
            model._features(dec_cat, dec_cont, model._time_decoder_vars), return_contributions=False)
    else:
        Tfeat = dec_cont.new_zeros(1, H, 0)
    Avals, M = _scenario_inputs(model, dec_cat, dec_cont, "forecast_only", None)

    out = {"update_stream_ms": [], "decode_horizon_ms": []}
    for i in range(reps + warmup):
        _sync(device); t = time.perf_counter()
        state = model.update_stream(state, obs0)
        _sync(device)
        if i >= warmup:
            out["update_stream_ms"].append((time.perf_counter() - t) * 1e3)
        _sync(device); t = time.perf_counter()
        model.decode_horizon(state, sctx, Tfeat, Avals, M)
        _sync(device)
        if i >= warmup:
            out["decode_horizon_ms"].append((time.perf_counter() - t) * 1e3)
    return out


def evaluate_streams(
    model,
    streams: Sequence[ParticipantStream],
    spec: StreamFeatureSpec,
    *,
    scenario_mode: str = "forecast_only",
    planned_spec: Optional[Dict[str, float]] = None,
    warmup_steps: int = 0,
    anchor_stride: int = 1,
    eval_split: Optional[str] = None,
    max_anchors_per_participant: Optional[int] = None,
    method: str = "scan",
    transform: Optional["TargetTransform"] = None,
    record_latency: bool = False,
    keep_unobserved: bool = False,
    progress: bool = False,
) -> Dict[str, object]:
    """Run :func:`run_participant_stream` over many participants and collect results.

    Returns ``{"predictions": DataFrame, "memory": DataFrame, "timings": {...},
    "n_participants": int}``. Bulk scoring uses ``method`` (``scan`` by default);
    per-call deployment latency is sampled separately on the first stream when
    ``record_latency`` so the §13.6 numbers reflect true single-step cost.
    """
    all_rows: List[dict] = []
    mem_rows: List[dict] = []
    for i, stream in enumerate(streams):
        res = run_participant_stream(
            model, stream, spec, scenario_mode=scenario_mode, planned_spec=planned_spec,
            warmup_steps=warmup_steps, anchor_stride=anchor_stride, eval_split=eval_split,
            max_anchors=max_anchors_per_participant, method=method, transform=transform)
        all_rows.extend(res["rows"])
        mem_rows.append({"participant_id": res["participant_id"], "n_steps": res["n_steps"],
                         "n_anchors": res["n_anchors"], "state_bytes": res["state_bytes"],
                         "state_kib": res["state_bytes"] / 1024.0})
        if progress:
            print(f"[stream-eval] {i + 1}/{len(streams)} {res['participant_id']}: "
                  f"{res['n_anchors']} anchors over {res['n_steps']} steps")

    timings = {"update_stream_ms": [], "decode_horizon_ms": []}
    if record_latency and len(streams):
        timings = sample_deployment_latency(model, streams[0], spec)

    pred = pd.DataFrame(all_rows)
    if len(pred) and not keep_unobserved:
        pred = pred[pred["observed"]].reset_index(drop=True)
    return {"predictions": pred, "memory": pd.DataFrame(mem_rows), "timings": timings,
            "n_participants": len(streams)}
