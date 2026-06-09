"""Truncated-BPTT stateful-stream trainer for SSM-CGM-Stream (spec §8).

The training loop mirrors deployment: for a batch of participant streams, init the
patient-specific state once, then walk the timeline in **chunks**. Each chunk is
consumed by one vectorized ``temporal.scan`` (so the recurrence is differentiable and
identical to the per-step ``update_stream``); forecast anchors inside the chunk are
decoded from their ``h_t`` and scored with the quantile loss; ``loss.backward()`` runs
**per chunk** and the carried state is **detached** at the chunk boundary (truncated
BPTT). The first chunk keeps the live ``h0`` so gradients train the static state
initializer and FiLM (the cold-start personalization path).

Leakage rules (spec §6) are inherited from the streaming primitives: only observed
steps advance the state; the raw-CGM target is never a decoder covariate; future
scenario values enter only the decoder under a (here training-sampled) mask.

Batching: participants are length-sorted and grouped into fixed-size lanes, padded to
the batch's longest stream. Padding sits *after* a lane's real steps, and only anchors
with a fully-observed in-range future are scored, so padded steps never corrupt a
scored forecast.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..data.streaming import ParticipantStream, StreamFeatureSpec
from ..evaluation import metrics as M
from ..evaluation.streaming import _baked_cont, evaluate_streams


@dataclass
class StreamTrainConfig:
    chunk_steps: int = 72                 # truncated-BPTT chunk length (72 = 6h @ 5min)
    detach_state_every_chunk: bool = True
    train_anchor_stride: int = 3          # steps between training anchors (3 = 15min)
    val_anchor_stride: int = 6            # steps between validation anchors (6 = 30min)
    batch_size: int = 8                   # participant lanes per batch
    max_epochs: int = 20
    lr: float = 1e-3
    grad_clip: float = 1.0
    scenario_train: str = "mixed"         # mixed (sampled mask) | forecast_only
    target_mode: str = "residual_current"  # group | global | residual_current (production default)
    patience: int = 5
    max_train_batches: Optional[int] = None   # cap batches/epoch (smoke)
    val_max_participants: Optional[int] = None
    seed: int = 0
    log_every: int = 20
    # ---- causal regularization (docs/causal.md; optional, off by default, logged apart) ----
    causal: bool = False                  # master switch: fit intervention library + causal losses
    rank_actions: Sequence[str] = ("carbs_g_at_t", "bolus_units_at_t")
    rank_score_kind: Dict[str, str] = field(default_factory=lambda: {
        "carbs_g_at_t": "max", "bolus_units_at_t": "delayed_mean"})  # carbs peak ↑; insulin delayed ↓
    rank_margin: float = 2.0              # mg/dL hinge margin on the dose→outcome ordering
    rank_n_dose_levels: int = 2
    rank_max_anchors: int = 16            # cap anchors used for the (expensive) ranking term per chunk
    alpha_rank: float = 0.0              # weight of the dose-ordering ranking loss (>0 activates it)
    lambda_slope: float = 0.0           # weight of the slope (rise/fall) auxiliary on the main forecast
    lambda_shape: float = 0.0           # weight of the effect-shape (smoothness/onset) prior on dose effects
    shape_immediate_penalty: float = 1.0  # h=0 onset penalty for delayed (insulin) actions in effect-shape
    deconfound: bool = False              # also rank derived-only (e.g. IOB-only) edits, so the model
    #                                       responds causally to on-board state, not just the intake scalar
    # ---- Weights & Biases experiment tracking (optional; no-op if wandb absent/disabled) ----
    wandb: bool = False                   # enable W&B logging of config + per-epoch metrics
    wandb_project: str = "ssmcgm-stream"
    wandb_entity: Optional[str] = None    # W&B team/user; None uses the logged-in default
    wandb_mode: str = "offline"           # online | offline (offline needs no API key; sync later) | disabled
    wandb_run_name: Optional[str] = None


# ---------------------------------------------------------------------------
# batching: length-sorted padded lanes
# ---------------------------------------------------------------------------
@dataclass
class _StreamBatch:
    cat: torch.Tensor          # (B, L, n_cat)
    cont: torch.Tensor         # (B, L, n_cont)  (rel_time_idx baked to "now")
    target: torch.Tensor       # (B, L) raw mg/dL
    observed: torch.Tensor     # (B, L) bool
    target_scale: torch.Tensor  # (B, 2)
    anchor: torch.Tensor       # (B, L) current known glucose mg/dL (residual_current)
    real_len: List[int]
    rel_decoder: Optional[torch.Tensor]   # (B, H) per-lane canonical decoder rel ramp
    anchors: List[List[int]] = field(default_factory=list)  # valid anchor t per lane


def _make_batches(streams: Sequence[ParticipantStream], spec: StreamFeatureSpec, cfg: StreamTrainConfig,
                  device, *, stride: int, rng: np.random.Generator) -> List[_StreamBatch]:
    H = spec.horizon
    order = sorted(range(len(streams)), key=lambda i: streams[i].n_steps)
    batches: List[_StreamBatch] = []
    for start in range(0, len(order), cfg.batch_size):
        idxs = order[start:start + cfg.batch_size]
        grp = [streams[i] for i in idxs]
        L = max(s.n_steps for s in grp)
        B = len(grp)
        n_cont = grp[0].full_cont.shape[1]
        n_cat = grp[0].full_cat.shape[1]
        cat = torch.zeros(B, L, n_cat, dtype=grp[0].full_cat.dtype)
        cont = torch.zeros(B, L, n_cont, dtype=grp[0].full_cont.dtype)
        target = torch.full((B, L), float("nan"))
        anchor = torch.zeros(B, L)
        observed = torch.zeros(B, L, dtype=torch.bool)
        tscale = torch.zeros(B, 2)
        real_len, anchors = [], []
        for b, s in enumerate(grp):
            n = s.n_steps
            cont_baked = _baked_cont(s, spec)
            cat[b, :n] = s.full_cat
            cont[b, :n] = cont_baked
            if n < L:                                   # pad with the last real row
                cat[b, n:] = s.full_cat[-1]
                cont[b, n:] = cont_baked[-1]
            target[b, :n] = s.target
            anchor[b, :n] = s.anchor if s.anchor is not None else s.target
            observed[b, :n] = s.target_observed
            tscale[b] = s.target_scale
            real_len.append(n)
            # valid anchors: full observed future, strided
            va = []
            last = -10 ** 9
            for t in range(0, n - H):
                if (t - last) < stride:
                    continue
                if bool(s.target_observed[t + 1:t + 1 + H].all()) and torch.isfinite(s.target[t + 1:t + 1 + H]).all():
                    va.append(t)
                    last = t
            anchors.append(va)
        batches.append(_StreamBatch(
            cat=cat.to(device), cont=cont.to(device), target=target.to(device),
            observed=observed.to(device), target_scale=tscale.to(device), anchor=anchor.to(device),
            real_len=real_len,
            rel_decoder=(torch.stack([s.rel_decoder for s in grp]).to(device)   # (B, H) per-lane
                         if grp[0].rel_decoder is not None else None),
            anchors=anchors))
    rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------------------
# causal regularization helpers (docs/causal.md) — used only when cfg.causal
# ---------------------------------------------------------------------------
def _median_idx(model) -> int:
    """Index of the median (q≈0.5) quantile in the model's output, else 0."""
    q = getattr(getattr(model, "loss", None), "quantiles", None)
    if not q:
        return 0
    return min(range(len(q)), key=lambda i: abs(float(q[i]) - 0.5))


def _reveal_mask(model, action, library, A, H, device, dtype) -> torch.Tensor:
    """(A, H, n_scenario) mask revealing ``action``'s intake + any of its derived features
    that are scenario vars (forecast-only — masked 0 — elsewhere)."""
    m = torch.zeros(A, H, model.n_scenario_vars, device=device, dtype=dtype)
    svars = getattr(model, "scenario_vars", [])
    for name in [action] + library.taxonomy.derived_features(action):
        if name in svars:
            m[:, :, svars.index(name)] = 1.0
    return m


def _decode_variant(model, e_s_sub, h_t, dec_cat, dec_cont, reveal_mask, transform,
                    anchor_mgdl, tscale_sub, qmid):
    """Decode one coherent dose variant from a shared ``h_t`` → median mg/dL trajectory (A, H)."""
    A, H = dec_cont.shape[0], dec_cont.shape[1]
    if model.decoder_time_fusion is not None:
        Tfeat, _ = model.decoder_time_fusion(
            model._features(dec_cat, dec_cont, model._time_decoder_vars), return_contributions=False)
    else:
        Tfeat = dec_cont.new_zeros(A, H, 0)
    Avals = model._scenario_values(dec_cat, dec_cont)
    raw = model.decoder(h_t, e_s_sub, Tfeat, Avals, reveal_mask.to(Avals.dtype))   # (A,H,Q)
    pred = transform.predict_mgdl(model, raw, anchor_mgdl, tscale_sub)
    return pred[..., qmid] if pred.dim() == 3 else pred                            # (A, H) mg/dL


def _ranking_loss_term(model, spec, library, cfg, h_t, e_s_sub, dec_cont, dec_cat,
                       anchor_mgdl, tscale_sub, transform):
    """Causal dose-ordering loss (+ optional effect-shape prior) over a set of anchors.

    For each configured action with a KNOWN global direction, decode K coherent dose
    variants from the SAME ``h_t`` — editing only the future decoder covariates (intake +
    its derived footprint) and revealing the action in the scenario mask — then penalize
    forecasts whose dose→outcome ordering violates the known sign (carbs↑, insulin↓). The
    effect-shape prior smooths each dose's effect vs the ``none`` baseline (and penalizes
    an immediate h=0 jump for delayed/insulin actions). Returns ``(rank, shape, n_terms)``.
    """
    from ..causal.losses import score_glucose, intervention_ranking_loss, effect_shape_loss
    A, H = dec_cont.shape[0], dec_cont.shape[1]
    ridx = spec.rel_time_idx_col
    qmid = _median_idx(model)
    rank_total, shape_total, nterms = h_t.new_zeros(()), h_t.new_zeros(()), 0
    for a in cfg.rank_actions:
        if a not in library.actions or not library.taxonomy.has_known_direction(a):
            continue
        direction = library.taxonomy.direction_of(a)
        levels = library.levels[a]
        # Edit modes for the dose ranking. "coherent" edits the intake scalar + its on-board/absorption
        # footprint together (the natural intervention). With ``deconfound`` we ALSO rank a
        # "derived_only" set: vary the on-board/absorption footprint with the intake left MASKED, so the
        # model must obey the direction w.r.t. the derived state itself (e.g. insulin-on-board ⇒ glucose↓)
        # and cannot satisfy the objective by keying only on the intake scalar (treatment-by-indication).
        modes = ["coherent"]
        if getattr(cfg, "deconfound", False) and library.derived_idx[a]:
            modes.append("derived_only")
        for mode in modes:
            trajs = []
            for lvl in levels:
                dc = dec_cont.clone()
                if mode == "coherent":
                    dc[:, library.onset_step:, library.intake_idx[a]] = library.intake_scaled[a][lvl]
                    rm = _reveal_mask(model, a, library, A, H, dc.device, dc.dtype)
                else:                                                      # intake masked (unknown)
                    rm = torch.zeros(A, H, model.n_scenario_vars, device=dc.device, dtype=dc.dtype)
                fp = library.footprint[a][lvl].to(dc.device, dc.dtype)     # (H, n_derived)
                for j, idx in enumerate(library.derived_idx[a]):
                    if idx != ridx:                                        # never let an edit touch the rel ramp
                        dc[:, :, idx] = fp[:, j]
                trajs.append(_decode_variant(model, e_s_sub, h_t, dec_cat, dc, rm, transform,
                                             anchor_mgdl, tscale_sub, qmid))    # (A, H)
            scores = torch.stack([score_glucose(t, cfg.rank_score_kind.get(a, "mean")) for t in trajs], dim=1)
            rank_total = rank_total + intervention_ranking_loss(
                scores, list(range(len(levels))), direction, margin=cfg.rank_margin)
            if cfg.lambda_shape > 0:
                imm = cfg.shape_immediate_penalty if direction < 0 else 0.0    # insulin shouldn't act at h=0
                for t in trajs[1:]:                                        # effect vs the `none` baseline
                    shape_total = shape_total + effect_shape_loss(t - trajs[0], immediate_penalty=imm)
            nterms += 1
    n = max(nterms, 1)
    return rank_total / n, shape_total / n, nterms


# ---------------------------------------------------------------------------
# one batch -> summed quantile loss over its anchors (truncated BPTT)
# ---------------------------------------------------------------------------
def _decode_loss(model, spec, batch: _StreamBatch, e_s, out, s, e, anchors_bt, transform,
                 *, library=None, cfg=None):
    """Quantile loss (mg/dL) over the anchors whose ``h_t`` lives in this chunk, plus the
    optional causal terms (ranking / slope / effect-shape) when ``library``+``cfg.causal``.

    Predictions are mapped to mg/dL via ``transform`` (group/global/residual_current),
    so the loss — and thus training — is in the same space the eval scores. Returns
    ``(forecast_loss, n_anchors, components)`` where ``components`` holds the (unweighted)
    causal terms for separate logging."""
    from ..causal.losses import slope_loss
    H, ridx = spec.horizon, spec.rel_time_idx_col
    device = out.device
    b_idx = torch.tensor([b for b, _ in anchors_bt], device=device)
    t_idx = torch.tensor([t for _, t in anchors_bt], device=device)
    h_t = out[b_idx, t_idx - s]                                   # (A, d)
    fut = (t_idx[:, None] + 1) + torch.arange(H, device=device)[None, :]   # (A, H)
    dec_cont = batch.cont[b_idx[:, None], fut].clone()            # (A, H, n_cont)
    dec_cat = batch.cat[b_idx[:, None], fut]
    if ridx is not None and batch.rel_decoder is not None:
        dec_cont[:, :, ridx] = batch.rel_decoder[b_idx]      # per-lane (A, H), matches eval
    if model.decoder_time_fusion is not None:
        Tfeat, _ = model.decoder_time_fusion(
            model._features(dec_cat, dec_cont, model._time_decoder_vars), return_contributions=False)
    else:
        Tfeat = dec_cont.new_zeros(len(anchors_bt), H, 0)
    Avals = model._scenario_values(dec_cat, dec_cont)            # (A, H, n_scn)
    if model.n_scenario_vars and getattr(model, "_train_scn", "mixed") == "mixed":
        from ..scenario import sample_scenario_dropout_mask
        Mmask = sample_scenario_dropout_mask(
            len(anchors_bt), H, model.n_scenario_vars, model.hparams.scenario_dropout_mode,
            model.hparams.scenario_dropout_p, device).to(Avals.dtype)
    else:
        Mmask = torch.zeros_like(Avals)
    pred = model.decoder(h_t, e_s[b_idx], Tfeat, Avals, Mmask)   # (A, H, Q) raw output
    anchor_mgdl = batch.anchor[b_idx, t_idx] if transform.needs_anchor else None
    pred = transform.predict_mgdl(model, pred, anchor_mgdl, batch.target_scale[b_idx])  # -> mg/dL
    tgt = batch.target[b_idx[:, None], fut]                      # (A, H) mg/dL
    floss = model.loss(pred, tgt)

    comps = {"rank": 0.0, "slope": 0.0, "shape": 0.0}
    if cfg is not None and cfg.causal and library is not None:
        qmid = _median_idx(model)
        med = pred[..., qmid] if pred.dim() == 3 else pred       # (A, H) median mg/dL
        if cfg.lambda_slope > 0:
            comps["slope"] = slope_loss(med, tgt)
        if cfg.alpha_rank > 0 or cfg.lambda_shape > 0:
            A = len(anchors_bt)
            k = min(A, cfg.rank_max_anchors) if cfg.rank_max_anchors else A
            sl = slice(0, k)
            rank, shape, _ = _ranking_loss_term(
                model, spec, library, cfg, h_t[sl], e_s[b_idx][sl], dec_cont[sl], dec_cat[sl],
                (anchor_mgdl[sl] if anchor_mgdl is not None else None),
                batch.target_scale[b_idx][sl], transform)
            comps["rank"], comps["shape"] = rank, shape
    return floss, len(anchors_bt), comps


def _run_batch(model, spec, batch: _StreamBatch, cfg, optimizer, transform, *, library=None):
    """Stream one padded batch chunk-by-chunk with truncated BPTT; return mean losses."""
    device = batch.cont.device
    B, L = batch.cont.shape[0], batch.cont.shape[1]
    enc_vars = model.encoder_variables
    x0 = {"encoder_cat": batch.cat[:, :1], "encoder_cont": batch.cont[:, :1]}
    sctx = model.encode_static(x0)
    state = model.init_stream(sctx)                              # live h0 for chunk 0
    total_loss, total_anchors, first = 0.0, 0, True
    caus = {"rank": 0.0, "slope": 0.0, "shape": 0.0}
    for s in range(0, L, cfg.chunk_steps):
        e = min(s + cfg.chunk_steps, L)
        # anchors whose h_t is in [s, e) and that exist in any lane
        anchors_bt = [(b, t) for b in range(B) for t in batch.anchors[b] if s <= t < e]
        sctx_f = model.encode_static(x0)                        # fresh e_s graph per chunk
        feats = model._features(batch.cat[:, s:e], batch.cont[:, s:e], enc_vars)
        u, _ = model._fuse_history(feats, sctx_f.embedding)     # (B, C, d)
        out, ls, cs, _ = model.temporal.scan(u, state.layer_states, state.conv_states,
                                             static_embedding=sctx_f.embedding)
        if anchors_bt:
            loss, na, comps = _decode_loss(model, spec, batch, sctx_f.embedding, out, s, e,
                                           anchors_bt, transform, library=library, cfg=cfg)
            total = (loss + cfg.alpha_rank * comps["rank"]
                     + cfg.lambda_slope * comps["slope"] + cfg.lambda_shape * comps["shape"])
            total.backward()                                    # forecast loss is already per-anchor mean
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * na; total_anchors += na   # anchor-weighted mean
            for kk in caus:
                v = comps[kk]
                caus[kk] += (float(v.detach()) if torch.is_tensor(v) else float(v)) * na
        # carry state forward, detached (truncated BPTT)
        from ..stream.state import StreamState
        state = StreamState(layer_states=[t.detach() for t in ls],
                            conv_states=None if cs is None else [None if c is None else c.detach() for c in cs],
                            last_output=out[:, -1].detach(), static_context=sctx, step=e)
        first = False
    na_tot = max(total_anchors, 1)
    return total_loss / na_tot, total_anchors, {k: v / na_tot for k, v in caus.items()}


# ---------------------------------------------------------------------------
# validation (reuses the production streaming eval)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _validate(model, val_streams, spec, cfg, transform) -> Dict[str, float]:
    if not val_streams:
        return {"pinball": float("nan"), "mae": float("nan"), "n": 0}
    streams = val_streams[:cfg.val_max_participants] if cfg.val_max_participants else val_streams
    res = evaluate_streams(model, streams, spec, scenario_mode="forecast_only",
                           anchor_stride=cfg.val_anchor_stride, method="scan", transform=transform)
    return M.overall_metrics(res["predictions"])


def _init_wandb(cfg: StreamTrainConfig, model, run_dir):
    """Best-effort W&B init. Returns the run handle or ``None``. Never raises — any failure
    (package missing, no key, offline) degrades to local-only training."""
    if not cfg.wandb or cfg.wandb_mode == "disabled":
        return None
    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", cfg.wandb_mode)   # offline needs no API key
        n_params = sum(p.numel() for p in model.parameters())
        run = wandb.init(
            project=cfg.wandb_project, entity=cfg.wandb_entity, mode=cfg.wandb_mode,
            name=cfg.wandb_run_name or (os.path.basename(run_dir) if run_dir else None),
            dir=run_dir or ".", config={**{k: getattr(cfg, k) for k in vars(cfg)},
                                         "n_params": n_params})
        print(f"[wandb] logging to project={cfg.wandb_project} mode={cfg.wandb_mode}")
        return run
    except Exception as e:  # pragma: no cover - tracking must never break training
        print(f"[wandb] disabled ({type(e).__name__}: {e})")
        return None


def _wandb_log(run, rec):
    if run is None:
        return
    try:
        run.log({k: v for k, v in rec.items() if isinstance(v, (int, float))}, step=rec.get("epoch"))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# top-level training loop
# ---------------------------------------------------------------------------
def train_stateful_stream(
    model,
    train_streams: Sequence[ParticipantStream],
    val_streams: Sequence[ParticipantStream],
    spec: StreamFeatureSpec,
    cfg: StreamTrainConfig,
    *,
    device: Optional[str] = None,
    run_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Train ``model`` with truncated-BPTT participant streaming; return history.

    Selects the best checkpoint by validation streaming pinball loss (forecast-only),
    with early stopping. Writes ``best_model_checkpoint.pt`` / ``training_history.csv``
    to ``run_dir`` when given.
    """
    from ..evaluation.target_transform import TargetTransform, fit_residual_scale
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model._train_scn = cfg.scenario_train
    # production-safe target transform (residual_current scale fit on TRAIN streams only)
    res_scale = (fit_residual_scale(train_streams, spec) if cfg.target_mode == "residual_current" else None)
    transform = TargetTransform(mode=cfg.target_mode, residual_scale=res_scale)
    print(f"[bptt] target_mode={cfg.target_mode}"
          + (f" residual_scale(5..60min)={[round(x,1) for x in res_scale]}" if res_scale else ""))
    # optional causal regularization: fit the dose-level intervention library on TRAIN streams
    library = None
    if cfg.causal:
        from ..causal.taxonomy import InterventionTaxonomy
        from ..causal.interventions import InterventionLibrary
        library = InterventionLibrary.fit_from_streams(
            train_streams, InterventionTaxonomy(), spec,
            actions=tuple(cfg.rank_actions), n_dose_levels=cfg.rank_n_dose_levels)
        print(f"[bptt] causal: library actions={library.actions} "
              f"levels={ {a: library.levels[a] for a in library.actions} } "
              f"alpha_rank={cfg.alpha_rank} lambda_slope={cfg.lambda_slope} lambda_shape={cfg.lambda_shape}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        transform.save(os.path.join(run_dir, "target_transform.json"))
    wandb_run = _init_wandb(cfg, model, run_dir)

    history: List[dict] = []
    best_val, best_epoch, bad = float("inf"), -1, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        batches = _make_batches(train_streams, spec, cfg, device, stride=cfg.train_anchor_stride, rng=rng)
        if cfg.max_train_batches:
            batches = batches[:cfg.max_train_batches]
        cuda = device.startswith("cuda") if isinstance(device, str) else (device.type == "cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        ep_loss, ep_anchors = 0.0, 0
        ep_caus = {"rank": 0.0, "slope": 0.0, "shape": 0.0}
        for bi, batch in enumerate(batches):
            bl, na, caus = _run_batch(model, spec, batch, cfg, optimizer, transform, library=library)
            ep_loss += bl * na; ep_anchors += na
            for kk in ep_caus:
                ep_caus[kk] += caus[kk] * na
            if cfg.log_every and (bi + 1) % cfg.log_every == 0:
                print(f"[bptt] epoch {epoch} batch {bi + 1}/{len(batches)} "
                      f"train_pinball={ep_loss / max(ep_anchors, 1):.3f}"
                      + (f" rank={ep_caus['rank'] / max(ep_anchors, 1):.3f}" if cfg.causal else ""))
        train_seconds = time.perf_counter() - t0          # pure training time (excludes validation)
        peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if cuda else float("nan")
        train_pinball = ep_loss / max(ep_anchors, 1)
        ep_caus = {k: v / max(ep_anchors, 1) for k, v in ep_caus.items()}

        model.eval()
        val = _validate(model, list(val_streams), spec, cfg, transform)
        dt = time.perf_counter() - t0                      # epoch wall-clock incl. validation
        rec = {"epoch": epoch, "train_pinball": train_pinball, "val_pinball": val["pinball"],
               "val_mae": val["mae"], "val_rmse": val.get("rmse", float("nan")),
               "n_train_anchors": ep_anchors, "seconds": dt,
               "train_seconds": train_seconds,            # efficiency metrics for the manuscript
               "anchors_per_s": ep_anchors / max(train_seconds, 1e-9),
               "peak_mem_mb": peak_mem_mb}
        if cfg.causal:
            rec.update({f"train_{k}": ep_caus[k] for k in ep_caus})
        history.append(rec)
        _wandb_log(wandb_run, rec)
        print(f"[bptt] epoch {epoch}: train_pinball={train_pinball:.3f} "
              f"val_pinball={val['pinball']:.3f} val_mae={val['mae']:.2f} "
              f"({train_seconds:.0f}s train, {ep_anchors / max(train_seconds, 1e-9):.0f} anc/s, "
              f"{peak_mem_mb:.0f}MB peak)"
              + (f" | rank={ep_caus['rank']:.3f} slope={ep_caus['slope']:.3f} shape={ep_caus['shape']:.3f}"
                 if cfg.causal else ""))

        improved = val["pinball"] < best_val - 1e-5
        if improved or math.isnan(best_val):
            best_val, best_epoch, bad = val["pinball"], epoch, 0
            if run_dir:
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model_checkpoint.pt"))
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[bptt] early stop at epoch {epoch} (best epoch {best_epoch}, "
                      f"val_pinball={best_val:.3f})")
                break

    if run_dir:
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "training_history.csv"), index=False)
        torch.save(model.state_dict(), os.path.join(run_dir, "final_model_checkpoint.pt"))
    if wandb_run is not None:
        try:
            wandb_run.summary["best_val_pinball"] = best_val
            wandb_run.summary["best_epoch"] = best_epoch
            wandb_run.finish()
        except Exception:
            pass
    return {"history": pd.DataFrame(history), "best_val_pinball": best_val,
            "best_epoch": best_epoch, "transform": transform}
