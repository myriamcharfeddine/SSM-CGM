"""Phase B — CGMacros retrospective teacher.

Reconstructs the exact source architecture, loads the checkpoint when present,
and runs **batched, boundary-safe, overlap-averaged** inference that keeps the
continuous probability (the `prob_list` the original notebook discarded).

If no checkpoint is available the loader degrades gracefully: it ingests the
surviving binary `predmeal_flag` as ``cgmacros_teacher_flag`` and leaves
``cgmacros_teacher_probability`` as NaN, tagging ``teacher_source`` so every
downstream consumer knows the continuous teacher signal is unavailable. It never
fabricates probabilities from untrained weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import TeacherConfig, TEACHER_CHECKPOINT_CANDIDATES
from .windows import add_segment_ids, build_windows, reconstruct_overlap_mean


# --------------------------------------------------------------------------- #
# Architecture (verbatim recovery of MealDetection/mealdetection.ipynb cell 9) #
# --------------------------------------------------------------------------- #
def build_source_model(cfg: TeacherConfig):
    import torch.nn as nn

    class CNN_Dilated_BiLSTM_Attn(nn.Module):
        def __init__(self, inputs_channel=1, c_out=16, h=64, dr=0.4):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(inputs_channel, c_out, 15, padding=7, dilation=1), nn.ReLU(),
                nn.Conv1d(c_out, c_out, 3, padding=2, dilation=2), nn.ReLU(),
                nn.Conv1d(c_out, c_out, 3, padding=4, dilation=4), nn.ReLU(),
                nn.BatchNorm1d(c_out),
                nn.Dropout(dr),
            )
            self.lstm = nn.LSTM(c_out, h, num_layers=2, batch_first=True,
                                bidirectional=True, dropout=dr)
            self.norm = nn.LayerNorm(2 * h)
            self.cls = nn.Linear(2 * h, 1)

        def forward(self, x):                 # x: (B, L, 1)
            x = self.cnn(x.transpose(1, 2))   # (B, C, L)
            x = x.transpose(1, 2)             # (B, L, C)
            x, _ = self.lstm(x)               # (B, L, 2h)
            x = self.norm(x)
            return self.cls(x).squeeze(-1)    # (B, L)

    return CNN_Dilated_BiLSTM_Attn(cfg.in_channels, cfg.c_out, cfg.hidden, cfg.dropout)


def find_checkpoint(explicit: str | Path | None = None) -> Path | None:
    """Return the first existing checkpoint path, or None."""
    cands = []
    if explicit:
        cands.append(Path(explicit))
    cands.extend(TEACHER_CHECKPOINT_CANDIDATES)
    for p in cands:
        if p.exists():
            return p
    return None


def load_teacher(cfg: TeacherConfig, checkpoint: str | Path | None = None):
    """Build the model and load weights with a tolerant state-dict matcher.

    Returns ``(model, device, checkpoint_path)`` or ``(None, None, None)`` when
    no checkpoint is available.
    """
    ckpt_path = find_checkpoint(checkpoint)
    if ckpt_path is None:
        return None, None, None

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_source_model(cfg).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Tolerant load: strip common prefixes, keep only matching shapes.
    model_sd = model.state_dict()
    cleaned = {}
    for k, v in state.items():
        kk = k
        for pre in ("module.", "model.", "net."):
            if kk.startswith(pre):
                kk = kk[len(pre):]
        if kk in model_sd and model_sd[kk].shape == v.shape:
            cleaned[kk] = v
    missing = set(model_sd) - set(cleaned)
    if missing:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} is incompatible with the recovered "
            f"architecture: {len(missing)} parameters unmatched "
            f"(e.g. {sorted(missing)[:4]}). Refusing to run a partially-loaded "
            f"teacher."
        )
    model.load_state_dict(cleaned)
    model.eval()
    return model, device, ckpt_path


# --------------------------------------------------------------------------- #
# Inference                                                                    #
# --------------------------------------------------------------------------- #
def _infer_segment(model, device, cgm_segment: np.ndarray, cfg: TeacherConfig):
    """Batched window inference on one contiguous segment.

    Returns ``(prob_mean, baseline_flag)`` both of length ``len(cgm_segment)``:

    * ``prob_mean`` — overlap **mean** of per-window probabilities (the corrected
      continuous teacher signal the original notebook discarded).
    * ``baseline_flag`` — the exact legacy rule: per-window binary votes
      (prob > threshold) reconstructed by **ratio voting** (positive where the
      fraction of covering windows voting 1 is >= ``baseline_ratio``).

    Segments shorter than ``seq_len`` cannot form a full window and return all
    NaN, matching the source notebook which skipped series shorter than the
    window (``if len(values) >= window_size``).
    """
    import torch

    seg_len = len(cgm_segment)
    if seg_len < cfg.seq_len:
        nan = np.full(seg_len, np.nan, dtype=np.float64)
        return nan, nan.copy()

    win, starts = build_windows(cgm_segment.astype(np.float32), cfg.seq_len, cfg.stride)
    probs_win = np.empty_like(win, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(win), cfg.batch_size):
            chunk = win[i:i + cfg.batch_size]
            xb = torch.from_numpy(chunk).float().unsqueeze(-1).to(device)  # (B,L,1)
            logits = model(xb)
            probs_win[i:i + cfg.batch_size] = torch.sigmoid(logits).cpu().numpy()

    prob_mean = reconstruct_overlap_mean(probs_win, starts, seg_len, cfg.seq_len)
    # Legacy ratio vote: mean of 0/1 window flags == fraction voting positive.
    flags_win = (probs_win > cfg.baseline_threshold).astype(np.float64)
    vote_frac = reconstruct_overlap_mean(flags_win, starts, seg_len, cfg.seq_len)
    baseline_flag = (vote_frac >= cfg.baseline_ratio).astype(np.float64)
    return prob_mean, baseline_flag


def _infer_segment_causal(model, device, cgm_segment: np.ndarray, cfg: TeacherConfig,
                          horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Strictly-causal teacher read for one segment.

    For each position ``t`` the probability is taken from the single window that
    *ends* at ``t`` (covering ``[t-seq_len+1, t]``), read at the window's last
    position. That window never contains a CGM sample with timestamp greater than
    ``t``, so the read is causal with respect to the anchor.

    Returns ``(causal, horizon_disjoint)``:
      * ``causal``            the trailing-window last-position probability.
      * ``horizon_disjoint``  the mean of the trailing ``horizon_steps`` causal
                              reads (all of which end at or before ``t``), a
                              smoother causal signal that still never enters the
                              forecast horizon.
    Positions with no full trailing window are NaN.
    """
    import torch

    n = len(cgm_segment)
    causal = np.full(n, np.nan, dtype=np.float64)
    if n < cfg.seq_len:
        return causal, causal.copy()
    from numpy.lib.stride_tricks import sliding_window_view
    seg = cgm_segment.astype(np.float32)
    if np.isnan(seg).any():
        seg = pd.Series(seg).interpolate(limit_direction="both").to_numpy(np.float32)
    win = sliding_window_view(seg, cfg.seq_len)        # (n-L+1, L); row i ends at i+L-1
    last = np.empty(len(win), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(win), cfg.batch_size):
            chunk = win[i:i + cfg.batch_size]
            xb = torch.from_numpy(np.ascontiguousarray(chunk)).float().unsqueeze(-1).to(device)
            logits = model(xb)                          # (B, L)
            last[i:i + cfg.batch_size] = torch.sigmoid(logits[:, -1]).cpu().numpy()
    causal[cfg.seq_len - 1:] = last                     # window ending at t -> position t
    # trailing mean over the finite causal tail (causal, no future contribution)
    tail = pd.Series(causal[cfg.seq_len - 1:])
    hd = np.full(n, np.nan, dtype=np.float64)
    hd[cfg.seq_len - 1:] = tail.rolling(horizon_steps, min_periods=1).mean().to_numpy()
    return causal, hd


def run_teacher_causal(
    df: pd.DataFrame,
    cfg: TeacherConfig,
    *,
    horizon_steps: int = 12,
    checkpoint: str | Path | None = None,
) -> pd.DataFrame:
    """Compute strictly-causal teacher probabilities for an AI-READI frame.

    Adds ``cgmacros_teacher_probability_causal`` and
    ``cgmacros_teacher_probability_horizon_disjoint``. Requires a real
    checkpoint (raises otherwise; we never fabricate a causal teacher).
    """
    work = df if "segment_id" in df.columns else add_segment_ids(
        df, resolution_min=cfg.target_resolution_min)
    work = work.sort_values(["participant_id", "ts"]).reset_index(drop=True)

    model, device, ckpt_path = load_teacher(cfg, checkpoint)
    if model is None:
        raise RuntimeError(
            "run_teacher_causal needs a real CGMacros checkpoint; none was found. "
            "The causal teacher ceiling cannot be reconstructed without it.")

    causal = np.full(len(work), np.nan, dtype=np.float64)
    hd = np.full(len(work), np.nan, dtype=np.float64)
    cgm_all = work["cgm_glucose"].to_numpy(dtype=np.float64)
    for _, sub in work.groupby("segment_id", sort=False):
        idx = sub.index.to_numpy()
        c, h = _infer_segment_causal(model, device, cgm_all[idx], cfg, horizon_steps)
        causal[idx] = c
        hd[idx] = h
    work["cgmacros_teacher_probability_causal"] = causal
    work["cgmacros_teacher_probability_horizon_disjoint"] = hd
    work["teacher_causal_source"] = f"causal_trailing_window:{ckpt_path.name}"
    return work


def run_teacher(
    df: pd.DataFrame,
    cfg: TeacherConfig,
    checkpoint: str | Path | None = None,
) -> pd.DataFrame:
    """Produce teacher columns for an AI-READI frame.

    Adds ``cgmacros_teacher_probability``, ``cgmacros_teacher_flag``,
    ``cgmacros_teacher_flag_baseline`` (threshold/ratio rule), ``segment_id``,
    and ``teacher_source``. Returns the frame sorted by participant/time.
    """
    work = add_segment_ids(
        df, resolution_min=cfg.target_resolution_min
    ).reset_index(drop=True)

    model, device, ckpt_path = load_teacher(cfg, checkpoint)

    if model is None:
        # --- Fallback: ingest surviving binary flag as the teacher flag. ---
        if "predmeal_flag" not in work.columns:
            raise RuntimeError(
                "No teacher checkpoint found AND no surviving `predmeal_flag` "
                "column to fall back on; teacher signal cannot be produced."
            )
        work["cgmacros_teacher_probability"] = np.nan
        work["cgmacros_teacher_flag"] = work["predmeal_flag"].fillna(0).astype(float)
        # Baseline == the surviving flag in this mode.
        work["cgmacros_teacher_flag_baseline"] = work["cgmacros_teacher_flag"]
        work["teacher_source"] = "precomputed_flag_artifact"
        return work

    # --- True inference path. ---
    prob = np.full(len(work), np.nan, dtype=np.float64)
    baseline = np.full(len(work), np.nan, dtype=np.float64)
    cgm_all = work["cgm_glucose"].to_numpy(dtype=np.float64)
    for _, sub in work.groupby("segment_id", sort=False):
        idx = sub.index.to_numpy()
        seg = cgm_all[idx]
        if np.isnan(seg).any():
            # Short intra-segment NaNs only (segments are gap-free by Δt); fill
            # for the model input but do not invent across true gaps.
            seg = pd.Series(seg).interpolate(limit_direction="both").to_numpy()
        p_mean, base = _infer_segment(model, device, seg, cfg)
        prob[idx] = p_mean
        baseline[idx] = base

    work["cgmacros_teacher_probability"] = prob
    # Primary flag thresholds the *reconstructed continuous probability* (the
    # corrected signal), not per-window votes.
    work["cgmacros_teacher_flag"] = np.where(
        np.isnan(prob), np.nan, (prob >= cfg.baseline_threshold).astype(float)
    )
    # Legacy threshold(0.40)+ratio(0.65) decision, retained as baseline only.
    work["cgmacros_teacher_flag_baseline"] = baseline
    work["teacher_source"] = f"checkpoint_inference:{ckpt_path.name}"
    return work
