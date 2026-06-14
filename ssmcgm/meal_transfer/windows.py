"""Boundary-safe segmentation and windowing for AI-READI CGM.

Windows are built **only within contiguous, gap-free segments of a single
participant**. A new segment starts whenever the sampling gap departs from the
nominal 5-minute cadence (the source model never saw cross-gap context, and a
window straddling a recording gap is physiologically meaningless).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_segment_ids(
    df: pd.DataFrame,
    *,
    participant_col: str = "participant_id",
    time_col: str = "ts",
    resolution_min: int = 5,
    gap_tol_min: float = 1.0,
) -> pd.DataFrame:
    """Add a ``segment_id`` that is constant over contiguous same-cadence runs.

    Returns a sorted copy with an integer ``segment_id`` unique across the frame.
    """
    out = df.sort_values([participant_col, time_col]).reset_index(drop=True).copy()
    dt_min = out.groupby(participant_col)[time_col].diff().dt.total_seconds() / 60.0
    # New segment when participant changes (NaN diff) or the gap != nominal.
    new_seg = dt_min.isna() | (np.abs(dt_min - resolution_min) > gap_tol_min)
    out["segment_id"] = new_seg.cumsum().astype(np.int64)
    return out


def iter_segments(df: pd.DataFrame, segment_col: str = "segment_id"):
    """Yield ``(segment_id, sub_df)`` for each contiguous segment, order kept."""
    for seg_id, sub in df.groupby(segment_col, sort=False):
        yield seg_id, sub


def build_windows(
    values: np.ndarray,
    seq_len: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(windows, starts)`` for one 1-D series.

    If the segment is shorter than ``seq_len`` it is left-padded with its first
    value to a single window and the start index is 0; the caller masks padded
    positions out during reconstruction. ``windows`` has shape
    ``(n_windows, seq_len)``; ``starts`` are window start offsets within the
    segment.
    """
    n = len(values)
    if n < seq_len:
        pad = np.full(seq_len - n, values[0], dtype=values.dtype)
        win = np.concatenate([pad, values])[None, :]
        return win, np.array([0], dtype=np.int64)

    starts = np.arange(0, n - seq_len + 1, stride, dtype=np.int64)
    idx = starts[:, None] + np.arange(seq_len)[None, :]
    return values[idx], starts


def reconstruct_overlap_mean(
    window_outputs: np.ndarray,
    starts: np.ndarray,
    seg_len: int,
    seq_len: int,
) -> np.ndarray:
    """Average per-step window outputs back onto a segment of length ``seg_len``.

    Mirrors ``_reconstruct_prob_average`` from the source notebook but is
    vectorised, handles arbitrary stride, and is the *only* reconstruction used
    for continuous probabilities (no information is thrown away).
    Positions never covered by a window are returned as NaN.
    """
    summed = np.zeros(seg_len, dtype=np.float64)
    counts = np.zeros(seg_len, dtype=np.float64)
    for w, s in zip(window_outputs, starts):
        # For a short segment the single window was left-padded; align its tail.
        if seg_len < seq_len:
            summed += w[-seg_len:]
            counts += 1.0
        else:
            summed[s:s + seq_len] += w
            counts[s:s + seq_len] += 1.0
    out = np.full(seg_len, np.nan, dtype=np.float64)
    nz = counts > 0
    out[nz] = summed[nz] / counts[nz]
    return out
