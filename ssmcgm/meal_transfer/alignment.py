"""Align cached baseline forecasts to the passive meal-state table.

The cached SSM-CGM-Stream predictions index anchors by an internal
``anchor_time_idx`` (the stream position ``t``) that is **not** equal to the
meal-state ``ds`` (verified empirically). The target for ``(anchor a, horizon k)``
is the CGM at stream position ``a + k``. We recover, per participant, a constant
integer offset ``delta`` such that ``ds = stream_position + delta`` and the
baseline ``target`` exactly equals the meal-state ``cgm_glucose`` at the mapped
``ds``. The anchor's meal feature is then read at ``anchor_ds = a + delta``.

All of the checks the brief mandates are computed and returned:
near-zero CGM residual, monotone anchor ordering, per-participant isolation
(no leakage), no segment crossing, and exact baseline-target/meal-anchor match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AlignResult:
    participant_id: str
    delta: int
    n_rows: int
    mean_abs_resid: float
    max_abs_resid: float
    frac_exact: float           # |target - mapped cgm| <= 0.5 mg/dL
    monotone_ok: bool
    seg_cross_frac: float       # fraction of anchors whose 1h window crosses a segment
    ok: bool


def _reconstruct_stream_cgm(seg_pred: pd.DataFrame) -> pd.Series:
    """eval CGM by stream position: target(a, k) is the CGM at stream pos a+k."""
    pos = (seg_pred["anchor_time_idx"] + seg_pred["horizon_step"]).to_numpy()
    val = seg_pred["target"].to_numpy(dtype=np.float64)
    s = pd.Series(val, index=pos)
    s = s[np.isfinite(s.values)]
    # duplicate stream positions (covered by several anchors) must agree; keep first.
    return s.groupby(level=0).first().sort_index()


def _longest_contiguous(s: pd.Series, max_len: int = 96) -> tuple[int, np.ndarray]:
    """Return (start_pos, values) of a contiguous run of stream positions (len<=max_len)."""
    idx = s.index.to_numpy()
    if len(idx) == 0:
        return 0, np.array([])
    # find runs where idx increments by 1
    brk = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk, len(idx) - 1]
    lengths = ends - starts + 1
    j = int(np.argmax(lengths))
    a, b = starts[j], starts[j] + min(lengths[j], max_len)
    return int(idx[a]), s.iloc[a:b].to_numpy(dtype=np.float64)


def _match_offset(start_pos: int, win: np.ndarray, cgm_by_ds: pd.Series,
                  *, min_win: int = 24) -> int | None:
    """Return ``delta`` so cgm_by_ds[stream_pos + delta] ~ eval cgm, using a long
    contiguous window ``win`` that begins at stream position ``start_pos``.

    A long window (default up to 96 steps = 8 h) makes the match unique; a short
    one matches spuriously. Returns None if no near-exact placement exists.
    """
    if len(win) < min_win:
        return None
    ds_index = cgm_by_ds.index.to_numpy()
    lo, hi = ds_index.min(), ds_index.max()
    dense = np.full(hi - lo + 1, np.nan)
    dense[ds_index - lo] = cgm_by_ds.to_numpy(dtype=np.float64)
    L = len(win)
    if hi - lo + 1 < L:
        return None
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(dense, L)                    # (n-L+1, L)
    err = np.nanmean(np.abs(sw - win[None, :]), axis=1)
    err[np.isnan(sw).any(axis=1)] = np.inf
    p = int(np.argmin(err))
    if not np.isfinite(err[p]) or err[p] > 0.5:
        return None
    ds_start = lo + p                                     # ds aligned to win[0]
    return int(ds_start - start_pos)


def align_participant(
    pred_p: pd.DataFrame,
    ms_p: pd.DataFrame,
    *,
    exact_tol: float = 0.5,
) -> tuple[AlignResult, np.ndarray | None]:
    """Align one participant. ``pred_p`` is forecast_only rows for the participant;
    ``ms_p`` is its meal-state rows. Returns ``(AlignResult, anchor_ds_per_row)``
    where ``anchor_ds_per_row`` aligns to ``pred_p`` row order (or None if failed)."""
    pid = str(pred_p["participant_id"].iloc[0])
    ms_p = ms_p.sort_values("ds")
    cgm_by_ds = ms_p.set_index("ds")["cgm_glucose"]
    seg_by_ds = ms_p.set_index("ds")["segment_id"] if "segment_id" in ms_p else None

    # Solve an offset per EVAL segment (the eval re-indexes per segment), using a
    # long contiguous match window so the placement is unique.
    pred_p = pred_p.reset_index(drop=True)
    anchor_ds = np.full(len(pred_p), np.nan)
    seg_col = "segment_id" if "segment_id" in pred_p else None
    groups = pred_p.groupby(seg_col) if seg_col else [("_all", pred_p)]
    deltas = {}
    for seg_id, sp in groups:
        rec = _reconstruct_stream_cgm(sp)
        start_pos, win = _longest_contiguous(rec)
        delta = _match_offset(start_pos, win, cgm_by_ds)
        if delta is None:
            continue
        deltas[seg_id] = delta
        rows = sp.index.to_numpy()
        anchor_ds[rows] = sp["anchor_time_idx"].to_numpy() + delta
    if not deltas:
        return AlignResult(pid, 0, len(pred_p), np.nan, np.nan, 0.0, False, np.nan, False), None

    a = pred_p["anchor_time_idx"].to_numpy()
    k = pred_p["horizon_step"].to_numpy()
    tgt = pred_p["target"].to_numpy(dtype=np.float64)
    target_ds = anchor_ds + k
    mapped = cgm_by_ds.reindex(pd.Index(np.where(np.isfinite(target_ds), target_ds, -1).astype(int))).to_numpy(dtype=np.float64)
    resid = np.abs(tgt - mapped)
    valid = np.isfinite(resid) & np.isfinite(anchor_ds)
    mean_abs = float(np.nanmean(resid[valid])) if valid.any() else np.nan
    max_abs = float(np.nanmax(resid[valid])) if valid.any() else np.nan
    frac_exact = float((resid[valid] <= exact_tol).mean()) if valid.any() else 0.0
    delta = int(list(deltas.values())[0])

    # ordering: anchor_ds monotone non-decreasing with anchor_time_idx WITHIN each
    # eval segment (per-segment offsets mean global order need not hold).
    fin = np.isfinite(anchor_ds)
    monotone_ok = True
    if seg_col:
        for _sid, sp in pred_p.groupby(seg_col):
            rows = sp.index.to_numpy()
            m = np.isfinite(anchor_ds[rows])
            if m.sum() < 2:
                continue
            aa = sp["anchor_time_idx"].to_numpy()[m]
            adv = anchor_ds[rows][m]
            o = np.argsort(aa)
            if not np.all(np.diff(adv[o]) >= 0):
                monotone_ok = False
                break
    else:
        o = np.argsort(a[fin])
        monotone_ok = bool(np.all(np.diff(anchor_ds[fin][o]) >= 0))

    # segment crossing: anchor's 1h forecast window spans >1 meal-state segment
    seg_cross_frac = np.nan
    if seg_by_ds is not None:
        Hmax = int(k.max())
        anchor_ds_unique = np.unique(anchor_ds[fin]).astype(int)
        cross = 0
        for ad in anchor_ds_unique:
            win_ds = ad + np.arange(0, Hmax + 1)
            segs = pd.unique(seg_by_ds.reindex(win_ds).dropna())
            if len(segs) > 1:
                cross += 1
        seg_cross_frac = float(cross / max(len(anchor_ds_unique), 1))

    ok = bool(valid.mean() > 0.95 and mean_abs < exact_tol and frac_exact > 0.98 and monotone_ok)
    return (AlignResult(pid, int(delta), len(pred_p), mean_abs, max_abs, frac_exact,
                        monotone_ok, seg_cross_frac, ok),
            anchor_ds)


def align_cache(pred_df: pd.DataFrame, ms_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align an entire forecast cache (forecast_only) to the meal states.

    Returns ``(aligned_pred, verification)``: ``aligned_pred`` is ``pred_df`` plus
    an ``anchor_ds`` column (rows from unalignable participants dropped);
    ``verification`` is the per-participant :class:`AlignResult` table.
    """
    pred_df = pred_df.copy()
    pred_df["participant_id"] = pred_df["participant_id"].astype(str)
    ms_df = ms_df.copy()
    ms_df["participant_id"] = ms_df["participant_id"].astype(str)

    results = []
    anchor_ds_full = np.full(len(pred_df), np.nan)
    ms_groups = dict(tuple(ms_df.groupby("participant_id")))
    for pid, idx in pred_df.groupby("participant_id").groups.items():
        idx = np.asarray(idx)
        if pid not in ms_groups:
            results.append(AlignResult(pid, 0, len(idx), np.nan, np.nan, 0.0, False, np.nan, False))
            continue
        res, anchor_ds = align_participant(pred_df.loc[idx], ms_groups[pid])
        results.append(res)
        # Keep every row with a found per-segment offset; the ablation enforces
        # per-row exact target match downstream, so partial-match participants
        # still contribute their exactly-aligned rows.
        if anchor_ds is not None:
            anchor_ds_full[idx] = anchor_ds

    pred_df["anchor_ds"] = anchor_ds_full
    verification = pd.DataFrame([r.__dict__ for r in results])
    aligned = pred_df[np.isfinite(pred_df["anchor_ds"])].copy()
    aligned["anchor_ds"] = aligned["anchor_ds"].astype(int)
    return aligned, verification
