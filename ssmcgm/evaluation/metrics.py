"""Metrics over a streaming-prediction table (spec §13).

Everything is computed from one tidy long-form ``DataFrame`` where each row is a
single ``(participant, forecast anchor, horizon step)`` triple — the schema emitted by
:func:`ssmcgm.evaluation.streaming.evaluate_streams`:

    participant_id, split, anchor_time_idx, steps_since_start, hours_since_start,
    horizon_step (1..H), horizon_minutes, q10, q50, q90, target, observed,
    warmup_hours, scenario_mode, segment_id, <optional subgroup columns>

Quantile columns are named ``q{level*100:02.0f}`` (e.g. ``q10``/``q50``/``q90``). The
median column (``q50``) is the point forecast; ``(q10, q90)`` is the 80% interval.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_QUANTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)


def qcol(level: float) -> str:
    """Column name for a quantile level (0.1 -> ``"q10"``)."""
    return f"q{round(level * 100):02d}"


# ---------------------------------------------------------------------------
# core point + probabilistic aggregation
# ---------------------------------------------------------------------------
def _pinball(target: np.ndarray, pred: np.ndarray, level: float) -> np.ndarray:
    e = target - pred
    return np.maximum(level * e, (level - 1.0) * e)


def aggregate(df: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> Dict[str, float]:
    """Point + probabilistic metrics over the (already-filtered) rows in ``df``."""
    if len(df) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "medae": np.nan, "bias": np.nan,
                "pinball": np.nan, "coverage80": np.nan, "interval_width80": np.nan,
                "crossing_rate": np.nan}
    qs = sorted(quantiles)
    med = qcol(min(qs, key=lambda q: abs(q - 0.5)))      # nearest 0.5, never aliased onto a bound
    # the "80" interval is the 0.1/0.9 pair when present (the canonical 80% band the
    # whole report keys on); fall back to the outer quantiles otherwise.
    lo, hi = qcol(0.1 if 0.1 in quantiles else qs[0]), qcol(0.9 if 0.9 in quantiles else qs[-1])
    t = df["target"].to_numpy(dtype="float64")
    p = df[med].to_numpy(dtype="float64")
    e = p - t
    pinball = np.mean([_pinball(t, df[qcol(q)].to_numpy(dtype="float64"), q).mean() for q in qs])
    lo_v, hi_v = df[lo].to_numpy(dtype="float64"), df[hi].to_numpy(dtype="float64")
    inside = ((t >= lo_v) & (t <= hi_v)).mean()
    # quantile crossing: any out-of-order pair across the sorted levels
    cols = np.stack([df[qcol(q)].to_numpy(dtype="float64") for q in qs], axis=1)
    crossing = (np.diff(cols, axis=1) < -1e-6).any(axis=1).mean()
    return {
        "n": int(len(df)),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "medae": float(np.median(np.abs(e))),
        "bias": float(e.mean()),
        "pinball": float(pinball),
        "coverage80": float(inside),
        "interval_width80": float((hi_v - lo_v).mean()),
        "crossing_rate": float(crossing),
    }


def overall_metrics(df: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> Dict[str, float]:
    """Spec §13.1/§13.3 headline metrics over all scored rows."""
    return aggregate(df, quantiles)


def metrics_by(
    df: pd.DataFrame, by: Sequence[str], quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Aggregate metrics grouped by one or more columns (horizon, warm-up, subgroup…)."""
    by = [c for c in by if c in df.columns]
    if not by:
        return pd.DataFrame()
    rows = []
    for key, g in df.groupby(by, dropna=False, observed=False):
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(aggregate(g, quantiles))
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(by).reset_index(drop=True)


def metrics_by_horizon(df: pd.DataFrame, quantiles=DEFAULT_QUANTILES) -> pd.DataFrame:
    """Spec §13.2 — per-horizon (5,10,…,60 min) forecasting + calibration."""
    return metrics_by(df, ["horizon_step", "horizon_minutes"], quantiles)


def metrics_by_warmup(df: pd.DataFrame, quantiles=DEFAULT_QUANTILES) -> pd.DataFrame:
    """Spec §13.5 / §11 — the central warm-up personalization curve."""
    return metrics_by(df, ["warmup_hours"], quantiles)


def metrics_by_participant(df: pd.DataFrame, quantiles=DEFAULT_QUANTILES) -> pd.DataFrame:
    return metrics_by(df, ["participant_id"], quantiles)


def warmup_threshold_curve(
    df: pd.DataFrame, warmup_hours: Sequence[float] = (0, 1, 6, 12, 24, 48),
    quantiles=DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Spec §11 — error when *only* scoring anchors observed for ≥ W hours.

    For each warm-up duration ``W`` we aggregate the anchors with
    ``hours_since_start >= W`` (i.e. the model has already observed ≥ W h of the new
    person). A strong drop as ``W`` grows means the recurrent state is personalizing
    online; good performance at ``W=0`` means the static ``h0`` cold-starts well.
    """
    if "hours_since_start" not in df.columns:
        return pd.DataFrame()
    rows = []
    for w in sorted(set(warmup_hours)):
        rec = {"warmup_hours": float(w)}
        rec.update(aggregate(df[df["hours_since_start"] >= w], quantiles))
        rows.append(rec)
    return pd.DataFrame(rows)


def time_since_start_curve(
    df: pd.DataFrame, edges: Sequence[float] = (0, 1, 6, 12, 24, 48, 1e9),
    quantiles=DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Spec §11 figure — error *within* each elapsed-time bucket (not cumulative)."""
    if "hours_since_start" not in df.columns or len(df) == 0:
        return pd.DataFrame()
    edges = sorted(set(edges))
    labels = [f"[{edges[i]:g},{edges[i + 1]:g})h" for i in range(len(edges) - 1)]
    binned = df.copy()
    binned["hours_bucket"] = pd.cut(binned["hours_since_start"], bins=edges, right=False,
                                    labels=labels, include_lowest=True)
    out = metrics_by(binned, ["hours_bucket"], quantiles)
    # metrics_by's DataFrame(rows)+sort_values degrades the ordered Categorical to an
    # object column sorted lexicographically ("[6,12)h" after "[48,..)h"); restore the
    # intended elapsed-time order so the figure x-axis is monotone.
    if len(out) and "hours_bucket" in out.columns:
        out["hours_bucket"] = pd.Categorical(out["hours_bucket"], categories=labels, ordered=True)
        out = out.sort_values("hours_bucket").reset_index(drop=True)
    return out


def calibration_by_horizon(df: pd.DataFrame, quantiles=DEFAULT_QUANTILES) -> pd.DataFrame:
    """Spec §13.3 — coverage / interval width / crossing-rate per horizon."""
    tab = metrics_by_horizon(df, quantiles)
    keep = ["horizon_step", "horizon_minutes", "n", "coverage80", "interval_width80", "crossing_rate"]
    return tab[[c for c in keep if c in tab.columns]]


# ---------------------------------------------------------------------------
# clinical (spec §13.4) — a deployable subset
# ---------------------------------------------------------------------------
def _binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and prec + rec > 0) else np.nan
    return {"n_pos": int(y_true.sum()), "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def clinical_metrics(
    df: pd.DataFrame, *, hypo_thresh: float = 70.0, hyper_thresh: float = 180.0,
    quantiles=DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Hypo/hyper event prediction + regime-specific MAE (median forecast)."""
    if len(df) == 0:
        return pd.DataFrame()
    med = qcol(0.5)
    t = df["target"].to_numpy(dtype="float64")
    p = df[med].to_numpy(dtype="float64")
    rows = []
    hypo = _binary_scores(t < hypo_thresh, p < hypo_thresh)
    hypo.update({"metric": "hypoglycemia_<70", **_regime_mae(t, p, t < hypo_thresh)})
    rows.append(hypo)
    hyper = _binary_scores(t > hyper_thresh, p > hyper_thresh)
    hyper.update({"metric": "hyperglycemia_>180", **_regime_mae(t, p, t > hyper_thresh)})
    rows.append(hyper)
    in_range = (t >= hypo_thresh) & (t <= hyper_thresh)
    rows.append({"metric": "in_range_70_180", "n_pos": int(in_range.sum()),
                 **_regime_mae(t, p, in_range)})
    out = pd.DataFrame(rows)
    front = ["metric", "n_pos", "precision", "recall", "f1", "regime_mae", "regime_n"]
    return out[[c for c in front if c in out.columns] + [c for c in out.columns if c not in front]]


def _regime_mae(t: np.ndarray, p: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if mask.sum() == 0:
        return {"regime_mae": np.nan, "regime_n": 0}
    return {"regime_mae": float(np.abs(p[mask] - t[mask]).mean()), "regime_n": int(mask.sum())}


def bias_audit(df: pd.DataFrame, transform_mode: str = "") -> pd.DataFrame:
    """Cold-start bias summary (spec §1/§6): level tracking vs systematic offset.

    Reports mean target / mean median / bias, the per-participant predicted-vs-target
    level correlation (ranking preserved?), and the per-participant bias spread — the
    signature that distinguishes a normalization center mismatch from random error."""
    if len(df) == 0:
        return pd.DataFrame()
    med = qcol(0.5)
    g = df.groupby("participant_id").agg(t=("target", "mean"), p=(med, "mean"))
    pbias = (g["p"] - g["t"])
    rec = {
        "transform_mode": transform_mode,
        "mean_target_mgdl": float(df["target"].mean()),
        "mean_pred_q50_mgdl": float(df[med].mean()),
        "overall_bias_mgdl": float((df[med] - df["target"]).mean()),
        "participant_level_corr": float(g["t"].corr(g["p"])) if len(g) > 1 else float("nan"),
        "participant_bias_mean": float(pbias.mean()),
        "participant_bias_std": float(pbias.std()),
        "participant_bias_min": float(pbias.min()),
        "participant_bias_max": float(pbias.max()),
        "n_participants": int(len(g)),
    }
    return pd.DataFrame([rec])


def time_in_range_error(df: pd.DataFrame, *, lo: float = 70.0, hi: float = 180.0) -> float:
    """|TIR(forecast median) − TIR(observed)| over the scored rows."""
    if len(df) == 0:
        return np.nan
    med = qcol(0.5)
    tir_pred = ((df[med] >= lo) & (df[med] <= hi)).mean()
    tir_true = ((df["target"] >= lo) & (df["target"] <= hi)).mean()
    return float(abs(tir_pred - tir_true))


# ---------------------------------------------------------------------------
# production metrics (spec §13.6)
# ---------------------------------------------------------------------------
def stream_state_bytes(state) -> int:
    """Total bytes held by a :class:`~ssmcgm.stream.state.StreamState` (spec §13.6/§17.6).

    Sums ``numel * element_size`` over the per-layer SSM states, conv buffers and
    ``last_output`` — the memory footprint of one active participant stream.
    """
    total = 0

    def _add(t):
        nonlocal total
        if t is not None and hasattr(t, "numel"):
            total += int(t.numel()) * int(t.element_size())

    for s in getattr(state, "layer_states", []) or []:
        _add(s)
    for c in getattr(state, "conv_states", None) or []:
        _add(c)
    _add(getattr(state, "last_output", None))
    return total


def latency_table(timings_ms: Dict[str, Sequence[float]]) -> pd.DataFrame:
    """Summarize per-call latency samples (ms) into mean/median/p95 rows (spec §13.6)."""
    rows = []
    for name, xs in timings_ms.items():
        a = np.asarray(list(xs), dtype="float64")
        if a.size == 0:
            continue
        rows.append({"op": name, "n": int(a.size), "mean_ms": float(a.mean()),
                     "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
                     "throughput_per_s": float(1000.0 / a.mean()) if a.mean() > 0 else np.nan})
    return pd.DataFrame(rows)
