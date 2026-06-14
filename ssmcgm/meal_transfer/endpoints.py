"""Block 1 endpoint set for the meal-feature evaluation gate.

The previous evaluation gated on whole-horizon ``MAE_60min``. Negative-control
work showed there is almost no timing-specific signal on that endpoint, because a
60-minute average washes the post-prandial excursion out. Meals act through the
shape of the rise (peak, hyperglycemia), not the smooth horizon mean, so the
gate is re-anchored on the post-prandial endpoints below.

Gate endpoints (a feature must move these):
  * ``MAE_eval_meal_window``      MAE on horizons whose target falls inside a
                                  decoded meal window.
  * ``peak_error_1h``             |max predicted - max true| over the 1 h horizon.
  * ``time_to_peak_error_min``    |argmax predicted - argmax true| * 5 min.
  * ``hyper_AUPRC_max1h``         AUPRC for ``max(y over <=12 steps) > 180``.
  * ``hyper_recall_at_p80_1h``    recall at precision 0.80 for the same label.

Report-only endpoints (shown, never gated):
  * ``MAE_30min``, ``MAE_60min``, ``hypo_brier_1h``, ``coverage_q10_q90``.

The hyperglycemia label is the per-anchor ``max`` of the true glucose across the
up-to-12 forecast horizons exceeding 180 mg/dL (not the single 60-minute point),
matching the clinical question "will this person go high in the next hour".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HYPER_MGDL = 180.0
HYPO_MGDL = 70.0
Z90_MINUS_Z10 = 2.5631031  # sigma ~= (q90 - q10) / this, normal interval scale
PEAK_HORIZON_STEPS = 12     # 1 hour at 5-minute cadence

# Direction: True means a LOWER value is better (errors); False means HIGHER is
# better (AUPRC, recall). Used by the selection gate to sign improvements.
GATE_ENDPOINTS = {
    "MAE_eval_meal_window": True,
    "peak_error_1h": True,
    "time_to_peak_error_min": True,
    "hyper_AUPRC_max1h": False,
    "hyper_recall_at_p80_1h": False,
}
REPORT_ENDPOINTS = ["MAE_30min", "MAE_60min", "hypo_brier_1h", "coverage_q10_q90"]


def _phi(z: np.ndarray) -> np.ndarray:
    from math import erf, sqrt
    return 0.5 * (1.0 + np.vectorize(lambda x: erf(x / sqrt(2.0)))(z))


def _per_horizon_hyper_prob(pred: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """P(glucose >= 180) at each (anchor, horizon) row from the corrected mean and
    the cached quantile spread (Gaussian interval approximation)."""
    sigma = np.maximum(spread / Z90_MINUS_Z10, 1e-6)
    return 1.0 - _phi((HYPER_MGDL - pred) / sigma)


def _per_horizon_hypo_prob(pred: np.ndarray, spread: np.ndarray) -> np.ndarray:
    sigma = np.maximum(spread / Z90_MINUS_Z10, 1e-6)
    return _phi((HYPO_MGDL - pred) / sigma)


def recall_at_precision(label: np.ndarray, score: np.ndarray, precision_floor: float = 0.80) -> float:
    """Max recall achievable at precision >= ``precision_floor``.

    Returns NaN if the label has a single class. Returns 0.0 if no operating
    point reaches the precision floor.
    """
    if len(label) == 0 or label.min() == label.max():
        return np.nan
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(label, score)
    ok = prec >= precision_floor
    if not ok.any():
        return 0.0
    return float(np.max(rec[ok]))


def anchor_hyper_label_score(df: pd.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-anchor hyperglycemia label/score for ``max(y over <=12 steps) > 180``.

    label    = 1 if the true glucose exceeds 180 at any horizon of the anchor.
    score    = 1 - prod_h (1 - p_h), the probability of at least one hyper step.
    Returns ``(participant_id, label, score)`` aligned to unique anchors.
    """
    spread = df["spread"].to_numpy(dtype=np.float64)
    p_h = _per_horizon_hyper_prob(pred, spread)
    tmp = pd.DataFrame({
        "participant_id": df["participant_id"].astype(str).to_numpy(),
        "anchor_ds": df["anchor_ds"].to_numpy(),
        "target": df["target"].to_numpy(dtype=np.float64),
        "log1m": np.log(np.clip(1.0 - p_h, 1e-12, 1.0)),
    })
    g = tmp.groupby(["participant_id", "anchor_ds"], sort=False)
    label = (g["target"].max() >= HYPER_MGDL).astype(int)
    score = 1.0 - np.exp(g["log1m"].sum())
    out = pd.DataFrame({"label": label, "score": score}).reset_index()
    return out["participant_id"].to_numpy(), out["label"].to_numpy(), out["score"].to_numpy()


def anchor_hypo_label_score(df: pd.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-anchor hypoglycemia (min over horizon < 70) label and probability."""
    spread = df["spread"].to_numpy(dtype=np.float64)
    p_h = _per_horizon_hypo_prob(pred, spread)
    tmp = pd.DataFrame({
        "participant_id": df["participant_id"].astype(str).to_numpy(),
        "anchor_ds": df["anchor_ds"].to_numpy(),
        "target": df["target"].to_numpy(dtype=np.float64),
        "log1m": np.log(np.clip(1.0 - p_h, 1e-12, 1.0)),
    })
    g = tmp.groupby(["participant_id", "anchor_ds"], sort=False)
    label = (g["target"].min() <= HYPO_MGDL).astype(int).to_numpy()
    score = (1.0 - np.exp(g["log1m"].sum())).to_numpy()
    return label, score


def forecast_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """All gate + report endpoints for one corrected forecast.

    ``df`` must carry ``participant_id, anchor_ds, horizon_step, target, q10, q90,
    spread`` and a boolean ``eval_in_meal_window`` column.
    """
    target = df["target"].to_numpy(dtype=np.float64)
    err = np.abs(pred - target)
    h = df["horizon_step"].to_numpy()
    out: dict[str, float] = {
        "MAE_30min": float(err[h == 6].mean()) if (h == 6).any() else np.nan,
        "MAE_60min": float(err[h == 12].mean()) if (h == 12).any() else np.nan,
        "MAE_all": float(err.mean()),
    }
    mw = df["eval_in_meal_window"].to_numpy(dtype=bool)
    out["MAE_eval_meal_window"] = float(err[mw].mean()) if mw.any() else np.nan
    out["eval_meal_window_rows"] = int(mw.sum())

    # Per-anchor peak and time-to-peak over the (up to) 1 h horizon.
    tmp = df[["participant_id", "anchor_ds", "horizon_step", "target"]].copy()
    tmp["pred"] = pred
    g = tmp.groupby(["participant_id", "anchor_ds"], sort=False)
    out["peak_error_1h"] = float((g["target"].max() - g["pred"].max()).abs().mean())
    idx_t = tmp.loc[g["target"].idxmax(), ["participant_id", "anchor_ds", "horizon_step"]]
    idx_p = tmp.loc[g["pred"].idxmax(), ["participant_id", "anchor_ds", "horizon_step"]]
    ttp = idx_t.merge(idx_p, on=["participant_id", "anchor_ds"], suffixes=("_t", "_p"))
    out["time_to_peak_error_min"] = float((ttp["horizon_step_t"] - ttp["horizon_step_p"]).abs().mean() * 5.0)

    # Per-anchor hyperglycemia (max over horizon > 180).
    _pid, label, score = anchor_hyper_label_score(df, pred)
    out["hyper_prevalence_1h"] = float(label.mean()) if len(label) else np.nan
    if len(label) and label.min() != label.max():
        from sklearn.metrics import average_precision_score, roc_auc_score
        out["hyper_AUPRC_max1h"] = float(average_precision_score(label, score))
        out["hyper_AUROC_max1h"] = float(roc_auc_score(label, score))
        out["hyper_recall_at_p80_1h"] = recall_at_precision(label, score, 0.80)
    else:
        out["hyper_AUPRC_max1h"] = out["hyper_AUROC_max1h"] = out["hyper_recall_at_p80_1h"] = np.nan

    # Report-only: hypoglycemia Brier (per-anchor min over horizon < 70).
    hlab, hsc = anchor_hypo_label_score(df, pred)
    out["hypo_brier_1h"] = float(((hsc - hlab) ** 2).mean()) if len(hlab) else np.nan

    # Report-only: nominal q10-q90 interval coverage of the realised target.
    q10 = df["q10"].to_numpy(dtype=np.float64)
    q90 = df["q90"].to_numpy(dtype=np.float64)
    out["coverage_q10_q90"] = float(((target >= q10) & (target <= q90)).mean())
    return out


# --------------------------------------------------------------------------- #
# Participant-resampling bootstrap support
# --------------------------------------------------------------------------- #
def _anchor_reduce_max(values: np.ndarray, aidx: np.ndarray, n_anchor: int) -> np.ndarray:
    out = np.full(n_anchor, -np.inf)
    np.maximum.at(out, aidx, values)
    return out


def _anchor_argmax_horizon(values: np.ndarray, peak: np.ndarray, aidx: np.ndarray,
                           h: np.ndarray, n_anchor: int) -> np.ndarray:
    """Horizon (steps) of the per-anchor max of ``values``; ties take the
    earliest horizon."""
    is_max = values >= peak[aidx] - 1e-9
    out = np.full(n_anchor, np.inf)
    np.minimum.at(out, aidx[is_max], h[is_max])
    return out


def _anchor_hyper_score(p_h: np.ndarray, aidx: np.ndarray, n_anchor: int) -> np.ndarray:
    log1m = np.log(np.clip(1.0 - p_h, 1e-12, 1.0))
    acc = np.zeros(n_anchor)
    np.add.at(acc, aidx, log1m)
    return 1.0 - np.exp(acc)


class AnchorCache:
    """Pre-computed per-row / per-anchor arrays for a fast participant bootstrap.

    Built once from the test frame. Each registered forecast is reduced to
    per-anchor peak / time-to-peak / hyper-score and per-row absolute error once;
    the bootstrap then only gathers participant slices and does numpy reductions
    (and one sort for AUPRC), so it scales to the full cohort.
    """

    def __init__(self, df: pd.DataFrame):
        pid_row = df["participant_id"].astype(str).to_numpy()
        h = df["horizon_step"].to_numpy()
        target = df["target"].to_numpy(dtype=np.float64)
        self.spread = df["spread"].to_numpy(dtype=np.float64)
        self.h = h
        self.target = target
        self.mw = df["eval_in_meal_window"].to_numpy(dtype=bool)
        self.h12 = h == 12
        self.h6 = h == 6

        # anchor id per row = (participant, anchor_ds)
        anchor_key = pd.factorize(
            pd.Series(list(zip(pid_row, df["anchor_ds"].to_numpy()))))[0]
        self.aidx = anchor_key.astype(np.int64)
        self.n_anchor = int(self.aidx.max()) + 1 if len(self.aidx) else 0
        self.peak_true = _anchor_reduce_max(target, self.aidx, self.n_anchor)
        self.ttp_true = _anchor_argmax_horizon(target, self.peak_true, self.aidx, h, self.n_anchor)
        self.hyper_label = (self.peak_true >= HYPER_MGDL).astype(int)
        # per-anchor participant (first row's pid for each anchor)
        anchor_pid = np.empty(self.n_anchor, dtype=object)
        anchor_pid[self.aidx] = pid_row  # last write wins, all equal within anchor
        self.anchor_pid = anchor_pid

        # group row and anchor indices by participant
        order = np.argsort(pid_row, kind="stable")
        self.rows_by_pid = {p: idx.to_numpy() for p, idx in
                            pd.Series(np.arange(len(df))).groupby(pid_row).groups.items()}
        self.anchors_by_pid = {p: idx.to_numpy() for p, idx in
                               pd.Series(np.arange(self.n_anchor)).groupby(anchor_pid).groups.items()}
        self.participants = np.array(sorted(self.rows_by_pid))
        del order
        self._row_abs_err: dict[str, np.ndarray] = {}
        self._peak_pred: dict[str, np.ndarray] = {}
        self._ttp_pred: dict[str, np.ndarray] = {}
        self._hyper_score: dict[str, np.ndarray] = {}

    def add_forecast(self, name: str, pred: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.float64)
        self._row_abs_err[name] = np.abs(pred - self.target)
        peak = _anchor_reduce_max(pred, self.aidx, self.n_anchor)
        self._peak_pred[name] = peak
        self._ttp_pred[name] = _anchor_argmax_horizon(pred, peak, self.aidx, self.h, self.n_anchor)
        p_h = _per_horizon_hyper_prob(pred, self.spread)
        self._hyper_score[name] = _anchor_hyper_score(p_h, self.aidx, self.n_anchor)

    @property
    def _forecasts(self) -> dict:
        return self._row_abs_err

    def _gather(self, participants) -> tuple[np.ndarray, np.ndarray]:
        rows = np.concatenate([self.rows_by_pid[p] for p in participants if p in self.rows_by_pid])
        anchors = np.concatenate([self.anchors_by_pid[p] for p in participants if p in self.anchors_by_pid])
        return rows, anchors

    def _endpoint(self, name: str, rows: np.ndarray, anchors: np.ndarray, endpoint: str) -> float:
        if endpoint == "MAE_30min":
            m = self.h6[rows]
            return float(self._row_abs_err[name][rows][m].mean()) if m.any() else np.nan
        if endpoint == "MAE_60min":
            m = self.h12[rows]
            return float(self._row_abs_err[name][rows][m].mean()) if m.any() else np.nan
        if endpoint == "MAE_eval_meal_window":
            m = self.mw[rows]
            return float(self._row_abs_err[name][rows][m].mean()) if m.any() else np.nan
        if endpoint == "peak_error_1h":
            return float(np.abs(self.peak_true[anchors] - self._peak_pred[name][anchors]).mean())
        if endpoint == "time_to_peak_error_min":
            return float(np.abs(self.ttp_true[anchors] - self._ttp_pred[name][anchors]).mean() * 5.0)
        if endpoint in ("hyper_AUPRC_max1h", "hyper_recall_at_p80_1h", "hyper_AUROC_max1h"):
            label = self.hyper_label[anchors]
            score = self._hyper_score[name][anchors]
            if len(label) == 0 or label.min() == label.max():
                return np.nan
            if endpoint == "hyper_recall_at_p80_1h":
                return recall_at_precision(label, score, 0.80)
            from sklearn.metrics import average_precision_score, roc_auc_score
            if endpoint == "hyper_AUPRC_max1h":
                return float(average_precision_score(label, score))
            return float(roc_auc_score(label, score))
        raise ValueError(endpoint)

    def point_endpoint(self, name: str, participants: np.ndarray, endpoint: str) -> float:
        rows, anchors = self._gather(participants)
        return self._endpoint(name, rows, anchors, endpoint)

    def _gain_value(self, combo, rows, anchors) -> float:
        model, control, endpoint = combo
        lower_better = GATE_ENDPOINTS.get(endpoint, True)
        mv = self._endpoint(model, rows, anchors, endpoint)
        cv = self._endpoint(control, rows, anchors, endpoint)
        if not (np.isfinite(mv) and np.isfinite(cv)):
            return np.nan
        return (cv - mv) if lower_better else (mv - cv)

    def batched_gains(self, combos: list[tuple], participants: np.ndarray,
                      n_boot: int, seed: int) -> dict:
        """Bootstrap many (model, control, endpoint) gains at once, gathering each
        participant resample a single time. Positive gain = model better."""
        parts = np.array([p for p in participants if p in self.rows_by_pid])
        if len(parts) == 0:
            return {c: {"gain": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                        "ci_excludes_zero": False} for c in combos}
        # unique (forecast name, endpoint) evaluations needed across all combos
        needed = set()
        for model, control, endpoint in combos:
            needed.add((model, endpoint))
            needed.add((control, endpoint))
        needed = list(needed)

        def gains_from_cache(cache_vals) -> dict:
            out_g = {}
            for model, control, endpoint in combos:
                lower_better = GATE_ENDPOINTS.get(endpoint, True)
                mv = cache_vals[(model, endpoint)]
                cv = cache_vals[(control, endpoint)]
                out_g[(model, control, endpoint)] = (
                    np.nan if not (np.isfinite(mv) and np.isfinite(cv))
                    else ((cv - mv) if lower_better else (mv - cv)))
            return out_g

        rows, anchors = self._gather(parts)
        cache_vals = {(nm, ep): self._endpoint(nm, rows, anchors, ep) for nm, ep in needed}
        point = gains_from_cache(cache_vals)
        rng = np.random.default_rng(seed)
        samples = {c: np.empty(n_boot, dtype=np.float64) for c in combos}
        n = len(parts)
        for b in range(n_boot):
            sel = parts[rng.integers(0, n, size=n)]
            rows, anchors = self._gather(sel)
            cache_vals = {(nm, ep): self._endpoint(nm, rows, anchors, ep) for nm, ep in needed}
            g = gains_from_cache(cache_vals)
            for c in combos:
                samples[c][b] = g[c]
        out = {}
        for c in combos:
            arr = samples[c][np.isfinite(samples[c])]
            if len(arr) < 2:
                out[c] = {"gain": point[c], "ci_lo": np.nan, "ci_hi": np.nan,
                          "ci_excludes_zero": False}
            else:
                lo, hi = np.percentile(arr, [2.5, 97.5])
                out[c] = {"gain": float(point[c]), "ci_lo": float(lo), "ci_hi": float(hi),
                          "ci_excludes_zero": bool(lo > 0)}
        return out

    def bootstrap_gain(self, model: str, control: str, endpoint: str,
                       participants: np.ndarray, n_boot: int, seed: int) -> dict:
        """Participant-resampling bootstrap of the improvement of ``model`` over
        ``control`` on ``endpoint``. Positive gain always means ``model`` better
        (sign follows the endpoint direction)."""
        lower_better = GATE_ENDPOINTS.get(endpoint, True)
        parts = np.array([p for p in participants if p in self.rows_by_pid])
        if len(parts) == 0:
            return {"gain": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "ci_excludes_zero": False}

        def gain_on(plist) -> float:
            rows, anchors = self._gather(plist)
            mv = self._endpoint(model, rows, anchors, endpoint)
            cv = self._endpoint(control, rows, anchors, endpoint)
            if not (np.isfinite(mv) and np.isfinite(cv)):
                return np.nan
            return (cv - mv) if lower_better else (mv - cv)

        point = gain_on(parts)
        rng = np.random.default_rng(seed)
        boots = np.empty(n_boot, dtype=np.float64)
        n = len(parts)
        for i in range(n_boot):
            boots[i] = gain_on(parts[rng.integers(0, n, size=n)])
        boots = boots[np.isfinite(boots)]
        if len(boots) < 2:
            return {"gain": point, "ci_lo": np.nan, "ci_hi": np.nan, "ci_excludes_zero": False}
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"gain": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                "ci_excludes_zero": bool(lo > 0)}
