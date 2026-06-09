"""Persistence and simple baselines for AI-READI diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .diagnostics import ANCHOR_KEY, Q50, aggregate_metrics, forecast_only, save_table, write_json


def current_glucose_by_anchor(streams, anchor_df: pd.DataFrame) -> Dict[Tuple[str, int, int], float]:
    wanted = {
        (str(r.participant_id), int(r.segment_id), int(r.anchor_time_idx))
        for r in anchor_df[list(ANCHOR_KEY)].drop_duplicates().itertuples(index=False)
    }
    by_stream: Dict[Tuple[str, int], set] = {}
    for pid, seg, t in wanted:
        by_stream.setdefault((pid, seg), set()).add(t)
    current: Dict[Tuple[str, int, int], float] = {}
    for stream in streams:
        skey = (str(stream.participant_id), int(stream.segment_id))
        if skey not in by_stream:
            continue
        target = stream.target.detach().cpu().numpy() if hasattr(stream.target, "detach") else np.asarray(stream.target)
        time_idx = stream.time_idx.detach().cpu().numpy() if hasattr(stream.time_idx, "detach") else np.asarray(stream.time_idx)
        pos_by_time = {int(t): i for i, t in enumerate(time_idx.tolist())}
        for anchor_t in by_stream[skey]:
            pos = pos_by_time.get(int(anchor_t))
            if pos is not None and 0 <= pos < len(target):
                current[(skey[0], skey[1], int(anchor_t))] = float(target[pos])
    missing = wanted - set(current)
    if missing:
        raise RuntimeError(f"Could not map current glucose for {len(missing)} anchors; first missing={next(iter(missing))}")
    return current


def persistence_baseline(predictions: pd.DataFrame, streams, metrics_dir) -> List[str]:
    metrics_dir = Path(metrics_dir)
    df = forecast_only(predictions)
    anchor_df = df[ANCHOR_KEY].drop_duplicates()
    current = current_glucose_by_anchor(streams, anchor_df)
    work = df.copy()
    keys = list(zip(work["participant_id"].astype(str), work["segment_id"].astype(int), work["anchor_time_idx"].astype(int)))
    work["persistence_pred"] = [current[k] for k in keys]

    pwork = work.copy()
    pwork[Q50] = pwork["persistence_pred"]
    persistence = aggregate_metrics(pwork, pred_col=Q50, lo_col="", hi_col="")
    model = aggregate_metrics(df)
    terminal = pwork[pwork["horizon_minutes"] == pwork["horizon_minutes"].max()]
    terminal_model = df[df["horizon_minutes"] == df["horizon_minutes"].max()]
    payload = {
        "comparison_scope": "forecast_only rows, exact same participant/segment/anchor/horizon rows as model",
        "n_rows": int(len(work)),
        "n_anchors": int(anchor_df.shape[0]),
        "terminal_horizon_minutes": int(work["horizon_minutes"].max()) if not work.empty else None,
        "model": model,
        "persistence": persistence,
        "model_terminal_mae": aggregate_metrics(terminal_model).get("mae"),
        "persistence_terminal_mae": aggregate_metrics(terminal, pred_col=Q50, lo_col="", hi_col="").get("mae"),
        "delta_mae_model_minus_persistence": model.get("mae") - persistence.get("mae"),
    }
    json_path = metrics_dir / "persistence_baseline.json"
    write_json(json_path, payload)

    rows = []
    for (step, minutes), g in work.groupby(["horizon_step", "horizon_minutes"], dropna=False):
        gp = g.copy()
        gp[Q50] = gp["persistence_pred"]
        pmetrics = aggregate_metrics(gp, pred_col=Q50, lo_col="", hi_col="")
        mmetrics = aggregate_metrics(df[(df["horizon_step"] == step) & (df["horizon_minutes"] == minutes)])
        rows.append({
            "horizon_step": int(step),
            "horizon_minutes": int(minutes),
            "model_mae": mmetrics.get("mae"),
            "persistence_mae": pmetrics.get("mae"),
            "model_rmse": mmetrics.get("rmse"),
            "persistence_rmse": pmetrics.get("rmse"),
            "model_bias": mmetrics.get("bias"),
            "persistence_bias": pmetrics.get("bias"),
            "model_tir_predicted": mmetrics.get("tir_predicted"),
            "persistence_tir_predicted": pmetrics.get("tir_predicted"),
            "model_tir_gap": mmetrics.get("tir_gap"),
            "persistence_tir_gap": pmetrics.get("tir_gap"),
        })
    horizon_path = metrics_dir / "persistence_by_horizon.csv"
    save_table(pd.DataFrame(rows).sort_values("horizon_step"), horizon_path)
    return [str(json_path), str(horizon_path)]
