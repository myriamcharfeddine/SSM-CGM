"""Block 1 selection gate, implemented as code (not prose).

A meal feature is *selected* only if, on the non-insulin primary cohort:

  1. it improves at least 2 of the five gate endpoints over the baseline
     residual model (A', the slope/level/clock model from Block 2), and
  2. every endpoint it claims as an improvement also beats the shuffled,
     time-shifted, and block-shuffled negative controls, with a
     participant-bootstrap 95% confidence interval that excludes zero in the
     improving direction.

A feature that is not leakage-free (for example the bidirectional teacher) is
never eligible to be selected, regardless of its numbers; it is recorded as a
flagged diagnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .endpoints import GATE_ENDPOINTS, AnchorCache

CONTROL_MODES = ["shuffle", "time_shift", "block_shuffle"]
MIN_IMPROVED_ENDPOINTS = 2


def _point_gain(cache: AnchorCache, model: str, baseline: str, endpoint: str,
                participants: np.ndarray) -> tuple[float, float, float, bool]:
    lower_better = GATE_ENDPOINTS[endpoint]
    mv = cache.point_endpoint(model, participants, endpoint)
    bv = cache.point_endpoint(baseline, participants, endpoint)
    if not (np.isfinite(mv) and np.isfinite(bv)):
        return mv, bv, np.nan, False
    gain = (bv - mv) if lower_better else (mv - bv)
    return mv, bv, gain, bool(gain > 0)


def evaluate_selection(cache: AnchorCache, baseline: str, feature_setups: list[str],
                       participants: np.ndarray, *, leaky_setups: set[str] | None = None,
                       n_boot: int = 300, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the gate for each feature setup against ``baseline`` on ``participants``.

    For each setup the cache must hold the registered real forecast under
    ``setup`` and the three controls under ``f"{setup}__{mode}"``.
    Returns ``(evidence, summary)``.
    """
    leaky_setups = leaky_setups or set()

    # 1) point improvements vs baseline (cheap) -> decide which endpoints to bootstrap
    point: dict[tuple, tuple] = {}
    combos: list[tuple] = []
    for setup in feature_setups:
        for endpoint in GATE_ENDPOINTS:
            mv, bv, gain, improves = _point_gain(cache, setup, baseline, endpoint, participants)
            point[(setup, endpoint)] = (mv, bv, gain, improves)
            if improves:  # only bootstrap controls for claimed improvements
                for mode in CONTROL_MODES:
                    if f"{setup}__{mode}" in cache._forecasts:
                        combos.append((setup, f"{setup}__{mode}", endpoint))

    # 2) one batched participant bootstrap for all needed control comparisons
    boot = cache.batched_gains(combos, participants, n_boot, seed) if combos else {}

    evidence_rows: list[dict] = []
    summary_rows: list[dict] = []
    for setup in feature_setups:
        n_passing = 0
        for endpoint in GATE_ENDPOINTS:
            mv, bv, gain, improves = point[(setup, endpoint)]
            rec = {"setup": setup, "endpoint": endpoint, "model_value": mv,
                   "baseline_value": bv, "gain_vs_baseline": gain,
                   "improves_over_baseline": improves}
            beats_all = improves
            for mode in CONTROL_MODES:
                b = boot.get((setup, f"{setup}__{mode}", endpoint),
                             {"gain": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                              "ci_excludes_zero": False})
                rec[f"gain_vs_{mode}"] = b["gain"]
                rec[f"ci_lo_vs_{mode}"] = b["ci_lo"]
                rec[f"ci_hi_vs_{mode}"] = b["ci_hi"]
                rec[f"beats_{mode}"] = bool(b["ci_excludes_zero"])
                beats_all = beats_all and bool(b["ci_excludes_zero"])
            rec["beats_all_controls"] = bool(beats_all)
            rec["passing_endpoint"] = bool(improves and beats_all)
            n_passing += int(rec["passing_endpoint"])
            evidence_rows.append(rec)

        leaky = setup in leaky_setups
        selected = bool(n_passing >= MIN_IMPROVED_ENDPOINTS and not leaky)
        summary_rows.append({
            "setup": setup, "n_passing_endpoints": int(n_passing),
            "min_required": MIN_IMPROVED_ENDPOINTS, "leakage_free": not leaky,
            "selected": selected,
        })
    return pd.DataFrame(evidence_rows), pd.DataFrame(summary_rows)
