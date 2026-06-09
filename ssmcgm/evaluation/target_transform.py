"""Production-safe target transforms for SSM-CGM-Stream cold-start (spec: fix bias).

A held-out participant has no learned per-participant target center at deployment, so
the default `GroupNormalizer(groups=["USUBJID"])` falls back to a *global* center and
the forecast is systematically biased toward the cohort mean (observed: ~-20 mg/dL on
held-out T1DEXI participants whose true mean differs from the global center). These
transforms make the target normalization compatible with new people:

* ``group``  — legacy per-participant GroupNormalizer inverse (``transform_output``).
  Fine for *seen* participants; **biased** for held-out cold start. Not recommended.
* ``global`` — a single train-fit center/scale applied to everyone (no per-person
  center). Leakage-safe; removes the train↔test center mismatch. Uses the model's
  ``transform_output`` with the (now global) ``target_scale``.
* ``residual_current`` (production default) — the model predicts the *deviation from
  the current observed glucose*: ``ŷ_{t+h} = anchor_t + residual̂``, where ``anchor_t``
  is the CGM known at the forecast anchor. Zero-bias by construction, leakage-safe
  (current glucose is observed), and the natural "predict the change" framing. The
  residual is scaled by a train-fit per-horizon robust scale so the decoder output is
  well-conditioned.

Everything downstream computes metrics in **mg/dL**. ``predict_mgdl`` is the single
inverse used by both the streaming evaluation and the BPTT trainer, so train and eval
agree by construction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

MODES = ("group", "global", "residual_current")


@dataclass
class TargetTransform:
    """Inverse-transform from the decoder's raw output to mg/dL (production-safe)."""

    mode: str = "group"
    residual_scale: Optional[List[float]] = None   # per-horizon (len H) for residual modes

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"target transform mode must be one of {MODES}, got {self.mode!r}")
        if self.mode == "residual_current" and not self.residual_scale:
            raise ValueError("residual_current requires a fitted residual_scale (use fit_residual_scale)")

    @property
    def needs_anchor(self) -> bool:
        return self.mode == "residual_current"

    def predict_mgdl(self, model, raw_pred: torch.Tensor, anchor_mgdl: Optional[torch.Tensor],
                     target_scale: torch.Tensor) -> torch.Tensor:
        """Map the decoder output ``(A, H, Q)`` to mg/dL quantile forecasts ``(A, H, Q)``.

        ``group``/``global`` use the model's normalizer inverse; ``residual_current``
        adds the per-anchor current glucose to the (scaled) predicted residual. Quantile
        ordering is preserved because the inverse is monotone in the raw output.
        """
        if self.mode in ("group", "global"):
            return model.transform_output(raw_pred, target_scale=target_scale)
        # residual_current: anchor_t (A,) + residual̂ (A,H,Q) * scale (1,H,1)
        scale = torch.tensor(self.residual_scale, device=raw_pred.device, dtype=raw_pred.dtype)
        scale = scale[: raw_pred.shape[1]].view(1, -1, 1)
        return anchor_mgdl.view(-1, 1, 1) + raw_pred * scale

    # ----- persistence (saved alongside the checkpoint) -----
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mode": self.mode, "residual_scale": self.residual_scale}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TargetTransform":
        with open(path) as f:
            d = json.load(f)
        return cls(mode=d.get("mode", "group"), residual_scale=d.get("residual_scale"))


def fit_residual_scale(
    streams: Sequence, spec, *, robust: bool = True, max_per_stream: int = 2000, stride: int = 3,
) -> List[float]:
    """Per-horizon scale of ``glucose[t+h] - anchor_t`` over **train** anchors (len H).

    Robust (MAD-based) by default. Computed only from the streams passed in (which must
    be train streams) and only from observed targets — leakage-safe."""
    H = spec.horizon
    cols: List[List[float]] = [[] for _ in range(H)]
    for s in streams:
        tgt = s.target.cpu().numpy()
        anc = (s.anchor if s.anchor is not None else s.target).cpu().numpy()
        obs = s.target_observed.cpu().numpy()
        n = len(tgt)
        taken = 0
        for t in range(0, n - H, stride):
            if not obs[t + 1:t + 1 + H].all():
                continue
            a = anc[t]
            for h in range(H):
                cols[h].append(tgt[t + 1 + h] - a)
            taken += 1
            if taken >= max_per_stream:
                break
    out = []
    for h in range(H):
        arr = np.asarray(cols[h], dtype="float64")
        if arr.size == 0:
            out.append(1.0)
        elif robust:
            mad = np.median(np.abs(arr - np.median(arr)))
            out.append(float(max(1.4826 * mad, 1e-3)))
        else:
            out.append(float(max(arr.std(), 1e-3)))
    return out
