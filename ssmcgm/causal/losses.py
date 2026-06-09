"""Causal-coherence loss modules for the scenario decoder (docs/causal.md).

All optional and config-gated; the forecasting loss stays the main objective and default
behavior is unchanged. Each term is a pure function over forecasts / scores so it can be
unit-tested and logged separately. They are wired into the trainer in
:mod:`ssmcgm.training.stream_trainer`.

* :func:`intervention_ranking_loss` — for interventions with a KNOWN global direction
  (carbs +, insulin −): given the per-variant outcome scores for an intervention set, penalize
  orderings that violate the known dose→outcome ranking (hybrid-causal style). For
  context-dependent actions (exercise) pass per-variant ``gates`` so the ranking is only
  enforced where a domain rule is believed valid (context-gated ranking).
* :func:`score_glucose` — outcome scalar u(ŷ): mean / max / min / final / delayed-mean /
  time-above-180 / time-below-70 (insulin uses delayed-mean; rescue carbs use min/TBR).
* :func:`slope_loss` — match predicted vs true step-to-step glucose change (rise/fall).
* :func:`mask_consistency_loss` — a masked (M=0) scenario value must not change the forecast.
* :func:`effect_shape_loss` — scenario effect should be smooth/physiologically bounded; an
  optional immediate-onset penalty (insulin shouldn't act in the first 5 min; sleep no jumps).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


def score_glucose(pred: torch.Tensor, kind: str = "mean", *, delayed_from: int = 6) -> torch.Tensor:
    """Outcome scalar per sample from a median forecast ``pred`` (B, H) in mg/dL."""
    if kind == "mean":
        return pred.mean(dim=-1)
    if kind == "max":
        return pred.amax(dim=-1)
    if kind == "min":
        return pred.amin(dim=-1)
    if kind == "final":
        return pred[..., -1]
    if kind == "delayed_mean":                      # for insulin: score after action onset
        return pred[..., min(delayed_from, pred.shape[-1] - 1):].mean(dim=-1)
    if kind == "tir_above":                         # fraction of horizon above 180
        return (pred > 180.0).float().mean(dim=-1)
    if kind == "tir_below":                         # fraction below 70 (rescue carbs)
        return (pred < 70.0).float().mean(dim=-1)
    raise ValueError(f"unknown score kind {kind!r}")


def intervention_ranking_loss(
    scores: torch.Tensor, dose_ranks: Sequence[int], direction: int, *,
    gates: Optional[torch.Tensor] = None, support: Optional[Sequence[bool]] = None,
    margin: float = 2.0,
) -> torch.Tensor:
    """Pairwise hinge ranking loss over an intervention set's outcome ``scores`` (B, K).

    For ``direction=+1`` a higher dose must yield a higher score (carbs→glucose↑); for
    ``-1`` a higher dose must yield a *lower* score (insulin→glucose↓, scored on a delayed
    window). Pairs are weighted by the per-variant context gate (``gates``, defaults 1) and
    skipped if either variant is ``out_of_support``. ``direction==0`` (ambiguous) ⇒ no loss
    unless gated externally. Returns a scalar.
    """
    B, K = scores.shape
    if direction == 0 and gates is None:
        return scores.new_zeros(())
    w = torch.ones(K, device=scores.device) if gates is None else gates.to(scores.device)
    if support is not None:
        w = w * torch.tensor([1.0 if s else 0.0 for s in support], device=scores.device)
    ranks = list(dose_ranks)
    total, npairs = scores.new_zeros(()), 0
    for i in range(K):
        for j in range(K):
            if ranks[j] <= ranks[i]:
                continue
            wij = torch.minimum(w[i], w[j])
            if float(wij) <= 0:
                continue
            # want  direction * (score_j - score_i) >= margin
            viol = torch.relu(margin - direction * (scores[:, j] - scores[:, i]))
            total = total + wij * viol.mean()
            npairs += 1
    return total / max(npairs, 1)


def slope_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on consecutive-step glucose *changes* — rewards getting rise/fall right even when
    the level is right (``pred``/``target`` (B, H) mg/dL)."""
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    return ((dp - dt) ** 2).mean()


def mask_consistency_loss(pred_low: torch.Tensor, pred_high: torch.Tensor) -> torch.Tensor:
    """Penalize any forecast change between two batches that differ only in a *masked*
    (M=0) scenario value — prevents leakage through arbitrary imputed future covariates.
    ``pred_low``/``pred_high`` are forecasts (B, H[, Q]) of the two masked variants."""
    return ((pred_low - pred_high) ** 2).mean()


def effect_shape_loss(effect: torch.Tensor, *, immediate_penalty: float = 0.0,
                      bound_mgdl: Optional[float] = None) -> torch.Tensor:
    """Plausibility prior on a scenario effect curve ``effect`` (B, H) = ŷ(scenario) − ŷ(base):
    penalize roughness (2nd difference), optionally penalize an immediate (h=0) jump
    (``immediate_penalty`` — insulin/sleep act with delay), and optionally penalize
    effects beyond ``bound_mgdl``."""
    loss = effect.new_zeros(())
    if effect.shape[-1] >= 3:
        d2 = effect[:, 2:] - 2 * effect[:, 1:-1] + effect[:, :-2]
        loss = loss + (d2 ** 2).mean()
    if immediate_penalty:
        loss = loss + immediate_penalty * (effect[:, 0] ** 2).mean()
    if bound_mgdl is not None:
        loss = loss + torch.relu(effect.abs() - bound_mgdl).pow(2).mean()
    return loss
