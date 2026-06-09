"""Per-user personalization hooks for the scenario effect Δŷ (docs/causal.md §personalization).

A deployed user gives feedback like "my meal response was larger than predicted", "insulin
was weaker", "the effect was delayed". We must **not** fold one user's feedback into the
global model. Instead a small, per-user adapter calibrates **only** the decoder's
scenario-effect output Δŷ_scenario (never the base forecast or the shared weights):

    Δŷ_user = (1 + e·(g − 1)) ⊙ Δŷ_pop + e·b

where ``g`` (per-horizon multiplicative gain, exp-parameterized so g>0) and ``b``
(per-horizon additive shift) are the user's calibrated parameters, and ``e ∈ [0, 1]`` is an
*evidence* weight that shrinks the adapter toward the **population** effect (g→1, b→0) when
the user has few observed responses. So the model can express, from one checkpoint:
  * population-level expected effect (``e = 0`` → identity, the honest cold-start default),
  * user-specific calibrated effect (``e → 1`` with enough evidence),
  * graceful uncertainty (limited evidence ⇒ stays near population).

This module only requires a ``scenario_decompose=True`` decoder (so Δŷ is explicit); it is
attached to a model via :meth:`SSMCGMStream.set_effect_adapter` and is a no-op until fit.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import nn

# textual-feedback nudges: (gain factor, additive-bias sign) applied to the user's params.
# Multiplicative on gain (response larger/smaller), additive on bias (effect stronger/weaker).
FEEDBACK = {
    "meal_response_larger": ("gain_up",),
    "meal_response_smaller": ("gain_down",),
    "insulin_response_stronger": ("gain_up",),
    "insulin_response_weaker": ("gain_down",),
    "effect_delayed": ("delay",),
    "effect_too_strong": ("gain_down",),
    "effect_too_weak": ("gain_up",),
}


class ScenarioEffectAdapter(nn.Module):
    """Per-user FiLM/bias calibration of the scenario effect Δŷ (identity at init).

    ``horizon`` forecast steps; ``per_horizon`` gives each step its own gain/bias (else a
    single scalar gain + per-horizon bias). The ``evidence`` buffer (∈[0,1]) shrinks the
    adapter toward the population effect; it is 0 at init (pure population) and raised by
    :func:`calibrate_least_squares` / :func:`apply_feedback` as a user accumulates data.
    """

    def __init__(self, horizon: int, *, per_horizon: bool = True, user_id: Optional[str] = None):
        super().__init__()
        self.horizon = int(horizon)
        self.per_horizon = bool(per_horizon)
        self.user_id = user_id
        g = horizon if per_horizon else 1
        self.log_gain = nn.Parameter(torch.zeros(g))      # gain = exp(log_gain); init 1 (identity)
        self.bias = nn.Parameter(torch.zeros(horizon))    # per-horizon additive shift (scaled space)
        self.register_buffer("evidence", torch.zeros(()))  # 0 -> population; 1 -> fully user-calibrated

    # ------------------------------------------------------------------
    def gain(self) -> torch.Tensor:
        return torch.exp(self.log_gain)

    def forward(self, effect: torch.Tensor, *, evidence: Optional[float] = None) -> torch.Tensor:
        """Calibrate a population effect ``effect`` (B, H[, Q]) → user effect (same shape).

        ``effect`` is the decoder's Δŷ_scenario (raw output space). Shrinks toward the
        population effect by ``evidence`` (defaults to the stored buffer)."""
        e = self.evidence if evidence is None else torch.as_tensor(
            float(evidence), device=effect.device, dtype=effect.dtype)
        H = effect.shape[1]
        g = self.gain()[:H] if self.per_horizon else self.gain()
        gain_eff = 1.0 + e * (g - 1.0)                    # e=0 -> 1 (population)
        bias_eff = e * self.bias[:H]
        if effect.dim() == 3:                              # (B, H, Q): broadcast over quantiles
            gain_eff = gain_eff.view(1, -1, 1) if self.per_horizon else gain_eff.view(1, 1, 1)
            bias_eff = bias_eff.view(1, -1, 1)
        else:                                              # (B, H)
            gain_eff = gain_eff.view(1, -1) if self.per_horizon else gain_eff.view(1, 1)
            bias_eff = bias_eff.view(1, -1)
        return gain_eff * effect + bias_eff


# ---------------------------------------------------------------------------
# calibration from a user's observed scenario responses (the "memory summary")
# ---------------------------------------------------------------------------
@torch.no_grad()
def calibrate_least_squares(adapter: ScenarioEffectAdapter, pop_effect: torch.Tensor,
                            obs_effect: torch.Tensor, *, evidence_halflife: int = 8,
                            ridge: float = 1e-3) -> ScenarioEffectAdapter:
    """Fit the user's gain/bias so ``gain·pop + bias ≈ obs`` over their observed responses.

    ``pop_effect`` / ``obs_effect`` are ``(N, H)`` matched population-predicted vs observed
    scenario effects (mg/dL deviations) from the user's recent events. A scalar gain + a
    per-horizon bias are solved per horizon by ridge least squares; ``evidence`` is raised
    toward 1 with the number of events ``N`` (soft, via ``evidence_halflife``). Modifies and
    returns ``adapter`` in place. This is a per-user op — it never touches global weights."""
    pop = pop_effect.detach().float()
    obs = obs_effect.detach().float()
    N, H = pop.shape
    # per-horizon ordinary least squares for obs_h ≈ g_h·pop_h + b_h: slope from the centered
    # covariance, intercept from the means (jointly unbiased when a true bias is present).
    pm, om = pop.mean(0), obs.mean(0)
    pc, oc = pop - pm, obs - om
    g = ((pc * oc).sum(0) / ((pc * pc).sum(0) + ridge)).clamp(0.05, 20.0)
    bias = om - g * pm
    if adapter.per_horizon:
        adapter.log_gain.data = torch.log(g.clamp_min(1e-3))[: adapter.horizon]
    else:
        adapter.log_gain.data = torch.log(g.mean().clamp_min(1e-3)).reshape(1)
    adapter.bias.data = bias[: adapter.horizon]
    adapter.evidence.data = torch.tensor(1.0 - 0.5 ** (N / max(evidence_halflife, 1)))
    return adapter


@torch.no_grad()
def apply_feedback(adapter: ScenarioEffectAdapter, kind: str, *, strength: float = 0.15,
                   evidence_step: float = 0.1) -> ScenarioEffectAdapter:
    """Nudge a user's adapter from a single qualitative feedback label (``FEEDBACK`` keys).

    A coarse, bounded update for the deployed-feedback loop ("response larger/smaller",
    "effect delayed/too strong/weak"); much weaker than a data fit and capped so one report
    cannot dominate. Raises ``evidence`` by a small step. Per-user only."""
    if kind not in FEEDBACK:
        raise ValueError(f"unknown feedback {kind!r}; known: {sorted(FEEDBACK)}")
    action = FEEDBACK[kind][0]
    if action == "gain_up":
        adapter.log_gain.data = (adapter.log_gain.data + strength).clamp(-3.0, 3.0)
    elif action == "gain_down":
        adapter.log_gain.data = (adapter.log_gain.data - strength).clamp(-3.0, 3.0)
    elif action == "delay":
        # shift the bias curve one step later (a crude "effect arrives later" correction)
        adapter.bias.data = torch.roll(adapter.bias.data, shifts=1)
        adapter.bias.data[0] = 0.0
    adapter.evidence.data = (adapter.evidence.data + evidence_step).clamp(0.0, 1.0)
    return adapter
