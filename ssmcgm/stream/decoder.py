"""Scenario-aware horizon decoder.

Maps the current stream state ``h_t`` plus the future known/scenario inputs to the
``H``-step quantile forecast — **without** running the future through the temporal
SSM. Because it only processes the short horizon, it is streamable.

    ŷ_{t+h}^{(q)} = d_q(h_t, e_s, τ_{t+h}, A_{t+h}·M_{t+h}, M_{t+h}, horizon_emb[h])

The ``A·M`` + ``M`` representation makes an *unknown* scenario input (mask 0)
distinguishable from a supplied *real zero* (mask 1) — no future scenario value can
leak into ``h_t``. ``decoder_mode``:

* ``"shared_mlp_with_horizon_embedding"`` (default) — one MLP over all horizons,
  with a learned per-step horizon embedding.
* ``"mlp_per_horizon"`` — an independent MLP per horizon step.
* ``"linear"`` — a single linear map per step.
"""

from __future__ import annotations

from typing import List, Union

import torch
from torch import nn

_DECODER_MODES = ("mlp_per_horizon", "shared_mlp_with_horizon_embedding", "linear")


class ScenarioHorizonDecoder(nn.Module):
    """Decode ``(h_t, e_s, T, A, M)`` → quantile forecast ``(B, H, output_size)``.

    Args:
        d_model: width of ``h_t``.
        e_s_dim: width of the static embedding ``e_s``.
        n_time_features: number of known-future time features per step (``T``).
        n_scenario: number of scenario variables (``A``/``M``).
        horizon: max forecast horizon ``H``.
        output_size: ``n_quantiles`` (int) or a list for multi-target.
        hidden_size: decoder MLP width.
        decoder_mode: see module docstring.
        horizon_emb_dim: width of the learned horizon embedding.
    """

    def __init__(
        self,
        d_model: int,
        e_s_dim: int,
        n_time_features: int,
        n_scenario: int,
        horizon: int,
        output_size: Union[int, List[int]] = 7,
        hidden_size: int = 128,
        decoder_mode: str = "shared_mlp_with_horizon_embedding",
        horizon_emb_dim: int = 16,
        dropout: float = 0.1,
        scenario_decompose: bool = False,
    ):
        super().__init__()
        if decoder_mode not in _DECODER_MODES:
            raise ValueError(f"decoder_mode must be one of {_DECODER_MODES}")
        self.decoder_mode = decoder_mode
        self.scenario_decompose = bool(scenario_decompose)
        self.horizon = int(horizon)
        self.n_scenario = int(n_scenario)
        self.n_time_features = int(n_time_features)
        self.multi_target = isinstance(output_size, (list, tuple))
        self.output_sizes = list(output_size) if self.multi_target else [int(output_size)]

        use_horizon_emb = decoder_mode == "shared_mlp_with_horizon_embedding"
        self.horizon_emb = nn.Embedding(self.horizon, horizon_emb_dim) if use_horizon_emb else None
        # input per horizon step: h_t, e_s, T, A·M, M, [horizon_emb]
        in_dim = d_model + e_s_dim + self.n_time_features + 2 * self.n_scenario
        if use_horizon_emb:
            in_dim += horizon_emb_dim
        self.in_dim = in_dim

        def _mlp():
            return nn.Sequential(
                nn.Linear(in_dim, hidden_size), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size), nn.GELU(),
            )

        if decoder_mode == "linear":
            self.trunk = None
            trunk_out = in_dim
        elif decoder_mode == "mlp_per_horizon":
            self.trunk = nn.ModuleList([_mlp() for _ in range(self.horizon)])
            trunk_out = hidden_size
        else:
            self.trunk = _mlp()
            trunk_out = hidden_size

        self.heads = nn.ModuleList([nn.Linear(trunk_out, o) for o in self.output_sizes])

        # optional scenario-effect decomposition: ŷ = ŷ_base + Δŷ_scenario (docs/causal.md).
        # The base trunk/heads (above) see the scenario inputs zeroed; a parallel effect
        # trunk+heads consume the full scenario context and predict the additive change. The
        # effect heads are zero-initialized so Δŷ=0 at init (final == base; safe warm start).
        if self.scenario_decompose:
            self.effect_trunk = (_mlp() if decoder_mode != "linear" else None)
            if decoder_mode == "mlp_per_horizon":
                self.effect_trunk = nn.ModuleList([_mlp() for _ in range(self.horizon)])
            self.effect_heads = nn.ModuleList([nn.Linear(trunk_out, o) for o in self.output_sizes])
            for hh in self.effect_heads:
                nn.init.zeros_(hh.weight)
                nn.init.zeros_(hh.bias)

    def _apply_trunk(self, trunk, dec_in: torch.Tensor, H: int) -> torch.Tensor:
        if self.decoder_mode == "linear":
            return dec_in
        if self.decoder_mode == "mlp_per_horizon":
            return torch.stack([trunk[h](dec_in[:, h]) for h in range(H)], dim=1)
        return trunk(dec_in)

    def forward(
        self,
        h_t: torch.Tensor,            # (B, d_model)
        e_s: torch.Tensor,            # (B, e_s_dim)
        time_features: torch.Tensor,  # (B, H, n_time_features)
        scenario_values: torch.Tensor,  # (B, H, n_scenario)
        scenario_mask: torch.Tensor,    # (B, H, n_scenario)
        *,
        return_decomposition: bool = False,
    ):
        B, H = time_features.shape[0], time_features.shape[1]
        h_rep = h_t.unsqueeze(1).expand(B, H, h_t.shape[-1])
        s_rep = e_s.unsqueeze(1).expand(B, H, e_s.shape[-1])
        gated = scenario_values * scenario_mask           # A·M (unknown -> 0, mask separate)
        parts = [h_rep, s_rep, time_features, gated, scenario_mask]
        he = None
        if self.horizon_emb is not None:
            steps = torch.arange(H, device=h_t.device).clamp(max=self.horizon - 1)
            he = self.horizon_emb(steps).unsqueeze(0).expand(B, H, -1)
            parts.append(he)
        dec_in = torch.cat(parts, dim=-1)                 # (B, H, in_dim)

        if not self.scenario_decompose:
            rep = self._apply_trunk(self.trunk, dec_in, H)
            preds = [head(rep) for head in self.heads]    # each (B, H, o)
            out = preds[0] if not self.multi_target else preds
            if return_decomposition:
                zero = out * 0 if not self.multi_target else [p * 0 for p in preds]
                return out, out, zero
            return out

        # decomposition: base sees the scenario zeroed (known-future only); effect head
        # consumes the full scenario context and predicts the additive Δŷ.
        z = torch.zeros_like(gated)
        base_parts = [h_rep, s_rep, time_features, z, z]
        if he is not None:
            base_parts.append(he)
        base_in = torch.cat(base_parts, dim=-1)
        base_rep = self._apply_trunk(self.trunk, base_in, H)
        eff_rep = self._apply_trunk(self.effect_trunk, dec_in, H)
        base = [head(base_rep) for head in self.heads]
        effect = [eh(eff_rep) for eh in self.effect_heads]
        final = [b + e for b, e in zip(base, effect)]
        if self.multi_target:
            return (final, base, effect) if return_decomposition else final
        return (final[0], base[0], effect[0]) if return_decomposition else final[0]
