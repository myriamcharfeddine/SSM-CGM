"""Grouped linear feature fusion — the stream model's VSN replacement.

The full ``SSMCGM`` uses a :class:`VariableSelectionNetwork` (per-variable GRNs +
softmax selection) for every time step. That preserves feature identity but is
expensive. :class:`GroupedLinearFusion` keeps *some* feature identity (each group
gets its own projection, so per-feature contributions are recoverable for
attribution) at a fraction of the cost — no GRNs, no per-step softmax.

For a set of time-varying feature groups ``i`` with per-step features
``x_{i,t}``:

    grouped_sum:            u_t = Σ_i W_i x_{i,t}
    dense_linear:           u_t = W [concat_i x_{i,t}]              (no attribution)
    grouped_concat_project: u_t = W_out [concat_i (W_i x_{i,t})]
    cheap_gated:            u_t = Σ_i σ(w_i·x_{i,t}) (W_i x_{i,t})  (lightweight gate, not a VSN)

``forward`` returns ``(fused, contributions)`` where ``contributions`` is a dict
``{name: u_{i,t}}`` (the per-feature term that summed/concatenated into ``fused``)
or ``None`` when the mode has no meaningful per-feature decomposition. These
contributions feed the offline MES feature-time attribution.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union

import torch
from torch import nn

_FUSION_MODES = ("grouped_sum", "dense_linear", "grouped_concat_project", "cheap_gated")


class GroupedLinearFusion(nn.Module):
    """Fuse named time-varying feature groups into a single ``hidden_size`` vector.

    Args:
        input_sizes: ordered mapping ``{feature_name: feature_dim}``. The order
            fixes the concatenation order for the dense modes.
        hidden_size: output (and per-group projection) width.
        fusion_mode: one of :data:`_FUSION_MODES`.
        dropout: applied to the fused output.
        bias: whether the per-group / dense projections carry a bias. For
            ``grouped_sum`` a per-group bias would break the
            ``Σ contributions == fused`` identity, so it is forced off there.
    """

    def __init__(
        self,
        input_sizes: Mapping[str, int],
        hidden_size: int,
        fusion_mode: str = "grouped_sum",
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        if fusion_mode not in _FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {_FUSION_MODES}, got {fusion_mode!r}")
        self.names = list(input_sizes)
        self.input_sizes = dict(input_sizes)
        self.hidden_size = int(hidden_size)
        self.fusion_mode = fusion_mode
        self.total_in = sum(self.input_sizes.values())

        if fusion_mode == "dense_linear":
            self.dense = nn.Linear(self.total_in, hidden_size, bias=bias)
        else:
            # per-group projections W_i. grouped_sum must have no bias so the
            # contributions sum *exactly* to the fused output.
            grp_bias = bias and (fusion_mode != "grouped_sum")
            self.group_proj = nn.ModuleDict({
                name: nn.Linear(size, hidden_size, bias=grp_bias)
                for name, size in self.input_sizes.items()
            })
        if fusion_mode == "grouped_concat_project":
            self.out_proj = nn.Linear(hidden_size * len(self.names), hidden_size, bias=bias)
        if fusion_mode == "cheap_gated":
            self.gate = nn.ModuleDict({
                name: nn.Linear(size, 1, bias=True) for name, size in self.input_sizes.items()
            })
        self.drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def _as_dict(self, features: Union[Mapping[str, torch.Tensor], torch.Tensor]
                 ) -> Dict[str, torch.Tensor]:
        if isinstance(features, torch.Tensor):
            if len(self.names) != 1:
                raise ValueError(
                    "a single tensor was passed but GroupedLinearFusion has "
                    f"{len(self.names)} feature groups; pass a {{name: tensor}} dict.")
            return {self.names[0]: features}
        return {n: features[n] for n in self.names}

    def forward(
        self,
        features: Union[Mapping[str, torch.Tensor], torch.Tensor],
        return_contributions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Fuse ``features`` (dict or single tensor) → ``(fused, contributions)``.

        Each feature tensor is ``(..., feature_dim)`` (typically ``(B, T, dim)`` or
        ``(B, dim)`` for a single step). ``fused`` is ``(..., hidden_size)``.
        """
        feats = self._as_dict(features)

        if self.fusion_mode == "dense_linear":
            cat = torch.cat([feats[n] for n in self.names], dim=-1)
            fused = self.dense(cat)
            return self.drop(fused), (None if not return_contributions else None)

        contribs: Dict[str, torch.Tensor] = {}
        if self.fusion_mode == "grouped_sum":
            fused = None
            for n in self.names:
                u_i = self.group_proj[n](feats[n])
                contribs[n] = u_i
                fused = u_i if fused is None else fused + u_i
        elif self.fusion_mode == "cheap_gated":
            fused = None
            for n in self.names:
                g_i = torch.sigmoid(self.gate[n](feats[n]))     # (..., 1)
                u_i = g_i * self.group_proj[n](feats[n])
                contribs[n] = u_i
                fused = u_i if fused is None else fused + u_i
        elif self.fusion_mode == "grouped_concat_project":
            parts = []
            for n in self.names:
                u_i = self.group_proj[n](feats[n])
                contribs[n] = u_i
                parts.append(u_i)
            fused = self.out_proj(torch.cat(parts, dim=-1))
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(self.fusion_mode)

        fused = self.drop(fused)
        return fused, (contribs if return_contributions else None)
