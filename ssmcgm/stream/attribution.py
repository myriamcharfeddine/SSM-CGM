"""Offline MES feature-time-head attribution for the streaming model.

The streaming MES block records a small per-layer cache during a
``scan(record=True)`` — ``(dt_raw, A_log, B_h, C)``, the *same* layout as
:meth:`ssmcgm.modules.mes_mamba2.FastMESMamba2._build_cache` — plus the per-feature
contributions ``u_{i,j}`` from :class:`~ssmcgm.stream.fusion.GroupedLinearFusion`.

This module turns that cache into a retrospective attribution map. It reuses the
validated :func:`ssmcgm.ops.mes_reference.hidden_attention`, which reconstructs the
head-wise hidden-attention magnitude

    K_{ℓj}^{(m)} = C_shared (∏_{k=j}^{ℓ-1} A_k^{(m)}) B_j^{(m)}     (summed over states)

and combines it with the feature contributions to give

    I(i, j, ℓ, m) = ‖ K_{ℓj}^{(m)} · u_{i,j} ‖.

It is **offline** — never called from the default forward — and may be O(L²) over a
selected window, unlike the O(1)-per-step forecasting path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..ops.mes_reference import hidden_attention


@dataclass
class AttributionResult:
    """Result of :func:`attribute_window`.

    ``influence`` is ``(B, [M,] [F,] L_sel, L)`` — head ``M`` present unless
    ``aggregate_heads``; feature ``F`` present unless ``aggregate_features``;
    ``L_sel`` are the attributed target rows ``ℓ`` and ``L`` the source steps ``j``.
    """

    influence: torch.Tensor
    feature_names: List[str]
    target_rows: torch.Tensor
    layer: int
    aggregate_heads: bool
    aggregate_features: bool
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return tuple(self.influence.shape)


@torch.no_grad()
def attribute_window(
    caches: List[tuple],
    contributions: Dict[str, torch.Tensor],
    *,
    layer: int = -1,
    target_rows: Optional[torch.Tensor] = None,
    target_horizon: Optional[int] = None,
    max_lag: Optional[int] = None,
    aggregate_heads: bool = False,
    aggregate_features: bool = False,
    normalize: str = "softmax",
    ngroups: int = 1,
    sample_idx: Optional[int] = None,
) -> AttributionResult:
    """Feature×lag×target(×head) attribution from a recorded streaming scan.

    Args:
        caches: per-layer caches from ``StreamingMESStack.scan(record=True)``.
        contributions: ``{feature_name: (B, L, hidden)}`` from the input fusion over
            the same history window (source steps ``j`` live on the ``L`` axis).
        layer: which layer's hidden-attention to use (default last).
        target_rows: explicit target steps ``ℓ`` to attribute. Defaults to the final
            history step (the state that feeds the horizon decoder), i.e. ``[L-1]``.
        target_horizon: convenience — attribute the last ``target_horizon`` steps.
        max_lag: band the source steps to ``ℓ - j ≤ max_lag`` (others zeroed).
        aggregate_heads / aggregate_features: sum over heads / features.
    """
    cache = caches[layer]
    dt_raw, A_log, B_in, C_in = cache
    device = dt_raw.device
    L = dt_raw.shape[-1]

    if target_rows is None:
        if target_horizon is not None:
            target_rows = torch.arange(max(0, L - target_horizon), L, device=device)
        else:
            target_rows = torch.tensor([L - 1], device=device)
    target_rows = target_rows.to(device).long()

    # head-wise hidden-attention magnitude K_{ℓj}^{(m)} -> (B, H, L_sel, L)
    attn_heads, _ = hidden_attention(
        dt_raw, A_log, B_in, C_in, rows=target_rows, normalize=normalize,
        ngroups=ngroups, sample_idx=sample_idx,
    )
    B, H, L_sel, _ = attn_heads.shape

    if max_lag is not None:
        j = torch.arange(L, device=device).view(1, 1, 1, L)
        l = target_rows.view(1, 1, L_sel, 1)
        attn_heads = attn_heads.masked_fill((l - j) > max_lag, 0.0)

    # per-feature contribution norms over the hidden dim -> (B, F, L)
    names = list(contributions)
    if sample_idx is not None:
        u_norm = torch.stack(
            [contributions[n][sample_idx:sample_idx + 1].norm(dim=-1) for n in names], dim=1)
    else:
        u_norm = torch.stack([contributions[n].norm(dim=-1) for n in names], dim=1)  # (B,F,L)

    # I(i,j,ℓ,m) = attn[b,m,ℓ,j] * ||u_{i,j}||  -> (B, H, F, L_sel, L)
    influence = attn_heads[:, :, None, :, :] * u_norm[:, None, :, None, :]
    if aggregate_heads:
        influence = influence.sum(dim=1)                  # (B, F, L_sel, L)
        if aggregate_features:
            influence = influence.sum(dim=1)              # (B, L_sel, L)
    elif aggregate_features:
        influence = influence.sum(dim=2)                  # (B, H, L_sel, L)

    return AttributionResult(
        influence=influence, feature_names=names, target_rows=target_rows, layer=layer,
        aggregate_heads=aggregate_heads, aggregate_features=aggregate_features,
        meta={"max_lag": max_lag, "normalize": normalize, "n_heads": H, "L": L},
    )
