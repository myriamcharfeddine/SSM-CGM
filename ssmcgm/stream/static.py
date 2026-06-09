"""Static personalization: encode S once, then condition the stream on it.

Three modules, all pure-PyTorch and standalone-testable:

* :class:`StaticEncoder` — maps static categorical + continuous covariates to a
  single static embedding ``e_s`` (computed once per participant/window, cached in
  :class:`~ssmcgm.stream.state.StaticContext`). It owns its own embedding tables so
  it can be unit-tested without ``pytorch_forecasting``.
* :class:`StaticStateInitializer` — maps ``e_s`` to a patient-specific initial SSM
  state ``h0`` per layer (``state_init_mode`` in ``{zero, learned_global,
  patient_static}``). Differentiable, so gradients flow into the initializer and
  ``e_s`` through the streaming scan.
* :class:`StaticFiLM` — FiLM conditioning ``u_t = γ(e_s) ⊙ z_t + β(e_s)`` with
  ``γ≈1``/``β≈0`` at init (so the model starts at the non-FiLM baseline).

Design note on the ``h0`` shape. The SSM state is ``(B, H, P, N)`` (heads ×
head-dim × state). A per-(H,P,N) initializer would be huge, so ``h0`` is
parameterized as ``(B, H, N)`` from ``e_s`` and broadcast over the head-dim ``P``
at scan time (a rank-1-in-P initial state). Gradients still flow because the
broadcast/expand is differentiable.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

_STATE_INIT_MODES = ("zero", "learned_global", "patient_static")
_FILM_MODES = ("scale_shift", "scale_only", "none")


class StaticEncoder(nn.Module):
    """Encode static categorical + continuous covariates into ``e_s`` (B, hidden).

    Args:
        cat_cardinalities: per static-categorical number of classes (embedding
            table size), in the column order of ``static_cat``.
        cat_emb_dims: matching embedding dims (same length/order).
        n_continuous: number of static continuous features.
        hidden_size: output embedding width.
        dropout: dropout on the fused embedding.
    """

    def __init__(
        self,
        cat_cardinalities: Sequence[int],
        cat_emb_dims: Sequence[int],
        n_continuous: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        cat_cardinalities = list(cat_cardinalities)
        cat_emb_dims = list(cat_emb_dims)
        assert len(cat_cardinalities) == len(cat_emb_dims)
        self.n_cat = len(cat_cardinalities)
        self.n_cont = int(n_continuous)
        self.hidden_size = int(hidden_size)

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cat_cardinalities, cat_emb_dims)
        ])
        fused_in = sum(cat_emb_dims) + self.n_cont
        # Guard the degenerate no-static case so the module is still constructible.
        self.has_static = fused_in > 0
        self.fuse = nn.Sequential(
            nn.Linear(max(fused_in, 1), hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, static_cat: Optional[torch.Tensor],
                static_cont: Optional[torch.Tensor]) -> torch.Tensor:
        """``static_cat``: ``(B, n_cat)`` long. ``static_cont``: ``(B, n_cont)`` float."""
        ref = static_cont if static_cont is not None else static_cat
        if ref is None:
            raise ValueError("StaticEncoder needs at least one of static_cat / static_cont")
        B = ref.shape[0]
        device = ref.device
        parts: List[torch.Tensor] = []
        for i, emb in enumerate(self.embeddings):
            parts.append(emb(static_cat[:, i]))
        if self.n_cont and static_cont is not None:
            parts.append(static_cont.to(self.fuse[0].weight.dtype))
        if parts:
            fused_in = torch.cat(parts, dim=-1)
        else:  # no static features at all -> zeros (still a valid e_s)
            fused_in = torch.zeros(B, 1, device=device, dtype=self.fuse[0].weight.dtype)
        return self.fuse(fused_in)


class StaticStateInitializer(nn.Module):
    """Patient-specific initial SSM state ``h0`` per layer, from ``e_s``.

    Returns a list of ``depth`` reduced states, each ``(B, H, N)``; the streaming
    SSM broadcasts these over the head-dim ``P``. ``state_init_mode``:

    * ``"zero"`` — zeros (the classic Mamba init; not patient-specific).
    * ``"learned_global"`` — a learned global state shared by all participants.
    * ``"patient_static"`` — a per-participant state ``g(e_s)`` (differentiable).
    """

    def __init__(self, e_s_dim: int, depth: int, nheads: int, d_state: int,
                 state_init_mode: str = "patient_static"):
        super().__init__()
        if state_init_mode not in _STATE_INIT_MODES:
            raise ValueError(f"state_init_mode must be one of {_STATE_INIT_MODES}")
        self.mode = state_init_mode
        self.depth = int(depth)
        self.nheads = int(nheads)
        self.d_state = int(d_state)
        per_layer = nheads * d_state

        if state_init_mode == "learned_global":
            # one learned (H, N) per layer, broadcast over batch
            self.global_state = nn.Parameter(torch.zeros(depth, nheads, d_state))
        elif state_init_mode == "patient_static":
            # g_ℓ(e_s) -> (H*N) per layer; small init so we start near zero-state
            self.proj = nn.Linear(e_s_dim, depth * per_layer)
            nn.init.zeros_(self.proj.bias)
            nn.init.normal_(self.proj.weight, std=1e-3)

    def forward(self, e_s: torch.Tensor) -> List[torch.Tensor]:
        B = e_s.shape[0]
        H, N = self.nheads, self.d_state
        if self.mode == "zero":
            z = e_s.new_zeros(B, H, N)
            return [z for _ in range(self.depth)]
        if self.mode == "learned_global":
            return [self.global_state[i].unsqueeze(0).expand(B, H, N)
                    for i in range(self.depth)]
        # patient_static
        h = self.proj(e_s).view(B, self.depth, H, N)
        return [h[:, i] for i in range(self.depth)]


class StaticFiLM(nn.Module):
    """FiLM conditioning ``u_t = γ(e_s) ⊙ z_t + β(e_s)`` from the static embedding.

    Initialized to the identity (``γ=1``, ``β=0``) so the model starts at the
    non-FiLM baseline. ``film_mode``: ``scale_shift`` (γ and β), ``scale_only`` (γ
    only), ``none`` (identity).
    """

    def __init__(self, e_s_dim: int, hidden_size: int, film_mode: str = "scale_shift"):
        super().__init__()
        if film_mode not in _FILM_MODES:
            raise ValueError(f"film_mode must be one of {_FILM_MODES}")
        self.film_mode = film_mode
        self.hidden_size = int(hidden_size)
        if film_mode != "none":
            self.gamma = nn.Linear(e_s_dim, hidden_size)
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)          # γ = 1 + 0 = 1 at init
        if film_mode == "scale_shift":
            self.beta = nn.Linear(e_s_dim, hidden_size)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)           # β = 0 at init

    def coeffs(self, e_s: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(gamma, beta)`` of shape ``(B, hidden)`` (``beta`` may be None)."""
        if self.film_mode == "none":
            return torch.ones_like(e_s[:, :1]).expand(-1, self.hidden_size), None
        gamma = 1.0 + self.gamma(e_s)
        beta = self.beta(e_s) if self.film_mode == "scale_shift" else None
        return gamma, beta

    def forward(self, z: torch.Tensor, e_s: torch.Tensor) -> torch.Tensor:
        """Apply FiLM to ``z`` ``(B, T, hidden)`` (or ``(B, hidden)``) using ``e_s``."""
        if self.film_mode == "none":
            return z
        gamma, beta = self.coeffs(e_s)
        if z.dim() == 3:
            gamma = gamma[:, None, :]
            beta = None if beta is None else beta[:, None, :]
        out = gamma * z
        if beta is not None:
            out = out + beta
        return out
