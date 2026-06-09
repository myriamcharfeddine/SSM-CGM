"""Stable chunked SSD for the MES scan — O(L · chunk) memory, autograd-friendly.

This is the chunked (vs. naive quadratic) MES forward. It implements the same
selective-state-space recurrence as :func:`ssmcgm.ops.mes_reference.mes_scan_reference`
but in the canonical block/chunk decomposition (Mamba-2 "minimal SSD"), so peak
memory is ``O(L · chunk_size)`` instead of ``O(L²)`` and it stays numerically
stable for long contexts (within-chunk decays are bounded; cross-chunk coupling
uses a segment-sum, never a ``1/exp`` reverse-recompute).

Because it is pure PyTorch, autograd gives a matching ``O(L · chunk)`` backward
for free — which is exactly what the custom-Triton backend uses instead of the
old ``O(L²)`` recompute, and what the ``scan_backend="chunked"`` path runs
(CPU-capable, no Triton/CUDA kernel needed).

MES adaptation of the minimal SSD:
  * ``x`` is shared across heads and scaled by per-head ``dt`` -> ``X = dt · x``  (B,L,H,P)
  * ``A`` is scalar-per-head -> per-step log-decay ``Astep = dt · A``           (B,L,H)
  * ``B`` is per-head                                                            (B,L,H,N)
  * ``C`` is shared across heads (kept ``(B,L,N)``, broadcast in the einsums)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Lower-triangular segment sum.

    ``out[..., i, j] = sum_{j < k <= i} x_k`` for ``i >= j`` (0 on the diagonal),
    ``-inf`` above the diagonal. ``exp(out)`` is then the decay product from
    step ``j`` to step ``i``.
    """
    T = x.size(-1)
    xc = torch.cumsum(x, dim=-1)
    seg = xc.unsqueeze(-1) - xc.unsqueeze(-2)            # [..., i, j] = xc_i - xc_j
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    return seg.masked_fill(~mask, float("-inf"))


def mes_ssd_chunked(
    x_shared: torch.Tensor,   # (B, L, P)    shared value path
    dt: torch.Tensor,         # (B, L, H)    per-head timesteps (post-softplus)
    A: torch.Tensor,          # (H,)         scalar-per-head dynamics (<= 0)
    B_h: torch.Tensor,        # (B, L, H, N) per-head input matrix
    C_shared: torch.Tensor,   # (B, L, N)    shared readout
    D: Optional[torch.Tensor] = None,   # (H,) or (H, P) or None
    chunk_size: int = 64,
    dt_limit: Optional[tuple] = None,
) -> torch.Tensor:
    """Chunked SSD MES output. Returns ``y`` of shape ``(B, L, H, P)``.

    Differentiable, CPU- and GPU-capable, ``O(L · chunk_size)`` peak memory.
    """
    Bsz, L, P = x_shared.shape
    H = dt.shape[-1]
    N = C_shared.shape[-1]
    cs = int(chunk_size)
    if dt_limit is not None:
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])

    # dt-scaled shared input (per head via dt) and per-step log-decay
    X = dt.unsqueeze(-1) * x_shared.unsqueeze(2)        # (B,L,H,P)
    Astep = dt * A.view(1, 1, H)                        # (B,L,H)

    # pad sequence to a whole number of chunks (padded steps are causal-future
    # of every real step, so they cannot affect outputs we keep)
    pad = (cs - L % cs) % cs
    if pad:
        X = F.pad(X, (0, 0, 0, 0, 0, pad))
        Astep = F.pad(Astep, (0, 0, 0, pad))
        B_h = F.pad(B_h, (0, 0, 0, 0, 0, pad))
        C_shared = F.pad(C_shared, (0, 0, 0, pad))
    Lp = L + pad
    nc = Lp // cs

    # reshape to chunks (B, nc, cs, ...)
    Xc = X.view(Bsz, nc, cs, H, P)
    Bc = B_h.view(Bsz, nc, cs, H, N)
    Cc = C_shared.view(Bsz, nc, cs, N)
    Ac = Astep.view(Bsz, nc, cs, H).permute(0, 1, 3, 2)    # (B,nc,H,cs)
    A_cumsum = torch.cumsum(Ac, dim=-1)                    # (B,nc,H,cs)

    # 1) intra-chunk (diagonal) contribution
    Ldecay = torch.exp(_segsum(Ac))                                       # (B,nc,H,cs,cs)
    CB = torch.einsum("bcin,bcjhn->bchij", Cc, Bc)                        # (B,nc,H,cs,cs)
    Y_diag = torch.einsum("bchij,bcjhp->bcihp", Ldecay * CB, Xc)          # (B,nc,cs,H,P)

    # 2) each chunk's end state
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)           # (B,nc,H,cs)
    states = torch.einsum("bchs,bcshn,bcshp->bchpn", decay_states, Bc, Xc)  # (B,nc,H,P,N)

    # 3) inter-chunk recurrence on the chunk states (prepend zero initial state)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)  # (B,nc+1,H,P,N)
    A_chunk = A_cumsum[:, :, :, -1].permute(0, 2, 1)                      # (B,H,nc) chunk totals
    decay_chunk = torch.exp(_segsum(F.pad(A_chunk, (1, 0))))              # (B,H,nc+1,nc+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)  # (B,nc+1,H,P,N)
    states_in = new_states[:, :-1]                                        # (B,nc,H,P,N)

    # 4) off-diagonal output: chunk-entry state decayed to each step, read by C
    state_decay_out = torch.exp(A_cumsum)                                # (B,nc,H,cs)
    Y_off = torch.einsum("bcin,bchpn,bchi->bcihp", Cc, states_in, state_decay_out)  # (B,nc,cs,H,P)

    y = (Y_diag + Y_off).reshape(Bsz, Lp, H, P)[:, :L]                   # (B,L,H,P)

    if D is not None:
        if D.dim() == 1:
            y = y + D.view(1, 1, H, 1) * x_shared.unsqueeze(2)
        else:
            y = y + D.view(1, 1, H, P) * x_shared.unsqueeze(2)
    return y
