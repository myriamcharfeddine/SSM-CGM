"""Vectorised, head-loop-free reference implementations of the MES scan.

Two distinct quantities live here and must not be confused:

1. :func:`mes_scan_reference` -- the **forward output** ``y`` of the MES-style
   selective state-space recurrence, computed in the naive quadratic SSD form
   (a single chunk).  It is fully vectorised over heads and runs on CPU, so it
   is the ground truth used by the numerical-equivalence tests and the autograd
   fallback for the Triton path.

2. :func:`hidden_attention` -- the **interpretability map** (MES hidden
   attention).  This reproduces, bit-for-bit, ``hidden_attention_per_head_mes``
   from the original interpretability notebook: signed ``B``/``C`` are replaced
   by their magnitudes, decays are row-normalised (softmax / exp) and summed
   over the state dimension.  It is *not* the forward output -- it is the
   temporal-attribution map the paper visualises.

Both are written without a ``for h in range(nheads)`` loop.

Recurrence being modelled (per head ``h``, batch ``b``, head-dim ``p``,
state ``n``), with a **scalar-per-head** ``A`` (the configuration the paper
uses):

    state_t[p, n] = exp(dt_t * A) * state_{t-1}[p, n] + dt_t * B_t[n] * x_t[p]
    y_t[p]        = sum_n C_t[n] * state_t[p, n]  (+ D[p] * x_t[p])

with ``x`` shared across heads, ``C`` shared across heads, ``B`` per head.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _segsum_decay(dt: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Lower-triangular decay matrix ``L[..., l, j] = exp(A * sum_{t=j+1..l} dt_t)``.

    Args:
        dt: (B, H, L) positive timesteps (already soft-plussed / clamped).
        A:  (H,) scalar-per-head continuous dynamics (<= 0).

    Returns:
        (B, H, L, L) decay, zero above the diagonal (causal).
    """
    Bsz, H, L = dt.shape
    # cumulative dt: cum[l] = sum_{t=1..l} dt_t
    cum = torch.cumsum(dt, dim=-1)                                   # (B,H,L)
    # exponent[l, j] = A * (cum[l] - cum[j]) for j <= l
    diff = cum.unsqueeze(-1) - cum.unsqueeze(-2)                     # (B,H,L,L) = cum_l - cum_j
    expo = A.view(1, H, 1, 1) * diff
    mask = torch.tril(torch.ones(L, L, device=dt.device, dtype=torch.bool))
    decay = torch.where(mask, torch.exp(expo), torch.zeros_like(expo))
    return decay


def mes_scan_reference(
    x_shared: torch.Tensor,   # (B, L, P)   shared value path
    dt: torch.Tensor,         # (B, L, H)   per-head timesteps (post-softplus)
    A: torch.Tensor,          # (H,)        scalar-per-head, A <= 0
    B_h: torch.Tensor,        # (B, L, H, N) per-head input matrix
    C_shared: torch.Tensor,   # (B, L, N)   shared readout
    D: Optional[torch.Tensor] = None,  # (H,) or (H, P) or None
    dt_limit: Optional[tuple] = None,
) -> torch.Tensor:
    """Naive (quadratic, single-chunk) SSD output for the MES structure.

    Returns ``y`` of shape (B, L, H, P).  Differentiable; CPU-friendly.

    This is the *output* path (signed B/C, dt input scaling), used to validate
    the fused Triton-kernel path numerically.
    """
    Bsz, L, P = x_shared.shape
    H = dt.shape[-1]
    N = C_shared.shape[-1]

    dt = dt.transpose(1, 2)                      # (B,H,L)
    if dt_limit is not None:
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])

    decay = _segsum_decay(dt, A)                 # (B,H,L,L)

    # CB[b,h,l,j] = sum_n C_l[n] * B_{h,j}[n]
    CB = torch.einsum("bln,bjhn->bhlj", C_shared, B_h)   # (B,H,L,L)

    # M[b,h,l,j] = decay[l,j] * CB[l,j] * dt[b,h,j]
    M = decay * CB * dt.unsqueeze(-2)            # (B,H,L,L)

    # y[b,h,l,p] = sum_j M[l,j] * x_j[p]
    y = torch.einsum("bhlj,bjp->blhp", M, x_shared)      # (B,L,H,P)

    if D is not None:
        if D.dim() == 1:                         # (H,)
            y = y + D.view(1, 1, H, 1) * x_shared.unsqueeze(2)
        else:                                    # (H,P)
            y = y + D.view(1, 1, H, P) * x_shared.unsqueeze(2)
    return y


@torch.no_grad()
def hidden_attention(
    dt_raw: torch.Tensor,        # (B, H, L)   pre-softplus per-head dt
    A_log: torch.Tensor,         # (H, N)      N = ngroups * d_state
    B_in: torch.Tensor,          # (B, H, N, L) per-head  OR (B, N, L) shared
    C_in: torch.Tensor,          # (B, N, L)   shared readout
    *,
    sample_idx: Optional[int] = None,
    rows: Optional[torch.Tensor] = None,
    causal: bool = True,
    normalize: str = "softmax",
    return_per_group: bool = False,
    ngroups: Optional[int] = None,
    reduce_states: str = "sum",
    compute_dtype: Optional[torch.dtype] = torch.float32,
):
    """MES hidden-attention map -- vectorised port of the notebook function.

    Returns:
        attn_heads:   (B, H, L_sel, L)
        attn_heads_g: (B, H, G, L_sel, L) if ``return_per_group`` else ``None``

    ``attn_heads[b, h, l, j]`` is the (optionally row-normalised) influence of
    source step ``j`` on target step ``l`` for head ``h``: a decay term times
    the magnitudes ``|C_l|`` and ``|B_j|`` summed over the state dimension.
    """
    if sample_idx is not None:
        dt_raw = dt_raw[sample_idx:sample_idx + 1]
        C_in = C_in[sample_idx:sample_idx + 1]
        B_in = B_in[sample_idx:sample_idx + 1]

    Bsz, H, L = dt_raw.shape
    B_is_per_head = (B_in.dim() == 4)
    if B_is_per_head:
        _, Hb, N, Lb = B_in.shape
        assert Hb == H, "per-head B_in: head axis mismatch with dt_raw"
    elif B_in.dim() == 3:
        _, N, Lb = B_in.shape
    else:
        raise ValueError("B_in must be (B,H,N,L) or (B,N,L)")
    assert Lb == L and C_in.shape[-1] == L, "inconsistent temporal dimensions"

    device = dt_raw.device
    if compute_dtype is not None:
        dt_raw = dt_raw.to(compute_dtype)
        A_log = A_log.to(compute_dtype)
        B_in = B_in.to(compute_dtype)
        C_in = C_in.to(compute_dtype)

    dt = F.softplus(dt_raw)                       # (B,H,L)
    A = -torch.exp(A_log)                         # (H,N), A <= 0

    expo = torch.einsum("bhl,hn->bhnl", dt, A)    # (B,H,N,L)
    S = torch.cumsum(expo, dim=-1)                # (B,H,N,L)

    if rows is not None:
        rows = rows.to(device)
        L_sel = rows.numel()
        S_l = S.index_select(-1, rows)            # (B,H,N,L_sel)
        logK = S_l.unsqueeze(-1) - S.unsqueeze(-2)  # (B,H,N,L_sel,L)
        if causal:
            j_idx = torch.arange(L, device=device)[None, :]
            l_idx = rows[:, None]
            mask = (j_idx > l_idx)                # (L_sel,L)
            logK = logK.masked_fill(mask[None, None, None, :, :], float("-inf"))
    else:
        L_sel = L
        logK = S.unsqueeze(-1) - S.unsqueeze(-2)  # (B,H,N,L,L)
        if causal:
            mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
            logK = logK.masked_fill(mask, float("-inf"))

    logKc = logK - torch.amax(logK, dim=-1, keepdim=True)
    if normalize == "softmax":
        weights = torch.softmax(logKc, dim=-1)
    elif normalize == "exp":
        weights = torch.exp(logKc)
    else:
        raise ValueError("normalize must be 'softmax' or 'exp'")

    C_rows = C_in.index_select(-1, rows) if rows is not None else C_in  # (B,N,L_sel)
    C_b = C_rows.unsqueeze(1).unsqueeze(-1).abs()                       # (B,1,N,L_sel,1)
    if B_is_per_head:
        B_b = B_in.unsqueeze(-2).abs()                                 # (B,H,N,1,L)
    else:
        B_b = B_in.unsqueeze(1).unsqueeze(-2).abs()                    # (B,1,N,1,L)

    contrib = weights * C_b * B_b                                      # (B,H,N,L_sel,L)
    attn_heads = contrib.sum(dim=2)                                    # (B,H,L_sel,L)

    attn_heads_g = None
    if return_per_group:
        assert ngroups is not None and (N % ngroups == 0), "invalid ngroups for N"
        contrib_g = contrib.reshape(Bsz, H, ngroups, N // ngroups, L_sel, L)
        attn_heads_g = (
            contrib_g.mean(dim=3) if reduce_states == "mean" else contrib_g.sum(dim=3)
        )  # (B,H,G,L_sel,L)
    return attn_heads, attn_heads_g
