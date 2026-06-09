"""MES hidden-attention reconstruction from a (small) attribution cache.

This is the *attribution mode*: it is never on the training hot path.  It takes
the detached cache produced by ``FastMESMamba2(..., return_cache=True)`` --

    cache = (dt_save, A_log_exp, B_in_h, C_in)
            (B,H,L)   (H,GN)      (B,H,GN,L) (B,GN,L)

-- and reconstructs the head-wise temporal-attribution maps the paper uses.

Two regimes:

* ``max_lags is None``: full (B, H, L_sel, L) causal map (memory ~ O(B H N L^2)
  before the state reduction; head-chunked to stay bounded).
* ``max_lags = K``: **banded** (B, H, L_sel, K+1) map -- for each target step
  only the last ``K`` source steps are scored.  Memory ~ O(B H N L_sel K).
  This is the recommended form for long contexts (e.g. ``max_lags=60``).

Selection of layers / heads / windows / target rows and CPU offload all live
here so attribution cost never leaks into ``mode="forecast"``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .mes_reference import hidden_attention as _hidden_attention_full


def _slice_heads(cache, heads):
    dt_save, A_log_exp, B_in_h, C_in = cache
    if heads is None:
        return cache
    heads = torch.as_tensor(list(heads), device=dt_save.device, dtype=torch.long)
    return (
        dt_save.index_select(1, heads),
        A_log_exp.index_select(0, heads),
        B_in_h.index_select(1, heads),
        C_in,
    )


@torch.no_grad()
def hidden_attention_banded(
    dt_raw: torch.Tensor,    # (B,H,L) pre-softplus
    A_log: torch.Tensor,     # (H,N)
    B_in: torch.Tensor,      # (B,H,N,L) per-head
    C_in: torch.Tensor,      # (B,N,L) shared
    *,
    max_lags: int,
    rows: Optional[torch.Tensor] = None,
    normalize: str = "softmax",
    return_per_group: bool = False,
    ngroups: Optional[int] = None,
    reduce_states: str = "sum",
    h_chunk: int = 4,
    compute_dtype: torch.dtype = torch.float32,
    out_device="cpu",
):
    """Banded hidden attention: only the last ``max_lags`` source steps per row.

    Returns ``attn`` (B,H,L_sel,K+1) where lag index ``k`` is source ``l-k``
    (``k=0`` is the target itself).  Optionally ``attn_g`` (B,H,G,L_sel,K+1).
    """
    Bsz, H, L = dt_raw.shape
    N = A_log.shape[1]
    K = int(max_lags)
    device = dt_raw.device
    if rows is None:
        rows = torch.arange(L, device=device)
    else:
        rows = rows.to(device)
    L_sel = rows.numel()

    dt = F.softplus(dt_raw.to(compute_dtype))               # (B,H,L)
    A = -torch.exp(A_log.to(compute_dtype))                 # (H,N)
    B_pos = B_in.abs().to(compute_dtype)
    C_pos = C_in.abs().to(compute_dtype)

    lags = torch.arange(K + 1, device=device)              # (K+1,)
    src = rows[:, None] - lags[None, :]                    # (L_sel, K+1)
    valid = src >= 0
    src_c = src.clamp(min=0)                                # for gather

    attn = torch.zeros((Bsz, H, L_sel, K + 1), dtype=compute_dtype, device=out_device)
    attn_g = None
    if return_per_group:
        assert ngroups is not None and (N % ngroups == 0)
        G = int(ngroups)
        attn_g = torch.zeros((Bsz, H, G, L_sel, K + 1), dtype=compute_dtype, device=out_device)

    C_rows = C_pos.index_select(-1, rows)                  # (B,N,L_sel)

    for h0 in range(0, H, h_chunk):
        h1 = min(h0 + h_chunk, H)
        dt_h = dt[:, h0:h1]                                # (B,hc,L)
        A_h = A[h0:h1]                                     # (hc,N)
        expo = torch.einsum("bhl,hn->bhnl", dt_h, A_h)     # (B,hc,N,L)
        S = torch.cumsum(expo, dim=-1)                     # (B,hc,N,L)

        S_tgt = S.index_select(-1, rows)                   # (B,hc,N,L_sel)
        # gather source S: build (B,hc,N,L_sel,K+1)
        src_flat = src_c.reshape(-1)                       # (L_sel*(K+1),)
        S_src = S.index_select(-1, src_flat).reshape(
            Bsz, h1 - h0, N, L_sel, K + 1
        )
        logK = S_tgt.unsqueeze(-1) - S_src                 # (B,hc,N,L_sel,K+1)
        logK = logK.masked_fill(~valid[None, None, None], float("-inf"))

        logKc = logK - torch.amax(logK, dim=-1, keepdim=True)
        if normalize == "softmax":
            w = torch.softmax(logKc, dim=-1)
        elif normalize == "exp":
            w = torch.exp(logKc)
        else:
            raise ValueError("normalize must be 'softmax' or 'exp'")

        # |B| at source, |C| at target
        B_h = B_pos[:, h0:h1]                              # (B,hc,N,L)
        B_src = B_h.index_select(-1, src_flat).reshape(
            Bsz, h1 - h0, N, L_sel, K + 1
        )
        C_b = C_rows.unsqueeze(1).unsqueeze(-1)            # (B,1,N,L_sel,1)
        contrib = w * C_b * B_src                          # (B,hc,N,L_sel,K+1)
        attn[:, h0:h1] = contrib.sum(dim=2).to(out_device)
        if return_per_group:
            cg = contrib.reshape(Bsz, h1 - h0, ngroups, N // ngroups, L_sel, K + 1)
            attn_g[:, h0:h1] = (
                cg.mean(dim=3) if reduce_states == "mean" else cg.sum(dim=3)
            ).to(out_device)

    return attn, attn_g


@torch.no_grad()
def compute_hidden_attention(
    cache,
    *,
    sample_idx: Optional[int] = None,
    rows: Optional[torch.Tensor] = None,
    max_lags: Optional[int] = None,
    heads: Optional[Sequence[int]] = None,
    ngroups: Optional[int] = None,
    normalize: str = "softmax",
    reduce_states: str = "sum",
    return_per_group: bool = False,
    h_chunk: int = 4,
    out_device="cpu",
    compute_dtype: torch.dtype = torch.float32,
):
    """Reconstruct MES hidden-attention maps from one layer's cache.

    Args:
        cache: ``(dt_save, A_log_exp, B_in_h, C_in)`` from ``return_cache=True``.
        sample_idx: restrict to a single batch element (avoids OOM).
        rows: target time indices to score (default: all).
        max_lags: if set, return the banded last-``K`` map (memory efficient);
            otherwise the full ``(B,H,L_sel,L)`` map.
        heads: subset of heads to compute.
        ngroups / return_per_group / reduce_states: per-group decomposition.
        out_device: where to place the (potentially large) result; default CPU.

    Returns:
        ``(attn, attn_g)``.  ``attn`` is (B,H,L_sel,K+1) when ``max_lags`` is set,
        else (B,H,L_sel,L).  ``attn_g`` is the per-group map or ``None``.
    """
    if return_per_group and ngroups is None:
        ngroups = 1  # MES default: a single shared B/C group
    cache = _slice_heads(cache, heads)
    dt_save, A_log_exp, B_in_h, C_in = cache
    if sample_idx is not None:
        dt_save = dt_save[sample_idx:sample_idx + 1]
        A_log_exp = A_log_exp
        B_in_h = B_in_h[sample_idx:sample_idx + 1]
        C_in = C_in[sample_idx:sample_idx + 1]

    if max_lags is not None:
        return hidden_attention_banded(
            dt_save, A_log_exp, B_in_h, C_in,
            max_lags=max_lags, rows=rows, normalize=normalize,
            return_per_group=return_per_group, ngroups=ngroups,
            reduce_states=reduce_states, h_chunk=h_chunk,
            compute_dtype=compute_dtype, out_device=out_device,
        )
    return _hidden_attention_full(
        dt_save, A_log_exp, B_in_h, C_in,
        rows=rows, normalize=normalize, return_per_group=return_per_group,
        ngroups=ngroups, reduce_states=reduce_states, compute_dtype=compute_dtype,
    )
