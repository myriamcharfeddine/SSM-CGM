"""Custom Triton selective-scan for the MES-Mamba2 structure.

This is the bespoke kernel called for by the rewrite spec (constraint C).  It
implements the MES recurrence directly -- shared value path ``x``, shared
readout ``C``, per-head ``B`` and scalar-per-head ``A`` -- in a single fused
launch, **without** folding into the official kernel and **without** a Python
per-head loop.

Design (forward)
----------------
One Triton program per ``(batch, head, headdim)`` triple.  Each program holds a
state vector of length ``d_state`` in registers and walks the sequence once:

    decay     = exp(dt_t * A_h)                       # scalar
    state[n]  = decay * state[n] + (dt_t * x_t) * B_t[n]
    y_t       = sum_n C_t[n] * state[n]  (+ D_hp * x_t)

The whole head axis is covered by the launch grid, so there is exactly one
kernel launch for all heads and layers -- the per-head loop is gone.

Autograd
--------
The forward uses the Triton kernel.  The backward is provided through a
``torch.autograd.Function`` that recomputes gradients with the differentiable
PyTorch reference (:func:`ssmcgm.ops.mes_reference.mes_scan_reference`).  This
is correct but not as fast as a hand-written fused backward -- documented as the
remaining kernel-level work.  For training, the fused official-kernel path
(``scan_backend="fused"``) is recommended; this kernel is intended for
inference / attribution and as the explicit custom-kernel reference.

This kernel is the parallel (closed-form) sibling of the banded hidden-attention
computation: both score source->target via the same per-head decay.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _mes_scan_fwd_kernel(
        x_ptr,        # (B, L, P)        shared value path
        dt_ptr,       # (B, H, L)        per-head dt (post-softplus)
        A_ptr,        # (H,)             scalar-per-head A (<= 0)
        B_ptr,        # (B, H, L, N)     per-head B
        C_ptr,        # (B, L, N)        shared C
        D_ptr,        # (H, P) or dummy
        out_ptr,      # (B, L, H, P)
        B_batch, H, L, P, N,
        sx_b, sx_l, sx_p,
        sdt_b, sdt_h, sdt_l,
        sB_b, sB_h, sB_l, sB_n,
        sC_b, sC_l, sC_n,
        sO_b, sO_l, sO_h, sO_p,
        HAS_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        p = pid % P
        h = (pid // P) % H
        b = pid // (P * H)

        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        A_h = tl.load(A_ptr + h)                       # scalar
        state = tl.zeros((BLOCK_N,), dtype=tl.float32)

        x_base = x_ptr + b * sx_b + p * sx_p
        dt_base = dt_ptr + b * sdt_b + h * sdt_h
        B_base = B_ptr + b * sB_b + h * sB_h
        C_base = C_ptr + b * sC_b
        out_base = out_ptr + b * sO_b + h * sO_h + p * sO_p
        if HAS_D:
            d_hp = tl.load(D_ptr + h * P + p)
        else:
            d_hp = 0.0

        for t in range(0, L):
            dt_t = tl.load(dt_base + t * sdt_l).to(tl.float32)
            x_t = tl.load(x_base + t * sx_l).to(tl.float32)
            b_row = tl.load(B_base + t * sB_l + offs_n * sB_n, mask=mask_n, other=0.0).to(tl.float32)
            c_row = tl.load(C_base + t * sC_l + offs_n * sC_n, mask=mask_n, other=0.0).to(tl.float32)

            decay = tl.exp(dt_t * A_h)
            state = decay * state + (dt_t * x_t) * b_row
            y_t = tl.sum(c_row * state, axis=0) + d_hp * x_t
            tl.store(out_base + t * sO_l, y_t)


def _mes_scan_fwd(x_shared, dt, A, B_h, C_shared, D):
    """Run the Triton forward. Layouts:
        x_shared: (B,L,P)   dt: (B,H,L)   A: (H,)
        B_h: (B,H,L,N)      C_shared: (B,L,N)   D: (H,P) or None
    Returns out (B,L,H,P).
    """
    B, L, P = x_shared.shape
    H = dt.shape[1]
    N = B_h.shape[-1]
    out = torch.empty((B, L, H, P), device=x_shared.device, dtype=torch.float32)

    x32 = x_shared.contiguous().float()
    dt32 = dt.contiguous().float()
    A32 = A.contiguous().float()
    B32 = B_h.contiguous().float()
    C32 = C_shared.contiguous().float()
    has_D = D is not None
    D32 = (D.contiguous().float() if has_D
           else torch.empty((1,), device=x_shared.device, dtype=torch.float32))

    BLOCK_N = triton.next_power_of_2(N)
    grid = (B * H * P,)
    _mes_scan_fwd_kernel[grid](
        x32, dt32, A32, B32, C32, D32, out,
        B, H, L, P, N,
        x32.stride(0), x32.stride(1), x32.stride(2),
        dt32.stride(0), dt32.stride(1), dt32.stride(2),
        B32.stride(0), B32.stride(1), B32.stride(2), B32.stride(3),
        C32.stride(0), C32.stride(1), C32.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        HAS_D=has_D, BLOCK_N=BLOCK_N,
    )
    return out


class _MESScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_shared, dt, A, B_h, C_shared, D, chunk_size=64):
        # x_shared (B,L,P) dt (B,L,H) A (H,) B_h (B,L,H,N) C_shared (B,L,N)
        ctx.save_for_backward(x_shared, dt, A, B_h, C_shared,
                              D if D is not None else torch.empty(0))
        ctx.has_D = D is not None
        ctx.chunk_size = chunk_size
        dt_bhl = dt.transpose(1, 2).contiguous()          # (B,H,L)
        B_bhln = B_h.permute(0, 2, 1, 3).contiguous()     # (B,H,L,N)
        out = _mes_scan_fwd(x_shared, dt_bhl, A, B_bhln, C_shared, D)
        return out.to(x_shared.dtype)                     # (B,L,H,P)

    @staticmethod
    def backward(ctx, grad_out):
        # Gradients via the STABLE CHUNKED SSD (O(L·chunk) memory), not the old
        # O(L²) naive recompute. Same math as the Triton forward, so the grads
        # are correct; chunked autograd keeps backward memory linear in L.
        from ..mes_chunked import mes_ssd_chunked
        x_shared, dt, A, B_h, C_shared, D = ctx.saved_tensors
        D = D if ctx.has_D else None
        with torch.enable_grad():
            xs = x_shared.detach().requires_grad_(x_shared.requires_grad)
            dtt = dt.detach().requires_grad_(dt.requires_grad)
            Ad = A.detach().requires_grad_(A.requires_grad)
            Bd = B_h.detach().requires_grad_(B_h.requires_grad)
            Cd = C_shared.detach().requires_grad_(C_shared.requires_grad)
            Dd = D.detach().requires_grad_(D.requires_grad) if D is not None else None
            y = mes_ssd_chunked(xs, dtt, Ad, Bd, Cd, D=Dd, chunk_size=ctx.chunk_size)
            inputs = [t for t in (xs, dtt, Ad, Bd, Cd, Dd) if t is not None]
            grads = torch.autograd.grad(y, inputs, grad_out.to(y.dtype),
                                        allow_unused=True)
        gi = iter(grads)
        gx = next(gi) if xs.requires_grad else None
        gdt = next(gi) if dtt.requires_grad else None
        gA = next(gi) if Ad.requires_grad else None
        gB = next(gi) if Bd.requires_grad else None
        gC = next(gi) if Cd.requires_grad else None
        gD = next(gi) if (Dd is not None and Dd.requires_grad) else None
        return gx, gdt, gA, gB, gC, gD, None  # last None for chunk_size


def mes_selective_scan_triton(x_shared, dt, A, B_h, C_shared, D=None, chunk_size=64):
    """MES selective scan via the custom Triton kernel.

    Forward uses the custom sequential Triton kernel; backward uses the stable
    chunked SSD (O(L·chunk) memory). ``chunk_size`` controls the backward chunking.

    Args:
        x_shared: (B, L, P) shared value path.
        dt:       (B, L, H) per-head timesteps (post-softplus).
        A:        (H,) scalar-per-head dynamics (<= 0).
        B_h:      (B, L, H, N) per-head input matrix.
        C_shared: (B, L, N) shared readout.
        D:        (H,) / (H, P) / None.
    Returns:
        y: (B, L, H, P)
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton is not available")
    if D is not None and D.dim() == 1:        # (H,) -> (H,P)
        D = D.view(-1, 1).expand(-1, x_shared.shape[-1]).contiguous()
    return _MESScanTriton.apply(x_shared, dt, A, B_h, C_shared, D, chunk_size)


# ===========================================================================
# Streaming variant: h0-aware fused forward + final-state out (deployment /
# long-context inference). One launch, O(1) state memory (no materialized
# intra-chunk L matrix). Backward routes through the differentiable chunked SSD
# so the patient h0 gets an exact gradient (dL/dh0) — the fused kernel the agenda
# called for. NOTE: this in-register sequential walk is ~4x slower than the
# chunked-torch `ssd_chunk_scan` on bulk GPU work, which stays the training/eval
# default; this kernel is for h0-differentiable single-launch inference + as the
# explicit custom-kernel reference.
# ===========================================================================
if _HAS_TRITON:

    @triton.jit
    def _mes_stream_scan_fwd_kernel(
        x_ptr,        # (B, L, P)        shared value path
        dt_ptr,       # (B, H, L)        per-head dt (post-softplus)
        A_ptr,        # (H,)             scalar-per-head A (<= 0)
        B_ptr,        # (B, H, L, N)     per-head B
        C_ptr,        # (B, L, N)        shared C
        h0_ptr,       # (B, H, P, N)     initial state
        out_ptr,      # (B, L, H, P)     y_core = C·state (NO D term)
        final_ptr,    # (B, H, P, N)     final state
        B_batch, H, L, P, N,
        sx_b, sx_l, sx_p,
        sdt_b, sdt_h, sdt_l,
        sB_b, sB_h, sB_l, sB_n,
        sC_b, sC_l, sC_n,
        sh_b, sh_h, sh_p, sh_n,
        sO_b, sO_l, sO_h, sO_p,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        p = pid % P
        h = (pid // P) % H
        b = pid // (P * H)
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        A_h = tl.load(A_ptr + h)
        h0_base = h0_ptr + b * sh_b + h * sh_h + p * sh_p
        state = tl.load(h0_base + offs_n * sh_n, mask=mask_n, other=0.0).to(tl.float32)

        x_base = x_ptr + b * sx_b + p * sx_p
        dt_base = dt_ptr + b * sdt_b + h * sdt_h
        B_base = B_ptr + b * sB_b + h * sB_h
        C_base = C_ptr + b * sC_b
        out_base = out_ptr + b * sO_b + h * sO_h + p * sO_p

        for t in range(0, L):
            dt_t = tl.load(dt_base + t * sdt_l).to(tl.float32)
            x_t = tl.load(x_base + t * sx_l).to(tl.float32)
            b_row = tl.load(B_base + t * sB_l + offs_n * sB_n, mask=mask_n, other=0.0).to(tl.float32)
            c_row = tl.load(C_base + t * sC_l + offs_n * sC_n, mask=mask_n, other=0.0).to(tl.float32)
            decay = tl.exp(dt_t * A_h)
            state = decay * state + (dt_t * x_t) * b_row
            y_t = tl.sum(c_row * state, axis=0)
            tl.store(out_base + t * sO_l, y_t)
        final_base = final_ptr + b * sh_b + h * sh_h + p * sh_p
        tl.store(final_base + offs_n * sh_n, state, mask=mask_n)


def _mes_stream_scan_fwd(x_shared, dt_bhl, A, B_bhln, C_shared, h0):
    """Triton forward. x_shared(B,L,P) dt_bhl(B,H,L) A(H,) B_bhln(B,H,L,N)
    C_shared(B,L,N) h0(B,H,P,N) -> (out (B,L,H,P), final (B,H,P,N))."""
    B, L, P = x_shared.shape
    H, N = dt_bhl.shape[1], B_bhln.shape[-1]
    x32, dt32, A32 = x_shared.float(), dt_bhl.float(), A.float()
    B32, C32, h32 = B_bhln.float(), C_shared.float(), h0.float()
    out = torch.empty((B, L, H, P), device=x_shared.device, dtype=torch.float32)
    final = torch.empty((B, H, P, N), device=x_shared.device, dtype=torch.float32)
    grid = (B * H * P,)
    _mes_stream_scan_fwd_kernel[grid](
        x32, dt32, A32, B32, C32, h32, out, final,
        B, H, L, P, N,
        x32.stride(0), x32.stride(1), x32.stride(2),
        dt32.stride(0), dt32.stride(1), dt32.stride(2),
        B32.stride(0), B32.stride(1), B32.stride(2), B32.stride(3),
        C32.stride(0), C32.stride(1), C32.stride(2),
        h32.stride(0), h32.stride(1), h32.stride(2), h32.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_N=triton.next_power_of_2(N),
    )
    return out, final


class _MESStreamScanTriton(torch.autograd.Function):
    """Fused h0-aware MES scan: Triton forward; backward via the differentiable
    chunked SSD (exact grads incl. dL/dh0)."""

    @staticmethod
    def forward(ctx, x_shared, dt, A, B_h, C_shared, h0, chunk_size=64):
        # x_shared (B,L,P) dt (B,L,H) A (H,) B_h (B,L,H,N) C_shared (B,L,N) h0 (B,H,P,N)
        ctx.save_for_backward(x_shared, dt, A, B_h, C_shared, h0)
        ctx.chunk_size = chunk_size
        dt_bhl = dt.transpose(1, 2).contiguous()          # (B,H,L)
        B_bhln = B_h.permute(0, 2, 1, 3).contiguous()     # (B,H,L,N)
        out, final = _mes_stream_scan_fwd(x_shared, dt_bhl, A, B_bhln, C_shared, h0)
        return out.to(x_shared.dtype), final.to(x_shared.dtype)

    @staticmethod
    def backward(ctx, grad_out, grad_final):
        from ...stream.ssm import ssd_chunk_scan
        x_shared, dt, A, B_h, C_shared, h0 = ctx.saved_tensors
        B, L, P = x_shared.shape
        H, N = dt.shape[-1], B_h.shape[-1]
        with torch.enable_grad():
            xs = x_shared.detach().requires_grad_(x_shared.requires_grad)
            dtt = dt.detach().requires_grad_(dt.requires_grad)
            Ad = A.detach().requires_grad_(A.requires_grad)
            Bd = B_h.detach().requires_grad_(B_h.requires_grad)
            Cs = C_shared.detach().requires_grad_(C_shared.requires_grad)
            h0d = h0.detach().requires_grad_(h0.requires_grad)
            x_h = xs.unsqueeze(2).expand(B, L, H, P)
            C_h = Cs.unsqueeze(2).expand(B, L, H, N)
            y, fin = ssd_chunk_scan(dtt.unsqueeze(-1) * x_h, dtt * Ad.view(1, 1, -1),
                                    Bd, C_h, ctx.chunk_size, h0d)
            leaves = [t for t in (xs, dtt, Ad, Bd, Cs, h0d)]
            grads = torch.autograd.grad([y, fin], leaves, [grad_out.to(y.dtype), grad_final.to(fin.dtype)],
                                        allow_unused=True)
        return (*grads, None)   # last None = chunk_size


def mes_stream_scan_triton(x_shared, dt, A, B_h, C_shared, h0, chunk_size=64):
    """h0-aware MES scan via the fused Triton kernel; exact dL/dh0 backward.

    Returns ``(y_core (B,L,H,P), final_state (B,H,P,N))`` where ``y_core = Σ_n C·state``
    (the ``D·x`` skip term is added by the caller). Requires the MES layout (shared
    ``x``/``C``)."""
    if not _HAS_TRITON:
        raise RuntimeError("triton is not available")
    return _MESStreamScanTriton.apply(x_shared, dt, A, B_h, C_shared, h0, chunk_size)
