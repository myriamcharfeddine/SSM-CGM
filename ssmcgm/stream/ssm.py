"""Streaming MES selective state-space block — pure PyTorch.

This is the streamable core of SSM-CGM-Stream. It reimplements the MES-style
Mamba-2 recurrence already validated in :mod:`ssmcgm.ops.mes_reference`
(scalar-per-head ``A``, shared value path ``x``, shared readout ``C``, **per-head
``B``** via ``B_head_scale``) but with two additions the fused ``mamba_ssm`` kernel
cannot give us:

* a **patient-specific initial state** ``h0`` that gradients flow into, and
* a true **single-step** update for deployment.

``scan`` (batched, training) and ``step`` (one timestep, streaming) run the *same*
recurrence, so ``scan``'s final output equals repeated ``step`` **by
construction** (no tolerance needed beyond float noise). The causal depthwise conv
is likewise consistent: ``scan`` left-pads with the carried ``conv_state`` and
returns the updated buffer, so streaming continues seamlessly after an
``encode_history``.

The recurrence (per head ``h``, batch ``b``, head-dim ``p``, state ``n``):

    state_t[h,p,n] = exp(dt_t[h]·A_h)·state_{t-1}[h,p,n] + dt_t[h]·B_t[h,n]·x_t[p]
    y_t[h,p]       = Σ_n C_t[h,n]·state_t[h,p,n] + D_h·x_t[p]

``mamba_style="mes"`` shares ``x`` and ``C`` across heads (attribution-comparable,
the paper's design); ``mamba_style="standard"`` uses a per-head value path.

Correctness first (spec §15): the scan is a sequential loop over L. A chunked /
parallel-segsum scan and an optional fused ``mamba_ssm`` backend are documented
follow-up optimizations; both must reproduce this recurrence.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:  # the official autotuned chunked-SSD Triton kernel (fastest, h0-differentiable)
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401
    _HAS_MAMBA_CHUNK = True
except Exception:  # pragma: no cover - falls back to the pure-PyTorch chunked scan
    _HAS_MAMBA_CHUNK = False


# ---------------------------------------------------------------------------
# gated RMSNorm (pure PyTorch; mamba_ssm's is a GPU-only Triton kernel)
# ---------------------------------------------------------------------------
class GatedRMSNorm(nn.Module):
    """RMSNorm gated by ``silu(z)`` (Mamba-2 default: gate then normalize)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = x * F.silu(z)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment-sum: ``out[...,i,j] = Σ_{j<k<=i} x_k`` for ``i>=j`` else ``-inf``.

    Used to build the intra-chunk decay matrix ``L = exp(segsum(A))`` (Mamba-2 SSD)."""
    T = x.shape[-1]
    x = x.unsqueeze(-1).expand(*x.shape, T)
    m1 = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), -1)
    x = x.masked_fill(~m1, 0)
    xs = torch.cumsum(x, dim=-2)
    m0 = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), 0)
    return xs.masked_fill(~m0, float("-inf"))


def ssd_chunk_scan(X, A, B, C, chunk_len: int, initial_states=None):
    """Chunked-parallel (Mamba-2 "SSD") evaluation of the diagonal SSM recurrence

        state_t = exp(A_t)·state_{t-1} + B_t·X_t ,   y_t = Σ_n C_t·state_t

    over a window, with an optional differentiable ``initial_states`` (``h0``). Pure
    PyTorch, so autograd yields ``dL/dh0`` for free; equivalent to the sequential scan to
    fp precision (validated to ~1e-15 fp64). Shapes: ``X (b,L,h,p)``, ``A (b,L,h)``,
    ``B/C (b,L,h,n)``, ``initial_states (b,h,p,n)``; returns ``(Y (b,L,h,p), final (b,h,p,n))``.
    """
    b, L, h, p = X.shape
    n = B.shape[-1]
    pad = (chunk_len - L % chunk_len) % chunk_len
    if pad:
        X = F.pad(X, (0, 0, 0, 0, 0, pad)); A = F.pad(A, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad)); C = F.pad(C, (0, 0, 0, 0, 0, pad))
    Lp, c, cl = L + pad, (L + pad) // chunk_len, chunk_len
    Xc = X.reshape(b, c, cl, h, p); Bc = B.reshape(b, c, cl, h, n); Cc = C.reshape(b, c, cl, h, n)
    Ac = A.reshape(b, c, cl, h).permute(0, 3, 1, 2)               # (b,h,c,cl)
    Acs = torch.cumsum(Ac, dim=-1)
    Lmat = torch.exp(_segsum(Ac))                                # (b,h,c,cl,cl)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", Cc, Bc, Lmat, Xc)
    decay_states = torch.exp(Acs[..., -1:] - Acs)                # (b,h,c,cl)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", Bc, decay_states, Xc)   # (b,c,h,p,n)
    init = (torch.zeros(b, 1, h, p, n, dtype=X.dtype, device=X.device)
            if initial_states is None else initial_states.unsqueeze(1))
    states = torch.cat([init, states], dim=1)                    # (b,c+1,h,p,n)
    decay_chunk = torch.exp(_segsum(F.pad(Acs[..., -1], (1, 0))))  # (b,h,c+1,c+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states_prev, final = new_states[:, :-1], new_states[:, -1]
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", Cc, states_prev, torch.exp(Acs))
    return (Y_diag + Y_off).reshape(b, Lp, h, p)[:, :L], final


class StreamingMESSSM(nn.Module):
    """One streaming MES-Mamba2 layer (pure PyTorch, ``scan`` + ``step`` + ``h0``).

    Parameter layout mirrors :class:`ssmcgm.modules.mes_mamba2.FastMESMamba2`
    (``in_proj``, depthwise ``conv1d``, ``A_log``, ``dt_bias``, ``D``,
    ``B_head_scale``, gated norm, ``out_proj``) so it is the *same* block — only
    the scan differs.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
        ngroups: int = 1,
        mamba_style: str = "mes",     # "mes" | "standard"
        x_share_mode: str = "mean",   # "mean" | "first"  (mes value path)
        conv_bias: bool = True,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        dt_limit: Optional[Tuple[float, float]] = None,
        # --- optional static-conditioned timescale (experimental ablation) ---
        static_timescale_mode: str = "none",   # "none" | "additive"
        e_s_dim: Optional[int] = None,
        delta_min: float = 1e-4,
        delta_max: float = 1.0,
        # --- scan backend (history encoding only; `step` is always sequential) ---
        scan_mode: str = "sequential",         # "sequential" | "chunked" (fast, h0-differentiable)
        chunk_len: int = 64,
    ):
        super().__init__()
        if scan_mode not in ("sequential", "chunked", "triton", "mamba"):
            raise ValueError("scan_mode must be 'sequential', 'chunked', 'triton' or 'mamba'")
        self.scan_mode = scan_mode
        self.chunk_len = int(chunk_len)
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.headdim = int(headdim)
        self.ngroups = int(ngroups)
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0, "expand*d_model must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim
        assert self.nheads % self.ngroups == 0, "nheads must be divisible by ngroups"
        if mamba_style not in ("mes", "standard"):
            raise ValueError("mamba_style must be 'mes' or 'standard'")
        self.mamba_style = mamba_style
        self.x_share_mode = x_share_mode
        self.dt_limit = dt_limit

        H, N, G, P = self.nheads, self.d_state, self.ngroups, self.headdim
        self.conv_dim = self.d_inner + 2 * G * N
        d_in_proj = 2 * self.d_inner + 2 * G * N + H        # z, x, B, C, dt

        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.d_conv,
                                groups=self.conv_dim, padding=0, bias=conv_bias)
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1.0, 16.0)))
        self.D = nn.Parameter(torch.ones(H))
        self.B_head_scale = nn.Parameter(torch.ones(H, G * N))  # per-head B (MES diagonal)

        # dt bias initialized so softplus(dt_bias) lands in [dt_min, dt_max]
        dt = torch.exp(torch.rand(H) * (torch.log(torch.tensor(dt_max)) -
                                        torch.log(torch.tensor(dt_min))) +
                       torch.log(torch.tensor(dt_min))).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))          # inverse softplus
        self.dt_bias = nn.Parameter(inv_dt)

        # static-conditioned timescale: Δ_t = softplus(W_δ·u_t + U_δ·e_s + b_δ).
        # U_δ is zero-initialized so "additive" starts identical to "none" and only
        # changes behavior if training learns to use it.
        if static_timescale_mode not in ("none", "additive"):
            raise ValueError("static_timescale_mode must be 'none' or 'additive'")
        self.static_timescale_mode = static_timescale_mode
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        if static_timescale_mode == "additive":
            if e_s_dim is None:
                raise ValueError("static_timescale_mode='additive' requires e_s_dim")
            self.static_delta_proj = nn.Linear(int(e_s_dim), H)
            nn.init.zeros_(self.static_delta_proj.weight)
            nn.init.zeros_(self.static_delta_proj.bias)
        else:
            self.static_delta_proj = None

        self.norm = GatedRMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    def zero_state(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        """Zero SSM state ``(B, H, P, N)``."""
        device = device or self.A_log.device
        dtype = dtype or self.A_log.dtype
        return torch.zeros(batch, self.nheads, self.headdim, self.d_state,
                           device=device, dtype=dtype)

    def expand_state(self, reduced: torch.Tensor) -> torch.Tensor:
        """Broadcast a reduced initial state ``(B, H, N)`` to ``(B, H, P, N)``."""
        B, H, N = reduced.shape
        return reduced.unsqueeze(2).expand(B, H, self.headdim, N).contiguous()

    def zero_conv(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        """Zero causal-conv buffer ``(B, conv_dim, d_conv-1)``."""
        device = device or self.A_log.device
        dtype = dtype or self.A_log.dtype
        return torch.zeros(batch, self.conv_dim, self.d_conv - 1, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # shared projection / per-head tensor construction
    # ------------------------------------------------------------------
    def _A(self) -> torch.Tensor:
        return -torch.exp(self.A_log)                       # (H,)

    def _group_index(self, device) -> torch.Tensor:
        heads_per_group = self.nheads // self.ngroups
        return torch.arange(self.nheads, device=device) // heads_per_group  # (H,)

    def _per_head_xBC(self, x, B, C):
        """Build per-head ``x (.,H,P)``, ``B (.,H,N)``, ``C (.,H,N)`` from the split
        conv output. Works for a full sequence (leading dims ``B,L``) or a single
        step (leading dim ``B``)."""
        lead = x.shape[:-1]
        H, P, N, G = self.nheads, self.headdim, self.d_state, self.ngroups
        x_heads = x.reshape(*lead, H, P)                    # (...,H,P)
        B_g = B.reshape(*lead, G, N)                        # (...,G,N)
        C_g = C.reshape(*lead, G, N)                        # (...,G,N)
        gidx = self._group_index(x.device)                  # (H,)
        B_head = B_g.index_select(-2, gidx)                 # (...,H,N)
        C_head = C_g.index_select(-2, gidx)                 # (...,H,N)
        # per-head B scaling (MES diagonal) — also applied in "standard" so the
        # block is a strict generalization; set scales to 1 to disable.
        B_head = B_head * self.B_head_scale.view(*([1] * len(lead)), H, N)
        if self.mamba_style == "mes":
            if self.x_share_mode == "first":
                x_shared = x_heads[..., :1, :]
            else:
                x_shared = x_heads.mean(dim=-2, keepdim=True)
            x_head = x_shared.expand(*lead, H, P)
        else:                                               # standard: per-head value
            x_head = x_heads
        return x_head.contiguous(), B_head.contiguous(), C_head.contiguous()

    def _project(self, u: torch.Tensor):
        """``in_proj`` + split into ``z, xBC, dt`` (no conv yet)."""
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1)
        return z, xBC, dt

    def _dt(self, dt_raw: torch.Tensor, static_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        pre = dt_raw + self.dt_bias
        if self.static_timescale_mode == "additive" and static_embedding is not None:
            ds = self.static_delta_proj(static_embedding)        # (B, H)
            pre = pre + (ds.unsqueeze(1) if pre.dim() == 3 else ds)  # broadcast over L for scan
        dt = F.softplus(pre)
        if self.static_timescale_mode == "additive":
            dt = dt.clamp(min=self.delta_min, max=self.delta_max)   # keep the SSM stable
        elif self.dt_limit is not None:
            dt = dt.clamp(min=self.dt_limit[0], max=self.dt_limit[1])
        return dt

    # ------------------------------------------------------------------
    # batched scan (training / history encoding)
    # ------------------------------------------------------------------
    def scan(
        self,
        u: torch.Tensor,                    # (B, L, d_model)
        state0: Optional[torch.Tensor] = None,   # (B, H, P, N)
        conv0: Optional[torch.Tensor] = None,    # (B, conv_dim, d_conv-1)
        record: bool = False,
        static_embedding: Optional[torch.Tensor] = None,   # (B, e_s_dim) for static timescale
    ):
        """Run the recurrence over a window. Returns
        ``(out (B,L,d_model), final_state (B,H,P,N), final_conv, cache?)``.
        """
        Bsz, L, _ = u.shape
        device, dtype = u.device, u.dtype
        z, xBC, dt_raw = self._project(u)                   # (B,L,*)

        # causal depthwise conv: left-pad with carried conv state, no internal pad
        xBC_t = xBC.transpose(1, 2)                         # (B, conv_dim, L)
        pad = conv0 if conv0 is not None else self.zero_conv(Bsz, device, dtype)
        padded = torch.cat([pad, xBC_t], dim=-1)            # (B, conv_dim, L+K-1)
        conv_out = self.conv1d(padded)[..., :L]             # (B, conv_dim, L)
        final_conv = padded[..., -(self.d_conv - 1):] if self.d_conv > 1 else None
        xBC = F.silu(conv_out.transpose(1, 2))              # (B, L, conv_dim)

        x, B, C = torch.split(
            xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x_h, B_h, C_h = self._per_head_xBC(x, B, C)         # (B,L,H,P),(B,L,H,N),(B,L,H,N)
        dt = self._dt(dt_raw, static_embedding)             # (B,L,H)
        A = self._A()                                       # (H,)
        state0_ = self.zero_state(Bsz, device, dtype) if state0 is None else state0

        use_mamba = (self.scan_mode == "mamba" and L > 1 and x_h.is_cuda and _HAS_MAMBA_CHUNK)
        use_triton = (self.scan_mode == "triton" and L > 1 and x_h.is_cuda
                      and self.mamba_style == "mes" and self.ngroups == 1)
        if use_mamba:
            # official autotuned chunked-SSD Triton kernel (fastest at training scale;
            # backprops into h0). MES recurrence is an SSD instance: dt is post-softplus,
            # A<=0 scalar/head, per-head B_h/C_h (ngroups=nheads). D·x added below.
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
            y_core, state = mamba_chunk_scan_combined(
                x_h, dt, A, B_h, C_h, chunk_size=self.chunk_len,
                initial_states=state0_, return_final_states=True, dt_softplus=False)
            y = y_core + self.D.view(1, 1, -1, 1) * x_h                # (B,L,H,P)
        elif use_triton:
            # fused h0-aware Triton kernel (single launch, O(1) state memory; shared x/C)
            from ..ops.triton.mes_mamba2_scan import mes_stream_scan_triton
            y_core, state = mes_stream_scan_triton(
                x_h[:, :, 0].contiguous(), dt, A, B_h, C_h[:, :, 0].contiguous(),
                state0_, self.chunk_len)
            y = y_core + self.D.view(1, 1, -1, 1) * x_h                # (B,L,H,P)
        elif (self.scan_mode in ("chunked", "triton", "mamba")) and L > 1:
            # chunked-parallel SSD: same recurrence, no Python time-loop, h0-differentiable
            # (also the portable fallback for "triton"/"mamba" off-CUDA or when unavailable).
            y_core, state = ssd_chunk_scan(
                dt.unsqueeze(-1) * x_h,        # X = dt·x        (B,L,H,P)
                dt * A.view(1, 1, -1),         # A = dt·A (log-decay)
                B_h, C_h, self.chunk_len, state0_)
            y = y_core + self.D.view(1, 1, -1, 1) * x_h                # (B,L,H,P)
        else:
            decay = torch.exp(dt * A.view(1, 1, -1))                   # (B,L,H)
            state = state0_
            ys = []
            for t in range(L):
                a_t = decay[:, t].unsqueeze(-1).unsqueeze(-1)          # (B,H,1,1)
                dBx = (dt[:, t].unsqueeze(-1).unsqueeze(-1)            # (B,H,1,1)
                       * B_h[:, t].unsqueeze(-2)                       # (B,H,1,N)
                       * x_h[:, t].unsqueeze(-1))                      # (B,H,P,1)
                state = a_t * state + dBx                              # (B,H,P,N)
                y_t = (C_h[:, t].unsqueeze(-2) * state).sum(-1)        # (B,H,P)
                y_t = y_t + self.D.view(1, -1, 1) * x_h[:, t]
                ys.append(y_t)
            y = torch.stack(ys, dim=1)                                 # (B,L,H,P)
        y = y.reshape(Bsz, L, self.d_inner)
        y = self.norm(y, z)
        out = self.out_proj(y)

        cache = self._build_cache(dt_raw, B_h, C_h) if record else None
        return out, state, final_conv, cache

    # ------------------------------------------------------------------
    # single streaming step (deployment)
    # ------------------------------------------------------------------
    def step(
        self,
        u_t: torch.Tensor,                  # (B, d_model)
        state: torch.Tensor,                # (B, H, P, N)
        conv: Optional[torch.Tensor] = None,    # (B, conv_dim, d_conv-1)
        static_embedding: Optional[torch.Tensor] = None,   # (B, e_s_dim)
    ):
        """Advance one timestep. Returns ``(y_t (B,d_model), new_state, new_conv)``."""
        Bsz = u_t.shape[0]
        z, xBC, dt_raw = self._project(u_t)                 # (B,*)

        if self.d_conv > 1:
            if conv is None:
                conv = self.zero_conv(Bsz, u_t.device, u_t.dtype)
            window = torch.cat([conv, xBC.unsqueeze(-1)], dim=-1)       # (B,conv_dim,K)
            w = self.conv1d.weight.squeeze(1)                          # (conv_dim,K)
            conv_out = (window * w.unsqueeze(0)).sum(-1)               # (B,conv_dim)
            if self.conv1d.bias is not None:
                conv_out = conv_out + self.conv1d.bias
            new_conv = window[..., 1:]
        else:
            conv_out = xBC
            new_conv = None
        xBC = F.silu(conv_out)

        x, B, C = torch.split(
            xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x_h, B_h, C_h = self._per_head_xBC(x, B, C)         # (B,H,P),(B,H,N),(B,H,N)
        dt = self._dt(dt_raw, static_embedding)             # (B,H)
        A = self._A()
        a = torch.exp(dt * A.view(1, -1))                   # (B,H)

        dBx = (dt.unsqueeze(-1).unsqueeze(-1)
               * B_h.unsqueeze(-2)                          # (B,H,1,N)
               * x_h.unsqueeze(-1))                         # (B,H,P,1)
        new_state = a.unsqueeze(-1).unsqueeze(-1) * state + dBx        # (B,H,P,N)
        y = (C_h.unsqueeze(-2) * new_state).sum(-1)                    # (B,H,P)
        y = y + self.D.view(1, -1, 1) * x_h
        y = y.reshape(Bsz, self.d_inner)
        y = self.norm(y, z)
        out = self.out_proj(y)
        return out, new_state, new_conv

    # ------------------------------------------------------------------
    # attribution cache (layout identical to FastMESMamba2._build_cache)
    # ------------------------------------------------------------------
    def _build_cache(self, dt_raw, B_h, C_h):
        H, N = self.nheads, self.d_state
        dt_save = (dt_raw + self.dt_bias).permute(0, 2, 1).detach()    # (B,H,L)
        A_log_exp = self.A_log.view(H, 1).expand(H, N).contiguous().detach()  # (H,N)
        B_in = B_h.permute(0, 2, 3, 1).detach()                       # (B,H,N,L)
        # C is shared across heads in MES; take head 0 -> (B,N,L)
        C_in = C_h[:, :, 0, :].permute(0, 2, 1).detach()              # (B,N,L)
        return (dt_save, A_log_exp, B_in, C_in)


class StreamingMESBlock(nn.Module):
    """Pre-norm residual wrapper around a :class:`StreamingMESSSM`."""

    def __init__(self, d_model: int, dropout: float = 0.1, **ssm_kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = StreamingMESSSM(d_model, **ssm_kwargs)
        self.drop = nn.Dropout(dropout)

    def scan(self, x, state0=None, conv0=None, record=False, static_embedding=None):
        out, state, conv, cache = self.ssm.scan(self.norm(x), state0, conv0, record=record,
                                                static_embedding=static_embedding)
        return x + self.drop(out), state, conv, cache

    def step(self, x_t, state, conv=None, static_embedding=None):
        out, state, conv = self.ssm.step(self.norm(x_t), state, conv, static_embedding=static_embedding)
        return x_t + self.drop(out), state, conv


class StreamingMESStack(nn.Module):
    """Stack of :class:`StreamingMESBlock`; per-layer state/conv; ``h_t`` = top output."""

    def __init__(self, d_model: int, depth: int = 2, dropout: float = 0.1, **ssm_kwargs):
        super().__init__()
        self.depth = int(depth)
        self.blocks = nn.ModuleList([
            StreamingMESBlock(d_model, dropout=dropout, **ssm_kwargs) for _ in range(depth)
        ])

    @property
    def nheads(self) -> int:
        return self.blocks[0].ssm.nheads

    @property
    def d_state(self) -> int:
        return self.blocks[0].ssm.d_state

    def scan(self, u, layer_states=None, conv_states=None, record=False, static_embedding=None):
        """Run history through the stack. ``layer_states`` / ``conv_states`` are
        per-layer initial states (or None). Returns
        ``(out (B,L,d_model), new_layer_states, new_conv_states, caches)``."""
        new_states, new_convs, caches = [], [], []
        x = u
        for i, blk in enumerate(self.blocks):
            s0 = None if layer_states is None else layer_states[i]
            c0 = None if conv_states is None else conv_states[i]
            x, s, c, cache = blk.scan(x, s0, c0, record=record, static_embedding=static_embedding)
            new_states.append(s)
            new_convs.append(c)
            caches.append(cache)
        return x, new_states, new_convs, (caches if record else None)

    def step(self, u_t, layer_states, conv_states=None, static_embedding=None):
        """One streaming step through the stack. Returns
        ``(y_t (B,d_model), new_layer_states, new_conv_states)``."""
        new_states, new_convs = [], []
        x = u_t
        for i, blk in enumerate(self.blocks):
            c0 = None if conv_states is None else conv_states[i]
            x, s, c = blk.step(x, layer_states[i], c0, static_embedding=static_embedding)
            new_states.append(s)
            new_convs.append(c)
        return x, new_states, new_convs
