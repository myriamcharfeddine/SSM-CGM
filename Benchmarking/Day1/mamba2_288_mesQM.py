import torch
from torch import nn
from typing import Tuple, Dict
import torch
from torch import nn
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except Exception:
    mamba_chunk_scan_combined = None

from mamba_ssm import Mamba2

import os
import pickle
import gc
import io
import numpy as np
import pandas as pd
import torch
from torch import nn
from mamba_ssm.modules.mamba_simple import Mamba  # your Mamba
from pytorch_forecasting import TimeSeriesDataSet  # 🔧 FIX 1 – correc--t import
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.strategies import DDPStrategy

# from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader
from pytorch_forecasting.utils import move_to_device  # ✔ recursive

# Checkpointing and calls
import glob
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from google.cloud import storage
import datetime

# GCS constants
_BUCKET_NAME = "cgmproject2025"
_BASE_PREFIX = "models/predictions"
_gcs_client = storage.Client()


# -------------------------
# Hyperparam presets
# -------------------------
param_24 = {
    "dataset": {"context_length": 288, "horizon": 12, "batch_size": 32},
    "mamba_tft_init": {"mamba_depth": 4, "mamba_dropout": 0.2},
    "mamba_block": {
        "d_model": 128,
        "dropout": 0.2,
        "return_hidden_attn": False,
        "d_state": 128,   # ↑ un peu pour compenser ngroups=1 (avant 96)
        "d_conv": 8,
        "expand": 4,
        "headdim": 64,
        "ngroups": 1      # 👈 OBLIGATOIRE avec x_shared (nheads=1 à l’appel kernel)
    },
    "mamba2_mes_runtime": {
        "mes_diag": True,
        "x_share_mode": "mean",
        "chunk_size": 128, #256,
        "dt_limit": (1e-3, 10.0),
        "learnable_init_states": True,
        "D": None
    },
    "checkpoint": {True}
}

##########Improvement_param#######################

# A100: active TF32 (accelère les matmuls en FP32) + cuDNN autotune
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# Sanity checks GPU/bf16
assert torch.cuda.is_available(), "CUDA non dispo"
name = torch.cuda.get_device_name(0)
cap  = torch.cuda.get_device_capability(0)
print(f"[GPU] {name}  CC={cap}, BF16={'OK' if torch.cuda.is_bf16_supported() else 'NO'}")


# =============================
# Utility Functions
# =============================
def log_memory(message=""):
    """Log CPU and GPU memory usage."""
    pid = os.getpid()
    #process = psutil.Process(pid)
    #mem_cpu = process.memory_info().rss / 1e9  # in GB
    if torch.cuda.is_available():
        mem_gpu_allocated = torch.cuda.memory_allocated() / 1e9
        mem_gpu_reserved = torch.cuda.memory_reserved() / 1e9
    else:
        mem_gpu_allocated = mem_gpu_reserved = 0.0
    print(f"[{datetime.datetime.now()}] {message}")
    #print(f"ðŸ§  CPU Mem used: {mem_cpu:.2f} GB")
    print(f"GPU Mem allocated: {mem_gpu_allocated:.2f} GB | reserved: {mem_gpu_reserved:.2f} GB")


def upload_latest_checkpoint_to_gcs(local_dir, bucket_name, gcs_prefix):
    """Upload all .ckpt files in local_dir to GCS under gcs_prefix."""
    bucket = _gcs_client.bucket(bucket_name)
    for file_path in glob.glob(f"{local_dir}/*.ckpt"):
        blob = bucket.blob(f"{gcs_prefix}/{os.path.basename(file_path)}")
        blob.upload_from_filename(file_path)
        print(f"Uploaded {file_path} to gs://{bucket_name}/{gcs_prefix}/")

def download_latest_checkpoint_from_gcs(local_dir, bucket_name, gcs_prefix):
    """Download the latest .ckpt from GCS to local_dir. Returns local path or None."""
    bucket = _gcs_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    ckpt_blobs = [b for b in blobs if b.name.endswith('.ckpt')]
    if not ckpt_blobs:
        return None
    latest_blob = max(ckpt_blobs, key=lambda b: b.updated)
    local_path = os.path.join(local_dir, os.path.basename(latest_blob.name))
    latest_blob.download_to_filename(local_path)
    print(f"Downloaded {latest_blob.name} to {local_path}")
    return local_path


# -------------------------
# QuantileMSE loss
# -------------------------
class QuantileLossWithMSE(QuantileLoss):
    def __init__(self, quantiles=(0.1, 0.5, 0.9), mse_weight=0.3, huber_delta=None, **kwargs):
        super().__init__(quantiles=quantiles, **kwargs)  # <- LightningMetric OK
        assert 0.5 in self.quantiles, "Le quantile 0.5 (médiane) doit être présent."
        self.mse_weight = float(mse_weight)
        self.huber_delta = huber_delta

    def loss(self, y_pred, target):
        # Perte pinball classique (déjà moyennée)
        qloss = super().loss(y_pred, target)

        # Médiane
        midx = self.quantiles.index(0.5)
        median = y_pred[..., midx]

        # MSE/Huber, moyenne comme QuantileLoss
        if self.huber_delta is None:
            mse = torch.sqrt(F.mse_loss(median, target, reduction='mean'))

        else:
            mse = F.smooth_l1_loss(median, target, beta=self.huber_delta, reduction='mean')

        return qloss + self.mse_weight * mse


# -------------------------
# Blocs Mamba MES
# -------------------------

class Mamba2MESIdentical(Mamba2):
    """
    Mamba-2 en mode MES identique train/inférence:
      - Chemin non-fused (même code path).
      - VRAI MES : B par tête, C et X partagés entre têtes.
      - Compat A_log: (H,) ou (N,) ou (G*N,) ou (G,N)
      - Compat D: (H,P) ou (H,) ou (P,) ou scalaire; passé par tête au kernel.

    get_saved_internals() -> (dt_raw: B,H,L; A_log: H,G*N; B_in_h: B,H,G*N,L; C_in: B,G*N,L)
    """

    def __init__(
        self, *args,
        mes_diag: bool = True,
        record_default: bool = False,
        save_to_cpu: bool = False,
        debug_kernel_shapes: bool = False, #True,
        x_share_mode: str = "mean",  # "mean" (par défaut) ou "first"
        **kwargs
    ):
        # forcer chemin non-fused
        kwargs = {**kwargs, "use_mem_eff_path": False}
        super().__init__(*args, **kwargs)

        if mamba_chunk_scan_combined is None:
            raise RuntimeError("mamba_chunk_scan_combined introuvable (installe les ops Triton mamba-ssm).")

        self._record = bool(record_default)
        self._save_to_cpu = bool(save_to_cpu)
        self._saved_hidden = None
        self._debug_once = bool(debug_kernel_shapes)  # imprime les shapes une fois
        self.x_share_mode = str(x_share_mode)

        # --- MES params: B par tête ---
        N_total = self.ngroups * self.d_state
        self.mes_diag = bool(mes_diag)
        if self.mes_diag:
            self.B_head_scale = nn.Parameter(torch.ones(self.nheads, N_total))  # (H, G*N)
        else:
            self.B_head_mix = nn.Parameter(torch.zeros(self.nheads, N_total, N_total))  # (H, G*N, G*N)
            with torch.no_grad():
                for h in range(self.nheads):
                    self.B_head_mix[h].zero_()
                    self.B_head_mix[h].diagonal().fill_(1.0)

        # compat attributs suivant versions
        if not hasattr(self, "learnable_init_states"):
            self.learnable_init_states = False
        if not hasattr(self, "init_states"):
            self.init_states = None
        if not hasattr(self, "dt_limit"):
            self.dt_limit = (0.0, float("inf"))

        # garde-fou headdim
        assert (self.d_inner % self.headdim) == 0, \
            f"d_inner={self.d_inner} doit être divisible par headdim={self.headdim}"

    def set_record(self, on: bool = True):
        self._record = bool(on)

    def set_debug(self, on: bool = True):
        self._debug_once = bool(on)

    def get_saved_internals(self):
        return self._saved_hidden

    def _normalize_A(self):
        """Retourne (mode, A_head, A_scan) avec:
           - mode='per_head' -> A_head: (H,), A_scan=None
           - mode='per_state'-> A_head=None, A_scan: (G,N)
        """
        A_raw = self.A_log
        H, G, N = self.nheads, self.ngroups, self.d_state
        GN = G * N

        if A_raw.numel() == H:
            return "per_head", (-torch.exp(A_raw)).contiguous(), None
        elif A_raw.numel() == N:
            A_scan = (-torch.exp(A_raw)).view(1, N).expand(G, -1).contiguous()
            return "per_state", None, A_scan
        elif A_raw.numel() == GN:
            A_scan = (-torch.exp(A_raw)).view(G, N).contiguous()
            return "per_state", None, A_scan
        else:
            raise RuntimeError(
                f"A_log shape inattendue: {tuple(A_raw.shape)} ; "
                f"attendu H={H} ou N={N} ou G*N={GN}"
            )

    def _normalize_D(self, device, dtype):
        """Normalise self.D pour obtenir D_scan:
           - si None -> None
           - sinon -> (H,P) ou (H,), dtype/device ok
        """
        if not hasattr(self, "D") or self.D is None:
            return None

        D_raw = self.D
        if not isinstance(D_raw, torch.Tensor):
            D_raw = torch.as_tensor(D_raw)
        D_raw = D_raw.to(device=device, dtype=dtype)

        H, P = self.nheads, self.headdim
        if D_raw.dim() == 0:
            # scalaire -> (H,P)
            return D_raw.view(1, 1).expand(H, P).contiguous()
        if D_raw.dim() == 1:
            if D_raw.numel() == H:
                return D_raw.contiguous()  # (H,)
            if D_raw.numel() == P:
                return D_raw.view(1, P).expand(H, -1).contiguous()  # (H,P)
            if D_raw.numel() == self.d_inner:
                return D_raw.view(H, P).contiguous()  # (H,P)
            # fallback si possible
            if (D_raw.numel() % P) == 0 and (D_raw.numel() // P) == H:
                return D_raw.view(H, P).contiguous()
            raise RuntimeError(f"D shape inattendue (1D): {tuple(D_raw.shape)}")
        if D_raw.dim() == 2:
            if D_raw.shape == (H, P) or D_raw.shape == (H,):
                return D_raw.contiguous()
            if D_raw.shape == (P, H):
                return D_raw.t().contiguous()
            if D_raw.numel() == self.d_inner:
                return D_raw.reshape(H, P).contiguous()
            raise RuntimeError(f"D shape inattendue (2D): {tuple(D_raw.shape)}")
        raise RuntimeError(f"D dim inattendue: {D_raw.dim()}")

    def forward(self, u: torch.Tensor, seq_idx=None):
        """
        u: (B, L, D_model)
        """
        Bsz, seqlen, _ = u.shape

        # (1) in-proj -> [z, xBC, dt]
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner,
             self.d_inner + 2 * self.ngroups * self.d_state,
             self.nheads],
            dim=-1,
        )

        # (2) dt raw & pos (par tête)
        dt_raw = dt + self.dt_bias            # (B, L, H)
        dt_pos = F.softplus(dt_raw)           # (B, L, H)

        # (3) conv1d + SiLU sur xBC
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        xBC = xBC[:, :seqlen, :]

        # (4) split -> x, B_shared, C_shared
        x, B_shared, C_shared = torch.split(
            xBC,
            [self.d_inner,
             self.ngroups * self.d_state,
             self.ngroups * self.d_state],
            dim=-1,
        )

        # (5) construire B_h (B par tête)
        if self.mes_diag:
            B_h = B_shared.unsqueeze(2) * self.B_head_scale.unsqueeze(0).unsqueeze(0)  # (B,L,H,GN)
        else:
            B_h = torch.einsum("hnm,blm->blhn", self.B_head_mix, B_shared)             # (B,L,H,GN)

        # (6) préparer pour scan
        # MES strict: X partagé entre têtes (même entrée x pour toutes les têtes)
        x_heads  = rearrange(x, "b l (h p) -> b l h p", p=self.headdim).contiguous()   # (B,L,H,P)
        if self.x_share_mode == "mean":
            x_shared = x_heads.mean(dim=2, keepdim=True).contiguous()                 # (B,L,1,P)
        elif self.x_share_mode == "first":
            x_shared = x_heads[:, :, :1, :].contiguous()                               # (B,L,1,P)
        else:
            raise ValueError("x_share_mode doit être 'mean' ou 'first'")
        C_scan   = rearrange(C_shared, "b l (g n) -> b l g n", g=self.ngroups).contiguous()   # (B,L,G,N)

        # A & D normalisés
        A_mode, A_head, A_scan = self._normalize_A()
        D_scan = self._normalize_D(device=x.device, dtype=x.dtype)  # (H,P) ou (H,) ou None

        # kwargs communs (sans D ici !)
        initial_states = self.init_states if self.learnable_init_states else None
        scan_common = dict(
            chunk_size=self.chunk_size,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
        )
        if self.dt_limit != (0.0, float("inf")):
            scan_common["dt_limit"] = self.dt_limit

        # debug (une seule fois)
        if self._debug_once:
            print("[Mamba2MES] DEBUG SHAPES")
            #print("  u:", tuple(u.shape))
            #print("  x_heads:", tuple(x_heads.shape), "x_shared:", tuple(x_shared.shape))
            #print("  dt_raw:", tuple(dt_raw.shape), "dt_pos:", tuple(dt_pos.shape))
            #print("  B_h:", tuple(B_h.shape), "C_scan:", tuple(C_scan.shape))
            #print("  headdim:", self.headdim, "nheads:", self.nheads,
            #      "ngroups:", self.ngroups, "d_state:", self.d_state)
            if A_mode == "per_head":
                print("  A_mode: per_head ; A_head:", tuple(A_head.shape))
            else:
                print("  A_mode: per_state ; A_scan:", tuple(A_scan.shape))
            if D_scan is None:
                print("  D: None")
            else:
                if D_scan.dim() == 2:
                    print("  D_scan:", tuple(D_scan.shape), "(H,P)")
                else:
                    print("  D_scan:", tuple(D_scan.shape), "(H,)")

        y_heads = []
        for h in range(self.nheads):
            B_scan_h = rearrange(B_h[:, :, h, :], "b l (g n) -> b l g n", g=self.ngroups).contiguous()  # (B,L,G,N)

            # A par appel
            if A_mode == "per_state":
                A_arg = A_scan                               # (G,N)
            else:
                A_arg = A_head[h:h+1].contiguous()           # (1,)

            # D par appel (IMPORTANT: nheads=1 attendu par le kernel sur cet appel)
            if D_scan is None:
                D_arg = None
            else:
                if D_scan.dim() == 2:
                    D_arg = D_scan[h:h+1, :].contiguous()    # (1,P)
                else:
                    D_arg = D_scan[h:h+1].contiguous()       # (1,)

            # debug par-tête (première tête uniquement)
            if h == 0 and self._debug_once:
                print("  [per-head example] x:", tuple(x_shared.shape),
                      "dt:", tuple(dt_pos[:, :, h:h+1].shape),
                      "A:", (tuple(A_arg.shape) if isinstance(A_arg, torch.Tensor) else None),
                      "B:", tuple(B_scan_h.shape),
                      "C:", tuple(C_scan.shape),
                      "D:", (tuple(D_arg.shape) if isinstance(D_arg, torch.Tensor) else None))
                # couper le debug après le 1er exemple pour éviter le spam
                self._debug_once = False

            y_h = mamba_chunk_scan_combined(
                x_shared,                               # (B,L,1,P) -> X partagé
                dt_pos[:, :, h:h+1].contiguous(),      # (B,L,1)
                A_arg,                                  # (G,N) ou (1,)
                B_scan_h,                               # (B,L,G,N)
                C_scan,                                 # (B,L,G,N)
                D=D_arg,                                # (1,P) ou (1,) ou None
                **scan_common,
            )  # -> (B,L,1,P)
            y_heads.append(y_h)

        y = torch.cat(y_heads, dim=2)                         # (B,L,H,P)
        y = rearrange(y, "b l h p -> b l (h p)").contiguous() # (B,L,d_inner)

        # (7) gating+norm + out-proj
        y = self.norm(y, z)
        out = self.out_proj(y)

        # (8) enregistrement optionnel
        if self._record:
            dt_save = rearrange(dt_raw, "b l h -> b h l").detach()  # (B,H,L)
            G, N, H = self.ngroups, self.d_state, self.nheads
            GN = G * N
            if A_mode == "per_state":
                A_states_log = self.A_log
                if A_states_log.numel() == N:
                    A_states_log = A_states_log.view(1, N).expand(G, -1)  # (G,N)
                else:
                    A_states_log = A_states_log.view(G, N)                # (G,N)
                A_log_exp = A_states_log.reshape(1, GN).expand(H, -1).contiguous().detach()  # (H,GN)
            else:
                A_log_exp = self.A_log.view(H, 1).expand(H, GN).contiguous().detach()       # (H,GN)

            B_in_h = rearrange(B_h, "b l h n -> b h n l").detach()   # (B,H,GN,L)
            C_in   = rearrange(C_shared, "b l n -> b n l").detach()  # (B,GN,L)

            if self._save_to_cpu:
                dt_save = dt_save.cpu(); A_log_exp = A_log_exp.cpu()
                B_in_h = B_in_h.cpu();   C_in = C_in.cpu()

            self._saved_hidden = (dt_save, A_log_exp, B_in_h, C_in)

        return out



class ResidualMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        checkpoint: bool = False,
        return_hidden_attn: bool = False,
        # Hyperparams Mamba-2 "plats"
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 4, # combien de fois le hidden size
        headdim: int = 128,
        ngroups: int = 2,
        # compat: on ignore dt_rank s’il est passé par erreur
        dt_rank: int | None = None,
    ):
        # d_inner = expand*d_model
        # self.nheads = self.d_inner // self.headdim
        # d_model = hidden_size


        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.mamba = Mamba2MESIdentical(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            record_default=return_hidden_attn,
        )
        self.drop = nn.Dropout(dropout)
        self.checkpoint = checkpoint

    def _forward_inner(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mamba(self.norm(x))
        return x + self.drop(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_inner, x, use_reentrant=False)
        return self._forward_inner(x)


class StackedMamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        depth: int = 4,
        dropout: float = 0.1,
        checkpoint: bool = False,
        return_hidden_attn: bool = False,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 4,
        headdim: int = 128,
        ngroups: int = 1,
        dt_rank: int | None = None,  # ignoré
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualMambaBlock(
                d_model=d_model,
                dropout=dropout,
                checkpoint=checkpoint,
                return_hidden_attn=return_hidden_attn,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
                dt_rank=dt_rank,
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class DummyMambaAttn(nn.Module):
    """Remplace la MultiHeadAttention TFT par un petit mixeur Mamba sur q seulement.
    Renvoie des poids d'attention nuls pour préserver l'API & l'interprétabilité.
    """

    def __init__(
        self, d_model: int, n_head: int = 2, depth: int = 1, dropout: float = 0.0, return_hidden_attn = False
    ):
        super().__init__()
        self.n_head = n_head
        self.mixer = StackedMamba(d_model=d_model, depth=depth, dropout=dropout, return_hidden_attn=return_hidden_attn)

    def forward(self, q, k=None, v=None, mask=None):
        out = self.mixer(q)  # (B, L_q, D)
        B, L_q, _ = q.shape
        L_k = k.size(1) if k is not None else L_q
        attn = q.new_zeros(B, self.n_head, L_q, L_k)  # dummy weights
        # (Optionnel) appliquer mask => attn.masked_fill_(mask, 0.) si tu veux quelque chose de cohérent
        return out, attn


# ---------------------------------------------------------------------
# MambaTFT
# ---------------------------------------------------------------------
class MambaTFT(TemporalFusionTransformer):
    def __init__(
        self,
        *args,
        mamba_depth: int = 4,
        mamba_dropout: float = 0.1,
        mamba_checkpoint: bool = False,
        mamba_block_config: dict | None = None,   # 👈 ajouté
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        del self.lstm_encoder  
        del self.lstm_decoder
        del self.multihead_attn
        del self.static_context_initial_cell_lstm
        del self.static_context_initial_hidden_lstm
        # ... (suppression LSTM/MHA inchangée)

        d_model = self.hparams.hidden_size
        mb = mamba_block_config or {}

        self.lstm_encoder = StackedMamba(
            d_model=d_model,
            depth=mamba_depth,
            dropout=mamba_dropout,
            checkpoint=mamba_checkpoint,
            # 👇 tire les HP bloc depuis la config
            d_state=mb.get("d_state", 64),
            d_conv=mb.get("d_conv", 4),
            expand=mb.get("expand", 4),
            headdim=mb.get("headdim", 128),
            ngroups=mb.get("ngroups", 1),
            return_hidden_attn=mb.get("return_hidden_attn", False),
        )
        self.lstm_decoder = nn.Identity()

        self.multihead_attn = DummyMambaAttn(
            d_model=d_model,
            n_head=self.hparams.attention_head_size,
            dropout=mamba_dropout,
        )

    # ------------------------------------------------------------------
    # On ne touche qu'à la partie LSTM/attention, le reste reste identique.
    # ------------------------------------------------------------------
    def forward(self, x: Dict[str, torch.Tensor]):
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        total_lengths = encoder_lengths + decoder_lengths

        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        timesteps = x_cont.size(1)  # L_enc + L_dec
        max_encoder_length = int(encoder_lengths.max())

        # ==== 1) Embedding & variable selection (unchanged) =============
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )
        if len(self.static_variables) > 0:
            static_embedding_src = {
                name: input_vectors[name][:, 0] for name in self.static_variables
            }
            static_embedding, static_variable_selection = (
                self.static_variable_selection(static_embedding_src)
            )
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            static_variable_selection = torch.zeros(
                (x_cont.size(0), 0), dtype=self.dtype, device=self.device
            )
        static_context_varsel = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )
        embeddings_varying_encoder = {
            n: input_vectors[n][:, :max_encoder_length] for n in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = (
            self.encoder_variable_selection(
                embeddings_varying_encoder,
                static_context_varsel[:, :max_encoder_length],
            )
        )
        embeddings_varying_decoder = {
            n: input_vectors[n][:, max_encoder_length:] for n in self.decoder_variables
        }
        embeddings_varying_decoder, decoder_sparse_weights = (
            self.decoder_variable_selection(
                embeddings_varying_decoder,
                static_context_varsel[:, max_encoder_length:],
            )
        )

        # ==== 2) Mamba sur toute la fenêtre =============================
        embeddings_full = torch.cat(
            [embeddings_varying_encoder, embeddings_varying_decoder], dim=1
        )  # (B, L, D)
        full_seq = self.lstm_encoder(embeddings_full)  # (B, L, D)

        encoder_output = full_seq[:, :max_encoder_length]
        decoder_output = full_seq[:, max_encoder_length:]
        #h, c = _last_step_summary(full_seq, total_lengths)

        # ==== 3) Skip connection & gating (inchangé) ====================
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(
            lstm_output_encoder, embeddings_varying_encoder
        )
        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(
            lstm_output_decoder, embeddings_varying_decoder
        )
        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # ==== 4) Static enrichment & pseudo-attention ===================
        static_context_enrich = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrich, timesteps)
        )

        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input,
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths
            ),
        )

        attn_output = attn_output[:, max_encoder_length:]

        
        attn_output = self.post_attn_gate_norm(
            attn_output, attn_input[:, max_encoder_length:]
        )

        # ==== 5) Position‑wise FF & output head ========================
        output = self.pos_wise_ff(attn_output)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        if self.n_targets > 1:
            output = [layer(output) for layer in self.output_layer]
        else:
            output = self.output_layer(output)

        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )

def create_tft_dataloaders(train_df, horizon=12, context_length=72, batchsize=32):
    log_memory("🚀 Start of Dataloader Creation")

    # --- Extra Variables Outside Main Timeseries ---
    static_categoricals = [
        "participant_id",
        "clinical_site",
        "study_group",
    ]  # Have info on participant
    static_reals = ["age"]

    time_varying_known_categoricals = ["sleep_stage"]
    time_varying_known_reals = [
        "ds",
        "minute_of_day",
        "tod_sin",
        "tod_cos",
        "activity_steps",
        "calories_value",
        "heartrate",
        "oxygen_saturation",
        "respiration_rate",
        "stress_level",
        "predmeal_flag",
    ]  # To ensure time index (ordering is met)
    time_varying_unknown_reals = [
        "cgm_glucose",
        "cgm_lag_1",
        "cgm_lag_3",
        "cgm_lag_6",
        "cgm_diff_lag_1",
        "cgm_diff_lag_3",
        "cgm_diff_lag_6",
        "cgm_lagdiff_1_3",
        "cgm_lagdiff_3_6",
        "cgm_rolling_mean",
        "cgm_rolling_std",
    ]

    # --- Découpe temporelle ---
    # max_date = train_df["ds"].max()
    # cut_off_date = max_date - pd.Timedelta(days=context_length)

    # horizon = 12  # Predict 12 steps (1 hour)
    cut_off_date = train_df["ds"].max() - horizon  # Use latest horizon

    # --- Création du TimeSeriesDataSet ---
    training = TimeSeriesDataSet(
        train_df[train_df["ds"] < cut_off_date],
        time_idx="ds",
        target="cgm_glucose",
        group_ids=["participant_id"],
        max_encoder_length=context_length,
        max_prediction_length=horizon,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["participant_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = training.from_dataset(
        training, train_df, predict=True, stop_randomization=True
    )

    # --- Dataloaders ---
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batchsize,
        persistent_workers=False,
        num_workers=1,
        pin_memory=False, # CHANGE TO REDUCE MOMORY USAGE
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batchsize,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )


    return training, val_dataloader, train_dataloader, validation

# =============================
# Model Training
# =============================
class FirstEpochTimer(pl.Callback):
    def __init__(self, local_path, gcs_bucket, gcs_prefix):
        super().__init__()
        self.local_path = local_path
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.start_time = None
        self.logged = False

    def on_train_epoch_start(self, trainer, pl_module):
        # always record, not just epoch 0
        self.start_time = datetime.datetime.now()

    def on_train_epoch_end(self, trainer, pl_module):
        # only log+upload once, still guarding with `self.logged`
        if not self.logged:
            end_time = datetime.datetime.now()
            elapsed = (end_time - self.start_time).total_seconds()
            with open(self.local_path, 'w') as f:
                f.write(f"first_epoch_runtime_seconds: {elapsed}\n")
            # Upload to GCS
            bucket = _gcs_client.bucket(self.gcs_bucket)
            blob = bucket.blob(f"{self.gcs_prefix}/first_epoch_runtime.txt")
            blob.upload_from_filename(self.local_path)
            print(f"Uploaded first epoch runtime to gs://{self.gcs_bucket}/{self.gcs_prefix}/first_epoch_runtime.txt")
            self.logged = True

class GCSCheckpointUploader(pl.Callback):
    def __init__(self, local_dir, bucket_name, gcs_prefix):
        super().__init__()
        self.local_dir = local_dir
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # This hook is called after a checkpoint is saved
        upload_latest_checkpoint_to_gcs(self.local_dir, self.bucket_name, self.gcs_prefix)

class GCSLossLogger(pl.Callback):
    def __init__(self, log_path, bucket_name, gcs_prefix, log_every_n_batches=5000):
        super().__init__()
        self.log_path = log_path
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
        self.log_every_n_batches = log_every_n_batches
        self.batch_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_count += 1
        # Try to extract loss from outputs
        loss = None
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss'].item()
        elif hasattr(outputs, 'item'):
            loss = outputs.item()
        if self.batch_count % self.log_every_n_batches == 0 and loss is not None:
            with open(self.log_path, 'a') as f:
                f.write(f"Batch {self.batch_count}: loss={loss}\n")
            # Upload to GCS
            bucket = _gcs_client.bucket(self.bucket_name)
            blob = bucket.blob(f"{self.gcs_prefix}/loss_log.txt")
            blob.upload_from_filename(self.log_path)
            print(f"Uploaded loss log to gs://{self.bucket_name}/{self.gcs_prefix}/loss_log.txt")

def download_last_ckpt(local_dir: str, bucket_name: str, gcs_prefix: str):
    os.makedirs(local_dir, exist_ok=True)
    bucket = _gcs_client.bucket(bucket_name)
    blob = bucket.blob(f"{gcs_prefix}/last.ckpt")
    if blob.exists():
        local_path = os.path.join(local_dir, "last.ckpt")
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")
        return local_path
    return None

def TFT_train(train_df, param):
    # === tire les configs du preset ===
    ds_cfg = param["dataset"]
    mb_cfg = param["mamba_block"]
    mt_cfg = param["mamba_tft_init"]

    # heads = (expand * d_model) // headdim
    attention_heads = (mb_cfg["expand"] * mb_cfg["d_model"]) // mb_cfg["headdim"]

    horizon = ds_cfg["horizon"]
    context_length = ds_cfg["context_length"]
    batchsize = ds_cfg["batch_size"]

    log_memory("🚀 Début du chargement des dataloaders")

    training, val_dataloader, train_dataloader, validation = create_tft_dataloaders(
        train_df, horizon=horizon, context_length=context_length, batchsize=batchsize
    )

    del train_df
    gc.collect()
    log_memory("First creation of dataloaders")

    #loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    loss = QuantileLossWithMSE(quantiles=[0.1, 0.5, 0.9], mse_weight=0.3, huber_delta=None)

    # === MambaTFT.from_dataset avec HP issus de param ===
    tft = MambaTFT.from_dataset(
        training,
        learning_rate=0.001,                    # garde ta LR (ou mets-la aussi dans param si tu veux)
        hidden_size=mb_cfg["d_model"],          # 👈 d_model du bloc
        attention_head_size=attention_heads,    # 👈 cohérent avec expand/headdim
        dropout=mt_cfg["mamba_dropout"],        # 👈 dropout global TFT
        loss=loss,
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=4,
        # 👇 hyperparams spécifiques Mamba passés au __init__
        mamba_depth=mt_cfg["mamba_depth"],
        mamba_dropout=mt_cfg["mamba_dropout"],
        mamba_checkpoint=False,
        mamba_block_config=mb_cfg,
    )

    ###############
    # Checkpointing and Callbacks
    ###############
    log_memory("Model creation before training")
    # --- ModelCheckpoint callback ---
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="mamba288_MESqm-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_on_train_epoch_end=True
    )
    # --- First epoch timer callback ---
    first_epoch_timer = FirstEpochTimer(
        local_path="first_epoch_runtime.txt",
        gcs_bucket=_BUCKET_NAME,
        gcs_prefix="checkpoints_mamba288_MESqmv2"
    )
    # --- GCS checkpoint uploader callback ---
    gcs_ckpt_prefix = "checkpoints_mamba288_MESqmv2"
    gcs_checkpoint_uploader = GCSCheckpointUploader(
        local_dir="checkpoints",
        bucket_name=_BUCKET_NAME,
        gcs_prefix=gcs_ckpt_prefix
    )
    # --- GCS loss logger callback ---
    gcs_loss_logger = GCSLossLogger(
        log_path="loss_log.txt",
        bucket_name=_BUCKET_NAME,
        gcs_prefix=gcs_ckpt_prefix,
        log_every_n_batches=5000
    )
    # --- Download latest checkpoint from GCS if exists ---
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = download_last_ckpt("checkpoints", _BUCKET_NAME, gcs_ckpt_prefix)
    # log what happened
    if ckpt_path:
        print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
    else:
        print("[INFO] No checkpoint found: starting fresh")


    # === régler le runtime MES à partir du preset ===
    rt = param["mamba2_mes_runtime"]
    for blk in tft.lstm_encoder.blocks:
        blk.mamba.chunk_size = rt["chunk_size"]
        blk.mamba.dt_limit = rt["dt_limit"]
        blk.mamba.learnable_init_states = rt["learnable_init_states"]
        blk.mamba.set_debug(False)
        if rt.get("D", None) is not None:
            blk.mamba.D = rt["D"]

    # désactive la collecte pendant l'entraînement
    for blk in tft.lstm_encoder.blocks:
        blk.mamba.return_hidden_attn = False
    tft.lstm_encoder.return_hidden_attn = False

    log_memory("Model Creations before training")

    trainer = Trainer(
        max_epochs=30,
        gradient_clip_val=1.0,
        val_check_interval=0.2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15, mode="min"),
            #LearningRateMonitor(logging_interval="epoch"),
            checkpoint_callback,
            first_epoch_timer,
            gcs_checkpoint_uploader,
            gcs_loss_logger,
        ],
        enable_progress_bar=False,            # was True; disable to prevent large log entries
        enable_model_summary=False,           # suppress big model summary block
        logger=False,                         # rely on custom GCS logging
        #enable_progress_bar=True,
        accelerator="gpu",
        devices=4,
        strategy="ddp", #"ddp",
    )
    log_memory("Before training")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)
    log_memory("End training")

    return tft, trainer, val_dataloader, train_dataloader, validation, training

# =============================
# Model Save/Load Utilities
# =============================
def save_tft_to_gcs(
    tft: TemporalFusionTransformer,
    model_name: str = "default_model",
    bucket_name: str = _BUCKET_NAME,
    prefix: str = _BASE_PREFIX,
):
    """Save TFT state_dict and hyperparams to GCS."""
    bucket = _gcs_client.bucket(bucket_name)
    model_prefix = f"{prefix}/{model_name}"
    buf = io.BytesIO()
    torch.save(tft.state_dict(), buf)
    buf.seek(0)
    blob = bucket.blob(f"{model_prefix}/model.pth")
    blob.upload_from_file(buf)
    buf.close()
    buf = io.BytesIO()
    pickle.dump(tft.hparams, buf)
    buf.seek(0)
    blob = bucket.blob(f"{model_prefix}/model_kwargs.pkl")
    blob.upload_from_file(buf)
    buf.close()
    print(f"Model artifacts uploaded to gs://{bucket_name}/{model_prefix}/")

def load_tft_from_gcs(
    training_dataset,
    model_name: str = "default_model",
    map_location=None,
    bucket_name: str = _BUCKET_NAME,
    prefix: str = _BASE_PREFIX,
) -> MambaTFT:
    """Fetch model_kwargs.pkl and model.pth from GCS, rebuild the TFT, load weights, and return it."""
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = _gcs_client.bucket(bucket_name)
    model_prefix = f"{prefix}/{model_name}"
    blob = bucket.blob(f"{model_prefix}/model_kwargs.pkl")
    hparams_bytes = blob.download_as_bytes()
    model_kwargs = pickle.loads(hparams_bytes)
    tft = MambaTFT.from_dataset(training_dataset, **model_kwargs)
    blob = bucket.blob(f"{model_prefix}/model.pth")
    state_bytes = blob.download_as_bytes()
    buf = io.BytesIO(state_bytes)
    state_dict = torch.load(buf, map_location=map_location)
    tft.load_state_dict(state_dict)
    buf.close()
    print(f"Loaded TFT from gs://{bucket_name}/{model_prefix}/")
    return tft


# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    # Print rank and GPU info
    rank = os.environ.get("LOCAL_RANK", "NotSet")
    gpu_count = torch.cuda.device_count()
    print(f"[INFO] RANK={rank} | CUDA devices={gpu_count}")
    # Download dataset from GCS
    client = storage.Client()
    bucket = client.bucket(_BUCKET_NAME)
    blob = bucket.blob('ai-ready/data/train_finaltimeseries_meal.feather')
    data_bytes = blob.download_as_bytes()
    train = pd.read_feather(io.BytesIO(data_bytes))
    # Train model
    tft, trainer, val_dataloader, train_dataloader, validation, training = TFT_train(train, param_24) #Pick parameters HERE!!
    # Save model to GCS
    save_tft_to_gcs(tft, model_name="Mamba2_12h_288c_MESqm")