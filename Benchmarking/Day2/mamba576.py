# =============================
# Imports and Global Constants
# =============================
import os
import gc
import io
import pickle
import datetime
#import psutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Tuple, Dict
from google.cloud import storage
from mamba_ssm import Mamba  # your Mamba
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader
from pytorch_forecasting.utils import move_to_device
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
import glob
from einops import rearrange
from torch import nn
import torch, torch.nn.functional as F
import torch, torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
from typing import Tuple, Dict
from typing import Optional
#import psutil
import datetime



# GCS constants
_BUCKET_NAME = "cgmproject2025"
_BASE_PREFIX = "models/predictions"
_gcs_client = storage.Client()


# Configurations
# param_48h = {
#     # Fenêtres
#     "windows": {
#         "context_length": 576,   # 48h @ 5 min
#         "horizon": 12,           # 1h
#     },

#     # Modèle (Mamba avant static enrichment)
#     "model": {
#         "hidden_size": 256,
#         "dropout": 0.15,
#         "encoder": {
#             "mamba_depth": 6, #8,
#             "mamba_kwargs": {
#                 "d_state": 32, #64,
#                 "d_conv": 4,
#                 "expand": 2,
#                 "dt_rank": 32,    # <- si NON supporté par ta lib, supprime cette ligne
#             },
#         },

#         "post_static": {
#             "mamba_depth": 1,
#             "dropout": 0.15,
#             "mamba_kwargs": {
#                 "d_state": 32, #64,
#                 "d_conv": 4,     # mets 4 si tu veux préserver des pics très abrupts. 4 is the max!!
#                 "expand": 2,
#                 "dt_rank": 32,   # si NON supporté par ta lib, supprime cette ligne
#             },
#         },
#     },

#     # Perte
#     "loss": {
#         "type": "QuantileLoss",
#         "quantiles": [0.1, 0.5, 0.9],
#     },

#     # "loss": {
#     # "type": "QuantileMSELoss",
#     # "quantiles": [0.1, 0.5, 0.9],
#     # "mse_weight" : 0.2,
#     # },


#     # Optim & scheduler
#     "optim": {
#         "type": "AdamW",
#         "lr": 1e-3,
#         "betas": (0.9, 0.95),
#         "weight_decay": 5e-4,
#     },
#     "scheduler": {
#         "type": "ReduceLROnPlateau",
#         "patience": 15,
#         "factor": 0.2,
#         "min_lr": 1e-5,
#         "monitor": "val_loss",
#         "mode": "min",
#     },

#    # Entraînement
#     "training": {
#         "epochs": 30,
#         "devices": 4, # 4 GPUs
#         "gradient_clip_val": 1.0,
#         #"precision": "bf16-mixed",  # que sur A100 (DIDNT WORK!!!)
#         "strategy": "ddp", # DDP i don't know
#     },

#     # Dataloaders
#     "dataloader": {
#         "batch_size": 32, #128,        # ajuste selon ta VRAM
#         "num_workers": 1,        # 8–16 OK avec 48 vCPU check what u can 
#         "persistent_workers": False,
#         "pin_memory": False,
#         #"prefetch_factor": 4,
#     },
# }

param_48h = {
    "windows": {"context_length": 576, "horizon": 12},
    "model": {
        "hidden_size": 128,          # ↓ pour stabilité/Vram
        "dropout": 0.20,
        "encoder": {
            "mamba_depth": 4,         # 4 suffit avec expand=4, d_state=128
            "dropout": 0.20,
            "mamba_kwargs": {
                "d_state": 128,
                "d_conv": 4,
                "expand": 4,
                "dt_rank": None
            },
        },
        "post_static": {
            "mamba_depth": 1,
            "dropout": 0.20,
            "mamba_kwargs": {
                "d_state": 128,
                "d_conv": 4,
                "expand": 4,
                "dt_rank": None
            },
        },
    },
    "loss": {"type": "QuantileLoss", "quantiles": [0.1, 0.5, 0.9]},
    "optim": {
        "type": "AdamW",
        "lr": 1e-3, #3e-4,                  # ↓
        "betas": (0.9, 0.95),
        "weight_decay": 5e-4,
    },
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "patience": 15,
        "factor": 0.2,
        "min_lr": 1e-5,
        "monitor": "val_loss",
        "mode": "min",
    },
    "training": {
        "epochs": 30,
        "devices": 4,
        "gradient_clip_val": 1.0,
        "strategy": "ddp",
        "val_check_interval": 0.2
    },
    "dataloader": {
        "batch_size": 32,            # si OOM avec 576, passe à 24/16
        "num_workers": 1,
        "persistent_workers": False,
        "pin_memory": False,
    },
}


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

# =============================
# Mamba Blocks and TFT Variant
# =============================

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


#####################################################


class QuantileLossWithMSE(QuantileLoss):
    def __init__(self, quantiles=(0.1, 0.5, 0.9), mse_weight=0.2, huber_delta=None, **kwargs):
        super().__init__(quantiles=quantiles, **kwargs)  # <- LightningMetric OK
        assert 0.5 in self.quantiles, "Le quantile 0.5 (médiane) doit être présent."
        self.mse_weight = float(mse_weight)
        self.huber_delta = huber_delta

    def loss(self, y_pred, target):
        # pinball officielle PF
        qloss = super().loss(y_pred, target)

        # MSE/Huber sur la médiane
        midx = self.quantiles.index(0.5)
        median = y_pred[..., midx]
        if self.huber_delta is None:
            mse = F.mse_loss(median, target)
        else:
            mse = F.smooth_l1_loss(median, target, beta=self.huber_delta)

        return qloss + self.mse_weight * mse


################Overclassing_Mamba##################################



class MambaWithHiddenAttn(Mamba):
    """
    La même couche Mamba, mais expose self.last_hidden_attn   (B,L,L).
    Fonctionne même si selective_scan_fn est absent.
    """

    def __init__(self, *args, return_hidden_attn=False, **kw):
        super().__init__(*args, **kw)
        self._saved_raw : Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self.return_hidden_attn = return_hidden_attn

    # -----------------------------------------------------------------
    def _extract_dt_B_C(self, hidden_states):
        """
        Copie des 30 premières lignes du forward original : on s’arrête
        juste AVANT selective_scan_fn → on a dt_raw, B, C.
        """
        Bsz, L, _ = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=L,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        x, _ = xz.chunk(2, dim=1)                           # (B, D, L)

        # profondeur × conv 1-D + SiLU
        x = self.act(self.conv1d(x)[..., :L])

        # projette → dt_raw, B, C
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt_raw, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt_raw = self.dt_proj.weight @ dt_raw.T                  # (D, B·L)
        dt_raw = rearrange(dt_raw, "d (b l) -> b d l", l=L)

        B = rearrange(B, "(b l) n -> b n l", l=L)
        C = rearrange(C, "(b l) n -> b n l", l=L)
        return dt_raw, B, C

    # -----------------------------------------------------------------
    def forward(self, hidden_states, *fargs, **fkw):
        if self.return_hidden_attn:
            dt_raw, B, C = self._extract_dt_B_C(hidden_states)
            self._saved_raw = (dt_raw.detach(), B.detach(), C.detach())
        else:
            self._saved_raw = None

        # on appelle la vraie implémentation (fast-path ou slow-path)
        return super().forward(hidden_states, *fargs, **fkw)

    def extract_raw(self, hidden_states):
        """Retourne dt_raw, B, C sans rien modifier."""
        return self._extract_dt_B_C(hidden_states)     # ré-utilise ton code



class ResidualMambaBlock(nn.Module):
    """
    Norm -> Mamba -> Dropout -> Residual
    Tous les hyperparams Mamba sont des arguments explicites.
    """
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        checkpoint: bool = False,
        return_hidden_attn: bool = False,
        # ↓↓↓ hyperparams Mamba passés "à plat"
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = None,  # certains forks n'ont pas dt_rank
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # on passe directement les args à Mamba
        mamba_args = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        if dt_rank is not None:
            mamba_args["dt_rank"] = dt_rank

        self.mamba = MambaWithHiddenAttn(
            return_hidden_attn=return_hidden_attn,
            **mamba_args
        )
        self.drop = nn.Dropout(dropout)
        self.checkpoint = checkpoint


    def _forward_inner(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mamba(self.norm(x))  # (B, L, D)
        return x + self.drop(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_inner, x, use_reentrant=False
            )
        return self._forward_inner(x)


class StackedMamba(nn.Module):
    """
    Empile N blocs identiques ; on leur passe les mêmes hyperparams Mamba.
    """
    def __init__(
        self,
        d_model: int,
        depth: int = 4,
        dropout: float = 0.1,
        checkpoint: bool = False,
        return_hidden_attn: bool = False,
        # hyperparams Mamba "à plat"
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = None,
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
                dt_rank=dt_rank,
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x



class DummyMambaAttn(nn.Module):
    """
    Remplace la MHA par un petit mixeur Mamba sur q.
    On réutilise exactement les mêmes hyperparams Mamba, "à plat".
    """
    def __init__(
        self,
        d_model: int,
        n_head: int = 1,
        depth: int = 1,
        dropout: float = 0.0,
        return_hidden_attn: bool = False,
        # hyperparams Mamba "à plat"
        d_state: int = 32,
        d_conv: int = 8,
        expand: int = 2,
        dt_rank: int = None,
    ):
        super().__init__()
        self.n_head = n_head
        self.mixer = StackedMamba(
            d_model=d_model,
            depth=depth,
            dropout=dropout,
            return_hidden_attn=return_hidden_attn,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
        )
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
    """
    TFT où LSTM/MHA sont remplacés par Mamba.
    On passe des *arguments explicites* pour:
      - l'encoder (depth, d_state, d_conv, expand, dt_rank, dropout, checkpoint)
      - le bloc "post_static" (depth + mêmes hyperparams mamba)
    L’optim/scheduler restent gérés par Lightning si tu ne veux pas de custom.
    """
    def __init__(
        self,
        *args,
        # encoder (mamba sur toute la fenêtre)
        enc_depth: int = 4,
        enc_dropout: float = 0.1,
        enc_checkpoint: bool = False,
        enc_d_state: int = 32,
        enc_d_conv: int = 8,
        enc_expand: int = 2,
        enc_dt_rank: int = None,
        # post_static (pseudo-attention mamba)
        post_depth: int = 1,
        post_dropout: float = 0.1,
        post_d_state: int = 32,
        post_d_conv: int = 8,
        post_expand: int = 2,
        post_dt_rank: int = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # On enlève les modules TFT remplacés
        del self.lstm_encoder
        del self.lstm_decoder
        del self.multihead_attn
        del self.static_context_initial_cell_lstm
        del self.static_context_initial_hidden_lstm

        d_model = self.hparams.hidden_size

        # 1) Encoder Mamba
        self.lstm_encoder = StackedMamba(
            d_model=d_model,
            depth=enc_depth,
            dropout=enc_dropout,
            checkpoint=enc_checkpoint,
            d_state=enc_d_state,
            d_conv=enc_d_conv,
            expand=enc_expand,
            dt_rank=enc_dt_rank,
        )
        self.lstm_decoder = nn.Identity()

        # 2) "Attention" Mamba légère post-static
        self.multihead_attn = DummyMambaAttn(
            d_model=d_model,
            n_head=self.hparams.attention_head_size,
            depth=post_depth,
            dropout=post_dropout,
            d_state=post_d_state,
            d_conv=post_d_conv,
            expand=post_expand,
            dt_rank=post_dt_rank,
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
    
##############################################################################@

# =============================
# Data Preparation
# =============================

# ------- Dataloaders (inchangé sauf param lu simplement) -------
def create_tft_dataloaders(train_df, param: dict):
    log_memory("🚀 Start of Dataloader Creation (simple)")

    horizon = int(param["windows"]["horizon"])
    context_length = int(param["windows"]["context_length"])

    static_categoricals = ["participant_id", "clinical_site", "study_group"]
    static_reals = ["age"]
    time_varying_known_categoricals = ["sleep_stage"]
    time_varying_known_reals = [
        "ds","minute_of_day","tod_sin","tod_cos",
        "activity_steps","calories_value","heartrate",
        "oxygen_saturation","respiration_rate","stress_level","predmeal_flag",
    ]
    time_varying_unknown_reals = [
        "cgm_glucose","cgm_lag_1","cgm_lag_3","cgm_lag_6",
        "cgm_diff_lag_1","cgm_diff_lag_3","cgm_diff_lag_6",
        "cgm_lagdiff_1_3","cgm_lagdiff_3_6","cgm_rolling_mean","cgm_rolling_std",
    ]

    cut_off_date = train_df["ds"].max() - horizon

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

    validation = training.from_dataset(training, train_df, predict=True, stop_randomization=True)

    dl = param.get("dataloader", {})
    batch_size = int(dl.get("batch_size", 32))
    num_workers = int(dl.get("num_workers", 1))
    pin_memory = bool(dl.get("pin_memory", False))
    persistent_workers = bool(dl.get("persistent_workers", False)) if num_workers > 0 else False

    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0,
        persistent_workers=False,
        pin_memory=pin_memory,
    )
    return training, val_dataloader, train_dataloader, validation

# ------- LOSS simple (if/else) -------
def build_loss_from_param(param: dict):
    cfg = param.get("loss", {})
    typ = str(cfg.get("type", "QuantileLoss")).lower()
    quantiles = cfg.get("quantiles", [0.1, 0.5, 0.9])
    if typ in {"quantilemseloss", "quantilemse", "quantile_mse_loss"}:
        return QuantileLossWithMSE(
            quantiles=quantiles,
            mse_weight=float(cfg.get("mse_weight", 0.2)),
            huber_delta=cfg.get("huber_delta", None),
        )
    else:
        return QuantileLoss(quantiles=quantiles)

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

def TFT_train(train_df, param: dict):
    log_memory("Start loading dataloaders")

    training, val_dataloader, train_dataloader, validation = create_tft_dataloaders(train_df, param)
    del train_df
    gc.collect()
    log_memory("First creation of dataloaders")

    loss = build_loss_from_param(param)

    model_cfg = param.get("model", {})
    hidden_size = int(model_cfg.get("hidden_size", 64))
    dropout = float(model_cfg.get("dropout", 0.1))
    attn_heads = int(model_cfg.get("attention_head_size", 2))

    enc = model_cfg.get("encoder", {})
    enc_kwargs = enc.get("mamba_kwargs", {})
    post = model_cfg.get("post_static", {})
    post_kwargs = post.get("mamba_kwargs", {})

    tft = MambaTFT.from_dataset(
        training,
        learning_rate=float(param.get("optim", {}).get("lr", 3e-3)),
        hidden_size=hidden_size,
        attention_head_size=attn_heads,
        dropout=dropout,
        loss=loss,
        log_interval=10,
        log_val_interval=1,
        # Encoder Mamba
        enc_depth=int(enc.get("mamba_depth", enc.get("depth", 4))),
        enc_dropout=float(enc.get("dropout", dropout)),
        enc_checkpoint=bool(enc.get("checkpoint", False)),
        enc_d_state=int(enc_kwargs.get("d_state", 32)),
        enc_d_conv=int(enc_kwargs.get("d_conv", 8)),
        enc_expand=int(enc_kwargs.get("expand", 2)),
        enc_dt_rank=enc_kwargs.get("dt_rank", None),
        # Post-static Mamba
        post_depth=int(post.get("mamba_depth", post.get("depth", 1))),
        post_dropout=float(post.get("dropout", dropout)),
        post_d_state=int(post_kwargs.get("d_state", enc_kwargs.get("d_state", 32))),
        post_d_conv=int(post_kwargs.get("d_conv", enc_kwargs.get("d_conv", 8))),
        post_expand=int(post_kwargs.get("expand", enc_kwargs.get("expand", 2))),
        post_dt_rank=post_kwargs.get("dt_rank", enc_kwargs.get("dt_rank", None)),
    )

    ###############
    # Checkpointing and Callbacks
    ###############
    log_memory("Model creation before training")
    # --- ModelCheckpoint callback ---
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="mamba576-{epoch:02d}-{val_loss:.2f}",
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
        gcs_prefix="checkpoints_mamba576v2"
    )
    # --- GCS checkpoint uploader callback ---
    gcs_ckpt_prefix = "checkpoints_mamba576v2"
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



    # désactive la collecte d'attn pendant l'entraînement
    if isinstance(tft.lstm_encoder, StackedMamba):
        for blk in tft.lstm_encoder.blocks:
            blk.mamba.return_hidden_attn = False

    train_cfg = param.get("training", {})
    scheduler_cfg = param.get("scheduler", {})
    callbacks = [
        EarlyStopping(
            monitor=scheduler_cfg.get("monitor", "val_loss"),
            patience=int(scheduler_cfg.get("patience", 6)),
            mode=scheduler_cfg.get("mode", "min"),
        ),
        LearningRateMonitor(logging_interval="epoch"),
        checkpoint_callback,
        first_epoch_timer,
        gcs_checkpoint_uploader,
        gcs_loss_logger,
    ]

    trainer = Trainer(
        max_epochs=int(train_cfg.get("epochs", 30)),
        gradient_clip_val=float(train_cfg.get("gradient_clip_val", 1.0)),
        accelerator="gpu",
        devices=train_cfg.get("devices", 1),
        strategy=train_cfg.get("strategy", "auto"),
        val_check_interval=train_cfg.get("val_check_interval", 0.2),
        #**({"precision": train_cfg["precision"]} if "precision" in train_cfg else {}),
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    log_memory("Training")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)
    log_memory("Finish")

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
    tft, trainer, val_dataloader, train_dataloader, validation, training = TFT_train(train, param_48h) #Pick parameters HERE!!
    # Save model to GCS
    save_tft_to_gcs(tft, model_name="Mamba_12h_576c")