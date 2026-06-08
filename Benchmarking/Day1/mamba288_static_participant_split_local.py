"""
mamba288_static_participant_split_local.py
==========================================
Experiment C — dynamic + static features, participant-disjoint split.

Key design choices (updated):
  - participant_id is kept as group_ids (required to define each time series)
    but is REMOVED from static_categoricals by default.
    Reason: all unseen val/test participants map to the same OOV embedding,
    collapsing to a single vector that carries no signal and may hurt performance.
    Use --include-participant-id-embedding to restore old behaviour for ablation.

  - Val and test evaluation uses predict=False (rolling windows) by default.
    Reason: predict=True returns only the LAST window per participant (one window
    per participant for OOV participants), giving too few windows for stable metrics.
    Use --predict-last-only to restore old behaviour for ablation.

  - NaNLabelEncoder(add_nan=True) is applied to participant_id (for group_ids
    compatibility), clinical_site, and study_group so unseen categories never crash.

This lets you isolate the cause of poor Exp C results:
  Default (no flags)           → rolling windows, no pid embedding
  --include-participant-id-embedding → adds OOV pid embedding back
  --predict-last-only          → back to one window per participant
  Both flags                   → original behaviour for direct comparison

What differs from Experiment B (mamba288_static_local.py):
  - Reads THREE feathers: train / val / test from Data/ssmcgm_ready_exp_C/
  - Val and test participants are DISJOINT from train (participant-level split).
  - Outputs go to Data/results/exp_C_smoke/ or Data/results/exp_C_full/
  - Results saved to results_dynamic_static_participant_split.csv
  - Run config saved to experiment_c_run_config.json

Run:
  # Default (rolling windows, no pid embedding)
  python mamba288_static_participant_split_local.py --smoke --epochs 3

  # Old-like behaviour (OOV pid embedding + last-window only):
  python mamba288_static_participant_split_local.py --smoke --epochs 3 \\
      --include-participant-id-embedding --predict-last-only

  # Test only pid embedding effect:
  python mamba288_static_participant_split_local.py --smoke --epochs 3 \\
      --include-participant-id-embedding

  # Test only last-window effect:
  python mamba288_static_participant_split_local.py --smoke --epochs 3 \\
      --predict-last-only

  # Full run
  python mamba288_static_participant_split_local.py

  # Eval only
  python mamba288_static_participant_split_local.py --eval-only
"""

# =============================
# Imports
# =============================
import os
import gc
import json
import math
import argparse
import datetime
import copy
import re
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from typing import Tuple, Dict, Optional
from pathlib import Path

from mamba_ssm import Mamba
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
try:
    from pytorch_forecasting.metrics import MAE, RMSE, MAPE, SMAPE
except ImportError:
    MAE = RMSE = MAPE = SMAPE = None
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
try:
    from lightning.pytorch.loggers import WandbLogger
except ImportError:  # W&B is optional unless --use-wandb is requested.
    WandbLogger = None

import glob
from einops import rearrange


# =============================
# Hyperparameters (same as Exp A/B)
# =============================
param_24h = {
    "windows": {"context_length": 288, "horizon": 12},
    "model": {
        "hidden_size": 128,
        "dropout": 0.20,
        "encoder": {
            "mamba_depth": 4,
            "dropout": 0.20,
            "mamba_kwargs": {
                "d_state": 128,
                "d_conv": 4,
                "expand": 4,
                "dt_rank": None,
            },
        },
        "post_static": {
            "mamba_depth": 1,
            "dropout": 0.20,
            "mamba_kwargs": {
                "d_state": 128,
                "d_conv": 4,
                "expand": 4,
                "dt_rank": None,
            },
        },
    },
    "loss": {"type": "QuantileLoss", "quantiles": [0.1, 0.5, 0.9]},
    "optim": {
        "type": "AdamW",
        "lr": 1e-3,
        "betas": (0.9, 0.95),
        "weight_decay": 5e-4,
    },
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "patience": 2,
        "factor": 0.5,
        "min_lr": 1e-5,
        "monitor": "val_loss",
        "mode": "min",
    },
    "training": {
        "epochs": 30,
        "min_epochs": 10,
        "early_stopping_patience": 5,
        "devices": 4,          # a2-highgpu-4g (4×A100, 48 vCPU); fits standard CPU quota
        "gradient_clip_val": 1.0,
        "strategy": "auto",
        "val_check_interval": 1.0,
        "limit_val_batches": 1.0,
    },
    "dataloader": {
        "batch_size": 8,
        "val_batch_size": 4,   # rolling-window eval generates many windows; keep small to avoid OOM
        "val_num_workers": 2,  # defaults to train workers unless overridden for safe validation
        "num_workers": 2,      # conservative — avoid NCCL instability with many workers
        "persistent_workers": False,
        "pin_memory": False,
    },
}


# =============================
# Utility
# =============================
def log_memory(message=""):
    if torch.cuda.is_available():
        alloc    = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
    else:
        alloc = reserved = 0.0
    print(f"[{datetime.datetime.now():%H:%M:%S}] {message} | GPU {alloc:.2f}GB alloc / {reserved:.2f}GB reserved")


def log_memory_state(tag: str):
    import os
    import gc
    import torch

    print(f"\n[MEMORY] {tag}")
    print(f"[MEMORY] WORLD_SIZE={os.environ.get('WORLD_SIZE', '')} RANK={os.environ.get('RANK', '')} LOCAL_RANK={os.environ.get('LOCAL_RANK', '')} GLOBAL_RANK={os.environ.get('GLOBAL_RANK', '')}")

    try:
        import psutil
        vm = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        print(f"[MEMORY] RAM used     : {vm.used / 1024**3:.2f} GB")
        print(f"[MEMORY] RAM available: {vm.available / 1024**3:.2f} GB")
        print(f"[MEMORY] RAM percent  : {vm.percent:.1f}%")
        process_rss = process.memory_info().rss
        child_rss = 0
        child_count = 0
        for child in process.children(recursive=True):
            try:
                child_rss += child.memory_info().rss
                child_count += 1
            except psutil.Error:
                continue
        print(f"[MEMORY] Process RSS  : {process_rss / 1024**3:.2f} GB")
        print(f"[MEMORY] Child processes: {child_count}")
        print(f"[MEMORY] Children RSS : {child_rss / 1024**3:.2f} GB")
        print(f"[MEMORY] Proc+child RSS: {(process_rss + child_rss) / 1024**3:.2f} GB")
    except Exception as e:
        print(f"[MEMORY] psutil unavailable: {e}")

    try:
        st = os.statvfs("/dev/shm")
        total = st.f_frsize * st.f_blocks / 1024**3
        free = st.f_frsize * st.f_bavail / 1024**3
        print(f"[MEMORY] /dev/shm total: {total:.2f} GB")
        print(f"[MEMORY] /dev/shm free : {free:.2f} GB")
    except Exception as e:
        print(f"[MEMORY] /dev/shm stat failed: {e}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"[MEMORY] GPU {i}: allocated={alloc:.2f} GB | reserved={reserved:.2f} GB")

    gc.collect()


def cleanup_memory(tag: str = ""):
    if tag:
        print(f"[CLEANUP] {tag}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception as exc:
            print(f"[MEMORY] torch.cuda.ipc_collect unavailable: {exc}")


def _dataloader_can_yield_one_batch(dl, label: str) -> bool:
    if dl is None:
        print(f"[DataFrame] {label} dataloader is not built; skipping batch check.")
        return True
    try:
        batch = next(iter(dl))
        print(f"[DataFrame] {label} dataloader yielded one batch after construction.")
        del batch
        cleanup_memory(f"after {label} dataloader batch check")
        return True
    except Exception as exc:
        print(f"[DataFrame][WARN] {label} dataloader could not yield one batch: {exc}")
        return False


def _dataset_directly_holds_dataframe(dataset) -> bool:
    if dataset is None:
        return False
    try:
        return any(isinstance(value, pd.DataFrame) for value in vars(dataset).values())
    except Exception:
        return False


def maybe_delete_raw_dataframes_after_dataloaders(
    train_df, val_df, test_df,
    training, val_dataset, test_dataset,
    train_dl, val_dl, test_dl,
):
    datasets = [("train", training), ("val", val_dataset), ("test", test_dataset)]
    dataframe_refs = [name for name, dataset in datasets if _dataset_directly_holds_dataframe(dataset)]
    if dataframe_refs:
        print(f"[DataFrame] Raw DataFrames retained; dataset(s) still expose DataFrame refs: {dataframe_refs}")
        return False

    ok = (
        _dataloader_can_yield_one_batch(train_dl, "train")
        and _dataloader_can_yield_one_batch(val_dl, "val")
        and _dataloader_can_yield_one_batch(test_dl, "test")
    )
    if not ok:
        print("[DataFrame] Raw DataFrames retained because a dataloader batch check failed.")
        return False

    del train_df, val_df, test_df
    cleanup_memory("after deleting raw train/val/test DataFrames")
    print("[DataFrame] Raw train/val/test DataFrame references deleted after dataloader batch checks.")
    return True


def process_rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3
    except Exception:
        return None


def system_ram_percent():
    try:
        import psutil
        return float(psutil.virtual_memory().percent)
    except Exception:
        return None


def write_high_ram_warning(out_dir: Path, split_name: str, ram_percent, stage: str):
    warning_path = out_dir / f"high_ram_warning_{split_name}.txt"
    warning_path.write_text(
        f"[WARNING] Skipped nonessential final evaluation artifacts during {stage} "
        f"because system RAM usage was {ram_percent:.1f}% (>90%).\n"
    )
    print(f"[WARNING] High RAM during {stage}; warning saved -> {warning_path}")


class MemoryMonitorCallback(Callback):
    def __init__(self):
        super().__init__()
        self.previous_validation_end_rss_gb = None

    @staticmethod
    def _tag(trainer, label: str):
        return f"{label} | epoch={trainer.current_epoch} | global_step={trainer.global_step}"

    def on_fit_start(self, trainer, pl_module):
        log_memory_state(self._tag(trainer, "fit start"))

    def on_train_epoch_start(self, trainer, pl_module):
        log_memory_state(self._tag(trainer, "train epoch start"))

    def on_train_epoch_end(self, trainer, pl_module):
        log_memory_state(self._tag(trainer, "train epoch end"))
        cleanup_memory("after train epoch end")
        log_memory_state(self._tag(trainer, "after train epoch cleanup"))

    def on_validation_start(self, trainer, pl_module):
        log_memory_state(self._tag(trainer, "validation start"))

    def on_validation_epoch_start(self, trainer, pl_module):
        log_memory_state(self._tag(trainer, "validation epoch start"))

    def on_validation_epoch_end(self, trainer, pl_module):
        log_memory_state(self._tag(trainer, "validation epoch end"))
        current_rss_gb = process_rss_gb()
        if current_rss_gb is not None:
            print(f"[MEMORY] Validation end process RSS: {current_rss_gb:.2f} GB")
            if (
                self.previous_validation_end_rss_gb is not None
                and current_rss_gb - self.previous_validation_end_rss_gb > 10.0
            ):
                print("[WARNING] Process RSS increased by more than 10 GB since previous validation. Possible memory leak.")
            self.previous_validation_end_rss_gb = current_rss_gb
        cleanup_memory("after validation epoch end")
        log_memory_state(self._tag(trainer, "after validation epoch cleanup"))

    def on_validation_end(self, trainer, pl_module):
        log_memory_state(self._tag(trainer, "validation end"))
        cleanup_memory("after validation end")
        log_memory_state(self._tag(trainer, "after validation cleanup"))

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        log_memory_state(self._tag(trainer, "before checkpoint save"))
        cleanup_memory("before/after checkpoint save hook")
        log_memory_state(self._tag(trainer, "after checkpoint save hook cleanup"))


# =============================
# Mamba blocks (identical to Exp A/B)
# =============================

class MambaWithHiddenAttn(Mamba):
    def __init__(self, *args, return_hidden_attn=False, **kw):
        super().__init__(*args, **kw)
        self._saved_raw: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self.return_hidden_attn = return_hidden_attn

    def _extract_dt_B_C(self, hidden_states):
        Bsz, L, _ = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=L,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        x, _ = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :L])
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt_raw, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_raw = self.dt_proj.weight @ dt_raw.T
        dt_raw = rearrange(dt_raw, "d (b l) -> b d l", l=L)
        B = rearrange(B, "(b l) n -> b n l", l=L)
        C = rearrange(C, "(b l) n -> b n l", l=L)
        return dt_raw, B, C

    def forward(self, hidden_states, *fargs, **fkw):
        if self.return_hidden_attn:
            dt_raw, B, C = self._extract_dt_B_C(hidden_states)
            self._saved_raw = (dt_raw.detach(), B.detach(), C.detach())
        else:
            self._saved_raw = None
        return super().forward(hidden_states, *fargs, **fkw)

    def extract_raw(self, hidden_states):
        return self._extract_dt_B_C(hidden_states)


class ResidualMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1, checkpoint=False,
                 return_hidden_attn=False,
                 d_state=32, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        mamba_args = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        if dt_rank is not None:
            mamba_args["dt_rank"] = dt_rank
        self.mamba = MambaWithHiddenAttn(return_hidden_attn=return_hidden_attn, **mamba_args)
        self.drop = nn.Dropout(dropout)
        self.checkpoint = checkpoint

    def _forward_inner(self, x):
        return x + self.drop(self.mamba(self.norm(x)))

    def forward(self, x):
        if self.checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_inner, x, use_reentrant=False)
        return self._forward_inner(x)


class StackedMamba(nn.Module):
    def __init__(self, d_model, depth=4, dropout=0.1, checkpoint=False,
                 return_hidden_attn=False,
                 d_state=32, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualMambaBlock(d_model, dropout, checkpoint, return_hidden_attn,
                               d_state, d_conv, expand, dt_rank)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class DummyMambaAttn(nn.Module):
    def __init__(self, d_model, n_head=1, depth=1, dropout=0.0,
                 return_hidden_attn=False,
                 d_state=32, d_conv=8, expand=2, dt_rank=None):
        super().__init__()
        self.n_head = n_head
        self.mixer = StackedMamba(d_model, depth, dropout, False,
                                  return_hidden_attn, d_state, d_conv, expand, dt_rank)

    def forward(self, q, k=None, v=None, mask=None):
        out = self.mixer(q)
        B, L_q, _ = q.shape
        L_k = k.size(1) if k is not None else L_q
        attn = q.new_zeros(B, self.n_head, L_q, L_k)
        return out, attn


# =============================
# MambaTFT (identical to Exp A/B)
# =============================
class MambaTFT(TemporalFusionTransformer):
    def __init__(self, *args,
                 enc_depth=4, enc_dropout=0.1, enc_checkpoint=False,
                 enc_d_state=32, enc_d_conv=8, enc_expand=2, enc_dt_rank=None,
                 post_depth=1, post_dropout=0.1,
                 post_d_state=32, post_d_conv=8, post_expand=2, post_dt_rank=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        del self.lstm_encoder, self.lstm_decoder, self.multihead_attn
        del self.static_context_initial_cell_lstm, self.static_context_initial_hidden_lstm

        d_model = self.hparams.hidden_size
        self.lstm_encoder = StackedMamba(
            d_model, enc_depth, enc_dropout, enc_checkpoint,
            d_state=enc_d_state, d_conv=enc_d_conv, expand=enc_expand, dt_rank=enc_dt_rank,
        )
        self.lstm_decoder = nn.Identity()
        self.multihead_attn = DummyMambaAttn(
            d_model, n_head=self.hparams.attention_head_size,
            depth=post_depth, dropout=post_dropout,
            d_state=post_d_state, d_conv=post_d_conv, expand=post_expand, dt_rank=post_dt_rank,
        )

    def forward(self, x: Dict[str, torch.Tensor]):
        encoder_lengths  = x["encoder_lengths"]
        decoder_lengths  = x["decoder_lengths"]
        total_lengths    = encoder_lengths + decoder_lengths
        x_cat  = torch.cat([x["encoder_cat"],  x["decoder_cat"]],  dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        timesteps = x_cont.size(1)
        max_encoder_length = int(encoder_lengths.max())

        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update({
            name: x_cont[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.hparams.x_reals)
            if name in self.reals
        })

        if len(self.static_variables) > 0:
            static_embedding, static_variable_selection = self.static_variable_selection(
                {name: input_vectors[name][:, 0] for name in self.static_variables}
            )
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros(
                (x_cont.size(0), 0), dtype=self.dtype, device=self.device
            )

        static_context_varsel = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            {n: input_vectors[n][:, :max_encoder_length] for n in self.encoder_variables},
            static_context_varsel[:, :max_encoder_length],
        )
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            {n: input_vectors[n][:, max_encoder_length:] for n in self.decoder_variables},
            static_context_varsel[:, max_encoder_length:],
        )

        full_seq     = self.lstm_encoder(
            torch.cat([embeddings_varying_encoder, embeddings_varying_decoder], dim=1)
        )
        encoder_output = full_seq[:, :max_encoder_length]
        decoder_output = full_seq[:, max_encoder_length:]

        lstm_output_encoder = self.post_lstm_add_norm_encoder(
            self.post_lstm_gate_encoder(encoder_output), embeddings_varying_encoder
        )
        lstm_output_decoder = self.post_lstm_add_norm_decoder(
            self.post_lstm_gate_decoder(decoder_output), embeddings_varying_decoder
        )
        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        static_context_enrich = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrich, timesteps)
        )
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input, k=attn_input, v=attn_input,
            mask=self.get_attention_mask(encoder_lengths=encoder_lengths,
                                         decoder_lengths=decoder_lengths),
        )
        attn_output = self.post_attn_gate_norm(
            attn_output[:, max_encoder_length:], attn_input[:, max_encoder_length:]
        )
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


# =============================
# Data — participant-disjoint splits
# =============================
def _make_subsample_dl(
    dataset, max_windows, bs, pin_memory,
    num_workers=0, persistent_workers=False, seed=42,
):
    """
    Create a DataLoader from dataset, optionally subsampling to max_windows.
    This is used for capped rolling validation/test windows in large runs.
    """
    pw = bool(persistent_workers) if num_workers > 0 else False
    if max_windows is not None and len(dataset) > max_windows:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(dataset), size=max_windows, replace=False).tolist())
        # Borrow collate_fn from a minimal dataloader over the full dataset.
        tmp_dl = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        return DataLoader(
            Subset(dataset, idx),
            batch_size=bs,
            num_workers=num_workers,
            persistent_workers=pw,
            collate_fn=tmp_dl.collate_fn,
            pin_memory=pin_memory,
        )
    return dataset.to_dataloader(
        train=False,
        batch_size=bs,
        num_workers=num_workers,
        persistent_workers=pw,
        pin_memory=pin_memory,
    )


def create_tft_dataloaders_participant_split(
    train_df, val_df, test_df, param, extra_static_reals=None,
    include_participant_id_embedding=False,
    predict_last_only=False,
    max_val_windows=None,
    max_test_windows=None,
    enable_memory_monitor=False,
):
    """
    Build TSDS with participant-disjoint train / val / test.

    By default:
      - participant_id is used as group_ids only (not a predictive static categorical).
      - Val and test use predict=False (rolling windows) for stable metric estimation.

    Flags:
      include_participant_id_embedding=True → adds participant_id to static_categoricals.
        All unseen val/test participants collapse to the same OOV embedding.
      predict_last_only=True → uses predict=True (one window per participant).

    Returns (training, val_dl, test_dl, train_dl, val_dataset, test_dataset, static_categoricals).
    If test_df is None, test_dataset and test_dl are returned as None.
    """
    log_memory("Building TimeSeriesDataSets (participant-split)")

    horizon        = int(param["windows"]["horizon"])
    context_length = int(param["windows"]["context_length"])

    # participant_id excluded from predictive inputs by default to avoid OOV collapse
    static_categoricals = ["clinical_site", "study_group"]
    if include_participant_id_embedding:
        static_categoricals = ["participant_id", "clinical_site", "study_group"]

    static_reals = ["age"] + (extra_static_reals or [])

    time_varying_known_categoricals = ["sleep_stage"]
    time_varying_known_reals = [
        "ds", "minute_of_day", "tod_sin", "tod_cos",
        "activity_steps", "calories_value", "heartrate",
        "oxygen_saturation", "respiration_rate", "stress_level", "predmeal_flag",
    ]
    time_varying_unknown_reals = [
        "cgm_glucose",
        "cgm_lag_1", "cgm_lag_3", "cgm_lag_6",
        "cgm_diff_lag_1", "cgm_diff_lag_3", "cgm_diff_lag_6",
        "cgm_lagdiff_1_3", "cgm_lagdiff_3_6",
        "cgm_rolling_mean", "cgm_rolling_std",
    ]

    cut_off = train_df["ds"].max() - horizon

    # NaNLabelEncoder(add_nan=True) on participant_id is always needed so that
    # unseen val/test group_ids don't raise KeyError during from_dataset().
    # clinical_site and study_group get it too for robustness.
    categorical_encoders = {
        "participant_id": NaNLabelEncoder(add_nan=True),
        "clinical_site":  NaNLabelEncoder(add_nan=True),
        "study_group":    NaNLabelEncoder(add_nan=True),
    }

    training = TimeSeriesDataSet(
        train_df[train_df["ds"] < cut_off],
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
        categorical_encoders=categorical_encoders,
    )

    # Rolling windows (predict=False) give many windows per participant → stable metrics.
    # predict=True gives only the last window per participant → very few windows for OOV participants.
    val_dataset = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=predict_last_only, stop_randomization=True,
    )
    if test_df is None:
        print("[Data] Test DataFrame not loaded; skipping test TimeSeriesDataSet during training.")
        test_dataset = None
    else:
        test_dataset = TimeSeriesDataSet.from_dataset(
            training, test_df, predict=predict_last_only, stop_randomization=True,
        )
    if enable_memory_monitor:
        log_memory_state("after building TimeSeriesDataSets")

    dl  = param.get("dataloader", {})
    bs  = int(dl.get("batch_size", 32))
    val_bs = int(dl.get("val_batch_size", bs))
    nw  = int(dl.get("num_workers", 4))
    val_nw = int(dl.get("val_num_workers", nw))
    pin = bool(dl.get("pin_memory", True))
    pw  = bool(dl.get("persistent_workers", True)) if nw > 0 else False
    val_pw = bool(dl.get("val_persistent_workers", dl.get("persistent_workers", True))) if val_nw > 0 else False

    train_dl = training.to_dataloader(
        train=True, batch_size=bs, num_workers=nw, persistent_workers=pw, pin_memory=pin
    )
    val_dl  = _make_subsample_dl(
        val_dataset, max_val_windows, val_bs, pin,
        num_workers=val_nw, persistent_workers=val_pw, seed=42,
    )
    test_dl = None
    if test_dataset is not None:
        test_dl = _make_subsample_dl(
            test_dataset, max_test_windows, val_bs, pin,
            num_workers=val_nw, persistent_workers=val_pw, seed=43,
        )
    if enable_memory_monitor:
        log_memory_state("after building DataLoaders")

    return training, val_dl, test_dl, train_dl, val_dataset, test_dataset, static_categoricals


# =============================
# Pre-training validation
# =============================
def validate_before_training(train_df, val_df, test_df, extra_static_reals):
    print("\n[Pre-training Validation]")
    frames = [("train", train_df), ("val", val_df)]
    if test_df is not None:
        frames.append(("test", test_df))

    for name, sdf in frames:
        missing = [c for c in extra_static_reals if c not in sdf.columns]
        if missing:
            raise ValueError(f"Static columns missing from {name} DataFrame: {missing}")
        nan_counts = sdf[extra_static_reals].isna().sum()
        bad = nan_counts[nan_counts > 0]
        if len(bad) > 0:
            raise ValueError(f"NaN found in {name} static columns:\n{bad}")

    train_pids = set(train_df["participant_id"].unique())
    val_pids   = set(val_df["participant_id"].unique())
    overlap_tv = train_pids & val_pids
    if test_df is not None:
        test_pids  = set(test_df["participant_id"].unique())
        overlap_tt = train_pids & test_pids
        overlap_vt = val_pids   & test_pids
        if overlap_tv or overlap_tt or overlap_vt:
            raise ValueError(
                f"Participant overlap detected — splits are NOT disjoint!\n"
                f"  train∩val={len(overlap_tv)}, train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}"
            )
        print("  Participant disjointness : OK (train ∩ val ∩ test = ∅)")
    else:
        if overlap_tv:
            raise ValueError(f"Participant overlap detected — train∩val={len(overlap_tv)}")
        print("  Participant disjointness : OK (train ∩ val = ∅; test not loaded for training)")
    print("  Static cols OK           : no NaN")


# =============================
# Diagnostics
# =============================
def print_diagnostics(
    include_participant_id_embedding, predict_last_only,
    static_categoricals,
    train_df, val_df, test_df,
    train_dl, val_dl, test_dl,
    max_val_windows=None,
):
    def _dl_len(dl):
        try:
            return len(dl.dataset)
        except Exception:
            return "?"

    print("\n" + "=" * 62)
    print("  EXPERIMENT C — RUN CONFIGURATION")
    print("=" * 62)
    print(f"  include_participant_id_embedding : {include_participant_id_embedding}")
    print(f"  validation_regime                : {'last_window_only' if predict_last_only else 'rolling_windows'}")
    print(f"  predict_last_only (1 win/pid)    : {predict_last_only}")
    print(f"  rolling_windows (default)        : {not predict_last_only}")
    print(f"  max_val_windows                  : {max_val_windows}")
    print(f"  static_categoricals              : {static_categoricals}")
    print(f"  group_ids                        : ['participant_id']")
    print(f"  Train participants               : {train_df['participant_id'].nunique():,}")
    print(f"  Val   participants               : {val_df['participant_id'].nunique():,}")
    if test_df is not None:
        print(f"  Test  participants               : {test_df['participant_id'].nunique():,}")
    else:
        print("  Test  participants               : not loaded during training")
    print(f"  Train windows                   : {_dl_len(train_dl):,}")
    print(f"  Val   windows actually used     : {_dl_len(val_dl)}")
    if test_dl is not None:
        print(f"  Test  windows                   : {_dl_len(test_dl)}")
    else:
        print("  Test  windows                   : not built during training")
    print("=" * 62 + "\n")


# =============================
# Save run config
# =============================
def save_run_config(args, static_categoricals, param, out_dir, batch_diagnostics=None):
    context_hours = param["windows"]["context_length"] / 12
    horizon_hours = param["windows"]["horizon"] / 12
    global_batch_size = param["dataloader"]["batch_size"] * param["training"]["devices"]
    shm_size_mib = os.environ.get("SHM_SIZE_MIB")
    config = {
        "experiment_name":                 getattr(args, "experiment_name", "exp_C"),
        "run_name":                        getattr(args, "wandb_run_name", None),
        "gcs_output_path":                 getattr(args, "gcs_output_path", None),
        "output_root":                     os.environ.get("GCS_OUTPUT_ROOT") or getattr(args, "gcs_output_path", None),
        "machineType":                     os.environ.get("MACHINE_TYPE"),
        "machine_type":                    os.environ.get("MACHINE_TYPE"),
        "local_output_root":               os.environ.get("LOCAL_OUTPUT_ROOT") or str(out_dir),
        "include_participant_id_embedding": args.include_participant_id_embedding,
        "predict_last_only":               args.predict_last_only,
        "rolling_windows":                  not args.predict_last_only,
        "train_predict_last_only":         args.train_predict_last_only,
        "max_val_windows":                 args.max_val_windows,
        "max_test_windows":                args.max_test_windows,
        "skip_final_eval":                 args.skip_final_eval,
        "skip_test_eval":                  getattr(args, "skip_test_eval", False),
        "load_test_during_training":       bool(getattr(args, "load_test_during_training", True)),
        "save_all_epoch_checkpoints":       args.save_all_epoch_checkpoints,
        "limit_train_batches":             getattr(args, "limit_train_batches", None),
        "static_categoricals":             static_categoricals,
        "group_ids":                       ["participant_id"],
        "context_hours":                   context_hours,
        "horizon_hours":                   horizon_hours,
        "context_length":                  param["windows"]["context_length"],
        "horizon":                         param["windows"]["horizon"],
        "learning_rate":                   param["optim"]["lr"],
        "dropout":                         param["model"]["dropout"],
        "encoder_dropout":                 param["model"]["encoder"].get("dropout"),
        "post_static_dropout":             param["model"]["post_static"].get("dropout"),
        "weight_decay":                    param["optim"].get("weight_decay"),
        "epochs":                          param["training"]["epochs"],
        "max_epochs":                      param["training"]["epochs"],
        "min_epochs":                      param["training"].get("min_epochs"),
        "early_stopping_patience":         param["training"].get("early_stopping_patience"),
        "devices":                         param["training"]["devices"],
        "strategy":                        param["training"].get("strategy", "auto"),
        "num_sanity_val_steps":            param["training"].get("num_sanity_val_steps"),
        "batch_size":                      param["dataloader"]["batch_size"],
        "batch_size_per_gpu":              param["dataloader"]["batch_size"],
        "global_batch_size":               global_batch_size,
        "physical_global_batch_size":      global_batch_size,
        "val_batch_size":                  param["dataloader"].get("val_batch_size", param["dataloader"]["batch_size"]),
        "num_workers":                     param["dataloader"]["num_workers"],
        "train_num_workers":               param["dataloader"].get("num_workers"),
        "val_num_workers":                 param["dataloader"].get("val_num_workers", param["dataloader"].get("num_workers")),
        "persistent_workers":              param["dataloader"].get("persistent_workers", False),
        "pin_memory":                      param["dataloader"].get("pin_memory", False),
        "scheduler":                       param.get("scheduler", {}),
        "training_validation_regime":      "last_window_only" if args.train_predict_last_only else "rolling_or_capped",
        "final_eval_regime":               "last_window_only" if args.predict_last_only else "full_rolling_windows",
        "train_path":                      str(args.train),
        "val_path":                        str(args.val),
        "test_path":                       str(args.test),
        "use_wandb":                       bool(getattr(args, "use_wandb", False)),
        "enable_memory_monitor":           bool(getattr(args, "enable_memory_monitor", False)),
        "scalar_only_training_validation": bool(getattr(args, "scalar_only_training_validation", False)),
        "disable_training_prediction_storage": bool(getattr(args, "disable_training_prediction_storage", True)),
        "final_eval_return_x":             bool(getattr(args, "final_eval_return_x", True)),
        "final_eval_return_y":             bool(getattr(args, "final_eval_return_y", True)),
        "shm_size_mib":                    int(shm_size_mib) if shm_size_mib and shm_size_mib.isdigit() else shm_size_mib,
        "container_options":               os.environ.get("CONTAINER_OPTIONS"),
        "machineType":                     os.environ.get("MACHINE_TYPE"),
        "machine_type":                    os.environ.get("MACHINE_TYPE"),
        "run_timestamp":                   datetime.datetime.now().isoformat(),
    }
    if batch_diagnostics:
        config.update(batch_diagnostics)
    cfg_path = out_dir / "experiment_c_run_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Config] Run config saved → {cfg_path}")
    return config


# =============================
# W&B logging helpers
# =============================
def _metric_to_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _require_wandb_ready():
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "[W&B] WANDB_API_KEY is not set. Export WANDB_API_KEY before using --use-wandb."
        )
    if WandbLogger is None:
        raise RuntimeError(
            "[W&B] lightning.pytorch.loggers.WandbLogger is unavailable. Install wandb in the training environment."
        )


def _wandb_config(args, param, batch_diagnostics=None):
    context_hours = param["windows"]["context_length"] / 12
    horizon_hours = param["windows"]["horizon"] / 12
    cfg = {
        "experiment_name": getattr(args, "experiment_name", "exp_C_tuning"),
        "run_name": getattr(args, "wandb_run_name", None),
        "gcs_output_path": getattr(args, "gcs_output_path", None),
        "context_hours": context_hours,
        "horizon_hours": horizon_hours,
        "learning_rate": param["optim"]["lr"],
        "dropout": param["model"]["dropout"],
        "weight_decay": param["optim"].get("weight_decay"),
        "batch_size_per_gpu": param["dataloader"]["batch_size"],
        "global_batch_size": param["dataloader"]["batch_size"] * param["training"]["devices"],
        "num_workers": param["dataloader"].get("num_workers"),
        "max_val_windows": getattr(args, "max_val_windows", None),
        "load_test_during_training": getattr(args, "load_test_during_training", True),
        "limit_train_batches": getattr(args, "limit_train_batches", None),
        "max_epochs": param["training"]["epochs"],
        "devices": param["training"].get("devices"),
        "strategy": param["training"].get("strategy", "auto"),
        "num_sanity_val_steps": param["training"].get("num_sanity_val_steps"),
        "min_epochs": param["training"].get("min_epochs"),
        "early_stopping_patience": param["training"].get("early_stopping_patience"),
        "train_predict_last_only": getattr(args, "train_predict_last_only", None),
        "predict_last_only": getattr(args, "predict_last_only", None),
        "include_participant_id_embedding": getattr(args, "include_participant_id_embedding", None),
        "devices": param["training"].get("devices"),
        "strategy": param["training"].get("strategy", "auto"),
        "num_sanity_val_steps": param["training"].get("num_sanity_val_steps"),
        "load_test_during_training": bool(getattr(args, "load_test_during_training", True)),
    }
    if batch_diagnostics:
        cfg.update(batch_diagnostics)
    return cfg


def build_loggers(args, out_dir: Path, param, batch_diagnostics=None):
    csv_logger = CSVLogger(save_dir=str(out_dir), name="logs")
    loggers = [csv_logger]
    wandb_logger = None
    if args is not None and getattr(args, "use_wandb", False):
        try:
            _require_wandb_ready()
            wandb_config = _wandb_config(args, param, batch_diagnostics=batch_diagnostics)
            wandb_save_dir = out_dir / "wandb"
            wandb_save_dir.mkdir(parents=True, exist_ok=True)
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=args.wandb_run_name,
                version=os.environ.get("WANDB_RUN_ID") or None,
                save_dir=str(wandb_save_dir),
                log_model=False,
            )
            try:
                wandb_logger.log_hyperparams(wandb_config)
            except Exception as exc:
                print(f"[W&B][WARN] log_hyperparams failed; continuing without aborting training: {exc}")
            try:
                exp = wandb_logger.experiment
                config_obj = getattr(exp, "config", None)
                if hasattr(config_obj, "update"):
                    config_obj.update(wandb_config, allow_val_change=True)
                else:
                    print("[WARN] W&B config object does not support update(); using log_hyperparams only.")
            except Exception as exc:
                print(f"[WARN] W&B config update failed but training will continue: {exc}")
            loggers.append(wandb_logger)
        except Exception as exc:
            wandb_logger = None
            print(f"[W&B][WARN] W&B initialization failed; continuing with CSV logging only: {exc}")
    return (loggers[0] if len(loggers) == 1 else loggers), wandb_logger


def _parse_epoch_from_checkpoint(path):
    if not path:
        return None
    match = re.search(r"epoch[=_-](\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def _checkpoint_gcs_path(args, local_path):
    gcs_root = getattr(args, "gcs_output_path", None)
    if not gcs_root or not local_path:
        return None
    return f"{gcs_root.rstrip('/')}/checkpoints/{Path(local_path).name}"


def build_wandb_summary(args, param, trainer, checkpoint_cb, val_metrics=None, runtime_hours=None, exit_code=0):
    callback_metrics = getattr(trainer, "callback_metrics", {}) or {}
    logged_metrics = getattr(trainer, "logged_metrics", {}) or {}
    metrics = {**logged_metrics, **callback_metrics}
    best_model_path = checkpoint_cb.best_model_path if checkpoint_cb is not None else None
    val_mae = val_rmse = val_mape = val_smape = None
    if val_metrics is not None and len(val_metrics) > 0:
        row = val_metrics.iloc[-1]
        val_mae = _metric_to_float(row.get("mae"))
        val_rmse = _metric_to_float(row.get("rmse"))
        val_mape = _metric_to_float(row.get("mape_pct"))
        val_smape = _metric_to_float(row.get("smape_pct"))
    summary = {
        "experiment_name": getattr(args, "experiment_name", "exp_C_tuning"),
        "run_name": getattr(args, "wandb_run_name", None),
        "gcs_output_path": getattr(args, "gcs_output_path", None),
        "context_hours": param["windows"]["context_length"] / 12,
        "horizon_hours": param["windows"]["horizon"] / 12,
        "learning_rate": param["optim"]["lr"],
        "dropout": param["model"]["dropout"],
        "weight_decay": param["optim"].get("weight_decay"),
        "batch_size_per_gpu": param["dataloader"]["batch_size"],
        "global_batch_size": param["dataloader"]["batch_size"] * param["training"]["devices"],
        "num_workers": param["dataloader"].get("num_workers"),
        "max_val_windows": getattr(args, "max_val_windows", None),
        "limit_train_batches": getattr(args, "limit_train_batches", None),
        "max_epochs": param["training"]["epochs"],
        "min_epochs": param["training"].get("min_epochs"),
        "early_stopping_patience": param["training"].get("early_stopping_patience"),
        "train_loss_epoch": _metric_to_float(metrics.get("train_loss_epoch")),
        "val_loss": _metric_to_float(metrics.get("val_loss")),
        "val_MAE": val_mae,
        "val_RMSE": val_rmse,
        "val_MAPE": val_mape,
        "val_SMAPE": val_smape,
        "best_epoch": _parse_epoch_from_checkpoint(best_model_path),
        "best_val_loss": _metric_to_float(checkpoint_cb.best_model_score if checkpoint_cb is not None else None),
        "best_val_MAE": val_mae,
        "best_val_RMSE": val_rmse,
        "best_checkpoint_gcs_path": _checkpoint_gcs_path(args, best_model_path),
        "runtime_hours": runtime_hours,
        "exit_code": exit_code,
        "had_nan_warning": None,
        "had_unknown_classes_warning": None,
        "had_bus_error": None,
        "had_nccl_error": None,
        "had_oom_error": None,
    }
    return summary


def log_wandb_summary(wandb_logger, summary, out_dir: Path):
    out_path = out_dir / "wandb_run_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[W&B] Summary saved → {out_path}")
    if wandb_logger is None:
        return
    numeric = {
        k: v for k, v in summary.items()
        if isinstance(v, (int, float, bool, np.integer, np.floating)) and v is not None
    }
    if numeric:
        wandb_logger.log_metrics(numeric)
    wandb_logger.experiment.summary.update(summary)
    print("[W&B] Summary logged")


# =============================
# Training
# =============================
def build_loss(param):
    cfg = param.get("loss", {})
    return QuantileLoss(quantiles=cfg.get("quantiles", [0.1, 0.5, 0.9]))


def TFT_train_c(
    train_df, val_df, test_df, param, out_dir: Path,
    extra_static_reals=None,
    include_participant_id_embedding=False,
    predict_last_only=False,
    train_predict_last_only=False,
    max_val_windows=None,
    max_test_windows=None,
    save_all_epoch_checkpoints=False,
    limit_train_batches=None,
    args=None,
):
    train_eval_predict_last_only = train_predict_last_only or predict_last_only
    test_data_loaded_during_training = test_df is not None
    (training, val_dl, test_dl, train_dl,
     val_dataset, test_dataset, static_categoricals) = create_tft_dataloaders_participant_split(
        train_df, val_df, test_df, param,
        extra_static_reals=extra_static_reals,
        include_participant_id_embedding=include_participant_id_embedding,
        predict_last_only=train_eval_predict_last_only,
        max_val_windows=max_val_windows,
        max_test_windows=max_test_windows,
        enable_memory_monitor=bool(getattr(args, "enable_memory_monitor", False)) if args is not None else False,
    )

    print_diagnostics(
        include_participant_id_embedding, train_eval_predict_last_only,
        static_categoricals,
        train_df, val_df, test_df,
        train_dl, val_dl, test_dl,
        max_val_windows=max_val_windows,
    )

    # Diagnostic: first-batch shapes
    try:
        sample_x, sample_y = next(iter(train_dl))
        print(f"[Shapes] encoder_cont : {sample_x['encoder_cont'].shape}")
        print(f"[Shapes] decoder_cont : {sample_x['decoder_cont'].shape}")
        print(f"[Shapes] encoder_cat  : {sample_x['encoder_cat'].shape}")
        print(f"[Shapes] y (target)   : {sample_y[0].shape}")
        del sample_x, sample_y
    except Exception as e:
        print(f"[Shapes] Could not inspect first batch: {e}")

    raw_dataframes_deleted = maybe_delete_raw_dataframes_after_dataloaders(
        train_df, val_df, test_df,
        training, val_dataset, test_dataset,
        train_dl, val_dl, test_dl,
    )
    log_memory("DataLoaders ready")

    # ── Batch / epoch diagnostics ─────────────────────────────────────────────
    n_train_windows  = len(train_dl.dataset)
    bs_per_device    = int(param["dataloader"]["batch_size"])
    n_devices_cfg    = int(param["training"]["devices"])
    gpus_detected    = torch.cuda.device_count()
    global_bs        = bs_per_device * n_devices_cfg
    full_batches     = math.ceil(n_train_windows / global_bs)

    if isinstance(limit_train_batches, int):
        eff_batches = min(limit_train_batches, full_batches)
    elif isinstance(limit_train_batches, float):
        eff_batches = int(limit_train_batches * full_batches)
    else:
        eff_batches = full_batches
    eff_fraction = eff_batches / full_batches if full_batches > 0 else 1.0

    print("\n" + "=" * 62)
    print("  EXPERIMENT C — BATCH / EPOCH DIAGNOSTICS")
    print("=" * 62)
    print(f"  GPUs detected  (torch.cuda.device_count) : {gpus_detected}")
    print(f"  GPUs configured (devices)                : {n_devices_cfg}")
    print(f"  total_train_windows                      : {n_train_windows:,}")
    print(f"  train_batch_size_per_device              : {bs_per_device}")
    print(f"  global_batch_size                        : {global_bs}")
    print(f"  estimated_full_batches_per_epoch         : {full_batches:,}")
    print(f"  limit_train_batches                      : {limit_train_batches}")
    print(f"  effective_batches_per_epoch              : {eff_batches:,}")
    print(f"  effective_epoch_fraction                 : {eff_fraction:.3f}")
    print("=" * 62 + "\n")

    # Merge diagnostics into the run config that was written by save_run_config()
    _batch_diag = {
        "gpus_detected":                   gpus_detected,
        "devices_configured":              n_devices_cfg,
        "total_train_windows":             n_train_windows,
        "train_batch_size_per_device":     bs_per_device,
        "batch_size_per_gpu":              bs_per_device,
        "val_batch_size":                  int(param["dataloader"].get("val_batch_size", bs_per_device)),
        "train_num_workers":               int(param["dataloader"].get("num_workers", 0)),
        "val_num_workers":                 int(param["dataloader"].get("val_num_workers", param["dataloader"].get("num_workers", 0))),
        "global_batch_size":               global_bs,
        "physical_global_batch_size":      global_bs,
        "estimated_full_batches_per_epoch": full_batches,
        "limit_train_batches":             limit_train_batches,
        "effective_batches_per_epoch":     eff_batches,
        "effective_epoch_fraction":        round(eff_fraction, 4),
        "rolling_windows":                 not train_eval_predict_last_only,
        "predict_last_only":               bool(train_eval_predict_last_only),
        "use_wandb":                       bool(getattr(args, "use_wandb", False)) if args is not None else False,
        "enable_memory_monitor":           bool(getattr(args, "enable_memory_monitor", False)) if args is not None else False,
        "scalar_only_training_validation": bool(getattr(args, "scalar_only_training_validation", False)) if args is not None else False,
        "disable_training_prediction_storage": bool(getattr(args, "disable_training_prediction_storage", True)) if args is not None else True,
        "load_test_during_training":       bool(getattr(args, "load_test_during_training", True)) if args is not None else True,
        "test_data_loaded_during_training": test_data_loaded_during_training,
        "raw_dataframes_deleted_after_dataloaders": raw_dataframes_deleted,
        "final_eval_return_x":             bool(getattr(args, "final_eval_return_x", True)) if args is not None else True,
        "final_eval_return_y":             bool(getattr(args, "final_eval_return_y", True)) if args is not None else True,
        "shm_size_mib":                    int(os.environ["SHM_SIZE_MIB"]) if os.environ.get("SHM_SIZE_MIB", "").isdigit() else os.environ.get("SHM_SIZE_MIB"),
        "container_options":               os.environ.get("CONTAINER_OPTIONS"),
    }
    _runtime_env_path = out_dir / "runtime_env.txt"
    if _runtime_env_path.exists():
        with open(_runtime_env_path, "a") as _f:
            _f.write(f"total_train_windows={n_train_windows}\n")
            _f.write(f"global_batch_size={global_bs}\n")
            _f.write(f"physical_global_batch_size={global_bs}\n")
            _f.write(f"estimated_full_batches_per_epoch={full_batches}\n")
            _f.write(f"limit_train_batches={limit_train_batches}\n")
            _f.write(f"effective_batches_per_epoch={eff_batches}\n")
            _f.write(f"effective_epoch_fraction={eff_fraction:.4f}\n")
            _f.write(f"max_val_windows={max_val_windows}\n")
            _f.write(f"train_num_workers={int(param['dataloader'].get('num_workers', 0))}\n")
            _f.write(f"val_num_workers={int(param['dataloader'].get('val_num_workers', param['dataloader'].get('num_workers', 0)))}\n")
            _f.write(f"val_batch_size={int(param['dataloader'].get('val_batch_size', bs_per_device))}\n")
            _f.write(f"rolling_windows={str(not train_eval_predict_last_only).lower()}\n")
            _f.write(f"predict_last_only={str(bool(train_eval_predict_last_only)).lower()}\n")
            _f.write(f"use_wandb={str(bool(getattr(args, 'use_wandb', False)) if args is not None else False).lower()}\n")
            _f.write(f"enable_memory_monitor={str(bool(getattr(args, 'enable_memory_monitor', False)) if args is not None else False).lower()}\n")
            _f.write(f"scalar_only_training_validation={str(bool(getattr(args, 'scalar_only_training_validation', False)) if args is not None else False).lower()}\n")
            _f.write(f"disable_training_prediction_storage={str(bool(getattr(args, 'disable_training_prediction_storage', True)) if args is not None else True).lower()}\n")
            _f.write(f"load_test_during_training={str(bool(getattr(args, 'load_test_during_training', True)) if args is not None else True).lower()}\n")
            _f.write(f"test_data_loaded_during_training={str(test_data_loaded_during_training).lower()}\n")
            _f.write(f"raw_dataframes_deleted_after_dataloaders={str(raw_dataframes_deleted).lower()}\n")
            _f.write(f"final_eval_return_x={str(bool(getattr(args, 'final_eval_return_x', True)) if args is not None else True).lower()}\n")
            _f.write(f"final_eval_return_y={str(bool(getattr(args, 'final_eval_return_y', True)) if args is not None else True).lower()}\n")
            _f.write(f"SHM_SIZE_MIB={os.environ.get('SHM_SIZE_MIB')}\n")
            _f.write(f"container_options={os.environ.get('CONTAINER_OPTIONS')}\n")
            _f.write(f"machineType={os.environ.get('MACHINE_TYPE')}\n")

    _cfg_path = out_dir / "experiment_c_run_config.json"
    if _cfg_path.exists():
        with open(_cfg_path) as _f:
            _existing = json.load(_f)
        _existing.update(_batch_diag)
        with open(_cfg_path, "w") as _f:
            json.dump(_existing, _f, indent=2)
        print(f"[Config] Batch diagnostics merged → {_cfg_path}")

    loss      = build_loss(param)
    model_cfg = param["model"]
    enc       = model_cfg["encoder"]
    post      = model_cfg["post_static"]
    enc_kw    = enc["mamba_kwargs"]
    post_kw   = post["mamba_kwargs"]
    sched     = param.get("scheduler", {})
    if all(metric is not None for metric in (MAE, RMSE, MAPE, SMAPE)):
        logging_metrics = nn.ModuleList([MAE(), RMSE(), MAPE(), SMAPE()])
        print("[Validation] Scalar logging metrics enabled: val_loss, val_MAE, val_RMSE, val_MAPE, val_SMAPE")
    else:
        logging_metrics = nn.ModuleList([])
        print("[Validation][WARN] PyTorch Forecasting scalar metric classes unavailable; using val_loss only during trainer.fit.")

    tft = MambaTFT.from_dataset(
        training,
        learning_rate=param["optim"]["lr"],
        hidden_size=model_cfg["hidden_size"],
        attention_head_size=2,
        dropout=model_cfg["dropout"],
        loss=loss,
        logging_metrics=logging_metrics,
        log_interval=10,
        log_val_interval=1,
        # pytorch-forecasting 1.4.0 exposes ReduceLROnPlateau patience here.
        # TODO: wire factor/min_lr after verifying the cloud image accepts those kwargs.
        reduce_on_plateau_patience=int(sched.get("patience", 2)),
        weight_decay=param["optim"].get("weight_decay", 0.0),
        enc_depth=enc.get("mamba_depth", 4),
        enc_dropout=enc.get("dropout", 0.2),
        enc_checkpoint=False,
        enc_d_state=enc_kw.get("d_state", 128),
        enc_d_conv=enc_kw.get("d_conv", 4),
        enc_expand=enc_kw.get("expand", 4),
        enc_dt_rank=enc_kw.get("dt_rank", None),
        post_depth=post.get("mamba_depth", 1),
        post_dropout=post.get("dropout", 0.2),
        post_d_state=post_kw.get("d_state", 128),
        post_d_conv=post_kw.get("d_conv", 4),
        post_expand=post_kw.get("expand", 4),
        post_dt_rank=post_kw.get("dt_rank", None),
    )
    n_params = sum(p.numel() for p in tft.parameters())
    log_memory(f"Model created | {n_params/1e6:.2f}M parameters")
    if args is not None and getattr(args, "enable_memory_monitor", False):
        log_memory_state("after model creation")

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    context_steps = int(param["windows"]["context_length"])

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"mamba{context_steps}-static-c-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=-1 if save_all_epoch_checkpoints else 1,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=1,
    )
    last_ckpt = ckpt_dir / "last.ckpt"
    ckpt_path = str(last_ckpt) if last_ckpt.exists() else None
    print(f"[INFO] {'Resuming from: ' + ckpt_path if ckpt_path else 'No checkpoint — starting fresh'}")

    train_cfg  = param["training"]
    logger_arg, wandb_logger = build_loggers(args, out_dir, param, batch_diagnostics=_batch_diag)
    # limit_train_batches: int → absolute batch count, float → fraction of epoch, None → full epoch
    _ltb = limit_train_batches if limit_train_batches is not None else 1.0
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=train_cfg.get("early_stopping_patience", 5),
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        checkpoint_cb,
    ]
    if args is not None and getattr(args, "enable_memory_monitor", False):
        callbacks.append(MemoryMonitorCallback())

    print(f"[Trainer] devices={train_cfg['devices']} strategy={train_cfg.get('strategy', 'auto')} num_sanity_val_steps={train_cfg.get('num_sanity_val_steps', 2)}")
    print(f"[Trainer] env WORLD_SIZE={os.environ.get('WORLD_SIZE', '')} RANK={os.environ.get('RANK', '')} LOCAL_RANK={os.environ.get('LOCAL_RANK', '')} GLOBAL_RANK={os.environ.get('GLOBAL_RANK', '')}")

    trainer = Trainer(
        max_epochs=train_cfg["epochs"],
        min_epochs=train_cfg.get("min_epochs", 0),
        gradient_clip_val=train_cfg["gradient_clip_val"],
        accelerator="gpu",
        devices=train_cfg["devices"],
        strategy=train_cfg.get("strategy", "auto"),
        val_check_interval=train_cfg.get("val_check_interval", 1.0),
        limit_val_batches=train_cfg.get("limit_val_batches", 1.0),
        limit_train_batches=_ltb,
        num_sanity_val_steps=train_cfg.get("num_sanity_val_steps", 2),
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger_arg,
    )

    if args is not None and getattr(args, "disable_training_prediction_storage", True):
        print("[Validation] trainer.fit uses Lightning validation only; no predict(return_x/return_y) call is made during training validation.")
    log_memory("Starting training")
    if args is not None and getattr(args, "enable_memory_monitor", False):
        log_memory_state("before trainer.fit")
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt_path)
    if args is not None and getattr(args, "enable_memory_monitor", False):
        log_memory_state("after trainer.fit")
    cleanup_memory("after trainer.fit")
    log_memory("Training done")

    train_metadata = {
        "checkpoint_callback": checkpoint_cb,
        "wandb_logger": wandb_logger,
        "batch_diagnostics": _batch_diag,
    }
    return tft, trainer, val_dl, test_dl, val_dataset, test_dataset, training, static_categoricals, train_metadata


# =============================
# Evaluation — RMSE, MAE, TIR
# =============================
def evaluate(
    tft, test_dl, out_dir: Path, split_name: str = "test",
    include_participant_id_embedding: bool = False,
    predict_last_only: bool = False,
    context_hours: float = 24,
    horizon_hours: float = 1,
    final_eval_return_x: bool = True,
    final_eval_return_y: bool = True,
):
    print(f"\n[Eval] Running predictions on {split_name} set (participant-disjoint)...")
    tft.eval()
    preds = tft.predict(
        test_dl, mode="prediction", return_x=final_eval_return_x, return_y=final_eval_return_y,
        trainer_kwargs={
            "accelerator": "gpu",
            "logger": CSVLogger(save_dir=str(out_dir), name="eval_logs"),
        }
    )
    # mode="prediction" returns P50 for QuantileLoss: shape (n_windows, horizon)
    y_pred_tensor = preds.output if hasattr(preds, "output") else preds
    y_pred_arr = y_pred_tensor.detach().cpu().numpy() if isinstance(y_pred_tensor, torch.Tensor) else np.asarray(y_pred_tensor)
    y_pred = y_pred_arr.flatten()

    y_true_arr = None
    y_true = None
    if final_eval_return_y and hasattr(preds, "y") and preds.y is not None:
        y_true_tensor = preds.y[0] if isinstance(preds.y, (tuple, list)) else preds.y
        y_true_arr = y_true_tensor.detach().cpu().numpy() if isinstance(y_true_tensor, torch.Tensor) else np.asarray(y_true_tensor)
        y_true = y_true_arr.flatten()

    if y_true is not None:
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        mae  = np.mean(np.abs(y_pred - y_true))
        eps = 1e-8
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100
        tir  = np.mean((y_pred >= 70) & (y_pred <= 180)) * 100
    else:
        print("[Eval][WARN] final_eval_return_y=false or targets unavailable; scalar target metrics will be NaN.")
        rmse = mae = mape = smape = tir = float("nan")

    n_windows, horizon = y_pred_arr.shape
    rolling_windows = not predict_last_only

    print(f"\n{'='*55}")
    print(f"  Split                          : {split_name}")
    print(f"  Rolling windows (predict=False) : {rolling_windows}")
    print(f"  Include pid embedding           : {include_participant_id_embedding}")
    print(f"  N windows evaluated             : {n_windows:,}")
    print(f"  RMSE                           : {rmse:.4f} mg/dL")
    print(f"  MAE                            : {mae:.4f} mg/dL")
    print(f"  MAPE                           : {mape:.4f} %")
    print(f"  SMAPE                          : {smape:.4f} %")
    print(f"  TIR 70-180                     : {tir:.2f} %")
    print(f"{'='*55}\n")

    ram_pct = system_ram_percent()
    if ram_pct is not None and ram_pct > 90.0:
        write_high_ram_warning(out_dir, split_name, ram_pct, "prediction artifact write")
    else:
        export_data = {
            "window_idx": np.repeat(np.arange(n_windows), horizon),
            "step":       np.tile(np.arange(horizon), n_windows),
            "split":      split_name,
            "y_pred":     y_pred,
        }
        if y_true is not None:
            export_data["y_true"] = y_true
        pd.DataFrame(export_data).to_parquet(out_dir / f"predictions_{split_name}.parquet", index=False)
        print(f"Per-window predictions saved → {out_dir / f'predictions_{split_name}.parquet'}")

    new_row = pd.DataFrame([{
        "model":                           f"SSM-CGM-static-C (Exp C, participant-split, {context_hours:g}h ctx, {horizon_hours:g}h horizon)",
        "context_h":                       context_hours,
        "horizon_h":                       horizon_hours,
        "eval_split":                      split_name,
        "include_participant_id_embedding": include_participant_id_embedding,
        "predict_last_only":               predict_last_only,
        "rolling_windows":                 rolling_windows,
        "n_windows":                       n_windows,
        "rmse":                            rmse,
        "mae":                             mae,
        "mape_pct":                        mape,
        "smape_pct":                       smape,
        "tir_70_180_pct":                  tir,
        "run_time":                        datetime.datetime.now().isoformat(),
    }])

    results_path = out_dir / "results_dynamic_static_participant_split.csv"
    if results_path.exists():
        existing = pd.read_csv(results_path)
        new_row = pd.concat([existing, new_row], ignore_index=True)
    new_row.to_csv(results_path, index=False)
    print(f"Results saved → {results_path}")
    del preds, y_pred_arr, y_pred
    if y_true_arr is not None:
        del y_true_arr
    if y_true is not None:
        del y_true
    cleanup_memory(f"after {split_name} final evaluation")
    return new_row


# =============================
# Utilities
# =============================
def load_static_feature_list(path: Path):
    with open(path) as f:
        cols = json.load(f)
    print(f"[Static] Loaded {len(cols)} static features from {path}")
    return cols


def subsample_participants(df, n, seed=42, label=""):
    """Stratified subsample of n participants across study_group strata."""
    pids   = df[["participant_id", "study_group"]].drop_duplicates()
    groups = pids["study_group"].unique()
    per_group = max(1, n // len(groups))
    rng    = np.random.default_rng(seed)
    sampled = []
    for g in groups:
        pool = pids[pids["study_group"] == g]["participant_id"].tolist()
        k    = min(per_group, len(pool))
        sampled.extend(rng.choice(pool, size=k, replace=False).tolist())
    all_pids  = pids["participant_id"].tolist()
    remaining = [p for p in np.random.default_rng(seed + 1).permutation(all_pids)
                 if p not in set(sampled)]
    sampled  += remaining[: n - len(sampled)]
    result = df[df["participant_id"].isin(sampled[:n])].copy()
    actual = result["participant_id"].nunique()
    sg     = result["study_group"].value_counts().to_dict()
    print(f"  Subsampled {label}: {actual} participants {sg}")
    return result


def parse_args():
    DATA_ROOT = Path("/home/myriamcharfeddine/CGM/Data/ssmcgm_ready_exp_C")
    p = argparse.ArgumentParser(
        description="SSM-CGM mamba288-static (Experiment C — participant split)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data paths
    p.add_argument("--train", type=Path, default=DATA_ROOT / "train_timeseries_static.feather")
    p.add_argument("--val",   type=Path, default=DATA_ROOT / "val_timeseries_static.feather")
    p.add_argument("--test",  type=Path, default=DATA_ROOT / "test_timeseries_static.feather")
    p.add_argument("--out",   type=Path, default=Path("/home/myriamcharfeddine/CGM/Data/results/exp_C_full"))
    p.add_argument("--static-feature-list", type=Path, default=DATA_ROOT / "static_feature_list.json")

    # Run mode
    p.add_argument("--eval-only",  action="store_true",
                   help="Skip training; load best checkpoint and evaluate on test set")
    p.add_argument("--smoke",      action="store_true",
                   help="Smoke test: ~200 participants (160/25/15), 3 epochs")
    p.add_argument("--epochs",     type=int, default=None,
                   help="Override max_epochs (default: 30, smoke default: 3)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--context-hours", type=float, default=None,
                   help="Override encoder context length in hours. 48h -> 576 five-minute steps.")
    p.add_argument("--horizon-hours", type=float, default=None,
                   help="Override forecast horizon in hours. Default remains 1h.")

    # Ablation flags
    p.add_argument("--include-participant-id-embedding", action="store_true",
                   help="Add participant_id to static_categoricals (OOV → shared NaN embedding). "
                        "Default: participant_id is group_ids only, not a predictive feature.")
    p.add_argument("--predict-last-only", action="store_true",
                   help="Use predict=True for val/test (1 window per participant). "
                        "Default: rolling windows (predict=False) for stable metrics.")
    p.add_argument("--train-predict-last-only", action="store_true", default=False,
                   help="Use predict=True for training-time validation/test loaders. "
                        "Default: rolling/capped windows so --max-val-windows can reduce validation noise.")

    # Large-run controls
    p.add_argument("--max-val-windows",  type=int, default=None,
                   help="Randomly cap training-time val windows after dataset creation")
    p.add_argument("--max-test-windows", type=int, default=None,
                   help="Randomly cap final/eval-only test windows after dataset creation")
    p.add_argument("--skip-final-eval", action="store_true",
                   help="Train and save checkpoints only; skip post-training validation/test prediction.")
    p.add_argument("--skip-test-eval", action="store_true",
                   help="After training, evaluate validation only and skip test prediction.")
    p.add_argument("--save-all-epoch-checkpoints", action="store_true",
                   help="Save a checkpoint at every epoch instead of only best plus last.")
    p.add_argument("--devices", type=int, default=None,
                   help="Override Lightning Trainer devices, e.g. 4 for a2-highgpu-4g.")
    p.add_argument("--strategy", default=None,
                   help="Override Lightning Trainer strategy, e.g. ddp_fork.")
    p.add_argument("--num-sanity-val-steps", type=int, default=None,
                   help="Override Lightning Trainer num_sanity_val_steps.")

    # Epoch-size control
    def _parse_limit_train_batches(v: str):
        """'20000' → int(20000)  |  '0.5' / '1.0' → float"""
        if "." in v:
            return float(v)
        return int(v)

    p.add_argument("--limit-train-batches", type=_parse_limit_train_batches, default=None,
                   metavar="N_OR_FRAC",
                   help="Cap training batches per epoch: int (e.g. 20000) or float fraction "
                        "(e.g. 0.5). None = full epoch. Forwarded to Lightning Trainer.")
    p.add_argument("--val-batch-size", type=int, default=None,
                   help="Override val_batch_size from param defaults (default: 4).")

    def _parse_bool(v: str):
        value = str(v).strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return True
        if value in {"0", "false", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError("Expected true or false")

    # Hyperparameter tuning controls
    p.add_argument("--learning-rate", type=float, default=None,
                   help="Override AdamW learning rate.")
    p.add_argument("--dropout", type=float, default=None,
                   help="Override model, encoder, and post-static dropout.")
    p.add_argument("--weight-decay", type=float, default=None,
                   help="Override AdamW weight decay.")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Override training DataLoader num_workers.")
    p.add_argument("--val-num-workers", type=int, default=None,
                   help="Override validation/test DataLoader num_workers.")
    p.add_argument("--enable-memory-monitor", type=_parse_bool, default=False,
                   help="Print RAM, /dev/shm, and GPU memory around training and validation.")
    p.add_argument("--scalar-only-training-validation", type=_parse_bool, default=False,
                   help="Declare/enforce scalar-only training validation diagnostics; final evaluation remains separate.")
    p.add_argument("--disable-training-prediction-storage", type=_parse_bool, default=True,
                   help="Keep trainer.fit validation separate from prediction/export paths.")
    p.add_argument("--load-test-during-training", type=_parse_bool, default=True,
                   help="Load/build test data before trainer.fit. Set false for memory-survival training runs.")
    p.add_argument("--final-eval-return-x", type=_parse_bool, default=True,
                   help="Use return_x for post-training final evaluation/export only.")
    p.add_argument("--final-eval-return-y", type=_parse_bool, default=True,
                   help="Use return_y for post-training final evaluation/export only.")
    p.add_argument("--min-epochs", type=int, default=None,
                   help="Override Trainer min_epochs.")
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="Override early stopping patience on val_loss.")

    # W&B logging
    p.add_argument("--use-wandb", action="store_true",
                   help="Enable W&B logging. Requires WANDB_API_KEY in the environment.")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ssmcgm"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--gcs-output-path", default=os.environ.get("GCS_OUTPUT_PATH"),
                   help="GCS output path for logging artifact/checkpoint locations.")
    p.add_argument("--experiment-name", default="exp_C_tuning")

    return p.parse_args()


# =============================
# Main
# =============================
if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available — GPU required for Mamba"
    name = torch.cuda.get_device_name(0)
    cap  = torch.cuda.get_device_capability(0)
    print(f"[GPU] {name}  CC={cap}  BF16={'OK' if torch.cuda.is_bf16_supported() else 'NO'}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark        = True

    # PyTorch 2.6 changed weights_only default to True in torch.load, which
    # blocks pytorch-forecasting/pandas classes stored in Lightning checkpoints.
    # Patch torch.load to keep weights_only=False for checkpoints from this
    # trusted environment (same machine that created them).
    import functools as _functools
    _orig_load = torch.load
    @_functools.wraps(_orig_load)
    def _patched_load(f, *_a, **_kw):
        _kw["weights_only"] = False  # force override — Lightning passes True explicitly
        return _orig_load(f, *_a, **_kw)
    torch.load = _patched_load

    args = parse_args()
    run_start_time = time.time()
    if args.use_wandb:
        try:
            _require_wandb_ready()
        except Exception as exc:
            print(f"[W&B][WARN] W&B readiness check failed; training will continue with CSV logging only if W&B init also fails: {exc}")

    if args.enable_memory_monitor:
        log_memory_state("before loading data")

    # ── Validate input files ──────────────────────────────────────────────────
    requires_test_file = (
        args.eval_only
        or args.load_test_during_training
        or (not args.skip_final_eval and not args.skip_test_eval)
    )
    required_files = [
        (args.train, "train feather"),
        (args.val,   "val feather"),
        (args.static_feature_list, "static_feature_list.json"),
    ]
    if requires_test_file:
        required_files.append((args.test, "test feather"))
    for fpath, label in required_files:
        if not fpath.exists():
            raise FileNotFoundError(
                f"[ERROR] {label} not found: {fpath}\n"
                "  Run prepare_ssmcgm_data_experiment_c.py first."
            )

    extra_static_reals = load_static_feature_list(args.static_feature_list)

    # ── Apply smoke / tuning / epoch overrides ────────────────────────────────
    param = copy.deepcopy(param_24h)
    steps_per_hour = 12  # 5-minute samples

    if args.context_hours is not None:
        param["windows"]["context_length"] = int(round(args.context_hours * steps_per_hour))
    if args.horizon_hours is not None:
        param["windows"]["horizon"] = int(round(args.horizon_hours * steps_per_hour))

    n_train = n_val = n_test = None
    if args.smoke:
        print("\n[SMOKE TEST] ~200 participants (160 train / 25 val / 15 test) · 3 epochs")
        n_train, n_val, n_test = 160, 25, 15
        param["training"]["epochs"]             = 3
        param["training"]["min_epochs"]         = 1
        param["training"]["val_check_interval"] = 1.0
        args.out = Path("/home/myriamcharfeddine/CGM/Data/results/exp_C_smoke")

    if args.epochs is not None:
        param["training"]["epochs"] = args.epochs
    if args.devices is not None:
        param["training"]["devices"] = args.devices
    if args.strategy is not None:
        param["training"]["strategy"] = args.strategy
    if args.num_sanity_val_steps is not None:
        param["training"]["num_sanity_val_steps"] = args.num_sanity_val_steps
    if args.batch_size is not None:
        param["dataloader"]["batch_size"] = args.batch_size
    if args.val_batch_size is not None:
        param["dataloader"]["val_batch_size"] = args.val_batch_size
    if args.val_num_workers is not None:
        param["dataloader"]["val_num_workers"] = args.val_num_workers
    if args.learning_rate is not None:
        param["optim"]["lr"] = args.learning_rate
    if args.dropout is not None:
        param["model"]["dropout"] = args.dropout
        param["model"]["encoder"]["dropout"] = args.dropout
        param["model"]["post_static"]["dropout"] = args.dropout
    if args.weight_decay is not None:
        param["optim"]["weight_decay"] = args.weight_decay
    if args.num_workers is not None:
        param["dataloader"]["num_workers"] = args.num_workers
    if args.min_epochs is not None:
        param["training"]["min_epochs"] = args.min_epochs
    if args.early_stop_patience is not None:
        param["training"]["early_stopping_patience"] = args.early_stop_patience
    if param["training"].get("min_epochs", 0) > param["training"]["epochs"]:
        print("[Config] min_epochs exceeds max_epochs; capping min_epochs at max_epochs.")
        param["training"]["min_epochs"] = param["training"]["epochs"]
    param.setdefault("runtime", {})["enable_memory_monitor"] = bool(args.enable_memory_monitor)
    param["runtime"]["disable_training_prediction_storage"] = bool(args.disable_training_prediction_storage)
    param["runtime"]["load_test_during_training"] = bool(args.load_test_during_training)
    param["runtime"]["final_eval_return_x"] = bool(args.final_eval_return_x)
    param["runtime"]["final_eval_return_y"] = bool(args.final_eval_return_y)

    args.out.mkdir(parents=True, exist_ok=True)

    # ── Load feathers ─────────────────────────────────────────────────────────
    print(f"\nLoading train : {args.train}")
    train_df = pd.read_feather(args.train)
    print(f"Loading val   : {args.val}")
    val_df   = pd.read_feather(args.val)
    test_df = None
    if args.eval_only or args.load_test_during_training:
        print(f"Loading test  : {args.test}")
        test_df  = pd.read_feather(args.test)
    else:
        print("[Data] Skipping test DataFrame load before trainer.fit (--load-test-during-training=false).")
    if args.enable_memory_monitor:
        log_memory_state("after loading train/val data" + (" plus test" if test_df is not None else " (test skipped)"))

    if n_train:
        train_df = subsample_participants(train_df, n_train, seed=42,  label="train")
    if n_val:
        val_df   = subsample_participants(val_df,   n_val,   seed=100, label="val")
    if n_test and test_df is not None:
        test_df  = subsample_participants(test_df,  n_test,  seed=200, label="test")

    print(f"\n  Train : {train_df['participant_id'].nunique():,} pids | {len(train_df):,} rows")
    print(f"  Val   : {val_df['participant_id'].nunique():,} pids | {len(val_df):,} rows")
    if test_df is not None:
        print(f"  Test  : {test_df['participant_id'].nunique():,} pids | {len(test_df):,} rows")
    else:
        print("  Test  : not loaded during training")

    # ── Pre-training validation ───────────────────────────────────────────────
    validate_before_training(train_df, val_df, test_df, extra_static_reals)

    # ── Save run config ───────────────────────────────────────────────────────
    save_run_config(args, None, param, args.out)  # static_categoricals resolved inside dataloader fn

    if args.eval_only:
        ckpt_dir = args.out / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("mamba*-static-c-*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
        best_ckpt = ckpts[-1]
        print(f"Loading checkpoint: {best_ckpt}")
        (training, val_dl, test_dl, train_dl,
         val_dataset, test_dataset, static_categoricals) = create_tft_dataloaders_participant_split(
            train_df, val_df, test_df, param,
            extra_static_reals=extra_static_reals,
            include_participant_id_embedding=args.include_participant_id_embedding,
            predict_last_only=args.predict_last_only,
            max_val_windows=args.max_val_windows,
            max_test_windows=args.max_test_windows,
            enable_memory_monitor=args.enable_memory_monitor,
        )
        print_diagnostics(
            args.include_participant_id_embedding, args.predict_last_only,
            static_categoricals,
            train_df, val_df, test_df,
            train_dl, val_dl, test_dl,
            max_val_windows=args.max_val_windows,
        )
        tft = MambaTFT.load_from_checkpoint(str(best_ckpt), weights_only=False)
        context_hours = param["windows"]["context_length"] / 12
        horizon_hours = param["windows"]["horizon"] / 12
        evaluate(tft, val_dl, args.out, split_name="val",
                 include_participant_id_embedding=args.include_participant_id_embedding,
                 predict_last_only=args.predict_last_only,
                 context_hours=context_hours,
                 horizon_hours=horizon_hours,
                 final_eval_return_x=args.final_eval_return_x,
                 final_eval_return_y=args.final_eval_return_y)
        if not args.skip_test_eval:
            evaluate(tft, test_dl, args.out, split_name="test",
                     include_participant_id_embedding=args.include_participant_id_embedding,
                     predict_last_only=args.predict_last_only,
                     context_hours=context_hours,
                     horizon_hours=horizon_hours,
                     final_eval_return_x=args.final_eval_return_x,
                     final_eval_return_y=args.final_eval_return_y)
    else:
        training_dfs = [train_df, val_df, test_df]
        del train_df, val_df, test_df
        (tft, trainer, val_dl, test_dl,
         val_dataset, test_dataset, training,
         static_categoricals, train_metadata) = TFT_train_c(
            training_dfs.pop(0), training_dfs.pop(0), training_dfs.pop(0), param, args.out,
            extra_static_reals=extra_static_reals,
            include_participant_id_embedding=args.include_participant_id_embedding,
            predict_last_only=args.predict_last_only,
            train_predict_last_only=args.train_predict_last_only,
            max_val_windows=args.max_val_windows,
            max_test_windows=None,
            save_all_epoch_checkpoints=args.save_all_epoch_checkpoints,
            limit_train_batches=args.limit_train_batches,
            args=args,
        )
        # Update run config with resolved static_categoricals and batch diagnostics.
        save_run_config(
            args, static_categoricals, param, args.out,
            batch_diagnostics=train_metadata.get("batch_diagnostics"),
        )

        checkpoint_cb = train_metadata["checkpoint_callback"]
        wandb_logger = train_metadata.get("wandb_logger")
        best_model_path = checkpoint_cb.best_model_path
        eval_tft = tft
        if best_model_path and Path(best_model_path).exists():
            print(f"[Eval] Loading best checkpoint for validation metrics: {best_model_path}")
            eval_tft = MambaTFT.load_from_checkpoint(best_model_path, weights_only=False)

        val_metrics = None
        context_hours = param["windows"]["context_length"] / 12
        horizon_hours = param["windows"]["horizon"] / 12
        if args.skip_final_eval:
            print("\n[Eval] Skipped final validation/test evaluation (--skip-final-eval).")
            print("       W&B will contain Lightning validation loss but not post-training MAE/RMSE/MAPE/SMAPE.")
        else:
            # Reuse the training validation loader so final validation does not rebuild
            # train/val/test datasets or reload the raw DataFrames. Test eval, if enabled,
            # is built after trainer.fit from the saved training dataset.
            final_val_dl = val_dl
            val_metrics = evaluate(
                eval_tft, final_val_dl, args.out, split_name="val",
                include_participant_id_embedding=args.include_participant_id_embedding,
                predict_last_only=args.predict_last_only,
                context_hours=context_hours,
                horizon_hours=horizon_hours,
                final_eval_return_x=args.final_eval_return_x,
                final_eval_return_y=args.final_eval_return_y,
            )
            if args.enable_memory_monitor:
                log_memory_state("after final validation evaluation")
            cleanup_memory("after final validation evaluation")
            if args.skip_test_eval:
                print("\n[Eval] Skipped test evaluation (--skip-test-eval).")
            else:
                print(f"\n[Eval] Loading test data after trainer.fit: {args.test}")
                final_test_df = pd.read_feather(args.test)
                if n_test:
                    final_test_df = subsample_participants(final_test_df, n_test, seed=200, label="test")
                missing = [c for c in extra_static_reals if c not in final_test_df.columns]
                if missing:
                    raise ValueError(f"Static columns missing from final test DataFrame: {missing}")
                if args.enable_memory_monitor:
                    log_memory_state("after loading final test data")
                dl = param.get("dataloader", {})
                final_test_dataset = TimeSeriesDataSet.from_dataset(
                    training, final_test_df, predict=args.predict_last_only, stop_randomization=True,
                )
                final_test_dl = _make_subsample_dl(
                    final_test_dataset, args.max_test_windows,
                    int(dl.get("val_batch_size", dl.get("batch_size", 32))),
                    bool(dl.get("pin_memory", False)),
                    num_workers=int(dl.get("val_num_workers", dl.get("num_workers", 0))),
                    persistent_workers=bool(dl.get("val_persistent_workers", dl.get("persistent_workers", False))) if int(dl.get("val_num_workers", dl.get("num_workers", 0))) > 0 else False,
                    seed=43,
                )
                del final_test_df
                cleanup_memory("after final test dataloader construction")
                evaluate(
                    eval_tft, final_test_dl, args.out, split_name="test",
                    include_participant_id_embedding=args.include_participant_id_embedding,
                    predict_last_only=args.predict_last_only,
                    context_hours=context_hours,
                    horizon_hours=horizon_hours,
                    final_eval_return_x=args.final_eval_return_x,
                    final_eval_return_y=args.final_eval_return_y,
                )
                if args.enable_memory_monitor:
                    log_memory_state("after final test evaluation")
                cleanup_memory("after final test evaluation")

        runtime_hours = (time.time() - run_start_time) / 3600
        summary = build_wandb_summary(
            args, param, trainer, checkpoint_cb,
            val_metrics=val_metrics,
            runtime_hours=runtime_hours,
            exit_code=0,
        )
        log_wandb_summary(wandb_logger, summary, args.out)
