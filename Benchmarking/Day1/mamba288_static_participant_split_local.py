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
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

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
        "patience": 15,
        "factor": 0.2,
        "min_lr": 1e-5,
        "monitor": "val_loss",
        "mode": "min",
    },
    "training": {
        "epochs": 30,
        "devices": 4,          # a2-highgpu-4g (4×A100, 48 vCPU); fits standard CPU quota
        "gradient_clip_val": 1.0,
        "strategy": "auto",
        "val_check_interval": 1.0,
        "limit_val_batches": 1.0,
    },
    "dataloader": {
        "batch_size": 8,
        "val_batch_size": 4,   # rolling-window eval generates many windows; keep small to avoid OOM
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
def _make_subsample_dl(dataset, max_windows, bs, pin_memory, seed=42):
    """
    Create a DataLoader from dataset, optionally subsampling to max_windows.
    Used only for smoke/debug purposes — does not affect full runs.
    """
    if max_windows is not None and len(dataset) > max_windows:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(dataset), size=max_windows, replace=False).tolist())
        # Borrow collate_fn from a minimal dataloader over the full dataset
        tmp_dl = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        return DataLoader(
            Subset(dataset, idx),
            batch_size=bs,
            num_workers=0,
            collate_fn=tmp_dl.collate_fn,
            pin_memory=pin_memory,
        )
    return dataset.to_dataloader(train=False, batch_size=bs, num_workers=0, pin_memory=pin_memory)


def create_tft_dataloaders_participant_split(
    train_df, val_df, test_df, param, extra_static_reals=None,
    include_participant_id_embedding=False,
    predict_last_only=False,
    max_val_windows=None,
    max_test_windows=None,
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
    test_dataset = TimeSeriesDataSet.from_dataset(
        training, test_df, predict=predict_last_only, stop_randomization=True,
    )

    dl  = param.get("dataloader", {})
    bs  = int(dl.get("batch_size", 32))
    val_bs = int(dl.get("val_batch_size", bs))
    nw  = int(dl.get("num_workers", 4))
    pin = bool(dl.get("pin_memory", True))
    pw  = bool(dl.get("persistent_workers", True)) if nw > 0 else False

    train_dl = training.to_dataloader(
        train=True, batch_size=bs, num_workers=nw, persistent_workers=pw, pin_memory=pin
    )
    val_dl  = _make_subsample_dl(val_dataset,  max_val_windows,  val_bs, pin, seed=42)
    test_dl = _make_subsample_dl(test_dataset, max_test_windows, val_bs, pin, seed=43)

    return training, val_dl, test_dl, train_dl, val_dataset, test_dataset, static_categoricals


# =============================
# Pre-training validation
# =============================
def validate_before_training(train_df, val_df, test_df, extra_static_reals):
    print("\n[Pre-training Validation]")
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = [c for c in extra_static_reals if c not in sdf.columns]
        if missing:
            raise ValueError(f"Static columns missing from {name} DataFrame: {missing}")
        nan_counts = sdf[extra_static_reals].isna().sum()
        bad = nan_counts[nan_counts > 0]
        if len(bad) > 0:
            raise ValueError(f"NaN found in {name} static columns:\n{bad}")

    train_pids = set(train_df["participant_id"].unique())
    val_pids   = set(val_df["participant_id"].unique())
    test_pids  = set(test_df["participant_id"].unique())
    overlap_tv = train_pids & val_pids
    overlap_tt = train_pids & test_pids
    overlap_vt = val_pids   & test_pids
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"Participant overlap detected — splits are NOT disjoint!\n"
            f"  train∩val={len(overlap_tv)}, train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}"
        )
    print("  Participant disjointness : OK (train ∩ val ∩ test = ∅)")
    print("  Static cols OK           : no NaN")


# =============================
# Diagnostics
# =============================
def print_diagnostics(
    include_participant_id_embedding, predict_last_only,
    static_categoricals,
    train_df, val_df, test_df,
    train_dl, val_dl, test_dl,
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
    print(f"  predict_last_only (1 win/pid)    : {predict_last_only}")
    print(f"  rolling_windows (default)        : {not predict_last_only}")
    print(f"  static_categoricals              : {static_categoricals}")
    print(f"  group_ids                        : ['participant_id']")
    print(f"  Train participants               : {train_df['participant_id'].nunique():,}")
    print(f"  Val   participants               : {val_df['participant_id'].nunique():,}")
    print(f"  Test  participants               : {test_df['participant_id'].nunique():,}")
    print(f"  Train windows                   : {_dl_len(train_dl):,}")
    print(f"  Val   windows                   : {_dl_len(val_dl)}")
    print(f"  Test  windows                   : {_dl_len(test_dl)}")
    print("=" * 62 + "\n")


# =============================
# Save run config
# =============================
def save_run_config(args, static_categoricals, param, out_dir, batch_diagnostics=None):
    config = {
        "include_participant_id_embedding": args.include_participant_id_embedding,
        "predict_last_only":               args.predict_last_only,
        "train_predict_last_only":         args.train_predict_last_only,
        "max_val_windows":                 args.max_val_windows,
        "max_test_windows":                args.max_test_windows,
        "skip_final_eval":                 args.skip_final_eval,
        "save_all_epoch_checkpoints":       args.save_all_epoch_checkpoints,
        "limit_train_batches":             getattr(args, "limit_train_batches", None),
        "static_categoricals":             static_categoricals,
        "group_ids":                       ["participant_id"],
        "epochs":                          param["training"]["epochs"],
        "devices":                         param["training"]["devices"],
        "batch_size":                      param["dataloader"]["batch_size"],
        "val_batch_size":                  param["dataloader"].get("val_batch_size", param["dataloader"]["batch_size"]),
        "num_workers":                     param["dataloader"]["num_workers"],
        "training_validation_regime":      "last_window_only" if args.train_predict_last_only else "rolling_or_capped",
        "final_eval_regime":               "last_window_only" if args.predict_last_only else "full_rolling_windows",
        "train_path":                      str(args.train),
        "val_path":                        str(args.val),
        "test_path":                       str(args.test),
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
):
    train_eval_predict_last_only = train_predict_last_only or predict_last_only
    (training, val_dl, test_dl, train_dl,
     val_dataset, test_dataset, static_categoricals) = create_tft_dataloaders_participant_split(
        train_df, val_df, test_df, param,
        extra_static_reals=extra_static_reals,
        include_participant_id_embedding=include_participant_id_embedding,
        predict_last_only=train_eval_predict_last_only,
        max_val_windows=max_val_windows,
        max_test_windows=max_test_windows,
    )

    print_diagnostics(
        include_participant_id_embedding, train_eval_predict_last_only,
        static_categoricals,
        train_df, val_df, test_df,
        train_dl, val_dl, test_dl,
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

    del train_df
    gc.collect()
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
        "global_batch_size":               global_bs,
        "estimated_full_batches_per_epoch": full_batches,
        "effective_batches_per_epoch":     eff_batches,
        "effective_epoch_fraction":        round(eff_fraction, 4),
    }
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

    tft = MambaTFT.from_dataset(
        training,
        learning_rate=param["optim"]["lr"],
        hidden_size=model_cfg["hidden_size"],
        attention_head_size=2,
        dropout=model_cfg["dropout"],
        loss=loss,
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=4,
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

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="mamba288-static-c-{epoch:02d}-{val_loss:.2f}",
        save_top_k=-1 if save_all_epoch_checkpoints else 3,
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
    csv_logger = CSVLogger(save_dir=str(out_dir), name="logs")
    # limit_train_batches: int → absolute batch count, float → fraction of epoch, None → full epoch
    _ltb = limit_train_batches if limit_train_batches is not None else 1.0
    trainer = Trainer(
        max_epochs=train_cfg["epochs"],
        gradient_clip_val=train_cfg["gradient_clip_val"],
        accelerator="gpu",
        devices=train_cfg["devices"],
        strategy=train_cfg.get("strategy", "auto"),
        val_check_interval=train_cfg.get("val_check_interval", 1.0),
        limit_val_batches=train_cfg.get("limit_val_batches", 1.0),
        limit_train_batches=_ltb,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=2, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
            checkpoint_cb,
        ],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=csv_logger,
    )

    log_memory("Starting training")
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt_path)
    log_memory("Training done")

    return tft, trainer, val_dl, test_dl, val_dataset, test_dataset, training, static_categoricals


# =============================
# Evaluation — RMSE, MAE, TIR
# =============================
def evaluate(
    tft, test_dl, out_dir: Path, split_name: str = "test",
    include_participant_id_embedding: bool = False,
    predict_last_only: bool = False,
):
    print(f"\n[Eval] Running predictions on {split_name} set (participant-disjoint)...")
    tft.eval()
    preds = tft.predict(
        test_dl, mode="prediction", return_x=True, return_y=True,
        trainer_kwargs={
            "accelerator": "gpu",
            "logger": CSVLogger(save_dir=str(out_dir), name="eval_logs"),
        }
    )
    # mode="prediction" returns P50 for QuantileLoss: shape (n_windows, horizon)
    y_pred_arr = preds.output.cpu().numpy()  # (n_windows, horizon)
    y_true_arr = preds.y[0].cpu().numpy()    # (n_windows, horizon)
    y_pred = y_pred_arr.flatten()
    y_true = y_true_arr.flatten()

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae  = np.mean(np.abs(y_pred - y_true))
    tir  = np.mean((y_pred >= 70) & (y_pred <= 180)) * 100

    n_windows, horizon = y_pred_arr.shape
    rolling_windows = not predict_last_only

    print(f"\n{'='*55}")
    print(f"  Split                          : {split_name}")
    print(f"  Rolling windows (predict=False) : {rolling_windows}")
    print(f"  Include pid embedding           : {include_participant_id_embedding}")
    print(f"  N windows evaluated             : {n_windows:,}")
    print(f"  RMSE                           : {rmse:.4f} mg/dL")
    print(f"  MAE                            : {mae:.4f} mg/dL")
    print(f"  TIR 70-180                     : {tir:.2f} %")
    print(f"{'='*55}\n")

    pd.DataFrame({
        "window_idx": np.repeat(np.arange(n_windows), horizon),
        "step":       np.tile(np.arange(horizon), n_windows),
        "split":      split_name,
        "y_true":     y_true,
        "y_pred":     y_pred,
    }).to_parquet(out_dir / f"predictions_{split_name}.parquet", index=False)
    print(f"Per-window predictions saved → {out_dir / f'predictions_{split_name}.parquet'}")

    new_row = pd.DataFrame([{
        "model":                           "SSM-CGM-static-C (Exp C, participant-split, 24h ctx, 1h horizon)",
        "context_h":                       24,
        "horizon_h":                       1,
        "eval_split":                      split_name,
        "include_participant_id_embedding": include_participant_id_embedding,
        "predict_last_only":               predict_last_only,
        "rolling_windows":                 rolling_windows,
        "n_windows":                       n_windows,
        "rmse":                            rmse,
        "mae":                             mae,
        "tir_70_180_pct":                  tir,
        "run_time":                        datetime.datetime.now().isoformat(),
    }])

    results_path = out_dir / "results_dynamic_static_participant_split.csv"
    if results_path.exists():
        existing = pd.read_csv(results_path)
        new_row = pd.concat([existing, new_row], ignore_index=True)
    new_row.to_csv(results_path, index=False)
    print(f"Results saved → {results_path}")
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

    # Ablation flags
    p.add_argument("--include-participant-id-embedding", action="store_true",
                   help="Add participant_id to static_categoricals (OOV → shared NaN embedding). "
                        "Default: participant_id is group_ids only, not a predictive feature.")
    p.add_argument("--predict-last-only", action="store_true",
                   help="Use predict=True for val/test (1 window per participant). "
                        "Default: rolling windows (predict=False) for stable metrics.")
    p.add_argument("--train-predict-last-only", action="store_true", default=True,
                   help="Use predict=True for training-time validation/test loaders. "
                        "Default: enabled, so training monitors one window per participant; "
                        "final eval still uses --predict-last-only, so default final eval remains rolling.")

    # Large-run controls
    p.add_argument("--max-val-windows",  type=int, default=None,
                   help="Randomly cap training-time val windows after dataset creation")
    p.add_argument("--max-test-windows", type=int, default=None,
                   help="Randomly cap final/eval-only test windows after dataset creation")
    p.add_argument("--skip-final-eval", action="store_true",
                   help="Train and save checkpoints only; skip expensive post-training test prediction.")
    p.add_argument("--save-all-epoch-checkpoints", action="store_true",
                   help="Save a checkpoint at every epoch instead of only top-3 plus last.")

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

    # ── Validate input files ──────────────────────────────────────────────────
    for fpath, label in [
        (args.train, "train feather"),
        (args.val,   "val feather"),
        (args.test,  "test feather"),
        (args.static_feature_list, "static_feature_list.json"),
    ]:
        if not fpath.exists():
            raise FileNotFoundError(
                f"[ERROR] {label} not found: {fpath}\n"
                "  Run prepare_ssmcgm_data_experiment_c.py first."
            )

    extra_static_reals = load_static_feature_list(args.static_feature_list)

    # ── Apply smoke / epoch overrides ─────────────────────────────────────────
    param = dict(param_24h)
    param["training"]   = dict(param_24h["training"])
    param["dataloader"] = dict(param_24h["dataloader"])

    n_train = n_val = n_test = None
    if args.smoke:
        print("\n[SMOKE TEST] ~200 participants (160 train / 25 val / 15 test) · 3 epochs")
        n_train, n_val, n_test = 160, 25, 15
        param["training"]["epochs"]             = 3
        param["training"]["val_check_interval"] = 1.0
        args.out = Path("/home/myriamcharfeddine/CGM/Data/results/exp_C_smoke")

    if args.epochs is not None:
        param["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        param["dataloader"]["batch_size"] = args.batch_size
    if args.val_batch_size is not None:
        param["dataloader"]["val_batch_size"] = args.val_batch_size

    args.out.mkdir(parents=True, exist_ok=True)

    # ── Load feathers ─────────────────────────────────────────────────────────
    print(f"\nLoading train : {args.train}")
    train_df = pd.read_feather(args.train)
    print(f"Loading val   : {args.val}")
    val_df   = pd.read_feather(args.val)
    print(f"Loading test  : {args.test}")
    test_df  = pd.read_feather(args.test)

    if n_train:
        train_df = subsample_participants(train_df, n_train, seed=42,  label="train")
    if n_val:
        val_df   = subsample_participants(val_df,   n_val,   seed=100, label="val")
    if n_test:
        test_df  = subsample_participants(test_df,  n_test,  seed=200, label="test")

    print(f"\n  Train : {train_df['participant_id'].nunique():,} pids | {len(train_df):,} rows")
    print(f"  Val   : {val_df['participant_id'].nunique():,} pids | {len(val_df):,} rows")
    print(f"  Test  : {test_df['participant_id'].nunique():,} pids | {len(test_df):,} rows")

    # ── Pre-training validation ───────────────────────────────────────────────
    validate_before_training(train_df, val_df, test_df, extra_static_reals)

    # ── Save run config ───────────────────────────────────────────────────────
    save_run_config(args, None, param, args.out)  # static_categoricals resolved inside dataloader fn

    if args.eval_only:
        ckpt_dir = args.out / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("mamba288-static-c-*.ckpt"))
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
        )
        print_diagnostics(
            args.include_participant_id_embedding, args.predict_last_only,
            static_categoricals,
            train_df, val_df, test_df,
            train_dl, val_dl, test_dl,
        )
        tft = MambaTFT.load_from_checkpoint(str(best_ckpt), weights_only=False)
        evaluate(tft, val_dl, args.out, split_name="val",
                 include_participant_id_embedding=args.include_participant_id_embedding,
                 predict_last_only=args.predict_last_only)
        evaluate(tft, test_dl, args.out, split_name="test",
                 include_participant_id_embedding=args.include_participant_id_embedding,
                 predict_last_only=args.predict_last_only)
    else:
        (tft, trainer, val_dl, test_dl,
         val_dataset, test_dataset, training,
         static_categoricals) = TFT_train_c(
            train_df, val_df, test_df, param, args.out,
            extra_static_reals=extra_static_reals,
            include_participant_id_embedding=args.include_participant_id_embedding,
            predict_last_only=args.predict_last_only,
            train_predict_last_only=args.train_predict_last_only,
            max_val_windows=args.max_val_windows,
            max_test_windows=None,
            save_all_epoch_checkpoints=args.save_all_epoch_checkpoints,
            limit_train_batches=args.limit_train_batches,
        )
        # Update run config with resolved static_categoricals
        save_run_config(args, static_categoricals, param, args.out)
        if args.skip_final_eval:
            print("\n[Eval] Skipped final test evaluation (--skip-final-eval).")
            print("       Run later with --eval-only for full rolling-window metrics.")
        else:
            # Rebuild the eval loader separately so training can use cheap validation
            # while final metrics remain full rolling-window by default.
            (_, final_val_dl, final_test_dl, _, _, _, _) = create_tft_dataloaders_participant_split(
                train_df, val_df, test_df, param,
                extra_static_reals=extra_static_reals,
                include_participant_id_embedding=args.include_participant_id_embedding,
                predict_last_only=args.predict_last_only,
                max_val_windows=None,
                max_test_windows=args.max_test_windows,
            )
            evaluate(tft, final_val_dl, args.out, split_name="val",
                     include_participant_id_embedding=args.include_participant_id_embedding,
                     predict_last_only=args.predict_last_only)
            evaluate(tft, final_test_dl, args.out, split_name="test",
                     include_participant_id_embedding=args.include_participant_id_embedding,
                     predict_last_only=args.predict_last_only)
