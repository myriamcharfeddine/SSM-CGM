"""
mamba288_local.py
=================
SSM-CGM baseline (24h context, 1h horizon) — local version.

Changes from the original mamba288.py:
  1. Reads local feather files instead of GCS
  2. Saves checkpoints locally (no GCS upload)
  3. GPU assert moved inside __main__ (safe to import on any machine)

Run:
  conda activate ssmcgm
  cd /home/myriamcharfeddine/CGM/SSM-CGM/Benchmarking/Day1
  python mamba288_local.py \
      --train /home/myriamcharfeddine/CGM/Data/ssmcgm_ready/train_timeseries.feather \
      --test  /home/myriamcharfeddine/CGM/Data/ssmcgm_ready/test_timeseries.feather  \
      --out   /home/myriamcharfeddine/CGM/Data/results
"""

# =============================
# Imports
# =============================
import os
import gc
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from pathlib import Path

from mamba_ssm import Mamba
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import math
import glob
from einops import rearrange


# =============================
# Hyperparameter preset (24h context)
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
        "epochs": 20,
        "min_epochs": 3,
        "early_stop_patience": 3,
        "devices": 4,
        "gradient_clip_val": 1.0,
        "strategy": "auto",
        "val_check_interval": 0.2,
    },
    "dataloader": {
        "batch_size": 32,
        "num_workers": 8,
        "persistent_workers": True,
        "pin_memory": True,
    },
}


# =============================
# Utility
# =============================
def log_memory(message=""):
    if torch.cuda.is_available():
        alloc   = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
    else:
        alloc = reserved = 0.0
    print(f"[{datetime.datetime.now():%H:%M:%S}] {message} | GPU {alloc:.2f}GB alloc / {reserved:.2f}GB reserved")


# =============================
# Mamba blocks (identical to original mamba288.py)
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
# MambaTFT (identical to original)
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
# Data — same columns as Shakson's mamba288.py
# =============================
def _make_subsample_dl(dataset, max_windows, bs, pin_memory, seed=42):
    if max_windows is not None and len(dataset) > max_windows:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(dataset), size=max_windows, replace=False).tolist())
        sampler = torch.utils.data.SubsetRandomSampler(idx)
        return dataset.to_dataloader(
            train=False, batch_size=bs, num_workers=0,
            persistent_workers=False, pin_memory=pin_memory, sampler=sampler,
        )
    return dataset.to_dataloader(
        train=False, batch_size=bs, num_workers=0,
        persistent_workers=False, pin_memory=pin_memory,
    )


def create_tft_dataloaders(train_df, param, max_val_windows=None):
    log_memory("Building TimeSeriesDataSets")

    horizon        = int(param["windows"]["horizon"])
    context_length = int(param["windows"]["context_length"])

    static_categoricals          = ["participant_id", "clinical_site", "study_group"]
    static_reals                 = ["age"]
    time_varying_known_categoricals = ["sleep_stage"]
    time_varying_known_reals     = [
        "ds", "minute_of_day", "tod_sin", "tod_cos",
        "activity_steps", "calories_value", "heartrate",
        "oxygen_saturation", "respiration_rate", "stress_level", "predmeal_flag",
    ]
    time_varying_unknown_reals   = [
        "cgm_glucose",
        "cgm_lag_1", "cgm_lag_3", "cgm_lag_6",
        "cgm_diff_lag_1", "cgm_diff_lag_3", "cgm_diff_lag_6",
        "cgm_lagdiff_1_3", "cgm_lagdiff_3_6",
        "cgm_rolling_mean", "cgm_rolling_std",
    ]

    cut_off = train_df["ds"].max() - horizon

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
    )
    validation = training.from_dataset(training, train_df, predict=True, stop_randomization=True)

    dl = param.get("dataloader", {})
    bs  = int(dl.get("batch_size", 32))
    nw  = int(dl.get("num_workers", 4))
    pin = bool(dl.get("pin_memory", True))
    pw  = bool(dl.get("persistent_workers", True)) if nw > 0 else False

    train_dl = training.to_dataloader(train=True, batch_size=bs, num_workers=nw,
                                       persistent_workers=pw, pin_memory=pin)
    val_dl = _make_subsample_dl(validation, max_val_windows, bs, pin)
    return training, val_dl, train_dl, validation


# =============================
# Training
# =============================
def build_loss(param):
    cfg = param.get("loss", {})
    return QuantileLoss(quantiles=cfg.get("quantiles", [0.1, 0.5, 0.9]))


def TFT_train(train_df, param, out_dir: Path,
              limit_train_batches=None, save_all_checkpoints=False, max_val_windows=None):
    training, val_dl, train_dl, validation = create_tft_dataloaders(
        train_df, param, max_val_windows=max_val_windows
    )
    del train_df
    gc.collect()
    log_memory("DataLoaders ready")

    _bs          = int(param["dataloader"].get("batch_size", 32))
    _devs        = int(param["training"].get("devices", 4))
    _global_bs   = _bs * _devs
    _n_windows   = len(train_dl.dataset)
    _full_batches = math.ceil(_n_windows / _global_bs)
    _eff_batches  = min(_full_batches, limit_train_batches) if limit_train_batches else _full_batches
    _eff_frac     = _eff_batches / _full_batches * 100 if _full_batches else 100.0
    print(f"\n--- Training Diagnostics ---")
    print(f"  total_train_windows       : {_n_windows:,}")
    print(f"  global_batch_size         : {_global_bs}  ({_bs}/GPU × {_devs} GPUs)")
    print(f"  full_batches_per_epoch    : {_full_batches:,}")
    print(f"  effective_batches/epoch   : {_eff_batches:,}  (limit_train_batches={limit_train_batches})")
    print(f"  effective_epoch_fraction  : {_eff_frac:.1f}%")
    print(f"----------------------------\n")

    loss = build_loss(param)
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
        reduce_on_plateau_patience=2,  # TODO: factor=0.5, min_lr=1e-5 require pytorch_forecasting subclassing
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
    log_memory("Model created")

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="mamba288-{epoch:02d}-{val_loss:.2f}",
        save_top_k=-1 if save_all_checkpoints else 3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_on_train_epoch_end=True,
    )
    # Resume from last checkpoint if exists
    last_ckpt = ckpt_dir / "last.ckpt"
    ckpt_path = str(last_ckpt) if last_ckpt.exists() else None
    if ckpt_path:
        print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
    else:
        print("[INFO] No checkpoint found — starting fresh")

    train_cfg  = param["training"]
    csv_logger = CSVLogger(save_dir=str(out_dir), name="logs")
    _trainer_extra = {}
    if limit_train_batches is not None:
        _trainer_extra["limit_train_batches"] = limit_train_batches
    trainer = Trainer(
        max_epochs=train_cfg["epochs"],
        min_epochs=train_cfg.get("min_epochs", 3),
        gradient_clip_val=train_cfg["gradient_clip_val"],
        accelerator="gpu",
        devices=train_cfg["devices"],
        strategy=train_cfg.get("strategy", "auto"),
        val_check_interval=train_cfg.get("val_check_interval", 0.2),
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=train_cfg.get("early_stop_patience", 3),
                mode="min",
            ),
            LearningRateMonitor(logging_interval="epoch"),
            checkpoint_cb,
        ],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=csv_logger,
        **_trainer_extra,
    )

    log_memory("Starting training")
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt_path)
    log_memory("Training done")

    return tft, trainer, val_dl, train_dl, validation, training


# =============================
# Evaluation — RMSE, MAE, TIR
# =============================
def evaluate(tft, val_dl, out_dir: Path, ctx_h: int = 24):
    print("\n[Eval] Running predictions on validation set...")
    tft.eval()
    preds = tft.predict(
        val_dl, mode="prediction", return_x=True, return_y=True,
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

    print(f"\n{'='*40}")
    print(f"  RMSE : {rmse:.4f} mg/dL")
    print(f"  MAE  : {mae:.4f}  mg/dL")
    print(f"  TIR  : {tir:.2f}  %  (predicted in [70,180])")
    print(f"{'='*40}\n")

    n_windows, horizon = y_pred_arr.shape
    pd.DataFrame({
        "window_idx": np.repeat(np.arange(n_windows), horizon),
        "step":       np.tile(np.arange(horizon), n_windows),
        "y_true":     y_true,
        "y_pred":     y_pred,
    }).to_parquet(out_dir / "predictions.parquet", index=False)
    print(f"Per-window predictions saved → {out_dir / 'predictions.parquet'}")

    results = pd.DataFrame([{
        "model": f"SSM-CGM (mamba288, {ctx_h}h ctx, 1h horizon)",
        "context_h": ctx_h,
        "horizon_h": 1,
        "n_windows": n_windows,
        "rmse": rmse,
        "mae":  mae,
        "tir_70_180_pct": tir,
        "run_time": datetime.datetime.now().isoformat(),
    }])
    results_path = out_dir / "results_dynamic_only.csv"
    results.to_csv(results_path, index=False)
    print(f"Results saved → {results_path}")
    return results


# =============================
# Main
# =============================
def parse_args():
    p = argparse.ArgumentParser(description="SSM-CGM mamba288 — local training")
    p.add_argument("--train", type=Path,
                   default=Path("/home/myriamcharfeddine/CGM/Data/ssmcgm_ready/train_timeseries.feather"))
    p.add_argument("--test",  type=Path,
                   default=Path("/home/myriamcharfeddine/CGM/Data/ssmcgm_ready/test_timeseries.feather"))
    p.add_argument("--out",   type=Path,
                   default=Path("/home/myriamcharfeddine/CGM/Data/results"))
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load best checkpoint and evaluate only")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 200 stratified participants, 3 epochs — verifies convergence quickly")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override max_epochs (default: 30, smoke: 3)")
    p.add_argument("--min-epochs", type=int, default=None,
                   help="Override Lightning Trainer min_epochs (default: 3)")
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="Override EarlyStopping patience on val_loss (default: 3)")
    p.add_argument("--n-participants", type=int, default=None,
                   help="Subsample N participants (stratified by study_group)")
    p.add_argument("--context-length", type=int, default=288,
                   help="Encoder context length in bins (288=24h, 576=48h). Default: 288")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size per GPU (default: 32)")
    p.add_argument("--limit-train-batches", type=int, default=None,
                   help="Cap training batches per epoch per GPU (e.g. 20000)")
    p.add_argument("--max-val-windows", type=int, default=None,
                   help="Cap validation windows via random subsampling")
    p.add_argument("--save-all-epoch-checkpoints", action="store_true",
                   help="Save checkpoint every epoch (save_top_k=-1)")
    # DataLoader shared-memory flags — use these when workers crash with bus error / SIGBUS.
    # Root cause: multiple workers + pin_memory=True exhaust /dev/shm inside the container,
    # killing one worker; NCCL watchdog errors are secondary, not the root cause.
    p.add_argument("--num-workers", type=int, default=None,
                   help="Override DataLoader num_workers per GPU process (default: 8)")
    p.add_argument("--no-persistent-workers", action="store_true", dest="no_persistent_workers",
                   help="Disable persistent DataLoader workers (reduces /dev/shm pressure)")
    p.add_argument("--no-pin-memory", action="store_true", dest="no_pin_memory",
                   help="Disable DataLoader pin_memory (reduces shared-memory bus load)")
    return p.parse_args()


def subsample_participants(df, n, seed=42):
    """Stratified subsample of n participants across all study_group values."""
    pids = df[["participant_id", "study_group"]].drop_duplicates()
    groups = pids["study_group"].unique()
    per_group = max(1, n // len(groups))
    sampled = []
    for g in groups:
        pool = pids[pids["study_group"] == g]["participant_id"].tolist()
        k = min(per_group, len(pool))
        rng = np.random.default_rng(seed)
        sampled.extend(rng.choice(pool, size=k, replace=False).tolist())
    # top up to exactly n if rounding left us short
    all_pids = pids["participant_id"].tolist()
    rng = np.random.default_rng(seed + 1)
    remaining = [p for p in rng.permutation(all_pids) if p not in set(sampled)]
    sampled += remaining[: n - len(sampled)]
    return df[df["participant_id"].isin(sampled[:n])].copy()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available — GPU required for Mamba"
    name = torch.cuda.get_device_name(0)
    cap  = torch.cuda.get_device_capability(0)
    print(f"[GPU] {name}  CC={cap}  BF16={'OK' if torch.cuda.is_bf16_supported() else 'NO'}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark        = True

    args = parse_args()

    # ── Apply smoke-test overrides ────────────────────────────────────────
    param = dict(param_24h)   # shallow copy; mutate training sub-dict
    param["training"] = dict(param_24h["training"])
    param["windows"]  = dict(param_24h["windows"])

    # Apply context length override
    param["windows"]["context_length"] = args.context_length
    ctx_h = args.context_length // 12   # 5-min bins per hour

    param["dataloader"] = dict(param_24h["dataloader"])

    if args.batch_size is not None:
        param["dataloader"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        param["dataloader"]["num_workers"] = args.num_workers
    if args.no_persistent_workers:
        param["dataloader"]["persistent_workers"] = False
    if args.no_pin_memory:
        param["dataloader"]["pin_memory"] = False

    if args.smoke:
        print("\n[SMOKE TEST] 200 participants · 3 epochs · val every epoch")
        args.n_participants = args.n_participants or 200
        param["training"]["epochs"] = 3
        param["training"]["val_check_interval"] = 1.0   # once per epoch
        args.out = args.out / "smoke"

    if args.epochs is not None:
        param["training"]["epochs"] = args.epochs
    if args.min_epochs is not None:
        param["training"]["min_epochs"] = args.min_epochs
    if args.early_stop_patience is not None:
        param["training"]["early_stop_patience"] = args.early_stop_patience

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*56}")
    print(f"  Experiment A — SSM-CGM Dynamic-only (Mamba288)")
    print(f"{'='*56}")
    print(f"  context_length     : {param['windows']['context_length']} bins = {ctx_h}h")
    print(f"  horizon            : {param['windows']['horizon']} bins = 1h")
    print(f"  batch_size/GPU     : {param['dataloader']['batch_size']}")
    print(f"  num_workers        : {param['dataloader']['num_workers']}")
    print(f"  persistent_workers : {param['dataloader']['persistent_workers']}")
    print(f"  pin_memory         : {param['dataloader']['pin_memory']}")
    print(f"  devices            : {param['training']['devices']}")
    print(f"  max_epochs         : {param['training']['epochs']}")
    print(f"  min_epochs         : {param['training']['min_epochs']}")
    print(f"  early_stop_patience: {param['training']['early_stop_patience']}")
    print(f"  reduce_lr_patience : 2")
    print(f"  limit_train_batches: {args.limit_train_batches}")
    print(f"  max_val_windows    : {args.max_val_windows}")
    print(f"  output             : {args.out}")
    print(f"{'='*56}\n")

    print(f"\nLoading train feather: {args.train}")
    train_df = pd.read_feather(args.train)

    if args.n_participants:
        train_df = subsample_participants(train_df, args.n_participants)
        print(f"  Subsampled to {train_df['participant_id'].nunique()} participants "
              f"({train_df['study_group'].value_counts().to_dict()})")
    else:
        print(f"  {train_df['participant_id'].nunique()} participants | {len(train_df):,} rows")

    if args.eval_only:
        ckpt_dir = args.out / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("mamba288-*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        best_ckpt = ckpts[-1]
        print(f"Loading checkpoint: {best_ckpt}")
        _, val_dl, _, _ = create_tft_dataloaders(train_df, param, max_val_windows=args.max_val_windows)
        tft = MambaTFT.load_from_checkpoint(str(best_ckpt), weights_only=False)
        evaluate(tft, val_dl, args.out, ctx_h=ctx_h)
    else:
        tft, trainer, val_dl, train_dl, validation, training = TFT_train(
            train_df, param, args.out,
            limit_train_batches=args.limit_train_batches,
            save_all_checkpoints=args.save_all_epoch_checkpoints,
            max_val_windows=args.max_val_windows,
        )
        evaluate(tft, val_dl, args.out, ctx_h=ctx_h)
