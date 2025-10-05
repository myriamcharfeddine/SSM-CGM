#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import io
import gc
import glob
import pickle
import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn

# ⚡ Lightning
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

# 📦 GCS
from google.cloud import storage

# 🔮 PyTorch Forecasting
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.utils import move_to_device

# RecurrentNetwork import path varies across PF versions
try:
    from pytorch_forecasting.models import RecurrentNetwork
except Exception:
    from pytorch_forecasting.models.rnn import RecurrentNetwork  # PF ≥1.1 fallback

# Metrics (PF has MSE in newer versions)
try:
    from pytorch_forecasting.metrics import MSE as PF_MSE
    _HAS_PF_MSE = True
except Exception:
    _HAS_PF_MSE = False
    from pytorch_forecasting.metrics import RMSE  # always available

# =============================
# Globals & Runtime setup
# =============================
BUCKET_NAME = "cgmproject2025"
DATA_OBJECT = 'ai-ready/data/train_finaltimeseries_meal.feather'
GCS_CHECKPOINT_PREFIX = "checkpoints_lstm_baselineOPT"
LOCAL_CKPT_DIR = "checkpoints"
FIRST_EPOCH_METRICS_FILE = "first_epoch_runtime.txt"
LOSS_LOG_FILE = "loss_log.txt"

# A100 / Ampere: faster matmuls
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

assert torch.cuda.is_available(), "CUDA non dispo"

GPU_NAME = torch.cuda.get_device_name(0)
GPU_CAP = torch.cuda.get_device_capability(0)
print(f"[GPU] {GPU_NAME}  CC={GPU_CAP}, BF16={'OK' if torch.cuda.is_bf16_supported() else 'NO'}")

# =============================
# Utilities
# =============================
_gcs_client = storage.Client()


def log_memory(message: str = "") -> None:
    """Log GPU memory (simple, psutil-free)."""
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
    else:
        mem_alloc = mem_reserved = 0.0
    print(f"[{dt.datetime.now()}] {message}")
    print(f"GPU Mem allocated: {mem_alloc:.2f} GB | reserved: {mem_reserved:.2f} GB")


def upload_latest_checkpoint_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str) -> None:
    """Upload all .ckpt files in local_dir to GCS under gcs_prefix."""
    bucket = _gcs_client.bucket(bucket_name)
    for file_path in glob.glob(os.path.join(local_dir, "*.ckpt")):
        fname = os.path.basename(file_path)
        blob = bucket.blob(f"{gcs_prefix}/{fname}")
        blob.upload_from_filename(file_path)
        print(f"Uploaded {file_path} -> gs://{bucket_name}/{gcs_prefix}/{fname}")


def download_last_ckpt(local_dir: str, bucket_name: str, gcs_prefix: str) -> str | None:
    os.makedirs(local_dir, exist_ok=True)
    bucket = _gcs_client.bucket(bucket_name)
    blob = bucket.blob(f"{gcs_prefix}/last.ckpt")
    if blob.exists():
        local_path = os.path.join(local_dir, "last.ckpt")
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} -> {local_path}")
        return local_path
    return None


class FirstEpochTimer(pl.Callback):
    """Measure wall time of the first epoch and push to GCS once."""

    def __init__(self, local_path: str, gcs_bucket: str, gcs_prefix: str):
        super().__init__()
        self.local_path = local_path
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self._start: dt.datetime | None = None
        self._uploaded = False

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch == 0:
            self._start = dt.datetime.now()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch == 0 and not self._uploaded and self._start is not None:
            elapsed = (dt.datetime.now() - self._start).total_seconds()
            with open(self.local_path, "w") as f:
                f.write(f"first_epoch_runtime_seconds: {elapsed}\n")
            bucket = _gcs_client.bucket(self.gcs_bucket)
            blob = bucket.blob(f"{self.gcs_prefix}/first_epoch_runtime.txt")
            blob.upload_from_filename(self.local_path)
            print(f"Uploaded first epoch runtime -> gs://{self.gcs_bucket}/{self.gcs_prefix}/first_epoch_runtime.txt")
            self._uploaded = True


class GCSCheckpointUploader(pl.Callback):
    def __init__(self, local_dir: str, bucket_name: str, gcs_prefix: str):
        super().__init__()
        self.local_dir = local_dir
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        upload_latest_checkpoint_to_gcs(self.local_dir, self.bucket_name, self.gcs_prefix)


class GCSLossLogger(pl.Callback):
    def __init__(self, log_path: str, bucket_name: str, gcs_prefix: str, log_every_n_batches: int = 5000):
        super().__init__()
        self.log_path = log_path
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
        self.log_every_n_batches = log_every_n_batches
        self.batch_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.batch_count += 1
        loss_val = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_val = float(outputs["loss"])  # type: ignore[arg-type]
        elif hasattr(outputs, "item"):
            try:
                loss_val = float(outputs.item())
            except Exception:
                pass
        if loss_val is not None and self.batch_count % self.log_every_n_batches == 0:
            with open(self.log_path, "a") as f:
                f.write(f"Batch {self.batch_count}: loss={loss_val}\n")
            bucket = _gcs_client.bucket(self.bucket_name)
            blob = bucket.blob(f"{self.gcs_prefix}/loss_log.txt")
            blob.upload_from_filename(self.log_path)
            print(f"Uploaded loss log -> gs://{self.bucket_name}/{self.gcs_prefix}/loss_log.txt")


# =============================
# Data
# =============================

def create_tft_dataloaders(
    train_df: pd.DataFrame,
    horizon: int = 12,
    context_length: int = 72,
    batchsize: int = 32,
) -> Tuple[TimeSeriesDataSet, torch.utils.data.DataLoader, torch.utils.data.DataLoader, TimeSeriesDataSet]:
    """Build PF datasets + dataloaders.
    Uses the same feature config you validated in notebook.
    """
    log_memory("🚀 Start of Dataloader Creation")

    # --- Static & dynamic features ---
    static_categoricals = [
        "participant_id",
        "clinical_site",
        "study_group",
    ]
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
    ]
    time_varying_unknown_reals = [
        "cgm_glucose",
    ]

    # --- Temporal split ---
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

    validation = training.from_dataset(
        training, train_df, predict=True, stop_randomization=True
    )

    train_loader = training.to_dataloader(
        train=True,
        batch_size=batchsize,
        persistent_workers=False,
        num_workers=1,
        pin_memory=False,
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=batchsize,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    return training, val_loader, train_loader, validation


# =============================
# Model & Training
# =============================
LSTM_HP = {
    "hidden_size": 128,
    "rnn_layers": 2,
    "dropout": 0.2,
    "cell_type": "LSTM",  # or "GRU"
}

PARAM = {"dataset": {"context_length": 96, "horizon": 12, "batch_size": 32}}


def train_lstm(train_df: pd.DataFrame, param: dict) -> tuple:
    """Train PyTorch Forecasting RecurrentNetwork (LSTM) and return artifacts.

    Returns: (model, trainer, val_loader, train_loader, validation_ds, training_ds)
    """
    ds_cfg = param["dataset"]
    horizon = int(ds_cfg["horizon"])
    context_length = int(ds_cfg["context_length"])
    batchsize = int(ds_cfg["batch_size"])

    seed_everything(42, workers=True)

    log_memory("🚀 Début du chargement des dataloaders (LSTM)")
    training, val_loader, train_loader, validation = create_tft_dataloaders(
        train_df, horizon=horizon, context_length=context_length, batchsize=batchsize
    )

    # free heavy DF memory if needed
    del train_df
    gc.collect()
    log_memory("✅ Dataloaders prêts (LSTM)")

    # --- loss ---
    if _HAS_PF_MSE:
        loss = PF_MSE()
    else:
        loss = RMSE()

    # --- model ---
    lstm = RecurrentNetwork.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=LSTM_HP["hidden_size"],
        rnn_layers=LSTM_HP["rnn_layers"],
        dropout=LSTM_HP["dropout"],
        cell_type=LSTM_HP["cell_type"],
        loss=loss,
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=4,
    )

    log_memory("🧠 Modèle LSTM instancié")

    # --- callbacks ---
    os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=LOCAL_CKPT_DIR,
        filename="lstm_{epoch:02d}-{val_loss:.3f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_on_train_epoch_end=True,
    )
    first_epoch_timer = FirstEpochTimer(
        local_path=FIRST_EPOCH_METRICS_FILE,
        gcs_bucket=BUCKET_NAME,
        gcs_prefix=GCS_CHECKPOINT_PREFIX,
    )
    gcs_uploader = GCSCheckpointUploader(
        local_dir=LOCAL_CKPT_DIR,
        bucket_name=BUCKET_NAME,
        gcs_prefix=GCS_CHECKPOINT_PREFIX,
    )
    gcs_loss_logger = GCSLossLogger(
        log_path=LOSS_LOG_FILE,
        bucket_name=BUCKET_NAME,
        gcs_prefix=GCS_CHECKPOINT_PREFIX,
        log_every_n_batches=5000,
    )

    # --- resume if last.ckpt exists on GCS ---
    ckpt_path = download_last_ckpt(LOCAL_CKPT_DIR, BUCKET_NAME, GCS_CHECKPOINT_PREFIX)
    if ckpt_path:
        print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
    else:
        print("[INFO] No checkpoint found on GCS: starting fresh")

    # --- trainer ---
    trainer = Trainer(
        max_epochs=30,                 # match your quick bench; adjust freely
        gradient_clip_val=1,
        val_check_interval=0.2,       # idem
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15, mode="min"),
            ckpt_cb,
            first_epoch_timer,
            gcs_uploader,
            gcs_loss_logger,
        ],
        enable_progress_bar=True,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        enable_model_summary=False,
        logger=False,
    )

    log_memory("▶️ Début entraînement LSTM")
    trainer.fit(lstm, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    log_memory("🏁 Fin entraînement LSTM")

    return lstm, trainer, val_loader, train_loader, validation, training


# =============================
# Save/Load to GCS (optional convenience)
# =============================

def save_lstm_to_gcs(
    model: RecurrentNetwork,
    training_ds: TimeSeriesDataSet,
    model_name: str = "LSTM_baseline",
    bucket_name: str = BUCKET_NAME,
    prefix: str = "models/predictions",
) -> None:
    """Save state_dict and the dataset hparams needed to rebuild the model."""
    bucket = _gcs_client.bucket(bucket_name)
    model_prefix = f"{prefix}/{model_name}"

    # Save state_dict
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    bucket.blob(f"{model_prefix}/model.pth").upload_from_file(buf)
    buf.close()

    # Minimal kwargs to rebuild from dataset
    model_kwargs = dict(
        learning_rate=model.hparams.get("learning_rate", 0.003),
        hidden_size=LSTM_HP["hidden_size"],
        rnn_layers=LSTM_HP["rnn_layers"],
        dropout=LSTM_HP["dropout"],
        cell_type=LSTM_HP["cell_type"],
        loss=model.hparams.get("loss", None),
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=4,
    )
    buf = io.BytesIO()
    pickle.dump(model_kwargs, buf)
    buf.seek(0)
    bucket.blob(f"{model_prefix}/model_kwargs.pkl").upload_from_file(buf)
    buf.close()

    # Save dataset definition for reconstruction via from_dataset
    buf = io.BytesIO()
    pickle.dump(training_ds.get_parameters(), buf)
    buf.seek(0)
    bucket.blob(f"{model_prefix}/dataset_params.pkl").upload_from_file(buf)
    buf.close()

    print(f"Model artifacts uploaded to gs://{bucket_name}/{model_prefix}/")


# =============================
# Main
# =============================
if __name__ == "__main__":
    rank = os.environ.get("LOCAL_RANK", "NotSet")
    gpu_count = torch.cuda.device_count()
    print(f"[INFO] RANK={rank} | CUDA devices={gpu_count}")

    # Load dataset from GCS
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(DATA_OBJECT)
    data_bytes = blob.download_as_bytes()
    train_df = pd.read_feather(io.BytesIO(data_bytes))

    # Train
    model, trainer, val_loader, train_loader, validation_ds, training_ds = train_lstm(train_df, PARAM)

    # Optional: upload latest local ckpts to GCS one more time
    upload_latest_checkpoint_to_gcs(LOCAL_CKPT_DIR, BUCKET_NAME, GCS_CHECKPOINT_PREFIX)

    # Optional: persist a light-weight export for inference
    save_lstm_to_gcs(model, training_ds, model_name="LSTM_12h_OPT")
