import os
import io
import gc
import glob
import datetime
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn

from google.cloud import storage

# Pytorch Forecasting
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

# Lightning
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

# ------------------------------------------------------------
# GPU runtime tweaks (A100-friendly)
# ------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

assert torch.cuda.is_available(), "CUDA non dispo"
print(
    f"[GPU] {torch.cuda.get_device_name(0)} "
    f"CC={torch.cuda.get_device_capability(0)}, "
    f"BF16={'OK' if torch.cuda.is_bf16_supported() else 'NO'}"
)

# ------------------------------------------------------------
# GCS constants
# ------------------------------------------------------------
_BUCKET_NAME = "cgmproject2025"
_DATA_BLOB   = "ai-ready/data/train_finaltimeseries_meal.feather"  # adapte ici si besoin
_BASE_PREFIX = "models/predictions"
_gcs_client  = storage.Client()

# ------------------------------------------------------------
# TimeXer imports (compat PF v1/v2)
# ------------------------------------------------------------
try:
    # PF v2
    from pytorch_forecasting.models.timexer._timexer_v2 import TimeXer as _TimeXer
except Exception:
    # PF v1 (stable)
    from pytorch_forecasting.models.timexer._timexer import TimeXer as _TimeXer


# ------------------------------------------------------------
# Presets 24h / 48h (minutes)
# ------------------------------------------------------------
timexer_param_24 = {
    "dataset": {"context_length": 288, "horizon": 12, "batch_size": 32},
    "model": {
        "hidden_size": 128,   # dimension modèle
        "n_heads": 8,         # nb têtes (si applicable dans ta version)
        "e_layers": 3,        # nb couches encodeur
        "d_ff": 4 * 128,      # feed-forward
        "dropout": 0.20,
        "patch_length": 16,   # tokenisation patch
        "factor": 5,          # scaling attention (si applicable)
        "activation": "relu",
        "freq": "t",          # 't' minutes ; 'h' heures ; 'd' jours...
    },
    "optim": {"lr": 1e-3},
    "trainer": {
        "max_epochs": 30,
        "gradient_clip_val": 1.0,
        "limit_train_batches": 1.0,  # 1.0 pour tout; ex: 100 pour debug
        "val_check_interval": 0.2,
    },
    "checkpoint": True,
}

# Choix par défaut (change ici si tu veux 48h)
timexer_param = timexer_param_24


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def log_memory(message: str = ""):
    if torch.cuda.is_available():
        mem_gpu_allocated = torch.cuda.memory_allocated() / 1e9
        mem_gpu_reserved = torch.cuda.memory_reserved() / 1e9
    else:
        mem_gpu_allocated = mem_gpu_reserved = 0.0
    print(f"[{datetime.datetime.now()}] {message}")
    print(f"GPU Mem allocated: {mem_gpu_allocated:.2f} GB | reserved: {mem_gpu_reserved:.2f} GB")


def upload_latest_checkpoint_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str):
    bucket = _gcs_client.bucket(bucket_name)
    for file_path in glob.glob(f"{local_dir}/*.ckpt"):
        blob = bucket.blob(f"{gcs_prefix}/{os.path.basename(file_path)}")
        blob.upload_from_filename(file_path)
        print(f"Uploaded {file_path} to gs://{bucket_name}/{gcs_prefix}/")


def download_last_ckpt(local_dir: str, bucket_name: str, gcs_prefix: str):
    os.makedirs(local_dir, exist_ok=True)
    bucket = _gcs_client.bucket(bucket_name)
    blob = bucket.blob(f"{gcs_prefix}/last.ckpt")  # si "save_last=True"
    if blob.exists():
        local_path = os.path.join(local_dir, "last.ckpt")
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")
        return local_path
    # sinon, prend le .ckpt le plus récent (top_k)
    blobs = [b for b in _gcs_client.list_blobs(bucket_name, prefix=gcs_prefix) if b.name.endswith(".ckpt")]
    if blobs:
        latest = max(blobs, key=lambda b: b.updated)
        local_path = os.path.join(local_dir, os.path.basename(latest.name))
        latest.download_to_filename(local_path)
        print(f"Downloaded {latest.name} to {local_path}")
        return local_path
    return None


# ------------------------------------------------------------
# Callbacks (GCS logging)
# ------------------------------------------------------------
class FirstEpochTimer(pl.Callback):
    def __init__(self, local_path, gcs_bucket, gcs_prefix):
        super().__init__()
        self.local_path = local_path
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.start_time = None
        self.logged = False

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = datetime.datetime.now()

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.logged:
            end_time = datetime.datetime.now()
            elapsed = (end_time - self.start_time).total_seconds()
            with open(self.local_path, "w") as f:
                f.write(f"first_epoch_runtime_seconds: {elapsed}\n")
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
        loss = None
        if isinstance(outputs, dict) and "loss" in outputs:
            try:
                loss = float(outputs["loss"])
            except Exception:
                pass
        if loss is None:
            try:
                loss = float(outputs)
            except Exception:
                pass
        if (self.batch_count % self.log_every_n_batches == 0) and (loss is not None):
            with open(self.log_path, "a") as f:
                f.write(f"Batch {self.batch_count}: loss={loss}\n")
            bucket = _gcs_client.bucket(self.bucket_name)
            blob = bucket.blob(f"{self.gcs_prefix}/loss_log.txt")
            blob.upload_from_filename(self.log_path)
            print(f"Uploaded loss log to gs://{self.bucket_name}/{self.gcs_prefix}/loss_log.txt")


# ------------------------------------------------------------
# Data: PF dataset + dataloaders (mêmes features que ton script MambaTFT)
# ------------------------------------------------------------
def create_tft_dataloaders(train_df: pd.DataFrame, horizon=12, context_length=72, batchsize=32):
    log_memory("🚀 Start of Dataloader Creation")

    static_categoricals = ["participant_id", "clinical_site", "study_group"]
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

    cut_off = train_df["ds"].max() - horizon  # horizon en unités du time_idx "ds"

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
        persistent_workers=False,
        num_workers=0,
        pin_memory=False,
    )

    return training, val_loader, train_loader, validation


def _get_max_lengths(ds) -> tuple[int, int]:
    enc = getattr(ds, "max_encoder_length", None)
    pred = getattr(ds, "max_prediction_length", None)
    if (enc is None or pred is None) and hasattr(ds, "hparams"):
        enc = enc or ds.hparams.get("max_encoder_length", None)
        pred = pred or ds.hparams.get("max_prediction_length", None)
    return int(enc), int(pred)


# ------------------------------------------------------------
# Train TimeXer
# ------------------------------------------------------------
def TimeXer_train(train_df: pd.DataFrame, param: dict):
    ds_cfg = param["dataset"]
    m_cfg  = param["model"]
    tr_cfg = param["trainer"]
    lr     = float(param["optim"]["lr"])

    horizon        = int(ds_cfg["horizon"])
    context_len    = int(ds_cfg["context_length"])
    batchsize      = int(ds_cfg["batch_size"])

    log_memory("🚀 Début du chargement des dataloaders (TimeXer)")
    training, val_loader, train_loader, validation = create_tft_dataloaders(
        train_df, horizon=horizon, context_length=context_len, batchsize=batchsize
    )

    del train_df
    gc.collect()
    log_memory("📦 Dataloaders prêts (TimeXer)")

    loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

    enc_len, pred_len = _get_max_lengths(training)

    # Instanciation du modèle TimeXer
    timexer = _TimeXer.from_dataset(
        training,
        # séquences
        context_length=enc_len,
        prediction_length=pred_len,
        # archi
        hidden_size=m_cfg["hidden_size"],
        n_heads=m_cfg.get("n_heads", 8),
        e_layers=m_cfg.get("e_layers", 3),
        d_ff=m_cfg.get("d_ff", 4 * m_cfg["hidden_size"]),
        dropout=m_cfg.get("dropout", 0.1),
        patch_length=m_cfg.get("patch_length", 16),
        factor=m_cfg.get("factor", 5),
        activation=m_cfg.get("activation", "relu"),
        freq=m_cfg.get("freq", "t"),
        # optimisation
        learning_rate=lr,
        loss=loss,
        # logs PF par défaut (SMAPE/MAE/RMSE/MAPE)
    )

    log_memory("🧠 Modèle TimeXer instancié")

    # Callbacks + checkpoint + GCS
    ckpt_prefix = "checkpoints_timexer288"

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="timexer288-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_on_train_epoch_end=True,
    )
    first_epoch_timer = FirstEpochTimer(
        local_path="first_epoch_runtime.txt",
        gcs_bucket=_BUCKET_NAME,
        gcs_prefix=ckpt_prefix,
    )
    gcs_checkpoint_uploader = GCSCheckpointUploader(
        local_dir="checkpoints",
        bucket_name=_BUCKET_NAME,
        gcs_prefix=ckpt_prefix,
    )
    gcs_loss_logger = GCSLossLogger(
        log_path="loss_log.txt",
        bucket_name=_BUCKET_NAME,
        gcs_prefix=ckpt_prefix,
        log_every_n_batches=5000,
    )
    # --- Download latest checkpoint from GCS if exists ---
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = download_last_ckpt("checkpoints", _BUCKET_NAME, ckpt_prefix)
    if ckpt_path:
        print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
    else:
        print("[INFO] No checkpoint found: starting fresh")

    trainer = Trainer(
        max_epochs=tr_cfg["max_epochs"],
        gradient_clip_val=tr_cfg["gradient_clip_val"],
        limit_train_batches=tr_cfg["limit_train_batches"],
        val_check_interval=tr_cfg["val_check_interval"],
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15, mode="min"),
            checkpoint_cb,
            first_epoch_timer,
            gcs_checkpoint_uploader,
            gcs_loss_logger,
            # LearningRateMonitor(logging_interval="epoch"),  # active si besoin
        ],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
    )

    log_memory("▶️ Entraînement TimeXer - avant fit")
    trainer.fit(timexer, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    log_memory("✅ Fin entraînement TimeXer")

    return timexer, trainer, val_loader, train_loader, validation, training


# ------------------------------------------------------------
# Save/Load model (GCS)
# ------------------------------------------------------------
def save_timexer_to_gcs(
    model: _TimeXer,
    model_name: str = "TimeXer_12h_288c",
    bucket_name: str = _BUCKET_NAME,
    prefix: str = _BASE_PREFIX,
):
    """Sauvegarde state_dict + hparams (kwargs from_dataset) sur GCS."""
    bucket = _gcs_client.bucket(bucket_name)
    model_prefix = f"{prefix}/{model_name}"

    # state_dict
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    blob = bucket.blob(f"{model_prefix}/model.pth")
    blob.upload_from_file(buf)
    buf.close()

    # hparams (kwargs utilisés par from_dataset)
    buf = io.BytesIO()
    pickle.dump(model.hparams, buf)
    buf.seek(0)
    blob = bucket.blob(f"{model_prefix}/model_kwargs.pkl")
    blob.upload_from_file(buf)
    buf.close()

    print(f"Model artifacts uploaded to gs://{bucket_name}/{model_prefix}/")


def load_timexer_from_gcs(
    training_dataset: TimeSeriesDataSet,
    model_name: str = "TimeXer_12h_288c",
    map_location=None,
    bucket_name: str = _BUCKET_NAME,
    prefix: str = _BASE_PREFIX,
) -> _TimeXer:
    """Recrée le modèle via from_dataset + charge les poids depuis GCS."""
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bucket = _gcs_client.bucket(bucket_name)
    model_prefix = f"{prefix}/{model_name}"

    # hparams/kwargs
    blob = bucket.blob(f"{model_prefix}/model_kwargs.pkl")
    hparams_bytes = blob.download_as_bytes()
    model_kwargs = pickle.loads(hparams_bytes)

    model = _TimeXer.from_dataset(training_dataset, **model_kwargs)

    # state_dict
    blob = bucket.blob(f"{model_prefix}/model.pth")
    state_bytes = blob.download_as_bytes()
    buf = io.BytesIO(state_bytes)
    state_dict = torch.load(buf, map_location=map_location)
    buf.close()

    model.load_state_dict(state_dict)
    print(f"Loaded TimeXer from gs://{bucket_name}/{model_prefix}/")
    return model


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # Rank/devices info (utile sous DDP/SLURM)
    rank = os.environ.get("LOCAL_RANK", "NotSet")
    gpu_count = torch.cuda.device_count()
    print(f"[INFO] RANK={rank} | CUDA devices={gpu_count}")

    # Download dataset from GCS
    client = storage.Client()
    bucket = client.bucket(_BUCKET_NAME)
    blob = bucket.blob(_DATA_BLOB)
    data_bytes = blob.download_as_bytes()
    train_df = pd.read_feather(io.BytesIO(data_bytes))

    # Train
    timexer, trainer, val_loader, train_loader, validation, training = TimeXer_train(train_df, timexer_param)

    # Save artifacts to GCS
    save_timexer_to_gcs(timexer, model_name="TimeXer_12h_288c")
