# -*- coding: utf-8 -*-

import os
import io
import gc
import glob
import pickle
import argparse
import datetime
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn

# PyTorch Forecasting
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models.deepar import DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss

# Lightning
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

# GCS
from google.cloud import storage


# =============================
# Defaults & Presets
# =============================

BUCKET_DEFAULT = "cgmproject2025"
BLOB_DEFAULT = 'ai-ready/data/train_finaltimeseries_meal.feather'
GCS_MODEL_PREFIX = "models/predictions"
GCS_CKPT_PREFIX = "checkpoints_deepar576opt"

param_deepar_48 = {
    "dataset": {"context_length": 576, "horizon": 12, "batch_size": 32},
    "deepar_init": {"hidden_size": 256, "rnn_layers": 3, "dropout": 0.1},
}

# =============================
# Torch runtime knobs (A100 friendly)
# =============================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


# =============================
# Utils
# =============================

def log_gpu(message: str = ""):
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_res = torch.cuda.memory_reserved() / 1e9
        print(f"[{datetime.datetime.now()}] {message}")
        print(f"[GPU] {name} CC={cap} | allocated={mem_alloc:.2f} GB reserved={mem_res:.2f} GB")
    else:
        print(f"[{datetime.datetime.now()}] {message} (CPU mode)")

def assert_cuda():
    assert torch.cuda.is_available(), "CUDA non dispo — lance sur une machine avec GPU/CUDA."

def gcs_client() -> storage.Client:
    return storage.Client()

def upload_all_ckpts(local_dir: str, bucket_name: str, gcs_prefix: str):
    client = gcs_client()
    bucket = client.bucket(bucket_name)
    for path in glob.glob(os.path.join(local_dir, "*.ckpt")):
        blob = bucket.blob(f"{gcs_prefix}/{os.path.basename(path)}")
        blob.upload_from_filename(path)
        print(f"Uploaded {path} -> gs://{bucket_name}/{gcs_prefix}/")

def download_last_ckpt(local_dir: str, bucket_name: str, gcs_prefix: str) -> Optional[str]:
    os.makedirs(local_dir, exist_ok=True)
    client = gcs_client()
    bucket = client.bucket(bucket_name)
    # Try "last.ckpt"
    last_blob = bucket.blob(f"{gcs_prefix}/last.ckpt")
    if last_blob.exists():
        local_path = os.path.join(local_dir, "last.ckpt")
        last_blob.download_to_filename(local_path)
        print(f"Downloaded {last_blob.name} to {local_path}")
        return local_path
    # Otherwise, take most recent *.ckpt
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    ckpts = [b for b in blobs if b.name.endswith(".ckpt")]
    if not ckpts:
        return None
    latest = max(ckpts, key=lambda b: b.updated)
    local_path = os.path.join(local_dir, os.path.basename(latest.name))
    latest.download_to_filename(local_path)
    print(f"Downloaded {latest.name} to {local_path}")
    return local_path

class GCSCheckpointUploader(pl.Callback):
    def __init__(self, local_dir: str, bucket_name: str, gcs_prefix: str):
        super().__init__()
        self.local_dir = local_dir
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        upload_all_ckpts(self.local_dir, self.bucket_name, self.gcs_prefix)


def save_model_to_gcs(
    model: DeepAR,
    model_name: str,
    training_dataset: TimeSeriesDataSet,
    bucket_name: str = BUCKET_DEFAULT,
    prefix: str = GCS_MODEL_PREFIX,
):
    """
    Sauvegarde (state_dict + hparams) sur GCS pour rechargement ultérieur.
    On sérialise aussi les kwargs nécessaires via training_dataset.get_parameters().
    """
    client = gcs_client()
    bucket = client.bucket(bucket_name)
    model_prefix = f"{prefix}/{model_name}"

    # 1) hparams de DeepAR.from_dataset (kwargs)
    model_kwargs = model.hparams  # dict-like
    # Pour reproduire le dataset au chargement :
    dataset_kwargs = training_dataset.get_parameters()

    # 2) upload
    # state_dict
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    bucket.blob(f"{model_prefix}/model.pth").upload_from_file(buf)
    buf.close()

    # hparams
    buf = io.BytesIO()
    pickle.dump({"model_kwargs": model_kwargs, "dataset_kwargs": dataset_kwargs}, buf)
    buf.seek(0)
    bucket.blob(f"{model_prefix}/artifacts.pkl").upload_from_file(buf)
    buf.close()

    print(f"Model artifacts uploaded to gs://{bucket_name}/{model_prefix}/")


def load_model_from_gcs(
    bucket_name: str,
    model_name: str,
    map_location: Optional[torch.device] = None,
    prefix: str = GCS_MODEL_PREFIX,
) -> Tuple[DeepAR, TimeSeriesDataSet]:
    """
    Recharge (dataset -> DeepAR.from_dataset -> state_dict) depuis GCS.
    """
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client = gcs_client()
    bucket = client.bucket(bucket_name)
    model_prefix = f"{prefix}/{model_name}"

    # Récup hparams/dataset params
    art_bytes = bucket.blob(f"{model_prefix}/artifacts.pkl").download_as_bytes()
    art = pickle.loads(art_bytes)
    model_kwargs = art["model_kwargs"]
    dataset_kwargs = art["dataset_kwargs"]

    # Reconstruit le dataset "à vide" (utile pour from_dataset)
    training_dataset = TimeSeriesDataSet.from_parameters(dataset_kwargs, data=None)

    # Construit le modèle
    model = DeepAR.from_dataset(training_dataset, **model_kwargs)

    # Charge state_dict
    state_bytes = bucket.blob(f"{model_prefix}/model.pth").download_as_bytes()
    buf = io.BytesIO(state_bytes)
    state_dict = torch.load(buf, map_location=map_location)
    model.load_state_dict(state_dict)
    buf.close()

    print(f"Loaded model from gs://{bucket_name}/{model_prefix}/")
    return model, training_dataset


# =============================
# Data & Dataloaders
# =============================

def _filter_present(cols: List[str], df: pd.DataFrame) -> List[str]:
    return [c for c in cols if c in df.columns]

def create_deepar_dataloaders(
    train_df: pd.DataFrame,
    horizon: int = 12,
    context_length: int = 72,
    batchsize: int = 32,
) -> Tuple[TimeSeriesDataSet, torch.utils.data.DataLoader, torch.utils.data.DataLoader, TimeSeriesDataSet]:
    """
    Même logique que ton notebook, avec robustesse :
    - on garde 'ds' comme time index s'il est déjà numérique,
      sinon on crée un 'time_idx' entier par participant.
    - on filtre les colonnes optionnelles si absentes.
    """
    df = train_df.copy()

    # Features (filtrées si manquantes)
    static_categoricals = _filter_present(
        ["participant_id", "clinical_site", "study_group"], df
    )
    static_reals = _filter_present(["age"], df)

    time_varying_known_categoricals = _filter_present(["sleep_stage"], df)
    time_varying_known_reals = _filter_present(
        [
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
        ],
        df,
    )
    time_varying_unknown_reals = _filter_present(
        ["cgm_glucose"],
        df,
    )

    cut_off_date = train_df["ds"].max() - horizon  # Use latest horizon

    training = TimeSeriesDataSet(
        df[df["ds"] < cut_off_date],
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

    validation = training.from_dataset(training, df, predict=True, stop_randomization=True)

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
# Training
# =============================

def train_deepar(
    df: pd.DataFrame,
    preset: Dict,
    bucket_name: str,
    gcs_ckpt_prefix: str,
    max_epochs: int = 30,
    limit_train_batches: Optional[float] = None,
    devices: int = 1,
    strategy: Optional[str] = None,  # e.g. "ddp"
) -> Tuple[DeepAR, Trainer, torch.utils.data.DataLoader, torch.utils.data.DataLoader, TimeSeriesDataSet, TimeSeriesDataSet]:
    ds_cfg = preset["dataset"]
    ar_cfg = preset["deepar_init"]

    horizon = ds_cfg["horizon"]
    context_length = ds_cfg["context_length"]
    batchsize = ds_cfg["batch_size"]

    log_gpu("Début création dataloaders")
    training, val_loader, train_loader, validation = create_deepar_dataloaders(
        df, horizon=horizon, context_length=context_length, batchsize=batchsize
    )

    # libère le gros DF
    del df
    gc.collect()
    log_gpu("Dataloaders OK")

    loss = NormalDistributionLoss()

    model = DeepAR.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=ar_cfg["hidden_size"],
        rnn_layers=ar_cfg["rnn_layers"],
        dropout=ar_cfg["dropout"],
        loss=loss,
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=4,
        weight_decay=1e-4,    # For stability
    )

    # Callbacks
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="deepar576-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        save_on_train_epoch_end=True,
    )
    es_cb = EarlyStopping(monitor="val_loss", patience=15, mode="min")
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    gcs_cb = GCSCheckpointUploader(ckpt_dir, bucket_name=bucket_name, gcs_prefix=gcs_ckpt_prefix)

    # Resume if possible
    ckpt_path = download_last_ckpt(ckpt_dir, bucket_name, gcs_ckpt_prefix)
    if ckpt_path:
        print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
    else:
        print("[INFO] No remote checkpoint — starting fresh")

    trainer = Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=1.0,
        val_check_interval=0.2,
        callbacks=[es_cb, lr_cb, ckpt_cb, gcs_cb],
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        enable_progress_bar=True,
    )

    log_gpu("Avant entraînement")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    log_gpu("Fin entraînement")

    return model, trainer, val_loader, train_loader, validation, training


# =============================
# Main
# =============================

def main():
    parser = argparse.ArgumentParser(description="Train DeepAR on CGM data (GCS).")
    parser.add_argument("--bucket", default=BUCKET_DEFAULT, help="GCS bucket name")
    parser.add_argument("--blob", default=BLOB_DEFAULT, help="Path (blob) to feather dataset in bucket")
    parser.add_argument("--model-name", default="DeepAR_12h_576cOPT", help="Model name for GCS artifacts")
    parser.add_argument("--preset", choices=["24", "48"], default="48", help="Hyperparam preset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--strategy", default="ddp", help='e.g. "ddp" for multi-GPU')
    parser.add_argument("--limit-train-batches", type=float, default=None, help="Optional fraction/int for quick runs")
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)
    assert_cuda()

    # GPU sanity
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[GPU] {name}  CC={cap}, BF16={'OK' if torch.cuda.is_bf16_supported() else 'NO'}")

    # Download dataset from GCS
    print(f"[INFO] Downloading gs://{args.bucket}/{args.blob}")
    client = gcs_client()
    bucket = client.bucket(args.bucket)
    blob = bucket.blob(args.blob)
    data_bytes = blob.download_as_bytes()
    df = pd.read_feather(io.BytesIO(data_bytes))

    # Choose preset
    preset = param_deepar_48

    # Train
    model, trainer, val_loader, train_loader, validation, training = train_deepar(
        df=df,
        preset=preset,
        bucket_name=args.bucket,
        gcs_ckpt_prefix=GCS_CKPT_PREFIX,
        max_epochs=args.epochs,
        limit_train_batches=args.limit_train_batches,
        devices=args.devices,
        strategy=args.strategy,
    )

    # Save model artifacts to GCS
    save_model_to_gcs(
        model,
        model_name=args.model_name,
        training_dataset=training,
        bucket_name=args.bucket,
        prefix=GCS_MODEL_PREFIX,
    )
    print("[DONE] Training complete and artifacts uploaded.")


if __name__ == "__main__":
    main()
