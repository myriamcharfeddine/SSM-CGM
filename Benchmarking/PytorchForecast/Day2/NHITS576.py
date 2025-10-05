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
from pytorch_forecasting.models.nhits import NHiTS


# GCS constants
_BUCKET_NAME = "cgmproject2025"
_BASE_PREFIX = "models/predictions"
_gcs_client = storage.Client()


# -------------------------
# Hyperparam presets
# -------------------------
param_48 = {
    "dataset": {"context_length": 576, "horizon": 12, "batch_size": 32},}

param = param_48


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


def create_nhits_dataloaders(train_df, horizon=12, context_length=72, batchsize=32):
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
        add_relative_time_idx=False,
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
# 
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
        if getattr(trainer, "global_rank", 0) != 0 or self.logged:
            return
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        with open(self.local_path, 'w') as f:
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
        if getattr(trainer, "global_rank", 0) != 0:
            return
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
        if getattr(trainer, "global_rank", 0) != 0:
            return
        self.batch_count += 1
        loss = None
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = float(outputs['loss'])
        elif hasattr(outputs, 'item'):
            try: loss = float(outputs.item())
            except: pass
        if self.batch_count % self.log_every_n_batches == 0 and loss is not None:
            with open(self.log_path, 'a') as f:
                f.write(f"Batch {self.batch_count}: loss={loss}\n")
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
def Nhits_train(train_df, param):
    ds_cfg = param["dataset"]
    horizon = ds_cfg["horizon"]
    context_length = ds_cfg["context_length"]
    batchsize = ds_cfg["batch_size"]

    training, val_dataloader, train_dataloader, validation = create_nhits_dataloaders(
        train_df, horizon=horizon, context_length=context_length, batchsize=batchsize
    )
    del train_df; gc.collect()

    loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

    nhits = NHiTS.from_dataset(
        training,
        hidden_size=128,                 # try 128 for the accuracy preset
        dropout=0.2,
        backcast_loss_ratio=0.0,
        learning_rate=0.001,
        loss=loss,
        log_interval=10,
        # pas de log_val_interval ici
        reduce_on_plateau_patience=4,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="nhits_576-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3, monitor="val_loss", mode="min", save_last=True, save_on_train_epoch_end=True
    )
    first_epoch_timer = FirstEpochTimer("first_epoch_runtime.txt", _BUCKET_NAME, "checkpoints_nhits_576")
    gcs_ckpt_prefix = "checkpoints_nhits_576"
    gcs_checkpoint_uploader = GCSCheckpointUploader("checkpoints", _BUCKET_NAME, gcs_ckpt_prefix)
    gcs_loss_logger = GCSLossLogger("loss_log.txt", _BUCKET_NAME, gcs_ckpt_prefix, log_every_n_batches=5000)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = download_last_ckpt("checkpoints", _BUCKET_NAME, gcs_ckpt_prefix)
    print(f"[INFO] Resuming from checkpoint: {ckpt_path}" if ckpt_path else "[INFO] No checkpoint found: starting fresh")

    trainer = Trainer(
        max_epochs=30,
        gradient_clip_val=1.0,
        val_check_interval=0.2,
        callbacks=[EarlyStopping(monitor="val_loss", patience=15, mode="min"),
                   checkpoint_callback, first_epoch_timer, gcs_checkpoint_uploader, gcs_loss_logger],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
    )

    trainer.fit(nhits, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)

    return nhits, trainer, val_dataloader, train_dataloader, validation, training

def save_model_to_gcs(model, model_name: str, bucket_name: str = _BUCKET_NAME, prefix: str = _BASE_PREFIX):
    bucket = _gcs_client.bucket(bucket_name)
    model_prefix = f"{prefix}/{model_name}"

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf); buf.seek(0)
    bucket.blob(f"{model_prefix}/model.pth").upload_from_file(buf); buf.close()

    buf = io.BytesIO()
    pickle.dump(model.hparams, buf); buf.seek(0)
    bucket.blob(f"{model_prefix}/model_kwargs.pkl").upload_from_file(buf); buf.close()
    print(f"Model artifacts uploaded to gs://{bucket_name}/{model_prefix}/")

def load_nhits_from_gcs(training_dataset, model_name: str, map_location=None,
                        bucket_name: str = _BUCKET_NAME, prefix: str = _BASE_PREFIX) -> NHiTS:
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = _gcs_client.bucket(bucket_name)
    model_prefix = f"{prefix}/{model_name}"

    hparams_bytes = bucket.blob(f"{model_prefix}/model_kwargs.pkl").download_as_bytes()
    model_kwargs = pickle.loads(hparams_bytes)
    model = NHiTS.from_dataset(training_dataset, **model_kwargs)

    state_bytes = bucket.blob(f"{model_prefix}/model.pth").download_as_bytes()
    state_dict = torch.load(io.BytesIO(state_bytes), map_location=map_location)
    model.load_state_dict(state_dict)
    print(f"Loaded N-HiTS from gs://{bucket_name}/{model_prefix}/")
    return model


if __name__ == "__main__":
    rank = os.environ.get("LOCAL_RANK", "NotSet")
    print(f"[INFO] RANK={rank} | CUDA devices={torch.cuda.device_count()}")

    client = storage.Client()
    bucket = client.bucket(_BUCKET_NAME)
    blob = bucket.blob('ai-ready/data/train_finaltimeseries_meal.feather')
    data_bytes = blob.download_as_bytes()
    train = pd.read_feather(io.BytesIO(data_bytes))

    nhits, trainer, val_dataloader, train_dataloader, validation, training = Nhits_train(train, param)

    save_model_to_gcs(nhits, model_name="NHiTS_12h_576c")
