#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Usage: scripts/submit_exp_C_tuning_batch.sh \
  --learning-rate LR \
  --batch-size BATCH \
  --max-val-windows N \
  --run-name NAME \
  [--dropout DROPOUT] \
  [--weight-decay WD] \
  [--limit-train-batches N_OR_FRAC] \
  [--epochs N] \
  [--devices N] \
  [--strategy NAME] \
  [--machine-type TYPE] \
  [--num-sanity-val-steps N] \
  [--num-workers N] \
  [--train-num-workers N] \
  [--val-num-workers N] \
  [--val-batch-size N] \
  [--shm-size-mib MIB] \
  [--use-wandb true|false] \
  [--enable-memory-monitor true|false] \
  [--scalar-only-training-validation true|false] \
  [--disable-training-prediction-storage true|false] \
  [--load-test-during-training true|false] \
  [--skip-final-eval true|false] \
  [--final-eval-return-x true|false] \
  [--final-eval-return-y true|false] \
  [--gcs-output-root URI] \
  [--output-root URI] \
  [--local-output-root PATH] \
  [--experiment-name NAME] \
  [--dry-run] \
  [--job-id exp-c-tune-YYYYMMDD-HHMMSS-XX]

Required environment:
  WANDB_API_KEY            Required only when --use-wandb true.

Optional environment:
  WANDB_PROJECT            Default: ssmcgm
  WANDB_ENTITY             Default: unset
  GCP_PROJECT_ID           Required.
  GCP_LOCATION             Default: us-central1
  GCP_SERVICE_ACCOUNT_EMAIL Required for Batch jobs.
  IMAGE_URI                Required Batch container image.
  REMOTE_SCRIPT_ROOT       Required gs:// path for uploaded scripts.
  EXP_C_DATA_GCS_ROOT      Required gs:// path containing Experiment C data.
  LOCAL_EXP_C_DATA_ROOT    Default: ../Data/ssmcgm_ready_exp_C
  SKIP_DATA_UPLOAD         Default: 0; set to 1 if data is already staged
  SKIP_SCRIPT_UPLOAD       Default: 0; set to 1 if scripts are already staged
  GCS_OUTPUT_ROOT          Required unless passed via --gcs-output-root.
  LOCAL_OUTPUT_ROOT        Default: /tmp/exp_C_tuning_results
  CONTAINER_OPTIONS        Default from --shm-size-mib, e.g. --shm-size=20g
USAGE
}

sanitize_job_id() {
  local raw="${1:-}"
  local cleaned
  cleaned="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9-]+/-/g; s/-+/-/g; s/^-+//; s/-+$//')"
  if [[ -z "$cleaned" || ! "$cleaned" =~ ^[a-z] ]]; then
    cleaned="exp-c-tune${cleaned:+-$cleaned}"
  fi
  cleaned="${cleaned:0:63}"
  cleaned="$(printf '%s' "$cleaned" | sed -E 's/-+$//')"
  if [[ -z "$cleaned" || ! "$cleaned" =~ ^[a-z] ]]; then
    cleaned="exp-c-tune"
  fi
  printf '%s' "$cleaned"
}

validate_job_id() {
  local job_id="$1"
  if [[ ! "$job_id" =~ ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$ ]]; then
    fail "Generated Batch job ID is invalid: ${job_id}"
  fi
}

make_short_job_id() {
  local suffix="${1:-$(printf '%02d' $((RANDOM % 100)))}"
  local candidate
  candidate="$(sanitize_job_id "exp-c-tune-$(date -u +%Y%m%d-%H%M%S)-${suffix}")"
  validate_job_id "$candidate"
  printf '%s' "$candidate"
}

LEARNING_RATE=""
DROPOUT=""
WEIGHT_DECAY=""
BATCH_SIZE=""
MAX_VAL_WINDOWS=""
LIMIT_TRAIN_BATCHES=""
MAX_EPOCHS=""
DEVICES=""
STRATEGY=""
NUM_SANITY_VAL_STEPS=""
TRAIN_NUM_WORKERS=""
VAL_NUM_WORKERS=""
VAL_BATCH_SIZE=""
SHM_SIZE_MIB=""
MACHINE_TYPE_ARG=""
USE_WANDB="true"
ENABLE_MEMORY_MONITOR="false"
SCALAR_ONLY_TRAINING_VALIDATION="false"
DISABLE_TRAINING_PREDICTION_STORAGE="true"
LOAD_TEST_DURING_TRAINING="true"
SKIP_FINAL_EVAL="false"
FINAL_EVAL_RETURN_X="true"
FINAL_EVAL_RETURN_Y="true"
GCS_OUTPUT_ROOT_ARG=""
LOCAL_OUTPUT_ROOT_ARG=""
EXPERIMENT_NAME=""
RUN_NAME=""
JOB_ID=""
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-val-windows) MAX_VAL_WINDOWS="$2"; shift 2 ;;
    --limit-train-batches) LIMIT_TRAIN_BATCHES="$2"; shift 2 ;;
    --epochs|--max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --strategy) STRATEGY="$2"; shift 2 ;;
    --machine-type) MACHINE_TYPE_ARG="$2"; shift 2 ;;
    --num-sanity-val-steps) NUM_SANITY_VAL_STEPS="$2"; shift 2 ;;
    --num-workers|--train-num-workers) TRAIN_NUM_WORKERS="$2"; shift 2 ;;
    --val-num-workers) VAL_NUM_WORKERS="$2"; shift 2 ;;
    --val-batch-size) VAL_BATCH_SIZE="$2"; shift 2 ;;
    --shm-size-mib) SHM_SIZE_MIB="$2"; shift 2 ;;
    --use-wandb) USE_WANDB="$2"; shift 2 ;;
    --enable-memory-monitor) ENABLE_MEMORY_MONITOR="$2"; shift 2 ;;
    --scalar-only-training-validation) SCALAR_ONLY_TRAINING_VALIDATION="$2"; shift 2 ;;
    --disable-training-prediction-storage) DISABLE_TRAINING_PREDICTION_STORAGE="$2"; shift 2 ;;
    --load-test-during-training) LOAD_TEST_DURING_TRAINING="$2"; shift 2 ;;
    --skip-final-eval) SKIP_FINAL_EVAL="$2"; shift 2 ;;
    --final-eval-return-x) FINAL_EVAL_RETURN_X="$2"; shift 2 ;;
    --final-eval-return-y) FINAL_EVAL_RETURN_Y="$2"; shift 2 ;;
    --gcs-output-root|--output-root) GCS_OUTPUT_ROOT_ARG="$2"; shift 2 ;;
    --local-output-root) LOCAL_OUTPUT_ROOT_ARG="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --job-id) JOB_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

[[ -n "$LEARNING_RATE" ]] || fail "--learning-rate is required"
[[ -n "$BATCH_SIZE" ]] || fail "--batch-size is required"
[[ -n "$MAX_VAL_WINDOWS" ]] || fail "--max-val-windows is required"
[[ -n "$RUN_NAME" ]] || fail "--run-name is required"
DROPOUT="${DROPOUT:-0.2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-20000}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
DEVICES="${DEVICES:-4}"
STRATEGY="${STRATEGY:-auto}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-2}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-$TRAIN_NUM_WORKERS}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
SHM_SIZE_MIB="${SHM_SIZE_MIB:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_C_tuning_fullrolling_shmsize}"

case "$USE_WANDB" in
  true|false) ;;
  *) fail "--use-wandb must be true or false" ;;
esac
case "$ENABLE_MEMORY_MONITOR" in
  true|false) ;;
  *) fail "--enable-memory-monitor must be true or false" ;;
esac
case "$SCALAR_ONLY_TRAINING_VALIDATION" in
  true|false) ;;
  *) fail "--scalar-only-training-validation must be true or false" ;;
esac
case "$DISABLE_TRAINING_PREDICTION_STORAGE" in
  true|false) ;;
  *) fail "--disable-training-prediction-storage must be true or false" ;;
esac
case "$LOAD_TEST_DURING_TRAINING" in
  true|false) ;;
  *) fail "--load-test-during-training must be true or false" ;;
esac
case "$SKIP_FINAL_EVAL" in
  true|false) ;;
  *) fail "--skip-final-eval must be true or false" ;;
esac
case "$FINAL_EVAL_RETURN_X" in
  true|false) ;;
  *) fail "--final-eval-return-x must be true or false" ;;
esac
case "$FINAL_EVAL_RETURN_Y" in
  true|false) ;;
  *) fail "--final-eval-return-y must be true or false" ;;
esac
[[ "$SHM_SIZE_MIB" =~ ^[0-9]+$ ]] || fail "--shm-size-mib must be a non-negative integer"
[[ "$MAX_EPOCHS" =~ ^[0-9]+$ ]] || fail "--epochs must be a positive integer"
[[ "$DEVICES" =~ ^[0-9]+$ ]] || fail "--devices must be a positive integer"
[[ "$NUM_SANITY_VAL_STEPS" =~ ^[0-9]+$ ]] || fail "--num-sanity-val-steps must be a non-negative integer"
if [[ "$USE_WANDB" == "true" ]]; then
  : "${WANDB_API_KEY:?WANDB_API_KEY is not set. Export it before submitting W&B-enabled tuning jobs.}"
fi

WANDB_RUN_NAME="$RUN_NAME"
if [[ -n "$JOB_ID" ]]; then
  JOB_ID="$(sanitize_job_id "$JOB_ID")"
  validate_job_id "$JOB_ID"
else
  JOB_ID="$(make_short_job_id)"
fi

if [[ "$DRY_RUN" != "true" ]]; then
  command -v gcloud >/dev/null 2>&1 || fail "gcloud CLI is required"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAINER_PATH="${REPO_ROOT}/Benchmarking/Day1/mamba288_static_participant_split_local.py"
RUN_SCRIPT_PATH="${SCRIPT_DIR}/run_exp_C_tuning_cloud.sh"
[[ -f "$TRAINER_PATH" ]] || fail "Trainer not found: $TRAINER_PATH"
[[ -f "$RUN_SCRIPT_PATH" ]] || fail "Cloud run script not found: $RUN_SCRIPT_PATH"

GCP_PROJECT_ID="${GCP_PROJECT_ID:-}"
GCP_LOCATION="${GCP_LOCATION:-us-central1}"
GCP_SERVICE_ACCOUNT_EMAIL="${GCP_SERVICE_ACCOUNT_EMAIL:-}"
IMAGE_URI="${IMAGE_URI:-}"
REMOTE_SCRIPT_ROOT="${REMOTE_SCRIPT_ROOT:-}"
EXP_C_DATA_GCS_ROOT="${EXP_C_DATA_GCS_ROOT:-}"
LOCAL_EXP_C_DATA_ROOT="${LOCAL_EXP_C_DATA_ROOT:-${REPO_ROOT}/../Data/ssmcgm_ready_exp_C}"
GCS_OUTPUT_ROOT="${GCS_OUTPUT_ROOT_ARG:-${GCS_OUTPUT_ROOT:-}}"
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT_ARG:-${LOCAL_OUTPUT_ROOT:-/tmp/exp_C_tuning_results}}"
WANDB_PROJECT="${WANDB_PROJECT:-ssmcgm}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
MACHINE_TYPE="${MACHINE_TYPE_ARG:-${MACHINE_TYPE:-a2-highgpu-4g}}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-STANDARD}"
BOOT_DISK_GB="${BOOT_DISK_GB:-300}"
MAX_RETRY_COUNT="${MAX_RETRY_COUNT:-0}"
if [[ "$MACHINE_TYPE" == "a2-highgpu-8g" ]]; then
  CPU_MILLI="${CPU_MILLI:-96000}"
  MEMORY_MIB="${MEMORY_MIB:-696320}"
else
  CPU_MILLI="${CPU_MILLI:-48000}"
  MEMORY_MIB="${MEMORY_MIB:-348160}"
fi

[[ -n "$GCP_PROJECT_ID" ]] || fail "GCP_PROJECT_ID is required for Batch submission."
[[ -n "$GCP_SERVICE_ACCOUNT_EMAIL" ]] || fail "GCP_SERVICE_ACCOUNT_EMAIL is required for Batch submission."
[[ -n "$IMAGE_URI" ]] || fail "IMAGE_URI is required for Batch submission."
[[ -n "$REMOTE_SCRIPT_ROOT" ]] || fail "REMOTE_SCRIPT_ROOT is required, e.g. gs://BUCKET/scripts/exp_C_tuning."
[[ -n "$EXP_C_DATA_GCS_ROOT" ]] || fail "EXP_C_DATA_GCS_ROOT is required, e.g. gs://BUCKET/path/to/data_exp_C."
[[ -n "$GCS_OUTPUT_ROOT" ]] || fail "GCS_OUTPUT_ROOT is required or pass --gcs-output-root."

if [[ -z "${CONTAINER_OPTIONS:-}" && "$SHM_SIZE_MIB" -gt 0 ]]; then
  if (( SHM_SIZE_MIB % 1024 == 0 )); then
    CONTAINER_OPTIONS="--shm-size=$((SHM_SIZE_MIB / 1024))g"
  else
    CONTAINER_OPTIONS="--shm-size=${SHM_SIZE_MIB}m"
  fi
fi
CONTAINER_OPTIONS="${CONTAINER_OPTIONS:-}"

REQUIRED_EXP_C_DATA_FILES=(
  train_timeseries_static.feather
  val_timeseries_static.feather
  test_timeseries_static.feather
  static_feature_list.json
)

ensure_exp_c_data_file() {
  local file_name="$1"
  local local_path="${LOCAL_EXP_C_DATA_ROOT%/}/${file_name}"
  local remote_uri="${EXP_C_DATA_GCS_ROOT%/}/${file_name}"

  if gcloud storage ls "$remote_uri" >/dev/null 2>&1; then
    echo "[Data] Found ${remote_uri}"
    return 0
  fi

  [[ -f "$local_path" ]] || fail "Missing Exp C data file locally and in GCS: ${file_name}. Checked local ${local_path} and remote ${remote_uri}."
  echo "[Upload] Staging ${file_name} to ${remote_uri}"
  gcloud storage cp "$local_path" "$remote_uri"
}

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[Dry-run] Skipping data/script upload and Batch submission."
elif [[ "${SKIP_DATA_UPLOAD:-0}" != "1" ]]; then
  echo "[Data] Ensuring Experiment C data exists at ${EXP_C_DATA_GCS_ROOT}"
  for data_file in "${REQUIRED_EXP_C_DATA_FILES[@]}"; do
    ensure_exp_c_data_file "$data_file"
  done
else
  echo "[Data] SKIP_DATA_UPLOAD=1; expecting Experiment C data at ${EXP_C_DATA_GCS_ROOT}"
fi

if [[ "$DRY_RUN" != "true" && "${SKIP_SCRIPT_UPLOAD:-0}" != "1" ]]; then
  echo "[Upload] Syncing tuning scripts to ${REMOTE_SCRIPT_ROOT}"
  gcloud storage cp "$RUN_SCRIPT_PATH" "${REMOTE_SCRIPT_ROOT%/}/run_exp_C_tuning_cloud.sh"
  gcloud storage cp "$TRAINER_PATH" "${REMOTE_SCRIPT_ROOT%/}/mamba288_static_participant_split_local.py"
fi

CONFIG_FILE="$(mktemp -t exp_c_tuning_batch.XXXXXX.json)"
trap 'rm -f "$CONFIG_FILE"' EXIT
export CONFIG_FILE IMAGE_URI REMOTE_SCRIPT_ROOT EXP_C_DATA_GCS_ROOT GCS_OUTPUT_ROOT LOCAL_OUTPUT_ROOT
export WANDB_API_KEY="${WANDB_API_KEY:-}" WANDB_PROJECT WANDB_ENTITY WANDB_RUN_NAME RUN_NAME LEARNING_RATE DROPOUT WEIGHT_DECAY BATCH_SIZE MAX_VAL_WINDOWS JOB_ID
export LIMIT_TRAIN_BATCHES MAX_EPOCHS DEVICES STRATEGY NUM_SANITY_VAL_STEPS TRAIN_NUM_WORKERS VAL_NUM_WORKERS VAL_BATCH_SIZE SHM_SIZE_MIB CONTAINER_OPTIONS USE_WANDB ENABLE_MEMORY_MONITOR SCALAR_ONLY_TRAINING_VALIDATION DISABLE_TRAINING_PREDICTION_STORAGE LOAD_TEST_DURING_TRAINING SKIP_FINAL_EVAL FINAL_EVAL_RETURN_X FINAL_EVAL_RETURN_Y EXPERIMENT_NAME
export MACHINE_TYPE PROVISIONING_MODEL BOOT_DISK_GB MAX_RETRY_COUNT CPU_MILLI MEMORY_MIB GCP_LOCATION GCP_SERVICE_ACCOUNT_EMAIL

python3 - <<'PY'
import json
import os
from pathlib import Path

remote = os.environ["REMOTE_SCRIPT_ROOT"]
if remote.startswith("gs://"):
    remote = remote[len("gs://"):]
variables = {
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
    "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "ssmcgm"),
    "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", ""),
    "WANDB_RUN_NAME": os.environ["WANDB_RUN_NAME"],
    "RUN_NAME": os.environ["WANDB_RUN_NAME"],
    "BATCH_JOB_NAME": os.environ["JOB_ID"],
    "LEARNING_RATE": os.environ["LEARNING_RATE"],
    "DROPOUT": os.environ["DROPOUT"],
    "WEIGHT_DECAY": os.environ["WEIGHT_DECAY"],
    "BATCH_SIZE": os.environ["BATCH_SIZE"],
    "MAX_VAL_WINDOWS": os.environ["MAX_VAL_WINDOWS"],
    "LIMIT_TRAIN_BATCHES": os.environ["LIMIT_TRAIN_BATCHES"],
    "MAX_EPOCHS": os.environ["MAX_EPOCHS"],
    "DEVICES": os.environ["DEVICES"],
    "STRATEGY": os.environ["STRATEGY"],
    "NUM_SANITY_VAL_STEPS": os.environ["NUM_SANITY_VAL_STEPS"],
    "NUM_WORKERS": os.environ["TRAIN_NUM_WORKERS"],
    "TRAIN_NUM_WORKERS": os.environ["TRAIN_NUM_WORKERS"],
    "VAL_NUM_WORKERS": os.environ["VAL_NUM_WORKERS"],
    "VAL_BATCH_SIZE": os.environ["VAL_BATCH_SIZE"],
    "USE_WANDB": os.environ["USE_WANDB"],
    "ENABLE_MEMORY_MONITOR": os.environ["ENABLE_MEMORY_MONITOR"],
    "SCALAR_ONLY_TRAINING_VALIDATION": os.environ["SCALAR_ONLY_TRAINING_VALIDATION"],
    "DISABLE_TRAINING_PREDICTION_STORAGE": os.environ["DISABLE_TRAINING_PREDICTION_STORAGE"],
    "LOAD_TEST_DURING_TRAINING": os.environ["LOAD_TEST_DURING_TRAINING"],
    "SKIP_FINAL_EVAL": os.environ["SKIP_FINAL_EVAL"],
    "FINAL_EVAL_RETURN_X": os.environ["FINAL_EVAL_RETURN_X"],
    "FINAL_EVAL_RETURN_Y": os.environ["FINAL_EVAL_RETURN_Y"],
    "SHM_SIZE_MIB": os.environ["SHM_SIZE_MIB"],
    "CONTAINER_OPTIONS": os.environ.get("CONTAINER_OPTIONS", ""),
    "MACHINE_TYPE": os.environ["MACHINE_TYPE"],
    "EXP_C_DATA_GCS_ROOT": os.environ["EXP_C_DATA_GCS_ROOT"],
    "GCS_OUTPUT_ROOT": os.environ["GCS_OUTPUT_ROOT"],
    "LOCAL_OUTPUT_ROOT": os.environ["LOCAL_OUTPUT_ROOT"],
    "CONTEXT_HOURS": os.environ.get("CONTEXT_HOURS", "48"),
    "HORIZON_HOURS": os.environ.get("HORIZON_HOURS", "1"),
    "MIN_EPOCHS": os.environ.get("MIN_EPOCHS", "10"),
    "EARLY_STOP_PATIENCE": os.environ.get("EARLY_STOP_PATIENCE", "5"),
    "EXPERIMENT_NAME": os.environ["EXPERIMENT_NAME"],
}
container = {
    "imageUri": os.environ["IMAGE_URI"],
    "entrypoint": "/bin/bash",
    "commands": ["/mnt/disks/scripts/run_exp_C_tuning_cloud.sh"],
    "volumes": ["/mnt/disks/scripts:/mnt/disks/scripts:rw"],
}
if os.environ.get("CONTAINER_OPTIONS"):
    container["options"] = os.environ["CONTAINER_OPTIONS"]
config = {
    "taskGroups": [{
        "taskCount": "1",
        "parallelism": "1",
        "taskSpec": {
            "maxRetryCount": int(os.environ.get("MAX_RETRY_COUNT", "0")),
            "computeResource": {
                "cpuMilli": os.environ.get("CPU_MILLI", "48000"),
                "memoryMib": os.environ.get("MEMORY_MIB", "348160"),
            },
            "volumes": [{
                "gcs": {"remotePath": remote},
                "mountPath": "/mnt/disks/scripts",
            }],
            "environment": {"variables": variables},
            "runnables": [{"container": container}],
        },
    }],
    "allocationPolicy": {
        "instances": [{
            "policy": {
                "machineType": os.environ.get("MACHINE_TYPE", "a2-highgpu-4g"),
                "provisioningModel": os.environ.get("PROVISIONING_MODEL", "STANDARD"),
                "bootDisk": {"sizeGb": os.environ.get("BOOT_DISK_GB", "300")},
            },
            "installGpuDrivers": True,
        }],
        "location": {
            "allowedLocations": [f"regions/{os.environ.get('GCP_LOCATION', 'us-central1')}"],
        },
        "serviceAccount": {
            "email": os.environ["GCP_SERVICE_ACCOUNT_EMAIL"],
        },
    },
    "logsPolicy": {"destination": "CLOUD_LOGGING"},
}
Path(os.environ["CONFIG_FILE"]).write_text(json.dumps(config, indent=2) + "\n")
PY

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[Dry-run Batch JSON] ${CONFIG_FILE}"
  cat "$CONFIG_FILE"
  exit 0
fi

echo "[Submit] ${JOB_ID}"
echo "[Run] ${WANDB_RUN_NAME}"
gcloud batch jobs submit "$JOB_ID" \
  --project="$GCP_PROJECT_ID" \
  --location="$GCP_LOCATION" \
  --config="$CONFIG_FILE"

echo "[Submitted] ${JOB_ID}"
