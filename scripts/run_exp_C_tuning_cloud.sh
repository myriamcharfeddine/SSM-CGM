#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

sanitize_id() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g'
}

mask_secret() {
  local value="${1:-}"
  if [[ -z "$value" ]]; then
    printf ""
  elif [[ "${#value}" -le 8 ]]; then
    printf "***"
  else
    printf "%s...%s" "${value:0:4}" "${value: -4}"
  fi
}

: "${LEARNING_RATE:?LEARNING_RATE is required}"
: "${DROPOUT:?DROPOUT is required}"
: "${WEIGHT_DECAY:?WEIGHT_DECAY is required}"
: "${BATCH_SIZE:?BATCH_SIZE is required}"
: "${MAX_VAL_WINDOWS:?MAX_VAL_WINDOWS is required}"

USE_WANDB="${USE_WANDB:-false}"
case "$USE_WANDB" in
  true|false) ;;
  *) fail "USE_WANDB must be true or false" ;;
esac
if [[ "$USE_WANDB" == "true" ]]; then
  : "${WANDB_API_KEY:?WANDB_API_KEY is required when USE_WANDB=true}"
fi
WANDB_PROJECT="${WANDB_PROJECT:-ssmcgm}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT
export WANDB_ENTITY
if [[ "$USE_WANDB" == "true" ]]; then
  echo "WANDB_API_KEY is set: ${WANDB_API_KEY:0:8}********"
else
  echo "[W&B] Disabled; skipping W&B login and logger setup."
fi

PYTHON_BIN="${PYTHON_BIN:-/opt/micromamba/envs/cgmenv/bin/python}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/mnt/disks/scripts}"
TRAINING_SCRIPT="${TRAINING_SCRIPT:-${SCRIPT_ROOT}/mamba288_static_participant_split_local.py}"
EXP_C_DATA_GCS_ROOT="${EXP_C_DATA_GCS_ROOT:-}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/tmp/exp_C_tuning_data}"
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-/tmp/exp_C_tuning_results}"
GCS_OUTPUT_ROOT="${GCS_OUTPUT_ROOT:-}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-4g}"
CONTEXT_HOURS="${CONTEXT_HOURS:-48}"
HORIZON_HOURS="${HORIZON_HOURS:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
MIN_EPOCHS="${MIN_EPOCHS:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
DEVICES="${DEVICES:-4}"
STRATEGY="${STRATEGY:-auto}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-2}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-20000}"
NUM_WORKERS="${NUM_WORKERS:-${TRAIN_NUM_WORKERS:-2}}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-$NUM_WORKERS}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-$TRAIN_NUM_WORKERS}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
SHM_SIZE_MIB="${SHM_SIZE_MIB:-0}"
CONTAINER_OPTIONS="${CONTAINER_OPTIONS:-}"
ENABLE_MEMORY_MONITOR="${ENABLE_MEMORY_MONITOR:-false}"
SCALAR_ONLY_TRAINING_VALIDATION="${SCALAR_ONLY_TRAINING_VALIDATION:-false}"
DISABLE_TRAINING_PREDICTION_STORAGE="${DISABLE_TRAINING_PREDICTION_STORAGE:-true}"
LOAD_TEST_DURING_TRAINING="${LOAD_TEST_DURING_TRAINING:-true}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-false}"
FINAL_EVAL_RETURN_X="${FINAL_EVAL_RETURN_X:-true}"
FINAL_EVAL_RETURN_Y="${FINAL_EVAL_RETURN_Y:-true}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_C_tuning_fullrolling_shmsize}"
PYTHON="${PYTHON:-$PYTHON_BIN}"
PYTHON_BIN="$PYTHON"

: "${EXP_C_DATA_GCS_ROOT:?EXP_C_DATA_GCS_ROOT is required, e.g. gs://BUCKET/path/to/data_exp_C}"
: "${GCS_OUTPUT_ROOT:?GCS_OUTPUT_ROOT is required, e.g. gs://BUCKET/results/exp_C_tuning}"

TASK_INDEX="${BATCH_TASK_INDEX:-0}"
RAW_JOB_UID="${BATCH_JOB_UID:-${BATCH_JOB_ID:-manual-$(date -u +%Y%m%d%H%M%S)}}"
JOB_UID="$(sanitize_id "$RAW_JOB_UID")"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME:-exp_C_tuning_${JOB_UID}_${TASK_INDEX}}}"
RUN_NAME="$WANDB_RUN_NAME"
WANDB_RUN_ID="${WANDB_RUN_ID:-$(sanitize_id "${WANDB_RUN_NAME}-${JOB_UID}-${TASK_INDEX}")}"
export WANDB_RUN_NAME RUN_NAME WANDB_RUN_ID

RUN_STEM="${JOB_UID}_${TASK_INDEX}"
RUN_DIR="${LOCAL_OUTPUT_ROOT%/}/${RUN_STEM}"
RUN_GCS_OUT="${GCS_OUTPUT_ROOT%/}/${RUN_STEM}"
DATA_DIR="${LOCAL_DATA_ROOT%/}"
TRAINING_LOG="${RUN_DIR}/training.log"
RUN_STATUS_JSON="${RUN_DIR}/run_status.json"
export RUN_GCS_OUT RUN_STATUS_JSON TRAINING_LOG
export SHM_SIZE_MIB CONTAINER_OPTIONS USE_WANDB ENABLE_MEMORY_MONITOR SCALAR_ONLY_TRAINING_VALIDATION DISABLE_TRAINING_PREDICTION_STORAGE LOAD_TEST_DURING_TRAINING SKIP_FINAL_EVAL FINAL_EVAL_RETURN_X FINAL_EVAL_RETURN_Y EXPERIMENT_NAME GCS_OUTPUT_ROOT LOCAL_OUTPUT_ROOT MACHINE_TYPE

mkdir -p "$DATA_DIR" "$RUN_DIR"
: > "$TRAINING_LOG"

{
  printf 'BATCH_JOB_NAME=%s\n' "${BATCH_JOB_NAME:-}"
  printf 'BATCH_JOB_UID=%s\n' "${BATCH_JOB_UID:-}"
  printf 'BATCH_TASK_INDEX=%s\n' "$TASK_INDEX"
  printf 'JOB_UID=%s\n' "$JOB_UID"
  printf 'WANDB_RUN_NAME=%s\n' "$WANDB_RUN_NAME"
  printf 'RUN_NAME=%s\n' "$RUN_NAME"
  printf 'USE_WANDB=%s\n' "$USE_WANDB"
  printf 'use_wandb=%s\n' "$USE_WANDB"
  printf 'WANDB_PROJECT=%s\n' "$WANDB_PROJECT"
  printf 'WANDB_ENTITY=%s\n' "$WANDB_ENTITY"
  printf 'WANDB_API_KEY=%s\n' "$(mask_secret "$WANDB_API_KEY")"
  printf 'GCS_OUTPUT_ROOT=%s\n' "$GCS_OUTPUT_ROOT"
  printf 'output_root=%s\n' "$GCS_OUTPUT_ROOT"
  printf 'MACHINE_TYPE=%s\n' "$MACHINE_TYPE"
  printf 'machineType=%s\n' "$MACHINE_TYPE"
  printf 'GCS_OUTPUT_PATH=%s\n' "$RUN_GCS_OUT"
  printf 'LOCAL_OUTPUT_ROOT=%s\n' "$LOCAL_OUTPUT_ROOT"
  printf 'EXP_C_DATA_GCS_ROOT=%s\n' "$EXP_C_DATA_GCS_ROOT"
  printf 'CONTEXT_HOURS=%s\n' "$CONTEXT_HOURS"
  printf 'HORIZON_HOURS=%s\n' "$HORIZON_HOURS"
  printf 'LEARNING_RATE=%s\n' "$LEARNING_RATE"
  printf 'DROPOUT=%s\n' "$DROPOUT"
  printf 'WEIGHT_DECAY=%s\n' "$WEIGHT_DECAY"
  printf 'BATCH_SIZE=%s\n' "$BATCH_SIZE"
  printf 'BATCH_SIZE_PER_GPU=%s\n' "$BATCH_SIZE"
  printf 'GLOBAL_BATCH_SIZE=%s\n' "$((BATCH_SIZE * DEVICES))"
  printf 'physical_global_batch_size=%s\n' "$((BATCH_SIZE * DEVICES))"
  printf 'DEVICES=%s\n' "$DEVICES"
  printf 'STRATEGY=%s\n' "$STRATEGY"
  printf 'NUM_SANITY_VAL_STEPS=%s\n' "$NUM_SANITY_VAL_STEPS"
  printf 'MAX_VAL_WINDOWS=%s\n' "$MAX_VAL_WINDOWS"
  printf 'MAX_EPOCHS=%s\n' "$MAX_EPOCHS"
  printf 'MIN_EPOCHS=%s\n' "$MIN_EPOCHS"
  printf 'EARLY_STOP_PATIENCE=%s\n' "$EARLY_STOP_PATIENCE"
  printf 'LIMIT_TRAIN_BATCHES=%s\n' "$LIMIT_TRAIN_BATCHES"
  printf 'NUM_WORKERS=%s\n' "$TRAIN_NUM_WORKERS"
  printf 'TRAIN_NUM_WORKERS=%s\n' "$TRAIN_NUM_WORKERS"
  printf 'VAL_NUM_WORKERS=%s\n' "$VAL_NUM_WORKERS"
  printf 'VAL_BATCH_SIZE=%s\n' "$VAL_BATCH_SIZE"
  printf 'PERSISTENT_WORKERS=false\n'
  printf 'PIN_MEMORY=false\n'
  printf 'SHM_SIZE_MIB=%s\n' "$SHM_SIZE_MIB"
  printf 'container_options=%s\n' "$CONTAINER_OPTIONS"
  printf 'rolling_windows=true\n'
  printf 'predict_last_only=false\n'
  printf 'ENABLE_MEMORY_MONITOR=%s\n' "$ENABLE_MEMORY_MONITOR"
  printf 'enable_memory_monitor=%s\n' "$ENABLE_MEMORY_MONITOR"
  printf 'SCALAR_ONLY_TRAINING_VALIDATION=%s\n' "$SCALAR_ONLY_TRAINING_VALIDATION"
  printf 'DISABLE_TRAINING_PREDICTION_STORAGE=%s\n' "$DISABLE_TRAINING_PREDICTION_STORAGE"
  printf 'LOAD_TEST_DURING_TRAINING=%s\n' "$LOAD_TEST_DURING_TRAINING"
  printf 'load_test_during_training=%s\n' "$LOAD_TEST_DURING_TRAINING"
  printf 'SKIP_FINAL_EVAL=%s\n' "$SKIP_FINAL_EVAL"
  printf 'skip_final_eval=%s\n' "$SKIP_FINAL_EVAL"
  printf 'FINAL_EVAL_RETURN_X=%s\n' "$FINAL_EVAL_RETURN_X"
  printf 'FINAL_EVAL_RETURN_Y=%s\n' "$FINAL_EVAL_RETURN_Y"
  printf 'EXPERIMENT_NAME=%s\n' "$EXPERIMENT_NAME"
} > "$RUN_DIR/runtime_env.txt"

"${PYTHON}" -m pip install --no-cache-dir psutil || true

{
  echo "[MEMORY] Initial system memory"
  free -h || true
  echo "[SHM] Checking /dev/shm"
  df -h /dev/shm || true
  mount | grep shm || true
  echo "[GPU] Initial GPU status"
  nvidia-smi || true
} | tee -a "$TRAINING_LOG" "$RUN_DIR/runtime_env.txt"

[[ -x "$PYTHON_BIN" ]] || fail "Python executable not found: $PYTHON_BIN"
[[ -f "$TRAINING_SCRIPT" ]] || fail "Training script not found: $TRAINING_SCRIPT"
command -v gcloud >/dev/null 2>&1 || fail "gcloud CLI is required in the Batch container"

if [[ "$USE_WANDB" == "true" ]]; then
  WANDB_MISSING_MESSAGE="wandb is not installed in $PYTHON_BIN. Rebuild/update the Batch image with the environment.yml wandb dependency."
  if ! "${PYTHON}" -c "import wandb" >/dev/null 2>&1; then
    "${PYTHON}" -m pip install --no-cache-dir wandb || fail "$WANDB_MISSING_MESSAGE"
  fi
  "${PYTHON}" -c "import wandb; print(wandb.__version__)" || fail "$WANDB_MISSING_MESSAGE"
  if ! "${PYTHON}" -m wandb login --relogin "$WANDB_API_KEY" >/dev/null; then
    fail "wandb login failed. Check WANDB_API_KEY."
  fi
fi

echo "[Data] Downloading Experiment C data from ${EXP_C_DATA_GCS_ROOT}"
gcloud storage cp "${EXP_C_DATA_GCS_ROOT%/}/train_timeseries_static.feather" "$DATA_DIR/"
gcloud storage cp "${EXP_C_DATA_GCS_ROOT%/}/val_timeseries_static.feather" "$DATA_DIR/"
if [[ "$LOAD_TEST_DURING_TRAINING" == "true" ]]; then
  gcloud storage cp "${EXP_C_DATA_GCS_ROOT%/}/test_timeseries_static.feather" "$DATA_DIR/"
else
  echo "[Data] Skipping test feather download for training run (LOAD_TEST_DURING_TRAINING=false)."
fi
gcloud storage cp "${EXP_C_DATA_GCS_ROOT%/}/static_feature_list.json" "$DATA_DIR/"

CMD=(
  "$PYTHON_BIN" "$TRAINING_SCRIPT"
  --train "$DATA_DIR/train_timeseries_static.feather"
  --val "$DATA_DIR/val_timeseries_static.feather"
  --test "$DATA_DIR/test_timeseries_static.feather"
  --static-feature-list "$DATA_DIR/static_feature_list.json"
  --out "$RUN_DIR"
  --context-hours "$CONTEXT_HOURS"
  --horizon-hours "$HORIZON_HOURS"
  --epochs "$MAX_EPOCHS"
  --min-epochs "$MIN_EPOCHS"
  --devices "$DEVICES"
  --strategy "$STRATEGY"
  --num-sanity-val-steps "$NUM_SANITY_VAL_STEPS"
  --early-stop-patience "$EARLY_STOP_PATIENCE"
  --limit-train-batches "$LIMIT_TRAIN_BATCHES"
  --learning-rate "$LEARNING_RATE"
  --dropout "$DROPOUT"
  --weight-decay "$WEIGHT_DECAY"
  --batch-size "$BATCH_SIZE"
  --val-batch-size "$VAL_BATCH_SIZE"
  --num-workers "$TRAIN_NUM_WORKERS"
  --val-num-workers "$VAL_NUM_WORKERS"
  --max-val-windows "$MAX_VAL_WINDOWS"
  --enable-memory-monitor "$ENABLE_MEMORY_MONITOR"
  --scalar-only-training-validation "$SCALAR_ONLY_TRAINING_VALIDATION"
  --disable-training-prediction-storage "$DISABLE_TRAINING_PREDICTION_STORAGE"
  --load-test-during-training "$LOAD_TEST_DURING_TRAINING"
  --final-eval-return-x "$FINAL_EVAL_RETURN_X"
  --final-eval-return-y "$FINAL_EVAL_RETURN_Y"
  --gcs-output-path "$RUN_GCS_OUT"
  --experiment-name "$EXPERIMENT_NAME"
  --skip-test-eval
)
if [[ "$SKIP_FINAL_EVAL" == "true" ]]; then
  CMD+=(--skip-final-eval)
fi
if [[ "$USE_WANDB" == "true" ]]; then
  CMD+=(--use-wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME")
  if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

START_EPOCH="$(date +%s)"
echo "[Train] Starting ${WANDB_RUN_NAME}"
set +e
"${CMD[@]}" 2>&1 | tee -a "$TRAINING_LOG"
EXIT_CODE="${PIPESTATUS[0]}"
set -e
END_EPOCH="$(date +%s)"
RUNTIME_HOURS="$(awk "BEGIN {printf \"%.6f\", (${END_EPOCH}-${START_EPOCH})/3600}")"

flag_from_log() {
  local pattern="$1"
  if [[ -f "$TRAINING_LOG" ]] && grep -Eiq "$pattern" "$TRAINING_LOG"; then
    echo true
  else
    echo false
  fi
}

export EXIT_CODE RUNTIME_HOURS TRAIN_NUM_WORKERS VAL_NUM_WORKERS ENABLE_MEMORY_MONITOR SCALAR_ONLY_TRAINING_VALIDATION DISABLE_TRAINING_PREDICTION_STORAGE LOAD_TEST_DURING_TRAINING SKIP_FINAL_EVAL FINAL_EVAL_RETURN_X FINAL_EVAL_RETURN_Y
export HAD_NAN_WARNING="$(flag_from_log '(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_].*(warning|detected|loss|grad|value)|$)|warning.*(nan|inf)|loss.*(nan|inf)')"
export HAD_UNKNOWN_CLASSES_WARNING="$(flag_from_log 'unknown class|unknown categor|not in encoder|out-of-vocabulary|\bOOV\b')"
export HAD_BUS_ERROR="$(flag_from_log 'bus error')"
export HAD_NCCL_ERROR="$(flag_from_log '\bNCCL\b|nccl')"
export HAD_OOM_ERROR="$(flag_from_log 'out of memory|CUDA.*OOM|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory')"

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path


def parse_limit(value):
    text = str(value)
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return text

batch_size = int(os.environ["BATCH_SIZE"])
devices = int(os.environ.get("DEVICES", "4"))
status = {
    "experiment_name": os.environ.get("EXPERIMENT_NAME", "exp_C_tuning_fullrolling_shmsize"),
    "run_name": os.environ.get("WANDB_RUN_NAME"),
    "batch_job_name": os.environ.get("BATCH_JOB_NAME"),
    "gcs_output_root": os.environ.get("GCS_OUTPUT_ROOT"),
    "gcs_output_path": os.environ.get("RUN_GCS_OUT"),
    "local_output_root": os.environ.get("LOCAL_OUTPUT_ROOT"),
    "machineType": os.environ.get("MACHINE_TYPE"),
    "machine_type": os.environ.get("MACHINE_TYPE"),
    "context_hours": float(os.environ.get("CONTEXT_HOURS", "48")),
    "horizon_hours": float(os.environ.get("HORIZON_HOURS", "1")),
    "learning_rate": float(os.environ["LEARNING_RATE"]),
    "dropout": float(os.environ["DROPOUT"]),
    "weight_decay": float(os.environ["WEIGHT_DECAY"]),
    "batch_size_per_gpu": batch_size,
    "devices": devices,
    "strategy": os.environ.get("STRATEGY", "auto"),
    "num_sanity_val_steps": int(os.environ.get("NUM_SANITY_VAL_STEPS", "2")),
    "global_batch_size": batch_size * devices,
    "physical_global_batch_size": batch_size * devices,
    "train_num_workers": int(os.environ.get("TRAIN_NUM_WORKERS", os.environ.get("NUM_WORKERS", "2"))),
    "val_num_workers": int(os.environ.get("VAL_NUM_WORKERS", "0")),
    "val_batch_size": int(os.environ.get("VAL_BATCH_SIZE", "4")),
    "max_val_windows": int(os.environ["MAX_VAL_WINDOWS"]),
    "limit_train_batches": parse_limit(os.environ.get("LIMIT_TRAIN_BATCHES", "20000")),
    "max_epochs": int(os.environ.get("MAX_EPOCHS", "30")),
    "min_epochs": int(os.environ.get("MIN_EPOCHS", "10")),
    "early_stopping_patience": int(os.environ.get("EARLY_STOP_PATIENCE", "5")),
    "persistent_workers": False,
    "pin_memory": False,
    "predict_last_only": False,
    "rolling_windows": True,
    "use_wandb": os.environ.get("USE_WANDB", "false") == "true",
    "enable_memory_monitor": os.environ.get("ENABLE_MEMORY_MONITOR", "false") == "true",
    "scalar_only_training_validation": os.environ.get("SCALAR_ONLY_TRAINING_VALIDATION", "false") == "true",
    "disable_training_prediction_storage": os.environ.get("DISABLE_TRAINING_PREDICTION_STORAGE", "true") == "true",
    "load_test_during_training": os.environ.get("LOAD_TEST_DURING_TRAINING", "true") == "true",
    "skip_final_eval": os.environ.get("SKIP_FINAL_EVAL", "false") == "true",
    "final_eval_return_x": os.environ.get("FINAL_EVAL_RETURN_X", "true") == "true",
    "final_eval_return_y": os.environ.get("FINAL_EVAL_RETURN_Y", "true") == "true",
    "shm_size_mib": int(os.environ.get("SHM_SIZE_MIB", "0")),
    "container_options": os.environ.get("CONTAINER_OPTIONS", ""),
    "runtime_hours": float(os.environ["RUNTIME_HOURS"]),
    "exit_code": int(os.environ["EXIT_CODE"]),
    "had_nan_warning": os.environ["HAD_NAN_WARNING"] == "true",
    "had_unknown_classes_warning": os.environ["HAD_UNKNOWN_CLASSES_WARNING"] == "true",
    "had_bus_error": os.environ["HAD_BUS_ERROR"] == "true",
    "had_nccl_error": os.environ["HAD_NCCL_ERROR"] == "true",
    "had_oom_error": os.environ["HAD_OOM_ERROR"] == "true",
}
Path(os.environ["RUN_STATUS_JSON"]).write_text(json.dumps(status, indent=2) + "\n")
PY

if [[ "$USE_WANDB" == "true" ]]; then
  "$PYTHON_BIN" - <<'PY' || echo "[WARN] Could not update W&B run status" >&2
import json
import os
from pathlib import Path

import wandb

status = json.loads(Path(os.environ["RUN_STATUS_JSON"]).read_text())
run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "ssmcgm"),
    entity=os.environ.get("WANDB_ENTITY") or None,
    id=os.environ.get("WANDB_RUN_ID"),
    name=os.environ.get("WANDB_RUN_NAME"),
    resume="allow",
)
numeric = {k: v for k, v in status.items() if isinstance(v, (int, float, bool))}
if numeric:
    wandb.log(numeric)
run.summary.update(status)
run.finish(exit_code=int(status.get("exit_code", 0)))
PY
else
  echo "[W&B] Disabled; skipping W&B run status update."
fi

echo "[Upload] Syncing outputs to ${RUN_GCS_OUT}"
set +e
gcloud storage rsync --recursive "$RUN_DIR" "$RUN_GCS_OUT"
UPLOAD_CODE="$?"
set -e
if [[ "$UPLOAD_CODE" -ne 0 ]]; then
  echo "[ERROR] Output upload failed with exit code ${UPLOAD_CODE}" >&2
fi

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "[ERROR] Training failed with exit code ${EXIT_CODE}" >&2
  exit "$EXIT_CODE"
fi
exit "$UPLOAD_CODE"
