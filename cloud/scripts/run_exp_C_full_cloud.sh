#!/usr/bin/env bash
# run_exp_C_full_cloud.sh
# =======================
# Cloud Batch runner for Experiment C training — 4×A100 (a2-highgpu-4g), STANDARD.
#
# Training regime:
#   - 12 epochs, each capped at 20,000 batches (≈60% of a full 8-GPU epoch)
#   - Validation: predict_last_only=True (1 window/participant), capped at 2000 windows
#   - No full rolling-window evaluation during training (see --skip-final-eval)
#
# Full rolling-window evaluation should be run separately after training
# using the best checkpoint (--eval-only mode).
#
# Flow:
#   1. Start nvidia-smi logger
#   2. Download feathers + static feature list from GCS → /tmp/exp_C_data/
#   3. Run mamba288_static_participant_split_local.py (12 epochs, 4 GPUs)
#   4. Upload checkpoint + results to GCS on exit (via trap)

set -euo pipefail

GCS_SCRIPTS="gs://cgmproject2025/scripts/SSMCGM"
GCS_DATA="gs://cgmproject2025/scripts/SSMCGM/data_exp_C"
GCS_OUT="gs://cgmproject2025/results/exp_C"
PYTHON="/opt/micromamba/envs/cgmenv/bin/python"
SCRIPTS_DIR="/mnt/disks/scripts"
DATA_DIR="/tmp/exp_C_data"
RESULTS_DIR="/tmp/exp_C_results"
LOGDIR="/tmp/gpu_metrics_logs"
mkdir -p "${DATA_DIR}" "${RESULTS_DIR}" "${LOGDIR}"

NVIDIA_SMI_CSV="${LOGDIR}/nvidia_smi_metrics.csv"
NVIDIA_SMI_STDERR="${LOGDIR}/nvidia_smi_stderr.log"
TRAIN_LOG="${LOGDIR}/training_stdout_stderr.log"
RUNTIME_ENV="${LOGDIR}/batch_runtime_env.txt"

cleanup() {
  set +e

  echo "=== STOPPING NVIDIA-SMI LOGGER ==="
  [ -n "${NSMI_PID:-}" ] && kill "${NSMI_PID}" && wait "${NSMI_PID}" || true

  echo "=== TRAINING LOG TAIL ==="
  tail -n 80 "${TRAIN_LOG}" || true

  echo "=== WRITE RUNTIME ENV ==="
  {
    echo "BATCH_JOB_UID=${BATCH_JOB_UID:-NA}"
    echo "BATCH_TASK_INDEX=${BATCH_TASK_INDEX:-NA}"
    echo "START_UTC=${START_UTC:-NA}"
    echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "HOSTNAME=$(hostname)"
    echo "TRAIN_EXIT_CODE=${TRAIN_EXIT_CODE:-NA}"
    echo "N_GPUS=4"
    echo "LIMIT_TRAIN_BATCHES=20000"
    echo "MAX_EPOCHS=12"
  } > "${RUNTIME_ENV}"
  cat "${RUNTIME_ENV}" || true

  OUT_PREFIX="${GCS_OUT}/${BATCH_JOB_UID:-manual}_${BATCH_TASK_INDEX:-0}"
  echo "=== UPLOADING RESULTS ==="
  gsutil -m cp -r "${RESULTS_DIR}/" "${OUT_PREFIX}/results/" || true
  gsutil -m cp "${NVIDIA_SMI_CSV}"  "${OUT_PREFIX}/nvidia_smi_metrics.csv" || true
  gsutil -m cp "${TRAIN_LOG}"       "${OUT_PREFIX}/training.log" || true
  gsutil -m cp "${RUNTIME_ENV}"     "${OUT_PREFIX}/runtime_env.txt" || true
  echo "=== Results at: ${OUT_PREFIX}/ ==="
}
trap cleanup EXIT

echo "=== GPU CHECK ==="
nvidia-smi
ls -l /dev/nvidia* || true

echo "=== START NVIDIA-SMI LOGGER ==="
(
  echo "timestamp_utc,gpu_index,gpu_name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w,temperature_gpu_c"
  while true; do
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    nvidia-smi \
      --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits \
      | awk -v ts="$ts" -F ', ' '{print ts","$1","$2","$3","$4","$5","$6","$7}'
    sleep 5
  done
) > "${NVIDIA_SMI_CSV}" 2> "${NVIDIA_SMI_STDERR}" &
NSMI_PID=$!
echo "nvidia-smi logger PID: ${NSMI_PID}"
sleep 3

echo "=== DOWNLOADING DATA ==="
gsutil -m cp "${GCS_DATA}/train_timeseries_static.feather" "${DATA_DIR}/"
gsutil -m cp "${GCS_DATA}/val_timeseries_static.feather"   "${DATA_DIR}/"
gsutil -m cp "${GCS_DATA}/test_timeseries_static.feather"  "${DATA_DIR}/"
gsutil -m cp "${GCS_DATA}/static_feature_list.json"        "${DATA_DIR}/"
echo "  Data ready in ${DATA_DIR}/"
ls -lh "${DATA_DIR}/"

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== START TRAINING: ${START_UTC} ==="
echo "    Epochs            : 12"
echo "    GPUs              : 4  (a2-highgpu-4g)"
echo "    Batch/GPU         : 8  (global batch = 32)"
echo "    limit_train_batches: 20000  (~60% of full epoch)"
echo "    Validation        : predict_last_only, max 2000 windows"
echo "    Final rolling eval: SKIPPED (run separately with --eval-only)"

# Full rolling-window evaluation should be run separately after training
# using the best checkpoint (--eval-only mode).
"${PYTHON}" "${SCRIPTS_DIR}/mamba288_static_participant_split_local.py" \
    --train  "${DATA_DIR}/train_timeseries_static.feather" \
    --val    "${DATA_DIR}/val_timeseries_static.feather"   \
    --test   "${DATA_DIR}/test_timeseries_static.feather"  \
    --static-feature-list "${DATA_DIR}/static_feature_list.json" \
    --out    "${RESULTS_DIR}" \
    --epochs 12 \
    --batch-size 8 \
    --train-predict-last-only \
    --max-val-windows 2000 \
    --skip-final-eval \
    --save-all-epoch-checkpoints \
    --limit-train-batches 20000 \
    > "${TRAIN_LOG}" 2>&1
TRAIN_EXIT_CODE=$?

echo "=== TRAINING EXIT CODE: ${TRAIN_EXIT_CODE} ==="
exit "${TRAIN_EXIT_CODE}"
