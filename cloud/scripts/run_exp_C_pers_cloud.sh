#!/usr/bin/env bash
# run_exp_C_pers_cloud.sh
# =======================
# Cloud Batch runner for Experiment C personalization + adaptation sweep + subgroup analysis.
# Runs inside the gpu-base container on a2-highgpu-1g (1×A100).
#
# Flow:
#   1. Download checkpoint, feathers, split CSV, static data from GCS
#   2. Personalization — test split, 3 epochs, output_layer (C0 vs C1)
#   3. Adaptation sweep — 0h / 6h / 12h / 24h / 48h adaptation caps
#   4. Subgroup analysis — merge with static features, HbA1c/BMI quartiles, meds
#   5. Upload all results to GCS on exit (via trap)

set -euo pipefail

GCS_DATA="gs://cgmproject2025/scripts/SSMCGM/data_exp_C"
GCS_SPLIT="gs://cgmproject2025/scripts/SSMCGM/split_exp_C"
GCS_CKPT="gs://cgmproject2025/results/exp_C/best.ckpt"
GCS_OUT="gs://cgmproject2025/results/exp_C_pers"
PYTHON="/opt/micromamba/envs/cgmenv/bin/python"
SCRIPTS_DIR="/mnt/disks/scripts"
DATA_DIR="/tmp/exp_C_pers"
RESULTS_DIR="/tmp/exp_C_pers_results"
LOGDIR="/tmp/pers_logs"
mkdir -p "${DATA_DIR}" "${RESULTS_DIR}" "${LOGDIR}"

PERS_LOG="${LOGDIR}/personalization_stdout_stderr.log"
SWEEP_LOG="${LOGDIR}/sweep_stdout_stderr.log"
SUBGROUP_LOG="${LOGDIR}/subgroup_stdout_stderr.log"
RUNTIME_ENV="${LOGDIR}/batch_runtime_env.txt"

cleanup() {
  set +e

  echo "=== LOG TAILS ==="
  echo "--- personalization ---"; tail -n 40 "${PERS_LOG}" || true
  echo "--- sweep ---";           tail -n 40 "${SWEEP_LOG}" || true
  echo "--- subgroup ---";        tail -n 20 "${SUBGROUP_LOG}" || true

  echo "=== WRITE RUNTIME ENV ==="
  {
    echo "BATCH_JOB_UID=${BATCH_JOB_UID:-NA}"
    echo "BATCH_TASK_INDEX=${BATCH_TASK_INDEX:-NA}"
    echo "START_UTC=${START_UTC:-NA}"
    echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "HOSTNAME=$(hostname)"
    echo "PERS_EXIT=${PERS_EXIT:-NA}"
    echo "SWEEP_EXIT=${SWEEP_EXIT:-NA}"
    echo "SUBGROUP_EXIT=${SUBGROUP_EXIT:-NA}"
  } > "${RUNTIME_ENV}"
  cat "${RUNTIME_ENV}" || true

  OUT_PREFIX="${GCS_OUT}/${BATCH_JOB_UID:-manual}_${BATCH_TASK_INDEX:-0}"
  echo "=== UPLOADING RESULTS ==="
  gsutil -m cp -r "${RESULTS_DIR}/" "${OUT_PREFIX}/results/"      || true
  gsutil -m cp "${PERS_LOG}"        "${OUT_PREFIX}/personalization.log" || true
  gsutil -m cp "${SWEEP_LOG}"       "${OUT_PREFIX}/sweep.log"      || true
  gsutil -m cp "${SUBGROUP_LOG}"    "${OUT_PREFIX}/subgroup.log"   || true
  gsutil -m cp "${RUNTIME_ENV}"     "${OUT_PREFIX}/runtime_env.txt" || true
  echo "=== Results at: ${OUT_PREFIX}/ ==="
}
trap cleanup EXIT

echo "=== GPU CHECK ==="
nvidia-smi
ls -l /dev/nvidia* || true

# ── Download inputs ────────────────────────────────────────────────────────────
echo ""
echo "=== DOWNLOADING CHECKPOINT ==="
gsutil cp "${GCS_CKPT}" "${DATA_DIR}/best.ckpt"
echo "  Checkpoint: $(du -sh ${DATA_DIR}/best.ckpt)"

echo ""
echo "=== DOWNLOADING DATA ==="
gsutil -m cp "${GCS_DATA}/train_timeseries_static.feather"    "${DATA_DIR}/"
gsutil -m cp "${GCS_DATA}/test_timeseries_static.feather"     "${DATA_DIR}/"
gsutil -m cp "${GCS_DATA}/static_feature_list.json"           "${DATA_DIR}/"
gsutil -m cp "${GCS_DATA}/participant_static_features.parquet" "${DATA_DIR}/"
gsutil -m cp "${GCS_DATA}/cohort.csv"                         "${DATA_DIR}/"
gsutil cp    "${GCS_SPLIT}/test_personalization_windows.csv"   "${DATA_DIR}/"
echo "  Data ready:"
ls -lh "${DATA_DIR}/"

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ── Step 1: Full personalization (48h, 3 epochs) ───────────────────────────────
echo ""
echo "=== STEP 1/3: PERSONALIZATION (48h cap, 3 epochs) — ${START_UTC} ==="
"${PYTHON}" "${SCRIPTS_DIR}/personalize_exp_C_head.py" \
    --split test \
    --checkpoint  "${DATA_DIR}/best.ckpt" \
    --train       "${DATA_DIR}/train_timeseries_static.feather" \
    --feather-dir "${DATA_DIR}" \
    --split-dir   "${DATA_DIR}" \
    --static-feature-list "${DATA_DIR}/static_feature_list.json" \
    --output-dir  "${RESULTS_DIR}" \
    --adapt-epochs 3 \
    --unfreeze-mode output_layer \
    --overwrite \
    > "${PERS_LOG}" 2>&1
PERS_EXIT=$?
echo "=== PERSONALIZATION EXIT: ${PERS_EXIT} ==="

# ── Step 2: Adaptation sweep (0h / 6h / 12h / 24h / 48h) ─────────────────────
echo ""
echo "=== STEP 2/3: ADAPTATION SWEEP (0h/6h/12h/24h/48h) ==="
"${PYTHON}" "${SCRIPTS_DIR}/sweep_adaptation.py" \
    --split test \
    --python "${PYTHON}" \
    --personalize-script "${SCRIPTS_DIR}/personalize_exp_C_head.py" \
    --sweep-output-root  "${RESULTS_DIR}/sweep" \
    --checkpoint  "${DATA_DIR}/best.ckpt" \
    --train       "${DATA_DIR}/train_timeseries_static.feather" \
    --feather-dir "${DATA_DIR}" \
    --split-dir   "${DATA_DIR}" \
    --static-feature-list "${DATA_DIR}/static_feature_list.json" \
    --adapt-epochs 3 \
    --unfreeze-mode output_layer \
    --overwrite \
    > "${SWEEP_LOG}" 2>&1
SWEEP_EXIT=$?
echo "=== SWEEP EXIT: ${SWEEP_EXIT} ==="

# ── Step 3: Subgroup analysis ──────────────────────────────────────────────────
echo ""
echo "=== STEP 3/3: SUBGROUP ANALYSIS ==="
"${PYTHON}" "${SCRIPTS_DIR}/subgroup_analysis.py" \
    --split test \
    --metrics-dir    "${RESULTS_DIR}/test" \
    --static-features "${DATA_DIR}/participant_static_features.parquet" \
    --cohort          "${DATA_DIR}/cohort.csv" \
    --out-dir         "${RESULTS_DIR}/test" \
    > "${SUBGROUP_LOG}" 2>&1
SUBGROUP_EXIT=$?
echo "=== SUBGROUP EXIT: ${SUBGROUP_EXIT} ==="

echo ""
echo "=== ALL STEPS COMPLETE ==="
echo "  Personalization : exit ${PERS_EXIT}"
echo "  Sweep           : exit ${SWEEP_EXIT}"
echo "  Subgroup        : exit ${SUBGROUP_EXIT}"

# Fail the job if personalization (the core step) failed
exit "${PERS_EXIT}"
