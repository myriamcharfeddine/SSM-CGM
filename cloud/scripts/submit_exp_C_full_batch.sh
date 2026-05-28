#!/usr/bin/env bash
# submit_exp_C_full_batch.sh
# ==========================
# Uploads Exp C scripts + data to GCS, then submits a Cloud Batch job
# for 12-epoch training on a2-highgpu-8g (8×A100, STANDARD).
#
# Prerequisites:
#   - ssmcgm_ready_exp_C feathers exist locally
#   - gcloud authenticated, project set to cgmsy-467321
#
# Usage:
#   bash scripts/submit_exp_C_full_batch.sh
#
# Outputs (on Cloud Batch completion):
#   gs://cgmproject2025/results/exp_C/<JOB_ID>/results/
#     checkpoints/  (one per epoch)
#     training.log
#     runtime_env.txt
#     experiment_c_run_config.json

set -euo pipefail

CGM_ROOT="/home/myriamcharfeddine/CGM"
GCS_SSMCGM="gs://cgmproject2025/scripts/SSMCGM"
GCS_DATA="${GCS_SSMCGM}/data_exp_C"
LOCATION="us-central1"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_ID="exp-c-full-${TIMESTAMP}"

echo "========================================================"
echo " Experiment C — Cloud Batch submission"
echo " Job ID  : ${JOB_ID}"
echo " Machine : a2-highgpu-4g (4×A100) STANDARD"
echo " Epochs  : 12  (limit_train_batches=20000 per epoch)"
echo " $(date)"
echo "========================================================"

# ── Step 1: Upload training script ────────────────────────────────────────────
echo ""
echo "[1/3] Uploading training script..."
gsutil cp \
    "${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/mamba288_static_participant_split_local.py" \
    "${GCS_SSMCGM}/mamba288_static_participant_split_local.py"

# ── Step 2: Upload runner script ──────────────────────────────────────────────
gsutil cp \
    "${CGM_ROOT}/scripts/run_exp_C_full_cloud.sh" \
    "${GCS_SSMCGM}/run_exp_C_full_cloud.sh"

# ── Step 3: Upload data ───────────────────────────────────────────────────────
echo ""
echo "[2/3] Uploading feathers + feature list (~280 MB)..."
gsutil -m cp \
    "${CGM_ROOT}/Data/ssmcgm_ready_exp_C/train_timeseries_static.feather" \
    "${CGM_ROOT}/Data/ssmcgm_ready_exp_C/val_timeseries_static.feather"   \
    "${CGM_ROOT}/Data/ssmcgm_ready_exp_C/test_timeseries_static.feather"  \
    "${CGM_ROOT}/Data/ssmcgm_ready_exp_C/static_feature_list.json"         \
    "${GCS_DATA}/"
echo "  Uploaded to ${GCS_DATA}/"

# ── Step 4: Submit batch job ──────────────────────────────────────────────────
echo ""
echo "[3/3] Submitting Cloud Batch job ${JOB_ID}..."
gcloud batch jobs submit "${JOB_ID}" \
    --location="${LOCATION}" \
    --config="${CGM_ROOT}/gcloud_batch/exp_c_full_batch_config.json"

echo ""
echo "========================================================"
echo " Job submitted: ${JOB_ID}"
echo ""
echo " Monitor:"
echo "   gcloud batch jobs describe ${JOB_ID} --location=${LOCATION}"
echo "   gcloud logging read 'resource.labels.job_id=${JOB_ID}' --limit=50"
echo ""
echo " Results (after completion):"
echo "   gs://cgmproject2025/results/exp_C/<task-uid>/results/"
echo "========================================================"
