#!/usr/bin/env bash
# submit_exp_AB_48h_batch.sh
# ==========================
# Uploads Exp A & B scripts + data to GCS, then submits two Cloud Batch jobs
# for 30-epoch training with 48h (576-bin) context on a2-highgpu-4g (4×A100, STANDARD).
#
# Prerequisites:
#   - ssmcgm_ready feathers exist locally (Exp A: Data/ssmcgm_ready/, Exp B: Data/ssmcgm_ready_enriched/)
#   - gcloud authenticated, project set to cgmsy-467321
#
# Usage:
#   bash scripts/submit_exp_AB_48h_batch.sh
#
# Outputs (on Cloud Batch completion):
#   gs://cgmproject2025/results/exp_A_48h/<JOB_ID>/results/
#   gs://cgmproject2025/results/exp_B_48h/<JOB_ID>/results/

set -euo pipefail

CGM_ROOT="/home/myriamcharfeddine/CGM"
GCS_SSMCGM="gs://cgmproject2025/scripts/SSMCGM"
GCS_DATA_A="${GCS_SSMCGM}/data_exp_A"
GCS_DATA_B="${GCS_SSMCGM}/data_exp_B"
LOCATION="us-central1"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_ID_A="exp-a-48h-${TIMESTAMP}"
JOB_ID_B="exp-b-48h-${TIMESTAMP}"

echo "========================================================"
echo " Experiments A & B — 48h context — Cloud Batch submission"
echo " Job A : ${JOB_ID_A}"
echo " Job B : ${JOB_ID_B}"
echo " Machine: a2-highgpu-4g (4×A100) STANDARD"
echo " Context: 576 bins = 48h"
echo " Epochs : 30"
echo " $(date)"
echo "========================================================"

# ── Step 1: Upload Python training scripts ─────────────────────────────────────
echo ""
echo "[1/5] Uploading training scripts..."
gsutil cp \
    "${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/mamba288_local.py" \
    "${GCS_SSMCGM}/mamba288_local.py"
gsutil cp \
    "${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/mamba288_static_local.py" \
    "${GCS_SSMCGM}/mamba288_static_local.py"

# ── Step 2: Upload runner scripts ──────────────────────────────────────────────
echo ""
echo "[2/5] Uploading runner scripts..."
gsutil cp \
    "${CGM_ROOT}/scripts/run_exp_A_48h_cloud.sh" \
    "${GCS_SSMCGM}/run_exp_A_48h_cloud.sh"
gsutil cp \
    "${CGM_ROOT}/scripts/run_exp_B_48h_cloud.sh" \
    "${GCS_SSMCGM}/run_exp_B_48h_cloud.sh"

# ── Step 3: Upload Exp A data ──────────────────────────────────────────────────
echo ""
echo "[3/5] Uploading Exp A feathers (~170 MB)..."
gsutil -m cp \
    "${CGM_ROOT}/Data/ssmcgm_ready/train_timeseries.feather" \
    "${CGM_ROOT}/Data/ssmcgm_ready/test_timeseries.feather"  \
    "${GCS_DATA_A}/"
echo "  Uploaded to ${GCS_DATA_A}/"

# ── Step 4: Upload Exp B data ──────────────────────────────────────────────────
echo ""
echo "[4/5] Uploading Exp B feathers + feature list (~180 MB)..."
gsutil -m cp \
    "${CGM_ROOT}/Data/ssmcgm_ready_enriched/train_timeseries_static.feather" \
    "${CGM_ROOT}/Data/ssmcgm_ready_enriched/test_timeseries_static.feather"  \
    "${CGM_ROOT}/Data/ssmcgm_ready_enriched/static_feature_list.json"         \
    "${GCS_DATA_B}/"
echo "  Uploaded to ${GCS_DATA_B}/"

# ── Step 5: Submit both batch jobs ─────────────────────────────────────────────
echo ""
echo "[5/5] Submitting Cloud Batch jobs..."

echo "  Submitting Exp A (48h)..."
gcloud batch jobs submit "${JOB_ID_A}" \
    --location="${LOCATION}" \
    --config="${CGM_ROOT}/scripts/exp_a_48h_batch_config.json"

echo "  Submitting Exp B (48h)..."
gcloud batch jobs submit "${JOB_ID_B}" \
    --location="${LOCATION}" \
    --config="${CGM_ROOT}/scripts/exp_b_48h_batch_config.json"

echo ""
echo "========================================================"
echo " Jobs submitted:"
echo "   Exp A 48h: ${JOB_ID_A}"
echo "   Exp B 48h: ${JOB_ID_B}"
echo ""
echo " Monitor:"
echo "   gcloud batch jobs describe ${JOB_ID_A} --location=${LOCATION}"
echo "   gcloud batch jobs describe ${JOB_ID_B} --location=${LOCATION}"
echo "   gcloud logging read 'resource.labels.job_id=${JOB_ID_A}' --limit=50"
echo "   gcloud logging read 'resource.labels.job_id=${JOB_ID_B}' --limit=50"
echo ""
echo " Results (after completion):"
echo "   gs://cgmproject2025/results/exp_A_48h/<task-uid>/results/"
echo "   gs://cgmproject2025/results/exp_B_48h/<task-uid>/results/"
echo "========================================================"
