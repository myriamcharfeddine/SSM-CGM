#!/usr/bin/env bash
# submit_exp_C_pers_batch.sh
# ==========================
# Uploads all scripts + data to GCS, resolves the training checkpoint,
# then submits a Cloud Batch job that runs:
#   1. Personalization  — test split, 3 epochs, output_layer
#   2. Adaptation sweep — 0h / 6h / 12h / 24h / 48h
#   3. Subgroup analysis — HbA1c quartile, BMI, study group, meds, site
#
# Run AFTER the 30-epoch training job (exp-c-full-*) has completed.
#
# Prerequisites:
#   - Training job done; best.ckpt exists under gs://cgmproject2025/results/exp_C/
#   - Feathers already uploaded (done by submit_exp_C_full_batch.sh)
#   - gcloud authenticated, project set to cgmsy-467321
#
# Usage:
#   bash scripts/submit_exp_C_pers_batch.sh
#
# Outputs (on Cloud Batch completion):
#   gs://cgmproject2025/results/exp_C_pers/<task-uid>/results/
#     test/personalization_metrics_test.csv
#     test/personalization_summary_test.csv
#     test/predictions_global_test.parquet
#     test/predictions_personalized_test.parquet
#     test/subgroup_analysis_test_detail.csv
#     test/personalization_metrics_test_enriched.csv
#     sweep/sweep_adapt{0,6,12,24,48}h/test/personalization_metrics_test.csv
#     sweep/adaptation_sweep_summary.csv

set -euo pipefail

CGM_ROOT="/home/myriamcharfeddine/CGM"
GCS_SSMCGM="gs://cgmproject2025/scripts/SSMCGM"
GCS_DATA="${GCS_SSMCGM}/data_exp_C"
GCS_SPLIT="${GCS_SSMCGM}/split_exp_C"
GCS_CKPT_FIXED="gs://cgmproject2025/results/exp_C/best.ckpt"
LOCATION="us-central1"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_ID="exp-c-pers-${TIMESTAMP}"

echo "========================================================"
echo " Experiment C — Personalization + Sweep + Subgroup"
echo " Cloud Batch submission"
echo " Job ID  : ${JOB_ID}"
echo " Machine : a2-highgpu-1g (1×A100) SPOT"
echo " Split   : test (144 participants, 3 adapt epochs)"
echo " $(date)"
echo "========================================================"

# ── Step 1: Resolve checkpoint ─────────────────────────────────────────────────
echo ""
echo "[1/5] Resolving checkpoint..."
if gsutil -q stat "${GCS_CKPT_FIXED}" 2>/dev/null; then
    echo "  Found at fixed path: ${GCS_CKPT_FIXED}"
else
    echo "  Not at fixed path — searching training results..."
    LATEST_CKPT=$(gsutil ls "gs://cgmproject2025/results/exp_C/*/results/checkpoints/best.ckpt" 2>/dev/null | sort | tail -1)
    if [ -z "${LATEST_CKPT:-}" ]; then
        echo ""
        echo "ERROR: No checkpoint found. Make sure the training job has completed."
        echo "       Look under: gs://cgmproject2025/results/exp_C/"
        exit 1
    fi
    echo "  Found: ${LATEST_CKPT}"
    echo "  Copying to fixed path: ${GCS_CKPT_FIXED}"
    gsutil cp "${LATEST_CKPT}" "${GCS_CKPT_FIXED}"
    echo "  Done."
fi

# ── Step 2: Upload Python scripts ──────────────────────────────────────────────
echo ""
echo "[2/5] Uploading Python scripts..."
gsutil cp \
    "${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/personalize_exp_C_head.py" \
    "${GCS_SSMCGM}/personalize_exp_C_head.py"
gsutil cp \
    "${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/sweep_adaptation.py" \
    "${GCS_SSMCGM}/sweep_adaptation.py"
gsutil cp \
    "${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/subgroup_analysis.py" \
    "${GCS_SSMCGM}/subgroup_analysis.py"
gsutil cp \
    "${CGM_ROOT}/scripts/run_exp_C_pers_cloud.sh" \
    "${GCS_SSMCGM}/run_exp_C_pers_cloud.sh"
echo "  Uploaded to ${GCS_SSMCGM}/"

# ── Step 3: Upload split CSV ───────────────────────────────────────────────────
echo ""
echo "[3/5] Uploading split CSV..."
gsutil cp \
    "${CGM_ROOT}/Data/experiment_c_split_adapt48h_seed42/test_personalization_windows.csv" \
    "${GCS_SPLIT}/test_personalization_windows.csv"
echo "  Uploaded to ${GCS_SPLIT}/"

# ── Step 4: Upload static features + cohort (for subgroup analysis) ────────────
echo ""
echo "[4/5] Uploading static features + cohort..."
gsutil cp \
    "${CGM_ROOT}/Data/enriched_multimodal/participant_static_features.parquet" \
    "${GCS_DATA}/participant_static_features.parquet"
gsutil cp \
    "${CGM_ROOT}/Data/enriched_multimodal/cohort.csv" \
    "${GCS_DATA}/cohort.csv"
echo "  Uploaded to ${GCS_DATA}/"

# ── Step 5: Submit batch job ───────────────────────────────────────────────────
echo ""
echo "[5/5] Submitting Cloud Batch job ${JOB_ID}..."
gcloud batch jobs submit "${JOB_ID}" \
    --location="${LOCATION}" \
    --config="${CGM_ROOT}/scripts/exp_c_pers_batch_config.json"

echo ""
echo "========================================================"
echo " Job submitted: ${JOB_ID}"
echo ""
echo " Monitor:"
echo "   gcloud batch jobs describe ${JOB_ID} --location=${LOCATION}"
echo "   gcloud logging read 'resource.labels.job_id=${JOB_ID}' --limit=50"
echo ""
echo " Results (after completion):"
echo "   gs://cgmproject2025/results/exp_C_pers/<task-uid>/results/"
echo "   ├── test/personalization_metrics_test.csv"
echo "   ├── test/subgroup_analysis_test_detail.csv"
echo "   ├── test/personalization_metrics_test_enriched.csv"
echo "   └── sweep/adaptation_sweep_summary.csv"
echo "========================================================"
