#!/usr/bin/env bash
# run_exp_C_personalization_smoke.sh
# ====================================
# Smoke test for Experiment C head personalization.
#
# What it does:
#   Loads the best Exp C checkpoint (full or smoke), then for 5 test participants:
#     1. Evaluates the global model on their evaluation windows  (C0 baseline)
#     2. Fine-tunes only the output_layer for 1 epoch on their adaptation windows
#     3. Evaluates the personalized model on the same evaluation windows (C1)
#     4. Reports per-participant MAE / RMSE improvement
#
# Prerequisites:
#   - Experiment C training completed:
#       Data/results/exp_C_full/checkpoints/   (or exp_C_smoke as fallback)
#   - Experiment C data prepared:
#       Data/ssmcgm_ready_exp_C/
#   - Experiment C split generated:
#       Data/experiment_c_split_adapt48h_seed42/test_personalization_windows.csv
#
# Outputs (smoke):
#   Data/results/exp_C_personalization_smoke/test/
#     personalization_config.json
#     personalization_metrics_test.csv
#     personalization_summary_test.csv
#     skipped_participants_test.csv
#     predictions_global_test.parquet
#     predictions_personalized_test.parquet
#
# Usage:
#   bash scripts/run_exp_C_personalization_smoke.sh

set -e

CGM_ROOT="/home/myriamcharfeddine/CGM"
SCRIPT="${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/personalize_exp_C_head.py"

echo "========================================================"
echo " Experiment C — Head Personalization (SMOKE TEST)"
echo " $(date)"
echo "========================================================"
echo ""
echo "  Split           : test"
echo "  Participants    : 5 (smoke)"
echo "  Adapt epochs    : 1 (smoke)"
echo "  Unfreeze mode   : output_layer"
echo ""

# Default smoke run: test split, 5 participants, 1 epoch, output_layer only
conda run -n ssmcgm python "${SCRIPT}" \
    --split test \
    --smoke \
    --adapt-epochs 1 \
    --unfreeze-mode output_layer

echo ""
echo "========================================================"
echo " Smoke complete. Check results at:"
echo "   ${CGM_ROOT}/Data/results/exp_C_personalization_smoke/test/"
echo "========================================================"

# ── Full-run alternatives (uncomment to run) ───────────────────────────────────
#
# Full test-set personalization (3 epochs, output_layer only):
# conda run -n ssmcgm python "${SCRIPT}" \
#     --split test \
#     --adapt-epochs 3 \
#     --unfreeze-mode output_layer
#
# Full test-set with broader head unfreeze:
# conda run -n ssmcgm python "${SCRIPT}" \
#     --split test \
#     --adapt-epochs 3 \
#     --unfreeze-mode head
#
# Val-set personalization:
# conda run -n ssmcgm python "${SCRIPT}" \
#     --split val \
#     --adapt-epochs 3 \
#     --unfreeze-mode output_layer
#
# Point at a specific checkpoint:
# conda run -n ssmcgm python "${SCRIPT}" \
#     --split test \
#     --adapt-epochs 3 \
#     --checkpoint "${CGM_ROOT}/Data/results/exp_C_full/checkpoints/last.ckpt"
