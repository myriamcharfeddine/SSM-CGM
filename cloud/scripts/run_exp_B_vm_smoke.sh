#!/usr/bin/env bash
# run_exp_B_vm_smoke.sh
# =====================
# VM smoke test for Experiment B (dynamic + static features).
# Runs 200 participants × 3 epochs to verify the pipeline before a full run.
#
# Usage:
#   bash scripts/run_exp_B_vm_smoke.sh
#
# Prerequisites:
#   - Experiment A outputs exist (Data/ssmcgm_ready/ feathers)
#   - ssmcgm conda environment is set up
#
# Output:
#   Data/ssmcgm_ready_enriched/   ← enriched feathers + JSON metadata
#   Data/results/exp_B_smoke/     ← checkpoint + results_dynamic_static.csv

set -e   # stop on first error

CGM_ROOT="/home/myriamcharfeddine/CGM"
PYTHON="/home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python"

echo "========================================================"
echo " Experiment B — VM Smoke Test"
echo " $(date)"
echo "========================================================"

# ── Step 1: Data preparation ──────────────────────────────────────────────────
echo ""
echo "[Step 1/2] Running prepare_ssmcgm_data_enriched.py..."
echo "  Input : ${CGM_ROOT}/Data/enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet"
echo "  Output: ${CGM_ROOT}/Data/ssmcgm_ready_enriched/"
echo ""
"$PYTHON" "${CGM_ROOT}/Preprocess/prepare_ssmcgm_data_enriched.py"

# ── Step 2: Smoke training ────────────────────────────────────────────────────
echo ""
echo "[Step 2/2] Running mamba288_static_local.py --smoke (200 pids, 3 epochs)..."
echo "  Input : ${CGM_ROOT}/Data/ssmcgm_ready_enriched/train_timeseries_static.feather"
echo "  Output: ${CGM_ROOT}/Data/results/exp_B_smoke/"
echo ""
"$PYTHON" "${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/mamba288_static_local.py" --smoke

echo ""
echo "========================================================"
echo " Smoke test complete — check loss curve and:"
echo "   ${CGM_ROOT}/Data/results/exp_B_smoke/results_dynamic_static.csv"
echo ""
echo " Next: compare Exp A vs Exp B metrics:"
echo "   Exp A: ${CGM_ROOT}/Data/results/smoke/results_dynamic_only.csv"
echo "   Exp B: ${CGM_ROOT}/Data/results/exp_B_smoke/results_dynamic_static.csv"
echo "========================================================"
