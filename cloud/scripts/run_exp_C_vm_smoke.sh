#!/usr/bin/env bash
# run_exp_C_vm_smoke.sh
# =====================
# VM smoke test for Experiment C (participant-level split, dynamic + static features).
#
# Runs the full 3-step pipeline end to end:
#   1. create_experiment_c_split.py   — participant split with 48h adaptation (default)
#   2. prepare_ssmcgm_data_experiment_c.py — build 3 feathers (train/val/test)
#   3. mamba288_static_participant_split_local.py --smoke --epochs 3
#                                     — ~200 participants, 3 epochs, GPU
#
# Usage:
#   bash scripts/run_exp_C_vm_smoke.sh
#
# Prerequisites:
#   - Experiment C split inputs exist:
#       Data/enriched_multimodal/cohort.csv
#       Data/enriched_multimodal/segments.csv
#       Data/enriched_multimodal/forecast_windows.csv
#   - ssmcgm conda environment is set up
#
# Outputs:
#   Data/experiment_c_split_adapt48h_seed42/       ← split CSVs + metadata
#   Data/ssmcgm_ready_exp_C/                       ← train/val/test feathers + feature metadata
#   Data/results/exp_C_smoke/                      ← checkpoint + results CSV

set -e   # stop on first error

CGM_ROOT="/home/myriamcharfeddine/CGM"

echo "========================================================"
echo " Experiment C — VM Smoke Test (participant-level split)"
echo " $(date)"
echo "========================================================"

# ── Step 1: Generate participant-level split ───────────────────────────────────
echo ""
echo "[Step 1/3] Running create_experiment_c_split.py (48h adaptation, seed 42)..."
echo "  Inputs : ${CGM_ROOT}/Data/enriched_multimodal/{cohort,segments,forecast_windows}.csv"
echo "  Output : ${CGM_ROOT}/Data/experiment_c_split_adapt48h_seed42/"
echo ""
conda run -n ssmcgm python \
    "${CGM_ROOT}/Preprocess/create_experiment_c_split.py" \
    --adaptation-h 48

# ── Step 2: Build train / val / test feathers ─────────────────────────────────
echo ""
echo "[Step 2/3] Running prepare_ssmcgm_data_experiment_c.py..."
echo "  Split  : ${CGM_ROOT}/Data/experiment_c_split_adapt48h_seed42/"
echo "  Output : ${CGM_ROOT}/Data/ssmcgm_ready_exp_C/"
echo ""
conda run -n ssmcgm python \
    "${CGM_ROOT}/Preprocess/prepare_ssmcgm_data_experiment_c.py" \
    --split-dir "${CGM_ROOT}/Data/experiment_c_split_adapt48h_seed42" \
    --out-dir   "${CGM_ROOT}/Data/ssmcgm_ready_exp_C"

# ── Step 3: Smoke training (3 epochs, ~200 participants) ───────────────────────
echo ""
echo "[Step 3/3] Running mamba288_static_participant_split_local.py --smoke --epochs 3..."
echo "  Train  : ${CGM_ROOT}/Data/ssmcgm_ready_exp_C/train_timeseries_static.feather"
echo "  Val    : ${CGM_ROOT}/Data/ssmcgm_ready_exp_C/val_timeseries_static.feather"
echo "  Test   : ${CGM_ROOT}/Data/ssmcgm_ready_exp_C/test_timeseries_static.feather"
echo "  Output : ${CGM_ROOT}/Data/results/exp_C_smoke/"
echo ""

# Default: rolling windows (predict=False), no participant_id embedding
conda run -n ssmcgm python \
    "${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/mamba288_static_participant_split_local.py" \
    --smoke \
    --epochs 3

# ── Ablation alternatives (uncomment one to test a specific configuration) ────
#
# Old-like behavior (participant_id embedding + last-window-only evaluation):
# conda run -n ssmcgm python \
#     "${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/mamba288_static_participant_split_local.py" \
#     --smoke --epochs 3 \
#     --include-participant-id-embedding --predict-last-only
#
# Test only participant_id embedding effect (rolling windows, pid in static_categoricals):
# conda run -n ssmcgm python \
#     "${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/mamba288_static_participant_split_local.py" \
#     --smoke --epochs 3 \
#     --include-participant-id-embedding
#
# Test only last-window effect (no pid embedding, predict=True):
# conda run -n ssmcgm python \
#     "${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/mamba288_static_participant_split_local.py" \
#     --smoke --epochs 3 \
#     --predict-last-only

echo ""
echo "========================================================"
echo " Smoke test complete — check:"
echo "   ${CGM_ROOT}/Data/results/exp_C_smoke/results_dynamic_static_participant_split.csv"
echo ""
echo " Compare experiments:"
echo "   Exp A (dynamic only, window split)  :"
echo "     ${CGM_ROOT}/Data/results/smoke/results_dynamic_only.csv"
echo "   Exp B (dynamic+static, window split):"
echo "     ${CGM_ROOT}/Data/results/exp_B_smoke/results_dynamic_static.csv"
echo "   Exp C (dynamic+static, pid split)   :"
echo "     ${CGM_ROOT}/Data/results/exp_C_smoke/results_dynamic_static_participant_split.csv"
echo "========================================================"
