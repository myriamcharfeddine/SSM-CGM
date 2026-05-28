#!/usr/bin/env bash
# run_adaptation_sweep.sh
# =======================
# Runs the adaptation-length sweep for Exp C personalization.
# Tests how much adaptation data (0h/6h/12h/24h/48h) is needed
# to improve over the global model.
#
# All runs use the same evaluation windows (48h split),
# varying only how much of the adaptation period is used for fine-tuning.
#
# Prerequisites:
#   - Exp C checkpoint exists: Data/results/exp_C_full/ (or exp_C_smoke/)
#   - ssmcgm_ready_exp_C feathers exist
#
# Outputs:
#   Data/results/exp_C_personalization/sweep_adapt{0,6,12,24,48}h/
#   Data/results/exp_C_personalization/adaptation_sweep_summary.csv
#
# Usage:
#   bash scripts/run_adaptation_sweep.sh
#   bash scripts/run_adaptation_sweep.sh --split val

set -e

CGM_ROOT="/home/myriamcharfeddine/CGM"
PYTHON="/home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python"
SCRIPT="${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/sweep_adaptation.py"

echo "========================================================"
echo " Experiment C — Adaptation Sweep"
echo " Lengths: 0h (C0 baseline) / 6h / 12h / 24h / 48h"
echo " $(date)"
echo "========================================================"

"${PYTHON}" "${SCRIPT}" "$@"

echo ""
echo "========================================================"
echo " Sweep complete. Summary:"
echo "   ${CGM_ROOT}/Data/results/exp_C_personalization/adaptation_sweep_summary.csv"
echo "========================================================"
