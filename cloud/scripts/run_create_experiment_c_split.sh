#!/usr/bin/env bash
# run_create_experiment_c_split.sh
# ==================================
# Wrapper for create_experiment_c_split.py using the ssmcgm conda environment.
# All CLI arguments are forwarded, so you can do the adaptation-length sweep:
#
#   bash scripts/run_create_experiment_c_split.sh                     # 48 h (default)
#   bash scripts/run_create_experiment_c_split.sh --adaptation-h 6
#   bash scripts/run_create_experiment_c_split.sh --adaptation-h 12
#   bash scripts/run_create_experiment_c_split.sh --adaptation-h 24
#   bash scripts/run_create_experiment_c_split.sh --adaptation-h 48
#
# Each run saves to its own directory:
#   /home/myriamcharfeddine/CGM/Data/experiment_c_split_adapt{N}h_seed42/

set -e

PYTHON="/home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python"
SCRIPT="/home/myriamcharfeddine/CGM/Preprocess/create_experiment_c_split.py"

echo "========================================================"
echo " Experiment C Split Generator"
echo " $(date)"
echo " Args: $*"
echo "========================================================"

"$PYTHON" "$SCRIPT" "$@"
