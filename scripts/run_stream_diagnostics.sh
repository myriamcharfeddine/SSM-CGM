#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/myriamcharfeddine/CGM/SSM-CGM"
PY="/home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python3.10"
CONFIG="${CONFIG:-configs/aireadi_stream_full.yaml}"
CKPT="${CKPT:-outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/best_model_checkpoint.pt}"
EVAL_DIR="${EVAL_DIR:-outputs/aireadi_stream_mamba_stateful_10epoch_eval_test}"
OUT_DIR="${OUT_DIR:-${EVAL_DIR}/diagnostics}"
DEVICE="${DEVICE:-cuda}"

cd "$ROOT"
PYTHONUNBUFFERED=1 "$PY" scripts/evaluate_stream_diagnostics.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --eval-dir "$EVAL_DIR" \
  --device "$DEVICE" \
  --output-dir "$OUT_DIR" \
  --tasks persistence participant_metrics scenario_audit q01_hypo subgroup_plots

PYTHONUNBUFFERED=1 "$PY" scripts/evaluate_stream_diagnostics.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --eval-dir "$EVAL_DIR" \
  --device "$DEVICE" \
  --output-dir "$OUT_DIR" \
  --tasks matched_personalization
