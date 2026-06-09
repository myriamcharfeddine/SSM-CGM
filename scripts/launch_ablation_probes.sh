#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/myriamcharfeddine/CGM/SSM-CGM"
PY="/home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python3.10"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-3}"
MAX_PARTICIPANTS="${MAX_PARTICIPANTS:-200}"
OUT_ROOT="${OUT_ROOT:-outputs/ablations}"

cd "$ROOT"

run_probe() {
  local name="$1"
  local cfg="$2"
  echo "[ablation] running $name config=$cfg epochs=$EPOCHS max_participants=$MAX_PARTICIPANTS"
  PYTHONUNBUFFERED=1 "$PY" scripts/train_stream_aireadi.py \
    --config "$cfg" \
    --device "$DEVICE" \
    --train-mode stateful_stream \
    --epochs "$EPOCHS" \
    --max-participants "$MAX_PARTICIPANTS" \
    --output-dir "$OUT_ROOT/$name"
}

run_probe residual_current_full configs/ablations/aireadi_full.yaml
run_probe no_static_film configs/ablations/aireadi_no_static_film.yaml
run_probe no_decomposition configs/ablations/aireadi_no_decomposition.yaml

echo "[ablation] skipped absolute_target: current model supports residual_current only."
echo "[ablation] skipped no_scenario_masks: no config switch exists yet for mask removal."
