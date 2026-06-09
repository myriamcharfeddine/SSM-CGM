#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
CONFIG="${CONFIG:-${ROOT}/configs/aireadi_stream_full.yaml}"
DEVICE="${DEVICE:-auto}"

SMOKE_ARGS=()
for arg in "$@"; do
  if [[ "${arg}" == "--smoke" ]]; then
    SMOKE_ARGS=(--smoke)
  fi
done

"${PYTHON_BIN}" "${ROOT}/scripts/train_stream_aireadi.py" --config "${CONFIG}" --device "${DEVICE}" "$@"
"${PYTHON_BIN}" "${ROOT}/scripts/evaluate_stream_aireadi.py" --config "${CONFIG}" --device "${DEVICE}" "${SMOKE_ARGS[@]}"
"${PYTHON_BIN}" "${ROOT}/scripts/benchmark_stream_aireadi.py" --config "${CONFIG}" --device "${DEVICE}" "${SMOKE_ARGS[@]}"
