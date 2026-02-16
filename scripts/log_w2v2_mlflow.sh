#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/andres/ups-challenge/fairseq"
TB_DIR="${ROOT_DIR}/tb/w2v2_ups_ps"
CKPT_DIR="${ROOT_DIR}/checkpoints/w2v2_ups_ps"
EXPERIMENT="ups-w2v2"
RUN_NAME="w2v2_ups_ps"

uv run python "${ROOT_DIR}/scripts/log_run_to_mlflow.py" \
  --tb-dir "${TB_DIR}" \
  --ckpt-dir "${CKPT_DIR}" \
  --experiment "${EXPERIMENT}" \
  --run-name "${RUN_NAME}"
