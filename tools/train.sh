#!/usr/bin/env bash
# train_reid.sh
#
# Usage:
#   ./train_reid.sh <GPU_ID> <model_name>
#
# Example:
#   ./train_reid.sh 0 osnet_ibn_x1_0
#
# The script sets the selected GPU, then launches torchpack-distributed
# training with the chosen model-specific config.

set -euo pipefail

# ---- Parse arguments --------------------------------------------------------
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <GPU_ID> <model_name>"
  exit 1
fi

GPU_ID="$1"
MODEL_NAME="$2"

# ---- Launch training --------------------------------------------------------
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
MASTER_ADDR=localhost \
torchpack dist-run -v -np 1 \
  python tools/train.py \
  "configs_reid/JeongokPort/training/training_${MODEL_NAME}.py" \
  --seed 66 \
  --run-dir runs/
