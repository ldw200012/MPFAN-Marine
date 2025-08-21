#!/usr/bin/env bash
# test_reid.sh
#
# Usage:
#   ./test_reid.sh <GPU_ID> <model_name> <checkpoint_name> <port_name>
#
# Example:
#   ./test_reid.sh 0 osnet_ibn_x1_0 epoch_120 JeongokPort
#
# The script sets the GPU, then launches torchpack-distributed
# evaluation with the chosen model config and checkpoint.

set -euo pipefail

# ---- Parse arguments --------------------------------------------------------
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <GPU_ID> <model_name> <checkpoint_name> <data_name>"
  exit 1
fi

GPU_ID="$1"
MODEL_NAME="$2"
CKPT_NAME="$3"
DATA_NAME="$4"

# ---- Launch evaluation ------------------------------------------------------
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
MASTER_ADDR=localhost \
torchpack dist-run -v -np 1 \
  python tools/train.py \
  "configs_reid/${DATA_NAME}/testing/testing_${MODEL_NAME}.py" \
  --checkpoint "weights/${CKPT_NAME}.pth"

# ./test_reid.sh 0 pointnet epoch_120 JeongokPort