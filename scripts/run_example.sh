#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_example.sh /path/to/panoramic_frames [config] [output_dir]" >&2
  exit 1
fi

IMAGE_DIR="$1"
CONFIG="${2:-configs/base_config.yaml}"
OUTPUT_DIR="${3:-./exps}"

python panovggt_long.py \
  --image_dir "${IMAGE_DIR}" \
  --config "${CONFIG}" \
  --exp_folder_name "${OUTPUT_DIR}"
