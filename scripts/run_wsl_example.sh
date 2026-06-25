#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_wsl_example.sh /mnt/c/path/to/panoramic_frames [config] [output_dir]" >&2
  exit 1
fi

source ~/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-panovggt-long}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

python panovggt_long.py \
  --image_dir "$1" \
  --config "${2:-configs/base_config.yaml}" \
  --exp_folder_name "${3:-./exps}"
