#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${ROOT_DIR}/weights"
mkdir -p "${WEIGHTS_DIR}"

download_file() {
  local url="$1"
  local output="$2"
  if [ -e "${output}" ]; then
    echo "[skip] ${output}"
    return
  fi
  echo "[download] ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${output}"
  elif command -v wget >/dev/null 2>&1; then
    wget "${url}" -O "${output}"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
}

download_hf_file() {
  local repo_id="$1"
  local filename="$2"
  local output="$3"
  if [ -e "${output}" ]; then
    echo "[skip] ${output}"
    return
  fi
  echo "[download] ${repo_id}/${filename}"
  python - "$repo_id" "$filename" "$output" <<'PY'
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id, filename, output = sys.argv[1:4]
path = hf_hub_download(repo_id=repo_id, filename=filename)
Path(output).parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(path, output)
PY
}

if [ -n "${PANO_MODEL:-}" ]; then
  if [ ! -e "${PANO_MODEL}" ]; then
    echo "PANO_MODEL does not exist: ${PANO_MODEL}" >&2
    exit 1
  fi
  if [ ! -e "${WEIGHTS_DIR}/model.pt" ]; then
    ln -s "$(realpath "${PANO_MODEL}")" "${WEIGHTS_DIR}/model.pt"
  fi
else
  download_hf_file "YijingGuo/PanoVGGT" "model.pt" "${WEIGHTS_DIR}/model.pt"
fi

if [ "${DOWNLOAD_LOOP_WEIGHTS:-1}" = "1" ]; then
  download_file \
    "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt" \
    "${WEIGHTS_DIR}/dino_salad.ckpt"

  download_file \
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth" \
    "${WEIGHTS_DIR}/dinov2_vitb14_pretrain.pth"

  if [ ! -e "${WEIGHTS_DIR}/ORBvoc.txt" ]; then
    tmp="${WEIGHTS_DIR}/ORBvoc.txt.tar.gz"
    download_file \
      "https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz" \
      "${tmp}"
    tar -xzf "${tmp}" -C "${WEIGHTS_DIR}"
    rm -f "${tmp}"
  else
    echo "[skip] ${WEIGHTS_DIR}/ORBvoc.txt"
  fi
fi

echo "Weights are ready under ${WEIGHTS_DIR}"
