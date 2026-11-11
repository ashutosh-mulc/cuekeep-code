#!/usr/bin/env bash
# One-time environment setup on the NVIDIA GPU server.
#   bash scripts/setup_server.sh            # creates .venv with uv (falls back to python -m venv)
#   CUDA_TAG=cu128 bash scripts/setup_server.sh
# Afterwards run `python scripts/env_check.py` and log the output in the day's notebook entry.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CUDA_TAG="${CUDA_TAG:-cu128}"          # match the server driver: nvidia-smi shows the max supported CUDA version
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
PY="${PY:-python3.11}"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: is this the GPU server?" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

if command -v uv >/dev/null; then
  uv venv --python "$PY" .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PIP="uv pip"
else
  "$PY" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install -U pip
  PIP="python -m pip"
fi

# torch first, from the CUDA-specific index, so later installs do not pull a CPU wheel.
$PIP install "torch==${TORCH_VERSION}" torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
$PIP install -e ".[gpu,eval,dev]"

# flash-attn is optional (SDPA works without it); building it needs nvcc and can take a while.
if [[ "${WITH_FLASH_ATTN:-0}" == "1" ]]; then
  $PIP install flash-attn --no-build-isolation
fi

# VLMEvalKit from source (pin the commit in third_party/README.md once E002 works).
if [[ "${WITH_VLMEVALKIT:-1}" == "1" ]]; then
  mkdir -p third_party/src
  [[ -d third_party/src/VLMEvalKit ]] || git clone https://github.com/open-compass/VLMEvalKit third_party/src/VLMEvalKit
  if [[ -n "${VLMEVALKIT_COMMIT:-}" ]]; then git -C third_party/src/VLMEvalKit checkout "$VLMEVALKIT_COMMIT"; fi
  $PIP install -e third_party/src/VLMEvalKit --no-deps
  echo "VLMEvalKit commit: $(git -C third_party/src/VLMEvalKit rev-parse --short HEAD)  <- record in third_party/README.md"
fi

python scripts/env_check.py
echo "Done. Activate with: source .venv/bin/activate"
