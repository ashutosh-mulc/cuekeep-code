#!/usr/bin/env bash
# Launch an experiment detached on the GPU server, pinned to chosen GPUs, with logs captured.
#   GPUS=0 bash scripts/launch.sh experiments/E000-env-smoke/config.yaml [key=value ...] [--tag t]
# For a SLURM cluster use scripts/slurm_template.sbatch instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ $# -ge 1 ]] || { echo "usage: GPUS=0 $0 <config.yaml> [overrides...]" >&2; exit 1; }
CFG="$1"; shift
EXP="$(basename "$(dirname "$CFG")")"
mkdir -p "runs/$EXP/_launch"
LOG="runs/$EXP/_launch/$(date +%Y%m%d-%H%M%S).log"

if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  echo "WARNING: working tree is dirty; commit first so the run is reproducible from its SHA." >&2
fi

export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
export PYTHONUNBUFFERED=1
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

nohup python scripts/run.py "$CFG" "$@" >"$LOG" 2>&1 &
echo "pid $!  GPUs=$CUDA_VISIBLE_DEVICES  log=$LOG"
echo "follow: tail -f $LOG"
