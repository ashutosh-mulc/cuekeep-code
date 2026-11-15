#!/usr/bin/env bash
# Launch an experiment detached on the GPU server, pinned to chosen GPUs, with logs captured.
#   GPUS=0 bash scripts/launch.sh experiments/E000-env-smoke/config.gate.yaml [key=value ...] [--tag t]
#   GPUS=0 SCRIPT=scripts/dump_visual_kv.py bash scripts/launch.sh configs/dump/gate.yaml
# For a SLURM cluster use scripts/slurm_template.sbatch instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ $# -ge 1 ]] || { echo "usage: GPUS=0 $0 <config.yaml> [overrides...]" >&2; exit 1; }
CFG="$1"; shift
SCRIPT="${SCRIPT:-scripts/run.py}"
EXP="$(basename "$(dirname "$CFG")")"
[[ "$SCRIPT" == *dump_visual_kv.py ]] && EXP="DUMP-$(basename "$CFG" .yaml)"
mkdir -p "runs/$EXP/_launch"
LOG="runs/$EXP/_launch/$(date +%Y%m%d-%H%M%S).log"

if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  echo "WARNING: working tree is dirty. The server must run a clean checkout of a pushed commit" >&2
  echo "         (git stash, then git pull); otherwise the run is not reproducible from its SHA." >&2
fi

export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
export PYTHONUNBUFFERED=1
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

nohup python "$SCRIPT" "$CFG" "$@" >"$LOG" 2>&1 &
echo "pid $!  GPUs=$CUDA_VISIBLE_DEVICES  log=$LOG"
echo "follow: tail -f $LOG"
