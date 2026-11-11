#!/usr/bin/env bash
# SLURM variant of scripts/launch.sh. Adjust partition/account to the cluster.
#   sbatch --export=ALL,CFG=experiments/E000-env-smoke/config.yaml scripts/slurm_template.sbatch
#SBATCH --job-name=cuekeep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python scripts/run.py "${CFG}" ${OVERRIDES:-} --tag "slurm${SLURM_JOB_ID}"
