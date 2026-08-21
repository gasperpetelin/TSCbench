#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=logs/benchmark2_%A_%a.out
#SBATCH --error=logs/benchmark2_%A_%a.err

set -x
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONFAULTHANDLER=1

mkdir -p logs

uv run --no-sync python scripts/run_benchmark2.py -j $SLURM_CPUS_PER_TASK -f $SLURM_ARRAY_TASK_ID "$@"
