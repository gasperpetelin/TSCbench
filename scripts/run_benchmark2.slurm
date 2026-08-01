#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/benchmark2_%A_%a.out
#SBATCH --error=logs/benchmark2_%A_%a.err
#SBATCH --array=0-3

# SLURM array task ID is used as the fold index.
# All other args (--classifiers, --datasets, ...) are passed through via "$@".
#
# Usage:
#   sbatch scripts/run_benchmark2.slurm -c ROCKET,MiniRocket
#   sbatch --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlue-Accuracy-GPU

set -x
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONFAULTHANDLER=1

mkdir -p logs

uv run --no-sync python scripts/run_benchmark2.py -j $SLURM_CPUS_PER_TASK -f $SLURM_ARRAY_TASK_ID "$@"
