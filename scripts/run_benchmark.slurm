#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --time=08:00:00
#SBATCH --time-min=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err
#SBATCH --partition=e7
#SBATCH --reservation=e7
#SBATCH --array=0-29

# Folds are handled by the SLURM array (0-29), storage is always disk.
# All other args (--models, --datasets, ...) are passed through via "$@".
#
# Usage:
#   sbatch scripts/run_benchmark.slurm -m catch22
#   sbatch scripts/run_benchmark.slurm -m rocket,minirocket,catch22
#   sbatch scripts/run_benchmark.slurm -m tscglue -d Worms

set -x
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONFAULTHANDLER=1

mkdir -p logs

if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    N_GPUS=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | grep -c .)
else
    N_GPUS=0
fi

uv run --no-sync python scripts/run_benchmark.py --storage disk -j $SLURM_CPUS_PER_TASK -g $N_GPUS -f $SLURM_ARRAY_TASK_ID "$@"
