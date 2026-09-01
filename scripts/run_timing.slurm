#!/bin/bash
#SBATCH --job-name=timing
#SBATCH --output=logs/timing_%j.out
#SBATCH --error=logs/timing_%j.err

set -x
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONFAULTHANDLER=1

mkdir -p logs

# Every job asks for a GPU or it never gets scheduled on these nodes, so the
# CPU-only configs have to hide the one they were given.
if [ -n "$TIMING_NO_GPU" ]; then
    export CUDA_VISIBLE_DEVICES=
fi

# Belt and braces: OpenBLAS and numba size their pools from the node's full core
# count unless cgroups constrain the job's affinity mask. Redundant where the
# cluster does bind cores, wrong-free either way.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Confirm from the log that the allocation bound us to the cores asked for, and
# that the CPU-only configs really see no GPU -- if TIMING_NO_GPU failed to reach
# the job, this is the only place the mistake shows up before the timings are wrong.
uv run --no-sync python -c "import os, torch; print('AFFINITY', len(os.sched_getaffinity(0)), 'cores', sorted(os.sched_getaffinity(0))); print('GPUS_VISIBLE', torch.cuda.device_count())"

uv run --no-sync python scripts/run_benchmark2.py -j $SLURM_CPUS_PER_TASK -f 0 "$@"
