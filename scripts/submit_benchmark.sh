#!/bin/bash
# Submit benchmark jobs for all model groups.
# Datasets: 21 UCR datasets of varying sizes.
# Usage: bash scripts/submit_benchmark.sh

CPUS=8
MEM="64G"
FOLDS="1-5"

# rocket, minirocket, catch22 — CPU only
sbatch --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS scripts/run_benchmark.slurm \
    -m rocket,minirocket,catch22 \

# tscglue — CPU only
# sbatch --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS scripts/run_benchmark.slurm \
#     -m tscglue \

# tscglue — 1 GPU
sbatch --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS --gres=gpu:1 scripts/run_benchmark.slurm \
    -m tscglue \

# ROCKET, MiniRocket, MultiRocketHydra, Catch22 — CPU only
sbatch --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS scripts/run_benchmark2.slurm \
    -c ROCKET,MiniRocket,MultiRocketHydra,Catch22 \

# TSCGlue (Accuracy / LogLoss / ROCAUC) — 1 GPU each
sbatch --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS --gres=gpu:1 scripts/run_benchmark2.slurm \
    -c TSCGlue-Accuracy-GPU,TSCGlue-LogLoss-GPU,TSCGlue-ROCAUC-GPU \
