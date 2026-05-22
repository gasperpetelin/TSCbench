#!/bin/bash
# Submit benchmark jobs for all model groups.
# Datasets: 5 small UCR datasets.
# Usage: bash scripts/submit_benchmark.sh

DATASETS="ArrowHead,Beef,Car,Coffee,Fish"
CPUS=8
FOLDS="1-5"

# rocket, minirocket, catch22 — CPU only
sbatch --cpus-per-task=$CPUS --array=$FOLDS scripts/run_benchmark.slurm \
    -m rocket,minirocket,catch22 \
    -d "$DATASETS"

# tscglue — CPU only
sbatch --cpus-per-task=$CPUS --array=$FOLDS scripts/run_benchmark.slurm \
    -m tscglue \
    -d "$DATASETS"

# tscglue — 1 GPU
sbatch --cpus-per-task=$CPUS --array=$FOLDS --gres=gpu:1 scripts/run_benchmark.slurm \
    -m tscglue \
    -d "$DATASETS"
