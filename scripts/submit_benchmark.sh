#!/bin/bash
# Submit benchmark jobs for all model groups.
# Datasets: 15 UCR datasets of varying sizes.
# Usage: bash scripts/submit_benchmark.sh

DATASETS="ArrowHead,Beef,Car,Coffee,Fish,GunPoint,ECG200,ItalyPowerDemand,SwedishLeaf,FaceAll,Wafer,FordA,ElectricDevices,Crop,NonInvasiveFatalECGThorax1"
CPUS=8
MEM="64G"
FOLDS="1-5"

# rocket, minirocket, catch22 — CPU only
sbatch --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS scripts/run_benchmark.slurm \
    -m rocket,minirocket,catch22 \
    -d "$DATASETS"

# tscglue — CPU only
sbatch --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS scripts/run_benchmark.slurm \
    -m tscglue \
    -d "$DATASETS"

# tscglue — 1 GPU
sbatch --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS --gres=gpu:1 scripts/run_benchmark.slurm \
    -m tscglue \
    -d "$DATASETS"
