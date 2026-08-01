#!/bin/bash
# Submit benchmark jobs for all model groups.
# Datasets: 21 UCR datasets of varying sizes.
# Usage: bash scripts/submit_benchmark.sh

CPUS=8
MEM="64G"
FOLDS="1-5"

JOB_IDS=()

# rocket, minirocket, catch22 — CPU only
JOB_IDS+=($(sbatch --parsable --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS scripts/run_benchmark.slurm \
    -m rocket,minirocket,catch22))

# tscglue — CPU only
# sbatch --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS scripts/run_benchmark.slurm \
#     -m tscglue \

# tscglue — 1 GPU
JOB_IDS+=($(sbatch --parsable --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS --gres=gpu:1 scripts/run_benchmark.slurm \
    -m tscglue))

# ROCKET, MiniRocket, MultiRocketHydra, Catch22 — CPU only
JOB_IDS+=($(sbatch --parsable --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS scripts/run_benchmark2.slurm \
    -c ROCKET,MiniRocket,MultiRocketHydra,Catch22))

# TSCGlue (Accuracy / LogLoss / ROCAUC) — 1 GPU each
JOB_IDS+=($(sbatch --parsable --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS --gres=gpu:1 scripts/run_benchmark2.slurm \
    -c TSCGlue-Accuracy-GPU,TSCGlue-LogLoss-GPU,TSCGlue-ROCAUC-GPU))

# TSCGlueMean / TSCGlueETAll — 1 GPU
JOB_IDS+=($(sbatch --parsable --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS --gres=gpu:1 scripts/run_benchmark2.slurm \
    -c TSCGlueMean-GPU,TSCGlueETAll-GPU))

echo "Submitted benchmark jobs: ${JOB_IDS[*]}"

# Once every benchmark job above has finished (afterany: even if some tasks
# failed), plot the 4 CD diagrams from notebook 066 into generated_results/figures/.
DEP=$(IFS=:; echo "${JOB_IDS[*]}")
PLOT_JOB=$(sbatch --parsable --dependency=afterany:$DEP scripts/plot_figures.slurm)
echo "Plot job $PLOT_JOB waiting on: $DEP"
