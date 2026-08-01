#!/bin/bash
# Submit benchmark jobs.
# Usage: bash scripts/submit_benchmark.sh

CPUS=8                      # cores for the CPU baselines
GPU_CPUS=4                  # cores for the EnhancedV2 GPU runs
MEM="64G"
FOLDS="1-5"

# CPU timing sweep: same models, one job per core count, each to its own folder.
TIMING_CORES="1 2 4 8"
TIMING_FOLD=0
TIMING_DATASETS=""          # empty = all 112 UCR datasets
TIMING_TIME="72:00:00"

# Only accuracy and log_loss are worth running. At medium/high the served head for
# accuracy, f1 and roc_auc is the same (probability-stack-mean[-balanced]); at low it
# is f1, roc_auc and log_loss that share a head (probability-et). accuracy and log_loss
# therefore land in different groups at every preset and cover all four metrics — the
# other six preset x metric cells produce byte-identical predictions.
ENHANCED_V2=(
    TSCGlueEnhancedV2-Low-Accuracy-GPU
    TSCGlueEnhancedV2-Low-LogLoss-GPU
    TSCGlueEnhancedV2-Medium-Accuracy-GPU
    TSCGlueEnhancedV2-Medium-LogLoss-GPU
    TSCGlueEnhancedV2-High-Accuracy-GPU
    TSCGlueEnhancedV2-High-LogLoss-GPU
)

# SLURM opens the --output/--error paths before the job script runs, so the
# `mkdir -p logs` inside the .slurm files is too late — create it here.
mkdir -p logs

JOB_IDS=()
TIMING_JOB_IDS=()

# ---------------------------------------------------------------- CPU baselines ----
# rocket, minirocket, catch22 — CPU only
JOB_IDS+=($(sbatch --parsable --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS scripts/run_benchmark.slurm \
    -m rocket,minirocket,catch22))

# ROCKET, MiniRocket, MultiRocketHydra, Catch22 — CPU only
JOB_IDS+=($(sbatch --parsable --cpus-per-task=$CPUS --mem=$MEM --array=$FOLDS scripts/run_benchmark2.slurm \
    -c ROCKET,MiniRocket,MultiRocketHydra,Catch22))

# --------------------------------------------------- TSCGlueEnhancedV2 — accuracy ----
# GPU nodes, 1 GPU, $GPU_CPUS cores. One job per model rather than a single comma-list
# job, so a slow preset cannot drag the others past the walltime.
for MODEL in "${ENHANCED_V2[@]}"; do
    JOB_IDS+=($(sbatch --parsable --cpus-per-task=$GPU_CPUS --mem=$MEM --array=$FOLDS --gres=gpu:1 \
        scripts/run_benchmark2.slurm -c "$MODEL"))
done

# ----------------------------------------------------- TSCGlueEnhancedV2 — timing ----
# CPU nodes, no GPU. --exclusive keeps a noisy neighbour out of the measurements, and
# clearing CUDA_VISIBLE_DEVICES forces n_gpus=0 even if the job lands on a GPU node.
#
# Each core count needs its own -o: run_benchmark2.py runs with overwrite=False and
# tsml_eval only builds results that are not already present, so a shared folder would
# make the 2/4/8-core jobs skip everything the 1-core job wrote.
for N in $TIMING_CORES; do
    OUT="model_results_timing_cpu_c${N}"
    for MODEL in "${ENHANCED_V2[@]}"; do
        ARGS=(-c "$MODEL" -o "$OUT")
        [ -n "$TIMING_DATASETS" ] && ARGS+=(-d "$TIMING_DATASETS")
        TIMING_JOB_IDS+=($(sbatch --parsable --cpus-per-task=$N --mem=$MEM --array=$TIMING_FOLD \
            --exclusive --time=$TIMING_TIME --export=ALL,CUDA_VISIBLE_DEVICES= \
            scripts/run_benchmark2.slurm "${ARGS[@]}"))
    done
done

echo "Submitted benchmark jobs: ${JOB_IDS[*]}"
echo "Submitted CPU timing jobs: ${TIMING_JOB_IDS[*]}"

# Once every benchmark job above has finished (afterany: even if some tasks
# failed), plot the 4 CD diagrams from notebook 066 into generated_results/figures/.
# The timing jobs write to model_results_timing_cpu_c*, which the figures do not read,
# so they deliberately do not gate this.
DEP=$(IFS=:; echo "${JOB_IDS[*]}")
PLOT_JOB=$(sbatch --parsable --dependency=afterany:$DEP scripts/plot_figures.slurm)
echo "Plot job $PLOT_JOB waiting on: $DEP"
