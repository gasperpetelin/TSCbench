#!/bin/bash

# --time is the *restart interval*, not the budget: run_benchmark2.slurm requeues
# itself 5 min before the limit and resumes at dataset granularity, so the only
# rule is that --time must clear the classifier's slowest single dataset.
# Measured worst case (fit+predict, from model_results): Catch22 0.01h,
# ROCKET 0.16h, MultiRocketHydra 0.18h, TSCGlueV4 Low/Med/High 1.8/3.2/5.0h,
# HIVECOTEV2 69.7h (ElectricDevices).

COMMON=(--reservation=e7 --cpus-per-task=4 --mem=32G --array=0-1 --time-min=01:00:00)

# CPU baselines
sbatch "${COMMON[@]}" --time=02:00:00 scripts/run_benchmark2.slurm -c ROCKET
sbatch "${COMMON[@]}" --time=02:00:00 scripts/run_benchmark2.slurm -c MultiRocketHydra
sbatch "${COMMON[@]}" --time=02:00:00 scripts/run_benchmark2.slurm -c Catch22
sbatch "${COMMON[@]}" --time=72:00:00 scripts/run_benchmark2.slurm -c HIVECOTEV2

sbatch "${COMMON[@]}" --time=08:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Low-Accuracy-GPU
sbatch "${COMMON[@]}" --time=08:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Medium-Accuracy-GPU
sbatch "${COMMON[@]}" --time=08:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-High-Accuracy-GPU

#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU
#
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Low-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Low-LogLoss-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Medium-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Medium-LogLoss-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-High-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-High-LogLoss-GPU
#
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU
#
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Low-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Low-LogLoss-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Medium-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Medium-LogLoss-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-High-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-High-LogLoss-GPU
#
#sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU
#sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU
#sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU
#
#sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Low-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Low-LogLoss-GPU
#sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Medium-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Medium-LogLoss-GPU
#sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-High-Accuracy-GPU
#sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-High-LogLoss-GPU
#
#sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=01:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c HIVECOTEV2,ROCKET,MiniRocket,MultiRocketHydra,Catch22
