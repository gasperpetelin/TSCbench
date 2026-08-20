#!/bin/bash


# CPU baselines
sbatch --reservation=e7 --cpus-per-task=4 --mem=32G --array=0-1 --time=72:00:00 --time-min=01:00:00 scripts/run_benchmark2.slurm -c ROCKET
sbatch --reservation=e7 --cpus-per-task=4 --mem=32G --array=0-1 --time=72:00:00 --time-min=01:00:00 scripts/run_benchmark2.slurm -c MultiRocketHydra
sbatch --reservation=e7 --cpus-per-task=4 --mem=32G --array=0-1 --time=72:00:00 --time-min=01:00:00 scripts/run_benchmark2.slurm -c Catch22

sbatch --reservation=e7 --cpus-per-task=4 --mem=32G --array=0-1 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Low-Accuracy-GPU
sbatch --reservation=e7 --cpus-per-task=4 --mem=32G --array=0-1 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Medium-Accuracy-GPU
sbatch --reservation=e7 --cpus-per-task=4 --mem=32G --array=0-1 --time=72:00:00 --time-min=01:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-High-Accuracy-GPU

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
