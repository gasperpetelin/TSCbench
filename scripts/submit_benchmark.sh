#!/bin/bash
# Submit the accuracy benchmark jobs. Usage: bash scripts/submit_benchmark.sh
#
# The CPU/GPU core-scaling sweep lives in scripts/submit_timing.sh.
#
# No node pinning: the scheduler places these wherever there is room. The CPU lines
# blank CUDA_VISIBLE_DEVICES because they can land on a GPU node, and a node with
# two cards would otherwise trip the assert on n_gpus in tscglue.

mkdir -p logs

# CPU baselines
sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c ROCKET,MiniRocket,MultiRocketHydra,Catch22

sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU
sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU
sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU
sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU
sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU
sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU

sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU
sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU
sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU
sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU
sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU
sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU

sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=04:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU
sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=04:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU
sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=04:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU
sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=04:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU
sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=04:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU
sbatch --cpus-per-task=4 --mem=64G --array=0-5 --time=72:00:00 --time-min=04:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU

sbatch --cpus-per-task=4 --mem=32G --array=0-5 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= scripts/run_benchmark2.slurm -c HIVECOTEV2,ROCKET,MiniRocket,MultiRocketHydra,Catch22
