#!/bin/bash

COMMONHC=(--cpus-per-task=1 --array=0-10)
COMMON0=(--reservation=e7 --cpus-per-task=2 --mem-per-cpu=2G --array=0-10 --time-min=01:00:00)
COMMON1=(--reservation=e7 --cpus-per-task=4 --mem-per-cpu=8G --array=0-10 --time-min=01:00:00)
COMMON2=(--reservation=e7 --cpus-per-task=24 --mem-per-cpu=8G --array=0-10 --time-min=01:00:00)

# CPU baselines
sbatch "${COMMONHC[@]}" --time=72:00:00 --time-min=06:00:00 --mem-per-cpu=6G scripts/run_benchmark2.slurm -c HIVECOTEV2
sbatch "${COMMONHC[@]}" --time=72:00:00 --time-min=06:00:00 --mem-per-cpu=6G --reservation=e7 scripts/run_benchmark2.slurm -c HIVECOTEV2
sleep 2
sbatch "${COMMONHC[@]}" --time=72:00:00 --time-min=24:00:00 --mem-per-cpu=6G scripts/run_benchmark2.slurm -c HIVECOTEV2
sbatch "${COMMONHC[@]}" --time=72:00:00 --time-min=24:00:00 --mem-per-cpu=6G --reservation=e7 scripts/run_benchmark2.slurm -c HIVECOTEV2
sleep 2
sbatch "${COMMONHC[@]}" --time=72:00:00 --mem-per-cpu=6G scripts/run_benchmark2.slurm -c HIVECOTEV2
sbatch "${COMMONHC[@]}" --time=72:00:00 --mem-per-cpu=6G --reservation=e7 scripts/run_benchmark2.slurm -c HIVECOTEV2
sleep 2
sbatch "${COMMONHC[@]}" --time=72:00:00 --mem-per-cpu=12G scripts/run_benchmark2.slurm -c HIVECOTEV2
sbatch "${COMMONHC[@]}" --time=72:00:00 --mem-per-cpu=12G --reservation=e7 scripts/run_benchmark2.slurm -c HIVECOTEV2
sleep 2
sbatch "${COMMON0[@]}" --time=04:00:00 scripts/run_benchmark2.slurm -c ROCKET
sbatch "${COMMON0[@]}" --time=04:00:00 scripts/run_benchmark2.slurm -c MultiRocketHydra
sbatch "${COMMON0[@]}" --time=04:00:00 scripts/run_benchmark2.slurm -c Catch22
sleep 2
sbatch "${COMMON1[@]}" --time=04:00:00 scripts/run_benchmark2.slurm -c ROCKET
sbatch "${COMMON1[@]}" --time=04:00:00 scripts/run_benchmark2.slurm -c MultiRocketHydra
sbatch "${COMMON1[@]}" --time=04:00:00 scripts/run_benchmark2.slurm -c Catch22
sleep 2
sbatch "${COMMON0[@]}" --time=24:00:00 scripts/run_benchmark2.slurm -c WEASEL_V2
sbatch "${COMMON0[@]}" --time=24:00:00 scripts/run_benchmark2.slurm -c FreshPRINCE
sbatch "${COMMON0[@]}" --time=24:00:00 scripts/run_benchmark2.slurm -c RDST
sbatch "${COMMON0[@]}" --time=04:00:00 scripts/run_benchmark2.slurm -c QUANT
sleep 2
sbatch "${COMMON1[@]}" --time=24:00:00 scripts/run_benchmark2.slurm -c WEASEL_V2
sbatch "${COMMON1[@]}" --time=24:00:00 scripts/run_benchmark2.slurm -c FreshPRINCE
sbatch "${COMMON1[@]}" --time=24:00:00 scripts/run_benchmark2.slurm -c RDST
sbatch "${COMMON1[@]}" --time=04:00:00 scripts/run_benchmark2.slurm -c QUANT
sleep 2
sbatch "${COMMON2[@]}" --time=08:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Low-Accuracy-GPU
sbatch "${COMMON2[@]}" --time=08:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Medium-Accuracy-GPU
sbatch "${COMMON2[@]}" --time=08:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-High-Accuracy-GPU
sleep 2
sbatch "${COMMON2[@]}" --time=08:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Low-LogLoss-GPU
sbatch "${COMMON2[@]}" --time=08:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-Medium-LogLoss-GPU
sbatch "${COMMON2[@]}" --time=08:00:00 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV4-High-LogLoss-GPU

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
