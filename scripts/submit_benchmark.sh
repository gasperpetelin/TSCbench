#!/bin/bash
# Submit benchmark jobs. Usage: bash scripts/submit_benchmark.sh [run_timing]
#
# run_timing defaults to 1, which also submits the CPU timing sweep. Pass 0 to submit
# only the accuracy jobs:
#   bash scripts/submit_benchmark.sh 0
RUN_TIMING=${1:-1}

mkdir -p logs

COMMON_OPTS="--cpus-per-task=4 --array=0-5 --time=72:00:00 --time-min=04:00:00"
CPU_OPTS="$COMMON_OPTS --exclude=compute01"
GPU_OPTS="$COMMON_OPTS --gres=gpu:1 --nodelist=compute01"

# CPU baselines
sbatch $CPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c ROCKET,MiniRocket,MultiRocketHydra,Catch22
sbatch $CPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c HIVECOTEV2,ROCKET,MiniRocket,MultiRocketHydra,Catch22

# TSCGlueEnhancedV2 accuracy — GPU, folds 0-2 (accuracy + log_loss cover all four metrics)
sbatch $GPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU
sbatch $GPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU
sbatch $GPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU
sbatch $GPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU
sbatch $GPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU
sbatch $GPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU

sbatch $CPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU
sbatch $CPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU
sbatch $CPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU
sbatch $CPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU
sbatch $CPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU
sbatch $CPU_OPTS --mem=32G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU

# Return again with more memory as some of the larger datasets are running out of memory with 32G
sbatch $GPU_OPTS --mem=64G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU
sbatch $GPU_OPTS --mem=64G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU
sbatch $GPU_OPTS --mem=64G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU
sbatch $GPU_OPTS --mem=64G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU
sbatch $GPU_OPTS --mem=64G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU
sbatch $GPU_OPTS --mem=64G scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU

# if [ "$RUN_TIMING" != "1" ]; then
#     echo "RUN_TIMING=$RUN_TIMING — skipping CPU timing sweep."
#     exit 0
# fi
# 
# # CPU timing, 1 core — fold 0, own -o per core count so nothing is skipped as already built
# sbatch --cpus-per-task=1 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_cpu_c1
# sbatch --cpus-per-task=1 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_cpu_c1
# sbatch --cpus-per-task=1 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_cpu_c1
# sbatch --cpus-per-task=1 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_cpu_c1
# sbatch --cpus-per-task=1 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_cpu_c1
# sbatch --cpus-per-task=1 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_cpu_c1
# 
# # CPU timing, 2 cores
# sbatch --cpus-per-task=2 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_cpu_c2
# sbatch --cpus-per-task=2 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_cpu_c2
# sbatch --cpus-per-task=2 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_cpu_c2
# sbatch --cpus-per-task=2 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_cpu_c2
# sbatch --cpus-per-task=2 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_cpu_c2
# sbatch --cpus-per-task=2 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_cpu_c2
# 
# # CPU timing, 4 cores
# sbatch --cpus-per-task=4 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_cpu_c4
# sbatch --cpus-per-task=4 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_cpu_c4
# sbatch --cpus-per-task=4 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_cpu_c4
# sbatch --cpus-per-task=4 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_cpu_c4
# sbatch --cpus-per-task=4 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_cpu_c4
# sbatch --cpus-per-task=4 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_cpu_c4
# 
# # CPU timing, 8 cores
# sbatch --cpus-per-task=8 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_cpu_c8
# sbatch --cpus-per-task=8 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_cpu_c8
# sbatch --cpus-per-task=8 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_cpu_c8
# sbatch --cpus-per-task=8 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_cpu_c8
# sbatch --cpus-per-task=8 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_cpu_c8
# sbatch --cpus-per-task=8 --mem=32G --array=0 --exclude=compute01 --time=72:00:00 --time-min=04:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
#     scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_cpu_c8
# 