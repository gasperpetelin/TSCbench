#!/bin/bash
# Submit benchmark jobs. Usage: bash scripts/submit_benchmark.sh

mkdir -p logs

# CPU baselines
sbatch --cpus-per-task=8 --mem=64G --array=1-5 scripts/run_benchmark.slurm -m rocket,minirocket,catch22
sbatch --cpus-per-task=8 --mem=64G --array=1-5 scripts/run_benchmark2.slurm -c ROCKET,MiniRocket,MultiRocketHydra,Catch22

# TSCGlueEnhancedV2 accuracy — GPU, folds 1-5 (accuracy + log_loss cover all four metrics)
sbatch --cpus-per-task=4 --mem=64G --array=1-5 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU
sbatch --cpus-per-task=4 --mem=64G --array=1-5 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU
sbatch --cpus-per-task=4 --mem=64G --array=1-5 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU
sbatch --cpus-per-task=4 --mem=64G --array=1-5 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU
sbatch --cpus-per-task=4 --mem=64G --array=1-5 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU
sbatch --cpus-per-task=4 --mem=64G --array=1-5 --gres=gpu:1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU

# CPU timing, 1 core — fold 0, own -o per core count so nothing is skipped as already built
sbatch --cpus-per-task=1 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_cpu_c1
sbatch --cpus-per-task=1 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_cpu_c1
sbatch --cpus-per-task=1 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_cpu_c1
sbatch --cpus-per-task=1 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_cpu_c1
sbatch --cpus-per-task=1 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_cpu_c1
sbatch --cpus-per-task=1 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_cpu_c1

# CPU timing, 2 cores
sbatch --cpus-per-task=2 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_cpu_c2
sbatch --cpus-per-task=2 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_cpu_c2
sbatch --cpus-per-task=2 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_cpu_c2
sbatch --cpus-per-task=2 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_cpu_c2
sbatch --cpus-per-task=2 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_cpu_c2
sbatch --cpus-per-task=2 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_cpu_c2

# CPU timing, 4 cores
sbatch --cpus-per-task=4 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_cpu_c4
sbatch --cpus-per-task=4 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_cpu_c4
sbatch --cpus-per-task=4 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_cpu_c4
sbatch --cpus-per-task=4 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_cpu_c4
sbatch --cpus-per-task=4 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_cpu_c4
sbatch --cpus-per-task=4 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_cpu_c4

# CPU timing, 8 cores
sbatch --cpus-per-task=8 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_cpu_c8
sbatch --cpus-per-task=8 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_cpu_c8
sbatch --cpus-per-task=8 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_cpu_c8
sbatch --cpus-per-task=8 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_cpu_c8
sbatch --cpus-per-task=8 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_cpu_c8
sbatch --cpus-per-task=8 --mem=64G --array=0 --time=72:00:00 --export=ALL,CUDA_VISIBLE_DEVICES= \
    scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_cpu_c8
