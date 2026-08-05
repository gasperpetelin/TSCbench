#!/bin/bash
# CPU/GPU scaling sweep: one fold of every model at 1, 2, 4 and 8 cores.
# Split out of submit_benchmark.sh, which now only submits the accuracy runs.
#
# Usage: bash scripts/submit_timing.sh
#
# Every core count writes to its own -o directory, so nothing is skipped as
# "already built" — re-timing the same work at each width is the whole point.

mkdir -p logs

COMMON_OPTS="--nodelist=nsc-vfp003,nsc-vfp004 --array=0 --mem=64G --time=72:00:00 --time-min=04:00:00"


# The CPU sets blank CUDA_VISIBLE_DEVICES so the models build with n_gpus=0 —
# these nodes have GPUs, so a job without --gres could otherwise still see them.
CPU_C1="$COMMON_OPTS --cpus-per-task=1 --export=ALL,CUDA_VISIBLE_DEVICES="
GPU_C1="$COMMON_OPTS --cpus-per-task=1 --gres=gpu:1"
CPU_C2="$COMMON_OPTS --cpus-per-task=2 --export=ALL,CUDA_VISIBLE_DEVICES="
GPU_C2="$COMMON_OPTS --cpus-per-task=2 --gres=gpu:1"
CPU_C4="$COMMON_OPTS --cpus-per-task=4 --export=ALL,CUDA_VISIBLE_DEVICES="
GPU_C4="$COMMON_OPTS --cpus-per-task=4 --gres=gpu:1"
CPU_C8="$COMMON_OPTS --cpus-per-task=8 --export=ALL,CUDA_VISIBLE_DEVICES="
GPU_C8="$COMMON_OPTS --cpus-per-task=8 --gres=gpu:1"

# ---- 1 core ----

sbatch $CPU_C1 scripts/run_benchmark2.slurm -c ROCKET -o model_results_timing_cpu_c1
sbatch $CPU_C1 scripts/run_benchmark2.slurm -c MiniRocket -o model_results_timing_cpu_c1
sbatch $CPU_C1 scripts/run_benchmark2.slurm -c MultiRocketHydra -o model_results_timing_cpu_c1
sbatch $CPU_C1 scripts/run_benchmark2.slurm -c Catch22 -o model_results_timing_cpu_c1
sbatch $CPU_C1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_cpu_c1
sbatch $CPU_C1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_cpu_c1
sbatch $CPU_C1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_cpu_c1
sbatch $CPU_C1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_cpu_c1
sbatch $CPU_C1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_cpu_c1
sbatch $CPU_C1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_cpu_c1

# Same TSCGlueEnhancedV2 configs with a GPU attached
sbatch $GPU_C1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_gpu_c1
sbatch $GPU_C1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_gpu_c1
sbatch $GPU_C1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_gpu_c1
sbatch $GPU_C1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_gpu_c1
sbatch $GPU_C1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_gpu_c1
sbatch $GPU_C1 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_gpu_c1

# ---- 2 cores ----

# CPU only
# sbatch $CPU_C2 scripts/run_benchmark2.slurm -c HIVECOTEV2 -o model_results_timing_cpu_c2
sbatch $CPU_C2 scripts/run_benchmark2.slurm -c ROCKET -o model_results_timing_cpu_c2
sbatch $CPU_C2 scripts/run_benchmark2.slurm -c MiniRocket -o model_results_timing_cpu_c2
sbatch $CPU_C2 scripts/run_benchmark2.slurm -c MultiRocketHydra -o model_results_timing_cpu_c2
sbatch $CPU_C2 scripts/run_benchmark2.slurm -c Catch22 -o model_results_timing_cpu_c2
sbatch $CPU_C2 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_cpu_c2
sbatch $CPU_C2 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_cpu_c2
sbatch $CPU_C2 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_cpu_c2
sbatch $CPU_C2 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_cpu_c2
sbatch $CPU_C2 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_cpu_c2
sbatch $CPU_C2 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_cpu_c2

# Same TSCGlueEnhancedV2 configs with a GPU attached
sbatch $GPU_C2 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_gpu_c2
sbatch $GPU_C2 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_gpu_c2
sbatch $GPU_C2 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_gpu_c2
sbatch $GPU_C2 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_gpu_c2
sbatch $GPU_C2 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_gpu_c2
sbatch $GPU_C2 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_gpu_c2

# ---- 4 cores ----

# CPU only
# sbatch $CPU_C4 scripts/run_benchmark2.slurm -c HIVECOTEV2 -o model_results_timing_cpu_c4
sbatch $CPU_C4 scripts/run_benchmark2.slurm -c ROCKET -o model_results_timing_cpu_c4
sbatch $CPU_C4 scripts/run_benchmark2.slurm -c MiniRocket -o model_results_timing_cpu_c4
sbatch $CPU_C4 scripts/run_benchmark2.slurm -c MultiRocketHydra -o model_results_timing_cpu_c4
sbatch $CPU_C4 scripts/run_benchmark2.slurm -c Catch22 -o model_results_timing_cpu_c4
sbatch $CPU_C4 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_cpu_c4
sbatch $CPU_C4 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_cpu_c4
sbatch $CPU_C4 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_cpu_c4
sbatch $CPU_C4 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_cpu_c4
sbatch $CPU_C4 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_cpu_c4
sbatch $CPU_C4 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_cpu_c4

# Same TSCGlueEnhancedV2 configs with a GPU attached
sbatch $GPU_C4 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_gpu_c4
sbatch $GPU_C4 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_gpu_c4
sbatch $GPU_C4 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_gpu_c4
sbatch $GPU_C4 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_gpu_c4
sbatch $GPU_C4 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_gpu_c4
sbatch $GPU_C4 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_gpu_c4

# ---- 8 cores ----

# CPU only
# sbatch $CPU_C8 scripts/run_benchmark2.slurm -c HIVECOTEV2 -o model_results_timing_cpu_c8
sbatch $CPU_C8 scripts/run_benchmark2.slurm -c ROCKET -o model_results_timing_cpu_c8
sbatch $CPU_C8 scripts/run_benchmark2.slurm -c MiniRocket -o model_results_timing_cpu_c8
sbatch $CPU_C8 scripts/run_benchmark2.slurm -c MultiRocketHydra -o model_results_timing_cpu_c8
sbatch $CPU_C8 scripts/run_benchmark2.slurm -c Catch22 -o model_results_timing_cpu_c8
sbatch $CPU_C8 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_cpu_c8
sbatch $CPU_C8 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_cpu_c8
sbatch $CPU_C8 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_cpu_c8
sbatch $CPU_C8 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_cpu_c8
sbatch $CPU_C8 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_cpu_c8
sbatch $CPU_C8 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_cpu_c8

# Same TSCGlueEnhancedV2 configs with a GPU attached
sbatch $GPU_C8 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-Accuracy-GPU -o model_results_timing_gpu_c8
sbatch $GPU_C8 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Low-LogLoss-GPU -o model_results_timing_gpu_c8
sbatch $GPU_C8 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-Accuracy-GPU -o model_results_timing_gpu_c8
sbatch $GPU_C8 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-Medium-LogLoss-GPU -o model_results_timing_gpu_c8
sbatch $GPU_C8 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-Accuracy-GPU -o model_results_timing_gpu_c8
sbatch $GPU_C8 scripts/run_benchmark2.slurm -c TSCGlueEnhancedV2-High-LogLoss-GPU -o model_results_timing_gpu_c8
