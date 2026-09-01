#!/bin/bash
# Timing sweep: TSCGlueEnhancedV4-LogLoss (Low/Medium/High) and MultiRocketHydra
# at 1/4/8/16 CPUs, plus a 16-CPU + GPU run for each. Fold 0, all UCR.
#
# Every job requests a GPU -- these nodes will not schedule without one -- so the
# CPU-only configs get TIMING_NO_GPU=1 and blank CUDA_VISIBLE_DEVICES in the job.
# Fixed --mem for every job: memory has to be constant across the sweep or it
# confounds the CPU-count comparison. Each config writes its own results dir.

COMMON=(--nodes=1 --nodelist=nsc-vfp003,nsc-vfp004 --gres=gpu:1 --mem=84G --time=72:00:00)
NOGPU="--export=ALL,TIMING_NO_GPU=1"
O=timing_results

sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=1  scripts/run_timing.slurm -c TSCGlueEnhancedV4-Low-LogLoss-GPU    -o $O/tscglue-low_1cpu
sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=4  scripts/run_timing.slurm -c TSCGlueEnhancedV4-Low-LogLoss-GPU    -o $O/tscglue-low_4cpu
sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=8  scripts/run_timing.slurm -c TSCGlueEnhancedV4-Low-LogLoss-GPU    -o $O/tscglue-low_8cpu
sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=16 scripts/run_timing.slurm -c TSCGlueEnhancedV4-Low-LogLoss-GPU    -o $O/tscglue-low_16cpu
sbatch "${COMMON[@]}"        --cpus-per-task=16 scripts/run_timing.slurm -c TSCGlueEnhancedV4-Low-LogLoss-GPU    -o $O/tscglue-low_16cpu_gpu

sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=1  scripts/run_timing.slurm -c TSCGlueEnhancedV4-Medium-LogLoss-GPU -o $O/tscglue-medium_1cpu
sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=4  scripts/run_timing.slurm -c TSCGlueEnhancedV4-Medium-LogLoss-GPU -o $O/tscglue-medium_4cpu
sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=8  scripts/run_timing.slurm -c TSCGlueEnhancedV4-Medium-LogLoss-GPU -o $O/tscglue-medium_8cpu
sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=16 scripts/run_timing.slurm -c TSCGlueEnhancedV4-Medium-LogLoss-GPU -o $O/tscglue-medium_16cpu
sbatch "${COMMON[@]}"        --cpus-per-task=16 scripts/run_timing.slurm -c TSCGlueEnhancedV4-Medium-LogLoss-GPU -o $O/tscglue-medium_16cpu_gpu

sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=1  scripts/run_timing.slurm -c TSCGlueEnhancedV4-High-LogLoss-GPU   -o $O/tscglue-high_1cpu
sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=4  scripts/run_timing.slurm -c TSCGlueEnhancedV4-High-LogLoss-GPU   -o $O/tscglue-high_4cpu
sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=8  scripts/run_timing.slurm -c TSCGlueEnhancedV4-High-LogLoss-GPU   -o $O/tscglue-high_8cpu
sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=16 scripts/run_timing.slurm -c TSCGlueEnhancedV4-High-LogLoss-GPU   -o $O/tscglue-high_16cpu
sbatch "${COMMON[@]}"        --cpus-per-task=16 scripts/run_timing.slurm -c TSCGlueEnhancedV4-High-LogLoss-GPU   -o $O/tscglue-high_16cpu_gpu

sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=1  scripts/run_timing.slurm -c MultiRocketHydra                     -o $O/mrhydra_1cpu
sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=4  scripts/run_timing.slurm -c MultiRocketHydra                     -o $O/mrhydra_4cpu
sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=8  scripts/run_timing.slurm -c MultiRocketHydra                     -o $O/mrhydra_8cpu
sbatch "${COMMON[@]}" $NOGPU --cpus-per-task=16 scripts/run_timing.slurm -c MultiRocketHydra                     -o $O/mrhydra_16cpu
sbatch "${COMMON[@]}"        --cpus-per-task=16 scripts/run_timing.slurm -c MultiRocketHydra                     -o $O/mrhydra_16cpu_gpu
