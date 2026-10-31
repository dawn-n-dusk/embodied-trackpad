#!/bin/bash
# LEAP v1 Full bounded one-H800 runtime smoke launcher.
#
# This script intentionally contains NO site-specific values. Supply the
# scheduler identity at submit time through the environment or CLI, e.g.:
#   sbatch --account=<ACCOUNT> --partition=<PARTITION> tools/leap_v1_gpu_smoke.sbatch
# (or export SBATCH_ACCOUNT / SBATCH_PARTITION before calling sbatch).
#
# Resource bounds are part of the design: 1 GPU and 20 minutes, nothing more.
# CPUs and memory are deliberately LEFT AT SCHEDULER DEFAULTS for this task
# (historical --cpus-per-task=8/--mem values are NOT carried over); the actual
# allocation is recorded by the smoke run from SLURM_* environment variables.
#SBATCH --job-name=leap-v1-gpu-smoke
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Required environment (paths on the target cluster):
#   LEAP_V1_GPU_VENV    - virtualenv with torch 2.8 + mujoco 3.17 +
#                         mujoco-warp 3.17 + warp-lang (env/leap-v1-gpu-runtime.txt)
#   LEAP_V1_GPU_WORKDIR - deployed versioned bundle root containing
#                         tools/leap_v1_gpu_smoke.py, src/, assets/
# Optional:
#   LEAP_V1_GPU_SOURCE_COMMIT - git commit of the sources, recorded in output
#   LEAP_V1_GPU_STAGES        - default "1,32" (fixed smoke set; no expansion)
#   LEAP_V1_GPU_OUT           - default results/leap-v1-gpu-${SLURM_JOB_ID}.json
: "${LEAP_V1_GPU_VENV:?set LEAP_V1_GPU_VENV to the runtime virtualenv}"
: "${LEAP_V1_GPU_WORKDIR:?set LEAP_V1_GPU_WORKDIR to the deployed bundle root}"

source "${LEAP_V1_GPU_VENV}/bin/activate"
cd "${LEAP_V1_GPU_WORKDIR}"

# Task bound: at most 2 numerical threads, following the ACTUAL allocation;
# when SLURM_CPUS_PER_TASK is absent the default is 1 (never assume 2).
THREADS="${SLURM_CPUS_PER_TASK:-1}"
[ "${THREADS}" -gt 2 ] 2>/dev/null && THREADS=2
[ "${THREADS}" -lt 1 ] 2>/dev/null && THREADS=1
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

OUT="${LEAP_V1_GPU_OUT:-results/leap-v1-gpu-${SLURM_JOB_ID:-nominal}.json}"
echo "leap-v1-gpu-smoke: writing ${OUT}"
python tools/leap_v1_gpu_smoke.py --out "${OUT}"
