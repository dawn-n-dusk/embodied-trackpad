#!/bin/bash
# LEAP v2 Advanced bounded H800 validation launcher.
#
# This script intentionally contains NO site-specific values. Supply the
# scheduler identity at submit time through the environment or CLI, e.g.:
#   sbatch --account=<ACCOUNT> --partition=<PARTITION> tools/leap_h800_probe.sbatch
# (or export SBATCH_ACCOUNT / SBATCH_PARTITION before calling sbatch).
#
# Resource bounds are part of the design: 1 GPU, 8 CPUs, 20 minutes.
# Memory is left at the scheduler default.
#SBATCH --job-name=leap-h800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Required environment (paths on the target cluster):
#   LEAP_H800_VENV    - virtualenv with mujoco 3.17 + mujoco-warp 3.17 + warp-lang
#   LEAP_H800_WORKDIR - deployed versioned bundle root containing
#                       tools/leap_h800_probe.py, src/, assets/
# Optional:
#   LEAP_H800_SOURCE_COMMIT - git commit of the sources, recorded in output
#   LEAP_H800_STAGES        - default "1,32,128,512" (512 additionally gated)
#   LEAP_H800_OUT           - default results/leap-h800-${SLURM_JOB_ID}.json
: "${LEAP_H800_VENV:?set LEAP_H800_VENV to the probe virtualenv}"
: "${LEAP_H800_WORKDIR:?set LEAP_H800_WORKDIR to the deployed bundle root}"

source "${LEAP_H800_VENV}/bin/activate"
cd "${LEAP_H800_WORKDIR}"

# Allocation-time awareness for the stage-512 gate (>=5 min remaining).
export LEAP_H800_JOB_START_EPOCH="${LEAP_H800_JOB_START_EPOCH:-$(date +%s)}"
export LEAP_H800_JOB_WALL_S="${LEAP_H800_JOB_WALL_S:-1200}"

OUT="${LEAP_H800_OUT:-results/leap-h800-${SLURM_JOB_ID:-nominal}.json}"
echo "leap-h800: writing ${OUT}"
python tools/leap_h800_probe.py --out "${OUT}"
