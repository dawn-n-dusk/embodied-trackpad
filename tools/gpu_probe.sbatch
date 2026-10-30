#!/bin/bash
# Generic bounded single-GPU environment probe for Embodied Trackpad.
#
# This script intentionally contains NO site-specific values. Supply the
# scheduler identity at submit time through the environment or CLI, e.g.:
#   sbatch --account=<ACCOUNT> --partition=<PARTITION> tools/gpu_probe.sbatch
# (or export SBATCH_ACCOUNT / SBATCH_PARTITION before calling sbatch).
#
# Resource bounds are part of the probe design: 1 GPU, 4 CPUs, 20 minutes.
# Memory is left at the scheduler default.
#SBATCH --job-name=gpu-probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Required environment (paths on the target cluster):
#   PROBE_VENV    - virtualenv with the pinned probe requirements installed
#   PROBE_WORKDIR - probe working directory containing tools/gpu_probe.py
# Optional:
#   PROBE_SOURCE_COMMIT - git commit of the probe sources, recorded in output
#   PROBE_STEPS / PROBE_BENCH_STEPS / PROBE_NWORLDS - probe sizing
: "${PROBE_VENV:?set PROBE_VENV to the probe virtualenv}"
: "${PROBE_WORKDIR:?set PROBE_WORKDIR to the probe working directory}"

source "${PROBE_VENV}/bin/activate"
cd "${PROBE_WORKDIR}"

OUT="${PROBE_OUT:-results/probe-${SLURM_JOB_ID:-nominal}.json}"
echo "probe: writing ${OUT}"
python tools/gpu_probe.py --out "${OUT}"
