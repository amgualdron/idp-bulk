#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Per-task SLURM array script for this pipeline, CPU-only (Stokes has no GPUs).
# Do not submit this directly — use slurm/submit_batch_array.sh, which picks the
# --array range, the working directory and the output location for you.
#
# One array task == one row of job_manifest.csv == one simulation.py run:
# the array index (plus an offset for very large manifests, see below) is the
# task_id column generate_jobs.py wrote.
#
# Lines starting with "#SBATCH" are read by sbatch as if they were command-line
# flags ("##SBATCH" is a disabled one). They must come before the first real
# command, and any flag given on the sbatch command line overrides them.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=idp-md
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One core per task, same as the PI's runs: a single short chain gets nothing
# from extra threads, they would only burn core-hours against the monthly cap.
#SBATCH --cpus-per-task=1
# Walltime as D-HH:MM:SS. There is no checkpoint/restart in simulation.py, so a
# task killed for hitting this limit loses everything. 10 days is what the PI's
# Stokes jobs request; shorten it per batch once you have a calibrated runtime
# (shorter requests usually start sooner), e.g. `submit_batch_array.sh <run_dir>
# --time=3-00:00:00`.
#SBATCH --time=10-00:00:00
# Optional: pin every task to one CPU generation so timings are comparable (the
# PI's runs do this). Enable by removing one '#'.
##SBATCH --constraint=cascadelake

if [ -z "${1:-}" ] || [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Usage: slurm/submit_batch_array.sh <run_dir>  (submits this script as an array)" >&2
    exit 1
fi
RUN_DIR="$1"
# Manifests larger than the cluster's array size limit are submitted as several
# arrays, each with an offset to add to its array index (0 for the first).
TASK_ID=$(( ${IDP_TASK_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))

# ── Environment ────────────────────────────────────────────────────────────
# Override either of these at submit time without editing this file, e.g.
#   IDP_CONDA_ENV=/groups/<group>/my-envs/md-env slurm/submit_batch_array.sh <run_dir>
# The env must hold a CPU build of HOOMD v4+ (not the PI's hoomd_2.9.7 env —
# simulation.py uses the v3+ API — and not the gpu build from the workstation).
module load "${IDP_MODULE:-anaconda}"
eval "$(conda shell.bash hook)"
conda activate "${IDP_CONDA_ENV:-md-env}"

set -euo pipefail
export PYTHONUNBUFFERED=1  # so progress shows up in the .out file while running
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Host: $(hostname)   Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}   Manifest task: $TASK_ID   Started: $(date)"
echo "Python: $(which python)"
python -c "import hoomd; print('HOOMD', hoomd.version.version, hoomd.__file__)"

# The submit wrapper starts this job in the repo root (sbatch --chdir), so
# src/ and the run dir resolve exactly as they do for a local run.
python src/simulation.py \
    --manifest "$RUN_DIR/job_manifest.csv" \
    --task_id "$TASK_ID" \
    --device cpu

echo "Finished: $(date)"
