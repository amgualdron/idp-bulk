#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Submits one SLURM array job covering every unfinished row of a run's
# job_manifest.csv (produced by `src/generate_jobs.py`). This is the
# cluster-side counterpart to src/run_local.py: where run_local.py loops the
# manifest in-process one task at a time, this submits all tasks as an array
# job that SLURM schedules in parallel across available nodes.
#
# Run it on the login node, from anywhere:
#   slurm/submit_batch_array.sh <run_dir> [throttle] [extra sbatch flags...]
#
#   <run_dir>   runs/<name>_<MMDD> directory already containing job_manifest.csv
#   [throttle]  optional cap on array tasks running at once, e.g. 50 (applies
#               per array when a large manifest is split into several)
#   [extra...]  passed to sbatch as-is; they override the #SBATCH defaults in
#               submit_array.sh, e.g. --time=3-00:00:00 or --constraint=cascadelake
#
# Only tasks whose logs/*.json is not "completed" are submitted, so re-running
# this after some tasks failed or timed out resubmits just those. Don't re-run
# it while the previous array is still in the queue (check with `squeue -u
# $USER`) — tasks that are still running would be submitted a second time.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <run_dir> [throttle] [extra sbatch flags...]" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_DIR_ABS="$(realpath "$1")"
shift

THROTTLE=""
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    THROTTLE="$1"
    shift
fi

MANIFEST="$RUN_DIR_ABS/job_manifest.csv"
if [ ! -f "$MANIFEST" ]; then
    echo "Manifest not found: $MANIFEST (run src/generate_jobs.py first)" >&2
    exit 1
fi

N_TASKS=$(($(wc -l < "$MANIFEST") - 1))  # minus header row
if [ "$N_TASKS" -lt 1 ]; then
    echo "Manifest has no job rows: $MANIFEST" >&2
    exit 1
fi

# Collect task ids whose log does not say "completed" (missing log counts too).
PENDING=()
for id in $(seq 1 "$N_TASKS"); do
    log=("$RUN_DIR_ABS"/logs/*_task_"$id".json)
    if ! grep -qs '"status": "completed"' "${log[0]}"; then
        PENDING+=("$id")
    fi
done

if [ "${#PENDING[@]}" -eq 0 ]; then
    echo "All $N_TASKS tasks in $RUN_DIR_ABS are already completed — nothing to submit."
    exit 0
fi

# Turns a sorted id list into sbatch's array syntax: 1 2 3 7 -> 1-3,7
compress_ranges() {
    local out="" start="$1" prev="$1" n
    shift
    for n in "$@" ""; do
        if [ -n "$n" ] && [ "$n" -eq $((prev + 1)) ]; then
            prev="$n"
            continue
        fi
        out+="${out:+,}${start}"
        if [ "$prev" -ne "$start" ]; then out+="-${prev}"; fi
        start="$n"
        prev="$n"
    done
    echo "$out"
}

# SLURM will not create the output directory itself; a job whose --output
# directory is missing dies immediately without leaving any message.
mkdir -p "$RUN_DIR_ABS/slurm_logs"

# Hand the job a run dir relative to the repo root, so the trajectory paths
# simulation.py records in logs/*.json look the same as for a local run.
RUN_DIR_REL="$(realpath --relative-to="$REPO_ROOT" "$RUN_DIR_ABS")"

# A cluster only accepts array indices below its MaxArraySize (1001 unless the
# admins changed it), so a manifest with more rows than that goes in as several
# arrays. Each one carries an offset that submit_array.sh adds back on:
# manifest task_id = IDP_TASK_OFFSET + array index.
MAX_ARRAY="$(scontrol show config 2>/dev/null | awk '/^MaxArraySize/ {print $3}' || true)"
CHUNK=$(( ${MAX_ARRAY:-1001} - 1 ))

echo "Submitting ${#PENDING[@]} of $N_TASKS tasks for $RUN_DIR_REL"
for offset in $(seq 0 "$CHUNK" $((N_TASKS - 1))); do
    LOCAL_IDS=()
    for id in "${PENDING[@]}"; do
        if [ "$id" -gt "$offset" ] && [ "$id" -le $((offset + CHUNK)) ]; then
            LOCAL_IDS+=($((id - offset)))
        fi
    done
    if [ "${#LOCAL_IDS[@]}" -eq 0 ]; then
        continue
    fi

    ARRAY_SPEC="$(compress_ranges "${LOCAL_IDS[@]}")"
    if [ -n "$THROTTLE" ]; then
        ARRAY_SPEC="${ARRAY_SPEC}%${THROTTLE}"
    fi

    # %x = job name, %A = array job id, %a = array index (== manifest task_id
    # for the first array; later arrays say which offset to add in the name)
    OUT_NAME="%x_%A_%a.out"
    if [ "$offset" -gt 0 ]; then
        OUT_NAME="%x_%A_%a_plus${offset}.out"
    fi

    echo "  tasks $((offset + LOCAL_IDS[0]))..$((offset + LOCAL_IDS[-1])): array=$ARRAY_SPEC"
    sbatch \
        --array="$ARRAY_SPEC" \
        --chdir="$REPO_ROOT" \
        --output="$RUN_DIR_ABS/slurm_logs/$OUT_NAME" \
        --export="ALL,IDP_TASK_OFFSET=$offset" \
        "$@" \
        "$SCRIPT_DIR/submit_array.sh" "$RUN_DIR_REL"
done
