# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline for running coarse-grained (HPS/Mpipi-style, one-bead-per-residue) molecular
dynamics simulations of intrinsically disordered proteins (IDPs) in HOOMD-blue, extracting
structural observables (Rg, Ree, asphericity) from the resulting trajectories, and pooling
runs per sequence. The repo holds only the pipeline: run outputs (`runs/`) are git-ignored, and
notebooks, plots and ML dataset building live elsewhere.

## Environment

Python code runs in the conda env `md-env` (`/home/andres/miniforge3/envs/md-env`), which has
`hoomd`, `MDAnalysis`, `jax`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `tqdm`,
`scikit-learn`, `pyyaml`. There is no `environment.yml`/`requirements.txt` checked in — the README
has the `conda create` commands. Use `conda activate md-env` before running anything, or invoke
the env's Python directly. All commands assume the repo root as working directory.

`src/simulation.py` defaults to GPU (`--device gpu`, matching the author's workstation) but also
accepts `--device cpu` (on HOOMD v4, reads `SLURM_CPUS_PER_TASK` for the TBB thread count when set; v5 removed TBB
and is always single-threaded) and
`--device auto` (tries GPU, falls back to CPU). The local conda env's `hoomd` build here is the
`gpu` conda-forge variant; a CPU-only cluster should get its own env built from the `cpu` variant
instead of transplanting this one.

## Running on a SLURM cluster

`slurm/submit_batch_array.sh <run_dir> [throttle] [extra sbatch flags]` submits one SLURM array
job for an existing `job_manifest.csv` (produced by `generate_jobs.py` same as for a local run),
running `slurm/submit_array.sh` once per `task_id` — mirroring what `run_local.py` does
sequentially for a local run. It only submits tasks whose `logs/*.json` is not `"completed"`, so
re-running it resubmits just the failed/timed-out ones. Manifests longer than the cluster's
`MaxArraySize` are split into several arrays, each passing `IDP_TASK_OFFSET` so that
`task_id = offset + $SLURM_ARRAY_TASK_ID`. The wrapper sets `--chdir` to the repo
root and passes the run dir relative to it (sbatch copies the job script to a spool dir, so the
job script cannot locate the repo from its own path), and sends SLURM stdout/stderr to
`<run_dir>/slurm_logs/`. The conda module/env are overridable via `IDP_MODULE` /
`IDP_CONDA_ENV` (defaults `anaconda` / `md-env`); resources default to 1 core and a 10-day
walltime, modelled on a PI's working Stokes (UCF) job script but not yet run on the cluster. There is no checkpoint/restart in `simulation.py`: a task killed by hitting `--time`
loses all progress and must restart from scratch, so size `--time` from the longest sequence in
the batch, not the average, and calibrate CPU throughput with one real job before committing to
a walltime for a large batch (local GPU runs of the benchmark sets took ~34-42 h each).

There is no test suite, linter, or formatter configured in this repo.

## Pipeline / architecture

Everything flows through **run directories** under `runs/`, each one a self-contained record of
one experiment batch:

```
config/config.yaml          → control panel: which sequences, how many replicates, which
                               param_sets, runner (local|slurm)
config/benchmark_sequences.yaml → the only built-in sequence database: 110 named benchmark
                               proteins, name -> {sequence, length, description}
<any>.csv                   → the usual sequence source for real batches: `sequences.source`
                               ending in .csv is read by `name_column`/`sequence_column`;
                               `sequences.range: [first, last]` slices either kind of source
config/physics.yaml         → named physics parameter sets (e.g. "baseline", "cold"); entries
                               can `extends:` another set to inherit+override fields
                               (resolved by generate_jobs.py when it writes the manifest)

        │  src/generate_jobs.py --config config/config.yaml
        ▼
runs/<run_name>_<MMDD>/
  run_config.yaml            copy of the config.yaml used; analysis.py reads run-level options
                              from it (delete_trajectories_after_analysis, default false)
  job_manifest.csv           one row per task_id: sequence, sequence_name, seed, and every
                              physics param from physics.yaml flattened as param_<key>
  logs/<seq>_task_<id>.json  per-task status ("pending"→"running"→"completed"), written by
                              generate_jobs.py and updated in place by simulation.py — this is
                              how downstream steps know a job succeeded and where its data is
  data/timeseries/           pooled per-sequence CSVs written by analysis.py
  trajectories/              raw *.gsd trajectory files written by simulation.py

        │  src/run_local.py --run_dir runs/<run_name>_<MMDD>   (loops the manifest, shells out
        │                                                        to simulation.py per task_id)
        │  (or slurm/submit_batch_array.sh <run_dir>: an array job indexing the same manifest
        │   by $SLURM_ARRAY_TASK_ID)
        ▼
src/simulation.py --manifest <manifest> --task_id <N>
  Reads exactly one row of job_manifest.csv, builds a linear-chain HOOMD snapshot for that
  sequence, assigns per-residue amino-acid parameters (mass/charge/sigma/hydropathy from the
  hardcoded AA_PARAMS table — HPS1 = Dignon et al. 2018, HPS2 = Tesei et al. 2021 scales),
  derives electrostatics (Yukawa via a temperature/ionic-strength-dependent Debye length) and
  Rouse-time-based equilibration/production step counts when not explicitly overridden, runs
  equilibration then production, and streams frames to trajectories/<seq>_traj_<id>.gsd.
  Updates the matching logs/*.json with status/timing/frame count as it goes.

        │  src/analysis.py --run_dir runs/<run_name>_<MMDD>
        ▼
  Groups job_manifest.csv by sequence_name, and for every task whose log says "completed" and
  whose GSD file exists, uses MDAnalysis to compute per-frame Rg, Rg², Ree vector (x, y, z), Ree², asphericity
  (center-of-mass unwrapping handles PBC without needing bond topology). Concatenates all
  replicate/seed runs for a sequence into one
  data/timeseries/<seq>_pooled_timeseries.csv.

        │  src/quick_plot.py <path/to/*_pooled_timeseries.csv>
        ▼
  Fits an exponentially-modified-Gaussian to the Rg distribution and saves a two-panel
  histogram+KDE plot as <seq_name>_plot.png next to the CSV.
```

## Key conventions to preserve when editing this pipeline

- **The manifest is the single source of truth for a run.** Every stage (simulation, analysis)
  re-derives everything it needs from `job_manifest.csv` + the matching `logs/*.json` rather than
  taking parameters as flags. If you add a new physics parameter, add it in `physics.yaml` and it
  will automatically flow into the manifest as `param_<key>` — `simulation.py` reads it off the
  manifest row by that same name.
- **Derived quantities (KT, Yukawa epsilon/kappa, equilibration/production step counts) are
  computed in `simulation.py`, never stored in the YAML configs** — the physics configs only hold
  the inputs to those formulas (temperature, ionic concentration, multipliers).
- **`equilibration_steps`/`production_steps` being null in `physics.yaml` means "derive from the
  Rouse time"** (`rouse_multiplier`/`production_multiplier` × N^2.2 / dt); a non-null value is a
  hard override. Same null-means-derive pattern applies to `frames`.
- **Logs are the completion contract between stages.** `analysis.py` will silently skip any task
  whose log is missing, not `"status": "completed"`, or whose referenced trajectory file doesn't
  exist — it's designed to tolerate partial/failed runs in a batch rather than crash.
