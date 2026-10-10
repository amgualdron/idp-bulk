# IDP coarse-grained MD pipeline

Runs one-bead-per-residue (HPS-style) Langevin dynamics of single intrinsically disordered
protein chains in [HOOMD-blue](https://hoomd-blue.readthedocs.io), then reduces the trajectories
to per-frame structural observables (Rg, Rg², Ree², asphericity) pooled per sequence.

You describe a batch in three YAML files, generate a run directory from them, and execute it
either on a workstation or as a SLURM array job. The same run directory works for both.

```
config/*.yaml ──generate_jobs.py──▶ runs/<name>_<MMDD>/ ──run_local.py or SLURM──▶ trajectories ──analysis.py──▶ CSVs
```

## Layout

```
config/
  config.yaml                 control panel for one batch: which sequences, replicates, physics sets
  <your>.csv                  your own sequences, one per row: the usual source for real batches
  benchmark_sequences.yaml    110 named benchmark proteins (name -> sequence, length, description)
  physics.yaml                named physics parameter sets
src/
  generate_jobs.py            config -> run directory + job_manifest.csv
  simulation.py               runs exactly one manifest row (one task)
  run_local.py                runs every task of a run directory, one after another
  analysis.py                 trajectories -> pooled per-sequence timeseries CSVs
  quick_plot.py               Rg histogram + fit for one pooled CSV
slurm/
  submit_batch_array.sh       submit a run directory to the cluster (run this one)
  submit_array.sh             the per-task job script it submits (resources live here)
runs/                         generated output, git-ignored
```

## Setup

The code needs Python with HOOMD-blue v4 or newer, plus `gsd`, `MDAnalysis`, `pandas`, `numpy`,
`pyyaml`, `tqdm`, and for plotting `scipy`, `matplotlib`, `seaborn`.

HOOMD comes in separate GPU and CPU builds, so the workstation and the cluster each need their
own environment:

```bash
# workstation with an NVIDIA GPU
conda create -n md-env -c conda-forge "hoomd=*=gpu*" gsd mdanalysis pandas numpy pyyaml tqdm scipy matplotlib seaborn

# CPU-only cluster (load the cluster's conda first, e.g. `module load anaconda`)
conda create -n md-env -c conda-forge "hoomd=*=cpu*" gsd mdanalysis pandas numpy pyyaml tqdm scipy matplotlib seaborn
```

Run every command below from the repository root with that environment active.

## User guide

### 1. Describe the batch

Edit `config/config.yaml`:

```yaml
run_name: R15_R17_full_run      # run directory becomes runs/<run_name>_<MMDD>
sequences:
  source: benchmark_sequences.yaml   # the benchmark set, or a CSV of your own (see below)
  select: [R15, R17]            # a list of names, or `all`
num_replicates: 2               # independent seeds per sequence per param set
param_sets: [baseline]          # one or more sets from physics.yaml
delete_trajectories_after_analysis: false   # see "Deleting trajectories" below
runner: local                   # local | slurm (only changes the hint generate_jobs.py prints)
```

The batch is the product *sequences × replicates × param sets*, one task each.

To add a named protein to the benchmark set, add an entry to `config/benchmark_sequences.yaml`:

```yaml
MyProtein:
  sequence: MSEQ...             # one-letter codes, the 20 standard amino acids only
  length: 120
  description: "free text"
```

For real batches, point `source` at a CSV instead, with one sequence per row:

```yaml
sequences:
  source: my_sequences.csv      # relative to config/, or an absolute path
  name_column: name             # optional, this is the default
  sequence_column: sequence     # optional, this is the default
  range: [1, 100]               # optional: only rows 1 to 100 of the file
  select: all                   # or a list of names, as with YAML
```

- `range: [first, last]` takes that slice of the file, counted from 1 and inclusive, so
  `[1, 100]`, `[101, 200]`, ... split a big file into batches. It counts every row that has a
  sequence, including ones later skipped as invalid, so a range can yield slightly fewer tasks
  than its width. It works for YAML sources too.
- Other columns in the CSV are ignored.
- Names become file names, so characters other than letters, digits, `.`, `_` and `-` are
  replaced with `_`. Repeated names are an error.
- If the CSV has no name column, rows are named `seq_0001`, `seq_0002`, ... in file order.
- Sequences are upper-cased. Any sequence containing a letter outside the 20 standard amino
  acids is skipped with a warning (this check applies to YAML sources too).

To add physics conditions, add a set to `config/physics.yaml`. A set can inherit from another and
list only what it changes:

```yaml
cold:
  extends: baseline
  description: "Very low temperature run for collapse behavior"
  temperature_K: 5
```

| Field | Meaning |
|---|---|
| `temperature_K`, `ionic_concentration_M` | Thermodynamic state; sets kT and the Debye screening length |
| `hydropathy_scale` | `HPS1` (Dignon et al. 2018) or `HPS2` (Tesei et al. 2021) |
| `epsilon_lj`, `lj_r_cut`, `yukawa_r_cut` | Non-bonded well depth (kcal/mol) and cutoffs (nm) |
| `harmonic_k`, `harmonic_r0` | Bond stiffness and rest length |
| `box_size` | Cubic periodic box edge (nm) |
| `dt` | Integration timestep |
| `rouse_multiplier`, `production_multiplier` | Run lengths as multiples of N^2.2 (see below) |
| `equilibration_steps`, `production_steps` | Leave empty to derive from the multipliers; a number is a hard override |
| `frames` | Number of trajectory frames saved over the production run |

Run length scales with chain length N: equilibration is `rouse_multiplier × N^2.2 / dt` steps
and production is `production_multiplier × N^2.2 / dt` steps, unless overridden.

### 2. Generate the run directory

```bash
python src/generate_jobs.py --config config/config.yaml
```

This creates `runs/<run_name>_<MMDD>/` (a `_(1)`, `_(2)` suffix is added if it already exists):

```
runs/<run_name>_<MMDD>/
  run_config.yaml            copy of the config.yaml the run was generated from
  job_manifest.csv           one row per task: task_id, sequence, seed, and every physics field as param_<name>
  logs/<seq>_task_<id>.json  per-task status: pending -> running -> completed
  trajectories/              <seq>_traj_<id>.gsd, written by the simulation
  data/timeseries/           pooled CSVs, written by the analysis
  slurm_logs/                printed output of each cluster task (cluster runs only)
```

The manifest is the complete record of the run. Later edits to the YAML files do not affect a
run directory that already exists.

### 3a. Run on a workstation

```bash
python src/run_local.py --run_dir runs/<run_name>_<MMDD>
```

Tasks run one at a time on the GPU. To run a single task, or to run on CPU:

```bash
python src/simulation.py --manifest runs/<run_name>_<MMDD>/job_manifest.csv --task_id 3 --device cpu
```

`--device` is `gpu` (default), `cpu`, or `auto` (GPU if one is visible, else CPU).

### 3b. Run on a SLURM cluster

On the login node:

```bash
slurm/submit_batch_array.sh runs/<run_name>_<MMDD>
```

This submits the run as a SLURM array in which every task is its own single-core job. Options:

```bash
slurm/submit_batch_array.sh <run_dir> 50                      # at most 50 tasks running at once
slurm/submit_batch_array.sh <run_dir> --time=3-00:00:00       # any sbatch flag is passed through
IDP_CONDA_ENV=/path/to/env slurm/submit_batch_array.sh <run_dir>   # non-default conda env
IDP_MODULE=anaconda/anaconda-2023.09 slurm/submit_batch_array.sh <run_dir>   # non-default module
```

Defaults are in the `#SBATCH` lines of `slurm/submit_array.sh`: 1 core and a 10-day walltime.

| Task | Command |
|---|---|
| See your queued and running tasks | `squeue -u $USER` |
| Cancel a whole array, or one task | `scancel <jobid>`, `scancel <jobid>_<task_id>` |
| Read a task's output | `runs/<run>/slurm_logs/idp-md_<jobid>_<task_id>.out` |
| Resubmit what failed or timed out | run `submit_batch_array.sh` again |

Things to know:

- **There is no checkpointing.** A task that hits its walltime loses all progress, so set
  `--time` from the longest sequence in the batch. Run one or two tasks first to measure speed.
- **Resubmitting** only sends tasks whose log is not `completed`. Wait until the previous array
  has left the queue first, otherwise tasks still running are submitted a second time.
- **Large batches.** Clusters cap the size of one array (usually 1000 tasks). A manifest with
  more rows is submitted automatically as several arrays; output files of the later ones are
  named `..._<index>_plus<offset>.out`, where index + offset is the manifest `task_id`. A
  throttle applies to each array separately. Clusters may also cap how many jobs one user can
  have queued, in which case `sbatch` refuses the excess; re-run the script once tasks finish.

### 4. Analyse

```bash
python src/analysis.py --run_dir runs/<run_name>_<MMDD>
```

Writes `data/timeseries/<seq>_pooled_timeseries.csv` for each sequence, with all its replicates
concatenated, one row per saved frame. Lengths are in the simulation's units (nm).

| Column | Meaning |
|---|---|
| `sequence_name`, `task_id`, `seed` | Which task the frame came from |
| `frame` | Frame index within that task's production trajectory |
| `rg` | Mass-weighted radius of gyration |
| `rg_sq` | Its square, computed directly (for averaging ⟨Rg²⟩) |
| `ree_x`, `ree_y`, `ree_z` | End-to-end vector, last residue minus first, in the box frame |
| `ree_sq` | Squared end-to-end distance |
| `asphericity` | 0 for a sphere, 1 for a rod |

#### Deleting trajectories

Trajectories are kept by default. With `delete_trajectories_after_analysis: true`, `analysis.py`
deletes each `.gsd` once the CSV for its sequence has been written, and marks the task's log
with `"trajectory_deleted": true`.

- The CSV is then the only record: an observable that is not already a column can only be
  obtained by re-running the simulation.
- Re-running the analysis later is safe. Tasks whose trajectory is gone keep their existing rows
  in the CSV, and newly finished tasks are added.
- The setting is read from `runs/<run>/run_config.yaml`, so it can be switched for one existing
  run by editing that file. Changing `config/config.yaml` only affects runs generated afterwards.

To add an observable, compute it per frame in `extract_trajectory_metrics` in `src/analysis.py`
and append it to a new key of the `metrics` dictionary; it becomes a column.

Tasks that are not `completed`, or whose trajectory is missing, are skipped with a message, so
analysis can be run on a partially finished batch and re-run later. A run directory copied from
the cluster to another machine can be analysed there as is.

```bash
python src/quick_plot.py runs/<run>/data/timeseries/<seq>_pooled_timeseries.csv
```

Saves `<seq>_plot.png` next to the CSV: an Rg histogram with an exponentially-modified-Gaussian
fit, and a kernel density estimate.

## The model

Each residue is one bead carrying the mass, charge, diameter and hydropathy of its amino acid
(table `AA_PARAMS` in `src/simulation.py`). The chain starts as a straight line and is integrated
with a Langevin thermostat. Forces:

- **Bonds:** harmonic springs between neighbours.
- **Short-range:** the Ashbaugh–Hatch form, a Lennard-Jones potential whose attraction is scaled
  by the average hydropathy of the pair.
- **Electrostatics:** screened Coulomb (Yukawa), with a temperature-dependent water dielectric
  constant and a Debye length set by the ionic concentration.

Derived quantities such as kT, the Yukawa parameters and step counts are computed in
`simulation.py` from the manifest row and are never stored in the YAML files.
