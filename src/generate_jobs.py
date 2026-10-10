import argparse
import itertools
import json
import pandas as pd
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_yaml(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

VALID_RESIDUES = set("ARNDCQEGHILKMFPSTWYV")  # the residues simulation.py has parameters for

def load_sequences_csv(file_path: Path, name_column: str, sequence_column: str) -> dict:
    """Read a CSV of sequences into the same name -> {"sequence": ...} shape as the YAML files.

    Names end up in file names, so anything outside [A-Za-z0-9._-] becomes "_". Without a
    name column, rows are named seq_0001, seq_0002, ... in file order.
    """
    df = pd.read_csv(file_path, dtype=str, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    if sequence_column not in df.columns:
        raise ValueError(
            f"{file_path.name} has no '{sequence_column}' column (found: {list(df.columns)}). "
            "Set sequences.sequence_column in the config."
        )
    df = df.dropna(subset=[sequence_column])

    if name_column in df.columns:
        names = [re.sub(r"[^A-Za-z0-9._-]+", "_", str(n).strip()) for n in df[name_column]]
    else:
        print(f"No '{name_column}' column in {file_path.name}: naming rows seq_0001, seq_0002, ...")
        names = [f"seq_{i:04d}" for i in range(1, len(df) + 1)]

    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"{file_path.name} has repeated sequence names, which would be pooled together "
            f"in analysis: {duplicates[:10]}{' ...' if len(duplicates) > 10 else ''}"
        )
    return {
        name: {"sequence": seq.strip().upper()}
        for name, seq in zip(names, df[sequence_column])
    }

def resolve_param_set(params: dict, name: str) -> dict:
    """Return a physics param set with its `extends:` parent merged in underneath it."""
    entry = dict(params[name])
    parent = entry.pop("extends", None)
    if parent is None:
        return entry
    return {**resolve_param_set(params, parent), **entry}

def create_unique_dir(base_name):
    path = Path(base_name)
    counter = 1
    
    # Keep incrementing the counter if the path already exists
    # e.g., 'data', then 'data (1)', then 'data (2)'
    new_path = path
    while new_path.exists():
        new_path = Path(f"{base_name}_({counter})")
        counter += 1
        
    new_path.mkdir(parents=True)
    return new_path

# ── Main Generator Logic ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate job directories and manifests."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the master run_config.yaml"
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    run_config = load_yaml(config_path)

    params_path= project_root / "config" / "physics.yaml"
    params = load_yaml(params_path)

    timestamp = datetime.now().strftime("%m%d")
    exp_name = run_config.get("run_name", "unnamed_run")
    exp_id = f"{exp_name}_{timestamp}"

    # 1. Load Sequences
    # A .csv source is read by column; anything else is a YAML sequence database.
    seq_config = run_config["sequences"]
    seq_source_path = config_path.parent / seq_config["source"]
    if seq_source_path.suffix.lower() == ".csv":
        all_sequences = load_sequences_csv(
            seq_source_path,
            seq_config.get("name_column", "name"),
            seq_config.get("sequence_column", "sequence"),
        )
    else:
        all_sequences = load_yaml(seq_source_path)

    # Optional `range: [first, last]` narrows the source to those entries first, counted
    # from 1 in file order and inclusive, so [1, 100] and [101, 200] split a file cleanly.
    seq_range = seq_config.get("range")
    if seq_range is not None:
        if (not isinstance(seq_range, list) or len(seq_range) != 2
                or not all(isinstance(v, int) for v in seq_range)
                or not 1 <= seq_range[0] <= seq_range[1]):
            raise ValueError(f"sequences.range must be [first, last] with 1 <= first <= last, got {seq_range}")
        first, last = seq_range
        if first > len(all_sequences):
            raise ValueError(f"sequences.range starts at {first} but {seq_source_path.name} has only {len(all_sequences)} sequences")
        if last > len(all_sequences):
            print(f"Warning: range ends at {last} but {seq_source_path.name} has only {len(all_sequences)} sequences; stopping there")
        all_sequences = dict(list(all_sequences.items())[first - 1:last])

    # 2. Sequence Selection Logic
    selection = run_config["sequences"].get("select", "all")
    selected_seqs = {}

    if isinstance(selection, list):
        # Explicit list provided
        for name in selection:
            if name in all_sequences:
                selected_seqs[name] = all_sequences[name]
            else:
                print(f"Warning: Sequence '{name}' not found in {seq_source_path.name}")

    elif selection == "all":
        # Full database
        selected_seqs = dict(all_sequences)

    # Leave out anything simulation.py could not build, rather than have the task crash later
    for name in list(selected_seqs):
        unknown = set(selected_seqs[name]["sequence"]) - VALID_RESIDUES
        if unknown or not selected_seqs[name]["sequence"]:
            print(f"Warning: skipping '{name}': non-standard residues {sorted(unknown)}")
            del selected_seqs[name]
    if not selected_seqs:
        raise ValueError(f"No usable sequences selected from {seq_source_path.name}")

    # 3. Setup Experiment Directory
    exp_dir = create_unique_dir(project_root / "runs" / exp_id)
    # Create logs directory for SLURM output
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)
    (exp_dir / "trajectories").mkdir(exist_ok=True)

    # Keep the control panel with the run: a record of how it was set up, and where
    # later stages read run-level options (e.g. analysis.py's trajectory cleanup).
    shutil.copy(config_path, exp_dir / "run_config.yaml")

    manifest_path = exp_dir / "job_manifest.csv"
    manifest_rows = []

    # 4. The Cartesian Product Loop (Sequences x num_replicates)
    task_id = 1
    for seq_name, replicate, param_set in itertools.product(
        selected_seqs.keys(), range(run_config["num_replicates"]), run_config["param_sets"]
    ):
        seq_data = selected_seqs[seq_name]
        seq_string = seq_data["sequence"]

        # Write Stage 1 run_metadata.json
        meta = {
            "task_id": task_id,
            "status": "pending",  # Ready to be updated by simulation.py
        }

        with open(exp_dir / "logs" / f"{seq_name}_task_{task_id}.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Append to manifest tracking
        manifest_row = {
                "task_id": task_id,
                "sequence_name": seq_name,
                "sequence": seq_string,
                "seed": random.randint(1000, 9999),  # Using random seed for replicates
            }
        for key,value in resolve_param_set(params, param_set).items():
            manifest_row[f'param_{key}'] = value

        manifest_rows.append(manifest_row)
        task_id += 1

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False)

    # 6. Generate Execution Script
    total_jobs = len(manifest_rows)
    runner = run_config.get("runner", "local")
    print(f"✅ Generated {total_jobs} jobs in {exp_dir}")
    print(f"📄 Manifest written to: {manifest_path}")
    if runner == "slurm":
        print(f"Next: slurm/submit_batch_array.sh {exp_dir}")
    else:
        print(f"Next: python src/run_local.py --run_dir {exp_dir}")

if __name__ == "__main__":
    main()