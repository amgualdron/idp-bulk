import os
import json
import argparse
import pandas as pd
import numpy as np
import MDAnalysis as mda
import yaml
from pathlib import Path

def extract_trajectory_metrics(gsd_path, task_id, seed, seq_name):
    """
    Extracts structural metrics across all frames in a production GSD.
    Uses center-of-mass unwrapping to handle PBCs without requiring bond topology.
    """
    u = mda.Universe(gsd_path)
    protein = u.select_atoms('all')
    
    metrics = {
        'sequence_name': [],
        'task_id': [],
        'seed': [],
        'frame': [],
        'rg': [],
        'rg_sq': [],
        'ree_sq': [],
        'ree_x': [],
        'ree_y': [],
        'ree_z': [],
        'asphericity': []   
    }
    
    total_mass = np.sum(protein.masses)
    
    # Process every frame in the production trajectory
    for ts in u.trajectory:
        # 1. Unwrap coordinates ONCE per frame 
        u.atoms.unwrap(reference='com')
        
        # Grab properties after unwrapping
        positions = protein.positions
        com = protein.center_of_mass()
        masses = protein.masses
        
        # 2. Radius of Gyration (Standard)
        rg = protein.radius_of_gyration()
        
        # 3. Radius of Gyration Squared (Vectorized)
        ri_sq = (positions - com)**2
        sq = np.sum(ri_sq, axis=1)
        rg_sq = np.sum(masses * sq) / total_mass
        
        # 4. End-to-End vector (last bead minus first, box frame) and its squared length
        rend = positions[-1] - positions[0]
        rendsq_val = np.sum(rend**2)
        
        # 5. Append all metrics
        metrics['sequence_name'].append(seq_name)
        metrics['task_id'].append(task_id)
        metrics['seed'].append(seed)
        metrics['frame'].append(ts.frame)
        metrics['rg'].append(rg)
        metrics['rg_sq'].append(rg_sq)
        metrics['ree_sq'].append(rendsq_val)
        metrics['ree_x'].append(rend[0])
        metrics['ree_y'].append(rend[1])
        metrics['ree_z'].append(rend[2])
        metrics['asphericity'].append(protein.asphericity())
        
    return pd.DataFrame(metrics)

def main():
    parser = argparse.ArgumentParser(description="Analyze IDP trajectories and generate raw timeseries CSVs.")
    parser.add_argument("--run_dir", required=True, help="Path to the run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "job_manifest.csv"
    logs_dir = run_dir / "logs"
    
    data_dir = run_dir / "data"
    timeseries_dir = data_dir / "timeseries"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    # Run-level options live in the copy of the control panel generate_jobs.py left here.
    # Trajectories are kept unless the run explicitly asks for them to be deleted.
    run_config_path = run_dir / "run_config.yaml"
    delete_trajectories = False
    if run_config_path.exists():
        with open(run_config_path) as f:
            delete_trajectories = bool((yaml.safe_load(f) or {}).get("delete_trajectories_after_analysis", False))
    if delete_trajectories:
        print("delete_trajectories_after_analysis is on: each .gsd is removed once its "
              "sequence's CSV has been written.")

    # 1. Load Manifest
    df = pd.read_csv(manifest_path)

    # 2. Group by sequence (pooling replicates/seeds)
    seq_column = 'sequence_name' 
    grouped = df.groupby(seq_column)

    print(f"Found {len(grouped)} unique sequences. Extracting raw metrics...")

    for seq_name, group_df in grouped:
        print(f"\nProcessing {seq_name} ({len(group_df)} scheduled tasks)...")
        pooled_ts_dfs = []
        valid_task_count = 0
        analysed = []  # (log file, gsd file) of every trajectory read in this pass

        # Rows from an earlier analysis pass, so tasks whose trajectory has since been
        # deleted stay in the pooled CSV instead of being dropped when it is rewritten.
        ts_csv_path = timeseries_dir / f"{seq_name}_pooled_timeseries.csv"
        previous_df = pd.read_csv(ts_csv_path) if ts_csv_path.exists() else None

        for _, row in group_df.iterrows():
            task_id = row['task_id']
            seed = row['seed']
            log_file = logs_dir / f"{seq_name}_task_{task_id}.json"

            # 3. Check if log exists and job finished without error
            if log_file.exists():
                with open(log_file, 'r') as f:
                    try:
                        log_data = json.load(f)
                        if log_data.get("status") == "completed":

                            traj_path_str = log_data.get("trajectory_file")
                            
                            if not traj_path_str:
                                print(f"  [Skipping] Task {task_id}: No 'trajectory_file' key in log.")
                                continue
                                
                            gsd_file = Path(traj_path_str)
                            # The logged path is only valid where the job ran (e.g. the
                            # cluster); fall back to this run dir's own trajectories/.
                            if not gsd_file.exists():
                                gsd_file = run_dir / "trajectories" / gsd_file.name

                            if gsd_file.exists():
                                print(f"  [Extracting] Task {task_id} (Seed: {seed})...")
                                ts_df = extract_trajectory_metrics(str(gsd_file), task_id, seed, seq_name)
                                pooled_ts_dfs.append(ts_df)
                                valid_task_count += 1
                                analysed.append((log_file, gsd_file))
                            elif previous_df is not None and (previous_df['task_id'] == task_id).any():
                                print(f"  [Kept] Task {task_id}: GSD gone, reusing rows from the existing CSV.")
                                pooled_ts_dfs.append(previous_df[previous_df['task_id'] == task_id])
                                valid_task_count += 1
                            else:
                                print(f"  [Skipping] Task {task_id}: GSD missing at {gsd_file}")
                        else:
                            print(f"  [Skipping] Task {task_id}: Log status not completed.")
                            
                    except json.JSONDecodeError:
                        print(f"  [Error] Task {task_id}: Corrupted JSON log.")
            else:
                print(f"  [Skipping] Task {task_id}: No log found.")

        # 4. Pool data and save to a clean CSV
        if valid_task_count > 0:
            master_ts_df = pd.concat(pooled_ts_dfs, ignore_index=True)
            master_ts_df.to_csv(ts_csv_path, index=False)
            print(f"  -> Success: Saved {len(master_ts_df)} total frames to {ts_csv_path.name}")

            # Only now that the CSV is on disk is it safe to drop the raw trajectories
            if delete_trajectories:
                for log_file, gsd_file in analysed:
                    gsd_file.unlink()
                    with open(log_file) as f:
                        log_data = json.load(f)
                    log_data["trajectory_deleted"] = True
                    with open(log_file, "w") as f:
                        json.dump(log_data, f, indent=2)
                if analysed:
                    print(f"  -> Deleted {len(analysed)} trajectory file(s).")
        else:
            print(f"  -> Warning: No valid data extracted for {seq_name}.")

    print("\nExtraction pipeline complete.")

if __name__ == "__main__":
    main()