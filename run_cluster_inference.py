#!/usr/bin/env python3
"""
run_cluster_inference.py
========================
Cluster script for running multiple independent RJMCMC trajectories for
spin-bath inference on the first experimental data set (XY4 exp 1).

Each trajectory starts from a different random initialization (seed and
initial spin configuration).  All trajectories use the same experimental
data and config hyperparameters.

Modes
-----
sequential      Run all NUM_TRAJECTORIES trajectories one after another in a
                single process.  Saves one combined output file when done.

multiprocessing Run all trajectories in parallel using multiprocessing.Pool
                within a single node (one worker process per trajectory).

single          Run exactly one trajectory identified by --trajectory-id N.
                Saves a partial result file named <output-dir>/traj_N.pkl.
                Intended for use inside a SLURM array job (one job per
                trajectory).

merge           Collect all partial result files from --output-dir and
                combine them into a single output pkl.  Run this after all
                array tasks finish.

Output layout
-------------
Combined pkl (sequential / multiprocessing / merge) contains:
    {
        'num_trajectories' : int,
        'num_trials'       : int,
        'config_traj0'     : dict,          # config used for trajectory 0
        'hf_df'            : pd.DataFrame,  # stored once to save space
        'hf_dist_mat'      : np.ndarray,
        'exp_params'       : dict,          # experiment parameter dict
        'coherence_data'   : list[np.ndarray],
        'trajectories'     : list[dict],    # one entry per trajectory:
            {
                'traj_id'         : int,
                'seed'            : int,
                'elapsed_seconds' : float,
                'posterior'       : list[dict],
                'spin_samples'    : list,
                'T2_samples'      : list,
                'error_samples'   : list,
                'k_samples'       : list,
                'merged_params'   : dict,
            }
    }

Plotting in a local notebook
-----------------------------
Load the combined pkl with workflow_pipeline.load_state (or plain pickle)
and use plot_convergence / plot_posterior on individual trajectory states
reconstructed from the stored arrays.  See the companion plotting notebook
section at the bottom of this docstring.

Example loading code:
    import pickle
    with open('results_combined.pkl', 'rb') as f:
        results = pickle.load(f)

    # convergence of trajectory 2
    traj = results['trajectories'][2]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(traj['error_samples'])
    axes[0].set_ylabel('L2 error')
    axes[1].plot(traj['k_samples'])
    axes[1].set_ylabel('# spins k')
    plt.suptitle(f"Trajectory {traj['traj_id']}  seed={traj['seed']}")
    plt.tight_layout(); plt.show()

Usage examples
--------------
# All trajectories sequentially (one job):
python run_cluster_inference.py --mode sequential --output-dir ./results

# All trajectories in parallel within one node:
python run_cluster_inference.py --mode multiprocessing --output-dir ./results

# One trajectory (for SLURM array job, SLURM_ARRAY_TASK_ID handled automatically):
python run_cluster_inference.py --mode single --trajectory-id 2 --output-dir ./results

# Merge partial files after SLURM array completes:
python run_cluster_inference.py --mode merge --output-dir ./results
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── NumPy 2.0 compatibility: restore np.unicode_ before pycce/rjmcmc load ───
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_  # type: ignore[attr-defined]

# ── Inference pipeline helpers (unchanged from workflow_pipeline) ─────────────
from workflow_pipeline import (
    default_config,
    load_hyperfine_data,
    build_initial_experiment,
    load_coherence_data_from_arrays,
    run_inference,
)

# ─────────────────────────────────────────────────────────────────────────────
# USER-EDITABLE SETTINGS
# Mirrors the notebook's USER SETTINGS cell exactly.
# ─────────────────────────────────────────────────────────────────────────────

NUM_TRAJECTORIES = 5       # number of independent RJMCMC chains to run
NUM_TRIALS       = 25_000  # MCMC steps per trajectory (walker steps)
BURN_IN          = 12_500  # steps to discard as burn-in (50 % of NUM_TRIALS)

# Per-trajectory random seeds: trajectory i uses BASE_SEED + i * SEED_STRIDE
# Ensures fully independent initializations and random walks.
BASE_SEED   = 42
SEED_STRIDE = 1000

# Experimental data paths (relative to the script's working directory)
DATA_DIR       = "exp_data"
TIME_FILE      = "XY4_exp1_time.pkl"
COHERENCE_FILE = "XY4_exp1_coherence.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_config(trajectory_id: int) -> Dict[str, Any]:
    """
    Return the full config dict for one trajectory.

    All trajectories share the same hyperparameters; only the random seed
    differs, which drives a different initial spin configuration and RWMH
    random walk path.

    Parameters
    ----------
    trajectory_id : int  (0-indexed)

    Returns
    -------
    config : dict
    """
    config = default_config()

    # ── hyperfine spin library ────────────────────────────────────────────────
    config["hf_file"]       = "nv-2.txt"
    config["strong_thresh"] = 750.0   # kHz – remove spins stronger than this
    config["weak_thresh"]   = 5.0     # kHz – remove spins weaker than this

    # ── first experiment ──────────────────────────────────────────────────────
    config["init_num_pulses"] = 4
    config["init_mag_field"]  = 332.0  # Gauss
    config["init_noise"]      = 0.1
    config["init_T2"]         = 0.9   # ms
    config["init_tmin"]       = 0.0   # ms
    config["init_tmax"]       = 0.008 # ms
    config["init_num_tps"]    = 50    # placeholder – overwritten by real data

    # ── RJMCMC sampler ────────────────────────────────────────────────────────
    config["k_max"]              = 20
    config["num_trials"]         = NUM_TRIALS
    config["burn_in"]            = BURN_IN
    config["num_strands"]        = 18
    config["beta_base"]          = 2.0
    config["r_spin"]             = 5.0
    config["r_T2"]               = 0.1
    config["num_rjmcmc_steps"]   = 20
    config["num_parallel_steps"] = 20
    config["num_T2_steps"]       = 20

    # ── per-trajectory seed ───────────────────────────────────────────────────
    # Different seed → different random initial spin set AND different RWMH path
    config["random_seed"] = BASE_SEED + trajectory_id * SEED_STRIDE

    # ── plotting (not used on cluster; kept for pkl portability) ─────────────
    config["plot_top_n"]         = 5
    config["show_ground_truth"]  = False
    config["ground_truth_spins"] = None

    return config


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_experimental_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load measured time-points and coherence signal from the pkl files used
    in the notebook.  Applies the same pre-processing (0.5 * signal + 0.5).

    Returns
    -------
    times_meas  : 1-D array (ms)
    signal_meas : 1-D array (coherence in ~[0, 1])
    """
    time_path = Path(DATA_DIR) / TIME_FILE
    coh_path  = Path(DATA_DIR) / COHERENCE_FILE

    if not time_path.exists():
        raise FileNotFoundError(
            f"Time file not found: {time_path}\n"
            f"Make sure '{DATA_DIR}/' is in the working directory."
        )
    if not coh_path.exists():
        raise FileNotFoundError(
            f"Coherence file not found: {coh_path}\n"
            f"Make sure '{DATA_DIR}/' is in the working directory."
        )

    with open(time_path, "rb") as f:
        times_raw = pickle.load(f)
    with open(coh_path, "rb") as f:
        signal_raw = pickle.load(f)

    # same transformation as the notebook
    signal_processed = 0.5 * signal_raw + 0.5

    times_meas, signal_meas = load_coherence_data_from_arrays(
        times_raw, signal_processed
    )
    print(f"  Loaded {len(times_meas)} time-points from {time_path.name}")
    return times_meas, signal_meas


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TRAJECTORY RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_single_trajectory(
    traj_id: int,
    base_state: Dict[str, Any],
    exp1: Dict[str, Any],
    coherence_data_1: List[np.ndarray],
) -> Dict[str, Any]:
    """
    Run one independent RJMCMC trajectory and return a compact result dict.

    Parameters
    ----------
    traj_id          : trajectory index (0-indexed)
    base_state       : initial pipeline state from load_hyperfine_data()
                       (not mutated – deepcopy happens inside run_inference)
    exp1             : experiment-parameter dict for the first experiment
    coherence_data_1 : list of 1-D coherence arrays

    Returns
    -------
    result : dict with keys
        traj_id, seed, elapsed_seconds,
        posterior, spin_samples, T2_samples, error_samples, k_samples,
        merged_params
    """
    config = build_config(traj_id)
    seed   = config["random_seed"]

    print(f"\n{'─'*60}")
    print(f"  Trajectory {traj_id}  |  seed={seed}  |  {NUM_TRIALS} steps")
    print(f"{'─'*60}", flush=True)

    t0 = time.time()
    state = run_inference(
        state              = base_state,
        new_coherence_data = coherence_data_1,
        new_exp_params     = copy.deepcopy(exp1),
        config             = config,
    )
    elapsed = time.time() - t0

    print(f"  Trajectory {traj_id} finished in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min).", flush=True)

    # ── extract only what is needed for saving ────────────────────────────────
    # hf_df and hf_dist_mat are large and shared across trajectories;
    # they are stored once at the top level of the combined output.
    result = dict(
        traj_id          = traj_id,
        seed             = seed,
        elapsed_seconds  = elapsed,
        posterior        = state["posterior"],
        spin_samples     = state["spin_samples"],
        T2_samples       = state["T2_samples"],
        error_samples    = state["error_samples"],
        k_samples        = state["k_samples"],
        merged_params    = state["merged_params"],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# WORKER WRAPPER  (needed for multiprocessing.Pool – must be picklable)
# ─────────────────────────────────────────────────────────────────────────────

def _pool_worker(args: Tuple) -> Dict[str, Any]:
    """Unpack args tuple and call run_single_trajectory (pool-safe wrapper)."""
    traj_id, base_state, exp1, coherence_data_1 = args
    return run_single_trajectory(traj_id, base_state, exp1, coherence_data_1)


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED OUTPUT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_output(
    trajectory_results: List[Dict[str, Any]],
    base_state: Dict[str, Any],
    exp1: Dict[str, Any],
    coherence_data_1: List[np.ndarray],
) -> Dict[str, Any]:
    """
    Package all trajectory results plus shared metadata into a single dict
    ready for pickling.

    Parameters
    ----------
    trajectory_results : list of result dicts from run_single_trajectory()
    base_state         : the initial pipeline state (holds hf_df, hf_dist_mat)
    exp1               : experiment parameter dict
    coherence_data_1   : measured coherence data

    Returns
    -------
    combined : dict (see module docstring for schema)
    """
    # Sort by traj_id in case results arrived out of order (multiprocessing)
    trajectory_results = sorted(trajectory_results, key=lambda r: r["traj_id"])

    combined = dict(
        num_trajectories = NUM_TRAJECTORIES,
        num_trials       = NUM_TRIALS,
        burn_in          = BURN_IN,
        base_seed        = BASE_SEED,
        seed_stride      = SEED_STRIDE,
        config_traj0     = build_config(0),  # representative config for reference
        # shared data (stored once)
        hf_df            = base_state["hf_df"],
        hf_dist_mat      = base_state["hf_dist_mat"],
        exp_params       = exp1,
        coherence_data   = coherence_data_1,
        # per-trajectory results
        trajectories     = trajectory_results,
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# SAVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_combined(output: Dict[str, Any], output_dir: str) -> str:
    """Pickle the combined output to <output_dir>/results_combined.pkl."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_combined.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nCombined results saved → {out_path}  ({size_mb:.1f} MB)")
    return str(out_path)


def save_partial(result: Dict[str, Any], output_dir: str) -> str:
    """Pickle a single trajectory result to <output_dir>/traj_<N>.pkl."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"traj_{result['traj_id']}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Partial result saved → {out_path}  ({size_mb:.1f} MB)")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# MODES
# ─────────────────────────────────────────────────────────────────────────────

def mode_sequential(output_dir: str) -> None:
    """Run all trajectories one by one in a single process."""
    print(f"\n{'='*60}")
    print(f"  MODE: sequential  |  {NUM_TRAJECTORIES} trajectories  |  {NUM_TRIALS} steps each")
    print(f"{'='*60}\n")

    print("Loading hyperfine data …")
    base_state = load_hyperfine_data(build_config(0))
    print(f"Loading experimental data …")
    times_meas, signal_meas = load_experimental_data()

    exp1 = build_initial_experiment(build_config(0))
    exp1["timepoints"] = np.array([times_meas])
    coherence_data_1   = [signal_meas]

    results = []
    wall_t0 = time.time()
    for traj_id in range(NUM_TRAJECTORIES):
        r = run_single_trajectory(traj_id, base_state, exp1, coherence_data_1)
        results.append(r)

    total = time.time() - wall_t0
    print(f"\nAll trajectories complete.  Total wall time: {total/60:.1f} min.")

    combined = build_combined_output(results, base_state, exp1, coherence_data_1)
    save_combined(combined, output_dir)


def mode_multiprocessing(output_dir: str) -> None:
    """Run all trajectories in parallel using multiprocessing.Pool."""
    print(f"\n{'='*60}")
    print(f"  MODE: multiprocessing  |  {NUM_TRAJECTORIES} workers  |  {NUM_TRIALS} steps each")
    print(f"{'='*60}\n")

    print("Loading hyperfine data …")
    base_state = load_hyperfine_data(build_config(0))
    print("Loading experimental data …")
    times_meas, signal_meas = load_experimental_data()

    exp1 = build_initial_experiment(build_config(0))
    exp1["timepoints"] = np.array([times_meas])
    coherence_data_1   = [signal_meas]

    # Build argument tuples for each worker
    worker_args = [
        (traj_id, base_state, exp1, coherence_data_1)
        for traj_id in range(NUM_TRAJECTORIES)
    ]

    print(f"Launching {NUM_TRAJECTORIES} worker processes …", flush=True)
    wall_t0 = time.time()

    # 'spawn' start method avoids issues with numpy/OpenBLAS thread-safety
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=NUM_TRAJECTORIES) as pool:
        results = pool.map(_pool_worker, worker_args)

    total = time.time() - wall_t0
    print(f"\nAll workers finished.  Total wall time: {total/60:.1f} min.")

    combined = build_combined_output(results, base_state, exp1, coherence_data_1)
    save_combined(combined, output_dir)


def mode_single(traj_id: int, output_dir: str) -> None:
    """
    Run one trajectory and save a partial pkl.
    Called by each task in a SLURM array job.
    """
    print(f"\n{'='*60}")
    print(f"  MODE: single  |  trajectory {traj_id}  |  {NUM_TRIALS} steps")
    print(f"{'='*60}\n")

    print("Loading hyperfine data …")
    base_state = load_hyperfine_data(build_config(traj_id))
    print("Loading experimental data …")
    times_meas, signal_meas = load_experimental_data()

    exp1 = build_initial_experiment(build_config(traj_id))
    exp1["timepoints"] = np.array([times_meas])
    coherence_data_1   = [signal_meas]

    result = run_single_trajectory(traj_id, base_state, exp1, coherence_data_1)

    # Save partial result; merge step will combine everything later
    partial_path = save_partial(result, output_dir)

    # Also save the shared data alongside the first partial file so merge
    # can reconstruct the combined output without needing to re-run data loading
    if traj_id == 0:
        shared_path = Path(output_dir) / "shared_data.pkl"
        shared = dict(
            hf_df          = base_state["hf_df"],
            hf_dist_mat    = base_state["hf_dist_mat"],
            exp_params     = exp1,
            coherence_data = coherence_data_1,
        )
        with open(shared_path, "wb") as f:
            pickle.dump(shared, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Shared data saved → {shared_path}")


def mode_merge(output_dir: str) -> None:
    """
    Collect all traj_N.pkl partial files from output_dir and combine into
    results_combined.pkl.  Run this after all SLURM array tasks finish.
    """
    out_dir = Path(output_dir)
    print(f"\n{'='*60}")
    print(f"  MODE: merge  |  reading from {out_dir}")
    print(f"{'='*60}\n")

    # ── load shared metadata (saved by trajectory 0) ─────────────────────────
    shared_path = out_dir / "shared_data.pkl"
    if not shared_path.exists():
        raise FileNotFoundError(
            f"Shared data file not found: {shared_path}\n"
            "Make sure trajectory 0 has completed successfully."
        )
    with open(shared_path, "rb") as f:
        shared = pickle.load(f)

    # Reconstruct a minimal base_state for build_combined_output
    base_state = dict(
        hf_df       = shared["hf_df"],
        hf_dist_mat = shared["hf_dist_mat"],
        experiments = [],
        coherence_data = [],
        posterior   = None,
        iteration   = 0,
    )

    # ── load all partial trajectory files ────────────────────────────────────
    partial_files = sorted(out_dir.glob("traj_*.pkl"))
    print(f"Found {len(partial_files)} partial file(s): "
          f"{[f.name for f in partial_files]}")

    if len(partial_files) == 0:
        raise RuntimeError(f"No traj_*.pkl files found in {out_dir}")
    if len(partial_files) < NUM_TRAJECTORIES:
        print(f"WARNING: expected {NUM_TRAJECTORIES} trajectories, "
              f"found {len(partial_files)}.  Merging available results only.")

    results = []
    for p in partial_files:
        with open(p, "rb") as f:
            r = pickle.load(f)
        print(f"  Loaded {p.name}  (traj_id={r['traj_id']}, "
              f"elapsed={r['elapsed_seconds']/60:.1f} min)")
        results.append(r)

    combined = build_combined_output(
        results, base_state, shared["exp_params"], shared["coherence_data"]
    )
    save_combined(combined, output_dir)
    print("\nMerge complete.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-trajectory RJMCMC spin-bath inference on cluster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "multiprocessing", "single", "merge"],
        default="sequential",
        help=(
            "sequential: run all trajectories in series (single job); "
            "multiprocessing: run all in parallel within one node; "
            "single: run one trajectory (use with --trajectory-id, for SLURM array); "
            "merge: combine partial traj_N.pkl files into results_combined.pkl."
        ),
    )
    parser.add_argument(
        "--trajectory-id",
        type=int,
        default=None,
        help=(
            "Trajectory index to run (0-indexed, required for --mode single). "
            "If not given, the SLURM_ARRAY_TASK_ID environment variable is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./inference_results",
        help="Directory for output pkl files (created if it does not exist).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "sequential":
        mode_sequential(args.output_dir)

    elif args.mode == "multiprocessing":
        mode_multiprocessing(args.output_dir)

    elif args.mode == "single":
        # Resolve trajectory ID: CLI arg takes priority, then SLURM env var
        traj_id = args.trajectory_id
        if traj_id is None:
            slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
            if slurm_id is None:
                raise ValueError(
                    "--trajectory-id must be given (or SLURM_ARRAY_TASK_ID set) "
                    "when using --mode single."
                )
            traj_id = int(slurm_id)
        if not 0 <= traj_id < NUM_TRAJECTORIES:
            raise ValueError(
                f"--trajectory-id must be in [0, {NUM_TRAJECTORIES - 1}], "
                f"got {traj_id}."
            )
        mode_single(traj_id, args.output_dir)

    elif args.mode == "merge":
        mode_merge(args.output_dir)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
