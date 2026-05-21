"""
plot_cluster_results.py
=======================
Helper functions for loading and visualising the combined RJMCMC output
produced by run_cluster_inference.py.

Import this in a local Jupyter notebook:

    from plot_cluster_results import (
        load_results,
        plot_all_convergence,
        plot_convergence_single,
        plot_all_posteriors,
        plot_combined_posterior,
        print_trajectory_summary,
    )

    results = load_results("inference_results/results_combined.pkl")
    plot_all_convergence(results)
    plot_combined_posterior(results, config)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: str) -> Dict[str, Any]:
    """
    Load the combined results pkl produced by run_cluster_inference.py.

    Parameters
    ----------
    path : path to results_combined.pkl

    Returns
    -------
    results : dict  (see run_cluster_inference module docstring for schema)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {p}")
    with open(p, "rb") as f:
        results = pickle.load(f)
    n  = results["num_trajectories"]
    nt = results["num_trials"]
    bi = results["burn_in"]
    print(f"Loaded {p.name}")
    print(f"  Trajectories : {n}")
    print(f"  Steps / traj : {nt}  (burn-in: {bi})")
    print(f"  Spins in lib : {len(results['hf_df'])}")
    for t in results["trajectories"]:
        elapsed = t["elapsed_seconds"]
        print(f"  traj {t['traj_id']}  seed={t['seed']}  "
              f"elapsed={elapsed/60:.1f} min  "
              f"posterior configs={len(t['posterior'])}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CONVERGENCE PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence_single(
    results: Dict[str, Any],
    traj_id: int,
    burn_in: Optional[int] = None,
    figsize: tuple = (10, 5),
) -> None:
    """
    Plot the error trace and k-trace for one trajectory.

    Parameters
    ----------
    results  : combined results dict from load_results()
    traj_id  : which trajectory to show (0-indexed)
    burn_in  : draw a vertical burn-in cutoff line (uses results['burn_in'] if None)
    """
    traj = results["trajectories"][traj_id]
    bi   = burn_in if burn_in is not None else results["burn_in"]

    err = np.asarray(traj["error_samples"])
    k   = np.asarray(traj["k_samples"])

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(err, lw=0.6, color="steelblue")
    axes[0].axvline(bi, color="red", lw=1, ls="--", label=f"burn-in ({bi})")
    axes[0].set_ylabel("L2 error", fontsize=12)
    axes[0].set_title(
        f"Trajectory {traj_id}  |  seed={traj['seed']}  |  "
        f"elapsed={traj['elapsed_seconds']/60:.1f} min",
        fontsize=13,
    )
    axes[0].legend(fontsize=10)

    axes[1].plot(k, lw=0.6, color="darkorange")
    axes[1].axvline(bi, color="red", lw=1, ls="--")
    axes[1].set_ylabel("# spins  k", fontsize=12)
    axes[1].set_xlabel("MCMC step", fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_all_convergence(
    results: Dict[str, Any],
    burn_in: Optional[int] = None,
    figsize: tuple = (14, 3),
) -> None:
    """
    Plot error traces for all trajectories on a shared figure (one column per
    trajectory) for quick visual comparison.

    Parameters
    ----------
    results : combined results dict
    burn_in : burn-in cutoff (uses results['burn_in'] if None)
    """
    trajs = results["trajectories"]
    n     = len(trajs)
    bi    = burn_in if burn_in is not None else results["burn_in"]

    fig, axes = plt.subplots(2, n, figsize=(figsize[0], figsize[1] * 2),
                              sharex=True, sharey="row")
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, traj in enumerate(trajs):
        err = np.asarray(traj["error_samples"])
        k   = np.asarray(traj["k_samples"])

        axes[0, j].plot(err, lw=0.5, color="steelblue")
        axes[0, j].axvline(bi, color="red", lw=1, ls="--")
        axes[0, j].set_title(f"traj {traj['traj_id']}\nseed={traj['seed']}",
                              fontsize=10)
        if j == 0:
            axes[0, j].set_ylabel("L2 error", fontsize=11)

        axes[1, j].plot(k, lw=0.5, color="darkorange")
        axes[1, j].axvline(bi, color="red", lw=1, ls="--")
        axes[1, j].set_xlabel("step", fontsize=10)
        if j == 0:
            axes[1, j].set_ylabel("# spins k", fontsize=11)

    fig.suptitle("MCMC convergence — all trajectories", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# POSTERIOR PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_posterior_single(
    results: Dict[str, Any],
    traj_id: int,
    is_simulation: bool = False,
    ground_truth_spins: Optional[List[int]] = None,
    figsize: tuple = (7, 6),
) -> None:
    """
    Scatter-plot the posterior for one trajectory in (A_par, A_perp) space.

    Parameters
    ----------
    results             : combined results dict
    traj_id             : which trajectory to show
    is_simulation       : if True and ground_truth_spins is provided, overlay GT
    ground_truth_spins  : list of spin indices for ground-truth overlay
    """
    traj      = results["trajectories"][traj_id]
    posterior = traj["posterior"]
    hf_df     = results["hf_df"]

    weights = np.array([d["weight"] for d in posterior], dtype=float)
    weights /= weights.sum()

    fig, ax = plt.subplots(figsize=figsize)
    for d, alpha in zip(posterior, weights):
        pairs = np.array(d["pairs"])
        if pairs.ndim == 1:
            pairs = pairs.reshape(1, -1)
        ax.scatter(pairs[:, 0], pairs[:, 1],
                   alpha=float(np.clip(alpha * 5, 0.02, 0.9)),
                   s=25, color="steelblue")

    if is_simulation and ground_truth_spins:
        for i, spin in enumerate(ground_truth_spins):
            x = hf_df.iloc[spin]["A_par"]
            y = hf_df.iloc[spin]["A_perp"]
            ax.plot(x, y, "x", color="red", ms=10, markeredgewidth=2,
                    label="ground truth" if i == 0 else "_")
        ax.legend(fontsize=11)

    ax.set_xlabel(r"$A_{\parallel}$ (kHz)", fontsize=13)
    ax.set_ylabel(r"$A_{\perp}$ (kHz)", fontsize=13)
    ax.set_title(
        f"Posterior — traj {traj_id}  (seed={traj['seed']})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def plot_all_posteriors(
    results: Dict[str, Any],
    is_simulation: bool = False,
    ground_truth_spins: Optional[List[int]] = None,
    figsize: tuple = (18, 5),
) -> None:
    """
    Plot all trajectory posteriors side-by-side for comparison.
    """
    trajs = results["trajectories"]
    n     = len(trajs)
    hf_df = results["hf_df"]

    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, traj in zip(axes, trajs):
        posterior = traj["posterior"]
        weights   = np.array([d["weight"] for d in posterior], dtype=float)
        weights  /= weights.sum()

        for d, alpha in zip(posterior, weights):
            pairs = np.array(d["pairs"])
            if pairs.ndim == 1:
                pairs = pairs.reshape(1, -1)
            ax.scatter(pairs[:, 0], pairs[:, 1],
                       alpha=float(np.clip(alpha * 5, 0.02, 0.9)),
                       s=15, color="steelblue")

        if is_simulation and ground_truth_spins:
            for i, spin in enumerate(ground_truth_spins):
                x = hf_df.iloc[spin]["A_par"]
                y = hf_df.iloc[spin]["A_perp"]
                ax.plot(x, y, "x", color="red", ms=9, markeredgewidth=2,
                        label="GT" if i == 0 else "_")

        ax.set_title(f"traj {traj['traj_id']}\nseed={traj['seed']}", fontsize=10)
        ax.set_xlabel(r"$A_{\parallel}$ (kHz)", fontsize=10)

    axes[0].set_ylabel(r"$A_{\perp}$ (kHz)", fontsize=10)
    if is_simulation and ground_truth_spins:
        axes[0].legend(fontsize=9)

    fig.suptitle("Posterior — all trajectories", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_combined_posterior(
    results: Dict[str, Any],
    is_simulation: bool = False,
    ground_truth_spins: Optional[List[int]] = None,
    figsize: tuple = (7, 6),
) -> None:
    """
    Plot a single posterior by pooling samples from all trajectories.
    Weights are normalised across the full pool.
    """
    hf_df = results["hf_df"]
    all_baths  = []
    all_weights = []
    for traj in results["trajectories"]:
        for d in traj["posterior"]:
            all_baths.append(d)
            all_weights.append(d["weight"])

    weights = np.array(all_weights, dtype=float)
    weights /= weights.sum()

    fig, ax = plt.subplots(figsize=figsize)
    for d, alpha in zip(all_baths, weights):
        pairs = np.array(d["pairs"])
        if pairs.ndim == 1:
            pairs = pairs.reshape(1, -1)
        ax.scatter(pairs[:, 0], pairs[:, 1],
                   alpha=float(np.clip(alpha * 3, 0.02, 0.9)),
                   s=20, color="steelblue")

    if is_simulation and ground_truth_spins:
        for i, spin in enumerate(ground_truth_spins):
            x = hf_df.iloc[spin]["A_par"]
            y = hf_df.iloc[spin]["A_perp"]
            ax.plot(x, y, "x", color="red", ms=10, markeredgewidth=2,
                    label="ground truth" if i == 0 else "_")
        ax.legend(fontsize=11)

    n = results["num_trajectories"]
    ax.set_xlabel(r"$A_{\parallel}$ (kHz)", fontsize=13)
    ax.set_ylabel(r"$A_{\perp}$ (kHz)", fontsize=13)
    ax.set_title(f"Combined posterior  ({n} trajectories pooled)", fontsize=14)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_trajectory_summary(results: Dict[str, Any], top_n: int = 5) -> pd.DataFrame:
    """
    Print a summary table for every trajectory showing its top-N posterior
    spin-bath configurations.

    Returns a DataFrame indexed by (traj_id, rank).
    """
    rows = []
    for traj in results["trajectories"]:
        posterior = traj["posterior"]
        top = sorted(posterior, key=lambda d: d["weight"], reverse=True)[:top_n]
        for rank, bath in enumerate(top, start=1):
            pairs_str = "; ".join(
                f"({ap:.2f},{aperp:.2f})" for ap, aperp in bath["pairs"]
            )
            rows.append(dict(
                traj_id  = traj["traj_id"],
                seed     = traj["seed"],
                rank     = rank,
                weight   = round(bath["weight"], 4),
                count    = bath["count"],
                k        = len(bath["pairs"]),
                pairs    = pairs_str,
            ))
    df = pd.DataFrame(rows).set_index(["traj_id", "rank"])
    print(df.to_string())
    return df
