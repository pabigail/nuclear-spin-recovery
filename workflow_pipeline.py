"""
workflow_pipeline.py
====================
High-level orchestration helpers for the spin-bath inference pipeline.

This module wraps the existing RJMCMC / adaptive-experiment machinery
(rjmcmc.py, rwmh_implementations.py, adaptive_exp.py) behind a small,
experimentalist-friendly API.  No inference logic is modified here.

Typical usage (in the notebook)
--------------------------------
    config = default_config()           # tweak as needed
    state  = load_hyperfine_data(...)   # one-time setup

    # ── first experiment ──────────────────────────────
    data1 = load_coherence_data(...)    # your first measurement
    state = run_inference(state, data1, config)
    plot_posterior(state)
    rec = suggest_next_experiment(state, config)
    display_recommendation(rec)

    # ── second experiment ─────────────────────────────
    data2 = load_coherence_data(...)    # add your next measurement
    state = run_inference(state, data2, config, new_exp_params=rec['optimized_exp'])
    plot_posterior(state)
    ...
"""

from __future__ import annotations

import copy
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── NumPy 2.0 compatibility patch ────────────────────────────────────────────
# pycce (imported transitively through rjmcmc) references np.unicode_, which
# was removed in NumPy 2.0.  Restore the alias before pycce loads so that the
# attribute lookup in pycce/bath/cube.py succeeds on any NumPy version.
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_  # type: ignore[attr-defined]

# ── existing inference modules (unchanged) ───────────────────────────────────
from rjmcmc import (
    make_df_from_Ivady_file,
    get_distance_matrix,
    make_exp_params_dict,
    calculate_coherence_with_T2,
    calculate_coherence_with_T2_and_noise,
    get_specific_exp_parameters,
)
from rwmh_implementations import RJMCMC_RWMH_with_parallel_tempering
from adaptive_exp import (
    spin_bath_posterior_with_indices,
    merge_exp_params,
    optimize_experiment,
    make_dense_time_experiment,
    information_density,
    prediction_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def default_config() -> Dict[str, Any]:
    """
    Return a dictionary of all tunable hyperparameters with sensible defaults.

    Edit the values in the returned dict inside the notebook's USER SETTINGS
    cell.  Only modify fields you actually need to change.

    Returns
    -------
    config : dict
        Keys are grouped into logical sections:

        Hyperfine data loading
        ~~~~~~~~~~~~~~~~~~~~~~
        hf_file         : path to the ab-initio hyperfine text file
        strong_thresh   : kHz – remove spins with any coupling > this value
        weak_thresh     : kHz – remove spins with all couplings < this value

        Initial experiment
        ~~~~~~~~~~~~~~~~~~
        init_num_pulses : number of DD pulses for the first experiment
        init_mag_field  : magnetic field in Gauss
        init_noise      : measurement noise (sigma)
        init_T2         : initial T2 coherence time (ms)
        init_tmin       : start time of the first experiment (ms)
        init_tmax       : end time of the first experiment (ms)
        init_num_tps    : number of time-points sampled

        RJMCMC sampler
        ~~~~~~~~~~~~~~~
        k_max             : maximum number of spins in the bath
        num_trials        : total MCMC steps per inference call
        burn_in           : steps to discard as burn-in
        num_strands       : number of parallel-tempering replicas
        beta_base         : base for geometric temperature ladder (beta[i]=base^i)
        r_spin            : RWMH step radius in real space (Å)
        r_T2              : RWMH step size for T2
        num_rjmcmc_steps  : dimension-jump steps per outer iteration
        num_parallel_steps: parallel-tempering steps per outer iteration
        num_T2_steps      : T2 update steps per outer iteration
        random_seed       : integer seed (None = random)

        Candidate experiment design
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        pulse_options     : list of pulse numbers to try as candidates
        tmax_options      : list of maximum times (ms) to try as candidates
        num_candidate_tps : time-points in each candidate experiment
        candidate_mag_field: magnetic field for candidate experiments (G)
        candidate_noise   : noise level for candidate experiments
        candidate_T2      : T2 for candidate experiments (ms)

        Experiment optimisation (timepoint selection)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        T_budget          : total measurement-time budget
        Nt_dense          : dense-grid points for Fisher-information sweep
        M_eig             : Monte-Carlo samples for EIG estimation

        Plotting
        ~~~~~~~~
        plot_top_n        : how many top-weight baths to highlight
        show_ground_truth : if True and ground_truth_spins is set, overlay GT
        ground_truth_spins: list of spin indices (optional, for simulations)
    """
    return dict(
        # ── hyperfine data ──────────────────────────────────────────────────
        hf_file         = "nv-2.txt",
        strong_thresh   = 250.0,
        weak_thresh     = 10.0,

        # ── first / initial experiment ──────────────────────────────────────
        init_num_pulses = 4,
        init_mag_field  = 300.0,
        init_noise      = 0.008,
        init_T2         = 0.9,
        init_tmin       = 0.0,
        init_tmax       = 0.008,
        init_num_tps    = 50,

        # ── RJMCMC sampler ──────────────────────────────────────────────────
        k_max              = 20,
        num_trials         = 10_000,
        burn_in            = 5_000,
        num_strands        = 18,
        beta_base          = 2.0,
        r_spin             = 5.0,
        r_T2               = 0.1,
        num_rjmcmc_steps   = 20,
        num_parallel_steps = 20,
        num_T2_steps       = 20,
        random_seed        = None,

        # ── candidate experiments ───────────────────────────────────────────
        pulse_options      = [4, 8, 16, 32, 64, 128],
        tmax_options       = list(np.linspace(0.001, 0.01, 5)),
        num_candidate_tps  = 100,
        candidate_mag_field= 500.0,
        candidate_noise    = 0.01,
        candidate_T2       = 0.9,

        # ── experiment optimisation ─────────────────────────────────────────
        T_budget   = 10.0,
        Nt_dense   = 300,
        M_eig      = 20,

        # ── plotting ────────────────────────────────────────────────────────
        plot_top_n          = 5,
        show_ground_truth   = False,
        ground_truth_spins  = None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING / SETUP
# ─────────────────────────────────────────────────────────────────────────────

def load_hyperfine_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load the ab-initio hyperfine dataframe and pre-compute the distance matrix.

    Parameters
    ----------
    config : dict
        Must contain 'hf_file', 'strong_thresh', 'weak_thresh'.

    Returns
    -------
    state : dict
        Initial pipeline state with keys:
            'hf_df'        – hyperfine DataFrame
            'hf_dist_mat'  – pairwise distance matrix
            'experiments'  – list of accumulated experiment dicts (empty)
            'coherence_data' – list of measured coherence arrays (empty)
            'posterior'    – None (no inference run yet)
            'iteration'    – 0
    """
    print("Loading hyperfine data …", end=" ", flush=True)
    t0 = time.time()
    hf_df = make_df_from_Ivady_file(
        config["hf_file"],
        config["strong_thresh"],
        config["weak_thresh"],
    )
    hf_dist_mat = get_distance_matrix(hf_df)
    print(f"done ({time.time()-t0:.1f}s).  Found {len(hf_df)} spins in library.")

    return dict(
        hf_df        = hf_df,
        hf_dist_mat  = hf_dist_mat,
        experiments  = [],       # list of single-exp param dicts
        coherence_data = [],     # list of 1-D arrays
        posterior    = None,
        iteration    = 0,
    )


def build_initial_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the experiment-parameter dict for the very first measurement.

    Parameters
    ----------
    config : dict
        Uses init_* keys.

    Returns
    -------
    exp_params : dict   (single experiment, compatible with rjmcmc API)
    """
    times = np.linspace(config["init_tmin"], config["init_tmax"],
                        config["init_num_tps"])
    return make_exp_params_dict(
        num_exps  = 1,
        num_pulses= [config["init_num_pulses"]],
        mag_field = [config["init_mag_field"]],
        noise     = [config["init_noise"]],
        timepoints= [times],
        T2        = [config["init_T2"]],
    )


def load_coherence_data_from_arrays(
    timepoints: np.ndarray,
    signal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accept pre-loaded (time, coherence) arrays.  Returns them as-is after
    basic validation.

    Parameters
    ----------
    timepoints : 1-D array of floats (ms)
    signal     : 1-D array of floats in roughly [-1, 1]

    Returns
    -------
    (timepoints, signal) validated arrays
    """
    timepoints = np.asarray(timepoints, dtype=float).ravel()
    signal     = np.asarray(signal,     dtype=float).ravel()
    if len(timepoints) != len(signal):
        raise ValueError(
            f"timepoints length ({len(timepoints)}) != signal length ({len(signal)})"
        )
    return timepoints, signal


def simulate_coherence_data(
    spin_list: List[int],
    exp_params: Dict[str, Any],
    hf_df: pd.DataFrame,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Generate synthetic (noisy) coherence data from a known spin configuration.
    Useful for testing or demonstration.

    Parameters
    ----------
    spin_list  : ground-truth spin indices
    exp_params : single- or multi-experiment parameter dict
    hf_df      : hyperfine DataFrame
    seed       : random seed (None = random)

    Returns
    -------
    coherence_data : list of 1-D arrays (one per experiment)
    """
    if seed is not None:
        np.random.seed(seed)
    return calculate_coherence_with_T2_and_noise(spin_list, hf_df, exp_params)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  INFERENCE  (the main black-box call)
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    state: Dict[str, Any],
    new_coherence_data: List[np.ndarray],
    new_exp_params: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add one new experiment's data to the accumulated history and re-run RJMCMC
    to update the posterior.

    Parameters
    ----------
    state              : pipeline state dict (from load_hyperfine_data or a
                         previous run_inference call)
    new_coherence_data : list of 1-D arrays for the new experiment(s)
    new_exp_params     : experiment-parameter dict for the new data
                         (use build_initial_experiment for the first call)
    config             : configuration dict

    Returns
    -------
    state : updated pipeline state with keys
        'posterior'    – list of bath dicts (weight, pairs, indices, count)
        'spin_samples' – raw MCMC spin samples (for diagnostics)
        'T2_samples'   – raw MCMC T2 samples
        'error_samples'– raw MCMC error trace
        'merged_params'– merged multi-experiment params used for this inference
        'iteration'    – iteration counter (incremented)
    """
    state = copy.deepcopy(state)

    # ── accumulate data ──────────────────────────────────────────────────────
    state["experiments"].append(new_exp_params)
    state["coherence_data"].extend(new_coherence_data)
    n_iter = state["iteration"] + 1
    state["iteration"] = n_iter

    print(f"\n{'='*60}")
    print(f"  Iteration {n_iter}: running RJMCMC inference …")
    print(f"{'='*60}")

    # ── merge all experiments so far ─────────────────────────────────────────
    merged = merge_exp_params(state["experiments"])

    # ── random seed ──────────────────────────────────────────────────────────
    if config.get("random_seed") is not None:
        np.random.seed(config["random_seed"] + n_iter)

    # ── RJMCMC hyperparameters ───────────────────────────────────────────────
    k_max       = config["k_max"]
    num_strands = config["num_strands"]
    beta        = [1.0 / (config["beta_base"] ** i)
                   for i in range(num_strands)]

    rand_k     = np.random.randint(1, k_max + 1)
    init_spins = np.random.choice(len(state["hf_df"]),
                                  size=rand_k, replace=False).tolist()

    print(f"  Starting from k={rand_k} random spins, "
          f"{config['num_trials']} MCMC steps …", flush=True)
    t0 = time.time()

    k_samples, spin_samples, T2_samples, error_samples = \
        RJMCMC_RWMH_with_parallel_tempering(
            init_spins,
            state["hf_df"],
            state["hf_dist_mat"],
            config["r_spin"],
            config["r_T2"],
            merged,
            state["coherence_data"],
            config["num_trials"],
            k_max,
            num_strands,
            beta,
            config["num_rjmcmc_steps"],
            config["num_parallel_steps"],
            config["num_T2_steps"],
            sigma_sq=None,
        )

    elapsed = time.time() - t0
    print(f"  RJMCMC finished in {elapsed:.1f}s.")

    # ── extract posterior (discard burn-in) ──────────────────────────────────
    burn_in = config["burn_in"]
    post_samples = spin_samples[burn_in:]
    posterior = spin_bath_posterior_with_indices(post_samples, state["hf_df"])
    print(f"  Posterior: {len(posterior)} unique spin-bath configurations.")

    state.update(
        merged_params  = merged,
        posterior      = posterior,
        spin_samples   = spin_samples,
        T2_samples     = T2_samples,
        error_samples  = error_samples,
        k_samples      = k_samples,
    )
    return state


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_posterior(
    state: Dict[str, Any],
    config: Dict[str, Any],
    is_simulation: bool = False,
    fig_size: Tuple[float, float] = (7, 6),
) -> None:
    """
    Scatter-plot the posterior distribution over spin-bath configurations in
    the (A_par, A_perp) hyperfine plane.

    Each dot corresponds to one spin inside one sampled bath.  Opacity encodes
    posterior weight.

    Ground-truth spins (red crosses) are overlaid **only** when
    ``is_simulation=True`` and ``config['ground_truth_spins']`` is set.
    Set ``is_simulation=False`` (the default) for real experimental data where
    the ground truth is unknown.

    Parameters
    ----------
    state         : pipeline state (must have 'posterior' key)
    config        : configuration dict
    is_simulation : bool
        ``True``  – simulated data; ground-truth spin locations are known and
                    will be shown as red crosses on the plot.
        ``False`` – real experimental data; ground-truth overlay is suppressed
                    regardless of what is stored in config['ground_truth_spins'].
    """
    posterior = state.get("posterior")
    if posterior is None:
        print("No posterior yet – run run_inference() first.")
        return

    hf_df = state["hf_df"]
    n_iter = state["iteration"]

    weights     = np.array([d["weight"] for d in posterior], dtype=float)
    weights    /= weights.sum()

    fig, ax = plt.subplots(figsize=fig_size)

    for d, alpha in zip(posterior, weights):
        pairs = np.array(d["pairs"])
        if pairs.ndim == 1:
            pairs = pairs.reshape(1, -1)
        ax.scatter(pairs[:, 0], pairs[:, 1],
                   alpha=float(np.clip(alpha * 5, 0.02, 0.9)),
                   s=25, color="steelblue")

    # ground truth overlay – only shown for simulations where GT is known
    show_gt = is_simulation and bool(config.get("ground_truth_spins"))
    if show_gt:
        for spin in config["ground_truth_spins"]:
            x = hf_df.iloc[spin]["A_par"]
            y = hf_df.iloc[spin]["A_perp"]
            ax.plot(x, y, "x", color="red", ms=10, markeredgewidth=2,
                    label="ground truth" if spin == config["ground_truth_spins"][0]
                    else "_")

    mode_label = "simulation" if is_simulation else "experiment"
    ax.set_xlabel(r"$A_{\parallel}$ (kHz)", fontsize=13)
    ax.set_ylabel(r"$A_{\perp}$ (kHz)",    fontsize=13)
    ax.set_title(
        f"Spin-bath posterior  ·  iteration {n_iter}  ·  [{mode_label}]",
        fontsize=14,
    )
    if show_gt:
        ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_convergence(state: Dict[str, Any]) -> None:
    """
    Plot the MCMC error trace and the k (spin-count) trace as diagnostics.

    Parameters
    ----------
    state : pipeline state (must have 'error_samples' and 'k_samples')
    """
    err = state.get("error_samples")
    k   = state.get("k_samples")
    if err is None:
        print("No MCMC samples – run run_inference() first.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    axes[0].plot(err, lw=0.6, color="steelblue")
    axes[0].set_ylabel("L2 error", fontsize=12)
    axes[0].set_title(f"MCMC convergence diagnostics  ·  iteration {state['iteration']}",
                      fontsize=13)

    if k is not None:
        axes[1].plot(k, lw=0.6, color="darkorange")
        axes[1].set_ylabel("# spins  k", fontsize=12)
    axes[-1].set_xlabel("MCMC step", fontsize=12)

    if state.get("burn_in"):
        for ax in axes:
            ax.axvline(state["burn_in"], color="red", ls="--",
                       lw=1, label="burn-in cutoff")
        axes[0].legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_posterior_fits(
    state: Dict[str, Any],
    config: Dict[str, Any],
    exp_index: int = -1,
) -> None:
    """
    Overlay the top-N posterior bath coherence predictions on the measured data
    for one experiment.

    Parameters
    ----------
    state     : pipeline state
    config    : configuration dict  (uses 'plot_top_n')
    exp_index : which experiment to show (default: last one, -1)
    """
    posterior = state.get("posterior")
    if posterior is None:
        print("No posterior yet.")
        return

    hf_df        = state["hf_df"]
    merged       = state["merged_params"]
    coh_data     = state["coherence_data"]
    n_top        = config.get("plot_top_n", 5)

    # pick experiment
    n_exp = merged["num_experiments"]
    idx   = exp_index % n_exp
    times = merged["timepoints"][idx]
    meas  = coh_data[idx]

    # top-N baths
    top_n = sorted(posterior, key=lambda d: d["weight"], reverse=True)[:n_top]

    # dense time grid for smooth curves
    t_dense = np.linspace(times[0], times[-1], 500)

    # build a temporary single-exp dict for prediction
    single_exp = make_exp_params_dict(
        num_exps  = 1,
        num_pulses= [merged["num_pulses"][idx]],
        mag_field = [merged["mag_field"][idx]],
        noise     = [merged["noise"][idx]],
        timepoints= [t_dense],
        T2        = [merged["T2"][idx]],
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    for j, bath in enumerate(top_n):
        pred = calculate_coherence_with_T2(bath["indices"], hf_df, single_exp)
        label = f"bath #{j+1}  w={bath['weight']:.3f}"
        ax.plot(t_dense, pred[0], "--", lw=1.2, label=label)

    ax.plot(times, meas, ".", color="black", ms=6, zorder=5, label="measured data")
    ax.set_xlabel("time (ms)", fontsize=12)
    ax.set_ylabel("coherence", fontsize=12)
    ax.set_title(
        f"Top-{n_top} posterior baths vs. data  "
        f"(exp {idx+1}/{n_exp}, iteration {state['iteration']})",
        fontsize=13,
    )
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EXPERIMENT DESIGN
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_experiments(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Enumerate the grid of candidate experiment designs from config.

    Returns
    -------
    candidates : list of single-experiment param dicts
    """
    candidates = []
    for np_ in config["pulse_options"]:
        for tmax in config["tmax_options"]:
            times = np.linspace(0.0, tmax, config["num_candidate_tps"])
            exp   = make_exp_params_dict(
                num_exps  = 1,
                num_pulses= [np_],
                mag_field = [config["candidate_mag_field"]],
                noise     = [config["candidate_noise"]],
                timepoints= [times],
                T2        = [config["candidate_T2"]],
            )
            candidates.append(exp)
    print(f"Built {len(candidates)} candidate experiments "
          f"({len(config['pulse_options'])} pulse settings × "
          f"{len(config['tmax_options'])} tmax values).")
    return candidates


def suggest_next_experiment(
    state: Dict[str, Any],
    config: Dict[str, Any],
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Score all candidate experiments by information-gain utility and return the
    best (highest utility-per-cost) optimised design.

    Parameters
    ----------
    state      : pipeline state with a valid posterior
    config     : configuration dict
    candidates : list of candidate experiment dicts.  Built from config if None.

    Returns
    -------
    recommendation : dict with keys
        'optimized_exp' – optimised single-experiment param dict
        'base_exp'      – the best candidate (before time-point pruning)
        'utility'       – EIG / cost score
        'cost'          – total measurement-time cost
        'rank'          – list of all (candidate_index, utility) pairs, sorted
    """
    if state.get("posterior") is None:
        raise RuntimeError("run_inference() must be called before suggest_next_experiment().")

    if candidates is None:
        candidates = build_candidate_experiments(config)

    posterior = state["posterior"]
    hf_df     = state["hf_df"]

    print(f"\nScoring {len(candidates)} candidate experiments …")
    all_results = []
    for i, base_exp in enumerate(candidates):
        if i % 5 == 0:
            print(f"  candidate {i+1}/{len(candidates)} …", flush=True)
        try:
            opt_exp, utility, cost = optimize_experiment(
                base_exp, posterior, hf_df,
                T_budget = config["T_budget"],
                Nt_dense = config["Nt_dense"],
                M        = config["M_eig"],
            )
            all_results.append(dict(
                index         = i,
                base_exp      = base_exp,
                optimized_exp = opt_exp,
                utility       = utility,
                cost          = cost,
            ))
        except Exception as exc:
            warnings.warn(f"Candidate {i} failed: {exc}")

    if not all_results:
        raise RuntimeError("All candidate experiments failed – check config.")

    # rank by utility (highest first)
    all_results.sort(key=lambda r: r["utility"], reverse=True)
    best = all_results[0]

    print(f"\n✓ Best experiment selected:")
    print(f"  Pulses : {best['base_exp']['num_pulses'][0]}")
    print(f"  t_max  : {best['base_exp']['timepoints'][0][-1]:.4f} ms")
    print(f"  # timepoints (pruned): {len(best['optimized_exp']['timepoints'][0])}")
    print(f"  Utility: {best['utility']:.4f}   Cost: {best['cost']:.4f}")

    best["rank"] = [(r["index"], r["utility"]) for r in all_results]
    return best


def display_recommendation(recommendation: Dict[str, Any]) -> None:
    """
    Print and plot a human-readable summary of the recommended next experiment.

    Parameters
    ----------
    recommendation : dict returned by suggest_next_experiment()
    """
    opt = recommendation["optimized_exp"]
    base = recommendation["base_exp"]

    print("\n" + "═" * 55)
    print("  RECOMMENDED NEXT EXPERIMENT")
    print("═" * 55)
    print(f"  Pulse count   : {int(base['num_pulses'][0])}")
    print(f"  Magnetic field: {base['mag_field'][0]:.1f} G")
    print(f"  t_min         : {opt['timepoints'][0][0]:.5f} ms")
    print(f"  t_max         : {opt['timepoints'][0][-1]:.5f} ms")
    print(f"  # time-points : {len(opt['timepoints'][0])}")
    print(f"  Utility score : {recommendation['utility']:.4f}")
    print("═" * 55)

    fig, ax = plt.subplots(figsize=(8, 3))
    times = opt["timepoints"][0]
    ax.vlines(times, 0, 1, colors="steelblue", lw=1.5, alpha=0.7)
    ax.set_xlabel("time (ms)", fontsize=12)
    ax.set_yticks([])
    ax.set_title("Recommended measurement time-points", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_information_diagnostic(
    state: Dict[str, Any],
    recommendation: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Three-panel diagnostic: posterior predictions on a dense grid,
    information density, and the selected measurement time-points.

    Parameters
    ----------
    state          : pipeline state
    recommendation : dict from suggest_next_experiment()
    config         : configuration dict
    """
    base_exp    = recommendation["base_exp"]
    opt_exp     = recommendation["optimized_exp"]
    posterior   = state["posterior"]
    hf_df       = state["hf_df"]

    dense_exp = make_dense_time_experiment(base_exp, config["Nt_dense"])
    times_dense = dense_exp["timepoints"][0]
    times_opt   = opt_exp["timepoints"][0]

    P, w = prediction_matrix(dense_exp, posterior, hf_df)
    mean_pred  = np.average(P, axis=0, weights=w)
    sigma      = dense_exp["noise"][0]
    info       = information_density(dense_exp, posterior, hf_df)
    info_norm  = info / (info.max() + 1e-30)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # ── panel 1: posterior predictions ───────────────────────────────────────
    for k in range(min(P.shape[0], 30)):
        axes[0].plot(times_dense, P[k], color="gray", alpha=0.1, lw=0.7)
    axes[0].plot(times_dense, mean_pred, lw=2, label="posterior mean", color="steelblue")

    gt = config.get("ground_truth_spins")
    if config.get("show_ground_truth") and gt:
        gt_pred = calculate_coherence_with_T2(gt, hf_df, dense_exp)
        axes[0].plot(times_dense, gt_pred[0], lw=2, color="red", label="ground truth")

    axes[0].set_ylabel("coherence", fontsize=12)
    axes[0].set_title("Posterior predictions & information density", fontsize=13)
    axes[0].legend(fontsize=10)

    # ── panel 2: information density ─────────────────────────────────────────
    axes[1].fill_between(times_dense, info_norm, alpha=0.3, color="darkorange")
    axes[1].plot(times_dense, info_norm, lw=1.5, color="darkorange")
    axes[1].scatter(
        times_opt,
        np.interp(times_opt, times_dense, info_norm),
        s=50, zorder=5, color="red", label="selected time-points",
    )
    axes[1].set_xlabel("time (ms)", fontsize=12)
    axes[1].set_ylabel("info density (norm.)", fontsize=12)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_posterior_summary(
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Print and return a ranked summary table of the top spin-bath configurations.

    Parameters
    ----------
    state  : pipeline state
    config : configuration dict (uses 'plot_top_n')

    Returns
    -------
    df : pandas DataFrame  (rank, weight, count, num_spins, (A_par,A_perp) pairs)
    """
    posterior = state.get("posterior")
    if posterior is None:
        print("No posterior available.")
        return pd.DataFrame()

    n_top = config.get("plot_top_n", 5)
    top   = sorted(posterior, key=lambda d: d["weight"], reverse=True)[:n_top]

    rows = []
    for rank, bath in enumerate(top, start=1):
        formatted = "; ".join(
            f"({ap:.2f}, {aperp:.2f})" for ap, aperp in bath["pairs"]
        )
        rows.append(dict(
            rank      = rank,
            weight    = round(bath["weight"], 4),
            count     = bath["count"],
            num_spins = len(bath["pairs"]),
            pairs_Apar_Aperp = formatted,
        ))

    df = pd.DataFrame(rows)
    print(f"\nTop-{n_top} spin-bath configurations  (iteration {state['iteration']}):")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SAVE / RESTORE STATE
# ─────────────────────────────────────────────────────────────────────────────

def save_state(state: Dict[str, Any], path: str) -> None:
    """
    Pickle the full pipeline state to disk for later resumption.

    Parameters
    ----------
    state : pipeline state dict
    path  : file path (e.g. 'run_state.pkl')
    """
    import pickle
    # exclude hf_dist_mat (large) if desired – kept here for full reproducibility
    with open(path, "wb") as fh:
        pickle.dump(state, fh)
    print(f"State saved → {path}")


def load_state(path: str) -> Dict[str, Any]:
    """
    Restore a previously saved pipeline state from disk.

    Parameters
    ----------
    path : file path produced by save_state()

    Returns
    -------
    state : pipeline state dict
    """
    import pickle
    with open(path, "rb") as fh:
        state = pickle.load(fh)
    print(f"State loaded ← {path}  (iteration {state.get('iteration', '?')})")
    return state
