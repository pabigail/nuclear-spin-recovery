import numpy as np
import pandas as pd
from collections import Counter
from scipy.special import logsumexp
from rjmcmc import *


def spin_bath_posterior(list_of_index_lists, hf_df):
    """
    Parameters
    ----------
    list_of_index_lists : list[list[int]]
        Each inner list is a sampled spin bath (list of row indices).
    hf_df : pandas.DataFrame
        Must contain columns 'A_par' and 'A_perp'.

    Returns
    -------
    unique_baths : list[tuple[tuple[float, float], ...]]
        Each element is a canonical spin bath represented as a tuple of
        (A_par, A_perp) pairs, sorted to make order irrelevant.

    weights : np.ndarray
        Posterior probability of each unique bath (sums to 1).
    """

    bath_counter = Counter()

    for idx_list in list_of_index_lists:
        # Map indices -> (A_par, A_perp)
        pairs = [
            (hf_df.iloc[i]['A_par'], hf_df.iloc[i]['A_perp'])
            for i in idx_list
        ]

        # Canonicalize: sort so order doesn't matter, and convert to tuple
        canonical_bath = tuple(sorted(pairs))

        bath_counter[canonical_bath] += 1

    # Extract unique baths and normalize counts
    unique_baths = list(bath_counter.keys())
    counts = np.array(list(bath_counter.values()), dtype=float)

    weights = counts / counts.sum()

    return unique_baths, weights


def spin_bath_posterior_with_indices(list_of_index_lists, hf_df, round_decimals=None):
    """
    Parameters
    ----------
    list_of_index_lists : list[list[int]]
        Each inner list is a sampled spin bath (list of row indices).
    hf_df : pandas.DataFrame
        Must contain columns 'A_par' and 'A_perp'.
    round_decimals : int or None
        If not None, round A_par and A_perp to this many decimals before grouping.

    Returns
    -------
    results : list[dict]
        Each dict has keys:
            - 'pairs'   : canonical tuple of (A_par, A_perp)
            - 'indices' : one representative list of spin indices
            - 'weight'  : posterior probability
            - 'count'   : raw multiplicity in samples
    """

    bath_counter = Counter()
    representative_indices = {}

    for idx_list in list_of_index_lists:
        # Map indices -> (A_par, A_perp)
        pairs = []
        for i in idx_list:
            a_par = hf_df.iloc[i]['A_par']
            a_perp = hf_df.iloc[i]['A_perp']

            if round_decimals is not None:
                a_par = round(a_par, round_decimals)
                a_perp = round(a_perp, round_decimals)

            pairs.append((a_par, a_perp))

        # Canonicalize bath: order-independent, multiplicity preserved
        canonical_bath = tuple(sorted(pairs))

        bath_counter[canonical_bath] += 1

        # Store a representative index list (first one we see)
        if canonical_bath not in representative_indices:
            representative_indices[canonical_bath] = list(idx_list)

    # Normalize to get posterior weights
    unique_baths = list(bath_counter.keys())
    counts = np.array([bath_counter[b] for b in unique_baths], dtype=float)
    weights = counts / counts.sum()

    # Package results
    results = []
    for bath, w, c in zip(unique_baths, weights, counts):
        results.append({
            "pairs": bath,                          # tuple of (A_par, A_perp)
            "indices": representative_indices[bath],  # one index realization
            "weight": w,
            "count": int(c),
        })

    return results


def compute_EIG_particle(
    exp_params,
    posterior_results,
    hf_df,
    M=20,                 # number of synthetic datasets
):
    """
    Monte-Carlo Expected Information Gain estimator.
    """

    sigma = exp_params['noise'][0]
    t_points = exp_params['timepoints'][0]

    weights = np.array([b['weight'] for b in posterior_results])
    weights /= weights.sum()

    log_weights = np.log(weights)

    # Precompute forward predictions for ALL baths
    preds = []
    for bath in posterior_results:
        indices = bath['indices']
        pred = calculate_coherence_with_T2(indices, hf_df, exp_params)
        pred = np.array(pred).flatten()
        preds.append(pred)

    preds = np.array(preds)   # shape (K, Nt)
    print(preds.shape)

    K, Nt = preds.shape
    eig_terms = []

    for _ in range(M):

        # ---- 1. sample true bath ----
        k_true = np.random.choice(K, p=weights)
        mu_true = preds[k_true]

        # ---- 2. simulate data ----
        d = mu_true + sigma * np.random.randn(Nt)

        # ---- 3. likelihood under all baths ----
        # Gaussian log likelihoods
        residuals = preds - d[None, :]
        ll_all = -0.5 * np.sum(
            (residuals / sigma)**2 +
            np.log(2*np.pi*sigma**2),
            axis=1
        )

        # ---- 4. marginal likelihood ----
        log_evidence = logsumexp(log_weights + ll_all)

        # ---- 5. contribution ----
        eig_terms.append(ll_all[k_true] - log_evidence)

    return np.mean(eig_terms)

def precompute_predictions(experiments, posterior_results, hf_df):

    preds = {}

    for ei, exp_params in enumerate(experiments):
        if ei % 10 == 0:
            print(ei)
        bath_preds = []

        for bath in posterior_results:
            pred = calculate_coherence_with_T2(
                bath['indices'], hf_df, exp_params
            )
            pred = np.array(pred).flatten()
            bath_preds.append(pred)

        preds[ei] = np.array(bath_preds)

    return preds


def compute_EIG_all_experiments(
    experiments,
    posterior_results,
    preds,
    M=20,
):

    weights = np.array([b['weight'] for b in posterior_results])
    weights /= weights.sum()
    log_weights = np.log(weights)

    K = len(weights)
    utilities = np.zeros(len(experiments))

    # ---- shared randomness ----
    sampled_particles = np.random.choice(K, size=M, p=weights)

    for m in range(M):
        print(m)

        k_true = sampled_particles[m]

        for ei, exp_params in enumerate(experiments):

            sigma = exp_params['noise'][0]
            pred_all = preds[ei]

            Nt = pred_all.shape[1]

            # shared noise realization
            noise = np.random.randn(Nt)

            d = pred_all[k_true] + sigma * noise

            residuals = pred_all - d[None, :]

            ll_all = -0.5 * np.sum(
                (residuals / sigma)**2 +
                np.log(2*np.pi*sigma**2),
                axis=1
            )

            log_evidence = logsumexp(log_weights + ll_all)

            utilities[ei] += ll_all[k_true] - log_evidence

    utilities /= M

    return utilities

def merge_exp_params(exp_list):
    """
    Merge a list of single-experiment parameter dicts into one
    multi-experiment dict.

    Each exp in exp_list must represent exactly one experiment.
    """

    n = len(exp_list)
    if n == 0:
        raise ValueError("exp_list must contain at least one experiment")

    merged = {}

    # Number of experiments
    merged['num_experiments'] = n

    # Keys that are stored as length-1 arrays in each single experiment
    scalar_keys = ['mag_field', 'noise', 'num_pulses', 'T2']

    for key in scalar_keys:
        merged[key] = np.array([exp[key][0] for exp in exp_list])

    # Timepoints: stack along axis=0 → shape (n, num_tps)
    merged['timepoints'] = [exp['timepoints'][0] for exp in exp_list]

    return merged

def prediction_matrix(exp_params, posterior_results, hf_df):
    """
    Returns:
        P : (K, Nt) prediction matrix
        w : particle weights
    """

    preds = []
    weights = []

    for bath in posterior_results:
        pred = calculate_coherence_with_T2(
            bath["indices"], hf_df, exp_params
        )
        pred = np.array(pred).flatten()
        preds.append(pred)
        weights.append(bath["weight"])

    P = np.array(preds)
    w = np.array(weights)
    w /= w.sum()

    return P, w

def make_dense_time_experiment(base_exp, Nt_dense=300):
    """
    Replace timepoints with dense grid for design stage.
    """

    tmin = base_exp["timepoints"][0][0]
    tmax = base_exp["timepoints"][0][-1]

    dense_times = np.linspace(tmin, tmax, Nt_dense)

    new_exp = dict(base_exp)
    new_exp["timepoints"] = dense_times.reshape(1, -1)

    return new_exp

def information_density(exp_params, posterior_results, hf_df):
    """
    Returns information contribution of each timepoint.
    """

    sigma0 = exp_params["noise"][0]

    P, w = prediction_matrix(exp_params, posterior_results, hf_df)

    mean_pred = np.average(P, axis=0, weights=w)
    Pc = P - mean_pred

    # weighted variance across baths
    density = np.sum(w[:, None] * Pc**2, axis=0) / sigma0**2

    return density


def allocate_measurement_time(info_density, T_budget):
    """
    Allocate averaging time proportional to sqrt(info density).
    (Near-optimal D-design rule)
    """

    weights = np.sqrt(info_density + 1e-12)
    weights /= weights.sum()

    tau = T_budget * weights
    return tau

def prune_timepoints(times, tau, frac_threshold=0.05):
    """
    Keep only timepoints receiving meaningful measurement time.
    """

    mask = tau > frac_threshold * np.max(tau)

    return times[mask], tau[mask]

def build_optimized_experiment(base_exp, times, data_weight):
    exp = dict(base_exp)
    exp["timepoints"] = np.array(times).reshape(1, -1)
    exp["data_weight"] = np.array(data_weight).reshape(1, -1)
    return exp

def optimize_experiment(
    base_exp,
    posterior_results,
    hf_df,
    T_budget,
    Nt_dense=300,
    M=20
    
):

    # ----- STEP 1: dense grid -----
    dense_exp = make_dense_time_experiment(base_exp, Nt_dense)

    times = dense_exp["timepoints"][0]

    # ----- STEP 2: Fisher information -----
    info = information_density(dense_exp, posterior_results, hf_df)

    tau = allocate_measurement_time(info, T_budget)

    # ----- STEP 3: prune -----
    times_opt, tau_opt = prune_timepoints(times, tau)

    optimized_exp = build_optimized_experiment(base_exp, times_opt, tau_opt)

    # ----- STEP 4: MC refinement -----
    eig = compute_EIG_particle(
        optimized_exp,
        posterior_results,
        hf_df,
        M=M
    )

    cost = np.sum(tau_opt)

    utility = eig / cost

    return optimized_exp, utility, cost

def plot_information_diagnostic(
    exp_params_dense,
    optimized_exp,
    posterior_results,
    hf_df,
    spin_list_ground,
):
    """
    Visualize WHY adaptive design chose specific timepoints
    and show ground truth coherence signal.
    """

    # ----- predictions on dense grid -----
    P, w = prediction_matrix(
        exp_params_dense,
        posterior_results,
        hf_df
    )

    times_dense = exp_params_dense["timepoints"][0]
    times_opt = optimized_exp["timepoints"][0]

    # posterior mean
    mean_pred = np.average(P, axis=0, weights=w)

    # information density
    sigma = exp_params_dense["noise"][0]
    Pc = P - mean_pred
    info_density = np.sum(w[:, None] * Pc**2, axis=0) / sigma**2
    info_density /= info_density.max()  # normalize

    # ----- ground truth coherence -----
    coherence_dense = calculate_coherence_with_T2(spin_list_ground, hf_df, exp_params_dense)
    coherence_opt = calculate_coherence_with_T2(spin_list_ground, hf_df, optimized_exp)

    # -------------------------------------------------
    # PLOT
    # -------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # --- TOP: posterior predictions ---
    for k in range(P.shape[0]):
        ax[0].plot(times_dense, P[k], color="gray", alpha=0.15)
    ax[0].plot(times_dense, mean_pred, linewidth=2, label="posterior mean")
    ax[0].plot(times_dense, coherence_dense[0], linewidth=2, label="ground truth")
    ax[0].set_ylabel("Coherence")
    ax[0].set_title("Posterior Predictions")
    ax[0].legend()

    # --- MIDDLE: information density ---
    ax[1].plot(times_dense, info_density, linewidth=2)
    ax[1].scatter(
        times_opt,
        np.interp(times_opt, times_dense, info_density),
        s=40,
        zorder=3,
        color="red",
        label="selected points"
    )
    ax[1].set_ylabel("Information density (norm.)")
    ax[1].legend()

    # --- BOTTOM: ground truth coherence ---
    ax[2].plot(times_dense, coherence_dense[0], '.', label='dense grid points', color='black', ms=10)
    ax[2].plot(times_opt, coherence_opt[0], '.', label='optimized points to measure', color='red', ms=10)
    ax[2].set_xlabel("Time (ms)")
    ax[2].set_ylabel("Coherence")
    ax[2].set_title("Coherence signal (ground truth)")
    ax[2].legend()

    plt.tight_layout()
    plt.show()