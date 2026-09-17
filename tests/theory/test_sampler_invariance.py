"""T0: does the sampler target the distribution it claims to?

Recovery tests answer "did we find the right answer". They do not answer "are
we sampling the right distribution", and the two come apart: a chain with a
wrong acceptance ratio still concentrates near the likelihood maximum and so
still passes a recovery test, while reporting posterior widths and model
probabilities that are wrong.

These tests use targets whose stationary distribution is known in closed form,
so the chain can be checked against it directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from nuclear_spin_recovery import (
    ContinuousReflected,
    DiscreteLatticeWalk,
    NeighborIndex,
    ParameterBlock,
    RWMH,
    State,
    Trace,
)

pytestmark = pytest.mark.slow

# A line of sites with deliberately uneven spacing, so the number of
# neighbours within the walk radius differs from site to site.  On a uniform
# lattice the proposal ratio is one almost everywhere and a missing correction
# would go unnoticed.
UNEVEN = np.array(
    [[0.0, 0, 0], [1.0, 0, 0], [1.6, 0, 0], [2.1, 0, 0], [3.4, 0, 0], [4.2, 0, 0]]
)
RADIUS = 1.2


class GaussianLam:
    """log pi(lam) for a Gaussian truncated to the lambda domain."""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, state, beta=1.0):
        lam = state.lam[:, 0]
        return beta * (-0.5 * ((lam - self.mu) / self.sigma) ** 2)


class Flat:
    """Uniform over whatever the sampler is free to change."""

    def log_prob(self, state, beta=1.0):
        return np.zeros(state.n_replicas)


def _state(sites, n_sites, lam=0.5, k_max=4):
    return State.from_sites(
        sites, n_sites=n_sites, n_exp=1,
        lam=np.array([[lam]]), n_stretch=np.array([[1.0]]),
        sigma=np.array([[0.1]]), k_max=k_max,
    )


def _run(algorithm, state, target, n_steps, seed, n_sites, k_max):
    trace = Trace(n_sites=n_sites, k_max=k_max, n_exp=1)
    algorithm.run(state, target, np.random.default_rng(seed),
                  n_steps=n_steps, trace=trace)
    return trace


# ------------------------------------------------------- continuous kernel

def test_continuous_recovers_a_gaussian():
    """RWMH on a Gaussian target must reproduce that Gaussian."""
    mu, sigma = 0.5, 0.08
    mover = RWMH(ParameterBlock("lam"), ContinuousReflected(0.05, 0.0, 1.0))
    trace = _run(mover, _state((0,), 6, lam=mu), GaussianLam(mu, sigma),
                 n_steps=60_000, seed=0, n_sites=6, k_max=4)

    samples = trace.lam[10_000:, 0]
    assert np.mean(samples) == pytest.approx(mu, abs=0.01)
    assert np.std(samples) == pytest.approx(sigma, rel=0.1)


def test_continuous_passes_a_goodness_of_fit_test():
    mu, sigma = 0.5, 0.08
    mover = RWMH(ParameterBlock("lam"), ContinuousReflected(0.05, 0.0, 1.0))
    trace = _run(mover, _state((0,), 6, lam=mu), GaussianLam(mu, sigma),
                 n_steps=60_000, seed=1, n_sites=6, k_max=4)

    # Thin to reduce autocorrelation before a test that assumes independence.
    samples = trace.lam[10_000::50, 0]
    assert stats.kstest(samples, "norm", args=(mu, sigma)).pvalue > 0.01


def test_continuous_is_uniform_under_a_flat_target():
    """With no target gradient, reflection must fill the domain evenly."""
    mover = RWMH(ParameterBlock("lam"), ContinuousReflected(0.07, 0.0, 1.0))
    trace = _run(mover, _state((0,), 6, lam=0.5), Flat(),
                 n_steps=60_000, seed=2, n_sites=6, k_max=4)

    samples = trace.lam[10_000::50, 0]
    assert stats.kstest(samples, "uniform").pvalue > 0.01


# --------------------------------------------------------- discrete kernel

def test_discrete_is_uniform_under_a_flat_target():
    """The test that catches a missing proposal-ratio correction.

    Under a flat target the stationary distribution over sites is uniform.
    An implementation that treats the occupancy-constrained walk as symmetric
    instead converges to something proportional to 1 / |N_R(x)|, which on an
    unevenly spaced lattice is visibly non-uniform -- yet such an
    implementation passes every recovery test in the phase 3 ladder.
    """
    n_sites = len(UNEVEN)
    mover = RWMH(ParameterBlock("sites"),
                 DiscreteLatticeWalk(NeighborIndex(UNEVEN, RADIUS)))
    trace = _run(mover, _state((0,), n_sites), Flat(),
                 n_steps=200_000, seed=3, n_sites=n_sites, k_max=4)

    visits = np.bincount(trace.site_idx[20_000:, 0], minlength=n_sites)
    expected = np.full(n_sites, visits.sum() / n_sites)
    assert stats.chisquare(visits, expected).pvalue > 0.01


def test_neighbour_counts_actually_vary():
    """Guards the test above: on a uniform lattice it would prove nothing."""
    idx = NeighborIndex(UNEVEN, RADIUS)
    free = np.zeros(len(UNEVEN), dtype=bool)
    counts = {idx.count_available(i, free) for i in range(len(UNEVEN))}
    assert len(counts) > 1


def test_discrete_respects_occupancy_in_the_stationary_law():
    """With two spins, no sample may place both on one site."""
    n_sites = len(UNEVEN)
    mover = RWMH(ParameterBlock("sites"),
                 DiscreteLatticeWalk(NeighborIndex(UNEVEN, RADIUS)))
    trace = _run(mover, _state((0, 3), n_sites), Flat(),
                 n_steps=40_000, seed=4, n_sites=n_sites, k_max=4)

    for step in range(len(trace)):
        active = trace.site_idx[step, : trace.k[step]]
        assert len(set(active.tolist())) == len(active)


def test_discrete_reaches_every_connected_site():
    """A chain that cannot reach part of the space is not ergodic on it."""
    n_sites = len(UNEVEN)
    mover = RWMH(ParameterBlock("sites"),
                 DiscreteLatticeWalk(NeighborIndex(UNEVEN, RADIUS)))
    trace = _run(mover, _state((0,), n_sites), Flat(),
                 n_steps=40_000, seed=5, n_sites=n_sites, k_max=4)
    assert set(trace.site_idx[:, 0].tolist()) == set(range(n_sites))


# ------------------------------------------------------------ reversibility

def test_discrete_kernel_satisfies_detailed_balance():
    """Enumerate the one-spin transition matrix and check pi P = pi P^T.

    Under a flat target pi is uniform, so detailed balance reduces to the
    transition matrix being symmetric.  This is an exact algebraic check that
    does not depend on chain length or on any statistical threshold.
    """
    idx = NeighborIndex(UNEVEN, RADIUS)
    walk = DiscreteLatticeWalk(idx)
    n = len(UNEVEN)
    free = np.zeros(n, dtype=bool)

    # P(x -> z) = (1 / n_avail(x)) * min(1, exp(log_ratio)) for z a free
    # neighbour of x, under a flat target.
    P = np.zeros((n, n))
    for x in range(n):
        avail = [z for z in idx.neighbors(x) if not free[z]]
        if not avail:
            P[x, x] = 1.0
            continue
        for z in avail:
            fwd = 1.0 / len(avail)
            rev_avail = len([y for y in idx.neighbors(z) if not free[y]])
            accept = min(1.0, len(avail) / rev_avail)
            P[x, z] = fwd * accept
        P[x, x] = 1.0 - P[x].sum() + P[x, x]

    assert P == pytest.approx(P.T, abs=1e-12)
