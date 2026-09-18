"""Reversible-jump MCMC: birth and death of nuclear spins.

The number of spins is not known a priori, so the model dimension is inferred
alongside the parameters.  See docs/model-specification.md Sec. 8.3.

The prior on k is carried by the dimension kernel rather than appearing as a
separate factor in the acceptance ratio.  Swapping the kernel therefore changes
the effective prior on bath size without touching the sampler.
"""

from __future__ import annotations

import numpy as np

from .base import Algorithm


class BirthDeathKernel:
    """Proposes k -> k+1 or k -> k-1, and reports the kernel ratio.

    Births draw a site uniformly from those unoccupied; deaths remove one of
    the k live spins uniformly.  Those combinatorial factors are where the
    effective prior on bath size lives.
    """

    def __init__(self, k_max, birth_prob=0.5, log_prior_k=None):
        self.k_max = int(k_max)
        self.birth_prob = float(birth_prob)
        #: Callable k -> log p(k).  None means uniform on k up to k_max, which
        #: is the prior the published acceptance ratio implies.
        self.log_prior_k = log_prior_k

    def propose(self, rng, k, n_free):
        """Return ``(k_proposed, move)`` with move in {"birth", "death"}.

        At k = 0 only birth is possible; at k_max only death.  A proposal that
        cannot be made returns the current k with move None.
        """
        k = int(k)
        can_birth = k < self.k_max and n_free > 0
        can_death = k > 0
        if not can_birth and not can_death:
            return k, None
        if not can_birth:
            return k - 1, "death"
        if not can_death:
            return k + 1, "birth"
        if rng.uniform() < self.birth_prob:
            return k + 1, "birth"
        return k - 1, "death"

    def log_ratio(self, k, move, n_free):
        """log gamma(k', k) - log gamma(k, k') for the proposed move.

        Two terms, and omitting the second is a trap.

        The **proposal** ratio for a birth is (p_d / (k+1)) / (p_b / n_free):
        forward draws a site uniformly from the n_free unoccupied ones,
        backward removes one of the k+1 spins uniformly.

        The **prior** ratio carries a combinatorial factor, because a prior that
        is uniform over *configurations* is not uniform over *k* -- there are
        C(n, k) configurations of size k, and that count grows fast.  Writing
        p(config) = p(k) / C(n, k) so the induced prior on k is p(k),

            C(n, k) / C(n, k+1) = (k + 1) / (n - k) = (k + 1) / n_free

        since every unoccupied admissible site is available.  That is exactly
        the inverse of the proposal ratio, so the two cancel and

            log_ratio(birth) = log(p_d / p_b) + log p(k+1) - log p(k)

        with no dependence on n_free at all.

        Dropping the combinatorial term leaves log(n_free / (k+1)), which on a
        3557-site table is about +5.9 for a small bath -- a factor of 365
        favouring every birth, whatever the data says.  The model dimension then
        runs to k_max regardless of the evidence.
        """
        k = int(k)
        p_b, p_d = self.birth_prob, 1.0 - self.birth_prob
        if move == "birth":
            return float(np.log(p_d / p_b) + self._delta_log_prior(k, k + 1))
        if move == "death":
            return float(np.log(p_b / p_d) + self._delta_log_prior(k, k - 1))
        return 0.0

    def _delta_log_prior(self, k_from, k_to):
        """log p(k_to) - log p(k_from); zero for a uniform prior on k."""
        if self.log_prior_k is None:
            return 0.0
        return float(self.log_prior_k(k_to) - self.log_prior_k(k_from))


class RJMCMC(Algorithm):
    """Trans-dimensional moves over the number of spins."""

    def __init__(self, block, kernel):
        self.block = block
        self.kernel = kernel

    def step(self, state, target, rng, beta=1.0):
        current_lp = target.log_prob(state, beta=beta)
        proposed = state.copy()
        log_ratio = np.zeros(state.n_replicas)

        for r in range(state.n_replicas):
            free = np.flatnonzero(~proposed.occupied[r])
            k_new, move = self.kernel.propose(rng, int(proposed.k[r]), free.size)
            if move is None:
                continue
            log_ratio[r] = self.kernel.log_ratio(int(proposed.k[r]), move, free.size)
            if move == "birth":
                self._birth(proposed, r, int(rng.choice(free)))
            else:
                self._death(proposed, r, int(rng.integers(proposed.k[r])))

        proposed_lp = target.log_prob(proposed, beta=beta)
        accept = np.log(rng.uniform(size=state.n_replicas)) < (
            proposed_lp - current_lp + log_ratio)
        return self._merge(state, proposed, accept)

    @staticmethod
    def _birth(state, r, site):
        """Append a spin at ``site``, with no offset from its table value."""
        slot = int(state.k[r])
        state.site_idx[r, slot] = site
        state.dA_par[r, slot] = 0.0
        state.dA_perp[r, slot] = 0.0
        state.occupied[r, site] = True
        state.k[r] = slot + 1

    @staticmethod
    def _death(state, r, slot):
        """Remove one spin, keeping slots [0:k) contiguous.

        Swap-with-last rather than shift: O(1), and the offsets must move with
        their spin or they would be silently reassigned.
        """
        last = int(state.k[r]) - 1
        state.occupied[r, state.site_idx[r, slot]] = False
        if slot != last:
            for arr in (state.site_idx, state.dA_par, state.dA_perp):
                arr[r, slot] = arr[r, last]
        state.site_idx[r, last] = -1
        state.dA_par[r, last] = 0.0
        state.dA_perp[r, last] = 0.0
        state.k[r] = last

    @staticmethod
    def _merge(current, proposed, accept):
        out = current.copy()
        for name in ("site_idx", "k", "lam", "n_stretch", "sigma",
                     "dA_par", "dA_perp"):
            getattr(out, name)[accept] = getattr(proposed, name)[accept]
        out.occupied[accept] = proposed.occupied[accept]
        return out
