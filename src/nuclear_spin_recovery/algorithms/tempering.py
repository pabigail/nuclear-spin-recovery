"""Parallel tempering.

J replicas run at inverse temperatures beta_j = 2^-j, zero-indexed so replica 0
is the cold chain sampling the true posterior.  Hot replicas exist to carry
configurations across barriers; only the cold chain is retained.  See
docs/model-specification.md Sec. 8.4.
"""

from __future__ import annotations

import numpy as np

from .base import Algorithm


def geometric_ladder(n_replicas):
    """beta_j = 2 ** -j for j = 0 .. n_replicas - 1, so beta_0 = 1."""
    return [2.0 ** -j for j in range(int(n_replicas))]


class ParallelTempering(Algorithm):
    """Runs an inner schedule on each rung of a temperature ladder.

    The inner argument is a Schedule rather than a single algorithm: a rung
    typically advances both continuous and discrete blocks before a swap is
    attempted.
    """

    def __init__(self, inner, n_replicas=10, betas=None):
        self.inner = inner
        self.n_replicas = int(n_replicas)
        betas = geometric_ladder(self.n_replicas) if betas is None else list(betas)
        if len(betas) != self.n_replicas:
            raise ValueError(
                f"{len(betas)} inverse temperatures for {self.n_replicas} replicas")
        if not np.isclose(betas[0], 1.0):
            raise ValueError(f"beta_0 must be 1, got {betas[0]}")
        if np.any(np.diff(betas) > 0):
            raise ValueError("inverse temperatures must be non-increasing")
        self.betas = betas

    @property
    def block(self):
        """PT has no single block; it inherits whatever its inner schedule updates."""
        return None

    @property
    def label(self) -> str:
        blocks = "+".join(
            step.algorithm.block.name for step in self.inner
            if getattr(step.algorithm, "block", None) is not None)
        return f"pt:{blocks}" if blocks else "pt"

    def step(self, state, target, rng, beta=1.0):
        """Advance every rung one inner pass, then attempt one swap.

        Expects a state already expanded to ``n_replicas`` replicas.
        """
        for step in self.inner:
            for _ in range(step.n_steps):
                state = self._advance_rungs(state, step.algorithm, target, rng)
        return self.attempt_swap(state, target, rng)

    def _advance_rungs(self, state, algorithm, target, rng):
        """One inner step per rung, each against its own inverse temperature.

        Rungs are advanced one at a time because each needs a different beta;
        the algorithm itself is unchanged and unaware of the ladder.
        """
        out = state.copy()
        for j, beta in enumerate(self.betas):
            single = _extract(state, j)
            moved = algorithm.step(single, target, rng, beta=beta)
            _implant(out, j, moved)
        return out

    def attempt_swap(self, state, target, rng):
        """Propose exchanging two rungs, accept or reject, return the state.

        Computed in the reduced form of spec Sec. 8.4,

            log alpha = (beta_a - beta_b) (log L(p_b) - log L(p_a)),

        which needs only the untempered log-likelihoods and is numerically
        stable where the product of four tempered terms is not.
        """
        if self.n_replicas < 2:
            return state
        a, b = rng.choice(self.n_replicas, size=2, replace=False)
        untempered = target.log_prob(state, beta=1.0)
        log_alpha = (self.betas[a] - self.betas[b]) * (
            untempered[b] - untempered[a])
        if np.log(rng.uniform()) >= log_alpha:
            return state
        out = state.copy()
        for name in ("site_idx", "k", "lam", "n_stretch", "sigma",
                     "dA_par", "dA_perp"):
            arr = getattr(out, name)
            arr[[a, b]] = arr[[b, a]]
        out.occupied[[a, b]] = out.occupied[[b, a]]
        return out

    def run(self, state, target, rng, n_steps, trace=None, beta=1.0):
        """Expand to the ladder, advance, then collapse to the cold chain.

        The ladder persists across the whole block: expanding and collapsing
        every step would restart the hot rungs from the cold one each time,
        which is exactly the exploration the ladder exists to accumulate.
        """
        ladder = state.expand_replicas(self.n_replicas)
        for _ in range(int(n_steps)):
            ladder = self.step(ladder, target, rng)
            if trace is not None:
                cold = ladder.collapse_to_cold()
                trace.append(cold, target.log_prob(cold, beta=1.0), self.label)
        return ladder.collapse_to_cold()


def _extract(state, j):
    """Replica j as a standalone single-replica state."""
    out = state.collapse_to_cold()
    for name in ("site_idx", "k", "lam", "n_stretch", "sigma",
                 "dA_par", "dA_perp"):
        getattr(out, name)[0] = getattr(state, name)[j]
    out.occupied[0] = state.occupied[j]
    return out


def _implant(state, j, single):
    """Write a single-replica state back into rung j."""
    for name in ("site_idx", "k", "lam", "n_stretch", "sigma",
                 "dA_par", "dA_perp"):
        getattr(state, name)[j] = getattr(single, name)[0]
    state.occupied[j] = single.occupied[0]
