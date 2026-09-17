"""Random-walk Metropolis-Hastings, continuous and discrete.

A single parameter is proposed from a domain-aware kernel and accepted with

    alpha = min(1, [L(p') / L(p)] * [prior ratio] * [proposal ratio])

See docs/model-specification.md Sec. 8.1 and Sec. 8.2.
"""

from __future__ import annotations

import numpy as np

from .base import Algorithm


class RWMH(Algorithm):
    """Metropolis-Hastings over a fixed-dimension parameter block."""

    def __init__(self, block, proposal):
        self.block = block
        self.proposal = proposal

    def step(self, state, target, rng, beta=1.0):
        """Propose and accept or reject, independently per replica."""
        current_lp = target.log_prob(state, beta=beta)
        proposed = state.copy()
        log_ratio = (
            self._propose_sites(proposed, rng)
            if self.block.is_discrete
            else self._propose_continuous(proposed, rng)
        )
        proposed_lp = target.log_prob(proposed, beta=beta)

        log_alpha = proposed_lp - current_lp + log_ratio
        accept = np.log(rng.uniform(size=state.n_replicas)) < log_alpha
        return self._merge(state, proposed, accept)

    def _propose_continuous(self, state, rng):
        """Update one column of a per-experiment array, in place.

        Returns the log proposal ratio per replica.
        """
        values = getattr(state, self.block.name)
        column = rng.integers(state.n_exp)
        updated, log_ratio = self.proposal.propose(rng, values[:, column])
        values[:, column] = updated
        return np.full(state.n_replicas, log_ratio, dtype=float)

    def _propose_sites(self, state, rng):
        """Move one spin per replica to a neighbouring free site, in place."""
        log_ratio = np.zeros(state.n_replicas)
        for r in range(state.n_replicas):
            if state.k[r] == 0:
                continue
            slot = int(rng.integers(state.k[r]))
            current = int(state.site_idx[r, slot])
            site, ratio = self.proposal.propose(
                rng, current, occupied=state.occupied[r]
            )
            if site != current:
                state.site_idx[r, slot] = site
                state.occupied[r, current] = False
                state.occupied[r, site] = True
            log_ratio[r] = ratio
        return log_ratio

    @staticmethod
    def _merge(current, proposed, accept):
        """Take the proposed replicas where accepted, the current ones where not."""
        out = current.copy()
        for name in ("site_idx", "k", "lam", "n_stretch", "sigma"):
            getattr(out, name)[accept] = getattr(proposed, name)[accept]
        out.occupied[accept] = proposed.occupied[accept]
        return out
