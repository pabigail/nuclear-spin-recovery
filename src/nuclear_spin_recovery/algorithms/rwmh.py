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
        if self.block.is_discrete:
            log_ratio = self._propose_sites(proposed, rng)
        elif self.block.name == "offsets":
            log_ratio = self._propose_offsets(proposed, rng)
        else:
            log_ratio = self._propose_continuous(proposed, rng)
        proposed_lp = target.log_prob(proposed, beta=beta)

        log_alpha = proposed_lp - current_lp + log_ratio
        accept = np.log(rng.uniform(size=state.n_replicas)) < log_alpha
        out = self._merge(state, proposed, accept)
        if (self.block.name == "offsets"
                and getattr(self.proposal, "redraw_unoccupied", False)):
            self._redraw_unoccupied(out, rng)
        return out

    def _propose_continuous(self, state, rng):
        """Update one column of a per-experiment array, in place.

        Returns the log proposal ratio per replica.
        """
        values = getattr(state, self.block.name)
        column = rng.integers(state.n_exp)
        updated, log_ratio = self.proposal.propose(rng, values[:, column])
        values[:, column] = updated
        return np.full(state.n_replicas, log_ratio, dtype=float)

    def _propose_offsets(self, state, rng):
        """Update one spin's hyperfine offset, in place.

        Returns proposal ratio plus prior ratio: unlike the other blocks the
        offsets carry a proper prior, which must enter the acceptance ratio
        (spec Sec. 5.3).

        A site-scaled kernel is told which site and component it is moving,
        since its width depends on both.
        """
        site_scaled = getattr(self.proposal, "site_scaled", False)
        if site_scaled and not state.has_site_memory:
            raise ValueError(
                "a site-scaled offset kernel needs a state with site memory: "
                "without it an offset travels with its spin from one site's "
                "prior into another's. Build the state with site_memory=True."
            )
        log_ratio = np.zeros(state.n_replicas)
        which = rng.integers(2)
        values = state.dA_par if which == 0 else state.dA_perp
        for r in range(state.n_replicas):
            if state.k[r] == 0:
                continue
            slot = int(rng.integers(state.k[r]))
            current = float(values[r, slot])
            where = ({"site": int(state.site_idx[r, slot]), "component": int(which)}
                     if site_scaled else {})
            proposed, ratio = self.proposal.propose(rng, current, **where)
            state.set_offset(r, slot, which, proposed)
            log_ratio[r] = (ratio
                            + self.proposal.log_prior(proposed, **where)
                            - self.proposal.log_prior(current, **where))
        return log_ratio

    def _redraw_unoccupied(self, state, rng):
        """Draw one unoccupied site's remembered offset afresh from the prior.

        A Gibbs update, applied unconditionally: the likelihood does not
        depend on the offset of a site with no spin on it, so its conditional
        distribution is its prior.
        """
        for r in range(state.n_replicas):
            free = np.flatnonzero(~state.occupied[r])
            if free.size == 0:
                continue
            site = int(rng.choice(free))
            state.site_dA_par[r, site] = self.proposal.draw_prior(
                rng, site=site, component=0)
            state.site_dA_perp[r, site] = self.proposal.draw_prior(
                rng, site=site, component=1)

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
                state.move_spin(r, slot, site)
            log_ratio[r] = ratio
        return log_ratio

    @staticmethod
    def _merge(current, proposed, accept):
        """Take the proposed replicas where accepted, the current ones where not."""
        out = current.copy()
        for name in out.replica_fields():
            getattr(out, name)[accept] = getattr(proposed, name)[accept]
        out.occupied[accept] = proposed.occupied[accept]
        return out
