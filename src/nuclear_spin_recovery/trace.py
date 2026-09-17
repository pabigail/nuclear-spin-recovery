"""Recorded chain history.

A :class:`Trace` is the concatenated output of every sub-algorithm in a
schedule.  Only the cold chain is ever recorded: hot tempering replicas exist
to ferry configurations across barriers and do not target the posterior.
See docs/model-specification.md Sec. 8.4 and Sec. 8.5.
"""

from __future__ import annotations

import numpy as np


class Trace:
    """Append-only record of states visited by the sampler.

    Arrays are indexed (n_steps, ...) and hold replica 0 only.
    """

    def __init__(self, n_sites, k_max, n_exp):
        self.n_sites = int(n_sites)
        self.k_max = int(k_max)
        self.n_exp = int(n_exp)
        self._site_idx = []
        self._k = []
        self._lam = []
        self._n_stretch = []
        self._sigma = []
        self._log_prob = []
        self._algorithm = []

    def __len__(self) -> int:
        return len(self._k)

    def append(self, state, log_prob, algorithm=""):
        """Record the cold replica of ``state``.

        Values are copied, not referenced: the sampler mutates states in place
        between steps, and a trace holding views would silently rewrite its own
        history.
        """
        if state.k_max != self.k_max:
            raise ValueError(
                f"state k_max={state.k_max} does not match trace k_max={self.k_max}"
            )
        if state.n_exp != self.n_exp:
            raise ValueError(
                f"state n_exp={state.n_exp} does not match trace n_exp={self.n_exp}"
            )
        lp = np.asarray(log_prob, dtype=float)
        self._site_idx.append(np.array(state.site_idx[0], dtype=int))
        self._k.append(int(state.k[0]))
        self._lam.append(np.array(state.lam[0], dtype=float))
        self._n_stretch.append(np.array(state.n_stretch[0], dtype=float))
        self._sigma.append(np.array(state.sigma[0], dtype=float))
        self._log_prob.append(float(lp.reshape(-1)[0]))
        self._algorithm.append(str(algorithm))

    @property
    def site_idx(self):
        """(n_steps, k_max) int"""
        if not self._site_idx:
            return np.empty((0, self.k_max), dtype=int)
        return np.array(self._site_idx, dtype=int)

    @property
    def k(self):
        """(n_steps,) int"""
        if not self._k:
            return np.empty((0,), dtype=int)
        return np.array(self._k, dtype=int)

    @property
    def lam(self):
        """(n_steps, n_exp) float"""
        if not self._lam:
            return np.empty((0, self.n_exp), dtype=float)
        return np.array(self._lam, dtype=float)

    @property
    def n_stretch(self):
        """(n_steps, n_exp) float"""
        if not self._n_stretch:
            return np.empty((0, self.n_exp), dtype=float)
        return np.array(self._n_stretch, dtype=float)

    @property
    def sigma(self):
        """(n_steps, n_exp) float"""
        if not self._sigma:
            return np.empty((0, self.n_exp), dtype=float)
        return np.array(self._sigma, dtype=float)

    @property
    def log_prob(self):
        """(n_steps,) float"""
        if not self._log_prob:
            return np.empty((0,), dtype=float)
        return np.array(self._log_prob, dtype=float)

    @property
    def algorithm(self):
        """(n_steps,) str -- which sub-algorithm produced each step."""
        return np.array(self._algorithm, dtype=object)

    def discard_burn_in(self, n_burn):
        """Return a new Trace holding the steps after ``n_burn``."""
        n_burn = int(n_burn)
        if n_burn >= len(self):
            raise ValueError(
                f"discarding {n_burn} of {len(self)} steps would leave nothing"
            )
        out = Trace(self.n_sites, self.k_max, self.n_exp)
        out._site_idx = list(self._site_idx[n_burn:])
        out._k = list(self._k[n_burn:])
        out._lam = list(self._lam[n_burn:])
        out._n_stretch = list(self._n_stretch[n_burn:])
        out._sigma = list(self._sigma[n_burn:])
        out._log_prob = list(self._log_prob[n_burn:])
        out._algorithm = list(self._algorithm[n_burn:])
        return out
