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
        self._dA_par = []
        self._dA_perp = []
        self._log_prob = []
        self._algorithm = []
        # Building an array from the append lists is O(n).  Without a cache a
        # caller that reads a property once per step is O(n^2) -- 20,000 steps
        # cost 32 s of pure rebuilding.  Arrays are handed out read-only so the
        # cache cannot be invalidated behind our back.
        self._cache = {}

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
        self._dA_par.append(np.array(state.dA_par[0], dtype=float))
        self._dA_perp.append(np.array(state.dA_perp[0], dtype=float))
        self._log_prob.append(float(lp.reshape(-1)[0]))
        self._algorithm.append(str(algorithm))
        self._cache.clear()

    @property
    def site_idx(self):
        """(n_steps, k_max) int"""
        return self._build("site_idx", self._site_idx, int, (0, self.k_max))

    @property
    def k(self):
        """(n_steps,) int"""
        return self._build("k", self._k, int, (0,))

    @property
    def lam(self):
        """(n_steps, n_exp) float"""
        return self._build("lam", self._lam, float, (0, self.n_exp))

    @property
    def n_stretch(self):
        """(n_steps, n_exp) float"""
        return self._build("n_stretch", self._n_stretch, float, (0, self.n_exp))

    @property
    def sigma(self):
        """(n_steps, n_exp) float"""
        return self._build("sigma", self._sigma, float, (0, self.n_exp))

    @property
    def dA_par(self):
        """(n_steps, k_max) float -- offset from the table A_parallel, kHz."""
        return self._build("dA_par", self._dA_par, float, (0, self.k_max))

    @property
    def dA_perp(self):
        """(n_steps, k_max) float -- offset from the table A_perp, kHz."""
        return self._build("dA_perp", self._dA_perp, float, (0, self.k_max))

    @property
    def log_prob(self):
        """(n_steps,) float"""
        return self._build("log_prob", self._log_prob, float, (0,))

    @property
    def algorithm(self):
        """(n_steps,) str -- which sub-algorithm produced each step."""
        return self._build("algorithm", self._algorithm, object, (0,))

    def _build(self, name, store, dtype, empty_shape):
        """Return the cached array for ``name``, building it if needed.

        The result is read-only: it is shared with every other caller and with
        the next access, so a mutation would silently rewrite recorded history.
        """
        cached = self._cache.get(name)
        if cached is None:
            cached = (np.empty(empty_shape, dtype=dtype) if not store
                      else np.array(store, dtype=dtype))
            cached.flags.writeable = False
            self._cache[name] = cached
        return cached

    def save(self, path):
        """Write this trace to ``path`` as a compressed ``.npz``.

        Compressed because the padding compresses: site_idx beyond k is -1 and
        the offsets are mostly zero, so a 25,000-step trace at k_max = 64 goes
        from 39.4 MB in memory to 6.8 MB on disk.  Twenty ensembles are 136 MB,
        which keeps pooling a local operation.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        """Read a trace written by :meth:`save`.

        Round-trips everything, the per-step algorithm labels included -- they
        are what lets a pooled trace still be read block by block.
        """
        raise NotImplementedError

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
        out._dA_par = list(self._dA_par[n_burn:])
        out._dA_perp = list(self._dA_perp[n_burn:])
        out._log_prob = list(self._log_prob[n_burn:])
        out._algorithm = list(self._algorithm[n_burn:])
        return out
