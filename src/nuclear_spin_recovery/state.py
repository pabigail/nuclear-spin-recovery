"""The sampler state: nuclear spin configuration plus envelope parameters.

Spins are packed into slots ``[0:k)`` of fixed-width arrays; slots beyond k
are undefined.  A leading replica axis is 1 outside a tempering block and J
inside one.  See docs/model-specification.md Sec. 6.
"""

from __future__ import annotations

import numpy as np


class State:
    """Configuration of k nuclear spins and the per-experiment parameters.

    Arrays carry a leading replica axis R.

    site_idx   (R, k_max)   int    index into the SiteTable
    k          (R,)         int    active spin count
    occupied   (R, n_sites) bool   occupancy bitmap, derived from site_idx
    lam        (R, n_exp)   float  decay constant, ms
    n_stretch  (R, n_exp)   float  stretch exponent
    sigma      (R, n_exp)   float  noise std
    dA_par     (R, k_max)   float  hyperfine offset from the table value, kHz
    dA_perp    (R, k_max)   float  hyperfine offset from the table value, kHz

    Offsets are zero unless the ab initio constraint is relaxed (spec Sec. 5.3),
    in which case the model reduces exactly to the constrained one.
    """

    def __init__(self, site_idx, k, lam, n_stretch, sigma, n_sites, k_max,
                 dA_par=None, dA_perp=None):
        self.site_idx = np.asarray(site_idx, dtype=int)
        self.k = np.asarray(k, dtype=int)
        self.lam = np.asarray(lam, dtype=float)
        self.n_stretch = np.asarray(n_stretch, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.n_sites = int(n_sites)
        self.k_max = int(k_max)
        shape = (self.site_idx.shape[0], self.k_max)
        self.dA_par = np.zeros(shape) if dA_par is None else np.asarray(dA_par, float)
        self.dA_perp = np.zeros(shape) if dA_perp is None else np.asarray(dA_perp, float)
        self._occupied = self._derive_occupancy()

    def _derive_occupancy(self):
        occupied = np.zeros((self.n_replicas, self.n_sites), dtype=bool)
        for r in range(self.n_replicas):
            active = self.site_idx[r, : self.k[r]]
            occupied[r, active] = True
        return occupied

    def _active_mask(self):
        """(R, k_max) boolean: which slots hold a live spin."""
        return np.arange(self.k_max)[None, :] < self.k[:, None]

    def _gather(self, values):
        """Gather a per-site array onto active spin slots. (R, k_max)"""
        values = np.asarray(values, dtype=float)
        safe = np.clip(self.site_idx, 0, self.n_sites - 1)
        return np.where(self._active_mask(), values[safe], 0.0)

    @classmethod
    def from_sites(cls, sites, *, n_sites, n_exp, lam, n_stretch, sigma, k_max):
        """Build a single-replica state from a sequence of site indices."""
        sites = np.asarray(list(sites), dtype=int)
        if sites.size > k_max:
            raise ValueError(f"{sites.size} sites exceeds k_max={k_max}")
        if np.any(sites < 0) or np.any(sites >= n_sites):
            raise ValueError(f"site index outside [0, {n_sites})")
        if len(np.unique(sites)) != sites.size:
            raise ValueError("two spins may not occupy the same lattice site")

        site_idx = np.full((1, k_max), -1, dtype=int)
        site_idx[0, : sites.size] = sites
        lam = np.asarray(lam, dtype=float)
        if lam.shape[1] != n_exp:
            raise ValueError(f"lam has {lam.shape[1]} columns, n_exp={n_exp}")
        return cls(
            site_idx=site_idx,
            k=np.array([sites.size]),
            lam=lam,
            n_stretch=np.asarray(n_stretch, dtype=float),
            sigma=np.asarray(sigma, dtype=float),
            n_sites=n_sites,
            k_max=k_max,
        )

    @property
    def n_replicas(self) -> int:
        return int(self.site_idx.shape[0])

    @property
    def n_exp(self) -> int:
        return int(self.lam.shape[1])

    @property
    def occupied(self):
        return self._occupied

    def copy(self):
        """Deep copy; no array is shared with the original."""
        out = State(
            site_idx=self.site_idx.copy(),
            k=self.k.copy(),
            lam=self.lam.copy(),
            n_stretch=self.n_stretch.copy(),
            sigma=self.sigma.copy(),
            n_sites=self.n_sites,
            k_max=self.k_max,
        )
        out._occupied = self._occupied.copy()
        return out

    def expand_replicas(self, n_replicas):
        """Return a state with R identical replicas, for a tempering block."""
        def tile(a):
            return np.repeat(a[:1], n_replicas, axis=0)

        out = State(
            site_idx=tile(self.site_idx),
            k=tile(self.k),
            lam=tile(self.lam),
            n_stretch=tile(self.n_stretch),
            sigma=tile(self.sigma),
            n_sites=self.n_sites,
            k_max=self.k_max,
        )
        return out

    def collapse_to_cold(self):
        """Return replica 0 only, discarding the hot chains."""
        out = State(
            site_idx=self.site_idx[:1].copy(),
            k=self.k[:1].copy(),
            lam=self.lam[:1].copy(),
            n_stretch=self.n_stretch[:1].copy(),
            sigma=self.sigma[:1].copy(),
            n_sites=self.n_sites,
            k_max=self.k_max,
        )
        out._occupied = self._occupied[:1].copy()
        return out

    def gyro_per_spin(self, site_table):
        """Gyromagnetic ratio of each active spin. (R, k_max)

        Determined by the site, never sampled independently.  See spec Sec. 6.
        """
        return self._gather(site_table.gyro)

    def a_par_per_spin(self, site_table):
        """Parallel hyperfine component of each active spin, kHz. (R, k_max)"""
        return self._gather(site_table.a_par)

    def a_perp_per_spin(self, site_table):
        """Perpendicular hyperfine component of each active spin, kHz."""
        return self._gather(site_table.a_perp)

    def check_invariants(self):
        """Raise if the state is malformed.

        Checks that k is within bounds, that occupancy agrees with the active
        slice of site_idx, and that no site is doubly occupied.
        """
        if np.any(self.k < 0) or np.any(self.k > self.k_max):
            raise ValueError(f"k outside [0, {self.k_max}]: {self.k.tolist()}")
        for r in range(self.n_replicas):
            active = self.site_idx[r, : self.k[r]]
            if np.any(active < 0) or np.any(active >= self.n_sites):
                raise ValueError(f"replica {r}: site index outside the table")
            if len(np.unique(active)) != len(active):
                raise ValueError(f"replica {r}: a site is doubly occupied")
            expected = np.zeros(self.n_sites, dtype=bool)
            expected[active] = True
            if not np.array_equal(expected, self._occupied[r]):
                raise ValueError(
                    f"replica {r}: occupancy bitmap disagrees with site_idx"
                )
