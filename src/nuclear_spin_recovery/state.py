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

    **Whose offset is it?**  By default an offset belongs to the spin: it
    travels with the spin across a site move, and a newborn spin starts at
    zero.  With *site memory* an offset belongs to the site instead:

    site_dA_par   (R, n_sites) float  offset each site was last left at, kHz
    site_dA_perp  (R, n_sites) float

    A spin arriving at a site, by a move or by a birth, takes up the offset
    that site remembers, and a spin leaving one leaves its offset behind.  A
    site that has never been occupied remembers zero.  The per-spin arrays are
    still what the forward model reads; the memory is kept equal to them on
    every occupied site.  It is ``None`` when site memory is off.

    Moves change a state through :meth:`add_spin`, :meth:`remove_spin`,
    :meth:`move_spin` and :meth:`set_offset`, which is what keeps the two in
    step.  Writing to ``dA_par`` directly is fine with site memory off, and
    with it on must be followed by :meth:`remember_offsets`.
    """

    #: Per-replica arrays that every state carries.  Anything that copies,
    #: merges or swaps replicas iterates :meth:`replica_fields`, never a list
    #: of its own.
    FIELDS = ("site_idx", "k", "lam", "n_stretch", "sigma", "dA_par", "dA_perp")
    #: Per-replica arrays present only with site memory.
    MEMORY_FIELDS = ("site_dA_par", "site_dA_perp")

    def __init__(self, site_idx, k, lam, n_stretch, sigma, n_sites, k_max,
                 dA_par=None, dA_perp=None, site_dA_par=None, site_dA_perp=None):
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
        if (site_dA_par is None) != (site_dA_perp is None):
            raise ValueError("site memory needs both components or neither")
        self.site_dA_par = (None if site_dA_par is None
                            else np.asarray(site_dA_par, float))
        self.site_dA_perp = (None if site_dA_perp is None
                             else np.asarray(site_dA_perp, float))
        self._occupied = self._derive_occupancy()

    @property
    def has_site_memory(self) -> bool:
        """Whether offsets belong to sites rather than to spins."""
        return self.site_dA_par is not None

    def replica_fields(self):
        """Names of every per-replica array this state carries.

        ``occupied`` is not among them: it is derived, and is copied alongside.
        """
        return self.FIELDS + (self.MEMORY_FIELDS if self.has_site_memory else ())

    def enable_site_memory(self):
        """Turn site memory on, remembering the offsets the spins have now.

        Every unoccupied site starts at zero, its table value.
        """
        shape = (self.n_replicas, self.n_sites)
        self.site_dA_par = np.zeros(shape)
        self.site_dA_perp = np.zeros(shape)
        self.remember_offsets()
        return self

    def remember_offsets(self):
        """Write every live spin's offset to its site's memory."""
        if not self.has_site_memory:
            return
        for r in range(self.n_replicas):
            sites = self.site_idx[r, : self.k[r]]
            self.site_dA_par[r, sites] = self.dA_par[r, : self.k[r]]
            self.site_dA_perp[r, sites] = self.dA_perp[r, : self.k[r]]

    # -- mutation ----------------------------------------------------------

    def _arrive(self, r, slot, site):
        """Offset of a spin newly placed on ``site``: remembered, or zero."""
        if self.has_site_memory:
            self.dA_par[r, slot] = self.site_dA_par[r, site]
            self.dA_perp[r, slot] = self.site_dA_perp[r, site]

    def add_spin(self, r, site):
        """Append a spin at ``site`` in replica ``r``.

        It starts at its table value, or with site memory at the offset the
        site remembers.
        """
        slot = int(self.k[r])
        self.site_idx[r, slot] = site
        self.dA_par[r, slot] = 0.0
        self.dA_perp[r, slot] = 0.0
        self._arrive(r, slot, site)
        self._occupied[r, site] = True
        self.k[r] = slot + 1

    def remove_spin(self, r, slot):
        """Remove the spin in ``slot``, keeping slots [0:k) contiguous.

        Swap-with-last rather than shift: O(1), and the offsets must move with
        their spin or they would be silently reassigned.  With site memory the
        vacated site keeps its offset.
        """
        last = int(self.k[r]) - 1
        self._occupied[r, self.site_idx[r, slot]] = False
        if slot != last:
            for arr in (self.site_idx, self.dA_par, self.dA_perp):
                arr[r, slot] = arr[r, last]
        self.site_idx[r, last] = -1
        self.dA_par[r, last] = 0.0
        self.dA_perp[r, last] = 0.0
        self.k[r] = last

    def move_spin(self, r, slot, site):
        """Move the spin in ``slot`` to the unoccupied ``site``.

        Its offset goes with it, or with site memory is exchanged for the one
        the destination remembers.
        """
        current = int(self.site_idx[r, slot])
        self.site_idx[r, slot] = site
        self._occupied[r, current] = False
        self._occupied[r, site] = True
        self._arrive(r, slot, site)

    def set_offset(self, r, slot, component, value):
        """Set one offset of the spin in ``slot``.

        ``component`` is 0 for the parallel offset and 1 for the perpendicular.
        """
        live = self.dA_par if component == 0 else self.dA_perp
        live[r, slot] = value
        if self.has_site_memory:
            memory = self.site_dA_par if component == 0 else self.site_dA_perp
            memory[r, self.site_idx[r, slot]] = value

    def _derive_occupancy(self):
        occupied = np.zeros((self.n_replicas, self.n_sites), dtype=bool)
        for r in range(self.n_replicas):
            active = self.site_idx[r, : self.k[r]]
            occupied[r, active] = True
        return occupied

    def _active_mask(self):
        """(R, k_max) boolean: which slots hold a live spin."""
        return np.arange(self.k_max)[None, :] < self.k[:, None]

    def _gather(self, values, offset=None):
        """Gather a per-site array onto active spin slots. (R, k_max)

        ``offset`` is added to live slots only, so relaxing the ab initio
        constraint (spec Sec. 5.3) reduces exactly to the constrained model
        when the offsets are zero.
        """
        values = np.asarray(values, dtype=float)
        safe = np.clip(self.site_idx, 0, self.n_sites - 1)
        gathered = values[safe]
        if offset is not None:
            gathered = gathered + offset
        return np.where(self._active_mask(), gathered, 0.0)

    @classmethod
    def from_sites(cls, sites, *, n_sites, n_exp, lam, n_stretch, sigma, k_max,
                   site_memory=False):
        """Build a single-replica state from a sequence of site indices.

        ``site_memory`` makes offsets belong to sites rather than to spins.
        """
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
        out = cls(
            site_idx=site_idx,
            k=np.array([sites.size]),
            lam=lam,
            n_stretch=np.asarray(n_stretch, dtype=float),
            sigma=np.asarray(sigma, dtype=float),
            n_sites=n_sites,
            k_max=k_max,
        )
        return out.enable_site_memory() if site_memory else out

    @property
    def n_replicas(self) -> int:
        return int(self.site_idx.shape[0])

    @property
    def n_exp(self) -> int:
        return int(self.lam.shape[1])

    @property
    def occupied(self):
        return self._occupied

    def _rebuild(self, transform):
        """A new state with ``transform`` applied to every per-replica array."""
        arrays = {name: transform(getattr(self, name))
                  for name in self.replica_fields()}
        return State(n_sites=self.n_sites, k_max=self.k_max, **arrays)

    def copy(self):
        """Deep copy; no array is shared with the original."""
        out = self._rebuild(np.copy)
        out._occupied = self._occupied.copy()
        return out

    def expand_replicas(self, n_replicas):
        """Return a state with R identical replicas, for a tempering block."""
        return self._rebuild(lambda a: np.repeat(a[:1], n_replicas, axis=0))

    def collapse_to_cold(self):
        """Return replica 0 only, discarding the hot chains."""
        out = self._rebuild(lambda a: a[:1].copy())
        out._occupied = self._occupied[:1].copy()
        return out

    def gyro_per_spin(self, site_table):
        """Gyromagnetic ratio of each active spin. (R, k_max)

        Determined by the site, never sampled independently.  See spec Sec. 6.
        """
        return self._gather(site_table.gyro)

    def a_par_per_spin(self, site_table):
        """Parallel hyperfine component of each active spin, kHz. (R, k_max)"""
        return self._gather(site_table.a_par, self.dA_par)

    def a_perp_per_spin(self, site_table):
        """Perpendicular hyperfine component of each active spin, kHz."""
        return self._gather(site_table.a_perp, self.dA_perp)

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
            if self.has_site_memory:
                k = self.k[r]
                if not (np.array_equal(self.site_dA_par[r, active],
                                       self.dA_par[r, :k])
                        and np.array_equal(self.site_dA_perp[r, active],
                                           self.dA_perp[r, :k])):
                    raise ValueError(
                        f"replica {r}: a spin's offset disagrees with what "
                        f"its site remembers"
                    )
