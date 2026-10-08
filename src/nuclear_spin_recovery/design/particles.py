"""The posterior as a weighted set of distinct baths.

Everything downstream of the sampler sees particles, never a chain.  A particle
is one physical hypothesis -- a bath -- and its weight is the fraction of
posterior draws that visited it.

**Canonicalised on couplings, not on site index.**  The NV centre's symmetry
puts lattice sites into orbits that are spatially distinct but hyperfine-
identical, and no coherence measurement can tell them apart.  Grouping draws by
site index would split one hypothesis into as many particles as its orbit has
members, inflate K, and make the posterior look less certain than it is --
which is exactly the quantity a design engine reads.  Two draws are the same
particle when they have the same k and their spins can be paired one-to-one
with both couplings within ``tol``; offsets are included, so a relaxed run is
not read as a constrained one.

**Relaxed posteriors need a looser tolerance.**  The default ``tol`` is the
detection tolerance, 0.1 kHz, which suits couplings pinned to their table
values.  When offsets are sampled a coupling moves by kilohertz within one
hypothesis, and at 0.1 kHz nearly every draw is its own particle: measured on
a ten-site toy posterior, 1,725 particles from 1,800 draws, against 128 at
3 kHz.  :meth:`ParticleSet.from_trace` warns when that happens.

Sorting the pairs and comparing elementwise is **not** that test.  Under a
tolerance there is no canonical order: an orbit swap can move a spin past a
different spin whose A_par lies within ``tol`` of it, and the elementwise
comparison then pairs each spin with the wrong partner and splits one bath in
two.  On the NV table that is common rather than exotic -- 35,580 pairs of
distinct spins lie within 0.1 kHz of each other in A_par.

**Each particle carries its own envelope.**  Predictions depend on lambda, the
stretch exponent and sigma as well as the bath, and a particle is a group of
draws that agree on the bath but not necessarily on those.  A particle holds
their mean over its member draws -- the envelope conditional on that bath --
rather than a global value that no draw of the bath need have visited.

**Effective size, not draw count.**  Four hundred draws that collapsed onto two
baths are two hypotheses.  Kish's ``1 / sum(w^2)`` says so, and the finite-K
bias of the EIG estimator is a function of it rather than of the draw count.

See docs/phase-5-plan.md, unit 5a.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..post.detection import MATCH_TOL
from ..post.residual import predictive_from_arrays

#: A grouping that leaves more than this share of the draws as particles of
#: their own has probably not grouped anything; warned about, from this many
#: draws up.
_FRAGMENT_SHARE = 0.5
_FRAGMENT_MIN_DRAWS = 50


class ParticleSet:
    """K distinct baths with normalised weights.

    Arrays carry a leading particle axis K, laid out as :class:`~nuclear_spin_
    recovery.state.State` lays out replicas, so a particle set becomes a
    single K-replica state for one vectorised forward pass.

    site_idx   (K, k_max)   int    representative site of each spin
    k          (K,)         int    active spin count
    weight     (K,)         float  posterior mass, non-negative, sums to 1
    dA_par     (K, k_max)   float  hyperfine offsets of the representative, kHz
    dA_perp    (K, k_max)   float
    lam        (K, n_exp)   float  envelope, mean over the particle's draws
    n_stretch  (K, n_exp)   float
    sigma      (K, n_exp)   float

    Weights given by hand are normalised, so multiplicities may be passed
    directly.  Negative or all-zero weights raise.  Particles are held in
    order of decreasing weight, ties in the order given.

    A single particle is legal: it is what a collapsed posterior looks like,
    and saying so is the designer's job, not a reason to refuse construction.
    """

    def __init__(self, site_idx, k, weight, dA_par, dA_perp, lam, n_stretch,
                 sigma, n_sites, k_max):
        self.n_sites = int(n_sites)
        self.k_max = int(k_max)
        site_idx = np.asarray(site_idx, dtype=int).reshape(-1, self.k_max)
        n = site_idx.shape[0]
        k = np.asarray(k, dtype=int).reshape(-1)
        weight = np.asarray(weight, dtype=float).reshape(-1)
        if n == 0:
            raise ValueError("a particle set needs at least one particle")
        if k.shape != (n,) or weight.shape != (n,):
            raise ValueError(
                f"{n} particles but {k.size} spin counts and {weight.size} weights")
        if not np.all(np.isfinite(weight)) or np.any(weight < 0):
            raise ValueError("weights must be finite and non-negative")
        total = weight.sum()
        if total <= 0.0:
            raise ValueError("weights sum to zero, so the posterior has no mass")

        def per_particle(values, width, name):
            values = np.asarray(values, dtype=float)
            if values.ndim != 2 or values.shape[0] != n or (
                    width is not None and values.shape[1] != width):
                raise ValueError(f"{name} has shape {values.shape}, "
                                 f"expected ({n}, {width or 'n_exp'})")
            return values

        dA_par = per_particle(dA_par, self.k_max, "dA_par")
        dA_perp = per_particle(dA_perp, self.k_max, "dA_perp")
        lam = per_particle(lam, None, "lam")
        n_stretch = per_particle(n_stretch, lam.shape[1], "n_stretch")
        sigma = per_particle(sigma, lam.shape[1], "sigma")

        order = np.argsort(-weight, kind="stable")
        self.site_idx = site_idx[order]
        self.k = k[order]
        self.weight = weight[order] / total
        self.dA_par = dA_par[order]
        self.dA_perp = dA_perp[order]
        self.lam = lam[order]
        self.n_stretch = n_stretch[order]
        self.sigma = sigma[order]

    @classmethod
    def from_trace(cls, trace, site_table, *, burn=0, stride=1, tol=MATCH_TOL):
        """Group a trace's draws into weighted particles.

        ``burn`` steps are discarded, then every ``stride``-th draw is kept.
        Each surviving draw is one unit of multiplicity.  The representative of
        a particle -- its site indices and offsets -- is the first draw that
        founded it; the envelope is the mean over all of its draws.

        Pass a **pooled ensemble** trace, not one chain.  A single chain that
        never left a wrong configuration produces a confident one-particle
        posterior, and the designer will then very efficiently separate
        hypotheses that are all wrong (docs/phase-5-plan.md Sec. 5).
        """
        rows = np.arange(len(trace))[max(0, int(burn))::max(1, int(stride))]
        if rows.size == 0:
            raise ValueError(
                f"no draws survive burn={burn}, stride={stride} "
                f"from a trace of {len(trace)} steps")

        site_idx = np.asarray(trace.site_idx)[rows]
        k = np.asarray(trace.k)[rows]
        dA_par = np.asarray(trace.dA_par)[rows]
        dA_perp = np.asarray(trace.dA_perp)[rows]
        envelope = np.stack([np.asarray(trace.lam)[rows],
                             np.asarray(trace.n_stretch)[rows],
                             np.asarray(trace.sigma)[rows]])

        a_par = np.asarray(site_table.a_par, dtype=float)
        a_perp = np.asarray(site_table.a_perp, dtype=float)

        # Greedy first-match grouping, as coupling_posterior does.  Candidates
        # are bucketed by k, since baths of different size are never one.
        founders = []                  # row of the founding draw, per particle
        signatures = []                # (k, 2) couplings of the founder
        count = []
        env_sum = []
        by_k = {}
        for j in range(rows.size):
            n_spin = int(k[j])
            sites = site_idx[j, :n_spin]
            pairs = np.column_stack([a_par[sites] + dA_par[j, :n_spin],
                                     a_perp[sites] + dA_perp[j, :n_spin]])
            for p in by_k.get(n_spin, ()):
                if _same_bath(pairs, signatures[p], tol):
                    count[p] += 1
                    env_sum[p] += envelope[:, j]
                    break
            else:
                by_k.setdefault(n_spin, []).append(len(founders))
                founders.append(j)
                signatures.append(pairs)
                count.append(1)
                env_sum.append(envelope[:, j].copy())

        if rows.size >= _FRAGMENT_MIN_DRAWS and (
                len(founders) > _FRAGMENT_SHARE * rows.size):
            warnings.warn(
                f"{len(founders)} of {rows.size} draws became particles of "
                f"their own at tol={tol} kHz. If the hyperfine offsets were "
                f"sampled, a coupling wanders by more than that within one "
                f"hypothesis and the posterior has been split into its draws; "
                f"pass a larger tol.", stacklevel=2)
        founders = np.asarray(founders)
        count = np.asarray(count, dtype=float)
        env_mean = np.stack(env_sum) / count[:, None, None]
        return cls(
            site_idx=site_idx[founders], k=k[founders], weight=count,
            dA_par=dA_par[founders], dA_perp=dA_perp[founders],
            lam=env_mean[:, 0], n_stretch=env_mean[:, 1], sigma=env_mean[:, 2],
            n_sites=trace.n_sites, k_max=trace.k_max)

    @property
    def n_particles(self) -> int:
        """K, the number of distinct baths."""
        return int(self.weight.size)

    @property
    def effective_size(self) -> float:
        """Kish's effective sample size, ``1 / sum(w^2)``.

        K for uniform weights, 1 for a single particle, and near 1 whenever one
        particle carries almost all the mass however many others exist.
        """
        return float(1.0 / np.sum(self.weight ** 2))

    def predictions(self, expset, site_table, model):
        """Forward signal of every particle. (K, n_points)

        One vectorised pass: every particle becomes one replica of a single
        State, as :func:`~nuclear_spin_recovery.post.residual.predictive_from_
        arrays` does for posterior draws.  Agrees with a per-particle loop.
        """
        return predictive_from_arrays(
            self.site_idx, self.k, self.dA_par, self.dA_perp, self.lam,
            self.n_stretch, self.sigma, expset, site_table, model,
            k_max=self.k_max)


def _same_bath(a, b, tol):
    """Whether two (k, 2) coupling arrays pair up one-to-one within ``tol``.

    Each coordinate is checked on its own first, sorted: in one dimension a
    within-tolerance pairing exists exactly when the sorted sequences agree
    elementwise, so this is a necessary condition that rejects almost every
    distinct bath at the cost of two sorts.  What survives goes to an
    assignment over the Chebyshev distance, with pairs outside ``tol`` priced
    out; a pairing within ``tol`` exists exactly when the optimal one uses none
    of them.
    """
    if a.shape != b.shape:
        return False
    if a.shape[0] == 0:
        return True
    for c in range(2):
        if np.any(np.abs(np.sort(a[:, c]) - np.sort(b[:, c])) > tol):
            return False
    dist = np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2)
    feasible = dist <= tol
    if not feasible.any(axis=1).all():
        return False
    rows, cols = linear_sum_assignment(np.where(feasible, 0.0, 1.0))
    return bool(feasible[rows, cols].all())
