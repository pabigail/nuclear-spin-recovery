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
particle when they have the same k and their (A_par, A_perp) pairs, sorted,
agree within ``tol``; offsets are included, so a relaxed run is not read as a
constrained one.

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

from ..post.detection import MATCH_TOL


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
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def n_particles(self) -> int:
        """K, the number of distinct baths."""
        raise NotImplementedError

    @property
    def effective_size(self) -> float:
        """Kish's effective sample size, ``1 / sum(w^2)``.

        K for uniform weights, 1 for a single particle, and near 1 whenever one
        particle carries almost all the mass however many others exist.
        """
        raise NotImplementedError

    def predictions(self, expset, site_table, model):
        """Forward signal of every particle. (K, n_points)

        One vectorised pass: every particle becomes one replica of a single
        State, as :func:`~nuclear_spin_recovery.post.residual.predictive_from_
        arrays` does for posterior draws.  Agrees with a per-particle loop.
        """
        raise NotImplementedError
