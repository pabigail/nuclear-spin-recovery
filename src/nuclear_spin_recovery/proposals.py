"""Proposal kernels for random-walk Metropolis-Hastings.

Every proposal reports both the proposed value and the log of the proposal
ratio ``r(z -> x) / r(x -> z)``, which the acceptance rule needs.  Returning
the ratio from the proposal -- rather than assuming symmetry at the call site
-- is what keeps an asymmetric kernel correct.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Proposal(ABC):
    """A reversible proposal kernel."""

    @abstractmethod
    def propose(self, rng, current, occupied=None):
        """Return ``(proposed, log_ratio)``.

        ``log_ratio`` is log r(z -> x) - log r(x -> z), zero for a symmetric
        kernel.
        """

    def log_prior(self, value):
        """Log prior density at ``value``.

        Zero for kernels whose parameter has no proper prior -- the lattice
        constraint and the prior on k enter elsewhere (spec Sec. 7.2).  The
        hyperfine offsets of Sec. 5.3 are the exception: their Gaussian prior is
        proper and must appear in the acceptance ratio.
        """
        return 0.0


class ContinuousReflected(Proposal):
    """Uniform step of at most ``radius``, reflected at the domain bounds.

    Reflection preserves symmetry, so the log proposal ratio is exactly zero.
    Applied to lambda, the stretch exponent, and sigma.
    """

    def __init__(self, radius, lower=0.0, upper=1.0):
        self.radius = float(radius)
        self.lower = float(lower)
        self.upper = float(upper)
        if self.radius < 0.0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        if self.lower >= self.upper:
            raise ValueError(f"lower {lower} must be below upper {upper}")

    def propose(self, rng, current, occupied=None):
        step = rng.uniform(-self.radius, self.radius, size=np.shape(current))
        return self._reflect(np.asarray(current, dtype=float) + step), 0.0

    def _reflect(self, x):
        """Fold x back into [lower, upper] by repeated reflection.

        Reflection, not clipping: clipping piles probability mass onto the
        boundary and destroys the symmetry the zero proposal ratio assumes.
        """
        span = self.upper - self.lower
        y = np.mod(x - self.lower, 2.0 * span)
        y = np.where(y > span, 2.0 * span - y, y)
        return self.lower + y


class DiscreteLatticeWalk(Proposal):
    """Move one spin to an unoccupied site within ``radius``.

    The occupancy constraint makes this kernel asymmetric: the number of
    available neighbours differs between the current and proposed sites, so

        log_ratio = log |N_R(x) \\ O| - log |N_R(z) \\ O|

    with the moving spin excluded from the occupied set O.  Dropping this term
    yields a chain that still appears to recover correct configurations while
    targeting the wrong distribution.  See spec Sec. 8.2.
    """

    def __init__(self, neighbors):
        self.neighbors = neighbors

    @property
    def radius(self):
        return self.neighbors.radius

    def propose(self, rng, current, occupied=None):
        """Propose a new site for the spin currently at ``current``.

        If no unoccupied neighbour exists the move is a no-op: the current
        site is returned with a zero log ratio.
        """
        current = int(current)
        occupied = np.asarray(occupied, dtype=bool)

        candidates = self.neighbors.neighbors(current)
        # The spin being moved must not block its own departure.
        free = candidates[~occupied[candidates]] if candidates.size else candidates
        if free.size == 0:
            return current, 0.0

        proposed = int(rng.choice(free))
        forward = self.neighbors.count_available(current, occupied, ignore=current)
        reverse = self.neighbors.count_available(proposed, occupied, ignore=current)
        if forward == 0 or reverse == 0:
            return current, 0.0
        return proposed, float(np.log(forward) - np.log(reverse))


class GaussianOffset(ContinuousReflected):
    """Continuous walk over a hyperfine offset, under a Gaussian prior.

    Relaxes the hard *ab initio* constraint (spec Sec. 5.3): a spin's coupling
    becomes its table value plus an offset drawn against N(0, scale**2), so the
    prior is centred on the DFT prediction and its width encodes how far that
    prediction is trusted.

    Unlike the other kernels this one carries a proper prior, which enters the
    acceptance ratio explicitly.
    """

    def __init__(self, radius, scale, bound=None):
        bound = 5.0 * scale if bound is None else float(bound)
        super().__init__(radius, lower=-bound, upper=bound)
        self.scale = float(scale)

    def log_prior(self, value):
        """Gaussian centred on the table value, i.e. on a zero offset."""
        value = np.asarray(value, dtype=float)
        return -0.5 * (value / self.scale) ** 2


class SiteScaledOffset(Proposal):
    """Offset walk whose width is set by the site the spin is on.

    ``GaussianOffset`` trusts every coupling to the same number of kHz.  DFT
    error is closer to a fraction of the coupling, so a strongly coupled site
    should be allowed to move further than a weakly coupled one.  Here the
    width of each component at each site is

        width = max(floor, fraction * |table value|)

    with ``fraction`` given separately for the parallel and perpendicular
    components.  Both default to zero, and so does ``floor``: with nothing
    set the width is zero everywhere, no offset can move, and the model is
    exactly the constrained one.  A floor keeps a component whose table value
    is near zero from being pinned.

    ``prior`` is what the width means:

    ``"gaussian"``  N(0, width**2) centred on the table value, with the walk
                    bounded to ``n_sigma`` widths.  The default: the table
                    value is the most likely one and the width is how far it
                    is trusted.
    ``"flat"``      uniform on [-width, width] and zero outside.

    This kernel needs a state with **site memory**.  Without it an offset
    travels with its spin, so a site move would carry an offset drawn under
    one site's prior into another's, and the acceptance ratio of that move
    would need a prior term it does not have.  With site memory no move ever
    changes which site an offset belongs to, and the question does not arise.

    ``redraw_unoccupied`` adds, after every offset step, a fresh draw from the
    prior for the offset of one unoccupied site.  It is off by default.  With
    it off an unoccupied site keeps the offset it was left with for as long
    as it stays unoccupied; with it on, that memory decays.  Either way the
    sampler targets the same distribution, because the likelihood does not
    depend on the offset of a site with no spin on it.
    """

    #: Tells RWMH to pass the site and component to :meth:`propose`.
    site_scaled = True

    PRIORS = ("gaussian", "flat")

    def __init__(self, radius, site_table, fraction_par=0.0, fraction_perp=0.0,
                 floor=0.0, prior="gaussian", n_sigma=5.0,
                 redraw_unoccupied=False):
        self.radius = float(radius)
        self.fraction = (float(fraction_par), float(fraction_perp))
        self.floor = float(floor)
        self.prior = str(prior)
        self.n_sigma = float(n_sigma)
        self.redraw_unoccupied = bool(redraw_unoccupied)
        if self.radius < 0.0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        if min(self.fraction) < 0.0 or self.floor < 0.0:
            raise ValueError("fraction_par, fraction_perp and floor must be "
                             "non-negative")
        if self.prior not in self.PRIORS:
            raise ValueError(
                f"unknown prior {prior!r}; known priors are {list(self.PRIORS)}")
        if self.n_sigma <= 0.0:
            raise ValueError(f"n_sigma must be positive, got {n_sigma}")
        self._table_values = (np.asarray(site_table.a_par, dtype=float),
                              np.asarray(site_table.a_perp, dtype=float))

    def width(self, site, component):
        """Width of ``component`` (0 parallel, 1 perpendicular) at ``site``, kHz."""
        value = self._table_values[component][int(site)]
        return max(self.floor, self.fraction[component] * abs(float(value)))

    def bound(self, width):
        """Largest offset the walk may reach, for a given width."""
        return width if self.prior == "flat" else self.n_sigma * width

    def propose(self, rng, current, occupied=None, *, site, component):
        """A uniform step reflected at the bound, as ``ContinuousReflected``.

        The step is at most ``radius``, or the bound where that is smaller.
        A zero width leaves the offset where it is.
        """
        bound = self.bound(self.width(site, component))
        if bound <= 0.0:
            return float(current), 0.0
        kernel = ContinuousReflected(min(self.radius, bound), lower=-bound,
                                     upper=bound)
        return kernel.propose(rng, current)

    def log_prior(self, value, *, site, component):
        """Log prior density of an offset at ``site``, up to a constant."""
        if self.prior == "flat":
            return 0.0
        width = self.width(site, component)
        if width <= 0.0:
            return 0.0
        return float(-0.5 * (float(value) / width) ** 2)

    def draw_prior(self, rng, *, site, component):
        """One offset drawn from the prior at ``site``."""
        width = self.width(site, component)
        if width <= 0.0:
            return 0.0
        if self.prior == "flat":
            return float(rng.uniform(-width, width))
        bound = self.bound(width)
        while True:
            value = float(rng.normal(0.0, width))
            if abs(value) <= bound:
                return value
