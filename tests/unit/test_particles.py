"""The posterior as weighted particles.

The design engine reads the posterior's *uncertainty* -- how many distinct
hypotheses it holds and how the mass is spread over them -- so the thing most
worth pinning is that the particle count is a count of physical hypotheses.
Grouping by site index would split one symmetry orbit into several particles
and make a confident posterior look uncertain; most of this file is about
merging correctly, and about not merging what is genuinely different.

The rest pins the effective size, which the EIG bias depends on, and that the
vectorised prediction agrees with a plain loop.

docs/phase-5-plan.md, unit 5a.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    Experiment,
    ExperimentSet,
    ParticleSet,
    SiteTable,
    State,
    StretchedExponential,
    Trace,
    merge_traces,
)

K_MAX = 8


@pytest.fixture
def model():
    return AnalyticCCE1(StretchedExponential())


@pytest.fixture
def expset():
    return ExperimentSet(
        [Experiment(tau=np.linspace(1e-4, 8e-3, 40), n_pulses=16, b_z=311.0)]
    )


@pytest.fixture
def near_orbit_table():
    """Sites 0 and 1 differ in the fourth significant figure, as symmetry
    partners in the NV table do; site 2 is half a kilohertz away, which is a
    different spin."""
    return SiteTable(
        distance=np.array([1.5, 1.5, 1.6, 2.5]),
        positions=np.array(
            [[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0], [0.0, 1.6, 0.0], [0.0, 0.0, 2.5]]
        ),
        a_par=np.array([120.00, 120.03, 120.50, 30.0]),
        a_perp=np.array([45.00, 45.02, 45.00, 10.0]),
        isotope=np.array(["13C"] * 4),
        gyro=np.full(4, 6.7283),
    )


def make_state(sites, table, *, lam=3e-3, n_stretch=1.0, sigma=0.02,
               offsets=None, sort=True):
    """A one-replica state.  ``sort=False`` keeps the slot order as given."""
    sites = np.asarray(list(sites), dtype=int)
    if sort:
        sites = np.sort(sites)
    site_idx = np.full((1, K_MAX), -1, dtype=int)
    site_idx[0, : sites.size] = sites
    st = State(site_idx=site_idx, k=np.array([sites.size]),
               lam=np.full((1, 1), lam), n_stretch=np.full((1, 1), n_stretch),
               sigma=np.full((1, 1), sigma), n_sites=len(table), k_max=K_MAX)
    if offsets is not None:
        dpar, dperp = offsets
        st.dA_par[0, : len(dpar)] = dpar
        st.dA_perp[0, : len(dperp)] = dperp
    return st


def trace_of(states):
    first = states[0]
    tr = Trace(n_sites=first.n_sites, k_max=K_MAX, n_exp=1)
    for j, st in enumerate(states):
        tr.append(st, log_prob=-float(j))
    return tr


def trace_from_sites(configs, table, **kw):
    return trace_of([make_state(sites, table, **kw) for sites in configs])


def by_hand(sites_list, weight, *, n_sites=4, lam=None):
    """A ParticleSet typed in directly, as a user with no sampler would."""
    n = len(sites_list)
    site_idx = np.full((n, K_MAX), -1, dtype=int)
    for i, sites in enumerate(sites_list):
        site_idx[i, : len(sites)] = sites
    lam = np.full((n, 1), 3e-3) if lam is None else np.asarray(lam, float)
    return ParticleSet(
        site_idx=site_idx, k=np.array([len(s) for s in sites_list]),
        weight=np.asarray(weight, dtype=float),
        dA_par=np.zeros((n, K_MAX)), dA_perp=np.zeros((n, K_MAX)),
        lam=lam, n_stretch=np.ones((n, 1)), sigma=np.full((n, 1), 0.02),
        n_sites=n_sites, k_max=K_MAX)


# --------------------------------------------------------------------------
# construction by hand
# --------------------------------------------------------------------------


def test_weights_given_as_multiplicities_are_normalised():
    ps = by_hand([[0, 2], [3]], [3, 1])
    np.testing.assert_allclose(ps.weight, [0.75, 0.25])


def test_a_negative_weight_raises():
    with pytest.raises(ValueError):
        by_hand([[0, 2], [3]], [1.0, -0.5])


def test_all_zero_weights_raise():
    with pytest.raises(ValueError):
        by_hand([[0, 2], [3]], [0.0, 0.0])


def test_a_weight_per_particle_is_required():
    with pytest.raises(ValueError):
        by_hand([[0, 2], [3]], [1.0, 1.0, 1.0])


def test_particles_are_held_in_decreasing_weight():
    ps = by_hand([[3], [0, 2], [2]], [1, 5, 2])
    np.testing.assert_allclose(ps.weight, [5 / 8, 2 / 8, 1 / 8])
    assert ps.k.tolist() == [2, 1, 1]
    assert ps.site_idx[0, :2].tolist() == [0, 2]


def test_ties_keep_the_order_given():
    ps = by_hand([[3], [2]], [1, 1])
    assert ps.site_idx[:, 0].tolist() == [3, 2]


def test_a_single_particle_is_legal():
    """A collapsed posterior; reporting it is the designer's job."""
    ps = by_hand([[0, 2]], [1.0])
    assert ps.n_particles == 1
    np.testing.assert_allclose(ps.weight, [1.0])


# --------------------------------------------------------------------------
# from_trace: merging what is one hypothesis
# --------------------------------------------------------------------------


def test_identical_draws_merge_to_one_particle(tiny_site_table):
    ps = ParticleSet.from_trace(
        trace_from_sites([[0, 2]] * 5, tiny_site_table), tiny_site_table)
    assert ps.n_particles == 1
    np.testing.assert_allclose(ps.weight, [1.0])


def test_a_symmetry_orbit_merges(tiny_site_table):
    """Sites 0 and 1 carry identical couplings: one hypothesis, not two."""
    ps = ParticleSet.from_trace(
        trace_from_sites([[0, 2], [1, 2]], tiny_site_table), tiny_site_table)
    assert ps.n_particles == 1


def test_an_orbit_differing_in_the_fourth_figure_merges(near_orbit_table):
    """The NV table's orbits are not exactly equal -- tolerance does the
    merging, not rounding."""
    ps = ParticleSet.from_trace(
        trace_from_sites([[0, 3], [1, 3]], near_orbit_table), near_orbit_table)
    assert ps.n_particles == 1


def test_an_orbit_swap_that_reorders_the_couplings_still_merges():
    """Sorting is not a canonical form under a tolerance.

    Sites 0 and 1 are one orbit (A_par 120.00 / 120.03); site 2 is a different
    spin whose A_par, 120.02, falls between them.  Swapping 0 for 1 moves that
    spin past site 2 in sorted order, so an elementwise comparison of sorted
    couplings pairs each spin with the wrong partner and splits one bath in
    two.  On the NV table this is not a corner case: 35,580 pairs of distinct
    spins sit within 0.1 kHz of each other in A_par.
    """
    table = SiteTable(
        distance=np.array([1.5, 1.5, 2.0]),
        positions=np.array([[1.5, 0, 0], [-1.5, 0, 0], [0, 2.0, 0]], float),
        a_par=np.array([120.00, 120.03, 120.02]),
        a_perp=np.array([45.00, 45.02, 10.00]),
        isotope=np.array(["13C"] * 3),
        gyro=np.full(3, 6.7283),
    )
    ps = ParticleSet.from_trace(trace_from_sites([[0, 2], [1, 2]], table), table)
    assert ps.n_particles == 1


def test_couplings_outside_tolerance_stay_distinct(near_orbit_table):
    ps = ParticleSet.from_trace(
        trace_from_sites([[0, 3], [2, 3]], near_orbit_table), near_orbit_table)
    assert ps.n_particles == 2


def test_tolerance_is_a_parameter(near_orbit_table):
    tr = trace_from_sites([[0, 3], [2, 3]], near_orbit_table)
    ps = ParticleSet.from_trace(tr, near_orbit_table, tol=1.0)
    assert ps.n_particles == 1


def test_slot_order_is_irrelevant(tiny_site_table):
    tr = trace_of([make_state([0, 2], tiny_site_table, sort=False),
                   make_state([2, 0], tiny_site_table, sort=False)])
    assert ParticleSet.from_trace(tr, tiny_site_table).n_particles == 1


def test_a_different_spin_count_is_a_different_particle(tiny_site_table):
    ps = ParticleSet.from_trace(
        trace_from_sites([[0, 2], [0, 2, 3]], tiny_site_table), tiny_site_table)
    assert ps.n_particles == 2


def test_two_spins_on_one_orbit_are_not_one_spin(tiny_site_table):
    """Sites 0 and 1 together are two spins with equal couplings -- a k=2
    bath, not a duplicate of the k=1 bath at site 0."""
    ps = ParticleSet.from_trace(
        trace_from_sites([[0], [0, 1]], tiny_site_table), tiny_site_table)
    assert ps.n_particles == 2


def test_offsets_distinguish_particles(tiny_site_table):
    """Rebuilding from site index alone would read a relaxed run as
    constrained; the same bug test_post.py guards in the predictive."""
    tr = trace_of([
        make_state([0, 2], tiny_site_table),
        make_state([0, 2], tiny_site_table, offsets=([40.0, 40.0], [20.0, 20.0])),
    ])
    assert ParticleSet.from_trace(tr, tiny_site_table).n_particles == 2


def test_the_empty_bath_is_a_particle(tiny_site_table):
    ps = ParticleSet.from_trace(
        trace_from_sites([[], [], [0, 2]], tiny_site_table), tiny_site_table)
    assert ps.n_particles == 2
    assert ps.k.tolist() == [0, 2]


# --------------------------------------------------------------------------
# from_trace: weights, burn, stride, envelope
# --------------------------------------------------------------------------


def test_weights_follow_multiplicity(tiny_site_table):
    tr = trace_from_sites([[0, 2]] * 3 + [[3]], tiny_site_table)
    ps = ParticleSet.from_trace(tr, tiny_site_table)
    np.testing.assert_allclose(ps.weight, [0.75, 0.25])


def test_weights_sum_to_one(tiny_site_table):
    configs = [[0, 2], [3], [2], [0, 2], [2, 3], [0, 3], [3]]
    ps = ParticleSet.from_trace(trace_from_sites(configs, tiny_site_table),
                                tiny_site_table)
    assert ps.weight.sum() == pytest.approx(1.0)
    assert np.all(ps.weight > 0)


def test_burn_in_is_discarded(tiny_site_table):
    tr = trace_from_sites([[3]] * 4 + [[0, 2]] * 4, tiny_site_table)
    ps = ParticleSet.from_trace(tr, tiny_site_table, burn=4)
    assert ps.n_particles == 1
    assert ps.k.tolist() == [2]


def test_stride_thins_the_draws(tiny_site_table):
    tr = trace_from_sites([[0, 2], [3]] * 4, tiny_site_table)
    ps = ParticleSet.from_trace(tr, tiny_site_table, stride=2)
    assert ps.n_particles == 1
    assert ps.k.tolist() == [2]


def test_an_empty_trace_raises(tiny_site_table):
    tr = Trace(n_sites=len(tiny_site_table), k_max=K_MAX, n_exp=1)
    with pytest.raises(ValueError):
        ParticleSet.from_trace(tr, tiny_site_table)


def test_the_envelope_is_the_mean_over_the_particles_draws(tiny_site_table):
    tr = trace_of([make_state([0, 2], tiny_site_table, lam=2e-3, sigma=0.01),
                   make_state([1, 2], tiny_site_table, lam=4e-3, sigma=0.03),
                   make_state([3], tiny_site_table, lam=9e-3)])
    ps = ParticleSet.from_trace(tr, tiny_site_table)
    np.testing.assert_allclose(ps.lam[0], [3e-3])
    np.testing.assert_allclose(ps.sigma[0], [0.02])
    np.testing.assert_allclose(ps.lam[1], [9e-3])


def test_a_pooled_trace_is_read_like_any_other(tiny_site_table):
    """The engine cannot tell a pooled ensemble from a single chain."""
    a = trace_from_sites([[0, 2]] * 3, tiny_site_table)
    b = trace_from_sites([[3]], tiny_site_table)
    ps = ParticleSet.from_trace(merge_traces([a, b]), tiny_site_table)
    np.testing.assert_allclose(ps.weight, [0.75, 0.25])


# --------------------------------------------------------------------------
# effective size
# --------------------------------------------------------------------------


def test_effective_size_is_K_for_uniform_weights():
    ps = by_hand([[0], [2], [3], [0, 2]], [1, 1, 1, 1])
    assert ps.effective_size == pytest.approx(4.0)


def test_effective_size_of_one_particle_is_one():
    assert by_hand([[0, 2]], [1.0]).effective_size == pytest.approx(1.0)


def test_effective_size_of_a_spike_is_near_one():
    ps = by_hand([[0], [2], [3], [0, 2]], [0.997, 0.001, 0.001, 0.001])
    assert 1.0 <= ps.effective_size < 1.01


def test_many_draws_on_two_baths_have_effective_size_two(tiny_site_table):
    """The count that matters is hypotheses, not draws."""
    tr = trace_from_sites([[0, 2], [1, 2]] * 100 + [[3]] * 200, tiny_site_table)
    ps = ParticleSet.from_trace(tr, tiny_site_table)
    assert ps.effective_size == pytest.approx(2.0)


# --------------------------------------------------------------------------
# predictions
# --------------------------------------------------------------------------


def test_predictions_have_one_row_per_particle(tiny_site_table, expset, model):
    ps = by_hand([[0, 2], [3], [2]], [3, 2, 1])
    out = ps.predictions(expset, tiny_site_table, model)
    assert out.shape == (3, expset.n_points)


def test_predictions_match_a_per_particle_loop(tiny_site_table, expset, model):
    ps = by_hand([[0, 2], [3], [2], []], [4, 3, 2, 1],
                 lam=[[2e-3], [3e-3], [4e-3], [5e-3]])
    out = ps.predictions(expset, tiny_site_table, model)
    for i in range(ps.n_particles):
        one = State(site_idx=ps.site_idx[i:i + 1], k=ps.k[i:i + 1],
                    lam=ps.lam[i:i + 1], n_stretch=ps.n_stretch[i:i + 1],
                    sigma=ps.sigma[i:i + 1], n_sites=len(tiny_site_table),
                    k_max=K_MAX, dA_par=ps.dA_par[i:i + 1],
                    dA_perp=ps.dA_perp[i:i + 1])
        expected = model.coherence(one, expset, tiny_site_table)[0]
        np.testing.assert_allclose(out[i], expected, rtol=1e-12, atol=1e-12)


def test_each_particle_is_predicted_with_its_own_envelope(tiny_site_table,
                                                          expset, model):
    ps = by_hand([[0, 2], [0, 2]], [1, 1], lam=[[2e-3], [6e-3]])
    out = ps.predictions(expset, tiny_site_table, model)
    assert not np.allclose(out[0], out[1])


def test_offsets_reach_the_predictions(tiny_site_table, expset, model):
    tr = trace_of([
        make_state([0, 2], tiny_site_table),
        make_state([0, 2], tiny_site_table, offsets=([40.0, 40.0], [20.0, 20.0])),
    ])
    ps = ParticleSet.from_trace(tr, tiny_site_table)
    out = ps.predictions(expset, tiny_site_table, model)
    assert not np.allclose(out[0], out[1])


def test_from_trace_and_by_hand_agree(tiny_site_table, expset, model):
    """The same posterior, however it was obtained, predicts the same."""
    tr = trace_from_sites([[0, 2]] * 3 + [[3]], tiny_site_table)
    sampled = ParticleSet.from_trace(tr, tiny_site_table)
    typed = by_hand([[0, 2], [3]], [3, 1])
    np.testing.assert_allclose(sampled.weight, typed.weight)
    np.testing.assert_allclose(
        sampled.predictions(expset, tiny_site_table, model),
        typed.predictions(expset, tiny_site_table, model))


# --------------------------------------------------------------------------
# agnosticism
# --------------------------------------------------------------------------

SAMPLER_MODULES = {"algorithms", "driver", "ensemble", "config", "proposals",
                   "neighbors"}


def test_design_imports_no_sampler():
    """design/ may read a Trace; it may not know how one was produced."""
    from nuclear_spin_recovery import design

    root = pathlib.Path(design.__file__).parent
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                parts = set((node.module or "").split("."))
                assert not parts & SAMPLER_MODULES, (
                    f"{path.name} imports {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = set(alias.name.split("."))
                    assert not parts & SAMPLER_MODULES, (
                        f"{path.name} imports {alias.name}")


# --------------------------------------------------------------------------
# relaxed posteriors: a tolerance that groups nothing
# --------------------------------------------------------------------------


def _relaxed_trace(table, n_draws, spread, seed=0):
    """One bath, its offsets wandering by ``spread`` kHz from draw to draw."""
    gen = np.random.default_rng(seed)
    return trace_of([
        make_state([0, 3], table, offsets=(gen.uniform(-spread, spread, 2),
                                           gen.uniform(-spread, spread, 2)))
        for _ in range(n_draws)])


def test_a_relaxed_posterior_at_the_default_tolerance_warns(tiny_site_table):
    """One hypothesis whose couplings wander by kilohertz is split into its
    draws at 0.1 kHz, which a design would then read as many hypotheses."""
    trace = _relaxed_trace(tiny_site_table, 80, spread=3.0)
    with pytest.warns(UserWarning, match="larger tol"):
        ps = ParticleSet.from_trace(trace, tiny_site_table)
    assert ps.n_particles > 40


def test_a_looser_tolerance_groups_it_and_does_not_warn(tiny_site_table,
                                                        recwarn):
    trace = _relaxed_trace(tiny_site_table, 80, spread=3.0)
    ps = ParticleSet.from_trace(trace, tiny_site_table, tol=6.0)
    assert ps.n_particles == 1
    assert not [w for w in recwarn if "larger tol" in str(w.message)]


def test_a_handful_of_distinct_draws_does_not_warn(tiny_site_table, recwarn):
    """Too few draws to tell fragmentation from a posterior that is simply
    spread out."""
    ParticleSet.from_trace(_relaxed_trace(tiny_site_table, 10, spread=3.0),
                           tiny_site_table)
    assert not [w for w in recwarn if "larger tol" in str(w.message)]
