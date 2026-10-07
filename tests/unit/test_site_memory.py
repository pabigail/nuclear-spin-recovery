"""Site memory: hyperfine offsets that belong to sites, not to spins.

By default a spin carries its offset with it, so an offset fitted on one site
is applied to whatever site the spin hops to next.  With site memory the
offset stays behind, and a spin landing on a site -- by a hop, a birth, or a
tempering swap -- takes up the offset that site was last left with.

Two things are checked here.  The state keeps the memory and the live spins
in step through every mutation; and every move in the package, driven through
those mutations, leaves them in step.  The companion kernel, whose prior
width depends on the site, is checked at the end.

docs/model-specification.md Sec. 5.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    RJMCMC,
    RWMH,
    BirthDeathKernel,
    DiscreteLatticeWalk,
    GaussianOffset,
    HybridDriver,
    NeighborIndex,
    ParallelTempering,
    ParameterBlock,
    Schedule,
    SiteScaledOffset,
    State,
    Step,
)

N_SITES, K_MAX = 4, 3


class Flat:
    """A target with no preference, so every proposal is accepted on its
    proposal and prior ratios alone."""

    def __init__(self, site_table):
        self.site_table = site_table

    def log_prob(self, state, beta=1.0):
        return np.zeros(state.n_replicas)


def make_state(sites=(0, 2), site_memory=True, k_max=K_MAX):
    return State.from_sites(
        sites, n_sites=N_SITES, n_exp=1, lam=np.array([[3e-3]]),
        n_stretch=np.array([[1.0]]), sigma=np.array([[0.1]]), k_max=k_max,
        site_memory=site_memory)


def walk(table):
    return DiscreteLatticeWalk(NeighborIndex(table.positions, radius=50.0))


# ------------------------------------------------------------------- state

def test_memory_is_off_by_default():
    state = make_state(site_memory=False)
    assert not state.has_site_memory
    assert state.site_dA_par is None and state.site_dA_perp is None
    assert state.replica_fields() == State.FIELDS


def test_memory_starts_at_the_table_value():
    state = make_state()
    assert state.has_site_memory
    assert state.site_dA_par.shape == (1, N_SITES)
    assert np.all(state.site_dA_par == 0.0) and np.all(state.site_dA_perp == 0.0)
    assert state.replica_fields() == State.FIELDS + State.MEMORY_FIELDS


def test_memory_needs_both_components():
    with pytest.raises(ValueError, match="both"):
        State(site_idx=np.array([[0, -1]]), k=np.array([1]),
              lam=np.ones((1, 1)), n_stretch=np.ones((1, 1)),
              sigma=np.ones((1, 1)), n_sites=N_SITES, k_max=2,
              site_dA_par=np.zeros((1, N_SITES)))


def test_set_offset_writes_the_spin_and_the_site():
    state = make_state(sites=(0, 2))
    state.set_offset(0, 1, 0, 2.5)
    state.set_offset(0, 1, 1, -1.5)
    assert state.dA_par[0, 1] == 2.5 and state.dA_perp[0, 1] == -1.5
    assert state.site_dA_par[0, 2] == 2.5 and state.site_dA_perp[0, 2] == -1.5
    assert state.site_dA_par[0, 0] == 0.0
    state.check_invariants()


def test_set_offset_without_memory_writes_the_spin_only():
    state = make_state(site_memory=False)
    state.set_offset(0, 0, 0, 2.5)
    assert state.dA_par[0, 0] == 2.5
    assert state.site_dA_par is None


def test_a_move_leaves_the_offset_behind():
    state = make_state(sites=(0,))
    state.set_offset(0, 0, 0, 4.0)
    state.move_spin(0, 0, 3)
    assert state.dA_par[0, 0] == 0.0, "an unvisited site starts at its table value"
    assert state.site_dA_par[0, 0] == 4.0, "the site keeps what it was left with"
    assert list(state.occupied[0]) == [False, False, False, True]
    state.check_invariants()


def test_a_return_resumes_where_the_site_was_left():
    state = make_state(sites=(0,))
    state.set_offset(0, 0, 0, 4.0)
    state.set_offset(0, 0, 1, -2.0)
    state.move_spin(0, 0, 3)
    state.set_offset(0, 0, 0, -7.0)
    state.move_spin(0, 0, 0)
    assert state.dA_par[0, 0] == 4.0 and state.dA_perp[0, 0] == -2.0
    assert state.site_dA_par[0, 3] == -7.0
    state.check_invariants()


def test_without_memory_the_offset_travels_with_the_spin():
    """The behaviour every earlier result was produced with."""
    state = make_state(sites=(0,), site_memory=False)
    state.set_offset(0, 0, 0, 4.0)
    state.move_spin(0, 0, 3)
    assert state.dA_par[0, 0] == 4.0


def test_a_birth_resumes_the_site_and_a_death_leaves_it():
    state = make_state(sites=(0,))
    state.add_spin(0, 2)
    assert state.dA_par[0, 1] == 0.0
    state.set_offset(0, 1, 0, 3.0)
    state.remove_spin(0, 1)
    assert state.k[0] == 1
    assert state.site_dA_par[0, 2] == 3.0, "a death must not erase the memory"
    state.add_spin(0, 2)
    assert state.dA_par[0, 1] == 3.0, "a rebirth resumes it"
    state.check_invariants()


def test_without_memory_a_birth_starts_at_the_table_value():
    state = make_state(sites=(0,), site_memory=False)
    state.add_spin(0, 2)
    state.set_offset(0, 1, 0, 3.0)
    state.remove_spin(0, 1)
    state.add_spin(0, 2)
    assert state.dA_par[0, 1] == 0.0


def test_a_death_keeps_the_survivors_aligned():
    """Removal swaps the last spin into the gap; its offset must come too."""
    state = make_state(sites=(0, 1, 2))
    for slot, value in enumerate((1.0, 2.0, 3.0)):
        state.set_offset(0, slot, 0, value)
    state.remove_spin(0, 0)
    assert state.k[0] == 2
    for slot in range(2):
        assert state.dA_par[0, slot] == state.site_dA_par[0, state.site_idx[0, slot]]
    assert state.site_dA_par[0, 0] == 1.0
    state.check_invariants()


def test_copies_do_not_share_memory():
    state = make_state()
    state.set_offset(0, 0, 0, 2.0)
    other = state.copy()
    other.set_offset(0, 0, 0, 9.0)
    assert state.site_dA_par[0, 0] == 2.0
    assert other.has_site_memory


def test_replicas_each_get_the_cold_memory():
    state = make_state()
    state.set_offset(0, 0, 0, 2.0)
    ladder = state.expand_replicas(3)
    assert ladder.site_dA_par.shape == (3, N_SITES)
    assert np.all(ladder.site_dA_par[:, 0] == 2.0)
    ladder.set_offset(2, 0, 0, 5.0)
    assert ladder.site_dA_par[0, 0] == 2.0, "rungs must not share one memory"
    assert ladder.collapse_to_cold().site_dA_par.shape == (1, N_SITES)
    ladder.check_invariants()


def test_invariants_catch_a_memory_that_has_drifted():
    state = make_state()
    state.dA_par[0, 0] = 1.0          # written behind the state's back
    with pytest.raises(ValueError, match="remembers"):
        state.check_invariants()
    state.remember_offsets()
    state.check_invariants()


def test_enabling_memory_remembers_the_current_offsets():
    state = make_state(sites=(0, 2), site_memory=False)
    state.dA_par[0, :2] = [1.0, -2.0]
    state.enable_site_memory()
    assert list(state.site_dA_par[0]) == [1.0, 0.0, -2.0, 0.0]
    state.check_invariants()


# ------------------------------------------------------------------- moves

def test_site_move_resumes_the_destination(tiny_site_table):
    state = make_state(sites=(0, 1, 2))
    state.set_offset(0, 0, 0, 4.0)
    state.site_dA_par[0, 3] = -6.0          # as left by an earlier visit
    mover = RWMH(ParameterBlock("sites"), walk(tiny_site_table))
    rng = np.random.default_rng(0)
    for _ in range(20):                     # site 3 is the only free site
        state = mover.step(state, Flat(tiny_site_table), rng)
        state.check_invariants()
    assert set(state.site_dA_par[0]) >= {4.0, -6.0}


def test_rjmcmc_birth_resumes_the_site(tiny_site_table):
    state = make_state(sites=(0, 1, 2), k_max=4)
    state.site_dA_perp[0, 3] = 1.25          # as left by an earlier visit
    mover = RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=4))
    rng = np.random.default_rng(0)
    target = Flat(tiny_site_table)
    for _ in range(200):
        state = mover.step(state, target, rng)
        state.check_invariants()
        if state.occupied[0, 3]:
            slot = int(np.flatnonzero(state.site_idx[0] == 3)[0])
            assert state.dA_perp[0, slot] == 1.25
            break
    else:
        pytest.fail("no spin was ever born on site 3")


def test_a_swap_exchanges_the_memories(tiny_site_table):
    ladder = make_state(sites=(0,)).expand_replicas(2)
    ladder.set_offset(0, 0, 0, 1.0)
    ladder.set_offset(1, 0, 0, 2.0)
    ladder.site_dA_par[1, 3] = 7.0
    tempering = ParallelTempering(Schedule([]), n_replicas=2)
    # A flat target accepts every swap.
    swapped = tempering.attempt_swap(ladder, Flat(tiny_site_table),
                                     np.random.default_rng(0))
    assert swapped.dA_par[0, 0] == 2.0 and swapped.dA_par[1, 0] == 1.0
    assert swapped.site_dA_par[0, 3] == 7.0 and swapped.site_dA_par[1, 3] == 0.0
    swapped.check_invariants()


def test_tempering_reports_its_last_swap(tiny_site_table):
    """A diagnostic the notebooks draw from; the sampler never reads it."""
    tempering = ParallelTempering(Schedule([]), n_replicas=3)
    assert tempering.last_swap is None
    ladder = make_state(sites=(0,)).expand_replicas(3)
    tempering.attempt_swap(ladder, Flat(tiny_site_table), np.random.default_rng(0))
    a, b, accepted = tempering.last_swap
    assert {a, b} <= {0, 1, 2} and a != b
    assert accepted is True


@pytest.mark.parametrize("redraw", [False, True])
def test_a_full_schedule_keeps_spins_and_sites_in_step(tiny_site_table, redraw):
    """Every move the package has, interleaved, with the invariant checked
    after each one."""
    target = Flat(tiny_site_table)
    offsets = RWMH(ParameterBlock("offsets"), SiteScaledOffset(
        2.0, tiny_site_table, fraction_par=0.1, fraction_perp=0.1, floor=0.5,
        redraw_unoccupied=redraw))
    sites = RWMH(ParameterBlock("sites"), walk(tiny_site_table))
    jump = RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=K_MAX))
    tempering = ParallelTempering(
        Schedule([Step(jump, 1), Step(sites, 1), Step(offsets, 2)]), n_replicas=3)
    rng = np.random.default_rng(1)
    state = make_state(sites=(0,))
    seen_k = set()
    for _ in range(60):
        for algorithm in (jump, sites, offsets, offsets):
            state = algorithm.step(state, target, rng)
            state.check_invariants()
        state = tempering.run(state, target, rng, n_steps=2)
        state.check_invariants()
        seen_k.add(int(state.k[0]))
    assert len(seen_k) > 1
    assert np.any(state.site_dA_par != 0.0)


def test_memory_survives_the_hybrid_driver(tiny_site_table):
    offsets = RWMH(ParameterBlock("offsets"), SiteScaledOffset(
        2.0, tiny_site_table, fraction_par=0.1, fraction_perp=0.1))
    schedule = Schedule([
        Step(RWMH(ParameterBlock("sites"), walk(tiny_site_table)), 2),
        Step(offsets, 5)])
    out = HybridDriver(schedule).run(make_state(sites=(0, 1)),
                                     Flat(tiny_site_table),
                                     np.random.default_rng(2), 200)
    assert out.has_site_memory
    out.check_invariants()
    assert np.count_nonzero(out.site_dA_par) >= 3, "visited sites should remember"


# -------------------------------------------------------- site-scaled kernel

def kernel(table, **kwargs):
    kwargs.setdefault("fraction_par", 0.1)
    kwargs.setdefault("fraction_perp", 0.2)
    return SiteScaledOffset(kwargs.pop("radius", 1.0), table, **kwargs)


def test_width_is_a_fraction_of_the_table_value(tiny_site_table):
    k = kernel(tiny_site_table)
    assert k.width(0, 0) == pytest.approx(0.1 * 120.0)
    assert k.width(0, 1) == pytest.approx(0.2 * 45.0)
    assert k.width(3, 0) == pytest.approx(0.1 * 2.0)


def test_width_uses_the_magnitude_of_a_negative_coupling(tiny_site_table):
    tiny_site_table.a_par[1] = -120.0
    assert kernel(tiny_site_table).width(1, 0) == pytest.approx(12.0)


def test_floor_holds_up_a_weakly_coupled_site(tiny_site_table):
    k = kernel(tiny_site_table, floor=0.5)
    assert k.width(3, 0) == pytest.approx(0.5)      # 10% of 2 kHz is 0.2
    assert k.width(0, 0) == pytest.approx(12.0)     # unaffected where larger


def test_defaults_give_no_relaxation_at_all(tiny_site_table):
    """Fraction and floor default to zero: the constrained model, exactly."""
    k = SiteScaledOffset(1.0, tiny_site_table)
    assert all(k.width(s, c) == 0.0 for s in range(N_SITES) for c in (0, 1))
    mover = RWMH(ParameterBlock("offsets"), k)
    state = make_state(sites=(0, 2))
    rng = np.random.default_rng(0)
    for _ in range(50):
        state = mover.step(state, Flat(tiny_site_table), rng)
    assert np.all(state.dA_par == 0.0) and np.all(state.dA_perp == 0.0)
    assert np.all(state.site_dA_par == 0.0)


def test_gaussian_is_the_default_prior(tiny_site_table):
    k = kernel(tiny_site_table)
    assert k.prior == "gaussian"
    width = k.width(0, 0)
    assert k.log_prior(0.0, site=0, component=0) == 0.0
    assert k.log_prior(width, site=0, component=0) == pytest.approx(-0.5)
    assert k.log_prior(-2 * width, site=0, component=0) == pytest.approx(-2.0)


def test_the_same_offset_costs_more_on_a_weaker_site(tiny_site_table):
    k = kernel(tiny_site_table)
    assert (k.log_prior(1.0, site=3, component=0)
            < k.log_prior(1.0, site=0, component=0))


def test_flat_prior_has_no_preference_inside_its_box(tiny_site_table):
    k = kernel(tiny_site_table, prior="flat")
    assert k.log_prior(11.9, site=0, component=0) == 0.0
    assert k.bound(k.width(0, 0)) == pytest.approx(12.0)


def test_gaussian_walk_is_bounded_at_n_sigma(tiny_site_table):
    k = kernel(tiny_site_table, n_sigma=3.0)
    assert k.bound(k.width(0, 0)) == pytest.approx(36.0)


@pytest.mark.parametrize("prior", ["gaussian", "flat"])
def test_proposals_stay_inside_the_bound(tiny_site_table, prior):
    k = kernel(tiny_site_table, prior=prior, radius=50.0)
    rng = np.random.default_rng(0)
    for site in range(N_SITES):
        for component in (0, 1):
            bound = k.bound(k.width(site, component))
            value = 0.0
            for _ in range(200):
                value, ratio = k.propose(rng, value, site=site, component=component)
                assert abs(value) <= bound + 1e-12
                assert ratio == 0.0


def test_the_step_is_capped_at_the_bound(tiny_site_table):
    """A radius far wider than a narrow site's box must still move within it,
    and must not be reflected back and forth across it."""
    k = kernel(tiny_site_table, prior="flat", radius=1000.0)
    rng = np.random.default_rng(0)
    draws = np.array([k.propose(rng, 0.0, site=3, component=0)[0]
                      for _ in range(2000)])
    assert np.abs(draws).max() <= 0.2 + 1e-12
    assert np.abs(draws).max() > 0.19


def test_prior_draws_follow_the_prior(tiny_site_table):
    rng = np.random.default_rng(0)
    flat = kernel(tiny_site_table, prior="flat")
    draws = np.array([flat.draw_prior(rng, site=0, component=0)
                      for _ in range(4000)])
    assert np.abs(draws).max() <= 12.0
    assert draws.std() == pytest.approx(12.0 / np.sqrt(3.0), rel=0.05)
    gauss = kernel(tiny_site_table)
    draws = np.array([gauss.draw_prior(rng, site=0, component=0)
                      for _ in range(4000)])
    assert draws.std() == pytest.approx(12.0, rel=0.05)
    assert SiteScaledOffset(1.0, tiny_site_table).draw_prior(
        rng, site=0, component=0) == 0.0


@pytest.mark.parametrize("kwargs", [
    {"prior": "laplace"}, {"fraction_par": -0.1}, {"floor": -1.0},
    {"radius": -1.0}, {"n_sigma": 0.0}])
def test_meaningless_settings_raise(tiny_site_table, kwargs):
    with pytest.raises(ValueError):
        kernel(tiny_site_table, **kwargs)


def test_site_scaled_offsets_refuse_a_state_without_memory(tiny_site_table):
    """Without memory an offset would cross from one site's prior to another's
    on a hop, and the hop has no term for that."""
    mover = RWMH(ParameterBlock("offsets"), kernel(tiny_site_table))
    with pytest.raises(ValueError, match="site memory"):
        mover.step(make_state(site_memory=False), Flat(tiny_site_table),
                   np.random.default_rng(0))


def test_the_absolute_kernel_still_works_with_memory(tiny_site_table):
    """Site memory and a site-independent width are an allowed pairing."""
    mover = RWMH(ParameterBlock("offsets"), GaussianOffset(radius=1.5, scale=4.0))
    state = make_state(sites=(0, 2))
    rng = np.random.default_rng(0)
    for _ in range(30):
        state = mover.step(state, Flat(tiny_site_table), rng)
    state.check_invariants()
    assert np.any(state.site_dA_par != 0.0) or np.any(state.site_dA_perp != 0.0)


def test_unoccupied_sites_are_frozen_unless_redrawn(tiny_site_table):
    target = Flat(tiny_site_table)
    for redraw, expect_change in ((False, False), (True, True)):
        mover = RWMH(ParameterBlock("offsets"),
                     kernel(tiny_site_table, redraw_unoccupied=redraw))
        state = make_state(sites=(0,))
        state.site_dA_par[0, 3] = 0.1
        rng = np.random.default_rng(3)
        for _ in range(40):
            state = mover.step(state, target, rng)
            state.check_invariants()
        unoccupied = state.site_dA_par[0, 1:]
        changed = not np.array_equal(unoccupied, [0.0, 0.0, 0.1])
        assert changed == expect_change


def test_redrawn_offsets_respect_each_sites_bound(tiny_site_table):
    k = kernel(tiny_site_table, prior="flat", redraw_unoccupied=True)
    mover = RWMH(ParameterBlock("offsets"), k)
    state = make_state(sites=(0,))
    rng = np.random.default_rng(4)
    for _ in range(300):
        state = mover.step(state, Flat(tiny_site_table), rng)
        for site in range(N_SITES):
            assert abs(state.site_dA_par[0, site]) <= k.width(site, 0) + 1e-12
            assert abs(state.site_dA_perp[0, site]) <= k.width(site, 1) + 1e-12
