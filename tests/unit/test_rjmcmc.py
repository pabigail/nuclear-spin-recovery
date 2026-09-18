"""Birth-death kernel and trans-dimensional moves."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import BirthDeathKernel, ParameterBlock, RJMCMC, State


class Flat:
    def log_prob(self, state, beta=1.0):
        return np.zeros(state.n_replicas)


class PrefersMany:
    """Log-density increasing in k, so births are always favoured."""
    def __init__(self, sign=1.0):
        self.sign = sign

    def log_prob(self, state, beta=1.0):
        return beta * self.sign * 50.0 * state.k.astype(float)


def make_state(sites=(0, 2), n_sites=8, k_max=6, n_exp=1):
    return State.from_sites(
        sites, n_sites=n_sites, n_exp=n_exp,
        lam=np.full((1, n_exp), 3e-3), n_stretch=np.ones((1, n_exp)),
        sigma=np.full((1, n_exp), 0.1), k_max=k_max)


@pytest.fixture
def kernel():
    return BirthDeathKernel(k_max=6)


@pytest.fixture
def mover(kernel):
    return RJMCMC(ParameterBlock("sites"), kernel)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ------------------------------------------------------------------- kernel

def test_proposes_one_step_in_dimension(kernel, rng):
    for _ in range(50):
        k_new, move = kernel.propose(rng, 3, n_free=5)
        assert k_new in (2, 4)
        assert move in ("birth", "death")


def test_birth_moves_up_death_moves_down(kernel, rng):
    for _ in range(50):
        k_new, move = kernel.propose(rng, 3, n_free=5)
        assert (k_new == 4) == (move == "birth")


def test_empty_bath_can_only_be_born_into(kernel, rng):
    for _ in range(30):
        k_new, move = kernel.propose(rng, 0, n_free=5)
        assert (k_new, move) == (1, "birth")


def test_full_bath_can_only_die(kernel, rng):
    for _ in range(30):
        k_new, move = kernel.propose(rng, 6, n_free=5)
        assert (k_new, move) == (5, "death")


def test_birth_impossible_without_a_free_site(kernel, rng):
    """Every admissible site occupied: birth cannot be proposed."""
    k_new, move = kernel.propose(rng, 3, n_free=0)
    assert k_new == 2 and move == "death"


def test_birth_probability_is_respected(rng):
    kernel = BirthDeathKernel(k_max=100, birth_prob=0.8)
    moves = [kernel.propose(rng, 10, n_free=50)[1] for _ in range(4000)]
    assert np.mean([m == "birth" for m in moves]) == pytest.approx(0.8, abs=0.03)


def test_ratio_is_the_move_asymmetry_alone(kernel):
    """Under a uniform prior on k, symmetric moves give a ratio of exactly zero.

    The proposal ratio for a birth, (p_d/(k+1)) / (p_b/n_free), is cancelled
    exactly by the combinatorial prior term C(n,k)/C(n,k+1) = (k+1)/n_free.
    What survives is only the birth/death asymmetry.
    """
    assert kernel.log_ratio(3, "birth", n_free=5) == pytest.approx(0.0)
    assert kernel.log_ratio(3, "death", n_free=5) == pytest.approx(0.0)


def test_ratio_reflects_an_asymmetric_move_split():
    """Proposing births more often must make them correspondingly harder."""
    kernel = BirthDeathKernel(k_max=20, birth_prob=0.8)
    assert kernel.log_ratio(3, "birth", n_free=5) == pytest.approx(np.log(0.2 / 0.8))
    assert kernel.log_ratio(3, "death", n_free=5) == pytest.approx(np.log(0.8 / 0.2))


def test_death_ratio_is_the_birth_ratio_reversed():
    """Reversibility: death out of k+1 undoes birth into it."""
    kernel = BirthDeathKernel(k_max=20, birth_prob=0.7)
    birth = kernel.log_ratio(3, "birth", n_free=5)
    death = kernel.log_ratio(4, "death", n_free=4)
    assert death == pytest.approx(-birth)


def test_ratio_is_independent_of_free_sites(kernel):
    """The site count cancels between proposal and prior.

    A ratio that still varies with n_free has dropped the combinatorial term,
    and on a table of thousands of sites that omission favours every birth by
    a factor of hundreds -- the model dimension then runs to k_max whatever the
    data says.
    """
    assert kernel.log_ratio(3, "birth", n_free=10) == pytest.approx(
        kernel.log_ratio(3, "birth", n_free=5))
    assert kernel.log_ratio(3, "birth", n_free=3000) == pytest.approx(
        kernel.log_ratio(3, "birth", n_free=5))


def test_prior_on_k_enters_the_ratio():
    """Swapping the prior changes the effective prior on bath size.

    Spec Sec. 7.2: the prior on k is carried by the kernel, so an
    abundance-informed prior is a substitution rather than a sampler change.
    """
    penalty = BirthDeathKernel(k_max=20, log_prior_k=lambda k: -2.0 * k)
    assert penalty.log_ratio(3, "birth", n_free=5) == pytest.approx(-2.0)
    assert penalty.log_ratio(3, "death", n_free=5) == pytest.approx(+2.0)


def test_kernel_is_reproducible():
    k = BirthDeathKernel(k_max=6)
    a = [k.propose(np.random.default_rng(4), 3, 5) for _ in range(3)]
    b = [k.propose(np.random.default_rng(4), 3, 5) for _ in range(3)]
    assert a == b


# -------------------------------------------------------------------- moves

def test_step_returns_a_state(mover, rng):
    assert isinstance(mover.step(make_state(), Flat(), rng), State)


def test_step_does_not_mutate_the_input(mover, rng):
    st = make_state(sites=(0, 2))
    before = st.site_idx.copy(), st.k.copy()
    mover.step(st, Flat(), rng)
    assert st.site_idx.tolist() == before[0].tolist()
    assert st.k.tolist() == before[1].tolist()


def test_k_changes_by_at_most_one(mover, rng):
    st = make_state(sites=(0, 2))
    for _ in range(80):
        nxt = mover.step(st, Flat(), rng)
        assert abs(int(nxt.k[0]) - int(st.k[0])) <= 1
        st = nxt


def test_invariants_hold_throughout(mover, rng):
    st = make_state(sites=(0, 2))
    for _ in range(80):
        st = mover.step(st, Flat(), rng)
        st.check_invariants()


def test_never_exceeds_k_max(mover, rng):
    st = make_state(sites=(0, 2), k_max=6)
    for _ in range(300):
        st = mover.step(st, PrefersMany(+1.0), rng)
        assert int(st.k[0]) <= 6


def test_never_goes_below_zero(mover, rng):
    st = make_state(sites=(0, 2))
    for _ in range(300):
        st = mover.step(st, PrefersMany(-1.0), rng)
        assert int(st.k[0]) >= 0


def test_births_occupy_a_free_site(mover, rng):
    st = make_state(sites=(0,), n_sites=8)
    for _ in range(80):
        nxt = mover.step(st, PrefersMany(+1.0), rng)
        active = nxt.site_idx[0, : int(nxt.k[0])]
        assert len(set(active.tolist())) == len(active)
        st = nxt


def test_target_favouring_many_grows_the_bath(mover, rng):
    st = make_state(sites=(0,), k_max=6)
    for _ in range(200):
        st = mover.step(st, PrefersMany(+1.0), rng)
    assert int(st.k[0]) == 6


def test_target_favouring_few_empties_the_bath(mover, rng):
    st = make_state(sites=(0, 1, 2, 3), k_max=6)
    for _ in range(200):
        st = mover.step(st, PrefersMany(-1.0), rng)
    assert int(st.k[0]) == 0


def test_kernel_ratio_enters_acceptance(rng):
    """A kernel reporting a large negative ratio must suppress every move.

    Catches an implementation that computes the ratio but never uses it.
    """
    class Biased(BirthDeathKernel):
        def log_ratio(self, k, move, n_free):
            return -1e6

    st = make_state(sites=(0, 2))
    mover = RJMCMC(ParameterBlock("sites"), Biased(k_max=6))
    for _ in range(100):
        st = mover.step(st, Flat(), rng)
    assert int(st.k[0]) == 2


def test_continuous_parameters_untouched(mover, rng):
    st = make_state(sites=(0, 2))
    out = mover.step(st, Flat(), rng)
    assert out.lam[0, 0] == pytest.approx(3e-3)
    assert out.sigma[0, 0] == pytest.approx(0.1)


def test_replicas_advance_independently(mover, rng):
    st = make_state(sites=(0, 2)).expand_replicas(6)
    for _ in range(40):
        st = mover.step(st, PrefersMany(+1.0), rng)
    assert st.n_replicas == 6
    assert len(set(st.k.tolist())) > 1 or int(st.k[0]) == 6


def test_beta_tempers_acceptance(mover, rng):
    """At beta = 0 dimension moves are accepted on the kernel ratio alone."""
    st = make_state(sites=(0, 2))
    seen = set()
    for _ in range(200):
        st = mover.step(st, PrefersMany(-1.0), rng, beta=0.0)
        seen.add(int(st.k[0]))
    assert len(seen) > 2


def test_death_compacts_the_packed_slots(mover, rng):
    """Slots [0:k) must stay contiguous after a removal."""
    st = make_state(sites=(0, 1, 2, 3))
    for _ in range(60):
        st = mover.step(st, PrefersMany(-1.0), rng)
        k = int(st.k[0])
        assert np.all(st.site_idx[0, :k] >= 0)
        st.check_invariants()
