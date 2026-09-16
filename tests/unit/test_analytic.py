"""Analytic CCE-1 forward model.

The golden-value tests are the load-bearing ones: they pin the omega_L sign
and the placement of the 2*pi conversion against the reference implementation
that produced the published results (spec Sec. 11.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    State,
    StretchedExponential,
    single_spin_modulation,
)

GYRO_13C = 6.7283          # rad / (ms * G)
B_311 = 311.0              # G

# TODO(phase-1): regenerate from the reference implementation and paste here:
#   python -c "import rjmcmc, numpy as np; \
#              print(repr(rjmcmc.coherence_one_spin(np.array([...]), A_par, A_perp, N, B)))"
# run inside ~/Desktop/research/nuclear-spin-recovery.  Values below are
# placeholders and are expected to fail until they are filled in.
GOLDEN = [
    # (tau_ms, a_par_khz, a_perp_khz, n_pulses, b_z_G, expected_M)
    (2.0e-3, 50.0, 30.0, 16, B_311, None),
    (4.0e-3, -20.0, 80.0, 8, B_311, None),
    (1.0e-3, 120.0, 45.0, 32, 403.0, None),
]


@pytest.fixture
def model():
    return AnalyticCCE1(StretchedExponential())


# --------------------------------------------------------------- golden values

@pytest.mark.parametrize("tau,a_par,a_perp,n_pulses,b_z,expected", GOLDEN)
def test_matches_reference_implementation(tau, a_par, a_perp, n_pulses, b_z, expected):
    """Pins sign and 2*pi conventions against the published reference code."""
    assert expected is not None, "golden values not yet generated"
    got = single_spin_modulation(
        np.array([tau]), a_par, a_perp, n_pulses, b_z, GYRO_13C
    )
    assert got[0] == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------------------ analytic limits

def test_zero_a_perp_gives_no_modulation():
    """m_x = 0 kills the modulation term exactly, for any tau and N."""
    tau = np.linspace(1e-4, 8e-3, 40)
    got = single_spin_modulation(tau, a_par=50.0, a_perp=0.0,
                                 n_pulses=16, b_z=B_311, gyro=GYRO_13C)
    assert got == pytest.approx(np.ones_like(tau))


def test_zero_pulses_gives_no_modulation():
    """sin(N phi / 2) vanishes at N = 0."""
    tau = np.linspace(1e-4, 8e-3, 40)
    got = single_spin_modulation(tau, a_par=50.0, a_perp=30.0,
                                 n_pulses=0, b_z=B_311, gyro=GYRO_13C)
    assert got == pytest.approx(np.ones_like(tau))


def test_modulation_is_bounded():
    """M must stay in [-1, 1]; the denominator can approach zero."""
    rng = np.random.default_rng(0)
    tau = rng.uniform(1e-5, 1e-2, 400)
    for a_par in (-200.0, -5.0, 0.0, 5.0, 200.0):
        for a_perp in (0.0, 1.0, 50.0, 200.0):
            got = single_spin_modulation(tau, a_par, a_perp, 16, B_311, GYRO_13C)
            assert np.all(np.isfinite(got))
            assert np.all(got >= -1.0 - 1e-9)
            assert np.all(got <= 1.0 + 1e-9)


def test_sign_of_gyro_changes_the_result():
    """omega_L enters m_z, so its sign is physical, not cosmetic."""
    tau = np.array([2e-3])
    pos = single_spin_modulation(tau, 50.0, 30.0, 16, B_311, GYRO_13C)
    neg = single_spin_modulation(tau, 50.0, 30.0, 16, B_311, -GYRO_13C)
    assert pos[0] != pytest.approx(neg[0])


def test_two_pi_convention():
    """Couplings are kHz and converted internally; gyro is already angular.

    Doubling the field doubles omega_L, which must change the result in the
    same way as supplying a doubled gyro.
    """
    tau = np.array([2e-3])
    a = single_spin_modulation(tau, 50.0, 30.0, 16, 2 * B_311, GYRO_13C)
    b = single_spin_modulation(tau, 50.0, 30.0, 16, B_311, 2 * GYRO_13C)
    assert a[0] == pytest.approx(b[0], rel=1e-12)


# ------------------------------------------------------------- bath behaviour

def test_empty_bath_is_pure_envelope(model, single_experiment, tiny_site_table):
    """The empty product is 1, so coherence collapses to the envelope."""
    st = State.from_sites(
        (), n_sites=len(tiny_site_table), n_exp=1,
        lam=np.array([[3e-3]]), n_stretch=np.array([[1.0]]),
        sigma=np.array([[0.1]]), k_max=8,
    )
    got = model.coherence(st, single_experiment, tiny_site_table)
    expected = np.exp(-single_experiment.tau_all / 3e-3)
    assert got[0] == pytest.approx(expected)


def test_coherence_bounded(model, single_experiment, tiny_site_table):
    st = State.from_sites(
        (0, 2), n_sites=len(tiny_site_table), n_exp=1,
        lam=np.array([[3e-3]]), n_stretch=np.array([[1.0]]),
        sigma=np.array([[0.1]]), k_max=8,
    )
    got = model.coherence(st, single_experiment, tiny_site_table)
    assert np.all(got >= 0.0) and np.all(got <= 1.0)


def test_coherence_is_permutation_invariant(model, single_experiment, tiny_site_table):
    """Spin order is bookkeeping, not physics."""
    kw = dict(
        n_sites=len(tiny_site_table), n_exp=1,
        lam=np.array([[3e-3]]), n_stretch=np.array([[1.0]]),
        sigma=np.array([[0.1]]), k_max=8,
    )
    a = model.coherence(State.from_sites((0, 2), **kw), single_experiment, tiny_site_table)
    b = model.coherence(State.from_sites((2, 0), **kw), single_experiment, tiny_site_table)
    assert a == pytest.approx(b)


def test_bath_factorizes(model, single_experiment, tiny_site_table):
    """The k-spin signal is built from the per-spin modulations."""
    tau = single_experiment.tau_all
    m0 = single_spin_modulation(
        tau, tiny_site_table.a_par[0], tiny_site_table.a_perp[0],
        16, 311.0, tiny_site_table.gyro[0],
    )
    m2 = single_spin_modulation(
        tau, tiny_site_table.a_par[2], tiny_site_table.a_perp[2],
        16, 311.0, tiny_site_table.gyro[2],
    )
    st = State.from_sites(
        (0, 2), n_sites=len(tiny_site_table), n_exp=1,
        lam=np.array([[1e9]]), n_stretch=np.array([[1.0]]),
        sigma=np.array([[0.1]]), k_max=8,
    )
    got = model.coherence(st, single_experiment, tiny_site_table)[0]
    assert got == pytest.approx(0.5 * (1.0 + m0 * m2), rel=1e-6)


def test_mixed_isotopes_differ_from_single(model, single_experiment):
    """Per-spin gamma_n is the spec Sec. 4.1 generalization; it must matter."""
    from nuclear_spin_recovery import SiteTable

    common = dict(
        distance=np.array([2.0, 2.0]),
        positions=np.zeros((2, 3)),
        a_par=np.array([50.0, 50.0]),
        a_perp=np.array([30.0, 30.0]),
    )
    same = SiteTable(isotope=np.array(["13C", "13C"]),
                     gyro=np.array([6.7283, 6.7283]), **common)
    mixed = SiteTable(isotope=np.array(["13C", "29Si"]),
                      gyro=np.array([6.7283, -5.319]), **common)
    kw = dict(n_sites=2, n_exp=1, lam=np.array([[1e9]]),
              n_stretch=np.array([[1.0]]), sigma=np.array([[0.1]]), k_max=4)
    st = State.from_sites((0, 1), **kw)
    a = model.coherence(st, single_experiment, same)
    b = model.coherence(st, single_experiment, mixed)
    assert not np.allclose(a, b)


def test_all_same_isotope_matches_single_gamma_formula(model, single_experiment,
                                                       tiny_site_table):
    """The generalization must reduce exactly to the published expression."""
    tau = single_experiment.tau_all
    expected = np.ones_like(tau)
    for site in (0, 1):
        expected = expected * single_spin_modulation(
            tau, tiny_site_table.a_par[site], tiny_site_table.a_perp[site],
            16, 311.0, tiny_site_table.gyro[site],
        )
    st = State.from_sites(
        (0, 1), n_sites=len(tiny_site_table), n_exp=1,
        lam=np.array([[1e9]]), n_stretch=np.array([[1.0]]),
        sigma=np.array([[0.1]]), k_max=8,
    )
    got = model.coherence(st, single_experiment, tiny_site_table)[0]
    assert got == pytest.approx(0.5 * (1.0 + expected), rel=1e-6)


def test_replicas_match_loop(model, single_experiment, tiny_site_table):
    """Vectorizing over replicas must equal looping over them."""
    st = State.from_sites(
        (0, 2), n_sites=len(tiny_site_table), n_exp=1,
        lam=np.array([[3e-3]]), n_stretch=np.array([[1.0]]),
        sigma=np.array([[0.1]]), k_max=8,
    ).expand_replicas(3)
    st.lam[1, 0] = 5e-3
    st.lam[2, 0] = 9e-3
    batched = model.coherence(st, single_experiment, tiny_site_table)
    assert batched.shape == (3, single_experiment.n_points)
    for r in range(3):
        one = model.coherence(
            st.collapse_to_cold() if r == 0 else _replica(st, r),
            single_experiment,
            tiny_site_table,
        )
        assert batched[r] == pytest.approx(one[0])


def _replica(state, r):
    """Extract a single replica as its own state."""
    out = state.copy()
    out.site_idx = state.site_idx[r : r + 1].copy()
    out.k = state.k[r : r + 1].copy()
    out.lam = state.lam[r : r + 1].copy()
    out.n_stretch = state.n_stretch[r : r + 1].copy()
    out.sigma = state.sigma[r : r + 1].copy()
    return out


def test_multi_experiment_uses_per_experiment_pulses(model, two_experiments,
                                                     tiny_site_table):
    """N=8 and N=16 points must not be computed with the same pulse number."""
    st = State.from_sites(
        (0,), n_sites=len(tiny_site_table), n_exp=2,
        lam=np.array([[1e9, 1e9]]), n_stretch=np.array([[1.0, 1.0]]),
        sigma=np.array([[0.1, 0.1]]), k_max=8,
    )
    got = model.coherence(st, two_experiments, tiny_site_table)[0]
    # tau = 1e-3 appears in both experiments, at index 0 and index 3
    assert got[0] != pytest.approx(got[3])
