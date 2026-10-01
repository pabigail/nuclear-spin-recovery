"""Envelope extrapolation to pulse numbers the posterior has not measured.

The designer's default is to refuse a candidate at an unmeasured pulse number,
because it has no sampled decay constant for it.  ``DecouplingScaling`` makes
the alternative explicit: lambda scaled from the nearest measured pulse number
as ``(N / N_ref) ** (gamma - 1)``.  The tests pin the arithmetic, that the
designer uses exactly the scaled value, that it scales from the right
reference, and -- most important -- that a measured pulse number is never
overridden by the assumption.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    DecouplingScaling,
    Experiment,
    ExperimentDesigner,
    ExperimentSet,
    InformationDensity,
    ParticleSet,
    PredictiveVariance,
    StretchedExponential,
)

B_Z = 311.0
TAU = np.linspace(1e-4, 8e-3, 20)
W3 = np.array([0.5, 0.3, 0.2])


def rng(seed=0):
    return np.random.default_rng(seed)


def particles_with(lam_columns):
    """Three particles; ``lam_columns`` gives each experiment's lambda."""
    n, k_max = 3, 8
    n_exp = len(lam_columns)
    site_idx = np.full((n, k_max), -1)
    for i, c in enumerate([[0, 2], [0], [2]]):
        site_idx[i, : len(c)] = c
    return ParticleSet(
        site_idx=site_idx, k=np.array([2, 1, 1]), weight=W3,
        dA_par=np.zeros((n, k_max)), dA_perp=np.zeros((n, k_max)),
        lam=np.tile(lam_columns, (n, 1)), n_stretch=np.ones((n, n_exp)),
        sigma=np.full((n, n_exp), 0.002), n_sites=4, k_max=k_max)


def measured_at(*pulses, b_z=B_Z):
    return ExperimentSet([Experiment(tau=np.linspace(2e-4, 6e-3, 12),
                                     n_pulses=n, b_z=b_z) for n in pulses])


def designer(table, measured, envelope=None):
    return ExperimentDesigner(PredictiveVariance(), InformationDensity(),
                              AnalyticCCE1(StretchedExponential()), table,
                              measured, envelope=envelope)


def cand(n_pulses, b_z=B_Z):
    return Experiment(tau=TAU, n_pulses=n_pulses, b_z=b_z)


# --------------------------------------------------------------------------
# the scaling
# --------------------------------------------------------------------------


def test_scaling_at_the_reference_is_the_identity():
    assert DecouplingScaling(2 / 3).scale(3e-3, 16, 16) == pytest.approx(3e-3)


def test_two_thirds_gives_lambda_as_n_to_the_minus_one_third():
    """T ~ N^(2/3) in total time; a repetition lasts 2 N tau; so in tau the
    decay constant goes as N^(-1/3).  Eightfold N halves it."""
    assert DecouplingScaling(2 / 3).scale(3e-3, 8, 64) == pytest.approx(1.5e-3)


def test_gamma_one_keeps_lambda_fixed_in_tau():
    assert DecouplingScaling(1.0).scale(3e-3, 4, 64) == pytest.approx(3e-3)


def test_scaling_is_elementwise_over_particles():
    out = DecouplingScaling(0.0).scale(np.array([[2e-3], [4e-3]]), 4, 8)
    np.testing.assert_allclose(out, [[1e-3], [2e-3]])


@pytest.mark.parametrize("gamma", [np.nan, np.inf])
def test_gamma_must_be_finite(gamma):
    with pytest.raises(ValueError):
        DecouplingScaling(gamma)


@pytest.mark.parametrize("ends", [(0, 8), (8, 0)])
def test_scaling_needs_positive_pulse_numbers(ends):
    with pytest.raises(ValueError):
        DecouplingScaling(2 / 3).scale(3e-3, *ends)


# --------------------------------------------------------------------------
# the designer
# --------------------------------------------------------------------------


def test_without_an_envelope_an_unmeasured_pulse_number_is_still_refused(
        tiny_site_table):
    d = designer(tiny_site_table, measured_at(8))
    with pytest.raises(ValueError, match="n_pulses"):
        d.rank(particles_with([3e-3]), [cand(64)], budget=4.0, rng=rng())


def test_the_designer_uses_exactly_the_scaled_lambda(tiny_site_table):
    """Extrapolating from N = 8 at 3 us must equal having measured N = 64 at
    the scaled 1.5 us."""
    scaled = designer(tiny_site_table, measured_at(8),
                      envelope=DecouplingScaling(2 / 3)).rank(
        particles_with([3e-3]), [cand(64)], budget=4.0, rng=rng())
    direct = designer(tiny_site_table, measured_at(64)).rank(
        particles_with([1.5e-3]), [cand(64)], budget=4.0, rng=rng())
    np.testing.assert_allclose(scaled, direct, rtol=1e-12)


def test_a_measured_pulse_number_ignores_the_scaling(tiny_site_table):
    """N = 16 was measured, with a lambda the scaling would not predict; the
    posterior's value must win."""
    measured = measured_at(8, 16)
    ps = particles_with([3e-3, 5e-3])           # scaling would say 2.38 us
    with_env = designer(tiny_site_table, measured,
                        envelope=DecouplingScaling(2 / 3)).rank(
        ps, [cand(16)], budget=4.0, rng=rng())
    without = designer(tiny_site_table, measured).rank(
        ps, [cand(16)], budget=4.0, rng=rng())
    np.testing.assert_array_equal(with_env, without)


def test_the_reference_is_the_nearest_pulse_number_in_log(tiny_site_table):
    """Measured at 4 and 16; a CPMG-32 candidate scales from 16, not 4."""
    measured = measured_at(4, 16)
    ps = particles_with([6e-3, 3e-3])
    via_16 = DecouplingScaling(2 / 3).scale(3e-3, 16, 32)
    scaled = designer(tiny_site_table, measured,
                      envelope=DecouplingScaling(2 / 3)).rank(
        ps, [cand(32)], budget=4.0, rng=rng())
    direct = designer(tiny_site_table, measured_at(32)).rank(
        particles_with([float(via_16)]), [cand(32)], budget=4.0, rng=rng())
    np.testing.assert_allclose(scaled, direct, rtol=1e-12)


def test_a_different_field_is_still_refused(tiny_site_table):
    d = designer(tiny_site_table, measured_at(8),
                 envelope=DecouplingScaling(2 / 3))
    with pytest.raises(ValueError):
        d.rank(particles_with([3e-3]), [cand(64, b_z=400.0)], budget=4.0,
               rng=rng())


def test_a_proposal_can_be_at_a_new_pulse_number(tiny_site_table):
    d = designer(tiny_site_table, measured_at(8),
                 envelope=DecouplingScaling(2 / 3))
    out = d.propose(particles_with([3e-3]), [cand(32)], budget=4.0, rng=rng())
    assert out.n_pulses == 32
