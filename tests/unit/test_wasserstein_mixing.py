"""The transport term as a mixing fraction.

WassersteinL2 takes a rate, ``weight``, in log-likelihood per unit of
normalised transport; nothing says which value is "half and half".
MixedWassersteinL2 fixes a scale from the data so that ``zeta`` is a fraction:
at zeta = 0.5 the featureless prediction is charged equally by both terms.

Pinned here: the exact reduction at zeta = 0, the equivalence with
WassersteinL2 that makes the penalty form a reparametrisation rather than a new
model, the meaning of the flat scale, the temperature difference between the
two forms, and the pre-rewrite "level" measure.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from nuclear_spin_recovery import (
    RWMH,
    AnalyticCCE1,
    ContinuousReflected,
    Experiment,
    ExperimentSet,
    GaussianL2,
    Likelihood,
    MixedWassersteinL2,
    ParameterBlock,
    State,
    SiteScaledOffset,
    StretchedExponential,
    Target,
    Trace,
    WassersteinL2,
    level_measures,
    simulate_dataset,
    wasserstein_signal_distance,
)

TAU = np.linspace(0.0, 1.0, 201)


def bump(centre, width=0.02, depth=0.4, tau=TAU):
    return 1.0 - depth * np.exp(-0.5 * ((tau - centre) / width) ** 2)


@pytest.fixture
def model():
    return AnalyticCCE1(StretchedExponential())


@pytest.fixture
def expset():
    return ExperimentSet([
        Experiment(tau=np.linspace(1e-4, 8e-3, 40), n_pulses=16, b_z=311.0),
        Experiment(tau=np.linspace(1e-4, 6e-3, 30), n_pulses=32, b_z=311.0),
    ])


def make_state(sites, table, *, lam=3e-3, sigma=0.02, k_max=8, n_exp=2,
               site_memory=False):
    return State.from_sites(
        np.sort(np.asarray(list(sites), dtype=int)),
        n_sites=len(table), n_exp=n_exp,
        lam=np.full((1, n_exp), lam), n_stretch=np.ones((1, n_exp)),
        sigma=np.full((1, n_exp), sigma), k_max=k_max, site_memory=site_memory,
    )


@pytest.fixture
def scene(tiny_site_table, expset, model):
    truth = make_state([0, 2], tiny_site_table)
    data = simulate_dataset(truth, expset, tiny_site_table, model, sigma=0.002,
                            rng=np.random.default_rng(0))
    return truth, data


class Flat:
    """A forward model that predicts full coherence everywhere."""

    def coherence(self, state, expset, site_table):
        return np.ones((state.sigma.shape[0], expset.n_points))


def old_wasserstein_error(observed, simulated):
    """WassersteinError of the pre-rewrite error.py, one experiment, verbatim."""
    low = min(observed.min(), simulated.min())
    obs, sim = observed - low, simulated - low
    if obs.sum() > 0 and sim.sum() > 0:
        x = np.arange(len(observed))
        return wasserstein_distance(x, x, obs / obs.sum(), sim / sim.sum())
    return 0.0


# --------------------------------------------------------------------------
# the exact reduction at zeta = 0
# --------------------------------------------------------------------------


@pytest.mark.parametrize("form", MixedWassersteinL2.FORMS)
@pytest.mark.parametrize("sites", [(0,), (1, 2), (0, 2, 3)])
def test_zero_zeta_is_the_gaussian_bit_for_bit(scene, tiny_site_table, model,
                                               form, sites):
    _truth, data = scene
    state = make_state(sites, tiny_site_table)
    mixed = MixedWassersteinL2(zeta=0.0, form=form).log_prob(
        state, data, model, tiny_site_table)
    assert np.array_equal(mixed, GaussianL2().log_prob(state, data, model,
                                                       tiny_site_table))


def test_zero_zeta_gives_an_identical_accepted_path(scene, tiny_site_table, model):
    _truth, data = scene

    def run(likelihood):
        target = Target(data, model, likelihood, tiny_site_table)
        sampler = RWMH(ParameterBlock("lam"),
                       ContinuousReflected(radius=2e-4, lower=5e-4, upper=2e-2))
        trace = Trace(n_sites=len(tiny_site_table), k_max=8, n_exp=2)
        sampler.run(make_state([0, 2], tiny_site_table), target,
                    np.random.default_rng(4), n_steps=60, trace=trace)
        return np.asarray(trace.lam)

    assert np.array_equal(run(GaussianL2()), run(MixedWassersteinL2(zeta=0.0)))


def test_zero_zeta_gives_an_identical_offset_walk_with_site_memory(
        scene, tiny_site_table, model):
    """The relaxations reach the likelihood only through the forward model, so
    at zeta = 0 a site-scaled offset walk takes the GaussianL2 path exactly."""
    _truth, data = scene

    def run(likelihood):
        target = Target(data, model, likelihood, tiny_site_table)
        walk = RWMH(ParameterBlock("offsets"), SiteScaledOffset(
            2.0, tiny_site_table, fraction_par=0.1, fraction_perp=0.1, floor=0.5))
        state = make_state([0, 2], tiny_site_table, site_memory=True)
        rng = np.random.default_rng(5)
        path = []
        for _ in range(60):
            state = walk.step(state, target, rng)
            path.append(np.concatenate([state.dA_par[0], state.dA_perp[0]]))
        return np.asarray(path), state.site_dA_par.copy()

    gaussian, mixed = run(GaussianL2()), run(MixedWassersteinL2(zeta=0.0))
    assert np.any(gaussian[0] != 0.0), "the walk should have moved"
    assert np.array_equal(gaussian[0], mixed[0])
    assert np.array_equal(gaussian[1], mixed[1])


@pytest.mark.parametrize("form", MixedWassersteinL2.FORMS)
def test_an_offset_back_to_the_truth_lowers_the_transport_charge(
        tiny_site_table, expset, model, form):
    """Off-table truth: the spin on site 0 sits 10 kHz from its table value.
    Relaxing the candidate's coupling back onto it is rewarded by both terms."""
    truth = make_state([0, 2], tiny_site_table)
    shift = 10.0                               # kHz, the sweeps' largest delta
    truth.set_offset(0, 0, 0, shift)
    data = simulate_dataset(truth, expset, tiny_site_table, model, sigma=0.002,
                            rng=np.random.default_rng(0))
    on_table = make_state([0, 2], tiny_site_table)
    relaxed = make_state([0, 2], tiny_site_table)
    relaxed.set_offset(0, 0, 0, shift)

    mixed = MixedWassersteinL2(zeta=0.5, form=form)
    transport = {name: mixed._transport(model.coherence(s, data, tiny_site_table),
                                        data).sum()
                 for name, s in (("on_table", on_table), ("relaxed", relaxed))}
    assert transport["relaxed"] < transport["on_table"]
    assert (mixed.log_prob(relaxed, data, model, tiny_site_table)
            > mixed.log_prob(on_table, data, model, tiny_site_table)).all()


# --------------------------------------------------------------------------
# the penalty form is WassersteinL2 with a computed weight
# --------------------------------------------------------------------------


@pytest.mark.parametrize("zeta", [0.1, 0.5, 0.9])
def test_penalty_form_equals_wassersteinl2_at_the_implied_weight(
        tiny_site_table, model, zeta):
    """One experiment, one sigma: weight = zeta/(1-zeta) * s / (2 sigma**2)."""
    single = ExperimentSet([Experiment(tau=np.linspace(1e-4, 8e-3, 40),
                                       n_pulses=16, b_z=311.0)])
    truth = make_state([0, 2], tiny_site_table, n_exp=1)
    data = simulate_dataset(truth, single, tiny_site_table, model, sigma=0.002,
                            rng=np.random.default_rng(1))
    state = make_state([1, 3], tiny_site_table, n_exp=1, sigma=0.03)
    mixed = MixedWassersteinL2(zeta=zeta)
    weight = zeta / (1 - zeta) * mixed.scales(data)[0] / (2 * 0.03 ** 2)
    assert mixed.log_prob(state, data, model, tiny_site_table) == pytest.approx(
        WassersteinL2(weight=weight).log_prob(state, data, model, tiny_site_table),
        rel=1e-12)


@pytest.mark.parametrize("zeta", [0.25, 0.5, 0.75])
def test_the_two_forms_differ_only_by_the_factor_one_minus_zeta(
        scene, tiny_site_table, model, zeta):
    """Same ratio of the terms; the mixture form is the penalty form cooled."""
    _truth, data = scene
    state = make_state([1, 3], tiny_site_table)
    penalty = MixedWassersteinL2(zeta=zeta, form="penalty").log_prob(
        state, data, model, tiny_site_table)
    mixture = MixedWassersteinL2(zeta=zeta, form="mixture").log_prob(
        state, data, model, tiny_site_table)
    assert mixture == pytest.approx((1 - zeta) * penalty, rel=1e-12)


def test_the_penalty_only_ever_lowers_the_log_likelihood(scene, tiny_site_table,
                                                         model):
    _truth, data = scene
    state = make_state([1, 3], tiny_site_table)
    plain = GaussianL2().log_prob(state, data, model, tiny_site_table)
    for zeta in (0.1, 0.5, 0.9):
        assert np.all(MixedWassersteinL2(zeta=zeta).log_prob(
            state, data, model, tiny_site_table) <= plain)


def test_the_penalty_grows_with_zeta(scene, tiny_site_table, model):
    _truth, data = scene
    state = make_state([1, 3], tiny_site_table)
    values = [MixedWassersteinL2(zeta=z).log_prob(state, data, model,
                                                  tiny_site_table)[0]
              for z in (0.0, 0.25, 0.5, 0.75, 0.95)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


# --------------------------------------------------------------------------
# what the flat scale means
# --------------------------------------------------------------------------


def test_the_flat_scale_is_the_flat_squared_residual_for_the_dip_measure(scene):
    """A flat prediction has no dip, so it is at the maximal distance 1."""
    _truth, data = scene
    scales = MixedWassersteinL2(zeta=0.5).scales(data)
    expected = [np.sum((d - 1.0) ** 2) for d in data.split(data.data_all)]
    assert scales == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("measure", ["dip", "level"])
def test_at_one_half_the_flat_prediction_is_charged_equally(scene,
                                                            tiny_site_table,
                                                            measure):
    """The definition of 'half and half', checked through log_prob."""
    _truth, data = scene
    state = make_state([0], tiny_site_table)
    gaussian = GaussianL2().log_prob(state, data, Flat(), tiny_site_table)
    mixture = MixedWassersteinL2(zeta=0.5, form="mixture", measure=measure)
    total = mixture.log_prob(state, data, Flat(), tiny_site_table)
    assert total == pytest.approx(gaussian, rel=1e-12)   # 0.5 G + 0.5 T, T = G


def test_a_numeric_scale_is_used_as_given(scene):
    _truth, data = scene
    assert MixedWassersteinL2(zeta=0.5, scale=7.0).scales(data) == pytest.approx([7.0, 7.0])
    assert MixedWassersteinL2(zeta=0.5, scale=[2.0, 3.0]).scales(data) == \
        pytest.approx([2.0, 3.0])


def test_the_flat_scale_is_computed_once_per_experiment_set(scene):
    _truth, data = scene
    mixed = MixedWassersteinL2(zeta=0.5)
    assert mixed.scales(data) is mixed.scales(data)


def test_the_scale_follows_a_new_experiment_set(scene, tiny_site_table, model,
                                                expset):
    truth, data = scene
    louder = simulate_dataset(truth, expset, tiny_site_table, model, sigma=0.05,
                              rng=np.random.default_rng(9))
    mixed = MixedWassersteinL2(zeta=0.5)
    first = mixed.scales(data).copy()
    assert not np.allclose(mixed.scales(louder), first)


# --------------------------------------------------------------------------
# the pre-rewrite measure
# --------------------------------------------------------------------------


def test_level_measure_reproduces_the_pre_rewrite_distance():
    """On a uniform grid the old index positions only rescale by n - 1."""
    a, b = bump(0.3), bump(0.45, depth=0.25)
    old = old_wasserstein_error(2 * a - 1, 2 * b - 1)      # old code used +-1 coherence
    new = wasserstein_signal_distance(a, b, TAU, measure="level")
    assert old == pytest.approx((len(TAU) - 1) * new, rel=1e-9)


def test_level_measure_hardly_sees_a_translated_dip():
    """Why the rewrite changed the measure: most of the old mass is background."""
    a, b = bump(0.40), bump(0.41)
    dip = wasserstein_signal_distance(a, b, TAU, measure="dip")
    level = wasserstein_signal_distance(a, b, TAU, measure="level")
    assert dip == pytest.approx(0.01, abs=1e-3)
    assert level < 0.1 * dip


def test_level_measures_share_the_joint_minimum():
    a, b = np.array([0.9, 0.5, 1.0]), np.array([0.7, 0.8, 1.0])
    wa, wb = level_measures(a, b)
    assert wa == pytest.approx([0.4, 0.0, 0.5])
    assert wb == pytest.approx([0.2, 0.3, 0.5])


def test_an_unknown_measure_raises():
    with pytest.raises(ValueError):
        wasserstein_signal_distance(bump(0.3), bump(0.4), TAU, measure="values")


def test_the_default_measure_is_unchanged():
    a, b = bump(0.3), bump(0.5)
    assert wasserstein_signal_distance(a, b, TAU) == \
        wasserstein_signal_distance(a, b, TAU, measure="dip")


# --------------------------------------------------------------------------
# argument checking
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    dict(zeta=-0.1), dict(zeta=1.1), dict(zeta=1.0),            # penalty needs zeta < 1
    dict(zeta=0.5, form="product"), dict(zeta=0.5, measure="values"),
    dict(zeta=0.5, scale="empty"), dict(zeta=0.5, scale=0.0),
    dict(zeta=0.5, scale=[1.0, -1.0]), dict(zeta=0.5, scale=[[1.0]]),
])
def test_bad_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        MixedWassersteinL2(**kwargs)


def test_pure_transport_is_allowed_in_the_mixture_form(scene, tiny_site_table,
                                                       model):
    _truth, data = scene
    value = MixedWassersteinL2(zeta=1.0, form="mixture").log_prob(
        make_state([1, 3], tiny_site_table), data, model, tiny_site_table)
    assert np.all(np.isfinite(value)) and np.all(value <= 0.0)


# --------------------------------------------------------------------------
# it is a Likelihood like any other
# --------------------------------------------------------------------------


def test_it_subclasses_the_likelihood_abc():
    assert issubclass(MixedWassersteinL2, Likelihood)


def test_one_value_per_replica_with_per_replica_sigma(scene, tiny_site_table, model):
    _truth, data = scene
    replicas = make_state([0, 2], tiny_site_table).expand_replicas(3)
    replicas.sigma[:] = np.array([[0.01, 0.02], [0.02, 0.02], [0.04, 0.03]])
    value = MixedWassersteinL2(zeta=0.5).log_prob(replicas, data, model,
                                                  tiny_site_table)
    assert value.shape == (3,)
    assert value[0] < value[1] < value[2]      # larger sigma, smaller charge


def test_it_installs_in_a_target(scene, tiny_site_table, model):
    _truth, data = scene
    target = Target(data, model, MixedWassersteinL2(zeta=0.5), tiny_site_table)
    assert np.all(np.isfinite(target.log_prob(make_state([0, 2], tiny_site_table))))
