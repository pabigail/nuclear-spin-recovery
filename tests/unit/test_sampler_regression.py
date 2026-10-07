"""The sampler's trajectory, pinned against a stored trace.

Giving sites a memory of their hyperfine offsets (docs/model-specification.md
Sec. 5.3) rewires how every move writes to the state.  With the memory off,
none of that may change a single accepted or rejected proposal: the same seed
must give the same chain as before the change, to the last bit.

The stored trace was generated from the code as it stood before per-site
memory was added.  Regenerate it only for a deliberate change to the sampler,
never to make this test pass:

    python tests/unit/test_sampler_regression.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nuclear_spin_recovery import (
    RJMCMC,
    RWMH,
    AnalyticCCE1,
    BirthDeathKernel,
    ContinuousReflected,
    DiscreteLatticeWalk,
    Experiment,
    ExperimentSet,
    GaussianL2,
    GaussianOffset,
    HybridDriver,
    NeighborIndex,
    ParallelTempering,
    ParameterBlock,
    Schedule,
    SiteTable,
    State,
    Step,
    StretchedExponential,
    Target,
    Trace,
    simulate_dataset,
)

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "sampler_golden.npz"
FIELDS = ("site_idx", "k", "lam", "n_stretch", "sigma", "dA_par", "dA_perp",
          "log_prob")
N_SITES, K_MAX, N_STEPS = 12, 6, 600


def run_chain():
    """Every kind of move the package has, on a twelve-site table."""
    rng = np.random.default_rng(5)
    positions = rng.uniform(-4.0, 4.0, size=(N_SITES, 3))
    table = SiteTable(
        distance=np.linalg.norm(positions, axis=1), positions=positions,
        a_par=rng.uniform(-150.0, 150.0, N_SITES),
        a_perp=rng.uniform(10.0, 120.0, N_SITES),
        isotope=np.array(["13C"] * N_SITES), gyro=np.full(N_SITES, 6.7283))
    model = AnalyticCCE1(StretchedExponential())

    def make_state(sites):
        return State.from_sites(
            sites, n_sites=N_SITES, n_exp=1, lam=np.array([[3e-3]]),
            n_stretch=np.array([[1.0]]), sigma=np.array([[0.02]]), k_max=K_MAX)

    truth = make_state((1, 4, 7))
    truth.dA_par[0, :3] = [2.0, -1.5, 0.5]
    truth.dA_perp[0, :3] = [-1.0, 0.5, 1.5]
    blank = ExperimentSet([Experiment(tau=np.linspace(1e-4, 8e-3, 60),
                                      n_pulses=8, b_z=311.0)])
    data = simulate_dataset(truth, blank, table, model, sigma=0.002,
                            rng=np.random.default_rng(6))
    target = Target(data, model, GaussianL2(), table)

    walk = DiscreteLatticeWalk(NeighborIndex(positions, radius=6.0))
    sites = RWMH(ParameterBlock("sites"), walk)
    offsets = RWMH(ParameterBlock("offsets"), GaussianOffset(radius=1.5, scale=4.0))
    schedule = Schedule([
        Step(RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=K_MAX)), 7),
        Step(sites, 5),
        Step(offsets, 9),
        Step(RWMH(ParameterBlock("lam"),
                  ContinuousReflected(2e-4, lower=5e-4, upper=2e-2)), 3),
        Step(ParallelTempering(Schedule([Step(sites, 1), Step(offsets, 2)]),
                               n_replicas=4), 6),
    ])
    trace = Trace(n_sites=N_SITES, k_max=K_MAX, n_exp=1)
    HybridDriver(schedule).run(make_state((0, 2)), target,
                               np.random.default_rng(7), N_STEPS, trace=trace)
    return trace


def test_trajectory_is_unchanged():
    assert GOLDEN.exists(), f"golden trace not generated: {GOLDEN}"
    trace = run_chain()
    with np.load(GOLDEN) as stored:
        for name in FIELDS:
            assert np.array_equal(np.asarray(getattr(trace, name)), stored[name]), (
                f"{name} differs from the stored trajectory")
        assert list(trace.algorithm) == [str(a) for a in stored["algorithm"]]


def test_golden_chain_exercises_every_move():
    """A stored trajectory that never changed dimension, never relaxed an
    offset or never hopped would pin nothing."""
    with np.load(GOLDEN) as stored:
        assert len(set(stored["k"].tolist())) > 1
        assert np.any(stored["dA_par"] != 0.0)
        assert len({tuple(row) for row in stored["site_idx"]}) > 5
        assert len(set(stored["algorithm"].tolist())) == 5


if __name__ == "__main__":
    run_chain().save(GOLDEN)
    print(f"wrote {GOLDEN}")
