"""T9: does adaptive design beat a uniform grid at equal measurement time?

One round of the loop the design engine exists for, run on several simulated
baths so the answer is not one seed's:

1. **Fit** a sparse, noisy first experiment -- 20 points at noise 0.02 on the
   detectable table, k_true = 6 -- with pooled ensembles.  Piloted to leave
   the posterior ambiguous but at the noise: effective size 5.5, residual
   1.00 sigma, three of six spins detected at 0.16 or below.
2. **Design** a follow-up of the same total time three ways:
     adaptive  ExperimentDesigner, EIG ranking over five windows of a dense
               grid and the whole of it, InformationDensity allocation
     uniform   the same time spread evenly over the dense grid
     anti      the same time on the least informative points
3. **Measure** each at the truth, with noise sigma / sqrt(w_j).
4. **Refit** on the first experiment plus the follow-up, every design from
   the same starts and root seed, so the sampler's randomness is shared and
   the comparison is paired.

Two metrics, because they answer different questions:

  mean R_i    detection over the refitted posterior -- the ladder's metric,
              and the claim T9 makes.  Depends on the sampler as well as on
              the data.
  update      posterior mass on the true bath after reweighting the *first*
              posterior's particles by the follow-up likelihood.  No refit,
              so it isolates the design from the sampler; zero whenever the
              first posterior never visited the truth.

And the degenerate control: a posterior collapsed onto one bath must be
declined by the designer -- NothingToLearn -- while the uniform design, which
needs no posterior, is still built.

Thresholds are set from this output the standard way (test-plan Sec. 2): the
measured value, the value with the mechanism disabled, and a threshold between
them with margin.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from nuclear_spin_recovery import (
    RJMCMC,
    RWMH,
    AnalyticCCE1,
    BirthDeathKernel,
    DiscreteLatticeWalk,
    EnsembleRunner,
    ExpectedInformationGain,
    Experiment,
    ExperimentDesigner,
    ExperimentSet,
    GaussianL2,
    InformationDensity,
    LeastInformative,
    NeighborIndex,
    NothingToLearn,
    ParallelTempering,
    ParameterBlock,
    ParticleSet,
    PredictiveVariance,
    Schedule,
    SiteTable,
    State,
    Step,
    StretchedExponential,
    Target,
    UniformThinning,
    simulate_dataset,
    spread_across_k,
)
from nuclear_spin_recovery.design.particles import _same_bath
from nuclear_spin_recovery.post import summarize

FULL = SiteTable.from_ivady_file("nv-2.txt", strong_thresh=750.0, weak_thresh=5.0)
KEEP = np.hypot(FULL.a_par, FULL.a_perp) >= 100.0
TABLE = SiteTable(distance=FULL.distance[KEEP], positions=FULL.positions[KEEP],
                  a_par=FULL.a_par[KEEP], a_perp=FULL.a_perp[KEEP],
                  isotope=FULL.isotope[KEEP], gyro=FULL.gyro[KEEP])
MODEL = AnalyticCCE1(StretchedExponential())
N_PULSES, B_Z, LAM, K_MAX = 16, 311.0, 3e-3, 32
LIK_SIGMA = 0.02

K_TRUE = 6
FIRST_POINTS, NOISE = 20, 0.02
BUDGET = float(FIRST_POINTS)          # the follow-up costs what the first did
DENSE = np.linspace(0.0, 8e-3, 250, endpoint=False) + 8e-3 / 250
N_WINDOWS = 5
N_ENSEMBLES, STEPS, BURN, K_START = 4, 1200, 400, (3, 9)
PARTICLE_STRIDE = 10
EIG_DRAWS = 256

SEEDS = (7, 11, 23, 42, 404, 2026)


def grid(n):
    return np.linspace(0.0, 8e-3, n, endpoint=False) + 8e-3 / n


def state(sites, n_exp=1, sigma=LIK_SIGMA):
    return State.from_sites(np.sort(np.asarray(list(sites), int)),
                            n_sites=len(TABLE), n_exp=n_exp,
                            lam=np.full((1, n_exp), LAM),
                            n_stretch=np.ones((1, n_exp)),
                            sigma=np.full((1, n_exp), sigma), k_max=K_MAX)


def schedule():
    walk = DiscreteLatticeWalk(NeighborIndex(TABLE.positions, radius=5.0))
    return Schedule([
        Step(RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=K_MAX)), 80),
        Step(ParallelTempering(
            Schedule([Step(RWMH(ParameterBlock("sites"), walk), 1)]),
            n_replicas=6), 160),
    ])


def fit(data, root_seed):
    """Pooled ensembles over ``data``; every design refits with the same root."""
    n_exp = data.n_experiments
    runner = EnsembleRunner(
        schedule(), n_ensembles=N_ENSEMBLES, n_steps=STEPS, n_burn=BURN,
        init=spread_across_k(lambda s: state(s, n_exp=n_exp), K_START),
        init_name="spread_across_k")
    return runner.run(Target(data, MODEL, GaussianL2(), TABLE),
                      root_seed=root_seed).pooled


def candidates():
    """Five windows of the dense grid, and the whole of it."""
    windows = np.array_split(DENSE, N_WINDOWS)
    return [Experiment(tau=w, n_pulses=N_PULSES, b_z=B_Z) for w in windows] + [
        Experiment(tau=DENSE, n_pulses=N_PULSES, b_z=B_Z)]


def uniform_design():
    """Needs no posterior, so it survives the degenerate control."""
    idx, weight = UniformThinning(FIRST_POINTS).select(
        np.zeros((1, DENSE.size)), [1.0], NOISE, BUDGET, None)
    return Experiment(tau=DENSE[idx], n_pulses=N_PULSES, b_z=B_Z, sigma=NOISE,
                      weight=weight)


def design(kind, particles, measured, rng):
    if kind == "uniform":
        return uniform_design()
    if kind == "adaptive":
        designer = ExperimentDesigner(ExpectedInformationGain(n_draws=EIG_DRAWS),
                                      InformationDensity(), MODEL, TABLE, measured)
        return designer.propose(particles, candidates(), budget=BUDGET, rng=rng,
                                exclude=measured)
    if kind == "anti":
        designer = ExperimentDesigner(PredictiveVariance(),
                                      LeastInformative(FIRST_POINTS), MODEL,
                                      TABLE, measured)
        return designer.propose(particles, [candidates()[-1]], budget=BUDGET,
                                rng=rng, exclude=measured)
    raise ValueError(kind)


def truth_couplings(sites):
    return np.column_stack([TABLE.a_par[sites], TABLE.a_perp[sites]])


def mass_on_truth(particles, sites):
    """Posterior weight of the particle that is the true bath, or 0."""
    target = truth_couplings(sites)
    for i in range(particles.n_particles):
        k = int(particles.k[i])
        s = particles.site_idx[i, :k]
        pairs = np.column_stack([TABLE.a_par[s] + particles.dA_par[i, :k],
                                 TABLE.a_perp[s] + particles.dA_perp[i, :k]])
        if _same_bath(pairs, target, 0.1):
            return float(particles.weight[i])
    return 0.0


def updated_mass(particles, measured, followup, sites):
    """Mass on the truth after reweighting the first posterior's particles by
    the follow-up likelihood -- the design's effect, with no refit."""
    exp = followup.experiments[0]
    bare = ExperimentSet([Experiment(tau=exp.tau, n_pulses=exp.n_pulses,
                                     b_z=exp.b_z)])
    e = next(i for i, m in enumerate(measured.experiments)
             if m.n_pulses == exp.n_pulses)
    P = ParticleSet(particles.site_idx, particles.k, particles.weight,
                    particles.dA_par, particles.dA_perp, particles.lam[:, [e]],
                    particles.n_stretch[:, [e]], particles.sigma[:, [e]],
                    particles.n_sites, particles.k_max).predictions(
                        bare, TABLE, MODEL)
    ll = -0.5 * np.sum(exp.weight * (exp.data - P) ** 2, axis=1) / NOISE ** 2
    log_w = np.log(particles.weight) + ll
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    updated = ParticleSet(particles.site_idx, particles.k, w, particles.dA_par,
                          particles.dA_perp, particles.lam, particles.n_stretch,
                          particles.sigma, particles.n_sites, particles.k_max)
    return mass_on_truth(updated, sites)


def one_seed(seed):
    rng = np.random.default_rng(seed)
    sites = np.sort(rng.choice(len(TABLE), K_TRUE, replace=False))
    truth = state(sites, sigma=NOISE)
    first = simulate_dataset(
        truth, ExperimentSet([Experiment(tau=grid(FIRST_POINTS),
                                         n_pulses=N_PULSES, b_z=B_Z)]),
        TABLE, MODEL, sigma=NOISE, rng=np.random.default_rng(seed + 9000))

    pooled = fit(first, root_seed=seed)
    particles = ParticleSet.from_trace(pooled, TABLE, stride=PARTICLE_STRIDE)
    before = summarize(pooled, first, TABLE, MODEL, reference=sites, burn=0,
                       stride=50, noise=NOISE, tol=0.1)
    row = {"seed": seed, "eff": particles.effective_size,
           "R0": float(np.mean(before.R_i)),
           "m0": mass_on_truth(particles, sites)}

    for kind in ("adaptive", "uniform", "anti"):
        proposal = design(kind, particles, first, np.random.default_rng(seed + 1))
        followup = simulate_dataset(truth, ExperimentSet([proposal]), TABLE,
                                    MODEL, sigma=NOISE,
                                    rng=np.random.default_rng(seed + 5000))
        combined = ExperimentSet(first.experiments + followup.experiments)
        refit = fit(combined, root_seed=seed + 100)
        after = summarize(refit, combined, TABLE, MODEL, reference=sites,
                          burn=0, stride=50, noise=NOISE, tol=0.1)
        row[f"R_{kind}"] = float(np.mean(after.R_i))
        row[f"u_{kind}"] = updated_mass(particles, first, followup, sites)
        row[f"n_{kind}"] = proposal.tau.size
    return row


def degenerate_control(seed=SEEDS[0]):
    """A posterior collapsed onto the truth: the designer must decline."""
    sites = np.sort(np.random.default_rng(seed).choice(len(TABLE), K_TRUE,
                                                       replace=False))
    measured = ExperimentSet([Experiment(tau=grid(FIRST_POINTS),
                                         n_pulses=N_PULSES, b_z=B_Z)])
    one = state(sites, sigma=NOISE)
    collapsed = ParticleSet(one.site_idx, one.k, [1.0], one.dA_par, one.dA_perp,
                            one.lam, one.n_stretch, one.sigma, len(TABLE), K_MAX)
    try:
        design("adaptive", collapsed, measured, np.random.default_rng(0))
    except NothingToLearn:
        declined = True
    else:
        declined = False
    return declined, uniform_design().tau.size


if __name__ == "__main__":
    declined, n_uniform = degenerate_control()
    print(f"degenerate control: designer declined={declined}, "
          f"uniform still built with {n_uniform} points\n", flush=True)

    cols = ("seed", "eff", "R0", "m0", "R_adaptive", "R_uniform", "R_anti",
            "u_adaptive", "u_uniform", "u_anti", "n_adaptive")
    print("  ".join(f"{c:>10}" for c in cols), flush=True)
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        row = one_seed(seed)
        rows.append(row)
        print("  ".join(f"{row[c]:>10.3f}" if isinstance(row[c], float)
                        else f"{row[c]:>10}" for c in cols),
              f"  {time.time() - t0:.0f}s", flush=True)

    def paired(a, b, metric):
        d = np.array([r[f"{metric}_{a}"] - r[f"{metric}_{b}"] for r in rows])
        return (f"{a} - {b} on {metric}: mean {d.mean():+.3f}, min {d.min():+.3f}, "
                f"positive on {int(np.sum(d > 0))}/{d.size}")

    print()
    for metric in ("R", "u"):
        print(paired("adaptive", "uniform", metric))
        print(paired("uniform", "anti", metric))
