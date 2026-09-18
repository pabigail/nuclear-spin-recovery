"""How do fit and detection depend on the likelihood's sigma, for a bare
single-block discrete RWMH?  Measured, not assumed."""
import sys, time
import numpy as np
sys.path.insert(0, "src")
from nuclear_spin_recovery import (
    AnalyticCCE1, DiscreteLatticeWalk, Experiment, ExperimentSet, GaussianL2,
    NeighborIndex, ParameterBlock, RWMH, SiteTable, State, StretchedExponential,
    Target, Trace, simulate_coherence, simulate_dataset)

TABLE = SiteTable.from_ivady_file("nv-2.txt", strong_thresh=750.0, weak_thresh=5.0)
MODEL = AnalyticCCE1(StretchedExponential())
NEIGH, LOCAL = NeighborIndex(TABLE.positions, 5.0), NeighborIndex(TABLE.positions, 3.0)
TAU = np.linspace(0.0, 8e-3, 250, endpoint=False) + 8e-3/250
DATA_NOISE, LAM, K_MAX = 0.002, 3e-3, 16
STEPS, BURN = 10_000, 4_000
BANDS = [(5,25), (25,100), (100,750)]
MAG = np.hypot(TABLE.a_par, TABLE.a_perp)

def strat(rng, per=4):
    out = []
    for lo, hi in BANDS:
        pool = np.flatnonzero((MAG >= lo) & (MAG < hi))
        out.extend(rng.choice(pool, size=per, replace=False))
    return np.array(sorted(set(out)))

def st(sites, sigma):
    return State.from_sites(np.sort(np.asarray(sites,int)), n_sites=len(TABLE), n_exp=1,
        lam=np.array([[LAM]]), n_stretch=np.array([[1.]]),
        sigma=np.array([[sigma]]), k_max=K_MAX)

def perturb(sites, rng):
    out, taken = [], set()
    for s in sites:
        c = [x for x in LOCAL.neighbors(s) if x not in taken]
        p = int(rng.choice(c)) if c else int(s); out.append(p); taken.add(p)
    return np.array(out)

def pr(sites): return set(zip(np.round(TABLE.a_par[sites],4), np.round(TABLE.a_perp[sites],4)))

t0 = time.time()
print(f"{'sigma_lik':>10} {'resid/noise':>12} {'accept':>8} "
      + " ".join(f"{'R '+str(lo)+'-'+str(hi):>11}" for lo,hi in BANDS), flush=True)
for sigma in (0.002, 0.006, 0.02, 0.06, 0.316):
    res_all, R_all, mag_all, acc_all = [], [], [], []
    for trial in range(3):
        rng = np.random.default_rng(200+trial)
        truth_sites = strat(rng)
        data = simulate_dataset(st(truth_sites, DATA_NOISE), ExperimentSet(
            [Experiment(tau=TAU, n_pulses=16, b_z=311.)]), TABLE, MODEL,
            sigma=DATA_NOISE, rng=np.random.default_rng(trial+900))
        target = Target(data, MODEL, GaussianL2(), TABLE)
        tr = Trace(n_sites=len(TABLE), k_max=K_MAX, n_exp=1)
        RWMH(ParameterBlock("sites"), DiscreteLatticeWalk(NEIGH)).run(
            st(perturb(truth_sites, rng), sigma), target,
            np.random.default_rng(trial), n_steps=STEPS, trace=tr)
        post = tr.discard_burn_in(BURN)
        acc_all.append(np.mean(np.any(np.diff(post.site_idx, axis=0) != 0, axis=1)))
        samples = [pr(post.site_idx[j,:int(post.k[j])]) for j in range(len(post))]
        ref = list(zip(TABLE.a_par[truth_sites], TABLE.a_perp[truth_sites]))
        R_all.append(np.array([np.mean([any(abs(a-c)<=.1 and abs(b-d)<=.1 for c,d in S)
                                        for S in samples]) for a,b in ref]))
        mag_all.append(np.hypot(*np.array(ref).T))
        obs = data.data_all
        res_all.append([np.sqrt(np.mean((obs - simulate_coherence(
            st(post.site_idx[j,:int(post.k[j])], sigma), data, TABLE, MODEL))**2))
            for j in range(0, len(post), 200)])
    R, mag = np.concatenate(R_all), np.concatenate(mag_all)
    r = np.concatenate(res_all)/DATA_NOISE
    cells = [f"{R[(mag>=lo)&(mag<hi)].mean():11.3f}" for lo,hi in BANDS]
    print(f"{sigma:10.3f} {np.median(r):12.2f} {np.mean(acc_all):8.1%} "
          + " ".join(cells), flush=True)
print(f"\n({time.time()-t0:.0f}s)")
