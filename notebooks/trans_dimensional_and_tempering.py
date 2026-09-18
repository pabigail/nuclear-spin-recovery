# %% [markdown]
# # Inferring bath size, escaping local minima, and relaxing the DFT constraint
#
# The previous two tutorials built a coherence signal from a known bath
# (`spin_bath_and_coherence`) and ran a fixed-dimension sampler over it
# (`mcmc_algorithms`). Both assumed the number of spins $k$ was known.
#
# Phase 3 removes that assumption and adds the machinery for a landscape that is
# genuinely multimodal:
#
# | object | what it adds |
# |---|---|
# | `BirthDeathKernel`, `RJMCMC` | trans-dimensional moves — $k$ is inferred, not assumed |
# | `ParallelTempering`, `geometric_ladder` | a $\beta$ ladder, to cross barriers a cold chain cannot |
# | `Step`, `Schedule`, `HybridDriver` | cycles several algorithms into one chain and one trace |
# | `GaussianOffset` | relaxes the hard *ab initio* constraint on each hyperfine coupling |
#
# This notebook is organised around three questions:
#
# 1. **§1–§3** What does a complete workflow look like with a *single* algorithm,
#    and how do you tell whether it worked?
# 2. **§4–§6** What do the trans-dimensional and tempered moves buy on top of it?
# 3. **§7** What happens when the DFT couplings themselves are not quite right?
#
# Throughout, two things get visualised: where the spins actually **walk in real
# space** over the diamond lattice, and how the **error falls with step count**.
#
# The model is specified in `docs/model-specification.md`; what each measurement
# can and cannot demonstrate is in `docs/test-plan.md`.

# %%
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nuclear_spin_recovery import (
    RJMCMC,
    RWMH,
    AnalyticCCE1,
    BirthDeathKernel,
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
    geometric_ladder,
    simulate_coherence,
    simulate_dataset,
)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})

# %% [markdown]
# ## 1. A complete workflow with a single algorithm
#
# Everything phase 3 adds is optional. Start with the smallest thing that is
# still a real inference: one lattice, one experiment, one algorithm.
#
# ### 1.1 The candidate sites
#
# Candidate nuclear positions come from the DFT hyperfine table for NV in
# diamond. Two thresholds filter it, as in the first tutorial.

# %%
full_table = SiteTable.from_ivady_file(REPO / "nv-2.txt", strong_thresh=750.0,
                                       weak_thresh=5.0)
print(f"{len(full_table)} sites survive the 5–750 kHz filter")

# %% [markdown]
# For this tutorial the candidate pool is narrowed further, to sites a CPMG-16
# experiment at 311 G can actually resolve.
#
# This is not cosmetic. Below roughly 25 kHz a spin changes the signal by less
# than the noise, so adding a spurious one barely moves the likelihood and is
# accepted about half the time. On the full table that random walk carries $k$
# far above the truth — the posterior mode is 17 when the truth is 8 — which
# measures *identifiability*, not the sampler. Restricting the pool makes a
# spurious spin cost something. The numbers behind the cutoff are in
# `docs/test-plan.md` §5.6.

# %%
magnitude_all = np.hypot(full_table.a_par, full_table.a_perp)
keep = magnitude_all >= 100.0
table = SiteTable(
    distance=full_table.distance[keep], positions=full_table.positions[keep],
    a_par=full_table.a_par[keep], a_perp=full_table.a_perp[keep],
    isotope=full_table.isotope[keep], gyro=full_table.gyro[keep],
)
print(f"{len(table)} sites with coupling magnitude >= 100 kHz")

# %% [markdown]
# ### 1.2 Ground truth, an experiment, and data
#
# Six spins drawn from that pool, a CPMG-16 experiment at 311 G sampled at 250
# values of $\tau$, and data with 0.002 of Gaussian noise.

# %%
N_PULSES, B_Z, LAM = 16, 311.0, 3e-3
DATA_NOISE = 0.002
LIK_SIGMA = 0.02      # calibrated for the sampler, not the data noise — see §3.3
TAU = np.linspace(0.0, 8e-3, 250, endpoint=False) + 8e-3 / 250
K_MAX = 32

model = AnalyticCCE1(StretchedExponential())


def make_state(sites, sigma=LIK_SIGMA, k_max=K_MAX, tbl=None):
    """A single-replica State on ``tbl`` holding the given sites."""
    tbl = table if tbl is None else tbl
    return State.from_sites(
        np.sort(np.asarray(list(sites), dtype=int)),
        n_sites=len(tbl), n_exp=1,
        lam=np.array([[LAM]]), n_stretch=np.array([[1.0]]),
        sigma=np.array([[sigma]]), k_max=k_max,
    )


true_sites = np.sort(np.random.default_rng(7).choice(len(table), 6, replace=False))
truth = make_state(true_sites, sigma=DATA_NOISE)

blank = ExperimentSet([Experiment(tau=TAU, n_pulses=N_PULSES, b_z=B_Z)])
data = simulate_dataset(truth, blank, table, model, sigma=DATA_NOISE,
                        rng=np.random.default_rng(11))
obs = data.data_all

print(f"true sites : {true_sites.tolist()}")
print("couplings  : " + ", ".join(
    f"({table.a_par[s]:.0f}, {table.a_perp[s]:.0f})" for s in true_sites) + "  kHz")
print(f"data       : {data.n_points} points, one experiment")

# %%
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(TAU * 1e3, obs, lw=0.8, color="0.35", label="simulated data")
ax.plot(TAU * 1e3, simulate_coherence(truth, data, table, model), lw=1.4,
        color="crimson", label="noiseless truth")
ax.set_xlabel(r"$\tau$ (µs)")
ax.set_ylabel("coherence")
ax.set_title(f"CPMG-{N_PULSES} at {B_Z:.0f} G, six spins")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()

# %% [markdown]
# ### 1.3 The `Target`
#
# `Target` bundles the data, forward model, likelihood and site table behind a
# single `log_prob(state, beta)`. Every algorithm depends on this one object
# rather than on the four pieces separately — which is what lets the same
# algorithm run at any rung of a temperature ladder without knowing it.

# %%
target = Target(data, model, GaussianL2(), table)
print(f"log L at truth  : {target.log_prob(truth)[0]:.1f}")
print(f"tempered, β=0.25: {target.log_prob(truth, beta=0.25)[0]:.1f}")

# %% [markdown]
# ### 1.4 The neighbour index and the proposal
#
# A spin moves to another site within `radius` of its current one. The index is
# precomputed once; the proposal reports the asymmetry the occupancy constraint
# creates, since the number of free neighbours differs between the current and
# proposed sites.

# %%
WALK_RADIUS = 6.0
neighbors = NeighborIndex(table.positions, radius=WALK_RADIUS)
walk = DiscreteLatticeWalk(neighbors)

counts = [neighbors.neighbors(s).size for s in range(len(table))]
print(f"radius {WALK_RADIUS} Å: {np.mean(counts):.1f} neighbours per site "
      f"(min {min(counts)}, max {max(counts)})")

# %% [markdown]
# ### 1.5 One algorithm, one chain
#
# `RWMH` over the `sites` block, started from a random configuration of the
# right size. This is the whole workflow: everything above is setup.

# %%
start_sites = np.sort(np.random.default_rng(3).choice(len(table), 6, replace=False))
sampler = RWMH(ParameterBlock("sites"), walk)

t0 = time.time()
rwmh_trace = Trace(n_sites=len(table), k_max=K_MAX, n_exp=1)
sampler.run(make_state(start_sites), target, np.random.default_rng(5),
            n_steps=6000, trace=rwmh_trace)
print(f"6000 steps in {time.time() - t0:.1f} s, {len(rwmh_trace)} recorded")

# %% [markdown]
# ### 1.6 Judging it — and the one thing never to judge it by
#
# The recovery problem is ill-posed: many configurations reproduce the same
# signal within noise. So a single "final answer" is meaningless, and two
# separate questions get asked instead (spec §9.1):
#
# - **Criterion A — does the forward model reproduce the data?** RMS residual
#   against the noise level. This is the *only* criterion available on
#   experimental data.
# - **Criterion B — does the posterior contain the simulated spins?** Detection
#   rate over posterior samples. Available in simulation only.
#
# Criterion B is measured **over the posterior, never over the final state**, and
# matched on **couplings, never on site index** — §2 shows why that second point
# is not a technicality.

# %%
def residual(sites, dA_par=None, dA_perp=None, tbl=None):
    """RMS residual of a configuration, in units of the data noise."""
    tbl = table if tbl is None else tbl
    st = make_state(sites, tbl=tbl)
    k = int(st.k[0])
    if dA_par is not None:
        st.dA_par[0, :k] = dA_par[:k]
        st.dA_perp[0, :k] = dA_perp[:k]
    pred = simulate_coherence(st, data, tbl, model)
    return float(np.sqrt(np.mean((obs - pred) ** 2)) / DATA_NOISE)


def residual_curve(trace, stride=25, tbl=None):
    """Residual at every ``stride``-th recorded step. (steps, residuals)"""
    steps = np.arange(0, len(trace), stride)
    vals = [residual(trace.site_idx[j, : int(trace.k[j])],
                     trace.dA_par[j], trace.dA_perp[j], tbl=tbl) for j in steps]
    return steps, np.array(vals)


def detection_rates(trace, burn, reference=true_sites, stride=10, tol=0.1, tbl=None):
    """Fraction of posterior samples containing each reference spin's couplings.

    Matched on couplings, never on site index: symmetry-equivalent sites are
    physically indistinguishable, so an index match would score a correct answer
    as a miss.
    """
    tbl = table if tbl is None else tbl
    post = trace.discard_burn_in(burn)
    samples = [
        set(zip(np.round(tbl.a_par[post.site_idx[j, : int(post.k[j])]], 4),
                np.round(tbl.a_perp[post.site_idx[j, : int(post.k[j])]], 4)))
        for j in range(0, len(post), stride)
    ]
    out = []
    for s in reference:
        a, b = tbl.a_par[s], tbl.a_perp[s]
        out.append(np.mean([any(abs(a - c) <= tol and abs(b - d) <= tol
                                for c, d in S) for S in samples]))
    return np.array(out)


steps, rwmh_resid = residual_curve(rwmh_trace)
rwmh_R = detection_rates(rwmh_trace, burn=2000)

print(f"criterion A  best residual : {rwmh_resid.min():6.2f} σ   (1 σ = fits within noise)")
print(f"criterion A  final residual: {rwmh_resid[-1]:6.2f} σ")
print("criterion B  detection rate per true spin:")
for s, r in zip(true_sites, rwmh_R):
    mag = np.hypot(table.a_par[s], table.a_perp[s])
    print(f"    site {s:3d}  |A| = {mag:6.1f} kHz   R = {r:.2f}")

# %% [markdown]
# A residual of many $\sigma$ means the chain has **not** fit the data. Hold that
# thought — §3 shows what it is doing instead, and §5 fixes it.

# %% [markdown]
# ## 2. Watching the walk in real space
#
# The state is a list of site indices, but each index is a real position in the
# diamond lattice. Plotting the walk there is the most direct way to see what the
# sampler is doing.
#
# Positions are relative to the NV centre, with $z$ along the NV axis. The
# natural projection is therefore $\rho = \sqrt{x^2 + y^2}$ (distance from the
# axis) against $z$ (distance along it), since the dipolar coupling depends on
# exactly that angle.

# %%
pos = table.positions
rho = np.hypot(pos[:, 0], pos[:, 1])
z = pos[:, 2]


def lattice_axes(figsize=(11, 4.4)):
    """Two projections of the candidate lattice, with the true sites marked."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=figsize)
    for ax, xs, ys, xl, yl in (
        (axl, rho, z, r"$\rho = \sqrt{x^2+y^2}$  (Å)", r"$z$ along NV axis  (Å)"),
        (axr, pos[:, 0], pos[:, 1], r"$x$  (Å)", r"$y$  (Å)"),
    ):
        ax.scatter(xs, ys, s=14, color="0.82", edgecolor="none", zorder=1)
        ax.scatter(xs[true_sites], ys[true_sites], s=150, marker="*",
                   color="crimson", zorder=5, label="true sites")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal", adjustable="datalim")
    return fig, (axl, axr)


fig, (axl, axr) = lattice_axes()
axl.set_title(f"{len(table)} candidate sites, NV-axis projection")
axr.set_title("viewed down the NV axis")
axl.legend(loc="upper right", fontsize=8)
plt.tight_layout()

# %% [markdown]
# ### 2.1 The trajectory of one spin
#
# `RWMH` moves one spin per step, chosen uniformly from the $k$ slots, so slot 0
# is a single physical spin followed through the whole chain. (This is only true
# while $k$ is fixed — `RJMCMC` fills a vacated slot by swapping in the last
# spin, so slot identity does not survive a death. §4.)

# %%
SLOT = 0
path = rwmh_trace.site_idx[:, SLOT]
hops = np.flatnonzero(np.diff(path)) + 1
visited = np.concatenate([[path[0]], path[hops]])
arrive = np.concatenate([[0], hops])

print(f"slot {SLOT} occupied {len(np.unique(path))} distinct sites over "
      f"{len(path)} steps, {len(hops)} accepted hops")

fig, (axl, axr) = lattice_axes()
for ax, xs, ys in ((axl, rho, z), (axr, pos[:, 0], pos[:, 1])):
    ax.plot(xs[visited], ys[visited], "-", color="steelblue", lw=1.2,
            alpha=0.9, zorder=3)
    sc = ax.scatter(xs[visited], ys[visited], c=arrive, cmap="viridis",
                    s=55, zorder=4, edgecolor="k", linewidth=0.4)
    ax.scatter(xs[visited[0]], ys[visited[0]], s=170, marker="s",
               facecolor="none", edgecolor="darkorange", linewidth=2, zorder=6)
axl.set_title(f"path of spin in slot {SLOT} (square = start)")
axr.set_title("same path, down the NV axis")
axl.legend(loc="upper right", fontsize=8)
fig.colorbar(sc, ax=axr, label="step of arrival")
plt.tight_layout()

# %% [markdown]
# ### 2.2 Why site index is the wrong thing to score
#
# The NV centre has $C_{3v}$ symmetry, so lattice sites come in orbits that are
# spatially distinct but have **identical** hyperfine couplings. No coherence
# measurement can tell them apart — they produce the same signal exactly.
#
# `SiteTable.symmetry_groups` labels them.

# %%
groups = table.symmetry_groups(tol=0.1)
sizes = Counter(groups.tolist())
print(f"{len(table)} sites collapse into {len(sizes)} distinguishable groups")
print(f"most common orbit size: {Counter(sizes.values()).most_common(3)}")

example = groups[true_sites[0]]
members = np.flatnonzero(groups == example)
print(f"\ngroup {example} has {members.size} members, all with the same couplings:")
for s in members:
    p = pos[s]
    print(f"    site {s:3d}  at ({p[0]:6.2f}, {p[1]:6.2f}, {p[2]:6.2f}) Å   "
          f"A_par {table.a_par[s]:8.2f}  A_perp {table.a_perp[s]:7.2f} kHz")

# %%
fig, (axl, axr) = lattice_axes()
for ax, xs, ys in ((axl, rho, z), (axr, pos[:, 0], pos[:, 1])):
    ax.scatter(xs[members], ys[members], s=110, facecolor="none",
               edgecolor="darkviolet", linewidth=2, zorder=6)
axl.set_title(f"the {members.size} members of orbit {example}")
axr.set_title("spatially distinct, physically identical")
plt.tight_layout()

# %% [markdown]
# This also explains something visible in every left-hand panel above. An orbit
# shares one $\rho$ and one $z$ — it is generated by rotation about the NV axis —
# so the $\rho$–$z$ projection collapses each orbit to a single point. That
# projection is, in effect, a picture of what the experiment can distinguish,
# while the $x$–$y$ view shows what it cannot.

# %%
proj = np.column_stack([np.round(rho, 3), np.round(z, 3)])
print(f"{len(table)} sites   ->   {len(np.unique(proj, axis=0))} distinct (ρ, z) points"
      f"   ->   {len(sizes)} coupling orbits")

# %% [markdown]
# So the right way to read a final configuration is by orbit, not by index.

# %%
final_sites = rwmh_trace.site_idx[-1, : int(rwmh_trace.k[-1])]
print(f"true  orbits: {sorted(groups[true_sites].tolist())}")
print(f"final orbits: {sorted(groups[final_sites].tolist())}")
print(f"true  sites : {sorted(true_sites.tolist())}")
print(f"final sites : {sorted(final_sites.tolist())}")
hit = len(set(groups[final_sites]) & set(groups[true_sites]))
print(f"\n{hit} of {len(true_sites)} orbits recovered "
      f"(index overlap alone: {len(set(final_sites) & set(true_sites.tolist()))})")

# %% [markdown]
# ## 3. Error with respect to step count
#
# The residual curve says what the chain is doing far more directly than the
# configuration does.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))

ax1.plot(steps, rwmh_resid, lw=1.3, color="steelblue")
ax1.axhline(1.0, color="crimson", ls="--", lw=1,
            label="1 σ — fits within noise")
ax1.set_yscale("log")
ax1.set_xlabel("step")
ax1.set_ylabel(r"RMS residual  ($\sigma$)")
ax1.set_title("criterion A versus step count")
ax1.legend(fontsize=8)

ax2.plot(rwmh_trace.log_prob, lw=0.7, color="0.4")
ax2.set_xlabel("step")
ax2.set_ylabel(r"$\log L$")
ax2.set_title("log-likelihood, every step")
plt.tight_layout()

# %% [markdown]
# The curve drops fast and then flatlines well above 1 σ. The chain has found a
# configuration it cannot improve on by moving one spin at a time, and it is not
# the truth: it is stuck in a local mode.
#
# That is not a bug in `RWMH` — it is the honest behaviour of a single-spin walk
# on a rugged landscape, and it is precisely what the tempering ladder in §5
# exists to fix.

# %%
print(f"residual after   500 steps: {rwmh_resid[steps <= 500][-1]:6.2f} σ")
print(f"residual after  6000 steps: {rwmh_resid[-1]:6.2f} σ")
print(f"improvement over the last 5500 steps: "
      f"{rwmh_resid[steps <= 500][-1] - rwmh_resid[-1]:.2f} σ")

# %% [markdown]
# ### 3.1 A note on the two sigmas
#
# `DATA_NOISE = 0.002` is the noise actually added to the data. `LIK_SIGMA = 0.02`
# is the width the *likelihood* uses, and it is deliberately ten times looser.
#
# At the true noise level the posterior is so sharp that almost every proposal is
# rejected and the chain never moves. The calibration in `docs/test-plan.md` §5.2
# scans this: at σ = 0.002 the chain stalls, at σ = 0.316 it accepts 79% and
# wanders at 31 σ residual, and σ = 0.02 is optimal on residual, detection rate
# and acceptance simultaneously. The papers' σ² = 0.1 does not transfer to a bare
# sampler.

# %% [markdown]
# ## 4. `RJMCMC` — inferring the number of spins
#
# Everything so far held $k$ fixed at the true value, which is not knowable
# experimentally. `BirthDeathKernel` proposes $k \to k \pm 1$; `RJMCMC` accepts
# or rejects.

# %%
kernel = BirthDeathKernel(k_max=K_MAX, birth_prob=0.5)
rj = RJMCMC(ParameterBlock("sites"), kernel)

for k, n_free in ((6, len(table) - 6), (20, len(table) - 20)):
    print(f"k = {k:2d}: log ratio  birth {kernel.log_ratio(k, 'birth', n_free):+.3f}   "
          f"death {kernel.log_ratio(k, 'death', n_free):+.3f}")

# %% [markdown]
# Both ratios are zero, and that is the correct answer — worth dwelling on,
# because getting it wrong is subtle and nearly invisible.
#
# The **proposal** ratio for a birth is $\frac{p_d/(k+1)}{p_b/n_{\text{free}}}$:
# forward draws a site uniformly from the unoccupied ones, backward removes one
# of the $k+1$ spins uniformly. The **prior** ratio carries a combinatorial
# factor, because a prior uniform over *configurations* is not uniform over $k$ —
# there are $\binom{n}{k}$ configurations of size $k$. Writing
# $p(\text{config}) = p(k)/\binom{n}{k}$,
#
# $$\binom{n}{k} \Big/ \binom{n}{k+1} = \frac{k+1}{n-k} = \frac{k+1}{n_{\text{free}}}$$
#
# which is exactly the inverse of the proposal ratio. The two cancel, leaving
# $\log(p_d/p_b) + \Delta \log p(k)$ — zero for a symmetric kernel and a uniform
# prior on $k$.
#
# Dropping the combinatorial term leaves $\log(n_{\text{free}}/(k+1))$, about
# **+5.9** on the full table: a factor of 365 favouring every birth whatever the
# data says. The dimension then runs to `k_max` regardless of evidence. This bug
# was in the code, and the overfitting guard in the T3 rung is what caught it.

# %% [markdown]
# ### 4.1 Approaching the truth from below and from above

# %%
rj_traces = {}
for k0 in (1, 14):
    st = make_state(np.random.default_rng(31).choice(len(table), k0, replace=False))
    tr = Trace(n_sites=len(table), k_max=K_MAX, n_exp=1)
    rj.run(st, target, np.random.default_rng(17), n_steps=4000, trace=tr)
    rj_traces[k0] = tr
    post = tr.discard_burn_in(2000)
    print(f"start k = {k0:2d}  ->  posterior mode {np.bincount(post.k).argmax()}, "
          f"mean {post.k.mean():.2f}   (truth {len(true_sites)})")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
for k0, colour in ((1, "seagreen"), (14, "darkorange")):
    ax1.plot(rj_traces[k0].k, lw=0.8, color=colour, label=f"start $k$ = {k0}")
    post = rj_traces[k0].discard_burn_in(2000)
    ax2.hist(post.k, bins=np.arange(-0.5, 20.5), alpha=0.55, color=colour,
             label=f"start $k$ = {k0}", density=True)
ax1.axhline(len(true_sites), color="crimson", ls="--", lw=1, label="truth")
ax2.axvline(len(true_sites), color="crimson", ls="--", lw=1, label="truth")
ax1.set_xlabel("step"); ax1.set_ylabel("$k$"); ax1.set_title("dimension versus step count")
ax2.set_xlabel("$k$"); ax2.set_ylabel("posterior density"); ax2.set_title("posterior on $k$, after burn-in")
ax1.legend(fontsize=8); ax2.legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# The two chains do not agree, and that disagreement is a measured property of
# this problem rather than a burn-in artefact: `docs/test-plan.md` §5.6 records
# the same split holding at 8,000, 16,000 and 30,000 steps.
#
# **Birth–death moves alone do not mix across dimension.** A chain that arrives
# at $k$ from above has a different configuration than one arriving from below,
# and neither can reach the other by adding or removing one spin at a time. The
# practical consequence: report $k$ from several starts, or pair `RJMCMC` with
# moves that can cross the barrier — which is the next section.

# %% [markdown]
# ## 5. `ParallelTempering` — crossing the barrier
#
# $J$ replicas run at $\beta_j = 2^{-j}$, zero-indexed so replica 0 is the cold
# chain sampling the true posterior. Hot replicas accept almost anything and so
# wander freely; swaps let a configuration found at high temperature descend to
# the cold chain. Only the cold chain is ever recorded.

# %%
print("ladder:", [f"{b:.4f}" for b in geometric_ladder(6)])

inner = Schedule([Step(RWMH(ParameterBlock("sites"), walk), 1)])
pt = ParallelTempering(inner, n_replicas=6)

ladder_state = make_state(start_sites).expand_replicas(pt.n_replicas)
print(f"expand_replicas -> site_idx {ladder_state.site_idx.shape}  "
      f"({ladder_state.n_replicas} rungs)")
print(f"collapse_to_cold -> {ladder_state.collapse_to_cold().n_replicas} replica")

# %% [markdown]
# ### 5.1 How many rungs?
#
# The ladder only works if adjacent rungs actually exchange. Too few rungs and
# the temperature gap is too wide for a swap ever to be accepted, and the hot
# chains are pure wasted computation. This was measured directly: on a
# four-rung geometric ladder the swap is accepted **zero** times, and on six
# rungs at a rate of 0.058 (`docs/test-plan.md` §5.6).

# %%
pt_traces, pt_curves = {}, {}
for J in (4, 6, 8):
    tr = Trace(n_sites=len(table), k_max=K_MAX, n_exp=1)
    t0 = time.time()
    ParallelTempering(inner, n_replicas=J).run(
        make_state(start_sites), target, np.random.default_rng(5),
        n_steps=1500, trace=tr)
    pt_traces[J] = tr
    pt_curves[J] = residual_curve(tr, stride=10)
    print(f"J = {J}:  {time.time() - t0:5.1f} s   "
          f"best residual {pt_curves[J][1].min():6.2f} σ")

print(f"\nRWMH alone, 6000 steps: {rwmh_resid.min():6.2f} σ")

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(steps, rwmh_resid, lw=1.4, color="0.45", label="RWMH alone")
for J, colour in ((4, "#8ecae6"), (6, "#219ebc"), (8, "#023047")):
    s, r = pt_curves[J]
    ax.plot(s, r, lw=1.3, color=colour, label=f"tempered, $J$ = {J}")
ax.axhline(1.0, color="crimson", ls="--", lw=1, label="1 σ")
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel(r"RMS residual  ($\sigma$)")
ax.set_title("error versus step count: the ladder crosses what the walk cannot")
ax.legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# ### 5.2 The cold chain, in real space
#
# The clearest signature of tempering is spatial. A cold single-spin walk crawls
# between neighbouring sites. A tempered cold chain **teleports**: a
# configuration assembled at high temperature arrives through a swap, so
# successive recorded states can sit far apart on the lattice.

# %%
best_J = min(pt_curves, key=lambda J: pt_curves[J][1].min())
pt_path = pt_traces[best_J].site_idx[:, SLOT]
pt_hops = np.flatnonzero(np.diff(pt_path)) + 1
pt_visited = np.concatenate([[pt_path[0]], pt_path[pt_hops]])


def hop_distances(path):
    idx = np.flatnonzero(np.diff(path)) + 1
    a, b = path[idx - 1], path[idx]
    return np.linalg.norm(pos[a] - pos[b], axis=1)


d_rwmh, d_pt = hop_distances(path), hop_distances(pt_path)
print(f"RWMH        : {len(d_rwmh):4d} hops, median {np.median(d_rwmh):5.2f} Å, "
      f"max {d_rwmh.max():5.2f} Å")
print(f"tempered J={best_J}: {len(d_pt):4d} hops, median {np.median(d_pt):5.2f} Å, "
      f"max {d_pt.max():5.2f} Å")
print(f"\nproposal radius is {WALK_RADIUS} Å — anything beyond that arrived by swap")
print(f"hops exceeding it: RWMH {int((d_rwmh > WALK_RADIUS).sum())}, "
      f"tempered {int((d_pt > WALK_RADIUS).sum())}")

# %%
fig, (axl, axr) = lattice_axes()
for ax, xs, ys in ((axl, rho, z), (axr, pos[:, 0], pos[:, 1])):
    ax.plot(xs[visited], ys[visited], "-o", color="0.5", lw=1.1, ms=4,
            alpha=0.8, zorder=3, label="RWMH alone")
    ax.plot(xs[pt_visited], ys[pt_visited], "-o", color="#023047", lw=1.1,
            ms=4, alpha=0.85, zorder=4, label=f"tempered, $J$ = {best_J}")
axl.set_title(f"slot {SLOT}: crawling versus teleporting")
axr.set_title("down the NV axis")
axl.legend(loc="upper right", fontsize=8)
plt.tight_layout()

# %%
final_pt = pt_traces[best_J].site_idx[-1, : int(pt_traces[best_J].k[-1])]
print(f"true      orbits: {sorted(groups[true_sites].tolist())}")
print(f"tempered  orbits: {sorted(groups[final_pt].tolist())}")
print(f"RWMH      orbits: {sorted(groups[final_sites].tolist())}")

# %% [markdown]
# ### 5.3 One caveat, recorded rather than smoothed over
#
# The benefit is real but **seed-dependent**. Pooled over the calibration runs in
# `docs/test-plan.md` §5.6 the best residual improves 2.72 σ → 1.76 σ, but that
# ranges from 4.80 → 2.00 on one seed to no improvement at all on another. A
# single comparison — including the one plotted above — is an illustration, not
# evidence. This is why the T4 rung asserts on pooled runs and not per-seed.

# %% [markdown]
# ## 6. `HybridDriver` — composing them into one chain
#
# `Step(algorithm, n_steps)` names a block; `Schedule` orders them; `HybridDriver`
# cycles until the budget is spent. Each block leaves the target invariant, so
# the systematic-scan composition does too (spec §8.5).

# %%
schedule = Schedule([
    Step(RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=K_MAX)), 150),
    Step(ParallelTempering(inner, n_replicas=8), 400),
    Step(RWMH(ParameterBlock("sites"), walk), 100),
])
print(f"{len(schedule)} blocks, {schedule.steps_per_cycle} steps per cycle")

driver = HybridDriver(schedule)
hybrid_trace = Trace(n_sites=len(table), k_max=K_MAX, n_exp=1)

t0 = time.time()
driver.run(make_state(np.random.default_rng(31).choice(len(table), 3, replace=False)),
           target, np.random.default_rng(23), n_total=1300, trace=hybrid_trace)
print(f"{len(hybrid_trace)} steps in {time.time() - t0:.1f} s")
print(Counter(hybrid_trace.algorithm.tolist()))

# %% [markdown]
# Every recorded step is labelled with the sub-algorithm that produced it, so a
# single trace can be read block by block.

# %%
hyb_steps, hyb_resid = residual_curve(hybrid_trace, stride=10)
labels = hybrid_trace.algorithm[hyb_steps]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
colours = {"rjmcmc:sites": "seagreen", "pt:sites": "#023047", "rwmh:sites": "0.45"}
for name, colour in colours.items():
    sel = labels == name
    ax1.scatter(hyb_steps[sel], hyb_resid[sel], s=9, color=colour, label=name)
ax1.axhline(1.0, color="crimson", ls="--", lw=1, label="1 σ")
ax1.set_yscale("log")
ax1.set_ylabel(r"residual ($\sigma$)")
ax1.set_title("one chain, three algorithms")
ax1.legend(fontsize=8, ncol=4)

ax2.plot(hybrid_trace.k, lw=0.9, color="darkorange")
ax2.axhline(len(true_sites), color="crimson", ls="--", lw=1)
ax2.set_xlabel("step"); ax2.set_ylabel("$k$")
plt.tight_layout()

# %% [markdown]
# The trace is labelled, so the block that actually produced the breakthrough can
# be identified rather than read off the picture.

# %%
crossed = np.flatnonzero(hyb_resid < 2.0)
if crossed.size:
    j = hyb_steps[crossed[0]]
    print(f"first step below 2 σ: {j}, produced by {hybrid_trace.algorithm[j]!r}")
    print(f"residual there      : {hyb_resid[crossed[0]]:.2f} σ")
else:
    print("this run never reached 2 σ")

# %%
hyb_post = hybrid_trace.discard_burn_in(400)
hyb_final = hybrid_trace.site_idx[-1, : int(hybrid_trace.k[-1])]
hyb_R = detection_rates(hybrid_trace, burn=400)

print(f"criterion A  best residual : {hyb_resid.min():.2f} σ")
print(f"criterion B  mean detection: {hyb_R.mean():.2f}  (per spin {np.round(hyb_R, 2).tolist()})")
print(f"posterior mode k           : {np.bincount(hyb_post.k).argmax()}   (truth {len(true_sites)})")
print(f"\ntrue  orbits: {sorted(groups[true_sites].tolist())}")
print(f"final orbits: {sorted(groups[hyb_final].tolist())}")

# %% [markdown]
# ### 6.1 One property of the driver worth knowing
#
# `ParallelTempering.run` expands the ladder when the block starts and collapses
# to the cold chain when it ends. Since `HybridDriver` calls `run` once per
# cycle, **the hot rungs are rebuilt from the cold chain at every cycle
# boundary** — their accumulated exploration does not survive a `Step` boundary.

# %%
probe = make_state(start_sites)
returned = ParallelTempering(inner, n_replicas=8).run(
    probe, target, np.random.default_rng(0), n_steps=5)
print(f"input  : {probe.n_replicas} replica")
print(f"output : {returned.n_replicas} replica — the ladder is not carried out of the block")

# %% [markdown]
# Whether that costs anything at a given budget was not resolved here: at 600
# tempered steps, splitting them into 20 bursts of 30 versus one block of 600
# gave no consistent ordering across three seeds. Treat the block length as a
# knob to test, not one with a known-good default.

# %% [markdown]
# ## 7. Relaxing the *ab initio* constraint
#
# Everything above pins each spin's couplings to its DFT table value exactly.
# That treats the table as perfect. `GaussianOffset` relaxes it: a spin's
# coupling becomes its table value plus an offset under a Gaussian prior centred
# on zero, whose width encodes how far the DFT prediction is trusted
# (spec §5.3).
#
# Unlike every other kernel here, this one carries a **proper prior**, which
# enters the acceptance ratio explicitly.

# %%
OFFSET_SCALE = 4.0     # kHz — how far the table is allowed to be wrong
offset_prop = GaussianOffset(radius=1.5, scale=OFFSET_SCALE)
print(f"prior N(0, {OFFSET_SCALE}²) kHz, walk bounded to ±{offset_prop.upper:.0f} kHz")
for v in (0.0, 2.0, 8.0):
    print(f"  log prior at {v:4.1f} kHz: {offset_prop.log_prior(v):+.3f}")

# %% [markdown]
# ### 7.1 Data whose couplings are *not* the table values
#
# To test relaxation honestly the data must actually violate the constraint.
# True offsets are drawn for each spin, and the constrained model can no longer
# fit exactly no matter which sites it picks.

# %%
off_rng = np.random.default_rng(77)
true_dA_par = off_rng.normal(0.0, OFFSET_SCALE, size=len(true_sites))
true_dA_perp = off_rng.normal(0.0, OFFSET_SCALE, size=len(true_sites))

shifted_truth = make_state(true_sites, sigma=DATA_NOISE)
shifted_truth.dA_par[0, : len(true_sites)] = true_dA_par
shifted_truth.dA_perp[0, : len(true_sites)] = true_dA_perp

data = simulate_dataset(shifted_truth, blank, table, model, sigma=DATA_NOISE,
                        rng=np.random.default_rng(101))
obs = data.data_all
target = Target(data, model, GaussianL2(), table)

print("true offsets (kHz):")
for s, dp, dq in zip(true_sites, true_dA_par, true_dA_perp):
    print(f"    site {s:3d}   ΔA_par {dp:+6.2f}   ΔA_perp {dq:+6.2f}")
print(f"\nresidual of the constrained model at the true sites: "
      f"{residual(true_sites):.2f} σ")
print(f"residual with the true offsets applied              : "
      f"{residual(true_sites, true_dA_par, true_dA_perp):.2f} σ")

# %% [markdown]
# ### 7.2 Constrained versus relaxed
#
# The spins are **held at the true sites** for this comparison, and only the
# offsets are sampled. That is deliberate. A chain that searches configurations
# and relaxes couplings at the same time conflates the two, and the result says
# nothing about either — an earlier version of the T5 rung made exactly this
# mistake, and starting from a perturbed configuration *inverted* the comparison
# (`docs/test-plan.md` §5.6).
#
# With the sites fixed there is nothing left for the constrained model to sample:
# it is a single number.

# %%
offset_sampler = RWMH(ParameterBlock("offsets"), offset_prop)

relaxed = Trace(n_sites=len(table), k_max=K_MAX, n_exp=1)
t0 = time.time()
offset_sampler.run(make_state(true_sites), target, np.random.default_rng(13),
                   n_steps=3000, trace=relaxed)
print(f"3000 offset steps in {time.time() - t0:.1f} s")

constrained_resid = residual(true_sites)
_, r_resid = residual_curve(relaxed, stride=10)
floor = residual(true_sites, true_dA_par, true_dA_perp)

print(f"\nconstrained, offsets pinned at 0 : {constrained_resid:6.2f} σ")
print(f"relaxed,     best sampled offsets: {r_resid.min():6.2f} σ")
print(f"floor,       the true offsets    : {floor:6.2f} σ")

# %% [markdown]
# The comparison uses the **best** residual, not the median, and the reason is a
# trap worth naming. A model with extra sampled parameters has a *higher* median
# residual than one holding them at the prior mean, because a typical draw sits
# away from that mean. Judged on the median, relaxation looks worse precisely
# because it is exploring. The question to ask is whether it can **reach** a fit
# the constrained model cannot.

# %%
post = relaxed.discard_burn_in(1500)
sampled_par = post.dA_par[:, : len(true_sites)]
sampled_perp = post.dA_perp[:, : len(true_sites)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.plot(np.arange(0, len(relaxed), 10), r_resid, lw=1.2, color="darkviolet",
         label="relaxed")
ax1.axhline(constrained_resid, color="0.45", ls="-", lw=1.2,
            label="constrained (nothing to sample)")
ax1.axhline(floor, color="crimson", ls="--", lw=1, label="true-offset floor")
ax1.set_yscale("log")
ax1.set_xlabel("step")
ax1.set_ylabel(r"residual ($\sigma$)")
ax1.set_title("relaxation versus the hard constraint")
ax1.legend(fontsize=8)

colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(true_sites)))
for i, c in enumerate(colours):
    ax2.plot(sampled_par[:, i], lw=0.7, alpha=0.85, color=c)
    ax2.axhline(true_dA_par[i], color=c, ls=":", lw=1.2)
ax2.set_xlabel("step after burn-in")
ax2.set_ylabel(r"$\Delta A_\parallel$  (kHz)")
ax2.set_title("offset traces; dotted = truth")
plt.tight_layout()

# %%
print("posterior mean offset versus truth (kHz):")
print(f"{'site':>6}  {'ΔA_par':>16}  {'ΔA_perp':>16}")
for i, s in enumerate(true_sites):
    print(f"{s:6d}  {sampled_par[:, i].mean():+7.2f} vs {true_dA_par[i]:+6.2f}  "
          f"  {sampled_perp[:, i].mean():+7.2f} vs {true_dA_perp[i]:+6.2f}")

# %% [markdown]
# Relaxation recovers the offsets, including the one nearly three prior standard
# deviations out, and closes most of the gap to the floor. The residual it cannot
# close is the prior doing its job: an offset is pulled toward the DFT value
# unless the data pays for moving it.
#
# Two things this does **not** show. The chain started at the true sites, so it
# says nothing about relaxing and searching at once — that is harder, and the
# measured comparison is in the test plan. And a relaxed model will always fit at
# least as well as a constrained one, so a lower residual is not by itself
# evidence that the relaxation is physical.

# %% [markdown]
# ## Where to go next
#
# What this notebook covered:
#
# - a complete workflow with one algorithm (§1), judged by signal fit and
#   posterior containment rather than by a single configuration;
# - the walk in real space, and why symmetry orbits — not site indices — are the
#   unit of a correct answer (§2);
# - error against step count as the primary diagnostic (§3);
# - `RJMCMC` for the model dimension, and the multimodality that birth–death
#   moves alone cannot mix across (§4);
# - `ParallelTempering` for the barriers a single-spin walk cannot cross (§5);
# - `HybridDriver` composing all three into one labelled trace (§6);
# - `GaussianOffset` for when the DFT table itself is the thing in doubt (§7).
#
# Not yet implemented: the ensemble runner and SLURM integration, the posterior
# metrics and plotting package, a Wasserstein likelihood, and a PyCCE backend for
# baths where the CCE-1 analytic form is not enough.
#
# Two documents carry what a tutorial cannot. `docs/model-specification.md` is
# the authoritative statement of the model, including the five conventions that
# had to be resolved against conflicting published sources.
# `docs/test-plan.md` records what each rung of the test ladder can and cannot
# demonstrate, and every calibration run behind every threshold — including the
# runs that produced the numbers quoted here.
