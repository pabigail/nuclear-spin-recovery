# %% [markdown]
# # Sampling: proposals, Metropolis-Hastings, and the trace
#
# The previous tutorial built a coherence signal from a known spin bath. This one
# runs the inverse: given a signal, explore the distribution over baths that could
# have produced it.
#
# Phase 2 implements the fixed-dimension machinery — random-walk Metropolis-Hastings
# over continuous and discrete parameters. The number of spins $k$ is held fixed
# here; trans-dimensional moves arrive in phase 3.
#
# Four objects do the work:
#
# | object | role |
# |---|---|
# | `Target` | bundles data, forward model, likelihood and site table behind one call |
# | `Proposal` | suggests a move and reports its proposal ratio |
# | `RWMH` | proposes, then accepts or rejects |
# | `Trace` | records where the chain went |
#
# The specification is `docs/model-specification.md` §8.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nuclear_spin_recovery import (
    AnalyticCCE1,
    ContinuousReflected,
    DiscreteLatticeWalk,
    Experiment,
    ExperimentSet,
    GaussianL2,
    NeighborIndex,
    ParameterBlock,
    RWMH,
    SiteTable,
    State,
    StretchedExponential,
    Target,
    Trace,
    simulate_dataset,
)

table = SiteTable.from_ivady_file(REPO / "nv-2.txt", strong_thresh=750.0, weak_thresh=5.0)
model = AnalyticCCE1(StretchedExponential())
print(f"{len(table)} candidate sites")

# %% [markdown]
# ## 1. Ground truth to recover
#
# A ten-spin bath, a CPMG-16 experiment at 311 G, and noisy data generated from it.
# Because we made the data, we know the answer and can check the sampler against it.

# %%
def make_state(sites, lam=3e-3, sigma=0.01, k_max=64, n_exp=1):
    sites = np.sort(np.asarray(sites, dtype=int))
    return State.from_sites(
        sites, n_sites=len(table), n_exp=n_exp,
        lam=np.full((1, n_exp), lam),
        n_stretch=np.ones((1, n_exp)),
        sigma=np.full((1, n_exp), sigma),
        k_max=k_max,
    )


rng = np.random.default_rng(11)
true_sites = rng.choice(len(table), size=10, replace=False)
truth = make_state(true_sites, lam=3e-3)

tau = np.linspace(0.0, 8e-3, 250, endpoint=False) + 8e-3 / 250
blank = ExperimentSet([Experiment(tau=tau, n_pulses=16, b_z=311.0)])
data = simulate_dataset(truth, blank, table, model, sigma=0.002,
                        rng=np.random.default_rng(3))

print("true A_par  (kHz):", np.round(truth.a_par_per_spin(table)[0, :10], 1))
print("true lambda (ms) :", truth.lam[0, 0])

# %% [markdown]
# ## 2. The `Target`
#
# Everything the acceptance ratio needs, behind one method. An algorithm calls
# `target.log_prob(state, beta)` and knows nothing about coherence, lattices or
# hyperfine couplings — which is why the forward model and likelihood can be
# swapped without touching a sampler.

# %%
target = Target(data, model, GaussianL2(), table)

print(f"log L at the truth        : {target.log_prob(truth)[0]:12.1f}")
wrong = make_state(rng.choice(len(table), size=10, replace=False))
print(f"log L at a different bath : {target.log_prob(wrong)[0]:12.1f}")

# `beta` is the inverse temperature, used later by parallel tempering.
for beta in (1.0, 0.5, 0.25):
    print(f"  beta = {beta:4.2f} -> {target.log_prob(truth, beta=beta)[0]:10.1f}")

# %% [markdown]
# ## 3. Proposals
#
# A proposal returns two things: where to go, and the **log proposal ratio**
# $\log\frac{r(z \to x)}{r(x \to z)}$. Returning the ratio from the proposal — rather
# than assuming symmetry at the call site — is what keeps an asymmetric kernel correct.
#
# ### Continuous: a reflected random walk
#
# Used for $\lambda$, the stretch exponent, and $\sigma$. It steps uniformly within a
# radius and **reflects** at the domain bounds rather than clipping. Clipping would
# pile probability onto the boundary and break the symmetry that makes the ratio zero.

# %%
walk = ContinuousReflected(radius=0.15, lower=0.0, upper=1.0)
r = np.random.default_rng(0)

x, path = 0.92, []
for _ in range(400):
    x, log_ratio = walk.propose(r, x)
    path.append(x)

print(f"log proposal ratio: {log_ratio}   (exactly zero — the kernel is symmetric)")
print(f"stays in bounds   : {min(path):.4f} to {max(path):.4f}")
print(f"pile-up at 1.0    : {sum(p >= 0.9999 for p in path)} samples "
      f"(clipping would give many)")

# %% [markdown]
# ### Discrete: a lattice walk under an occupancy constraint
#
# A spin moves to another site within `radius`. Two spins may never share a site, and
# **that constraint makes the kernel asymmetric**: the number of available neighbours
# differs between where you are and where you are going.
#
# $$\log\text{ratio} = \log\bigl|N_R(x)\setminus O\bigr| - \log\bigl|N_R(z)\setminus O\bigr|$$
#
# Neighbour lists are precomputed once per table.

# %%
neighbors = NeighborIndex(table.positions, radius=5.0)
lattice_walk = DiscreteLatticeWalk(neighbors)

occupied = truth.occupied[0]
site = int(truth.site_idx[0, 0])
free = neighbors.count_available(site, occupied, ignore=site)
print(f"site {site}: {len(neighbors.neighbors(site))} neighbours within 5 Å, "
      f"{free} of them free")

proposed, log_ratio = lattice_walk.propose(r, site, occupied=occupied)
print(f"proposed move {site} -> {proposed},  log ratio = {log_ratio:+.4f}")

# %% [markdown]
# **Why this term matters.** Drop it and the chain still finds good configurations —
# it just samples the wrong distribution. Under a flat target the stationary law should
# be uniform over sites; without the correction it converges to something proportional
# to the neighbour count. No recovery test detects this, which is why
# `tests/theory/test_sampler_invariance.py` exists.

# %%
UNEVEN = np.array([[0., 0, 0], [1., 0, 0], [1.6, 0, 0],
                   [2.1, 0, 0], [3.1, 0, 0], [4.1, 0, 0]])
toy = NeighborIndex(UNEVEN, radius=1.2)


class Flat:
    """Every configuration equally likely."""
    def log_prob(self, state, beta=1.0):
        return np.zeros(state.n_replicas)


class NoRatio(DiscreteLatticeWalk):
    """The bug: treat the constrained walk as if it were symmetric."""
    def propose(self, rng, current, occupied=None):
        return super().propose(rng, current, occupied=occupied)[0], 0.0


def site_histogram(proposal, n_steps=40_000, seed=3):
    st = State.from_sites((0,), n_sites=6, n_exp=1, lam=np.array([[0.5]]),
                          n_stretch=np.array([[1.0]]), sigma=np.array([[0.1]]), k_max=4)
    tr = Trace(n_sites=6, k_max=4, n_exp=1)
    RWMH(ParameterBlock("sites"), proposal).run(
        st, Flat(), np.random.default_rng(seed), n_steps=n_steps, trace=tr)
    visits = np.bincount(tr.site_idx[4_000:, 0], minlength=6)
    return visits / visits.sum()


counts = np.array([toy.count_available(i, np.zeros(6, dtype=bool)) for i in range(6)])
correct = site_histogram(DiscreteLatticeWalk(toy))
buggy = site_histogram(NoRatio(toy))

fig, ax = plt.subplots(figsize=(9, 3.4))
idx = np.arange(6)
ax.bar(idx - 0.2, correct, 0.38, label="with proposal ratio")
ax.bar(idx + 0.2, buggy, 0.38, label="ratio omitted")
ax.axhline(1 / 6, color="k", ls="--", lw=1, label="uniform, the correct answer")
ax.plot(idx + 0.2, counts / counts.sum(), "k.", ms=9, label=r"$\propto |N(x)|$")
ax.set_xlabel("site"); ax.set_ylabel("visit fraction")
ax.set_title("Omitting the proposal ratio samples the wrong distribution")
ax.legend(fontsize=8)
fig.tight_layout()

print("neighbour counts :", counts.tolist())
print("with    ratio    :", np.round(correct, 4).tolist())
print("without ratio    :", np.round(buggy, 4).tolist())

# %% [markdown]
# ## 4. `RWMH` — setting up a sampler
#
# An algorithm pairs a **parameter block** (what it updates) with a **proposal**
# (how it moves). Everything outside the block is held fixed — that partition is how
# the algorithms will compose into a hybrid sampler in phase 3.

# %%
lam_sampler = RWMH(ParameterBlock("lam"), ContinuousReflected(0.0004, 1e-4, 2e-2))
site_sampler = RWMH(ParameterBlock("sites"), DiscreteLatticeWalk(neighbors))

print("blocks available:", ["sites", "lam", "n_stretch", "sigma"])
print("lam_sampler.label :", lam_sampler.label)
print("site_sampler.label:", site_sampler.label)

# %% [markdown]
# ## 5. Recovering $\lambda$ with the spins held at truth
#
# The simplest possible inference: one continuous parameter, everything else known.
# The chain should concentrate on the value used to generate the data.

# %%
start = make_state(true_sites, lam=1.2e-2)     # deliberately wrong
trace = Trace(n_sites=len(table), k_max=64, n_exp=1)
lam_sampler.run(start, target, np.random.default_rng(0), n_steps=4000, trace=trace)

burn = 1000
post = trace.lam[burn:, 0]
print(f"true lambda      : {truth.lam[0, 0] * 1e3:.3f} µs")
print(f"posterior mean   : {post.mean() * 1e3:.3f} µs")
print(f"posterior sd     : {post.std() * 1e3:.3f} µs")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4),
                               gridspec_kw={"width_ratios": [2, 1]})
ax1.plot(trace.lam[:, 0] * 1e3, lw=0.7)
ax1.axvspan(0, burn, color="k", alpha=0.07)
ax1.axhline(truth.lam[0, 0] * 1e3, color="C3", lw=1.2, label="truth")
ax1.set_xlabel("trace step"); ax1.set_ylabel(r"$\lambda$ (µs)")
ax1.set_title("Chain, burn-in shaded"); ax1.legend(fontsize=8)

ax2.hist(post * 1e3, bins=40, orientation="horizontal")
ax2.axhline(truth.lam[0, 0] * 1e3, color="C3", lw=1.2)
ax2.set_xlabel("count"); ax2.set_title("Posterior")
fig.tight_layout()

# %% [markdown]
# ## 6. Reading the trace
#
# `Trace` records the configuration, every continuous parameter, the log-likelihood,
# and which sub-algorithm produced each step. That last column matters once several
# algorithms cycle: it lets you see which block is responsible for a stall.

# %%
print(f"steps recorded : {len(trace)}")
print(f"site_idx shape : {trace.site_idx.shape}")
print(f"lam shape      : {trace.lam.shape}")
print(f"labels         : {set(trace.algorithm)}")

accept = np.mean(np.diff(trace.lam[:, 0]) != 0)
print(f"acceptance rate: {accept:.2%}")

after = trace.discard_burn_in(burn)
print(f"after burn-in  : {len(after)} steps (original still {len(trace)})")

# %% [markdown]
# ## 7. Recovering spin positions with $k$ fixed
#
# Now the discrete block. The number of spins is correct, but every spin starts on the
# wrong site; the sampler walks them over the lattice toward configurations that
# explain the data.

# %%
scrambled = make_state(rng.choice(len(table), size=10, replace=False), lam=3e-3)
site_trace = Trace(n_sites=len(table), k_max=64, n_exp=1)
final = site_sampler.run(scrambled, target, np.random.default_rng(1),
                         n_steps=6000, trace=site_trace)

print(f"log L at start : {target.log_prob(scrambled)[0]:12.1f}")
print(f"log L at end   : {target.log_prob(final)[0]:12.1f}")
print(f"log L at truth : {target.log_prob(truth)[0]:12.1f}")

fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(site_trace.log_prob, lw=0.7)
ax.axhline(target.log_prob(truth)[0], color="C3", lw=1.2, label="truth")
ax.set_xlabel("trace step"); ax.set_ylabel("log likelihood")
ax.set_title("Discrete random walk over lattice sites")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ### The likelihood is nearly at truth. The configuration is not.
#
# Measure recovery the way the specification does (§9): match spins on their hyperfine
# couplings within 0.1 kHz, never on site index, since symmetry-equivalent sites are
# genuinely indistinguishable in the data.

# %%
def coupling_pairs(state):
    k = int(state.k[0])
    return list(zip(state.a_par_per_spin(table)[0, :k],
                    state.a_perp_per_spin(table)[0, :k]))


def detected(recovered, reference, tol=0.1):
    return sum(
        any(abs(a - c) <= tol and abs(b - d) <= tol for c, d in coupling_pairs(recovered))
        for a, b in coupling_pairs(reference)
    )


n_found = detected(final, truth)
print(f"spins recovered within 0.1 kHz : {n_found} of {int(truth.k[0])}")
print(f"log L gap to truth             : {target.log_prob(truth)[0] - target.log_prob(final)[0]:.1f}")

# %% [markdown]
# A handful of spins found, yet the likelihood lands within a few units of the truth.
# That gap is the whole problem: **many distinct configurations explain this data almost
# equally well**, so a chain that maximises the likelihood has not thereby found the
# right answer. Weakly coupled spins modulate the signal by less than the noise, and the
# data cannot distinguish them.
#
# This is not a failure of the sampler. It is why the method reports a posterior rather
# than a point estimate, and why phase 3 adds parallel tempering (to escape the local
# minima a single block walks into) and RJMCMC (so the number of spins is inferred
# rather than assumed correct, as it was here).

# %%
def couplings(state):
    k = int(state.k[0])
    return np.sort(np.hypot(state.a_par_per_spin(table)[0, :k],
                            state.a_perp_per_spin(table)[0, :k]))


fig, ax = plt.subplots(figsize=(9, 3.4))
for values, label, style in ((couplings(truth), "truth", "-o"),
                             (couplings(final), "recovered", "-s"),
                             (couplings(scrambled), "start", ":^")):
    ax.plot(values, style, ms=4, lw=1, label=label)
ax.set_yscale("log")
ax.set_xlabel("spin, sorted by coupling")
ax.set_ylabel(r"$\sqrt{A_\parallel^2 + A_\perp^2}$ (kHz)")
ax.set_title("Strong couplings recover first")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## 8. Tempering, and the replica axis
#
# `beta` flattens the target. At $\beta = 1$ the chain samples the posterior; as
# $\beta \to 0$ every proposal is accepted and the chain explores freely. Parallel
# tempering in phase 3 will run a ladder $\beta_j = 2^{-j}$ and swap between rungs,
# keeping only the $\beta_0$ chain.
#
# `State` already carries the replica axis that ladder will use.

# %%
replicas = make_state(true_sites, lam=3e-3).expand_replicas(4)
print(f"replicas        : {replicas.n_replicas}")
print(f"site_idx shape  : {replicas.site_idx.shape}")

# At beta = 1 this posterior is so sharp that a 0.4 µs proposal is almost always
# rejected, and the replicas would sit on top of each other.  A hot rung accepts
# freely, which is exactly the point of the ladder.
stepped = lam_sampler.step(replicas, target, np.random.default_rng(0), beta=0.0)
print(f"lambda per replica, one hot step : {np.round(stepped.lam[:, 0] * 1e3, 3).tolist()} µs")
cold = lam_sampler.step(replicas, target, np.random.default_rng(0), beta=1.0)
print(f"lambda per replica, one cold step: {np.round(cold.lam[:, 0] * 1e3, 3).tolist()} µs")
print(f"collapse_to_cold -> {stepped.collapse_to_cold().n_replicas} replica "
      f"(hot chains do not target the posterior, so the Trace never sees them)")

for beta in (1.0, 0.3, 0.0):
    st = make_state(true_sites, lam=1e-2)
    tr = Trace(n_sites=len(table), k_max=64, n_exp=1)
    lam_sampler.run(st, target, np.random.default_rng(0), n_steps=800,
                    trace=tr, beta=beta)
    moved = np.mean(np.diff(tr.lam[:, 0]) != 0)
    print(f"  beta = {beta:4.2f}: acceptance {moved:5.1%}")

# %% [markdown]
# ## What phase 3 adds
#
# - **RJMCMC** — birth and death moves so $k$ is inferred rather than assumed.
# - **ParallelTempering** — the $\beta$ ladder, to escape the local minima visible in §7.
# - **HybridDriver** — a schedule cycling the three, concatenated into one trace.
#
# Until then, every algorithm here already speaks the interface those will use:
# a `ParameterBlock`, a `Proposal`, and `target.log_prob(state, beta)`.
