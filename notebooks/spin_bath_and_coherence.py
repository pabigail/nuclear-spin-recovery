# %% [markdown]
# # Generating nuclear spin baths and coherence signals
#
# This tutorial covers the two things the forward-model layer does:
#
# 1. building an instance of a nuclear spin bath around an NV center in diamond, and
# 2. computing the coherence signal that bath would produce in a dynamical
#    decoupling experiment, given $B_z$, the pulse number $N$, and the interpulse
#    spacings $\tau$.
#
# The physics and conventions are specified in `docs/model-specification.md`.
# Units throughout are **kHz** for hyperfine couplings, **ms** for $\tau$,
# **G** for the magnetic field, and **Å** for positions. That triple is chosen so
# that $\omega\tau$ is dimensionless with no conversion factor anywhere.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

# Import from the source tree directly. The editable install works too, but its
# .pth file is periodically flagged hidden by macOS under iCloud-synced folders,
# and CPython silently skips hidden .pth files -- so the package disappears with
# no warning. This makes the notebook independent of that.
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nuclear_spin_recovery import (
    AnalyticCCE1,
    Experiment,
    ExperimentSet,
    GaussianL2,
    SiteTable,
    State,
    StretchedExponential,
    simulate_coherence,
    simulate_dataset,
)

NV2 = REPO / "nv-2.txt"
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. The lattice site table
#
# Candidate nuclear positions are restricted to lattice sites whose hyperfine
# couplings were computed by DFT. This is the prior information that makes the
# inverse problem tractable: instead of searching a continuum of coupling values,
# a spin walks over a finite, physically meaningful set.
#
# Two thresholds filter the table, and both matter:
#
# - `strong_thresh` drops sites where *either* component is too large. Strongly
#   coupled spins are resolved by other means and are not what this method targets.
# - `weak_thresh` drops sites where *both* components are too small. Their
#   modulation falls below the detection floor set by shot noise and sampling, so
#   including them adds parameters the data cannot constrain.

# %%
table = SiteTable.from_ivady_file(NV2, strong_thresh=750.0, weak_thresh=5.0)
print(f"{len(table)} candidate sites survive filtering")
print(f"A_par  range: {table.a_par.min():8.2f} to {table.a_par.max():8.2f} kHz")
print(f"A_perp range: {table.a_perp.min():8.2f} to {table.a_perp.max():8.2f} kHz")
print(f"distance    : {table.distance.min():8.2f} to {table.distance.max():8.2f} A")

# %%
magnitude = np.hypot(table.a_par, table.a_perp)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.scatter(table.distance, magnitude, s=3, alpha=0.3)
ax1.set_yscale("log")
ax1.set_xlabel("distance from NV (Å)")
ax1.set_ylabel(r"$\sqrt{A_\parallel^2 + A_\perp^2}$ (kHz)")
ax1.set_title("Coupling falls off with distance")

ax2.scatter(table.a_par, table.a_perp, s=3, alpha=0.3)
ax2.set_xlabel(r"$A_\parallel$ (kHz)")
ax2.set_ylabel(r"$A_\perp$ (kHz)")
ax2.set_title("Available couplings")
fig.tight_layout()

# %% [markdown]
# ### Symmetry-equivalent sites
#
# Distinct lattice sites related by crystal symmetry have identical couplings and
# are therefore **indistinguishable in coherence data**. The table can label them,
# which matters later: posterior comparisons must be made on couplings, never on
# site indices.

# %%
labels = table.symmetry_groups(tol=0.1)
sizes = np.bincount(labels)
print(f"{labels.max() + 1} distinct coupling groups among {len(table)} sites")
print(f"largest group holds {sizes.max()} symmetry-equivalent sites")

# %% [markdown]
# ## 2. Generating a bath instance
#
# A bath is a set of occupied sites. One hard rule: **no two spins may occupy the
# same site**. `State.from_sites` enforces it.
#
# The simplest generator picks a fixed number of sites uniformly.

# %%
def random_bath(table, n_spins, rng, *, n_exp=1, lam=3e-3, n_stretch=1.0,
                sigma=0.01, k_max=64):
    """A bath of n_spins distinct sites drawn uniformly from the table."""
    sites = rng.choice(len(table), size=n_spins, replace=False)
    return State.from_sites(
        np.sort(sites),
        n_sites=len(table),
        n_exp=n_exp,
        lam=np.full((1, n_exp), lam),
        n_stretch=np.full((1, n_exp), n_stretch),
        sigma=np.full((1, n_exp), sigma),
        k_max=k_max,
    )


bath = random_bath(table, n_spins=10, rng=rng)
print(f"k = {bath.k[0]} spins")
print("A_par  (kHz):", np.round(bath.a_par_per_spin(table)[0, : bath.k[0]], 2))
print("A_perp (kHz):", np.round(bath.a_perp_per_spin(table)[0, : bath.k[0]], 2))

# %% [markdown]
# ### Sampling at natural abundance
#
# A more physical generator occupies each site independently with probability
# equal to the isotopic concentration — 1.1% for $^{13}$C in natural diamond.
#
# One caveat worth stating plainly: the table has already been filtered, so this
# gives the expected number of *detectable* spins, not the true bath size. The
# full bath contains many more nuclei whose couplings are below `weak_thresh`.

# %%
occupied = rng.random(len(table)) < 0.011
abundance_bath = State.from_sites(
    np.flatnonzero(occupied),
    n_sites=len(table),
    n_exp=1,
    lam=np.array([[3e-3]]),
    n_stretch=np.array([[1.0]]),
    sigma=np.array([[0.01]]),
    k_max=256,
)
print(f"natural abundance draw: {abundance_bath.k[0]} detectable spins")
abundance_bath.check_invariants()   # raises if the state is malformed

# %% [markdown]
# ## 3. Defining an experiment
#
# An `Experiment` is the measurement setting: a magnetic field, a pulse number,
# and the interpulse spacings sampled. Nothing about the sample.
#
# Below: 250 values of $\tau$ up to 8 µs (0.008 ms), a CPMG-16 sequence, at 311 G —
# the default settings used throughout the papers.

# %%
tau = np.linspace(0.0, 8e-3, 250, endpoint=False) + 8e-3 / 250
experiment = ExperimentSet([Experiment(tau=tau, n_pulses=16, b_z=311.0)])

model = AnalyticCCE1(StretchedExponential())
signal = simulate_coherence(bath, experiment, table, model)

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(tau * 1e3, signal, lw=1)
ax.set_xlabel(r"interpulse spacing $\tau$ (µs)")
ax.set_ylabel("coherence")
ax.set_title("CPMG-16 at 311 G, 10 spins")
ax.axhline(0.5, color="gray", ls=":", lw=1)
fig.tight_layout()

# %% [markdown]
# The dips are individual nuclear spins coming into resonance. The envelope decays
# toward **0.5**, not 0 — that is the fully mixed population, and it is why the
# decoherence envelope sits *inside* the half-sum:
#
# $$f(\tau) = \tfrac{1}{2}\left(1 + \left[\prod_i M_i(\tau)\right] e^{-(\tau/\lambda)^n}\right)$$
#
# See `docs/model-specification.md` §4.2 — two of the source papers print this
# differently, and the difference is large.

# %% [markdown]
# ## 4. Varying the experimental inputs
#
# Each of $N$, $B_z$, and the $\tau$ window changes what the experiment can see.
# More pulses sharpen the resonances; the field sets the Larmor frequency and so
# the spacing between them.

# %%
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

for n_pulses in (8, 16, 32):
    es = ExperimentSet([Experiment(tau=tau, n_pulses=n_pulses, b_z=311.0)])
    axes[0].plot(tau * 1e3, simulate_coherence(bath, es, table, model),
                 lw=1, label=f"N = {n_pulses}")
axes[0].set_ylabel("coherence")
axes[0].set_title("Pulse number: more pulses, sharper dips")
axes[0].legend(loc="lower left", fontsize=8)

for b_z in (311.0, 403.0, 500.0):
    es = ExperimentSet([Experiment(tau=tau, n_pulses=16, b_z=b_z)])
    axes[1].plot(tau * 1e3, simulate_coherence(bath, es, table, model),
                 lw=1, label=f"B = {b_z:.0f} G")
axes[1].set_ylabel("coherence")
axes[1].set_title("Magnetic field: sets the Larmor frequency")
axes[1].legend(loc="lower left", fontsize=8)

for lam in (1e-3, 3e-3, 1e-2):
    b = random_bath(table, 10, np.random.default_rng(0), lam=lam)
    es = ExperimentSet([Experiment(tau=tau, n_pulses=16, b_z=311.0)])
    axes[2].plot(tau * 1e3, simulate_coherence(b, es, table, model),
                 lw=1, label=rf"$\lambda$ = {lam * 1e3:.0f} µs")
axes[2].set_ylabel("coherence")
axes[2].set_xlabel(r"interpulse spacing $\tau$ (µs)")
axes[2].set_title(r"Decoherence envelope: all curves decay toward 0.5")
axes[2].legend(loc="lower left", fontsize=8)
fig.tight_layout()

# %% [markdown]
# ### Bath size

# %%
fig, ax = plt.subplots(figsize=(11, 3.5))
for n_spins in (1, 5, 20):
    b = random_bath(table, n_spins, np.random.default_rng(1), lam=1e9)
    es = ExperimentSet([Experiment(tau=tau, n_pulses=16, b_z=311.0)])
    ax.plot(tau * 1e3, simulate_coherence(b, es, table, model),
            lw=1, label=f"{n_spins} spins")
ax.set_xlabel(r"interpulse spacing $\tau$ (µs)")
ax.set_ylabel("coherence")
ax.set_title("Bath size, envelope switched off")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## 5. Several experiments at once
#
# Experiments on the same defect are fit jointly. The **spin configuration is
# shared** — it is a property of the sample. The decay constant $\lambda$, the
# noise $\sigma$, and the $\tau$ grid are **per-experiment**; a single $\lambda$
# cannot describe different pulse numbers, since decoupling extends coherence
# with $N$.
#
# Grids need not even be the same length.

# %%
joint = ExperimentSet([
    Experiment(tau=np.linspace(0.0, 8e-3, 250, endpoint=False) + 8e-3 / 250,
               n_pulses=8, b_z=311.0),
    Experiment(tau=np.linspace(0.0, 6e-3, 180, endpoint=False) + 6e-3 / 180,
               n_pulses=16, b_z=311.0),
])

joint_bath = random_bath(table, 10, np.random.default_rng(0), n_exp=2)
joint_bath.lam[0] = [4e-3, 6e-3]        # longer coherence at higher N
joint_bath.sigma[0] = [0.02, 0.01]      # the second was averaged longer

joint_signal = simulate_coherence(joint_bath, joint, table, model)
pieces = joint.split(joint_signal)

fig, axes = plt.subplots(2, 1, figsize=(11, 6))
for ax, exp, piece in zip(axes, joint.experiments, pieces):
    ax.plot(exp.tau * 1e3, piece, lw=1)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_ylabel("coherence")
    ax.set_title(f"N = {exp.n_pulses}, {len(exp)} points")
axes[-1].set_xlabel(r"interpulse spacing $\tau$ (µs)")
fig.tight_layout()

# %% [markdown]
# ## 6. Adding measurement noise
#
# `simulate_dataset` attaches noisy data to a copy of the experiment set. Pass an
# array of per-experiment $\sigma$ values to give each dataset its own noise level.

# %%
noisy = simulate_dataset(
    joint_bath, joint, table, model,
    sigma=np.array([0.02, 0.01]),
    rng=np.random.default_rng(42),
)
truth = simulate_coherence(joint_bath, joint, table, model)

fig, axes = plt.subplots(2, 1, figsize=(11, 6))
for ax, exp, t, d in zip(axes, joint.experiments,
                         joint.split(truth), joint.split(noisy.data_all)):
    ax.plot(exp.tau * 1e3, d, lw=0.6, color="0.6", label="measured")
    ax.plot(exp.tau * 1e3, t, lw=1.2, color="C3", label="truth")
    ax.set_ylabel("coherence")
    ax.set_title(f"N = {exp.n_pulses}")
    ax.legend(loc="lower left", fontsize=8)
axes[-1].set_xlabel(r"interpulse spacing $\tau$ (µs)")
fig.tight_layout()

# %% [markdown]
# ## 7. Scoring a configuration
#
# The likelihood is what the samplers will maximise. It is unnormalized, so a
# perfect fit scores exactly zero and everything else is negative.

# %%
likelihood = GaussianL2()
print(f"truth            : {likelihood.log_prob(joint_bath, noisy, model, table)[0]:12.2f}")

wrong = random_bath(table, 10, np.random.default_rng(7), n_exp=2)
wrong.lam[0] = joint_bath.lam[0]
wrong.sigma[0] = joint_bath.sigma[0]
print(f"a different bath : {likelihood.log_prob(wrong, noisy, model, table)[0]:12.2f}")

empty = State.from_sites(
    (), n_sites=len(table), n_exp=2,
    lam=joint_bath.lam.copy(), n_stretch=joint_bath.n_stretch.copy(),
    sigma=joint_bath.sigma.copy(), k_max=64,
)
print(f"no spins at all  : {likelihood.log_prob(empty, noisy, model, table)[0]:12.2f}")

# %% [markdown]
# The true configuration scores highest. Recovering it from the data alone — without
# knowing $k$ in advance — is the inverse problem the samplers solve in later phases.
#
# ## Where to go next
#
# - `docs/model-specification.md` — the physics, the statistical model, and the
#   places where the published papers disagree with each other.
# - `SiteTable.from_ase` — build a table for a lattice other than this one, using
#   the point-dipole approximation instead of DFT couplings.
