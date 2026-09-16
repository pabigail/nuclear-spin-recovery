# Model specification

Bayesian recovery of nuclear spin configurations around optically active spin
defects, from coherence data measured in dynamical decoupling experiments.

This document specifies the physics, the statistical model, and the sampling
scheme. It is the reference against which the implementation is written and
tested; it contains no code. Where the two source papers disagree with each
other or with the reference implementation, the resolution is stated explicitly
and marked.

**Sources.** The method is that of Poteshman, Yun, Taminiau and Galli,
*Trans-dimensional Hamiltonian model selection and parameter estimation from
sparse, noisy data* (arXiv:2506.18802; Quantum, 2026), applied as in Poteshman,
Onizhuk, Egerstrom, Mark, Awschalom, Heremans and Galli, *High-throughput
spin-bath characterization of spin defects in semiconductors* (arXiv:2506.19259;
Phys. Rev. Applied **24**, 054048, 2025). Referred to below as the **methods
paper** and the **application paper**.

---

## 1. Scope

The package addresses the inverse problem: given one or more measured coherence
curves from a single spin defect, recover the number of nuclear spins in the
surrounding bath, their lattice positions, and their hyperfine couplings to the
defect, together with the parameters of an empirical decoherence envelope.

The problem is ill-posed. Many distinct spin configurations produce coherence
signals that are indistinguishable at experimental resolution, particularly for
weakly coupled spins. The output is therefore a posterior distribution over
configurations — including over the *number* of spins, which is itself unknown
and makes the inference trans-dimensional — rather than a point estimate.

Two defect families are in scope: NV centers in diamond with a $^{13}$C bath,
and divacancies in SiC with a mixed $^{13}$C / $^{29}$Si bath. The mixed-isotope
case requires a per-spin nuclear gyromagnetic ratio, which is a generalization
of the single-isotope expressions printed in both papers.

---

## 2. Notation and units

| Symbol | Meaning | Unit |
|---|---|---|
| $\tau$ | interpulse spacing | ms |
| $N_e$ | number of CP pulses in experiment $e$ | — |
| $B_{z,e}$ | external field along the defect axis, experiment $e$ | G |
| $A_{\parallel}, A_{\perp}$ | secular hyperfine components | kHz |
| $\gamma_{n(i)}$ | nuclear gyromagnetic ratio, per spin, set by isotope | rad ms$^{-1}$ G$^{-1}$ |
| $\eta_i$ | Larmor phase $\omega_{L,i}\tau$, derived not stored | — |
| $\omega_L$ | nuclear Larmor frequency | rad ms$^{-1}$ |
| $\lambda_e$ | decoherence decay constant, experiment $e$ | ms |
| $n_e$ | stretch exponent, experiment $e$ | — |
| $\sigma_e$ | noise standard deviation, experiment $e$ | (coherence units) |
| $k$ | number of nuclear spins | — |
| $s_i$ | lattice site index of spin $i$ | — |
| positions | Cartesian coordinates of lattice sites | Å |

**Angular-frequency convention.** Hyperfine couplings are stored and reported in
kHz as ordinary frequencies, matching the *ab initio* tables, and are converted
to angular frequency by a factor $2\pi$ before entering any trigonometric
expression:

$$\tilde{A}_{\parallel} = 2\pi A_{\parallel}, \qquad \tilde{A}_{\perp} = 2\pi A_{\perp}.$$

Nuclear gyromagnetic ratios are taken already in angular units, as supplied by
PyCCE's isotope table, so $\omega_L$ carries no additional $2\pi$. Throughout
Sec. 4 the tilde is suppressed and $A_{\parallel}, A_{\perp}$ denote the angular
quantities.

The unit triple (kHz, ms, G) is chosen so that products such as $\omega\tau$ are
dimensionless with no conversion constant anywhere in the inner loop. Conversion
happens only at two boundaries: when the site table is loaded, and when an
experiment is constructed. Everything downstream is unit-consistent by
construction.

---

## 3. Physical model

A single central electronic spin in an external field $B_z$ applied along the
defect symmetry axis interacts with $k$ surrounding nuclear spins. The
Hamiltonian is

$$\mathcal{H}_k = D\mathcal{S}_z^2 + \gamma_e B_z \mathcal{S}_z + \sum_{i=1}^{k}\gamma_{n(i)} B_z \mathcal{I}_{z,i} + \sum_{i=1}^{k}\mathcal{S}_z\left(A_{\parallel,i}\mathcal{I}_{z,i} + A_{\perp,i}\mathcal{I}_{x,i}\right),$$

where $D$ is the axial zero-field splitting ($D = 2\pi \times 2.87$ GHz for the
NV center in diamond), $\gamma_e$ is the electronic gyromagnetic ratio, and
$n(i)$ denotes the isotope occupying site $s_i$ — so $\gamma_{n(i)}$ varies
between spins in a mixed-isotope bath.

Because the electronic level splitting dominates, the hyperfine interaction is
taken in the secular approximation and enters only through the two components
$A_{\parallel}$ and $A_{\perp}$ relative to the defect axis.

**Nuclear–nuclear interactions are neglected.** These scale as $\gamma_n^2$ and
are weak compared with the electron–nuclear coupling over the short interpulse
spacings ($\tau < 10\ \mu$s) and modest pulse numbers ($N \leq 32$) that the
high-throughput regime targets. This assumption defines the validity domain of
the analytic forward model and is revisited in Sec. 10.

---

## 4. Forward model

### 4.1 Single-spin modulation

Under a CPMG-$N$ sequence, each nuclear spin independently modulates the
electronic coherence. For spin $i$ in experiment $e$, at interpulse spacing
$\tau$:

$$M_i(\tau) = 1 - m_{i,x}^2\,\frac{(1-\cos\alpha_i)\bigl(1-\cos\eta_i\bigr)}{1 + \cos\alpha_i\cos\eta_i - m_{i,z}\sin\alpha_i\sin\eta_i}\,\sin^2\!\left(\frac{N_e\phi_i}{2}\right)$$

with

$$\cos\phi_i = \cos\alpha_i\cos\eta_i - m_{i,z}\sin\alpha_i\sin\eta_i,$$

$$m_{i,z} = \frac{A_{\parallel,i}+\omega_{L,i}}{\tilde\omega_i}, \qquad m_{i,x} = \frac{A_{\perp,i}}{\tilde\omega_i}, \qquad \tilde\omega_i = \sqrt{(A_{\parallel,i}+\omega_{L,i})^2 + A_{\perp,i}^2},$$

$$\alpha_i = \tilde\omega_i\,\tau, \qquad \eta_i = \omega_{L,i}\,\tau, \qquad \omega_{L,i} = \gamma_{n(i)}\,B_{z,e}.$$

The quantity carried per spin is the **nuclear gyromagnetic ratio**
$\gamma_{n(i)}$, fixed by the isotope occupying site $s_i$. Everything else
in the expression is derived from it at evaluation time: $\gamma_{n(i)}$ and
the experiment's field give the Larmor frequency $\omega_{L,i}$, which with
$\tau$ gives the phase $\eta_i$. Neither $\omega_{L,i}$ nor $\eta_i$ is stored:
they are composites of a physical constant with experimental settings, and
storing them would duplicate state that must then be kept consistent.

Two departures from the printed papers, both deliberate:

- The nuclear gyromagnetic ratio is per-spin. Both papers carry a single
  $\gamma_n$, and hence a single Larmor phase common to all spins, which holds
  only for a single-isotope bath. In a mixed $^{13}$C / $^{29}$Si bath each spin
  precesses at the rate set by its own isotope, so $\gamma_{n(i)}$ enters the
  configuration alongside the site index. For a single-isotope bath every
  $\gamma_{n(i)}$ is equal and the expression reduces exactly to the published
  one.
- $\omega_L = +\gamma_n B_z$, not $-\gamma_n B_z$ as printed. The sign is
  physically meaningful — it enters $m_{i,z}$ and therefore the modulation depth
  — and the positive convention is the one used in the reference implementation
  that produced the published results. See Sec. 11.

### 4.2 Coherence and decoherence envelope

The coherence for experiment $e$ is the product over all $k$ spins, attenuated by
an empirical envelope:

$$f_e(\tau) = \frac{1}{2}\left(1 + \prod_{i=1}^{k} M_i(\tau)\right)\exp\left[-\left(\frac{\tau}{\lambda_e}\right)^{n_e}\right].$$

The envelope absorbs dephasing not captured by the explicitly modeled spins:
coupling to lattice impurities other than the modeled isotope (substitutional
nitrogen, for instance), and to the distant bath. Including it is necessary for
the forward model to describe experimental data in the sparse, noisy regime;
without it the modeled spins are forced to account for decay they did not cause.

$\lambda_e$ is **per-experiment**. A single decay constant cannot describe
experiments at different pulse number, since dynamical decoupling extends
coherence with $N$.

$n_e$ defaults to a **single global value shared across experiments**, fixed at
$n = 1$ (recovering the exponential envelope of the application paper). It may be
fixed at another value, or sampled — globally or per-experiment.

**Discrepancy, resolved.** The methods paper writes the envelope as an exponent
applied to the bracket, $\left(\tfrac12(1+\prod_i M_i)\right)^{-\tau/\lambda}$.
That form is not equivalent to the multiplicative envelope above, and as printed
it does not decay: the base lies in $[0,1]$ and $\lambda$ is restricted to
$[0,1]$, so a negative exponent drives the expression above unity and growing in
$\tau$. The multiplicative form of the application paper (Eq. 6) is correct and
is what this package implements, generalized from $e^{-\tau/\lambda}$ to
$e^{-(\tau/\lambda)^{n}}$.

### 4.3 Multiple experiments

Several dynamical decoupling experiments on the same defect — differing in pulse
number, field, or $\tau$ sampling — are fit jointly. The joint model shares the
physical object and separates the empirical ones:

- **Shared:** the spin configuration, meaning $k$, the occupied sites, and their
  hyperfine couplings. This is a property of the sample, not of the measurement.
- **Per-experiment:** $\lambda_e$, $\sigma_e$, the $\tau$ grid, $N_e$, $B_{z,e}$,
  and optionally $n_e$.

Per-experiment $\sigma_e$ is not merely a convenience: an experiment averaged for
longer has genuinely smaller noise, and forcing a common $\sigma$ would let the
noisier dataset distort the configuration inferred from the cleaner one.

Experiments need not share a $\tau$ grid or even its length.

---

## 5. Discretization: lattice sites and *ab initio* data

### 5.1 The site table

Candidate nuclear positions are restricted to crystallographic lattice sites,
with hyperfine couplings computed at each site by density functional theory.
This is the prior information that makes the inverse problem tractable: it
replaces an unbounded continuous search over coupling values with a random walk
over a finite, physically meaningful set.

Each site carries a position, an isotope identity (determined by the site — in
SiC a given site is either Si or C, never both), the nuclear gyromagnetic ratio
following from that identity, and a hyperfine tensor.

From the tensor, the secular components are

$$A_{\parallel} = A_{zz}, \qquad A_{\perp} = \sqrt{A_{xz}^2 + A_{yz}^2},$$

with the defect axis along $z$.

**Site selection.** Two thresholds filter the table. Sites where either
component exceeds a strong threshold are removed, as strongly coupled spins are
resolved by other means and are not the target of this method. Sites where
*both* components fall below a weak threshold are removed, as their modulation
is below the detection floor set by shot noise and sampling. The application
paper establishes this floor quantitatively: for $\tau_{\max} = 8\ \mu$s,
$B_z = 311$ G and 250 sampled $\tau$, no coupling below 7.8125 kHz is
recoverable regardless of averaging.

**Symmetry.** Distinct lattice sites related by crystal symmetry have identical
hyperfine couplings and are therefore indistinguishable in the data. The site
table records symmetry-equivalent groups. Two consequences: the number of spins
sharing a given coupling is bounded by the size of its symmetry group, and all
posterior comparison must be made on couplings rather than on site indices.

### 5.2 Occupancy constraint

No two spins may occupy the same lattice site. This is a hard constraint, not a
penalty: configurations violating it have zero prior mass. It bounds the
multiplicity of any given hyperfine value and it makes the discrete proposal
asymmetric, with consequences for the acceptance ratio given in Sec. 8.2.

### 5.3 Relaxing the *ab initio* constraint

Pinning couplings exactly to DFT values assumes those values are exact. The
methods paper's robustness study shows recovery degrades once errors exceed
about 1 kHz, which is within reach for strongly coupled spins even at 1%
relative DFT accuracy.

The constraint is therefore optionally relaxed. Each spin's coupling becomes its
table value plus a continuous offset,

$$A_{\parallel,i} = A_{\parallel}^{\text{DFT}}(s_i) + \delta_{\parallel,i}, \qquad A_{\perp,i} = A_{\perp}^{\text{DFT}}(s_i) + \delta_{\perp,i},$$

with Gaussian priors $\delta_{\parallel,i}\sim\mathcal{N}(0,s_{\parallel}^2)$ and
$\delta_{\perp,i}\sim\mathcal{N}(0,s_{\perp}^2)$ centred on the DFT value. The
prior width encodes the trusted accuracy of the electronic-structure
calculation.

Sampling then mixes a discrete walk over sites with a continuous walk over the
offset at the occupied site. Unlike the priors of Sec. 7.2, this prior is proper
and explicit, and enters the acceptance ratio directly.

When the constraint is not relaxed, all offsets are identically zero and the
model reduces exactly to the published one.

---

## 6. Parameter inventory

| Parameter | Kind | Domain | Per | Sampler |
|---|---|---|---|---|
| $k$ | discrete, trans-dimensional | $\{0,\dots,k_{\max}\}$ | configuration | RJMCMC |
| $s_i$ | discrete | site table, distinct | spin | discrete RWMH |
| $\delta_{\parallel,i},\delta_{\perp,i}$ | continuous | $\mathbb{R}$ | spin | continuous RWMH (optional) |
| $\lambda_e$ | continuous | bounded, positive | experiment | continuous RWMH |
| $n_e$ | continuous | bounded, positive | global or experiment | continuous RWMH (optional) |
| $\sigma_e$ | continuous | positive | experiment | continuous RWMH |

Known, never sampled: $N_e$, $B_{z,e}$, the $\tau$ grids, $D$, $\gamma_e$, and
the site table itself.

Determined, not sampled: the per-spin gyromagnetic ratio $\gamma_{n(i)}$. It is a
function of the site index, since the site fixes the isotope, so it changes
whenever a spin moves or is born but is never proposed independently. It belongs
to the configuration as a looked-up attribute rather than as a free parameter,
and a proposal that altered it without moving the spin would be unphysical.

---

## 7. Statistical model

### 7.1 Likelihood

Coherence signals are acquired by photon counting and averaging over many
repetitions. The counts are Poisson, but at the thousands-to-tens-of-thousands
of photons typical of these experiments the Gaussian approximation is accurate
and is what the model assumes.

For data $\mathbf{d}$ across experiments indexed by $e$ with points $j$:

$$\log\mathcal{L}(\mathbf{d}\mid\theta) = -\sum_e \frac{1}{2\sigma_e^2}\sum_j \left(d_{e,j} - f_e(\tau_{e,j})\right)^2.$$

All computation is in log space.

$\sigma_e$ plays a dual role that deserves stating plainly. It is a noise
estimate, but it also acts as a temperature on the likelihood surface: too small
and chains trap in local minima, too large and the posterior fails to
concentrate. The methods paper finds strong sensitivity and adopts
$\sigma^2 = 0.1$ as the best compromise. Treating $\sigma_e$ as sampled rather
than fixed lets the data inform this tradeoff, at the cost of a softer posterior.

**Alternative: Wasserstein-penalized likelihood.** An optional variant adds a
distributional penalty between predicted and observed signals,

$$\mathcal{L}_{\text{mod}}(\mathbf{d}\mid\theta) = (1-\zeta)\exp\left(-\frac{1}{2\sigma^2}\sum_j (d_j - f_j)^2\right) - \zeta\,W\!\left(f,\mathbf{d}\right),$$

with $W$ the Wasserstein distance and $\zeta$ a weighting parameter. The default
is $\zeta = 0$, recovering the Gaussian likelihood exactly. The likelihood is a
replaceable component: any function of predicted and observed signal may be
substituted, and the sampling machinery — including the tempered likelihoods
used in parallel tempering — operates on whatever is installed.

### 7.2 Priors

Priors enter through three distinct routes, and conflating them is a source of
error:

1. **The lattice constraint** is a hard prior: zero mass outside the site table,
   zero mass on configurations with doubly-occupied sites.
2. **The prior on $k$** is folded into the RJMCMC dimension-changing kernel
   $\gamma$ rather than appearing as a separate factor. The acceptance ratio of
   Sec. 8.3 therefore contains a likelihood ratio and a kernel ratio only. The
   effective prior on model dimension lives in the combinatorial structure of
   the birth and death proposals and in the cutoff $k_{\max}$.
3. **The offset prior** of Sec. 5.3, when the DFT constraint is relaxed, is
   proper, explicit, and appears directly in the acceptance ratio for offset
   moves.

---

## 8. Sampling

The posterior is high-dimensional, multimodal, and trans-dimensional. No single
algorithm handles all three. The scheme composes four, each applied to the
parameter block it suits.

Throughout, parameters are partitioned into an updated set $\mathbf{p}$ and a
held-fixed set $\mathbf{q}$. Every algorithm updates $\mathbf{p}$ and leaves
$\mathbf{q}$ untouched. This partitioning is the mechanism by which the
algorithms compose.

### 8.1 Continuous random-walk Metropolis–Hastings

Applied to $\lambda_e$, $n_e$, $\sigma_e$, and hyperfine offsets. A proposal is
drawn within radius $R$ of the current value, reflected at the domain boundary.
Reflection preserves symmetry, so the proposal ratio is unity and

$$\alpha = \min\left\{1,\ \frac{\mathcal{L}(\mathbf{p}^*)}{\mathcal{L}(\mathbf{p})}\cdot\frac{\pi(\mathbf{p}^*)}{\pi(\mathbf{p})}\right\},$$

where $\pi$ is the offset prior when sampling offsets and unity otherwise.

### 8.2 Discrete random-walk Metropolis–Hastings

Applied to lattice sites. One spin is selected and proposed to move to another
site within radius $R_{\text{spin}}$ of its current position. The interpretation
of a random walk is geometric: the discrete domain inherits real-space structure
from the lattice, so a radius means what it usually means.

The occupancy constraint makes this proposal **asymmetric**. Let
$\mathcal{N}_R(x)$ be the sites within $R$ of site $x$, and $\mathcal{O}$ the set
of sites occupied by spins other than the one being moved. The proposal draws
uniformly from $\mathcal{N}_R(x)\setminus\mathcal{O}$, so

$$\frac{r(z\to x)}{r(x\to z)} = \frac{\left|\mathcal{N}_R(x)\setminus\mathcal{O}\right|}{\left|\mathcal{N}_R(z)\setminus\mathcal{O}\right|},$$

and

$$\alpha = \min\left\{1,\ \frac{\mathcal{L}(\mathbf{p}^*)}{\mathcal{L}(\mathbf{p})}\cdot\frac{\left|\mathcal{N}_R(x)\setminus\mathcal{O}\right|}{\left|\mathcal{N}_R(z)\setminus\mathcal{O}\right|}\right\}.$$

Both neighbourhood counts must be evaluated with the moving spin excluded from
$\mathcal{O}$. Omitting this ratio — treating the proposal as symmetric — yields
a chain that still appears to recover correct configurations while targeting the
wrong distribution.

$R_{\text{spin}}$ is bounded below by the nearest-neighbour distance (1.54 Å in
diamond). Recovery is otherwise insensitive to it: the acceptance rule, not the
neighbourhood size, determines where walkers settle.

### 8.3 Reversible-jump MCMC

Applied to $k$. A dimension-changing kernel $\gamma$ proposes a birth or a death:

- **Birth:** $k \to k+1$; a new spin is placed at a site drawn from the
  unoccupied admissible sites.
- **Death:** $k \to k-1$; one of the $k$ existing spins is removed uniformly at
  random.

$$\alpha = \min\left\{1,\ \frac{\mathcal{L}(\mathbf{d}\mid f_{k^*}(\mathbf{p}^*))}{\mathcal{L}(\mathbf{d}\mid f_{k}(\mathbf{p}))}\cdot\frac{\gamma(k^*, k)}{\gamma(k, k^*)}\right\}.$$

Because the prior on $k$ is carried by $\gamma$ (Sec. 7.2), no separate prior
ratio appears. The kernel is a replaceable component; substituting one that
encodes isotopic abundance changes the effective prior on bath size without
touching the sampler.

Dimension matching is trivial here because births draw from the same discrete
domain the model already uses — no auxiliary variables or Jacobian are required.
That would change if births proposed continuous couplings directly.

### 8.4 Parallel tempering

Applied to fixed-dimension exploration of the rugged configuration landscape.
$J$ replicas run at inverse temperatures $\beta_0 = 1 > \beta_1 > \dots >
\beta_{J-1}$, with $\beta_j = 2^{-j}$ under zero-based indexing, so replica 0 is
the cold chain sampling the true posterior.

Each replica advances by an inner update — which may itself be a composite of
continuous and discrete moves — against the tempered likelihood
$\mathcal{L}^{\beta_j}$. A pair of replicas $(a,b)$ is then drawn uniformly and a
swap attempted with

$$\alpha_{\text{PT}} = \min\left\{1,\ \frac{\mathcal{L}_a(\mathbf{p}_b)\,\mathcal{L}_b(\mathbf{p}_a)}{\mathcal{L}_a(\mathbf{p}_a)\,\mathcal{L}_b(\mathbf{p}_b)}\right\} = \min\left\{1,\ \exp\left[(\beta_a-\beta_b)\left(\log\mathcal{L}(\mathbf{p}_b)-\log\mathcal{L}(\mathbf{p}_a)\right)\right]\right\}.$$

The second form is what is computed: it is numerically stable and makes clear
that only the untempered log-likelihoods and the two inverse temperatures are
needed.

**Only the cold chain is retained.** Hot replicas exist to ferry configurations
across barriers; their samples do not target the posterior and are discarded
when the tempering block ends.

### 8.5 Hybrid composition

The sampler cycles deterministically through a user-specified schedule of
blocks, each naming an algorithm, a parameter block, a step count, and its
hyperparameters. A representative cycle: continuous RWMH on $\lambda$ with all
else fixed; then RJMCMC on $k$; then parallel tempering over sites at fixed $k$.
The cycle repeats until the total step budget is exhausted, and the outputs of
every block are concatenated into a single trace.

This is a systematic-scan Metropolis-within-Gibbs composition. Each block leaves
the target invariant, so the composition does; the composed chain is not
reversible, which is harmless for estimation but means reversibility-based
diagnostics do not apply to the concatenated trace.

Which algorithm updates which parameters is a user decision, not a fixed
property of the model. The schedule is the primary interface.

### 8.6 Ensembles

Several independent chains — differing in seed and initialization, never
exchanging information — are run per dataset. Post-burn-in samples pool into a
single posterior, and the spread *between* ensembles is the primary convergence
diagnostic: agreement is evidence of convergence, disagreement reveals that
individual chains remain trapped. Because ensembles are independent, they
parallelize trivially.

Burn-in is discarded before any posterior summary. The published runs use 5
ensembles, 25,000 steps, 10,000 discarded.

---

## 9. Posterior summaries

The posterior is over configurations of varying dimension, which rules out naive
per-parameter averaging: there is no stable correspondence between "spin 3" in
one sample and "spin 3" in another. Summaries are therefore defined on
couplings, matched within a tolerance that absorbs numerical differences between
symmetry-related sites.

**Detection rate.** For a reference spin $s_i$ (simulated ground truth, or
experimentally confirmed) and posterior samples $B^{(j)}$, $j = 1\dots M$, let
$\mathbb{I}_i^{(j)}$ indicate whether $s_i$ appears in sample $B^{(j)}$, matching
on hyperfine couplings. Then

$$R_i = \frac{1}{M}\sum_{j=1}^{M}\mathbb{I}_i^{(j)}, \qquad R = \frac{1}{n}\sum_{i=1}^{n}R_i.$$

$R_i = 1$ means the spin appears in every posterior sample; $R_i = 0$ means it
never appears.

**Dimension discrepancy.** The absolute difference between the mode of the
posterior over $k$ and the true number of spins. Zero indicates the inferred
model dimension is correct.

**Sample-averaged false-absence rate.** For $S^*$ the modal configuration of
size $n$,

$$FP = \frac{1}{nM}\sum_{s\in S^*}\sum_{j=1}^{M}\mathbb{I}_{s\notin B^{(j)}},$$

the fraction of posterior samples in which spins of the modal configuration are
absent. This is not a false-positive rate in the classification sense, and is
not to be reported as one.

All three are conventionally grouped by coupling magnitude
$\sqrt{A_{\parallel}^2 + A_{\perp}^2}$, since recovery quality depends strongly
on it.

**Trajectory diagnostics.** Residual error, $k$, and individual parameters are
tracked against trace step, with burn-in marked and ensembles overlaid.

A note on vocabulary: *walker* refers to an individual nuclear spin moving over
lattice sites — the object with radius $R_{\text{spin}}$ that cannot share a
site. The sequence of states produced by the sampler is the *trace*. The papers
use "walker steps" for what is here called trace steps.

---

## 10. Domain of validity

- **Neglected nuclear–nuclear coupling.** The analytic model is equivalent to
  CCE-1 and exact only when nuclear spins do not interact appreciably. This holds
  for dilute baths at short $\tau$ and low pulse number. NV centers in diamond
  are well described; divacancies in SiC are converged only at CCE-2, and require
  a numerical forward model.
- **Secular approximation**, requiring the electronic splitting to dominate.
- **Gaussian noise**, requiring sufficient photon counts.
- **Weak-coupling non-identifiability.** Below roughly 25 kHz, recovery degrades
  substantially and does not improve with more data at fixed experimental
  settings. This is a property of the inverse problem, not of the algorithm:
  many configurations produce indistinguishable signals, and additional samples
  reinforce the ambiguity rather than resolving it. Escaping it requires
  measurements that change the structure of the problem — different magnetic
  fields, for instance — not more of the same measurement. The posterior makes
  this explicit through broad, multimodal support, and reporting should preserve
  that rather than collapsing it to a point estimate.
- **DFT accuracy.** Recovery is robust to coupling errors up to about 1 kHz and
  degrades beyond. Since relative DFT error translates to larger absolute error
  for strongly coupled spins, accuracy is expected to fall off at the strong end
  unless the constraint is relaxed as in Sec. 5.3.

---

## 11. Conventions requiring confirmation

Recorded so that later disagreement with published results can be traced.

1. **Sign of $\omega_L$.** Both papers print $\omega_L = -\gamma_n B_z$; the
   reference implementation uses $+\gamma_n B_z$ with PyCCE's positive
   gyromagnetic ratios. This specification follows the implementation, on the
   grounds that it produced the published results. The sign affects modulation
   depth through $m_{i,z}$ and is not cosmetic.
2. **Envelope form.** Resolved in favour of the application paper's
   multiplicative form; see Sec. 4.2.
3. **Tempering ladder indexing.** The methods paper states both $\beta_1 = 1$ and
   $\beta_j = 2^{-j}$, which are inconsistent. Resolved as zero-based:
   $\beta_j = 2^{-j}$ with $\beta_0 = 1$.
4. **Per-spin $\gamma_{n(i)}$.** A generalization beyond both papers, required
   for mixed-isotope baths and reducing exactly to the published expression for a
   single isotope.
5. **Redundant distance column.** The site table carries both a radial distance
   and Cartesian coordinates; they agree to roughly four decimals but not
   exactly. The stored value is used as given rather than recomputed.

---

## 12. References

1. A. N. Poteshman, J. Yun, T. H. Taminiau, G. Galli, *Trans-dimensional
   Hamiltonian model selection and parameter estimation from sparse, noisy data*,
   arXiv:2506.18802; Quantum (2026).
2. A. N. Poteshman, M. Onizhuk, C. Egerstrom, D. P. Mark, D. D. Awschalom,
   F. J. Heremans, G. Galli, *High-throughput spin-bath characterization of spin
   defects in semiconductors*, arXiv:2506.19259; Phys. Rev. Applied **24**,
   054048 (2025).
3. T. H. Taminiau *et al.*, *Detection and control of individual nuclear spins
   using a weakly coupled electron spin*, Phys. Rev. Lett. **109**, 137602 (2012)
   — origin of the single-spin modulation expression.
4. P. J. Green, *Reversible jump Markov chain Monte Carlo computation and
   Bayesian model determination*, Biometrika **82**, 711 (1995).
5. M. Onizhuk, G. Galli, *PyCCE: A Python package for cluster correlation
   expansion simulations of spin qubit dynamics* (2021) — numerical forward model
   and isotope constants.
6. I. Takács, V. Ivády *et al.* — *ab initio* hyperfine couplings underlying the
   site table.
