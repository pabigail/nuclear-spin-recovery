# Phase 6 build order — PyCCE forward model

Companion to `model-specification.md` and `test-plan.md`. Covers one question:
**how does a numerical CCE forward model, computed by PyCCE, sit behind the
existing `ForwardModel` interface** — optionally installed, identical to the
analytic model at CCE-1, and extensible to higher orders?

Phases 1–5 are implemented. The design decisions are settled and recorded in
Sec. 8; no code or tests have been written yet.

---

## 1. Requirements

1. A forward model that computes coherence from `(state, expset, site_table)`
   using PyCCE.
2. PyCCE is an optional dependency. The package installs, imports and runs the
   analytic model without it.
3. At CCE-1 the PyCCE model returns the same numerical result as
   `AnalyticCCE1`.
4. Orders above one are supported, and take the extra information they need.

---

## 2. What was checked before planning

PyCCE is not installed in the development environment, so the PyCCE facts here
come from reading the source of the 1.1.1 wheel, not from running it. The
equivalence figures come from a scratch script that propagates one nuclear spin
exactly, using the Hamiltonian as PyCCE's source writes it, and compares the
result against `single_spin_modulation`.

### 2.1 CCE-1 equivalence holds for even pulse numbers only

Largest difference between the exact propagation and the analytic form over 400
values of tau, at 311 G, for three coupling pairs:

| N | ms = −1 | ms = +1 |
|---|---|---|
| 0 | 0.9 – 2.0 | 0.95 – 2.0 |
| 1 | 1.9 – 2.0 | 1.9 – 2.0 |
| 2 | 2e-15 – 6e-15 | 0.015 – 0.32 |
| 3 | 1.7 – 2.0 | 1.7 – 2.0 |
| 8 | 7e-15 – 1.4e-14 | 0.17 – 1.3 |
| 16 | 1.4e-14 – 2.9e-14 | 0.59 – 2.0 |
| 32 | 3.5e-14 – 7.6e-14 | 1.5 – 2.0 |

The analytic expression is the even-N CPMG form. For N = 0 and odd N it does
not describe the sequence, and the two models will disagree at order one.
`Experiment` currently accepts any non-negative integer N.

The imaginary part of the exact single-spin result is below 1e-14 everywhere.

### 2.2 The qubit must be ms = 0 / ms = −1

PyCCE writes the nuclear Zeeman term as `-gyro * B . I`. With the package's
convention `omega_L = +gyro * b_z` (spec Sec. 11.1), the analytic frequency
`sqrt((A_par + omega_L)**2 + A_perp**2)` is reproduced only when the second
qubit level is ms = −1. With ms = +1 the comparison fails, as the table shows.

### 2.3 PyCCE does not import on NumPy 2

Both PyCCE 1.1.1 and upstream master use `np.unicode_` in class bodies that
execute at import time (`bath/cube.py`, `bath/array.py`, `bath/cell.py`,
`io/xyz.py`). NumPy 2.0 removed that alias. The development environment has
NumPy 2.5.3. The `old_rjmcmc_code` branch worked around this by restoring the
alias before importing PyCCE.

### 2.4 Units already match

PyCCE works in kHz, ms, G and Angstrom with angular gyromagnetic ratios in
rad / (ms G), the same system as `units.py`. No conversion is needed. Passing
`as_delay=True` makes PyCCE's time axis the interpulse half-spacing, which is
this package's `tau` (PyCCE otherwise divides total time by `2 * N`).

### 2.5 Dependencies

PyCCE requires `numpy`, `scipy`, `ase`, `pandas` and `numba`. The last two are
new to this project. Whether the current numba release supports NumPy 2.5 has
not been tested.

---

## 3. Optional install

- A `pycce` extra in `pyproject.toml`, installed with `pip install .[pycce]`.
  The core dependency list is unchanged.
- A new module `forward/pycce_backend.py` holding `PyCCEForward(ForwardModel)`.
  The module itself imports without PyCCE. PyCCE is imported when the class is
  constructed, and a missing install raises an `ImportError` that names the
  extra.
- `PyCCEForward` is exported from `forward/` and from the package namespace, so
  the wiring checks in `tests/unit/test_imports.py` pass with or without PyCCE.
- The README dependency list, which still names `pycce` and `pandas` as
  required, is corrected.

---

## 4. CCE-1 backend

`PyCCEForward(envelope, order=1, ...)` returns an array of shape
`(n_replicas, n_points)`, like every `ForwardModel`.

For each replica and each experiment:

1. Build a bath from the active slots of the state. Positions come from
   `site_table.positions`. The hyperfine tensor is set directly, with
   `A_zz = a_par` and `A_xz = A_zx = a_perp`, both including the `dA` offsets.
   A non-zero tensor stops PyCCE from substituting its own point-dipole values.
2. Take gyromagnetic ratios from `site_table.gyro`, through custom spin types,
   not from PyCCE's lookup by isotope name. The test fixtures use 6.7283 where
   PyCCE's table has 6.7282853…, and the equivalence must hold for whatever the
   table carries.
3. Run conventional CCE with `order=1`, `pulses=N`, `as_delay=True`, the field
   `[0, 0, b_z]`, and the qubit levels of Sec. 2.2, on that experiment's tau
   grid.
4. Apply the same `Envelope` object the analytic model uses:
   `0.5 * (1 + Re(L) * envelope)`.

A state with k = 0 does not call PyCCE and returns the envelope term alone.

---

## 5. Higher orders

CCE-2 and above need information that CCE-1 does not:

- **Direction of the perpendicular hyperfine component.** `SiteTable` keeps
  only the magnitude `hypot(A_xz, A_yz)`. At CCE-1 only the magnitude enters.
  Once nuclear pairs couple through the dipolar interaction, the azimuth of
  each spin's perpendicular component relative to the pair axis changes the
  result. The table needs `A_xz` and `A_yz` as an optional field; the Ivady
  file already carries them, and `from_ase` computes them before discarding
  them.
- **Positions in the defect frame.** Origin at the defect and z along its axis.
  `from_ase` produces this. It has not been confirmed for `nv-2.txt`.
- **Cluster settings.** `order`, `r_dipole`, and optionally `n_clusters`, as
  constructor arguments.
- **Central spin.** Spin, qubit levels and zero-field splitting. These differ
  between the NV centre and the SiC divacancy, which the specification names
  as the case that requires CCE-2 (Sec. 10).

`PyCCEForward` with `order >= 2` raises a clear error when the site table lacks
the extra fields, and does not fall back to an assumed azimuth.

Two choices fix what the higher-order model computes (Sec. 8):

- **Clusters are built from the state's spins only.** The bath handed to PyCCE
  is the k active spins of the configuration. Everything else — unresolved
  spins, other impurities — stays in the envelope, as in the analytic model.
- **Offsets act along the site's existing azimuth.** `dA_perp` is a scalar
  offset on a magnitude. The perpendicular vector `(A_xz, A_yz)` is rescaled to
  the magnitude `a_perp + dA_perp` with its direction unchanged, so a zero
  offset reduces exactly to the table's tensor.

---

## 6. Configuration

`_build_model()` in `config.py` returns `AnalyticCCE1` unconditionally and notes
that it becomes a section when a second backend exists. It becomes a `[model]`
section:

```toml
[model]
backend = "analytic"   # or "pycce"
order = 1
r_dipole = 8.0         # Angstrom, read only when order >= 2
```

A configuration that asks for the PyCCE backend without PyCCE installed fails
when the configuration is loaded, not when the first job starts.

---

## 7. Tests

**Without PyCCE.** A subprocess test that blocks `pycce` from importing,
mirroring `test_importing_post_does_not_import_matplotlib`:

- the package imports;
- the analytic model runs;
- constructing `PyCCEForward` raises the error that names the extra.

**CCE-1 equivalence.** Skipped when PyCCE is absent. `PyCCEForward(order=1)`
against `AnalyticCCE1`, to an absolute tolerance of 1e-10 on coherence, on the
tiny site table and on a slow case from `nv-2.txt`, covering:

- even pulse numbers N >= 2 only (Sec. 2.1);
- two experiments with different grids, grid lengths and fields;
- a mixed 13C / 29Si bath;
- non-zero `dA_par` and `dA_perp` offsets;
- k = 0;
- several replicas with different configurations;
- tau values near resonance, where the analytic denominator vanishes.

**CCE-2.**

- It equals CCE-1 when `r_dipole` is below the smallest pair distance.
- For two spins it matches an exact 4×4 diagonalisation written in the test,
  since CCE-2 is exact for a two-spin bath.
- It is unchanged by the order of the slots, and by a joint rotation of all
  positions and tensors about z.
- It changes when one spin's azimuth changes at fixed `(a_par, a_perp)`.
- A non-zero `dA_perp` changes the magnitude of the perpendicular component and
  leaves its azimuth where it was.
- It raises when the site table has no perpendicular components.

**CI.** Two environments, one with the extra and one without.

**Performance.** Measured only after correctness. The analytic model is one
vectorised pass over replicas, spins and points. PyCCE rebuilds a simulator per
replica, per experiment, per MCMC step, and numba compiles on first use. If
that is too slow for sampling, cluster contributions are cached by site, so a
single-spin move recomputes only the clusters containing that spin. Non-zero
`dA` offsets defeat a cache keyed by site alone.

---

## 8. Decisions

1. **Equivalence is to a numerical tolerance.** Bitwise equality is not
   achievable: PyCCE diagonalises numerically and the analytic model is a
   closed form. The CCE-1 test uses an absolute tolerance of 1e-10 on
   coherence.
2. **The comparison with the analytic model is tested at even N >= 2 only.**
   The models disagree at N = 0 and odd N (Sec. 2.1), where the analytic
   expression does not describe the sequence. That behaviour of the analytic
   model is outside this phase.
3. **NumPy 2 is handled with the alias.** `np.unicode_ = np.str_` is restored
   immediately before PyCCE is imported, inside the lazy import and nowhere
   else. NumPy is not pinned.
4. **Clusters are built from the state's spins only.** No background bath; the
   envelope stands in for everything that is not in the configuration.
5. **Offsets act along the site's existing azimuth** at higher order (Sec. 5).
6. **Symmetry groups are deferred.** `SiteTable.symmetry_groups` treats sites
   with equal `(a_par, a_perp)` as indistinguishable, which is no longer true
   at CCE-2 and affects the detection metrics in `post/`. It is left as it is
   in this phase and revisited later.
7. **This phase is conventional CCE only,** with a maximally mixed bath. It is
   deterministic, which the sampler requires.

---

## 9. Out of scope

- **Generalised CCE.** It may be wanted later. It would be a separate
  `ForwardModel` beside `PyCCEForward`, not an option on it: it needs the full
  hyperfine tensor and the central-spin Hamiltonian, which conventional CCE
  does not.
- **Second-order corrections and bath-state sampling.** Sampling bath states
  makes the likelihood noisy, which the sampler cannot use as it stands.
- **The analytic model at N = 0 and odd N** (Sec. 8, item 2).
- **Symmetry groups at CCE-2** (Sec. 8, item 6).
