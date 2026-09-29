# Nuclear Spin Recovery

**Status:** This software is still under development, and the tutorials are a work in progress! All data supporting arXiv:2506.19259 and arXiv:2506.18802 are in the process of being migrated permanently to Qresp. You are welcome to email **poteshman@uchicago.edu** for software support or access to data.  


## Overview

This package provides tools reconstructing nuclear spin environments from experimental coherence data using hybrid MCMC techniques.  

## Dependencies

To run this software, the following Python packages are required:

- `python`  (version >= 3.9)
- `numpy`  (version >= 1.16)
- `pandas`  
- `pycce`  (version >= 1.1)

## Installation

The recommended way to install:

```bash
git clone <this-repo-url>
cd nuclear-spin-recovery
pip install .

```

## Development History

- **`main`** — the current implementation. A ground-up rewrite (2026) with an
  object-oriented sampler architecture, a written model specification and a
  calibrated test suite. Its history carries the original research
  implementation of 2024–2025 as an ancestor, so `git log` shows both: the 89
  commits that produced the results in arXiv:2506.19259 and arXiv:2506.18802,
  and the rewrite on top of them. See [REWRITE.md](REWRITE.md).
- **`old_rjmcmc_code`** — the research code as it stood in June 2026, kept for
  reference. It is the most complete of the pre-rewrite branches and contains
  the history of the earlier ones.

### Working on the code

```bash
pip install -e ".[dev]"
pytest                   # full suite, ~4 min
pytest -m "not slow"     # fast subset, ~1 s
```

Notebooks in `notebooks/` are paired with `.py:percent` files via jupytext.
Edit either; only the `.py` is committed.
