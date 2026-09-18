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

This repository holds two implementations of the same physics.

- **`main`** — Original research implementation (2024–2025). The code that
  produced the results in arXiv:2506.19259 and arXiv:2506.18802.
- **`claude-rewrite`** — Production rewrite (2026–), with an object-oriented
  sampler architecture, a written model specification, and a calibrated test
  suite. Active development happens here. See [REWRITE.md](REWRITE.md).

The rewrite branch carries both histories: `git log` shows the original 89
commits alongside the rewrite.

### Working on the rewrite branch

```bash
pip install -e ".[dev]"
pytest                   # full suite, ~4 min
pytest -m "not slow"     # fast subset, ~1 s
```

Notebooks in `notebooks/` are paired with `.py:percent` files via jupytext.
Edit either; only the `.py` is committed.
