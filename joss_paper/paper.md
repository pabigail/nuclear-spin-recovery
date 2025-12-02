---
title: "nuclear-spin-recovery: A Python package for hybrid MCMC sampling for nuclear spin reconstruction"
tags:
  - Python
  - spin-defects
  - nuclear spin baths
  - coherence simulations
  - dynamical decoupling
authors:
  - name: Abigail N. Poteshman
    orcid: 0000-0002-4873-4826
    affiliation: "1, 2"
  - name: Giulia Galli
    orcid: 0000-0002-8001-5290
    affiliation: "2, 3, 4"
affiliations:
  - name: Committee on Computational and Applied Mathematics, University of Chicago, USA
    index: 1
  - name: Materials Science Division, Argonne National Laboratory, USA
    index: 2
  - name: Department of Chemistry, University of Chicago, USA
    index: 3
  - name: Pritzker School of Molecular Engineering, University of Chicago, USA
    index: 4
date: 1 December 2025
bibliography: paper.bib
---

# Summary
We present a software package that implements a flexible Bayesian framework for reconstructing and obtaining parameters in ill-posed inverse problems and multi-modal optimization problems arising in the analysis of coherence properties of nuclear spin baths surrounding individual spin defects in semiconductors (e.g., nitrogen-vacancy (NV) centers in diamond, divacancies in silicon carbide (SiC)) in quantum sensing and spin-physics experiments.
Key features of our software package include joint sampling of continuous and discrete parameters, hybridization of several MCMC techniques include trans-dimensional model selection, modular forward-model interfaces to simulate coherence signals under a range of theoretical approaches, customizable error models and likelihood functions to support a range of experimental conditions and noise sources, and extensibility to arbitrary nuclear spin baths and semiconductors. This tool is designed for experimentalists seeking posterior distributions of nuclear spin baths from sparsely sampled standard coherence signals and for theorists developing optimized pulse sequences or adaptive data-assimilation schemes tailored to specific spin defects, nuclear baths, and experimental setups.  

# Statement of need

Experimental characterization of nuclear spin environments around solid-state spin defects, such as NV centers in diamond, is a central problem in quantum sensing and quantum information science. Extracting the configuration of surrounding nuclear spins from measurements—typically dynamical decoupling or coherence experiments—is an inverse problem that is often ill-posed, with multiple configurations producing similar signals.

Bayesian inference provides a natural framework for this task, but existing MCMC packages face limitations for nuclear spin bath reconstruction:

- **Mixed discrete and continuous parameters:** Nuclear spin systems involve both continuous parameters (e.g., hyperfine couplings, nuclear positions) and discrete parameters (e.g., isotopic identity, number of spins). Whether a parameter is modeled as discrete or continuous depends on available knowledge: highly accurate ab initio calculations or measurements may justify discrete modeling, while poorly known parameters are better treated as continuous. Few general-purpose MCMC frameworks support such mixed spaces directly.

- **Hybrid MCMC algorithm requirements:** Efficient sampling of complex, multi-modal posteriors often requires interleaving different MCMC techniques (e.g., Hamiltonian Monte Carlo for continuous parameters, Gibbs sampling or Random Walk Metropolis-Hastings for discrete parameters). Most existing packages make hybridization cumbersome or impractical.

- **Domain-specific modeling needs:** Forward models for coherence simulations (e.g., cluster-correlation expansion) are computationally intensive and non-linear, and standard MCMC packages do not easily integrate physics-informed likelihoods or custom constraints.

`nuclear-spin-recovery` addresses these gaps by providing a Python framework for hybrid MCMC sampling that:

- Supports joint sampling of discrete and continuous parameters informed by domain knowledge.
- Allows seamless hybridization of MCMC algorithms for different parameter types.
- Integrates with physics-based forward models and custom likelihood functions.
- Provides posterior distributions over experimentally relevant quantities, even in ill-posed regimes.
- Is extensible to new materials, spin baths, and experimental protocols.

This package enables both experimentalists and theorists to infer, quantify, and design nuclear spin environments effectively.

### Comparison of Existing MCMC Packages

Table: Feature comparison of general MCMC software packages with `nuclear-spin-recovery`, highlighting support for continuous, discrete, mixed, hybrid, and transdimensional sampling.

| Package                   | Continuous Parameters | Discrete Parameters | Mixed (Cont+Disc) | Transdimensional MCMC | Easy Hybridization |
|---------------------------|-----------------------|----------------------|--------------------|------------------------|---------------------|
| PyMC                      | $\checkmark$          | $\checkmark$         | $\times$           | $\times$               | $\times$            |
| Stan                      | $\checkmark$          | $\times$             | $\times$           | $\times$               | $\times$            |
| emcee                     | $\checkmark$          | $\times$             | $\times$           | $\times$               | $\times$            |
| pymc3                     | $\checkmark$          | $\checkmark$         | $\times$           | $\times$               | $\times$            |
| TensorFlow Probability    | $\checkmark$          | $\checkmark$         | $\times$           | $\times$               | $\times$            |
| nuclear-spin-recovery     | $\checkmark$          | $\checkmark$         | $\checkmark$       | $\checkmark$           | $\checkmark$        |

# Software overview
The software is designed with two goals:
- allow experimentalists to reconstruct nuclear spin configurations from standard dynamical decoupling experiments quickly
- allow theorists to design, implement, and integrate new forward models, MCMC algorithms, experiments, and error/likelihood functions.



# Acknowledgements

We thank Mykyta Onizhuk, F. Joseph Heremans, Benjamin Pingault, Christopher Egerstrom, Daniel Sanz-Alonso, and Nathan Waniorek for helpful conversations during the development of this project.
A.N.P acknowledges support from the DOE CSGF under Award No. DE-SC0022158 and AFOSR under ???.
G.G. acknowledges support from the Midwest Integrated Center for Computational Materials (MICCoM) as a part of the Computational Materials Sciences Program funded by the U.S. Department of Energy.
This research used resources of the University of Chicago Research Computing Center.

# References

