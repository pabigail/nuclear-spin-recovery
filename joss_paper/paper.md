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
`nuclear-spin-recovery` is a Python package that uses hybrid Markov chain Monte Carlo (MCMC) methods to reconstruct the arrangement and properties of nuclear spins around point defects in solids based on experimentally-measured coherence data.
In many quantum sensing and spin-physics experiments, these nuclear environments influence the beahvior of the spin in the point-defect, but they cannot be observed directly.
Instead, researchers rely on indirect measurements through the coherence signal of a point-defect that make the detection of nuclear spins difficult, often with many different nuclear spin configurations capable of producing a coherence signal that has a high likelihood of producing the experimental coherence data, especially when the coherence data is noisy or sparse.

This software package produces a flexible Bayesian framework for solving this inverse problem. 
It combines several MCMC techniques to sample both continuous parameters (such as coupling interactions among spins) and discrete parameters (such as lattice positions of spins), as well as models of different dimensions (such as nuclear spin configurations with different numbers of nuclear spins). 
The package is designed to be highly extensible: users can incorporate different forward models based on different levels of cluster-correlation expansion theory, choose among various sampling algorithms, and define custom likelihood or errorfunctions to suit their experimental setup. 

By producing full posterior distributions rather than single best-fit values, `nuclear-spin-recovery` allows users to quantify uncertainty and explroe multiple plausible nuclear configurations, even in ill-posed regimes. The package serves both experimentalists, who can use it to reconstruct nuclear spin environments from common dynamical decoupling experiments, and theorists, who can apply it as a general tool for nonlineiear, non-convex optimization for designing pulse sequences and experiments tailored to different recovery criteria or to specific spin-bath structures.

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

| Package       | Continuous Parameters | Discrete Parameters | Mixed (Cont+Disc) | Easy Hybridization |
|---------------|--------------------|-------------------|-----------------|-------------------------------------|
| PyMC          | ✅                  | ✅                 | ❌               | ❌                             |
| Stan          | ✅                  | ❌                 | ❌               | ❌                                   |
| emcee         | ✅                  | ❌                 | ❌               | ❌                                   |
| pymc3         | ✅                  | ✅                 | ❌               | ❌                                   |
| TensorFlow Probability | ✅          | ✅                 | ❌               | ❌                                   |
| nuclear-spin-recovery | ✅          | ✅                 | ✅               | ✅                                   |

*Table notes:*  
- “Mixed” indicates ability to sample continuous and discrete parameters jointly in the same framework.  
- “Easy Hybridization” indicates whether it is straightforward to interleave different MCMC techniques for different parameters.



# Software overview
The software is designed with two goals:
- allow experimentalists to reconstruct nuclear spin configurations from standard dynamical decoupling experiments quickly
- allow theorists to design, implement, and integrate new forward models, MCMC algorithms, experiments, and error/likelihood functions.



# Acknowledgements

We thank Mykyta Onizhuk, F. Joseph Heremans, Benjamin Pingault, Christopher Egerstrom,  
Daniel Sanz-Alonso, and Nathan Waniorek for helpful conversations during the development of this project.

A.N.P acknowledges support from the DOE CSGF under Award No. DE-SC0022158 and AFOSR under ???.

G.G. acknowledges support from the Midwest Integrated Center for Computational Materials (MICCoM) as a part of the Computational Materials Sciences Program funded by the U.S. Department of Energy.

This research used resources of the University of Chicago Research Computing Center.

# References

