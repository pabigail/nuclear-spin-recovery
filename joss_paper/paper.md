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
Key features of our software package include joint sampling of continuous and discrete parameters, hybridization of several MCMC techniques include trans-dimensional model selection, modular forward-model interfaces to simulate coherence signals under a range of theoretical approaches, customizable error models and likelihood functions to support a range of experimental conditions and noise sources, and extensibility to arbitrary nuclear spin baths and semiconductors. 
This tool is designed for experimentalists seeking posterior distributions of nuclear spin baths from sparsely sampled standard coherence signals and for theorists developing optimized pulse sequences or adaptive data-assimilation schemes tailored to specific spin defects, nuclear baths, and experimental setups.  

# Statement of need

Experimental characterization of nuclear spin environments around solid-state spin defects, such as NV centers in diamond, is a central problem in quantum sensing and quantum information science. 
On the one hand, these nuclear spins can be a main source of decoherence of quantum information for the spin-defect [@zhao2012decoherence; @seo2016quantum]. 
On the other hand, these nuclear spins can serve as local quantum registers or quantum memories, provided that their spatial configuration and couplings to the spin-defect are known [@bradley2019ten; @bourassa2020entanglement]. 
While direct experimental characterization of the nuclear spin environment is possible through, for example, correlated sensing [@van2024mapping], often experimentalists seek an initial rough characterization of the nuclear spin environment of individual spin-defects before deciding whether to use a specific spin-defect for further experiments [@poteshman2025high]. 
As a result, there is a need for nuclear spin-bath characterization methods from sparse, noisy coherence data that can be obtained relatively quickly. 
In such experiments, the inverse problem is ill-posed, where multiple nuclear spin bath configurations can yield the same experimental data in the noisy, sparse regime. 
As a result, direct machine learning approaches [@jung2021deep; @varona2024automatic] to this inverse problem require an order of magnitude more experimental data than the hybrid MCMC methods that can be implemented with this software package [@poteshman2025trans]. 
We also note that approaches based on variational Bayesian inference have also been successfully applied to this inverse problem [@belliardo2025multi], but these approaches have not yet incorporated spatial nuclear information or flexible interfacing with different forward models of coherence. 

Bayesian inference provides a natural framework for this task, but existing general MCMC packages (\autoref{mcmc-table}) in terms of both the statistical and domain-specific features necessary for nuclear spin bath reconstruction. 
We need joint sampling from mixed discrete and continuous parameters, since a nuclear spin may be modeled as having both continuous parameters, such as hyperfine coupligns, and discrete parameters, such as discrete lattice positions. 
Similarly, experimental settings may have continuous features, such as magnetic field, or discrete features, such as the number of $\pi$-pulses. 
Furthermore, whether specific parameters are modeled as continuous or discrete can be highly dependent on the available prior knowledge or experimental setup. 
For example, whether the hyperfine couplings of a nuclear spin are modeled as discrete or continuous depends on whether highly accurate _ab initio_ calculations are availble, and specific experimental equipment may model time as either continuous or discrete, depending on resolution limits. 
However, we would need flexbility to sample over discrete and continuous parameter spaces jointly, and few general-purpose MCMC frameworks support such mixed parameter spaces directly.

Efficient sampling of complex, multi-model posteriors often requires interleaving different MCMC techniques (e.g., random walk Metropolis Hastings, Gibbs sampling, parallel tempering, reverse jump MCMC). 
Most existing packages make hybridization cumbersome or impractical. 
Furthermore, nuclear spin bath characterization is inherently a multi-dimensional model selection problem, since we do not know how many nuclear spins are in the bath _a priori_, and existing MCMC softwares do not have support or easy extensbility to implement trans-dimensional MCMC-based model selection techniques, such as reverse jump MCMC.

Furthermore, the forward problem of simulating a coherence signal from a specific nuclear spin bath configuration with specific experimental settings is non-trivial and different approaches vary vastly in their computational expense.
The standard approach to simulate coherence signals is based on the cluster correlation expansion method [@yang2008quantum; @yang2020longitudinal], but the level of expansion required to obtain computationally converged coherence signals requires both theoretical consideration of the materials and nuclear spins being simulated and numerical convergence of hyperparameters.
These different levels of theory are implemented in `pyCCE` [@onizhuk2021pycce], and this software package is designed so that users can flexibly and interchangeably access different levels of theory in the forward model for the MCMC simulations.  

`nuclear-spin-recovery` addresses these shortocmings of these general MCMC software packages by providing a domain-specific Python framework for hybrid MCMC sampling that:

- Supports joint sampling of discrete and continuous parameters informed by domain knowledge.
- Allows seamless hybridization of MCMC algorithms for different parameter types.
- Integrates with physics-based forward models and custom likelihood functions.
- Provides posterior distributions over experimentally relevant quantities, even in ill-posed regimes.
- Is extensible to new materials, spin baths, and experimental protocols.

This package enables both experimentalists and theorists to infer, quantify, and design nuclear spin environments effectively.

### Comparison of Existing MCMC Packages

Table: Feature comparison of general MCMC software packages with `nuclear-spin-recovery`, highlighting support for continuous, discrete, mixed, hybrid, and transdimensional sampling. \label{mcmc-table}

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

The software is structured so that there are a series of high-level abstract objects that interact among themselves, and then specific logic related to 



# Acknowledgements

We thank Mykyta Onizhuk, F. Joseph Heremans, Benjamin Pingault, Christopher Egerstrom, Daniel Sanz-Alonso, and Nathan Waniorek for helpful conversations during the development of this project.
A.N.P acknowledges support from the DOE CSGF under Award No. DE-SC0022158 and AFOSR under ???.
G.G. acknowledges support from the Midwest Integrated Center for Computational Materials (MICCoM) as a part of the Computational Materials Sciences Program funded by the U.S. Department of Energy.
This research used resources of the University of Chicago Research Computing Center.

# References

