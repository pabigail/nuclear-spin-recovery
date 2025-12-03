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
csl: joss.csl

header-includes:
  - \usepackage{amssymb}
  - \usepackage{amsmath}
  - \usepackage{xcolor}
  - \newcommand{\cmark}{\textcolor{green}{\checkmark}}
  - \newcommand{\xmark}{\textcolor{red}{\times}}
---

# Summary
We present a software package that implements a flexible Bayesian framework for ill-posed inverse problems and multi-modal optimization problems that arise in the analysis of coherence properties of spin defects in semiconductors in quantum sensing and spin-physics experiments.
These tools are designed for experimentalists seeking posterior distributions of nuclear spin baths from sparsely sampled standard coherence signals and for theorists developing optimized pulse sequences or adaptive data-assimilation schemes tailored to specific spin defects, nuclear baths, and experimental setups.
Key features of our software package include joint sampling of continuous and discrete parameters, hybridization of several MCMC techniques include trans-dimensional model selection, modular forward-model interfaces to simulate coherence signals under a range of theoretical approaches, customizable error models and likelihood functions to support a range of experimental conditions and noise sources, and extensibility to arbitrary nuclear spin baths and semiconductors. 

# Statement of need

Experimental characterization of nuclear spin environments around solid-state spin defects, such as nitrogen vacancy (NV) centers in diamond or divacancies in silicon carbide (SiC), is a central problem in quantum sensing and quantum information science. 
On the one hand, these nuclear spins can be a main source of decoherence of quantum information for the spin-defect [@zhao2012decoherence; @seo2016quantum]. 
On the other hand, these nuclear spins can serve as local quantum registers or quantum memories, provided that their spatial configuration and couplings to the spin-defect are known [@bradley2019ten; @bourassa2020entanglement].

While direct experimental characterization of the full nuclear spin environment is possible through, for example, correlated sensing [@van2024mapping], these experiments are time-consuming and labor-intensive.
Often experimentalists do not require a full mapping of the nuclear spin environment but instead seek an initial rough characterization of the nuclear spin environment using standard dynamical decoupling experiments before deciding whether to use a specific spin-defect for further experiments [@poteshman2025high]. 
As a result, there is a need for nuclear spin-bath characterization methods from sparse, noisy coherence data that can be obtained relatively quickly. 

In such experiments, the inverse problem is ill-posed, where multiple nuclear spin bath configurations can yield the same experimental coherence data in the noisy, sparse regime. 
While there have been direct machine learning approaches [@jung2021deep; @varona2024automatic] to this same inverse problem, a direct inverse mapping of coherence signal to nuclear spin environment requires an order of magnitude more experimental data than is practical or necessary for a rough characterization [@poteshman2025trans]. 
We also note that approaches based on variational Bayesian inference have also been successfully applied to this inverse problem [@belliardo2025multi], but these approaches have not yet incorporated spatial nuclear information or flexible interfacing with different forward models to simulate coherence signals. 

Bayesian inference provides a natural framework for this task, but existing general, open-source Python MCMC packages (\autoref{mcmc-table}) lack both statistical and domain-specific features necessary for nuclear spin bath reconstruction. 
We need joint sampling from mixed discrete and continuous parameters, since a nuclear spin may be modeled as having both continuous parameters, such as hyperfine couplings, and discrete parameters, such as lattice positions. 
Similarly, experimental settings may have continuous features, such as magnetic field, or discrete features, such as the number of $\pi$-pulses. 
Whether specific parameters are modeled as continuous or discrete depends on the available prior knowledge or specific experimental setup. 
For example, the hyperfine couplings of a nuclear spin can be modeled as discrete if highly accurate _ab initio_ calculations are available or continuous if there is little prior knowledge about their distribution. Specific experimental equipment may model times for interpulse-spacings as either continuous or discrete, depending on equipment resolution limits. 
Few general-purpose MCMC frameworks support such mixed parameter spaces directly.

Efficient sampling of complex, multi-model posteriors often requires interleaving different MCMC techniques (e.g., random walk Metropolis Hastings, Gibbs sampling, parallel tempering, reverse jump MCMC). 
Most existing packages make hybridization cumbersome or impractical. 
Furthermore, nuclear spin bath characterization is inherently a multi-dimensional model selection problem, since we do not know how many nuclear spins are in the bath _a priori_, and existing MCMC softwares do not have support or easy extensbility to implement trans-dimensional MCMC-based model selection techniques, such as reverse jump MCMC.

Furthermore, the forward problem of simulating a coherence signal from a specific nuclear spin bath configuration with specific experimental settings is non-trivial and different approaches vary vastly in their computational expense.
The standard approach to simulate coherence signals is based on the cluster correlation expansion method [@yang2008quantum; @yang2020longitudinal], but the level of expansion required to obtain computationally converged coherence signals requires both theoretical consideration of the materials and nuclear spins being simulated and numerical convergence of hyperparameters.
These different levels of theory are implemented in `pyCCE` [@onizhuk2021pycce], and this software package is designed so that users can flexibly and interchangeably access different levels of theory in the forward model for the MCMC simulations.
This software package features domain-specific modular forward models that interface with `pyCCE` for easy switching between different levels of CCE theory.

`nuclear-spin-recovery` addresses these shortocmings of these general, open-source Python MCMC software packages by providing a domain-specific Python framework for hybrid MCMC sampling that:

- Supports joint sampling of discrete and continuous parameters informed by domain knowledge.
- Allows seamless hybridization of MCMC algorithms for different parameter types.
- Integrates with physics-based forward models and custom likelihood functions.
- Provides posterior distributions over experimentally relevant quantities, even in ill-posed regimes.
- Is extensible to new materials, spin baths, and experimental protocols.

This package enables both experimentalists and theorists to infer, quantify, and design nuclear spin environments effectively.

\begin{table}[ht]
\centering
\caption{Feature comparison of general, open-source Python MCMC packages with \texttt{nuclear-spin-recovery}, highlighting suport for sampling from continuous parameter distributions, discrete parameter distributions, jointly sampling from mixed continuous-discrete parameter distributions, hybridizing MCMC algorithms (on the level of individual parameters), and transdimensional (e.g., reverse jump) MCMC methods.}
\label{mcmc-table}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\hline
Package & \shortstack{Continuous \\ Parameters} & \shortstack{Discrete \\ Parameters} & \shortstack{Mixed \\ (Cont+Disc)} & \shortstack{Transdimensional \\ MCMC} & \shortstack{Straightforward \\ Hybridization} & Ref \\
\hline
PyMC & $\cmark$ & $\cmark$ & $\xmark$ & $\xmark$ & $\xmark$ & \cite{abril2023pymc} \\
Stan & $\cmark$ & $\xmark$ & $\xmark$ & $\xmark$ & $\xmark$ & \cite{carpenter2017stan} \\
emcee & $\cmark$ & $\xmark$ & $\xmark$ & $\xmark$ & $\xmark$ & \cite{foreman2013emcee} \\
TensorFlow Probability & $\cmark$ & $\cmark$ & $\xmark$ & $\xmark$ & $\xmark$ & \cite{dillon2017tensorflow} \\
pymcmcstat & $\cmark$ & $\xmark$ & $\xmark$ & $\xmark$ & $\xmark$ & \cite{miles2019pymcmcstat} \\
nuclear-spin-recovery & $\cmark$ & $\cmark$ & $\cmark$ & $\cmark$ & $\cmark$ \\
\hline
\end{tabular}%
}
\end{table}


# Software overview
The software is designed with two goals:

- allow experimentalists to reconstruct nuclear spin configurations from standard dynamical decoupling experiments quickly
- allow theorists to design, implement, and integrate new forward models, MCMC algorithms, experiments, and error/likelihood functions.

The software is structured so that there are a series of high-level abstract objects that interact among themselves, and then the logic for specific MCMC algorithms, forward models, error and likelihood functions can be implemented in subclasses of the abstract objects.
We highlight the main abstract objects below, and we refer users to the tutorials for examples of specific implementations of hybridized MCMC methods on specific nuclear spin baths.

The parameters to be estimated can be either properties of a `SpinBath` object consisting of `NuclearSpin` objects, which store physical properties such as xyz coordinates, hyperfine couplings, and gyromagnetic ratios, or of an `Experiment`, which contains experimental settings such as magnetic fields, interpulse spacings, the type of pulses (e.g., $\pi$ pulses).
Each of these properties can be either be a fixed parameter or a parameter than can be sampled from, and each parameter can be modeled as being drawn from a discrete or continuous distribution.
Any combination of these properties can be jointly sampled over, given some experimental coherence data.

The `ForwardModel` provides a link between a coherence signal and a combination of a specific set of `NuclearSpin` objects and an `Experiment`. The `ForwardModel` interfaces with the coherence signals computed using `pyCCE` or analytical formulas. In the future, a surrogate coherence model mapping nuclear spins and experiments to a coherence signal could also be used as the backend to a `ForwardModel`.

The `Error` functions and `LogLikelihood` objects connect a set of `NuclearSpins` and an `Experiment` with a set of coherence data using a `ForwardModel` given a specific underlying assumption about the noise distribution of the data (e.g., Gaussian, Poisson) or a specific optimization obective.

Different `Walkers` are defined for the parameters of the `SpinBath` and `Experiment` to be sampled over. 
The `ProposalKernel` and `AcceptanceProbability` objects provide the statistical engines to produce a proposal kernel (e.g., random walk Metropolis Hastings, reverse jump MCMC, parallel tempering) and acceptance probability for the different MCMC sampling methods.
A `Scheduler` takes in a list of `Walkers` and different MCMC algorithms defined by `ProposalKernel` and `AcceptanceProbability` objects, as well as hyperparameters for running the hybrid MCMC sampler, such as the number of steps that each MCMC algorithm should take and which algorithms sample which `Walkers`, intialization conditions, and the number of ensembles to run.

Once the `Scheduler` has run the algorithm, there are postprocessing features to obtain convergence information, estimates of the parameters being sampled, and uncertainty quantification. For specific examples, see the tutorials hosted on the github page.

# Acknowledgements

We thank Mykyta Onizhuk, F. Joseph Heremans, Benjamin Pingault, Christopher Egerstrom, Daniel Sanz-Alonso, and Nathan Waniorek for helpful conversations during the development of this project.
A.N.P acknowledges support from the DOE CSGF under Award No. DE-SC0022158 and AFOSR under ???.
G.G. acknowledges support from the Midwest Integrated Center for Computational Materials (MICCoM) as a part of the Computational Materials Sciences Program funded by the U.S. Department of Energy.
This research used resources of the University of Chicago Research Computing Center.

# References

