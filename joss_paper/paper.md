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


# Software overview



# Acknowledgements

We thank Mykyta Onizhuk, F. Joseph Heremans, Benjamin Pingault, Christopher Egerstrom,  
Daniel Sanz-Alonso, and Nathan Waniorek for helpful conversations during the development of this project.

A.N.P acknowledges support from the DOE CSGF under Award No. DE-SC0022158 and AFOSR under ???.

G.G. acknowledges support from the Midwest Integrated Center for Computational Materials (MICCoM) as a part of the Computational Materials Sciences Program funded by the U.S. Department of Energy.

This research used resources of the University of Chicago Research Computing Center.

# References

