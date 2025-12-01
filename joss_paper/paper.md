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
`nuclear-spin-recovery` is a Python package for hybrid markov chain Monte Carlo (MCMC) sampling to reconstruct nuclear spin baths from experimental coherence data. Key features of this Python package include the ability to sample from mixed-continuous parameter distributions jointly by interleaving different MCMC techniques. This Python package was designed for maximum extensibility in terms of the kinds of materials or spin-baths that can be modeled, the types of MCMC algorithms used for sampling, custom error and likelihood functions, and forward models based on different levels of cluster-correlation expansion theory, providing posterior distributions of different experimentally-relevant quantities that can be extracted even from ill-posed regimes, where multiple sets of parameters may have a high likelihood of producing some observed data. This software package is both a practical tool for experimentalists seeking to recover nuclear spin configurations from standard dynamical decoupling experiments and a numerical tool for non-linear, non-convex optimization for theorists in terms of inverse-design of pulse experiments for specific nuclear spin-bath configurations. 

# Statement of need


# Software overview



# Acknowledgements

We thank Mykyta Onizhuk, F. Joseph Heremans, Benjamin Pingault, Christopher Egerstrom,  
Daniel Sanz-Alonso, and Nathan Waniorek for helpful conversations during the development of this project.

A.N.P acknowledges support from the DOE CSGF under Award No. DE-SC0022158 and AFOSR under ???.

G.G. acknowledges support from the Midwest Integrated Center for Computational Materials (MICCoM) as a part of the Computational Materials Sciences Program funded by the U.S. Department of Energy.

This research used resources of the University of Chicago Research Computing Center.

# References

