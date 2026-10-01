"""Adaptive experiment design: which experiment to run next, and where.

The engine's only contact with the sampler is
:meth:`~nuclear_spin_recovery.design.particles.ParticleSet.from_trace`.  Nothing
in this subpackage imports an algorithm, driver, schedule or ensemble runner,
so a single chain, a pooled ensemble, or an array typed in by hand all look the
same to the designer.

See docs/phase-5-plan.md.
"""

from .cost import SequenceDuration
from .designer import ExperimentDesigner
from .extrapolation import DecouplingScaling
from .particles import ParticleSet
from .selection import (
    GreedyUtility,
    InformationDensity,
    LeastInformative,
    NothingToLearn,
    PointSelector,
    UniformThinning,
)
from .utility import (
    DesignUtility,
    ExpectedInformationGain,
    PredictiveVariance,
    information_density,
)

__all__ = [
    "DecouplingScaling",
    "DesignUtility",
    "ExpectedInformationGain",
    "ExperimentDesigner",
    "GreedyUtility",
    "InformationDensity",
    "LeastInformative",
    "NothingToLearn",
    "ParticleSet",
    "PointSelector",
    "PredictiveVariance",
    "SequenceDuration",
    "UniformThinning",
    "information_density",
]
