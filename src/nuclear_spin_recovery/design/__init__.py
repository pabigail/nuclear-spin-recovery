"""Adaptive experiment design: which experiment to run next, and where.

The engine's only contact with the sampler is
:meth:`~nuclear_spin_recovery.design.particles.ParticleSet.from_trace`.  Nothing
in this subpackage imports an algorithm, driver, schedule or ensemble runner,
so a single chain, a pooled ensemble, or an array typed in by hand all look the
same to the designer.

See docs/phase-5-plan.md.
"""

from .cost import SequenceDuration
from .designer import MIN_GAIN, CandidateDesign, DesignResult, ExperimentDesigner
from .extrapolation import DecouplingScaling
from .particles import ParticleSet
from .report import (
    TIMING_NOTE,
    format_measurement_table,
    measurement_rows,
    measurement_table_html,
    write_measurement_csv,
)
from .selection import InformationDensity, NothingToLearn, PointSelector
from .utility import DesignUtility, ExpectedInformationGain, information_density

__all__ = [
    "MIN_GAIN",
    "TIMING_NOTE",
    "CandidateDesign",
    "DecouplingScaling",
    "DesignResult",
    "DesignUtility",
    "ExpectedInformationGain",
    "ExperimentDesigner",
    "InformationDensity",
    "NothingToLearn",
    "ParticleSet",
    "PointSelector",
    "SequenceDuration",
    "format_measurement_table",
    "information_density",
    "measurement_rows",
    "measurement_table_html",
    "write_measurement_csv",
]
