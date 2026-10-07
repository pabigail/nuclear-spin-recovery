"""Posterior summaries: detection, residual, dimension, and diagnostics.

Spec Sec. 9.2.  Every metric is computed over the posterior rather than over a
single configuration, and matched on couplings rather than on site index.
"""

from .detection import (
    BANDS,
    MATCH_TOL,
    band_index,
    by_band,
    coupling_posterior,
    couplings,
    detection_rate,
    false_absence,
    matches,
)
from .metrics import PosteriorSummary, summarize
from .relaxation import (
    ModalConfiguration,
    RelaxedCouplings,
    compare_couplings,
    modal_configuration,
    relaxed_couplings,
)
from .residual import (
    predictive_from_arrays,
    predictive_signals,
    residual_distribution,
)

__all__ = [
    "BANDS",
    "MATCH_TOL",
    "ModalConfiguration",
    "PosteriorSummary",
    "RelaxedCouplings",
    "band_index",
    "by_band",
    "compare_couplings",
    "coupling_posterior",
    "couplings",
    "detection_rate",
    "false_absence",
    "matches",
    "modal_configuration",
    "predictive_from_arrays",
    "predictive_signals",
    "relaxed_couplings",
    "residual_distribution",
    "summarize",
]
