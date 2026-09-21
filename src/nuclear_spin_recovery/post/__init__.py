"""Posterior summaries: detection, residual, dimension, and diagnostics.

Spec Sec. 9.2.  Every metric is computed over the posterior rather than over a
single configuration, and matched on couplings rather than on site index.
"""

from .detection import (
    BANDS,
    MATCH_TOL,
    band_index,
    by_band,
    couplings,
    detection_rate,
    false_absence,
    matches,
)
from .metrics import PosteriorSummary, summarize
from .residual import (
    predictive_from_arrays,
    predictive_signals,
    residual_distribution,
)

__all__ = [
    "BANDS",
    "MATCH_TOL",
    "PosteriorSummary",
    "band_index",
    "by_band",
    "couplings",
    "detection_rate",
    "false_absence",
    "matches",
    "predictive_from_arrays",
    "predictive_signals",
    "residual_distribution",
    "summarize",
]
