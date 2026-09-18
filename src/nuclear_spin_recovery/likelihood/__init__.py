"""Likelihood models."""

from .base import Likelihood
from .gaussian import GaussianL2

__all__ = ["Likelihood", "GaussianL2"]
