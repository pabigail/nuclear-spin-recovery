"""Likelihood interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Likelihood(ABC):
    """Quantifies agreement between predicted and measured coherence."""

    @abstractmethod
    def log_prob(self, state, expset, model, site_table):
        """Log-likelihood per replica. (n_replicas,)"""
