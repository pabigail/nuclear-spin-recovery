"""Forward model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ForwardModel(ABC):
    """Maps a spin configuration to a predicted coherence signal."""

    @abstractmethod
    def coherence(self, state, expset, site_table):
        """Predicted coherence at every point of the experiment set.

        Returns an array of shape (n_replicas, n_points).
        """
