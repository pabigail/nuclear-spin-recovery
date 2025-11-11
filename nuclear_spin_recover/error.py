import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import wasserstein_distance


class ErrorModel(ABC):
    """
    Abstract base class for error (misfit) functions that compare
    simulated and observed NV coherence data.

    The error model computes a scalar misfit between observed experimental
    data and coherence signals simulated by a forward model. 

    Attributes
    ----------
    data : np.ndarray
        Observed coherence data for one or more experiments. Stored as an
        object array where each entry corresponds to an experiment.
    """

    def __init__(self, data):
        if data is None:
            raise ValueError("data must be provided.")
        self.data = np.asarray(data, dtype=object)

    @abstractmethod
    def __call__(self, spins, forward_model):
        """
        Compute a scalar error value for the given nuclear spin configuration
        using the provided forward model.

        Parameters
        ----------
        spins : list of NuclearSpin
            Nuclear spin configuration to simulate.
        forward_model : ForwardModel
            **Instance** of a concrete ForwardModel subclass, which contains
            the experiment internally. The forward model is used to simulate
            coherence signals.

        Returns
        -------
        float
            Scalar error value comparing simulated and observed data.
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.data)} experiments>"


class L2Error(ErrorModel):
    """
    Computes the total L2 norm error between observed and simulated coherence signals.
    """

    def __call__(self, spins, forward_model):
    
        # Check if the forward model class already has an experiment
        if hasattr(forward_model, 'experiment'):
            experiment = forward_model.experiment
        else:
            raise ValueError(
                "Forward model must have an experiment stored internally. "
                "Pass a concrete subclass of ForwardModel that contains experiment."
            )

        # instantiate model with spins and its internal experiment
        model = type(forward_model)(spins, experiment)

        total_error = 0.0
        for idx in range(len(model.experiment)):
            simulated = np.array(model.compute_coherence(idx))
            observed = np.array(self.data[idx])

            if len(observed) != len(simulated):
                raise ValueError(f"Length mismatch: observed={len(observed)}, simulated={len(simulated)}")

            total_error += np.sum((observed - simulated)**2)

        return float(total_error)


class WassersteinError(ErrorModel):
    """
    Computes total Wasserstein (Earth Mover's) distance between observed and simulated signals.
    """
    def __call__(self, spins, forward_model):
        if hasattr(forward_model, 'experiment'):
            experiment = forward_model.experiment
        else:
            raise ValueError("Forward model must have an experiment attribute.")

        model = type(forward_model)(spins, experiment)
        total_error = 0.0

        for idx in range(len(model.experiment)):
            simulated = np.array(model.compute_coherence(idx))
            observed = np.array(self.data[idx])

            min_val = min(observed.min(), simulated.min())
            obs_shift = observed - min_val
            sim_shift = simulated - min_val

            obs_sum = np.sum(obs_shift)
            sim_sum = np.sum(sim_shift)

            if obs_sum > 0 and sim_sum > 0 and np.isfinite(obs_sum) and np.isfinite(sim_sum):
                obs_norm = obs_shift / obs_sum
                sim_norm = sim_shift / sim_sum
                x_positions = np.arange(len(observed))
                wass = wasserstein_distance(x_positions, x_positions, obs_norm, sim_norm)
            else:
                wass = 0.0

            total_error += wass

        return float(total_error)


class CompositeError(ErrorModel):
    """
    Combines multiple ErrorModels linearly with specified weights.
    """
    def __init__(self, data, error_models, weights=None):
        """
        Parameters
        ----------
        data : list or array
            Observed data.
        error_models : list of ErrorModel
            List of instantiated error models (e.g., [L2Error(data), WassersteinError(data)]).
        weights : list of float, optional
            Linear weights for each error term. Defaults to equal weighting.
        """
        super().__init__(data)
        self.error_models = error_models
        self.weights = weights if weights is not None else [1.0 / len(error_models)] * len(error_models)

        if len(self.error_models) != len(self.weights):
            raise ValueError("Number of weights must match number of error models.")

    def __call__(self, spins, forward_model):
        total = 0.0
        for w, err_model in zip(self.weights, self.error_models):
            total += w * err_model(spins, forward_model)
        return float(total)



class GaussianLogLikelihoodFromError(ErrorModel):
    """
    Converts any ErrorModel into a Gaussian log-likelihood or negative log-likelihood.
    """
    def __init__(self, data, base_error_model, sigma_sq=None, n_data=None, as_negative=True):
        super().__init__(data)
        self.base_error_model = base_error_model
        self.sigma_sq = sigma_sq
        self.n_data = n_data
        self.as_negative = as_negative

    def __call__(self, spins, forward_model):
        E = self.base_error_model(spins, forward_model)

        if self.sigma_sq is not None:
            sigma_sq_eff = self.sigma_sq
        elif hasattr(forward_model, "experiment") and hasattr(forward_model.experiment, "noise"):
            sigma_sq_eff = np.mean(np.atleast_1d(forward_model.experiment.noise))
        else:
            raise ValueError("sigma_sq must be provided or inferable from forward_model.experiment.noise")

        if self.n_data is not None:
            n = self.n_data
        elif hasattr(forward_model, "experiment"):
            n = sum(len(self.data[i]) for i in range(len(self.data)))
        else:
            n = len(np.atleast_1d(self.data))

        loglike = -0.5 * E / sigma_sq_eff - 0.5 * n * np.log(2 * np.pi * sigma_sq_eff)
        return float(-loglike if self.as_negative else loglike)
