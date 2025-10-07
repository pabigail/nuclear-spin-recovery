import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import wasserstein_distance


class ErrorModel(ABC):
    """
    Abstract base class for error (misfit) functions that compare
    simulated and observed NV coherence data.
    """

    def __init__(self, data):
        if data is None:
            raise ValueError("data must be provided.")
        self.data = np.asarray(data, dtype=object)

    @abstractmethod
    def __call__(self, spins, experiment, forward_model_cls):
        """
        Compute a scalar error value.

        Parameters
        ----------
        spins : list of NuclearSpin
            Nuclear spin configuration.
        experiment : Experiment
            Experiment parameters.
        forward_model_cls : type
            A subclass of ForwardModel (not an instance), used to simulate coherence.

        Returns
        -------
        float
            Scalar error value.
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.data)} experiments>"



class L2Error(ErrorModel):
    """
    Computes the total L2 norm error between observed and simulated coherence signals.
    """

    def __call__(self, spins, experiment, forward_model_cls):
        # Instantiate the model with spins and experiment
        model = forward_model_cls(spins, experiment)

        total_error = 0.0
        for idx in range(len(experiment)):
            simulated = np.array(model.compute_coherence(idx))
            observed = np.array(self.data[idx])

            if len(observed) != len(simulated):
                raise ValueError(f"Length mismatch: observed={len(observed)}, simulated={len(simulated)}")

            total_error += np.linalg.norm(observed - simulated)

        return float(total_error)



class CompositeErrorL2andWasserstein(ErrorModel):
    """
    Computes a composite negative log-likelihood with L2 and Wasserstein distance terms.
    """

    def __init__(self, data, sigma_sq=None, lambda_wass=0.0):
        super().__init__(data)
        self.sigma_sq = sigma_sq
        self.lambda_wass = lambda_wass

    def __call__(self, spins, experiment, forward_model_cls):
        # Create model instance
        model = forward_model_cls(spins, experiment)

        total_error = 0.0
        for idx in range(len(experiment)):
            simulated = np.array(model.compute_coherence(idx))
            observed = np.array(self.data[idx])

            sigma_sq_eff = (
                self.sigma_sq if self.sigma_sq is not None else experiment.noise[idx]
            )

            # ----------------------
            # L2 component
            # ----------------------
            l2_term = 0.5 / sigma_sq_eff * np.sum((observed - simulated) ** 2)
            l2_term *= (1.0 - self.lambda_wass)

            # ----------------------
            # Wasserstein component
            # ----------------------
            min_val = min(observed.min(), simulated.min())
            obs_shift = observed - min_val
            sim_shift = simulated - min_val

            obs_sum = np.sum(obs_shift)
            sim_sum = np.sum(sim_shift)

            if obs_sum > 0 and sim_sum > 0 and np.isfinite(obs_sum) and np.isfinite(sim_sum):
                obs_norm = obs_shift / obs_sum
                sim_norm = sim_shift / sim_sum
                x_positions = np.arange(len(observed))
                wass_term = self.lambda_wass * wasserstein_distance(
                    x_positions, x_positions, obs_norm, sim_norm
                )
            else:
                wass_term = 0.0

            total_error += l2_term + wass_term

        return float(total_error)
