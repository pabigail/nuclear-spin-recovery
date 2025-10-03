import numpy as np
import pycce as pc
from abc import ABC, abstractmethod
from .spin_bath import SpinBath, NuclearSpin


class ForwardModel(ABC):
    """
    Abstract base class for forward models that compute NV (or other qubit)
    coherence signals given a nuclear spin environment and experiment parameters.
    """

    def __init__(self, spins, experiment):
        """
        Parameters
        ----------
        spins : list of NuclearSpin
            The nuclear spin bath configuration.
        experiment : Experiment
            Experimental conditions (magnetic field, timepoints, pulse sequence, etc.).
        """
        if not isinstance(spins, (list, tuple)):
            raise TypeError("spins must be a list of NuclearSpin objects.")
        if not all(isinstance(s, NuclearSpin) for s in spins):
            raise TypeError("all items in spins must be NuclearSpin instances.")
        if not isinstance(experiment, Experiment):
            raise TypeError("experiment must be an Experiment instance.")

        self.spins = spins
        self.experiment = experiment

    @abstractmethod
    def compute_coherence(self, idx=0):
        """
        Compute the coherence signal for experiment `idx`.

        Parameters
        ----------
        idx : int, optional
            Index of the experiment (default: 0).

        Returns
        -------
        signal : np.ndarray
            Complex or real-valued coherence signal, shape = (len(timepoints),).
        """
        pass

    def __repr__(self):
        return (f"<ForwardModel with {len(self.spins)} spins and "
                f"{len(self.experiment)} experiments>")



class AnalyticCoherence(ForwardModel):
    """
    Analytic forward model for NV coherence using a closed-form expression
    for a single nuclear spin, extended to multiple spins with lambda_decoherence envelope.
    """

    def coherence_one_spin(self, t_i, A_par, A_perp, N, B_mag, gyro):
        """
        Analytic formula for single-spin contribution to coherence.
        Units: A_par, A_perp in kHz; B_mag in Gauss.
        """
        # Convert to angular frequency
        A_par = A_par * 2 * np.pi
        A_perp = A_perp * 2 * np.pi

        # Larmor frequency (rad/s)
        w_L = gyro * B_mag

        w_1 = A_par + w_L
        w = np.sqrt(w_1**2 + A_perp**2)
        mz = w_1 / w
        mx = A_perp / w

        alpha = np.outer(t_i, w)
        beta = np.array(t_i * w_L)[:, np.newaxis]

        cos_a, cos_b = np.cos(alpha), np.cos(beta)
        sin_a, sin_b = np.sin(alpha), np.sin(beta)

        phi = np.arccos(cos_a * cos_b - mz * sin_a * sin_b)

        n0n1 = (mx**2 * (1 - cos_a) * (1 - cos_b) /
                 (1 + cos_a * cos_b - mz * sin_a * sin_b))

        M = (1 - n0n1 * np.sin(N * phi / 2)**2)
        return M.flatten()

    def calculate_coherence(self, A_par_list=None, A_perp_list=None):
        """
        Compute raw coherence signals (without lambda_decay envelope).
        """
        signals = []

        for idx in range(len(self.experiment)):
            params = self.experiment[idx]
            t_i = params["timepoints"]
            N = params["num_pulses"]
            B_mag = params["mag_field"]

            # Start with unity signal
            total_signal = np.ones_like(t_i, dtype=float)

            for s_i, spin in enumerate(self.spins):
                A_par = spin.A_par if A_par_list is None else A_par_list[s_i]
                A_perp = spin.A_perp if A_perp_list is None else A_perp_list[s_i]

                spin_signal = self.coherence_one_spin(
                    t_i, A_par, A_perp, N, B_mag, gyro
                )
                total_signal *= spin_signal

            signals.append(total_signal)

        return signals

    def compute_coherence(self, A_par_list=None, A_perp_list=None):
        """
        Compute coherence signals with T2 decay envelope.
        Exactly mirrors `calculate_coherence_with_T2`.
        """
        # Step 1: raw coherence
        coherence_signals = self.calculate_coherence(A_par_list, A_perp_list)

        # Step 2: apply T2 envelope
        coherence_signals_with_T2 = []

        for index in range(len(self.experiment)):
            params = self.experiment[index]
            T2 = params["T2"]
            time = params["timepoints"]

            neg_exp = [np.exp(-t / T2) for t in time]
            coherence_with_T2 = [L * e for L, e in zip(coherence_signals[index], neg_exp)]
            coherence_signals_with_T2.append(np.array(coherence_with_T2))

        return coherence_signals_with_T2
