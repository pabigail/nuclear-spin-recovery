import numpy as np
import pytest
from nuclear_spin_recover import AnalyticCoherenceModel, NuclearSpin, Experiment

@pytest.fixture
def single_spin():
    # Simple test spin with axial hyperfine couplings
    return NuclearSpin("C13", x=0.0, y=0.0, z=1.0,
                       A_par=10.0, A_perp=5.0,
                       gyro=1.0)

@pytest.fixture
def simple_experiment():
    # One experiment: 10 timepoints, Hahn echo (N=1), B=500 G
    timepoints = np.linspace(0, 1e-3, 10)  # seconds
    exp = Experiment(
        num_exps=1,
        num_pulses=[1],
        mag_field=[500.0],
        noise=[0.0],
        timepoints=[timepoints],
        lambda_decoherence=[1.0]  # replaced by lambda_decoherence in AnalyticCoherenceModel
    )
    # Patch in lambda_decoherence param for consistency
    exp.lambda_decoherence = np.array([1e-3])
    return exp

def test_calculate_coherence_single_spin(single_spin, simple_experiment):
    model = AnalyticCoherenceModel([single_spin], simple_experiment)
    signals = model.calculate_coherence()
    assert isinstance(signals, list)
    assert len(signals) == 1
    assert signals[0].shape == simple_experiment[0]["timepoints"].shape
    # Should be between -1 and 1
    assert np.all(np.abs(signals[0]) <= 1.0 + 1e-12)

def test_compute_coherence_with_decay(single_spin, simple_experiment):
    model = AnalyticCoherenceModel([single_spin], simple_experiment)
    signals_with_decay = model.compute_coherence()
    signals_no_decay = model.calculate_coherence()
    # Decay should reduce the amplitude
    assert np.all(np.abs(signals_with_decay[0]) <= np.abs(signals_no_decay[0]) + 1e-12)

def test_multiple_spins_multiplicative(simple_experiment):
    s1 = NuclearSpin("C13", x=0, y=0, z=1, A_par=10.0, A_perp=5.0, gyro=1.0)
    s2 = NuclearSpin("C13", x=1, y=0, z=0, A_par=7.0, A_perp=3.0, gyro=1.0)
    model = AnalyticCoherenceModel([s1, s2], simple_experiment)
    single_model = AnalyticCoherenceModel([s1], simple_experiment)
    signals_multi = model.calculate_coherence()[0]
    signals_single = single_model.calculate_coherence()[0]
    # Multi-spin coherence should be smaller or equal in mag

def test_one_spin_two_experiments():
    exp = Experiment(
        num_exps=2,
        num_pulses=[1, 2],
        mag_field=[100, 200],
        noise=[0.0, 0.0],
        timepoints=[[0.0, 1.0], [0.0, 1.0]],
        lambda_decoherence=[1.0, 2.0],
    )

    # Mock spin with dummy hyperfine couplings
    s2 = NuclearSpin("C13", x=1, y=0, z=0, A_par=7.0, A_perp=3.0, gyro=1.0)
    spins = [s2]
    model = AnalyticCoherenceModel(spins=spins, experiment=exp)

    signals = model.compute_coherence()
    assert isinstance(signals, list)
    assert len(signals) == 2  # two experiments
    assert all(isinstance(sig, np.ndarray) for sig in signals)

def test_two_spins_two_experiments():
    exp = Experiment(
        num_exps=2,
        num_pulses=[1, 2],
        mag_field=[100, 200],
        noise=[0.0, 0.0],
        timepoints=[[0.0, 1.0], [0.0, 1.0]],
        lambda_decoherence=[1.0, 2.0],
    )

    s1 = NuclearSpin("C13", x=0, y=0, z=1, A_par=10.0, A_perp=5.0, gyro=1.0)
    s2 = NuclearSpin("C13", x=1, y=0, z=0, A_par=7.0, A_perp=3.0, gyro=1.0)
    spins = [s1, s2]
    model = AnalyticCoherenceModel(spins=spins, experiment=exp)

    signals = model.compute_coherence()
    assert isinstance(signals, list)
    assert len(signals) == 2
    assert all(isinstance(sig, np.ndarray) for sig in signals)
    # At t=0, all signals should be exactly 1 (normalization check)
    assert np.allclose([sig[0] for sig in signals], [1.0, 1.0])
