# tests/test_coherence_signal.py

import pytest
import numpy as np
from nuclear_spin_recover.coherence_signal import (
    SingleCoherenceSignal,
    CoherenceSignalList,
)


# =============================================================================
# SingleCoherenceSignal tests
# =============================================================================

def test_single_coherence_signal_init():
    times = np.linspace(0, 1, 10)
    coherence = np.ones_like(times)

    signal = SingleCoherenceSignal(times=times, coherence=coherence)

    # Check types
    assert isinstance(signal.times, np.ndarray)
    assert isinstance(signal.coherence, np.ndarray)

    # Check shapes
    assert signal.times.shape == signal.coherence.shape
    assert signal.times.ndim == 1


def test_single_coherence_signal_invalid_shapes():
    times = np.linspace(0, 1, 10)
    coherence = np.ones((5,))  # mismatch

    with pytest.raises(ValueError):
        SingleCoherenceSignal(times=times, coherence=coherence)


def test_single_coherence_signal_non_1d_arrays():
    times = np.ones((2, 5))
    coherence = np.ones((2, 5))

    with pytest.raises(ValueError):
        SingleCoherenceSignal(times=times, coherence=coherence)


def test_single_coherence_signal_immutable():
    times = np.linspace(0, 1, 5)
    coherence = np.ones_like(times)

    signal = SingleCoherenceSignal(times=times, coherence=coherence)

    with pytest.raises(AttributeError):
        signal.times = np.zeros_like(times)

    with pytest.raises(AttributeError):
        signal.coherence = np.zeros_like(times)


# =============================================================================
# CoherenceSignalList tests
# =============================================================================

@pytest.fixture
def sample_signals():
    times = np.linspace(0, 1, 5)
    return [SingleCoherenceSignal(times=times, coherence=np.ones_like(times))
            for _ in range(3)]


def test_coherence_signal_list_init(sample_signals):
    sig_list = CoherenceSignalList(sample_signals)

    # Length
    assert len(sig_list) == len(sample_signals)

    # Iteration
    for s, original in zip(sig_list, sample_signals):
        assert s.times.shape == original.times.shape
        assert np.allclose(s.coherence, original.coherence)

    # Indexing
    for i in range(len(sig_list)):
        assert sig_list[i] is sample_signals[i]


def test_coherence_signal_list_empty():
    with pytest.raises(ValueError):
        CoherenceSignalList([])


def test_coherence_signal_list_invalid_element():
    times = np.linspace(0, 1, 5)
    bad_signals = [SingleCoherenceSignal(times, np.ones_like(times)), 42]

    with pytest.raises(TypeError):
        CoherenceSignalList(bad_signals)


def test_coherence_signal_list_is_iterable(sample_signals):
    sig_list = CoherenceSignalList(sample_signals)

    signals_iter = list(iter(sig_list))
    assert len(signals_iter) == len(sample_signals)
    for s1, s2 in zip(signals_iter, sample_signals):
        assert s1 is s2

