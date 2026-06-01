import numpy as np
import pytest
from nanoboost.scripts.discrete_wavelet_transform.wavelet_transform_setup import wavelet_transform_func, peak_tracer


def _gaussian_event(n=256, amplitude=50.0):
    t = np.linspace(-3, 3, n)
    return np.exp(-t**2) * amplitude


def test_wavelet_output_length():
    signal = _gaussian_event()
    reconstructed, coeffs = wavelet_transform_func(signal, thresh=0.2, wavelet="bior3.3")
    assert len(reconstructed) == len(signal)


def test_wavelet_returns_coefficients():
    signal = _gaussian_event()
    _, coeffs = wavelet_transform_func(signal, thresh=0.2, wavelet="bior3.3")
    assert isinstance(coeffs, list)
    assert len(coeffs) > 1  # at least approximation + one detail level


def test_wavelet_smoothing_reduces_noise():
    rng = np.random.default_rng(42)
    clean = _gaussian_event(n=256, amplitude=50.0)
    noisy = clean + rng.normal(0, 3.0, len(clean))
    reconstructed, _ = wavelet_transform_func(noisy, thresh=0.2, wavelet="bior3.3")
    assert np.std(reconstructed - clean) < np.std(noisy - clean)


def test_wavelet_preserves_peak_amplitude():
    signal = _gaussian_event(amplitude=100.0)
    reconstructed, _ = wavelet_transform_func(signal, thresh=0.1, wavelet="bior3.3")
    assert max(reconstructed) > 60.0  # peak is substantially retained


def test_wavelet_higher_threshold_more_smooth():
    rng = np.random.default_rng(1)
    signal = _gaussian_event() + rng.normal(0, 2, 256)
    _, coeffs_low = wavelet_transform_func(signal, thresh=0.05, wavelet="bior3.3")
    _, coeffs_high = wavelet_transform_func(signal, thresh=0.4, wavelet="bior3.3")
    # higher threshold zeros more coefficients → sparser detail levels
    detail_energy_low = sum(np.sum(c**2) for c in coeffs_low[1:])
    detail_energy_high = sum(np.sum(c**2) for c in coeffs_high[1:])
    assert detail_energy_high <= detail_energy_low


def test_peak_tracer_restores_raw_peak():
    signal = _gaussian_event(n=256, amplitude=50.0)
    # simulate a DWT that flattens the peak
    smoothed = np.zeros_like(signal)
    event_data = [signal]
    dwt_sigs = [smoothed]
    traced = peak_tracer(event_data, dwt_sigs, threshold_upper=10.0, thresh=0.2)
    # peak region (where raw > 10) should now match the raw signal
    assert max(traced[0]) > 10.0
