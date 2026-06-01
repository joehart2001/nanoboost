import numpy as np
import pytest
from nanoboost.scripts.data_preprocessing.baseline_correction import compute_moving_average
from nanoboost.scripts.data_preprocessing.peak_finder import define_threshold


def test_moving_average_output_length():
    y = np.ones(10000)
    result = compute_moving_average(y, window_size=100)
    assert len(result) == len(y) - 100 + 1


def test_moving_average_flat_signal():
    y = np.full(5000, 7.5)
    result = compute_moving_average(y, window_size=500)
    np.testing.assert_allclose(result, 7.5, atol=1e-10)


def test_moving_average_removes_offset():
    # a linearly drifting signal should have its drift captured by the moving average
    y = np.linspace(0, 100, 10000)
    result = compute_moving_average(y, window_size=1000)
    assert len(result) > 0
    assert result[0] < result[-1]  # drift is captured


def test_define_threshold_value():
    rng = np.random.default_rng(42)
    noise = rng.normal(loc=5.0, scale=2.0, size=50000)
    threshold, mean_noise, sd_noise = define_threshold(noise, n_sd_upper=10)
    assert threshold == pytest.approx(mean_noise + 10 * sd_noise)


def test_define_threshold_mean_and_sd():
    rng = np.random.default_rng(0)
    noise = rng.normal(loc=0.0, scale=1.0, size=50000)
    _, mean_noise, sd_noise = define_threshold(noise, n_sd_upper=5)
    assert abs(mean_noise) < 0.05
    assert abs(sd_noise - 1.0) < 0.05


def test_define_threshold_with_lower():
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1, 10000)
    upper, lower, mean_noise, sd_noise = define_threshold(noise, n_sd_upper=10, n_sd_lower=10)
    assert upper > mean_noise
    assert lower < mean_noise
    assert upper == pytest.approx(mean_noise + 10 * sd_noise)
    assert lower == pytest.approx(mean_noise - 10 * sd_noise)
