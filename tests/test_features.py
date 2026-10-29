"""Tests for feature engineering (windowing, extraction, imputation)."""

import numpy as np
import pytest
from pdm.features.windowing import slice_windows, calculate_n_windows
from pdm.features.extract import (
    impute_signal_nan,
    extract_time_features,
    extract_freq_features,
    JointFeatureExtractor,
)


def test_window_count_formula() -> None:
    """Asserts that window slicing obeys 1 + (N - window_len) // hop."""
    signal = np.arange(10000)
    window_len = 1024
    hop = 512

    windows = slice_windows(signal, window_len, hop)
    expected_count = calculate_n_windows(len(signal), window_len, hop)

    assert len(windows) == expected_count
    assert expected_count == 1 + (10000 - 1024) // 512
    assert windows[0].shape == (1024,)
    # Verify overlap
    np.testing.assert_array_equal(windows[0][512:], windows[1][:512])


def test_no_nan_after_imputation() -> None:
    """Asserts that raw sensor signals with dropouts contain no NaNs after imputation."""
    # 1D array with dropouts
    arr_1d = np.array([1.0, 2.0, np.nan, np.nan, 5.0, np.nan, 7.0])
    imputed_1d = impute_signal_nan(arr_1d)
    assert not np.any(np.isnan(imputed_1d))
    assert imputed_1d[2] == 2.0  # forward filled
    assert imputed_1d[3] == 2.0

    # 2D array with dropouts and all-NaN leading row
    arr_2d = np.array([
        [np.nan, 10.0],
        [np.nan, np.nan],
        [3.0, 12.0],
    ])
    imputed_2d = impute_signal_nan(arr_2d)
    assert not np.any(np.isnan(imputed_2d))


def test_fixed_feature_vector_length() -> None:
    """Asserts that every extracted feature vector has identical length and no NaNs."""
    extractor = JointFeatureExtractor(sample_rate_hz=1000, n_bands=8, trend_window=5)

    signals1 = {
        "vibration": np.random.randn(1000, 3),
        "current": np.random.randn(1000),
        "acoustic": np.random.randn(1000),
    }
    slow1 = {"temperature_c": 35.2}
    feats1 = extractor.extract_step(signals1, slow1, f_rot=5.0)

    # Corrupt step with random NaNs
    signals2 = {
        "vibration": np.random.randn(1000, 3),
        "current": np.full(1000, np.nan),
        "acoustic": np.random.randn(1000),
    }
    slow2 = {"temperature_c": float("nan")}
    feats2 = extractor.extract_step(signals2, slow2, f_rot=5.0)

    assert len(feats1) == len(feats2)
    # Total features should be in the expected range 120-150
    assert 120 <= len(feats1) <= 150
    assert set(feats1.keys()) == set(feats2.keys())
    assert not any(np.isnan(v) for v in feats1.values())
    assert not any(np.isnan(v) for v in feats2.values())
