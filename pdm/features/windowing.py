"""Windowing operations for time-series signal arrays."""

import numpy as np


def slice_windows(signal: np.ndarray, window_len: int, hop: int) -> np.ndarray:
    """Slice a continuous 1D or 2D signal array into overlapping windows.

    Parameters
    ----------
    signal : np.ndarray
        Shape (N,) or (N, C) where N is total samples.
    window_len : int
        Number of samples per window.
    hop : int
        Number of samples to advance between consecutive windows.

    Returns
    -------
    np.ndarray
        Array of windows with shape (n_windows, window_len) or (n_windows, window_len, C).
    """
    n_total = len(signal)
    if n_total < window_len:
        return np.empty((0, window_len) + signal.shape[1:], dtype=signal.dtype)

    n_windows = 1 + (n_total - window_len) // hop
    windows = []
    for i in range(n_windows):
        start = i * hop
        end = start + window_len
        windows.append(signal[start:end])

    return np.array(windows)


def calculate_n_windows(n_samples: int, window_len: int, hop: int) -> int:
    """Return the exact number of windows produced for a given sample count."""
    if n_samples < window_len:
        return 0
    return 1 + (n_samples - window_len) // hop
