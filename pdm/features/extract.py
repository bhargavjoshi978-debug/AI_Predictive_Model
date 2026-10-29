"""Feature extraction for time-domain and frequency-domain signals."""

from collections import deque
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

logger = logging.getLogger(__name__)


def impute_signal_nan(arr: np.ndarray, last_valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Impute NaNs in signal array using forward-fill, then fallback to median/zeros."""
    if not np.any(np.isnan(arr)):
        return arr.copy()

    logger.debug("NaN detected in raw sensor signal. Performing imputation.")
    out = arr.copy()

    # If completely NaN, fall back to last valid or zeros
    if np.all(np.isnan(out)):
        if last_valid is not None and not np.all(np.isnan(last_valid)):
            return last_valid.copy()
        return np.zeros_like(out)

    # 1D or 2D handling
    if out.ndim == 1:
        # Forward fill
        mask = np.isnan(out)
        idx = np.where(~mask, np.arange(len(out)), 0)
        np.maximum.accumulate(idx, out=idx)
        out = out[idx]
        # If leading elements are still NaN, fill with median of valid
        if np.any(np.isnan(out)):
            med = np.nanmedian(arr)
            med_val = 0.0 if np.isnan(med) else med
            out[np.isnan(out)] = med_val
    else:
        for col in range(out.shape[1]):
            out[:, col] = impute_signal_nan(out[:, col], last_valid[:, col] if last_valid is not None else None)

    return out


def extract_time_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """Extract 8 time-domain statistical features from a 1D signal window."""
    eps = 1e-12
    mean_val = float(np.mean(x))
    std_val = float(np.std(x))
    rms_val = float(np.sqrt(np.mean(x ** 2)))
    peak_val = float(np.max(np.abs(x)))
    ptp_val = float(np.ptp(x))

    diff = x - mean_val
    m2 = np.mean(diff ** 2) + eps
    m3 = np.mean(diff ** 3)
    m4 = np.mean(diff ** 4)

    skew_val = float(m3 / (m2 ** 1.5))
    kurt_val = float(m4 / (m2 ** 2.0))
    crest_val = float(peak_val / (rms_val + eps))

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_rms": rms_val,
        f"{prefix}_peak": peak_val,
        f"{prefix}_ptp": ptp_val,
        f"{prefix}_skew": skew_val,
        f"{prefix}_kurtosis": kurt_val,
        f"{prefix}_crest": crest_val,
    }


def extract_freq_features(
    x: np.ndarray,
    prefix: str,
    sample_rate_hz: int = 1000,
    f_rot: float = 5.0,
    n_bands: int = 8,
) -> Dict[str, float]:
    """Extract frequency-domain FFT features with a Hann window."""
    eps = 1e-12
    n = len(x)
    # Apply Hann window to detrended signal
    window = np.hanning(n)
    x_detrend = (x - np.mean(x)) * window
    rfft_vals = np.abs(np.fft.rfft(x_detrend))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)

    power = rfft_vals ** 2
    tot_power = float(np.sum(power)) + eps

    # Dominant frequency and amplitude (excluding DC)
    if len(rfft_vals) > 1:
        dom_idx = int(np.argmax(rfft_vals[1:])) + 1
        dom_freq = float(freqs[dom_idx])
        dom_amp = float(rfft_vals[dom_idx]) / n
    else:
        dom_freq = 0.0
        dom_amp = 0.0

    # Spectral centroid: sum(f * P) / sum(P)
    centroid = float(np.sum(freqs * power) / tot_power)

    # Spectral entropy: normalized Shannon entropy
    p_norm = power / tot_power
    p_norm_safe = np.where(p_norm > eps, p_norm, eps)
    entropy_raw = -float(np.sum(p_norm_safe * np.log2(p_norm_safe)))
    max_entropy = np.log2(len(p_norm)) if len(p_norm) > 1 else 1.0
    spectral_entropy = float(entropy_raw / (max_entropy + eps))

    feats: Dict[str, float] = {
        f"{prefix}_dom_freq": dom_freq,
        f"{prefix}_dom_amp": dom_amp,
        f"{prefix}_spectral_centroid": centroid,
        f"{prefix}_spectral_entropy": spectral_entropy,
    }

    # N equal frequency bands
    nyquist = sample_rate_hz / 2.0
    band_width = nyquist / n_bands
    for b in range(n_bands):
        b_low = b * band_width
        b_high = (b + 1) * band_width
        mask = (freqs >= b_low) & (freqs < b_high)
        band_energy = float(np.sum(power[mask])) / tot_power if np.any(mask) else 0.0
        feats[f"{prefix}_band_{b}"] = band_energy

    # Fault-specific frequency energies
    # 1x f_rot
    mask_1x = np.abs(freqs - f_rot) <= 0.8
    feats[f"{prefix}_energy_1x"] = float(np.sum(power[mask_1x])) / tot_power if np.any(mask_1x) else 0.0

    # 2x f_rot
    mask_2x = np.abs(freqs - 2.0 * f_rot) <= 1.0
    feats[f"{prefix}_energy_2x"] = float(np.sum(power[mask_2x])) / tot_power if np.any(mask_2x) else 0.0

    # Bearing defect: 3.5x f_rot
    mask_bpf = np.abs(freqs - 3.5 * f_rot) <= 1.5
    feats[f"{prefix}_energy_bpf"] = float(np.sum(power[mask_bpf])) / tot_power if np.any(mask_bpf) else 0.0

    # Gear mesh + sidebands: 12x f_rot and 11x, 13x
    mask_gm = (
        (np.abs(freqs - 12.0 * f_rot) <= 2.0)
        | (np.abs(freqs - 11.0 * f_rot) <= 1.5)
        | (np.abs(freqs - 13.0 * f_rot) <= 1.5)
    )
    feats[f"{prefix}_energy_gm"] = float(np.sum(power[mask_gm])) / tot_power if np.any(mask_gm) else 0.0

    return feats


class JointFeatureExtractor:
    """Extracts all features for a single joint while tracking rolling history for trend features."""

    def __init__(self, sample_rate_hz: int = 1000, n_bands: int = 8, trend_window: int = 5) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.n_bands = n_bands
        self.trend_window = trend_window

        # Rolling buffers for trend calculation
        self.rms_history: deque = deque(maxlen=trend_window)
        self.kurtosis_history: deque = deque(maxlen=trend_window)
        self.temp_history: deque = deque(maxlen=trend_window)

        # Last valid arrays for forward-fill imputation
        self.last_vib: Optional[np.ndarray] = None
        self.last_curr: Optional[np.ndarray] = None
        self.last_acou: Optional[np.ndarray] = None
        self.last_temp: float = 25.0

    def reset(self) -> None:
        """Reset rolling buffers."""
        self.rms_history.clear()
        self.kurtosis_history.clear()
        self.temp_history.clear()
        self.last_vib = None
        self.last_curr = None
        self.last_acou = None
        self.last_temp = 25.0

    def extract_step(
        self,
        signals: Dict[str, np.ndarray],
        slow: Dict[str, float],
        f_rot: float = 5.0,
    ) -> Dict[str, float]:
        """Extract a complete feature dictionary for one step window of joint telemetry."""
        # 1. Impute any NaNs in signals
        raw_vib = signals.get("vibration", np.zeros((self.sample_rate_hz, 3)))
        vib = impute_signal_nan(raw_vib, self.last_vib)
        self.last_vib = vib.copy()

        raw_curr = signals.get("current", np.zeros(self.sample_rate_hz))
        curr = impute_signal_nan(raw_curr, self.last_curr)
        self.last_curr = curr.copy()

        raw_acou = signals.get("acoustic", np.zeros(self.sample_rate_hz))
        acou = impute_signal_nan(raw_acou, self.last_acou)
        self.last_acou = acou.copy()

        temp_val = slow.get("temperature_c", self.last_temp)
        if np.isnan(temp_val):
            temp_val = self.last_temp
        self.last_temp = float(temp_val)

        features: Dict[str, float] = {}

        # 2. Extract high-rate vibration channels (x, y, z)
        for i, ax in enumerate(["x", "y", "z"]):
            channel_arr = vib[:, i] if vib.ndim > 1 else vib
            t_feats = extract_time_features(channel_arr, prefix=f"vib_{ax}")
            f_feats = extract_freq_features(
                channel_arr,
                prefix=f"vib_{ax}",
                sample_rate_hz=self.sample_rate_hz,
                f_rot=f_rot,
                n_bands=self.n_bands,
            )
            features.update(t_feats)
            features.update(f_feats)

        # 3. Extract high-rate current channel
        curr_t_feats = extract_time_features(curr, prefix="curr")
        curr_f_feats = extract_freq_features(
            curr,
            prefix="curr",
            sample_rate_hz=self.sample_rate_hz,
            f_rot=f_rot,
            n_bands=self.n_bands,
        )
        features.update(curr_t_feats)
        features.update(curr_f_feats)

        # 4. Extract acoustic channel
        acou_t_feats = extract_time_features(acou, prefix="acou")
        acou_f_feats = extract_freq_features(
            acou,
            prefix="acou",
            sample_rate_hz=self.sample_rate_hz,
            f_rot=f_rot,
            n_bands=self.n_bands,
        )
        features.update(acou_t_feats)
        features.update(acou_f_feats)

        # 5. Slow channel: temperature + slope
        self.temp_history.append(self.last_temp)
        if len(self.temp_history) > 1:
            temp_slope = float(self.temp_history[-1] - self.temp_history[0]) / (len(self.temp_history) - 1)
        else:
            temp_slope = 0.0
        features["temp_c"] = self.last_temp
        features["temp_slope"] = temp_slope

        # 6. Trend features (slope of RMS, rolling max of kurtosis)
        vib_rms = features["vib_x_rms"]
        vib_kurt = features["vib_x_kurtosis"]
        self.rms_history.append(vib_rms)
        self.kurtosis_history.append(vib_kurt)

        if len(self.rms_history) > 1:
            rms_slope = float(self.rms_history[-1] - self.rms_history[0]) / (len(self.rms_history) - 1)
        else:
            rms_slope = 0.0

        features["trend_rms_slope"] = rms_slope
        features["trend_kurtosis_max"] = float(np.max(self.kurtosis_history))

        return features
