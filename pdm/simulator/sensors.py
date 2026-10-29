"""Virtual sensors converting physical joint state to noisy observations."""

from typing import Dict, Tuple
import numpy as np
from pdm.config import SimulatorConfig


class VirtualSensors:
    """Simulates multi-modal sensors with realistic noise, drift, dropouts, and glitches."""

    def __init__(self, config: SimulatorConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

    def sample_step(
        self,
        t: int,
        joint_idx: int,
        speed: float,
        load: float,
        temp_nominal: float,
        f_rot: float,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Generate high-rate raw arrays and slow telemetry for one step."""
        n_samples = self.config.sample_rate_hz
        dt = 1.0 / n_samples
        t_arr = np.linspace(0.0, 1.0 - dt, n_samples)

        # Baseline vibration: A0 * sin(2*pi*f_rot*t) + 0.3*A0*sin(2*pi*2*f_rot*t) + noise
        a0 = 0.5 + 0.5 * load
        phase_x = 0.0 + joint_idx * (np.pi / 6.0)
        phase_y = (np.pi / 3.0) + joint_idx * (np.pi / 6.0)
        phase_z = (2.0 * np.pi / 3.0) + joint_idx * (np.pi / 6.0)

        fund = np.sin(2.0 * np.pi * f_rot * t_arr)
        harm2 = 0.3 * np.sin(2.0 * np.pi * 2.0 * f_rot * t_arr)

        vib_x = a0 * np.sin(2.0 * np.pi * f_rot * t_arr + phase_x) + 0.3 * a0 * np.sin(4.0 * np.pi * f_rot * t_arr + phase_x)
        vib_y = 0.8 * a0 * np.sin(2.0 * np.pi * f_rot * t_arr + phase_y) + 0.25 * a0 * np.sin(4.0 * np.pi * f_rot * t_arr + phase_y)
        vib_z = 0.6 * a0 * np.sin(2.0 * np.pi * f_rot * t_arr + phase_z) + 0.2 * a0 * np.sin(4.0 * np.pi * f_rot * t_arr + phase_z)

        # Add Gaussian noise
        sigma_v = self.config.noise.sigma_v
        vib_x += self.rng.normal(0.0, sigma_v, n_samples)
        vib_y += self.rng.normal(0.0, sigma_v, n_samples)
        vib_z += self.rng.normal(0.0, sigma_v, n_samples)
        vibration = np.stack([vib_x, vib_y, vib_z], axis=1)

        # Servo current: I0 * (0.4 + 0.6 * load) + ripple + noise
        i0 = self.config.i0_base
        i_base = i0 * (0.4 + 0.6 * load)
        # Ripple at PWM harmonic (approx 50 Hz carrier)
        pwm_ripple = 0.05 * i0 * np.sin(2.0 * np.pi * 50.0 * t_arr)
        current = i_base + pwm_ripple + self.rng.normal(0.0, self.config.noise.sigma_i, n_samples)

        # Slow calibration drift on current
        drift_factor = 1.0 + (self.config.drift.rate_per_1000 * (t / 1000.0))
        current *= drift_factor

        # Temperature: first-order thermal model + noise + drift
        temp = temp_nominal + self.rng.normal(0.0, self.config.noise.sigma_t)
        temp *= drift_factor

        # Acoustic proxy: band-limited energy proportional to speed + noise
        acoustic = (0.2 + 0.5 * speed) + self.rng.normal(0.0, self.config.noise.sigma_a, n_samples)

        # Apply sensor imperfections (dropouts and glitches)
        imp = self.config.sensor_imperfections
        # Temperature dropout
        if self.rng.random() < imp.dropout_prob:
            temp = float("nan")
        elif self.rng.random() < imp.glitch_prob:
            temp *= imp.glitch_multiplier

        # Current dropout / glitch (per-step chance to glitch array or drop)
        if self.rng.random() < imp.dropout_prob:
            current[:] = float("nan")
        elif self.rng.random() < imp.glitch_prob:
            current *= imp.glitch_multiplier

        # Vibration dropout / glitch
        if self.rng.random() < imp.dropout_prob:
            vibration[:] = float("nan")
        elif self.rng.random() < imp.glitch_prob:
            vibration *= imp.glitch_multiplier

        signals = {
            "vibration": vibration,
            "current": current,
            "acoustic": acoustic,
        }
        slow = {
            "temperature_c": temp,
        }

        return signals, slow
