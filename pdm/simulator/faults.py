"""Fault injection registry and implementations for robot simulator."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type
import numpy as np


class BaseFault(ABC):
    """Abstract base class for fault injectors."""

    name: str = "none"

    @abstractmethod
    def apply_dynamics(
        self,
        damage: float,
        speed: float,
        load: float,
        temp_delta: float,
        rng: np.random.Generator,
    ) -> Tuple[float, float, float]:
        """Modify dynamic quantities: (speed, load, temp_delta)."""
        pass

    @abstractmethod
    def apply_signals(
        self,
        damage: float,
        t_arr: np.ndarray,
        vib: np.ndarray,
        curr: np.ndarray,
        f_rot: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Modify high-rate vibration and current signals."""
        pass


class NoneFault(BaseFault):
    """Healthy baseline: no fault modification."""

    name = "none"

    def apply_dynamics(
        self,
        damage: float,
        speed: float,
        load: float,
        temp_delta: float,
        rng: np.random.Generator,
    ) -> Tuple[float, float, float]:
        return speed, load, temp_delta

    def apply_signals(
        self,
        damage: float,
        t_arr: np.ndarray,
        vib: np.ndarray,
        curr: np.ndarray,
        f_rot: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return vib, curr


class BearingFault(BaseFault):
    """Bearing fault: adds f_bpf = 3.5 * f_rot, impulsive spikes, broadband noise."""

    name = "bearing"

    def apply_dynamics(
        self,
        damage: float,
        speed: float,
        load: float,
        temp_delta: float,
        rng: np.random.Generator,
    ) -> Tuple[float, float, float]:
        # Slight thermal increase due to bearing friction
        return speed, load, temp_delta + 5.0 * damage

    def apply_signals(
        self,
        damage: float,
        t_arr: np.ndarray,
        vib: np.ndarray,
        curr: np.ndarray,
        f_rot: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if damage <= 0.0:
            return vib, curr

        # f_bpf = 3.5 * f_rot with amplitude proportional to damage
        f_bpf = 3.5 * f_rot
        vib_mod = vib.copy()
        amp_bpf = 0.8 * damage
        bpf_signal = amp_bpf * np.sin(2.0 * np.pi * f_bpf * t_arr)
        vib_mod = vib_mod + bpf_signal[:, np.newaxis]

        # Impulsive spikes with probability proportional to d^2 (spikes raise kurtosis)
        n_samples = len(t_arr)
        spike_prob = min(0.3, 0.25 * (damage ** 2))
        spike_mask = rng.random(n_samples) < spike_prob
        if np.any(spike_mask):
            spike_signs = rng.choice([-1.0, 1.0], size=np.count_nonzero(spike_mask))
            spike_amps = spike_signs * rng.uniform(1.5, 4.0, size=np.count_nonzero(spike_mask)) * damage
            vib_mod[spike_mask] += spike_amps[:, np.newaxis]

        # Broadband vibration noise raised by (1 + 2d)
        noise_boost = rng.normal(0.0, 0.05 * 2.0 * damage, size=vib_mod.shape)
        vib_mod += noise_boost

        return vib_mod, curr


class GearWearFault(BaseFault):
    """Gear wear: gear-mesh f_gm = 12 * f_rot with sidebands f_gm +- f_rot, small speed ripple."""

    name = "gear_wear"

    def apply_dynamics(
        self,
        damage: float,
        speed: float,
        load: float,
        temp_delta: float,
        rng: np.random.Generator,
    ) -> Tuple[float, float, float]:
        # Speed ripple from gear tooth slip/backlash
        ripple = 0.05 * damage * np.sin(2.0 * np.pi * 12.0 * speed)
        return speed * (1.0 + ripple), load, temp_delta + 3.0 * damage

    def apply_signals(
        self,
        damage: float,
        t_arr: np.ndarray,
        vib: np.ndarray,
        curr: np.ndarray,
        f_rot: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if damage <= 0.0:
            return vib, curr

        vib_mod = vib.copy()
        f_gm = 12.0 * f_rot
        # Gear mesh + sidebands at f_gm +- f_rot
        amp_gm = 0.7 * damage
        carrier = np.sin(2.0 * np.pi * f_gm * t_arr)
        modulator = 1.0 + 0.6 * damage * np.cos(2.0 * np.pi * f_rot * t_arr)
        mesh_signal = amp_gm * carrier * modulator
        vib_mod += mesh_signal[:, np.newaxis]

        # Current modulation from tooth engagement fluctuation
        curr_mod = curr + (0.15 * damage * np.sin(2.0 * np.pi * f_rot * t_arr))
        return vib_mod, curr_mod


class MotorOverheatFault(BaseFault):
    """Motor overheat: temp +35*d, thermal tau increases, current +25%*d, slight efficiency drop."""

    name = "motor_overheat"

    def apply_dynamics(
        self,
        damage: float,
        speed: float,
        load: float,
        temp_delta: float,
        rng: np.random.Generator,
    ) -> Tuple[float, float, float]:
        # Raises temp by +35 * d
        temp_extra = 35.0 * damage
        # Slight efficiency drop raises effective load demand
        load_eff = load * (1.0 + 0.1 * damage)
        return speed, load_eff, temp_delta + temp_extra

    def apply_signals(
        self,
        damage: float,
        t_arr: np.ndarray,
        vib: np.ndarray,
        curr: np.ndarray,
        f_rot: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if damage <= 0.0:
            return vib, curr

        # Current raised by +25% * d
        curr_mod = curr * (1.0 + 0.25 * damage)
        # Thermal vibration harmonic
        vib_mod = vib + (0.1 * damage * np.sin(2.0 * np.pi * 2.0 * f_rot * t_arr))[:, np.newaxis]
        return vib_mod, curr_mod


class CableFault(BaseFault):
    """Cable fault: intermittent current dropouts/spikes, vibration glitch bursts, no steady temp change."""

    name = "cable_fault"

    def apply_dynamics(
        self,
        damage: float,
        speed: float,
        load: float,
        temp_delta: float,
        rng: np.random.Generator,
    ) -> Tuple[float, float, float]:
        # No steady change in mean temperature
        return speed, load, temp_delta

    def apply_signals(
        self,
        damage: float,
        t_arr: np.ndarray,
        vib: np.ndarray,
        curr: np.ndarray,
        f_rot: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if damage <= 0.0:
            return vib, curr

        curr_mod = curr.copy()
        vib_mod = vib.copy()
        n_samples = len(t_arr)

        # Intermittent current dropouts / spikes with probability proportional to d
        glitch_prob = min(0.35, 0.3 * damage)
        glitch_mask = rng.random(n_samples) < glitch_prob
        if np.any(glitch_mask):
            # Alternates between dropouts (near 0) and high spikes (2x-3x)
            modes = rng.choice(["drop", "spike"], size=np.count_nonzero(glitch_mask))
            spike_factors = np.ones(np.count_nonzero(glitch_mask))
            spike_factors[modes == "drop"] = rng.uniform(0.0, 0.2, size=np.count_nonzero(modes == "drop"))
            spike_factors[modes == "spike"] = rng.uniform(2.0, 3.5, size=np.count_nonzero(modes == "spike"))
            curr_mod[glitch_mask] *= spike_factors

        # Intermittent sensor glitch bursts in vibration
        burst_prob = min(0.2, 0.15 * damage)
        if rng.random() < burst_prob:
            burst_len = rng.integers(10, min(80, n_samples))
            start_idx = rng.integers(0, n_samples - burst_len)
            burst_amp = rng.uniform(2.0, 5.0) * damage
            vib_mod[start_idx : start_idx + burst_len] += burst_amp * rng.standard_normal((burst_len, vib_mod.shape[1]))

        return vib_mod, curr_mod


FAULT_REGISTRY: Dict[str, BaseFault] = {
    "none": NoneFault(),
    "bearing": BearingFault(),
    "gear_wear": GearWearFault(),
    "motor_overheat": MotorOverheatFault(),
    "cable_fault": CableFault(),
}


def register_fault(fault_cls: Type[BaseFault]) -> None:
    """Register a new fault injector into the FAULT_REGISTRY."""
    instance = fault_cls()
    FAULT_REGISTRY[instance.name] = instance
