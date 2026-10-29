"""RobotArm and Joint physical simulation models."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np

from pdm.config import SimulatorConfig, DegradationConfig
from pdm.simulator.faults import FAULT_REGISTRY, BaseFault, NoneFault
from pdm.simulator.sensors import VirtualSensors


@dataclass
class EpisodeConfig:
    robot_id: int
    n_joints: int
    fault_type: Literal["none", "bearing", "gear_wear", "motor_overheat", "cable_fault"]
    fault_joint: Optional[int]
    onset_step: int
    degradation_rate: float
    duration_steps: int
    seed: int


@dataclass
class JointObservation:
    joint: int
    signals: Dict[str, np.ndarray]  # vibration (N, 3), current (N,), acoustic (N,)
    slow: Dict[str, float]  # temperature_c


@dataclass
class Observation:
    t: int
    signals: Dict[str, np.ndarray] = field(default_factory=dict)
    slow: Dict[str, float] = field(default_factory=dict)
    joints: List[JointObservation] = field(default_factory=list)


@dataclass
class GroundTruth:
    fault_type: str
    fault_joint: Optional[int]
    health: np.ndarray  # per joint
    rul_steps: Optional[float]


class Joint:
    """Simulates a single joint in an industrial robotic arm."""

    def __init__(
        self,
        joint_idx: int,
        sim_config: SimulatorConfig,
        deg_config: DegradationConfig,
        seed: int = 42,
    ) -> None:
        self.joint_idx = joint_idx
        self.sim_config = sim_config
        self.deg_config = deg_config
        self.rng = np.random.default_rng(seed)
        self.sensors = VirtualSensors(sim_config, seed=seed)

        self.health: float = 1.0
        self.fault: BaseFault = NoneFault()
        self.onset_step: int = 0
        self.degradation_rate: float = 0.0
        self.has_fault: bool = False

    def inject_fault(
        self,
        fault_type: str,
        onset_step: int,
        degradation_rate: float,
    ) -> None:
        """Inject a fault to be activated starting at onset_step."""
        self.fault = FAULT_REGISTRY.get(fault_type, NoneFault())
        self.onset_step = onset_step
        self.degradation_rate = degradation_rate
        self.has_fault = (fault_type != "none")

    def step(self, t: int, generate_signals: bool = True) -> Tuple[Optional[JointObservation], float]:
        """Advance physical state by 1 step and return observation and health."""
        # 1. Update pick-and-place cycle: periodic load and speed
        # Cycle period is ~20 seconds with phase offset per joint
        cycle_period = 20.0
        cycle_phase = (t / cycle_period) * 2.0 * np.pi + (self.joint_idx * np.pi / 3.0)

        # Baseline load in [0.2, 0.9]
        load = 0.35 + 0.35 * np.sin(cycle_phase) + 0.1 * np.sin(2.0 * cycle_phase)
        load = float(np.clip(load + self.rng.normal(0.0, 0.02), 0.05, 1.0))

        # Speed in rad/s, base ~ 1.0 rad/s
        speed = 0.6 + 0.4 * np.cos(cycle_phase)
        speed = float(np.clip(speed + self.rng.normal(0.0, 0.02), 0.1, 1.5))

        # Rotational frequency proportional to speed: f_rot = f_rot_base * speed
        f_rot = self.sim_config.f_rot_base * speed

        # 2. Update health degradation
        if self.has_fault and t >= self.onset_step:
            dt_fault = t - self.onset_step
            # h(t) = max(0, 1 - r * (t - onset)^p + eps_t)
            noise = self.rng.normal(0.0, self.deg_config.noise_sigma)
            ideal_decay = self.degradation_rate * (dt_fault ** self.deg_config.p)
            target_h = max(0.0, 1.0 - ideal_decay + noise)
            # Enforce near-monotonic decay
            self.health = float(min(self.health, target_h))
        else:
            # Healthy jitter around 1.0, never decaying
            self.health = 1.0

        damage = float(np.clip(1.0 - self.health, 0.0, 1.0))

        # 3. Thermal state: first-order thermal model
        t_amb = self.sim_config.t_amb
        k_load = self.sim_config.k_load
        tau = self.sim_config.tau_thermal
        # Steady state heating
        nominal_temp = t_amb + k_load * load * (1.0 - np.exp(-max(1, t) / tau))
        temp_delta = 0.0

        # Apply fault dynamics (e.g. overheating, speed ripple)
        speed, load, temp_delta = self.fault.apply_dynamics(
            damage=damage,
            speed=speed,
            load=load,
            temp_delta=temp_delta,
            rng=self.rng,
        )
        total_nominal_temp = nominal_temp + temp_delta

        if not generate_signals:
            return None, self.health

        # 4. Generate sensor observations
        signals, slow = self.sensors.sample_step(
            t=t,
            joint_idx=self.joint_idx,
            speed=speed,
            load=load,
            temp_nominal=total_nominal_temp,
            f_rot=f_rot,
        )

        # 5. Apply fault modifications to signals (e.g. bearing impacts, gear sidebands, cable glitches)
        if damage > 0.0:
            n_samples = self.sim_config.sample_rate_hz
            t_arr = np.linspace(0.0, 1.0 - (1.0 / n_samples), n_samples)
            mod_vib, mod_curr = self.fault.apply_signals(
                damage=damage,
                t_arr=t_arr,
                vib=signals["vibration"],
                curr=signals["current"],
                f_rot=f_rot,
                rng=self.rng,
            )
            signals["vibration"] = mod_vib
            signals["current"] = mod_curr

        joint_obs = JointObservation(
            joint=self.joint_idx,
            signals=signals,
            slow=slow,
        )
        return joint_obs, self.health


class RobotArm:
    """Multi-joint robot arm coordinating joints and tracking ground truth."""

    def __init__(
        self,
        sim_config: SimulatorConfig,
        deg_config: DegradationConfig,
        n_joints: int = 6,
        seed: int = 42,
    ) -> None:
        self.sim_config = sim_config
        self.deg_config = deg_config
        self.n_joints = n_joints
        self.seed = seed
        self.t: int = 0

        self.fault_type: str = "none"
        self.fault_joint: Optional[int] = None
        self.onset_step: int = 0
        self.degradation_rate: float = 0.0
        self.failure_step: Optional[int] = None

        # Build joints with deterministic distinct seeds
        self.joints: List[Joint] = [
            Joint(j, sim_config, deg_config, seed=seed + j * 100)
            for j in range(n_joints)
        ]

    def configure_episode(self, ep_cfg: EpisodeConfig) -> None:
        """Initialize an episode configuration."""
        self.t = 0
        self.fault_type = ep_cfg.fault_type
        self.fault_joint = ep_cfg.fault_joint
        self.onset_step = ep_cfg.onset_step
        self.degradation_rate = ep_cfg.degradation_rate
        self.failure_step = None

        for j in self.joints:
            if ep_cfg.fault_type != "none" and ep_cfg.fault_joint == j.joint_idx:
                j.inject_fault(ep_cfg.fault_type, ep_cfg.onset_step, ep_cfg.degradation_rate)
            else:
                j.inject_fault("none", 0, 0.0)

    def step(self, generate_signals: bool = True, primary_only: bool = False) -> Observation:
        """Advance the entire robot arm by one step and return observation."""
        self.t += 1
        joint_obs_list: List[JointObservation] = []
        health_list: List[float] = []

        primary_idx = self.fault_joint if self.fault_joint is not None else 0

        for j in self.joints:
            gen = generate_signals and (not primary_only or j.joint_idx == primary_idx)
            obs, h = j.step(self.t, generate_signals=gen)
            if obs is not None:
                joint_obs_list.append(obs)
            health_list.append(h)

        # Check for failure condition: worst joint health <= failure_threshold
        worst_health = min(health_list)
        if worst_health <= self.deg_config.failure_threshold and self.failure_step is None:
            self.failure_step = self.t

        # Arm-level aggregated observation (using worst joint or primary joint)
        arm_signals = joint_obs_list[0].signals if joint_obs_list else {}
        arm_slow = joint_obs_list[0].slow if joint_obs_list else {}

        return Observation(
            t=self.t,
            signals=arm_signals,
            slow=arm_slow,
            joints=joint_obs_list,
        )

    def ground_truth(self) -> GroundTruth:
        """Ground truth information hidden from models."""
        healths = np.array([j.health for j in self.joints], dtype=np.float64)
        rul: Optional[float] = None
        if self.fault_type != "none" and self.failure_step is not None:
            rul = max(0.0, float(self.failure_step - self.t))
        elif self.fault_type != "none":
            # Estimate theoretical failure time if not yet reached
            # 1 - r * (t_fail - onset)^p = failure_threshold
            # (t_fail - onset)^p = (1 - failure_threshold) / r
            p = self.deg_config.p
            r = self.degradation_rate
            fail_thresh = self.deg_config.failure_threshold
            if r > 0:
                steps_to_fail = ((1.0 - fail_thresh) / r) ** (1.0 / p)
                predicted_fail_t = self.onset_step + steps_to_fail
                rul = max(0.0, float(predicted_fail_t - self.t))
            else:
                rul = None
        else:
            rul = None

        return GroundTruth(
            fault_type=self.fault_type,
            fault_joint=self.fault_joint,
            health=healths,
            rul_steps=rul,
        )
