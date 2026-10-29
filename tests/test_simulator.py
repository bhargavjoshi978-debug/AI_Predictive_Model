"""Tests for the simulation layer (RobotArm, Joint, faults, sensors)."""

import numpy as np
import pytest
from pdm.config import SimulatorConfig, DegradationConfig
from pdm.simulator.robot import RobotArm, EpisodeConfig
from pdm.simulator.faults import FAULT_REGISTRY


def test_seed_determinism() -> None:
    """Same seed run twice produces identical observations and ground truth."""
    sim_cfg = SimulatorConfig()
    deg_cfg = DegradationConfig()
    ep_cfg = EpisodeConfig(
        robot_id=1,
        n_joints=6,
        fault_type="bearing",
        fault_joint=2,
        onset_step=10,
        degradation_rate=0.0001,
        duration_steps=50,
        seed=123,
    )

    arm1 = RobotArm(sim_cfg, deg_cfg, n_joints=6, seed=ep_cfg.seed)
    arm1.configure_episode(ep_cfg)
    obs1 = [arm1.step() for _ in range(30)]
    gt1 = arm1.ground_truth()

    arm2 = RobotArm(sim_cfg, deg_cfg, n_joints=6, seed=ep_cfg.seed)
    arm2.configure_episode(ep_cfg)
    obs2 = [arm2.step() for _ in range(30)]
    gt2 = arm2.ground_truth()

    # Check that health is identical
    np.testing.assert_array_almost_equal(gt1.health, gt2.health, decimal=6)
    assert gt1.rul_steps == gt2.rul_steps

    # Check that signals in observations are identical
    for o1, o2 in zip(obs1, obs2):
        assert o1.t == o2.t
        np.testing.assert_array_almost_equal(o1.signals["vibration"], o2.signals["vibration"], decimal=6)
        np.testing.assert_array_almost_equal(o1.signals["current"], o2.signals["current"], decimal=6)
        assert np.isclose(o1.slow["temperature_c"], o2.slow["temperature_c"], equal_nan=True)


def test_monotonic_health_degradation() -> None:
    """Health decreases monotonically after onset under a fault."""
    sim_cfg = SimulatorConfig()
    deg_cfg = DegradationConfig()
    ep_cfg = EpisodeConfig(
        robot_id=1,
        n_joints=6,
        fault_type="gear_wear",
        fault_joint=1,
        onset_step=20,
        degradation_rate=0.0002,
        duration_steps=200,
        seed=42,
    )

    arm = RobotArm(sim_cfg, deg_cfg, n_joints=6, seed=ep_cfg.seed)
    arm.configure_episode(ep_cfg)

    health_trajectory = []
    for _ in range(150):
        arm.step()
        gt = arm.ground_truth()
        health_trajectory.append(gt.health[1])

    health_arr = np.array(health_trajectory)
    # Up to onset (step 20), health should be 1.0
    assert np.all(health_arr[:19] == 1.0)
    # After onset, health must be non-increasing (monotonic degradation)
    diffs = np.diff(health_arr[20:])
    assert np.all(diffs <= 1e-6), f"Health increased during degradation: max increase = {np.max(diffs)}"
    # Final health should have decayed significantly
    assert health_arr[-1] < 0.9


def test_healthy_episodes_remain_healthy() -> None:
    """Healthy episodes keep h ≈ 1 across all steps."""
    sim_cfg = SimulatorConfig()
    deg_cfg = DegradationConfig()
    ep_cfg = EpisodeConfig(
        robot_id=1,
        n_joints=6,
        fault_type="none",
        fault_joint=None,
        onset_step=100,
        degradation_rate=0.0,
        duration_steps=100,
        seed=999,
    )

    arm = RobotArm(sim_cfg, deg_cfg, n_joints=6, seed=ep_cfg.seed)
    arm.configure_episode(ep_cfg)

    for _ in range(80):
        arm.step()
        gt = arm.ground_truth()
        assert np.all(gt.health >= 0.999), f"Healthy joint decayed: {gt.health}"


def test_all_fault_types_injected() -> None:
    """Verify that all fault types can be registered and stepped without errors."""
    for fault_name in ["bearing", "gear_wear", "motor_overheat", "cable_fault"]:
        assert fault_name in FAULT_REGISTRY
        sim_cfg = SimulatorConfig()
        deg_cfg = DegradationConfig()
        ep_cfg = EpisodeConfig(
            robot_id=1,
            n_joints=6,
            fault_type=fault_name,
            fault_joint=0,
            onset_step=5,
            degradation_rate=0.0003,
            duration_steps=30,
            seed=42,
        )
        arm = RobotArm(sim_cfg, deg_cfg, n_joints=6, seed=42)
        arm.configure_episode(ep_cfg)
        obs = [arm.step() for _ in range(25)]
        assert len(obs) == 25
        gt = arm.ground_truth()
        assert gt.fault_type == fault_name
