"""Synthetic robot telemetry dataset generator (digital-twin style)."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from pdm.config import Config, load_config
from pdm.features.extract import JointFeatureExtractor
from pdm.simulator.robot import RobotArm, EpisodeConfig

logger = logging.getLogger(__name__)


def generate_dataset(
    config: Optional[Config] = None,
    output_dir: str | Path = "data",
    stride: Optional[int] = None,
) -> Tuple[Path, Path]:
    """Generate synthetic robot telemetry episodes and extracted feature table."""
    if config is None:
        config = load_config()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    base_rng = np.random.default_rng(config.seed)
    n_episodes = config.dataset.n_episodes
    min_steps = config.dataset.episode_min_steps
    max_steps = config.dataset.episode_max_steps
    actual_stride = stride if stride is not None else getattr(config.dataset, "step_stride", 10)

    # 1. Fault mix probabilities
    fault_names = list(config.dataset.fault_mix.keys())
    fault_probs = np.array(list(config.dataset.fault_mix.values()), dtype=np.float64)
    fault_probs /= np.sum(fault_probs)

    r_min = config.dataset.degradation.r_min
    r_max = config.dataset.degradation.r_max
    p_deg = config.dataset.degradation.p
    h_fail = config.dataset.degradation.failure_threshold
    obs_damage = config.dataset.degradation.fault_observable_damage
    rul_cap = config.dataset.degradation.rul_cap

    # 2. Check theoretical failure rate across planned episodes
    planned_episodes: List[Dict[str, Any]] = []
    fault_episodes_count = 0
    failing_episodes_count = 0

    for ep_id in range(n_episodes):
        fault_type = str(base_rng.choice(fault_names, p=fault_probs))
        dur_steps = int(base_rng.integers(min_steps, max_steps + 1))
        ep_seed = int(base_rng.integers(0, 10_000_000))
        ep_rng = np.random.default_rng(ep_seed)

        if fault_type == "none":
            fault_joint = None
            onset_step = dur_steps + 1
            deg_rate = 0.0
        else:
            fault_joint = int(ep_rng.integers(0, config.simulator.n_joints))
            high_bound = max(20, min(dur_steps - 30, 1500))
            low_bound = min(200, max(10, high_bound // 3))
            if low_bound >= high_bound:
                onset_step = max(5, high_bound // 2)
            else:
                onset_step = int(ep_rng.integers(low_bound, high_bound))
            deg_rate = float(ep_rng.uniform(r_min, r_max))
            fault_episodes_count += 1

            # Theoretical steps to failure
            steps_to_fail = ((1.0 - h_fail) / deg_rate) ** (1.0 / p_deg)
            if onset_step + steps_to_fail <= dur_steps:
                failing_episodes_count += 1

        planned_episodes.append({
            "ep_id": ep_id,
            "fault_type": fault_type,
            "fault_joint": fault_joint,
            "onset_step": onset_step,
            "deg_rate": deg_rate,
            "dur_steps": dur_steps,
            "seed": ep_seed,
        })

    # Validate failure rate >= 70%
    failure_ratio = (failing_episodes_count / fault_episodes_count) if fault_episodes_count > 0 else 1.0
    adjusted = False
    if fault_episodes_count > 0 and failure_ratio < 0.70:
        logger.warning(
            f"Failure rate ({failure_ratio:.1%}) is below 70%. Auto-adjusting degradation rates."
        )
        adjust_multiplier = 1.8
        r_min *= adjust_multiplier
        r_max *= adjust_multiplier
        adjusted = True

        # Re-sample degradation rates for fault episodes
        for ep in planned_episodes:
            if ep["fault_type"] != "none":
                ep_rng = np.random.default_rng(ep["seed"])
                ep_rng.integers(0, config.simulator.n_joints)  # advance rng state
                h_b = max(20, min(ep["dur_steps"] - 30, 1500))
                l_b = min(200, max(10, h_b // 3))
                if l_b < h_b:
                    ep_rng.integers(l_b, h_b)
                ep["deg_rate"] = float(ep_rng.uniform(r_min, r_max))

    # Record run_info
    run_info = {
        "n_episodes": n_episodes,
        "fault_episodes": fault_episodes_count,
        "failing_episodes": failing_episodes_count,
        "failure_ratio": failure_ratio,
        "adjusted_degradation_rate": adjusted,
        "r_min": r_min,
        "r_max": r_max,
        "stride": actual_stride,
    }
    with open(reports_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    # 3. Simulate episodes and collect raw episodes and features
    raw_records: List[Dict[str, Any]] = []
    feature_records: List[Dict[str, Any]] = []

    for ep in planned_episodes:
        ep_id = ep["ep_id"]
        robot_id = (ep_id % 10) + 1
        fault_type = ep["fault_type"]
        fault_joint = ep["fault_joint"]
        onset_step = ep["onset_step"]
        deg_rate = ep["deg_rate"]
        dur_steps = ep["dur_steps"]
        ep_seed = ep["seed"]

        ep_cfg = EpisodeConfig(
            robot_id=robot_id,
            n_joints=config.simulator.n_joints,
            fault_type=fault_type,  # type: ignore[arg-type]
            fault_joint=fault_joint,
            onset_step=onset_step,
            degradation_rate=deg_rate,
            duration_steps=dur_steps,
            seed=ep_seed,
        )

        arm = RobotArm(
            sim_config=config.simulator,
            deg_config=config.dataset.degradation,
            n_joints=config.simulator.n_joints,
            seed=ep_seed,
        )
        arm.configure_episode(ep_cfg)

        # Extractor for the monitored joint
        monitored_joint = fault_joint if fault_joint is not None else 0
        extractor = JointFeatureExtractor(
            sample_rate_hz=config.simulator.sample_rate_hz,
            n_bands=config.features.fft_bands,
            trend_window=config.features.trend_windows,
        )

        window_idx = 0
        for step_t in range(1, dur_steps + 1):
            is_stride = (step_t % actual_stride == 0)
            obs = arm.step(generate_signals=is_stride, primary_only=True)
            gt = arm.ground_truth()

            # Record if on stride
            if is_stride:
                # Joint observation for monitored joint (only primary joint generated)
                j_obs = obs.joints[0]
                h_val = float(gt.health[monitored_joint])
                damage = 1.0 - h_val

                # Raw summary for episodes.parquet
                vib = j_obs.signals["vibration"]
                curr = j_obs.signals["current"]
                acou = j_obs.signals["acoustic"]
                temp = float(j_obs.slow["temperature_c"])

                vib_x = float(np.mean(vib[:, 0]))
                vib_y = float(np.mean(vib[:, 1]))
                vib_z = float(np.mean(vib[:, 2]))
                curr_a = float(np.mean(curr))
                acou_energy = float(np.mean(acou))

                rul_val = float(gt.rul_steps) if gt.rul_steps is not None else np.nan

                raw_records.append({
                    "episode_id": ep_id,
                    "robot_id": robot_id,
                    "t": step_t,
                    "joint": monitored_joint,
                    "vibration_x": vib_x,
                    "vibration_y": vib_y,
                    "vibration_z": vib_z,
                    "temperature_c": temp,
                    "current_a": curr_a,
                    "acoustic_band_energy": acou_energy,
                    "label_fault": fault_type,
                    "label_joint": fault_joint if fault_joint is not None else -1,
                    "health": h_val,
                    "rul_steps": rul_val,
                })

                # Features extraction for features.parquet
                f_rot = config.simulator.f_rot_base
                feats = extractor.extract_step(
                    signals=j_obs.signals,
                    slow=j_obs.slow,
                    f_rot=f_rot,
                )

                # Observable damage rule: only windows where d >= 0.15 are labelled with fault class
                is_observable = (damage >= obs_damage) and (fault_type != "none")
                y_fault = fault_type if is_observable else "none"
                y_anomaly = 1 if is_observable else 0
                y_rul = min(float(rul_val), float(rul_cap)) if not np.isnan(rul_val) else np.nan

                f_row: Dict[str, Any] = {
                    "episode_id": ep_id,
                    "robot_id": robot_id,
                    "joint": monitored_joint,
                    "t": step_t,
                    "window_idx": window_idx,
                    "health": h_val,
                    "damage": damage,
                    "label_fault": fault_type,
                    "y_fault": y_fault,
                    "y_anomaly": y_anomaly,
                    "y_rul": y_rul,
                }
                f_row.update(feats)
                feature_records.append(f_row)
                window_idx += 1

    # Save to Parquet
    episodes_df = pd.DataFrame(raw_records)
    features_df = pd.DataFrame(feature_records)

    episodes_file = out_path / "episodes.parquet"
    features_file = out_path / "features.parquet"

    episodes_df.to_parquet(episodes_file, index=False)
    features_df.to_parquet(features_file, index=False)

    logger.info(f"Dataset generated: {len(episodes_df)} raw rows -> {episodes_file}")
    logger.info(f"Features generated: {len(features_df)} window rows -> {features_file}")
    logger.info(f"Class distribution in features: \n{features_df['y_fault'].value_counts(normalize=True)}")

    return episodes_file, features_file
