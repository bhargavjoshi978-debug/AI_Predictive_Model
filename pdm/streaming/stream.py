"""Real-time streaming simulation: yields tick-by-tick telemetry from RobotArm or replay."""

from typing import Generator, Optional, Tuple
from pdm.config import Config, load_config
from pdm.simulator.robot import EpisodeConfig, GroundTruth, Observation, RobotArm


class TelemetryStream:
    """Simulates real-time streaming telemetry from a physical or simulated robot arm."""

    def __init__(
        self,
        config: Optional[Config] = None,
        episode_config: Optional[EpisodeConfig] = None,
    ) -> None:
        self.config = config or load_config()
        self.arm = RobotArm(
            sim_config=self.config.simulator,
            deg_config=self.config.dataset.degradation,
            n_joints=self.config.simulator.n_joints,
            seed=self.config.seed,
        )

        if episode_config is not None:
            self.configure(episode_config)

    def configure(self, ep_cfg: EpisodeConfig) -> None:
        """Configure stream scenario."""
        self.arm.configure_episode(ep_cfg)

    def step(self) -> Tuple[Observation, GroundTruth]:
        """Advance one tick and yield current observation and ground truth."""
        obs = self.arm.step()
        gt = self.arm.ground_truth()
        return obs, gt

    def iterate(self, max_steps: Optional[int] = None) -> Generator[Tuple[Observation, GroundTruth], None, None]:
        """Generator yielding tick observations indefinitely or up to max_steps."""
        steps = 0
        while max_steps is None or steps < max_steps:
            steps += 1
            yield self.step()
