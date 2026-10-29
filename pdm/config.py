"""Configuration loader and typed dataclasses for the predictive maintenance simulator."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class NoiseConfig:
    sigma_v: float = 0.05
    sigma_i: float = 0.03
    sigma_t: float = 0.3
    sigma_a: float = 0.02


@dataclass
class DriftConfig:
    rate_per_1000: float = 0.0002


@dataclass
class SensorImperfectionsConfig:
    dropout_prob: float = 0.001
    glitch_prob: float = 0.0005
    glitch_multiplier: float = 3.0


@dataclass
class SimulatorConfig:
    n_joints: int = 6
    sample_rate_hz: int = 1000
    step_seconds: float = 1.0
    f_rot_base: float = 5.0
    i0_base: float = 5.0
    t_amb: float = 25.0
    k_load: float = 15.0
    tau_thermal: float = 120.0
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    sensor_imperfections: SensorImperfectionsConfig = field(default_factory=SensorImperfectionsConfig)


@dataclass
class DegradationConfig:
    r_min: float = 0.00002
    r_max: float = 0.00008
    p: float = 1.5
    noise_sigma: float = 0.002
    failure_threshold: float = 0.1
    fault_observable_damage: float = 0.15
    rul_cap: int = 300


@dataclass
class DatasetConfig:
    n_episodes: int = 400
    episode_min_steps: int = 2000
    episode_max_steps: int = 6000
    step_stride: int = 10
    fault_mix: Dict[str, float] = field(default_factory=lambda: {
        "none": 0.20,
        "bearing": 0.20,
        "gear_wear": 0.20,
        "motor_overheat": 0.20,
        "cable_fault": 0.20
    })
    split: Dict[str, float] = field(default_factory=lambda: {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15
    })
    degradation: DegradationConfig = field(default_factory=DegradationConfig)


@dataclass
class FeaturesConfig:
    window_len: int = 1024
    hop: int = 512
    fft_bands: int = 8
    trend_windows: int = 5
    seq_len: int = 20
    seq_features_top_k: int = 30


@dataclass
class RandomForestConfig:
    n_estimators: int = 200
    max_depth: int = 12
    class_weight: str = "balanced"
    random_state: int = 42


@dataclass
class XGBoostConfig:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.08
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    tree_method: str = "hist"
    early_stopping_rounds: int = 20


@dataclass
class AutoencoderConfig:
    latent_dim: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    learning_rate: float = 0.001
    epochs: int = 60
    batch_size: int = 256
    patience: int = 10
    threshold_percentile: float = 99.0


@dataclass
class LSTMConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 40
    batch_size: int = 128
    patience: int = 10


@dataclass
class ModelsConfig:
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)


@dataclass
class AlertsConfig:
    warn_threshold: float = 0.5
    critical_threshold: float = 0.8
    hysteresis_warn_steps: int = 3
    hysteresis_critical_steps: int = 3
    recovery_warn_steps: int = 5
    recovery_critical_steps: int = 5


@dataclass
class EdgeConfig:
    profile: str = "raspberry_pi_4"
    latency_ms_max: float = 50.0
    size_mb_max: float = 50.0
    ram_mb_max: float = 512.0
    slowdown_factor: float = 5.0
    test_windows: int = 1000


@dataclass
class FederatedConfig:
    enabled: bool = False
    cells: int = 3
    rounds: int = 10
    local_epochs: int = 3


@dataclass
class Config:
    seed: int = 42
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)


def _dict_to_dataclass(cls: Any, data: Dict[str, Any]) -> Any:
    """Recursively convert nested dicts into dataclass instances."""
    if not isinstance(data, dict):
        return data
    fields = cls.__dataclass_fields__
    kwargs = {}
    for f_name, f_type in fields.items():
        if f_name in data:
            val = data[f_name]
            f_cls = f_type.type
            if isinstance(f_cls, type) and hasattr(f_cls, "__dataclass_fields__"):
                kwargs[f_name] = _dict_to_dataclass(f_cls, val)
            else:
                kwargs[f_name] = val
    return cls(**kwargs)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate the YAML configuration into a typed Config dataclass."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config(
        seed=raw.get("seed", 42),
        simulator=SimulatorConfig(
            n_joints=raw.get("simulator", {}).get("n_joints", 6),
            sample_rate_hz=raw.get("simulator", {}).get("sample_rate_hz", 1000),
            step_seconds=raw.get("simulator", {}).get("step_seconds", 1.0),
            f_rot_base=raw.get("simulator", {}).get("f_rot_base", 5.0),
            i0_base=raw.get("simulator", {}).get("i0_base", 5.0),
            t_amb=raw.get("simulator", {}).get("t_amb", 25.0),
            k_load=raw.get("simulator", {}).get("k_load", 15.0),
            tau_thermal=raw.get("simulator", {}).get("tau_thermal", 120.0),
            noise=NoiseConfig(**raw.get("simulator", {}).get("noise", {})),
            drift=DriftConfig(**raw.get("simulator", {}).get("drift", {})),
            sensor_imperfections=SensorImperfectionsConfig(
                **raw.get("simulator", {}).get("sensor_imperfections", {})
            ),
        ),
        dataset=DatasetConfig(
            n_episodes=raw.get("dataset", {}).get("n_episodes", 400),
            episode_min_steps=raw.get("dataset", {}).get("episode_min_steps", 2000),
            episode_max_steps=raw.get("dataset", {}).get("episode_max_steps", 6000),
            step_stride=raw.get("dataset", {}).get("step_stride", 10),
            fault_mix=raw.get("dataset", {}).get("fault_mix", {
                "none": 0.20,
                "bearing": 0.20,
                "gear_wear": 0.20,
                "motor_overheat": 0.20,
                "cable_fault": 0.20
            }),
            split=raw.get("dataset", {}).get("split", {"train": 0.70, "val": 0.15, "test": 0.15}),
            degradation=DegradationConfig(**raw.get("dataset", {}).get("degradation", {})),
        ),
        features=FeaturesConfig(**raw.get("features", {})),
        models=ModelsConfig(
            random_forest=RandomForestConfig(**raw.get("models", {}).get("random_forest", {})),
            xgboost=XGBoostConfig(**raw.get("models", {}).get("xgboost", {})),
            autoencoder=AutoencoderConfig(**raw.get("models", {}).get("autoencoder", {})),
            lstm=LSTMConfig(**raw.get("models", {}).get("lstm", {})),
        ),
        alerts=AlertsConfig(**raw.get("alerts", {})),
        edge=EdgeConfig(**raw.get("edge", {})),
        federated=FederatedConfig(**raw.get("federated", {})),
    )
    return cfg
