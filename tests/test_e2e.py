"""End-to-End pipeline test on a tiny configuration (< 2 minutes)."""

import tempfile
import time
from pathlib import Path
import pytest

from pdm.config import (
    AutoencoderConfig,
    Config,
    DatasetConfig,
    DegradationConfig,
    EdgeConfig,
    LSTMConfig,
    ModelsConfig,
    RandomForestConfig,
    SimulatorConfig,
    XGBoostConfig,
)
from pdm.evaluation.report import ExperimentRunner
from pdm.features.dataset import fit_healthy_scaler, split_by_episode
from pdm.models.autoencoder import AutoencoderModel
from pdm.models.lstm import LSTMModel
from pdm.models.random_forest import RandomForestModel
from pdm.models.xgboost_model import XGBoostModel
from pdm.simulator.generator import generate_dataset


def test_e2e_pipeline_fast() -> None:
    """Run full pipeline (generate -> train -> evaluate) on tiny config in < 2 minutes."""
    t_start = time.perf_counter()

    tiny_cfg = Config(
        seed=42,
        simulator=SimulatorConfig(n_joints=3, sample_rate_hz=500),
        dataset=DatasetConfig(
            n_episodes=15,
            episode_min_steps=80,
            episode_max_steps=150,
            step_stride=5,
            degradation=DegradationConfig(r_min=0.0005, r_max=0.002, rul_cap=100),
        ),
        models=ModelsConfig(
            random_forest=RandomForestConfig(n_estimators=10, max_depth=5),
            xgboost=XGBoostConfig(n_estimators=15, max_depth=4),
            autoencoder=AutoencoderConfig(epochs=3, batch_size=32),
            lstm=LSTMConfig(epochs=3, batch_size=32),
        ),
        edge=EdgeConfig(test_windows=50),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = Path(tmp_dir) / "data"
        models_dir = Path(tmp_dir) / "models"
        reports_dir = Path(tmp_dir) / "reports"

        # 1. Generate
        ep_file, feat_file = generate_dataset(tiny_cfg, output_dir=data_dir)
        assert ep_file.exists()
        assert feat_file.exists()

        # 2. Evaluate & Train via ExperimentRunner
        runner = ExperimentRunner(config=tiny_cfg, data_dir=data_dir, reports_dir=reports_dir)
        res_df = runner.run_all()

        assert not res_df.empty
        assert (reports_dir / "results.csv").exists()
        assert (reports_dir / "results.md").exists()

        # Verify results.md contains required disclaimer
        with open(reports_dir / "results.md", "r", encoding="utf-8") as f:
            content = f.read()
            assert "Results are from synthetic data and are not validated on real robots." in content

    total_time = time.perf_counter() - t_start
    assert total_time < 120.0, f"E2E test took {total_time:.1f}s, exceeding 2 minute budget."
