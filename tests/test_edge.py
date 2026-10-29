"""Tests for edge feasibility benchmark and budget verification."""

import numpy as np
import pytest
from pdm.config import EdgeConfig
from pdm.edge.benchmark import run_single_model_benchmark, benchmark_models
from pdm.models.xgboost_model import XGBoostModel


def test_edge_benchmark_fields_and_budget() -> None:
    """Assert edge benchmark returns all required fields and handles budget logic."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 20)).astype(np.float32)
    y_cls = rng.choice(["none", "bearing"], size=100)

    model = XGBoostModel()
    model.fit((X, y_cls))

    edge_cfg = EdgeConfig(
        profile="raspberry_pi_4",
        latency_ms_max=50.0,
        size_mb_max=50.0,
        ram_mb_max=512.0,
        slowdown_factor=5.0,
        test_windows=50,
    )

    res = run_single_model_benchmark(model, X, edge_cfg, n_runs=50)

    # Required fields
    expected_fields = [
        "model",
        "latency_median_ms",
        "latency_p95_ms",
        "edge_latency_median_ms",
        "edge_latency_p95_ms",
        "size_mb",
        "ram_delta_mb",
        "throughput_win_per_sec",
        "pass_latency",
        "pass_size",
        "pass_ram",
        "edge_ok",
        "slowdown_factor",
    ]
    for field in expected_fields:
        assert field in res, f"Missing field in edge benchmark: {field}"

    assert res["latency_median_ms"] > 0.0
    assert res["edge_latency_median_ms"] == pytest.approx(res["latency_median_ms"] * 5.0, rel=1e-3)
    assert isinstance(res["edge_ok"], (bool, np.bool_))

    # Test failure when budget is impossibly tight
    tight_cfg = EdgeConfig(latency_ms_max=0.000001, size_mb_max=0.000001, ram_mb_max=0.000001)
    tight_res = run_single_model_benchmark(model, X, tight_cfg, n_runs=10)
    assert not tight_res["edge_ok"]
    assert not tight_res["pass_latency"]
