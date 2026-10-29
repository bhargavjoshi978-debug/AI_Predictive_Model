"""Edge feasibility benchmark: single-window latency, memory, size, and budget check."""

import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import psutil

from pdm.config import EdgeConfig
from pdm.models.base import BaseModel


def run_single_model_benchmark(
    model: BaseModel,
    test_windows: np.ndarray,
    edge_config: EdgeConfig,
    n_runs: int = 1000,
) -> Dict[str, Any]:
    """Measure single-window inference latency, peak RAM delta, and model size against an edge budget."""
    process = psutil.Process(os.getpid())
    n_samples = min(len(test_windows), n_runs)
    if n_samples == 0:
        raise ValueError("test_windows is empty.")

    # Warm-up (10 runs)
    for i in range(min(10, n_samples)):
        sample = test_windows[i : i + 1]
        try:
            model.predict(sample)
        except Exception:
            pass

    # Latency measurement (single window, batch size 1)
    latencies_ms: List[float] = []
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB

    t_start = time.perf_counter()
    for i in range(n_samples):
        sample = test_windows[i : i + 1]
        t0 = time.perf_counter()
        _ = model.predict(sample)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    t_end = time.perf_counter()

    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
    ram_delta_mb = max(0.0, mem_after - mem_before)

    total_time_sec = t_end - t_start
    throughput = float(n_samples / total_time_sec) if total_time_sec > 0 else 0.0

    lat_p50 = float(np.median(latencies_ms))
    lat_p95 = float(np.percentile(latencies_ms, 95))

    # Apply slowdown factor for edge emulation (e.g. 5x for Raspberry Pi 4 vs laptop)
    slowdown = edge_config.slowdown_factor
    edge_lat_p50 = lat_p50 * slowdown
    edge_lat_p95 = lat_p95 * slowdown

    # Serialized model size
    size_mb = float(model.size_bytes() / (1024 * 1024))

    # Edge budget pass/fail check
    pass_latency = edge_lat_p95 <= edge_config.latency_ms_max
    pass_size = size_mb <= edge_config.size_mb_max
    pass_ram = ram_delta_mb <= edge_config.ram_mb_max
    edge_ok = pass_latency and pass_size and pass_ram

    return {
        "model": model.name,
        "latency_median_ms": lat_p50,
        "latency_p95_ms": lat_p95,
        "edge_latency_median_ms": edge_lat_p50,
        "edge_latency_p95_ms": edge_lat_p95,
        "size_mb": size_mb,
        "ram_delta_mb": ram_delta_mb,
        "throughput_win_per_sec": throughput,
        "pass_latency": pass_latency,
        "pass_size": pass_size,
        "pass_ram": pass_ram,
        "edge_ok": edge_ok,
        "slowdown_factor": slowdown,
    }


def benchmark_models(
    models: List[BaseModel],
    test_windows: np.ndarray,
    edge_config: EdgeConfig,
    n_runs: int = 1000,
) -> pd.DataFrame:
    """Run edge benchmark on multiple models and return comparison table."""
    records = []
    for m in models:
        res = run_single_model_benchmark(m, test_windows, edge_config, n_runs=n_runs)
        records.append(res)
    return pd.DataFrame(records)
