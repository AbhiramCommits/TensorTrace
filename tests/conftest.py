"""Shared pytest fixtures for the TensorTrace test suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tensortrace.profilers.schema import LayerTiming, RunResult

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the captured-text fixture directory."""
    return FIXTURE_DIR


@pytest.fixture
def result_factory() -> Callable[..., RunResult]:
    """Return a factory for synthetic RunResult objects with sensible defaults."""

    def make(**overrides: Any) -> RunResult:
        defaults: dict[str, Any] = {
            "run_id": "run-test-torch-mlp",
            "backend": "torch",
            "model": "mlp",
            "config": {
                "model": "mlp",
                "backend": "torch",
                "batch_size": 1,
                "iterations": 4,
                "warmup": 1,
                "threads": 4,
                "effective_threads": 4,
                "output_dir": "reports",
                "seed": 42,
            },
            "environment": {
                "platform": "Linux-test",
                "python_version": "3.11.0",
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "packages": {"torch": "2.14.1", "onnxruntime": "1.31.0"},
            },
            "layers": [
                LayerTiming(
                    name="fc1",
                    op_type="Linear",
                    mean_ms=2.0,
                    total_ms=20.0,
                    calls=10,
                    output_bytes=1024,
                    self_memory_bytes=512,
                    percent_of_total=100.0,
                )
            ],
            "op_types": [],
            "operators": [],
            "iteration_latencies_ms": [10.0, 10.0, 10.0, 10.0],
            "latency": {
                "count": 4,
                "mean_ms": 10.0,
                "median_ms": 10.0,
                "p95_ms": 10.0,
                "stddev_ms": 0.0,
                "min_ms": 10.0,
                "max_ms": 10.0,
                "throughput_ips": 100.0,
            },
            "snapshot": {
                "procfs_available": False,
                "affinity": [0, 1],
                "memory": {"source": "psutil", "rss_peak_kb": 102400},
            },
            "time_series": [],
            "time_series_summary": {
                "count": 4,
                "duration_ms": 100.0,
                "rss_start_mb": 90.0,
                "rss_peak_mb": 100.0,
                "rss_end_mb": 95.0,
                "cpu_mean_percent": 50.0,
                "cpu_peak_percent": 80.0,
                "cpu_p95_percent": 75.0,
                "process_cpu_mean_percent": 60.0,
                "process_cpu_peak_percent": 90.0,
            },
            "perf": {
                "available": True,
                "reason": "",
                "counters": {
                    "cache-misses": {"value": 20.0, "unit": "", "run_pct": 100.0},
                    "cache-references": {"value": 1000.0, "unit": "", "run_pct": 100.0},
                    "instructions": {"value": 3000.0, "unit": "", "run_pct": 100.0},
                    "cycles": {"value": 1000.0, "unit": "", "run_pct": 100.0},
                },
            },
            "system": {
                "host": {"system": "Linux", "release": "test"},
                "cpu": {
                    "model": "Test CPU",
                    "physical_cores": 4,
                    "logical_cores": 8,
                    "affinity": {"cpus": [0, 1], "mask": "3"},
                    "affinity_numa_nodes": ["node0"],
                },
                "numa": {
                    "available": True,
                    "reason": None,
                    "nodes": {
                        "node0": {
                            "node_id": 0,
                            "cpus": [0, 1, 2, 3],
                            "cpu_list": "0-3",
                            "meminfo": {},
                            "distance": [10, 21],
                        },
                        "node1": {
                            "node_id": 1,
                            "cpus": [4, 5, 6, 7],
                            "cpu_list": "4-7",
                            "meminfo": {},
                            "distance": [21, 10],
                        },
                    },
                },
                "cache": {"available": True, "reason": None, "caches": [], "levels": {}},
                "perf": {"available": True, "reason": ""},
            },
            "warnings": [],
            "extra": {},
        }
        defaults.update(overrides)
        return RunResult(**defaults)

    return make
