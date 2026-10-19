"""Tests for pure hotspot calculations."""

from __future__ import annotations

from tensortrace.analysis.hotspots import (
    cache_analysis,
    cache_miss_rate,
    cumulative_attribution,
    ipc,
    memory_hotspots,
    numa_placement_check,
    top_layers_by_latency,
    variance_flag,
)
from tensortrace.profilers.schema import LayerTiming


def _layer(name: str, total: float, output: int | None = None, self_mem: int | None = None):
    return LayerTiming(
        name=name,
        op_type="Op",
        mean_ms=total / 10,
        total_ms=total,
        calls=10,
        output_bytes=output,
        self_memory_bytes=self_mem,
    )


def test_top_layers_ordering(result_factory) -> None:
    result = result_factory(
        layers=[_layer("a", 30), _layer("b", 50), _layer("c", 20)]
    )
    assert [layer.name for layer in top_layers_by_latency(result, 2)] == ["b", "a"]
    assert [layer.name for layer in top_layers_by_latency(result, 10)] == ["b", "a", "c"]


def test_cumulative_attribution_80_percent_math(result_factory) -> None:
    result = result_factory(layers=[_layer("a", 30), _layer("b", 50), _layer("c", 20)])
    attribution = cumulative_attribution(result, 0.80)
    assert attribution["layer_count"] == 2
    assert attribution["covered_percent"] == 80.0
    assert [row["name"] for row in attribution["layers"]] == ["b", "a"]
    assert attribution["layers"][0]["cumulative_percent"] == 50.0
    assert attribution["layers"][1]["cumulative_percent"] == 80.0
    assert attribution["total_ms"] == 100.0


def test_cumulative_attribution_empty_layers(result_factory) -> None:
    attribution = cumulative_attribution(result_factory(layers=[]), 0.80)
    assert attribution["layers"] == []
    assert attribution["covered_percent"] == 0.0


def test_memory_hotspots_rankings(result_factory) -> None:
    result = result_factory(
        layers=[
            _layer("a", 10, output=100, self_mem=10),
            _layer("b", 10, output=300, self_mem=50),
            _layer("c", 10, output=None, self_mem=100),
        ]
    )
    hotspots = memory_hotspots(result, n=2)
    assert [row["name"] for row in hotspots["by_output_bytes"]] == ["b", "a"]
    assert hotspots["by_output_bytes"][0]["percent_of_output_bytes"] == 75.0
    assert [row["name"] for row in hotspots["by_self_memory"]] == ["c", "b"]


def test_cache_miss_rate_math() -> None:
    nested = {
        "counters": {
            "cache-misses": {"value": 400.0},
            "cache-references": {"value": 1000.0},
        }
    }
    assert cache_miss_rate(nested) == 0.4
    raw = {"cache-misses": (1.0, "", 100.0), "cache-references": (10.0, "", 100.0)}
    assert cache_miss_rate(raw) == 0.1
    assert (
        cache_miss_rate({"cache-misses": {"value": None}, "cache-references": {"value": 10}})
        is None
    )
    assert (
        cache_miss_rate({"cache-misses": {"value": 1}, "cache-references": {"value": 0}})
        is None
    )
    assert cache_miss_rate({}) is None


def test_ipc_math() -> None:
    nested = {
        "counters": {
            "instructions": {"value": 3000.0},
            "cycles": {"value": 1000.0},
        }
    }
    assert ipc(nested) == 3.0
    assert ipc({"instructions": {"value": 1}, "cycles": {"value": 0}}) is None
    assert ipc({}) is None


def test_cache_analysis_flags(result_factory) -> None:
    result = result_factory(
        perf={
            "available": True,
            "reason": "",
            "counters": {
                "cache-misses": {"value": 500.0},
                "cache-references": {"value": 1000.0},
                "instructions": {"value": 500.0},
                "cycles": {"value": 1000.0},
            },
        }
    )
    analysis = cache_analysis(result.perf)
    assert analysis["cache_miss_rate"] == 0.5
    assert analysis["ipc"] == 0.5
    assert set(analysis["flags"]) == {"high_cache_miss_rate", "low_ipc"}

    clean = cache_analysis(
        {
            "available": True,
            "counters": {
                "cache-misses": {"value": 1},
                "cache-references": {"value": 100},
                "instructions": {"value": 300},
                "cycles": {"value": 100},
            },
        }
    )
    assert clean["flags"] == []

    unavailable = cache_analysis({"available": False, "reason": "no perf"})
    assert unavailable["available"] is False
    assert unavailable["reason"] == "no perf"
    assert unavailable["flags"] == []


def test_variance_flag(result_factory) -> None:
    stable = variance_flag(result_factory(iteration_latencies_ms=[10.0, 10.0, 10.0, 10.0]))
    assert stable["cv"] == 0.0
    assert stable["flagged"] is False

    noisy = variance_flag(result_factory(iteration_latencies_ms=[10.0, 20.0, 10.0, 20.0]))
    assert noisy["mean_ms"] == 15.0
    assert noisy["stddev_ms"] == 5.0
    assert noisy["cv"] == 0.333333
    assert noisy["flagged"] is True


def test_numa_placement_detects_span_and_oversubscription(result_factory) -> None:
    result = result_factory(
        config={"effective_threads": 8, "threads": 8},
        snapshot={"affinity": [0, 4], "procfs_available": False},
        system={
            "cpu": {
                "physical_cores": 4,
                "logical_cores": 8,
                "affinity": {"cpus": [0, 4], "mask": "11"},
            },
            "numa": {
                "available": True,
                "reason": None,
                "nodes": {
                    "node0": {"cpus": [0, 1, 2, 3]},
                    "node1": {"cpus": [4, 5, 6, 7]},
                },
            },
        },
    )
    check = numa_placement_check(result)
    assert check["numa_nodes"] == ["node0", "node1"]
    assert check["spans_multiple_numa_nodes"] is True
    assert check["thread_count"] == 8
    assert check["physical_cores"] == 4
    assert check["exceeds_physical_cores"] is True


def test_numa_placement_unavailable_and_empty_system(result_factory) -> None:
    unavailable = numa_placement_check(
        result_factory(
            system={
                "cpu": {"physical_cores": 4, "affinity": {"cpus": [0, 4]}},
                "numa": {"available": False, "reason": "not present", "nodes": {}},
            }
        )
    )
    assert unavailable["numa_available"] is False
    assert unavailable["spans_multiple_numa_nodes"] is False
    assert unavailable["exceeds_physical_cores"] is False

    empty = numa_placement_check(result_factory(system={}, snapshot={"affinity": []}))
    assert empty["affinity_cpus"] == []
    assert empty["physical_cores"] is None
