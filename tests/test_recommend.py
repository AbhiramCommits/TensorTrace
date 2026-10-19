"""Tests for deterministic tuning recommendations."""

from __future__ import annotations

from tensortrace.analysis.recommend import generate_recommendations


def _codes(result) -> list[str]:
    return [rec.code for rec in generate_recommendations(result)]


def test_clean_run_emits_no_recommendations(result_factory) -> None:
    assert generate_recommendations(result_factory()) == []


def test_pin_numa_fires_only_when_affinity_spans_nodes(result_factory) -> None:
    result = result_factory(
        snapshot={"affinity": [0, 4], "procfs_available": False},
        system={
            "cpu": {
                "physical_cores": 8,
                "logical_cores": 8,
                "affinity": {"cpus": [0, 4]},
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
    recommendations = generate_recommendations(result)
    assert [rec.code for rec in recommendations] == ["PIN_NUMA"]
    assert "affinity CPUs [0, 4]" in recommendations[0].evidence
    assert "node0" in recommendations[0].evidence and "node1" in recommendations[0].evidence

    single_node = result_factory(
        snapshot={"affinity": [0, 1]},
        system={
            "cpu": {"physical_cores": 8, "affinity": {"cpus": [0, 1]}},
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
    assert "PIN_NUMA" not in _codes(single_node)

    numa_unavailable = result_factory(
        config={"effective_threads": 8, "threads": 8},
        snapshot={"affinity": [0, 4]},
        system={
            "cpu": {"physical_cores": 8, "affinity": {"cpus": [0, 4]}},
            "numa": {"available": False, "reason": "not present", "nodes": {}},
        },
    )
    assert "PIN_NUMA" not in _codes(numa_unavailable)


def test_thread_oversubscription_and_set_threads(result_factory) -> None:
    result = result_factory(config={"effective_threads": 8, "threads": 8})
    recommendations = generate_recommendations(result)
    codes = [rec.code for rec in recommendations]
    assert codes == ["SET_THREADS", "THREAD_OVERSUBSCRIBE"]
    for rec in recommendations:
        assert "thread_count=8" in rec.evidence
        assert "physical_cores=4" in rec.evidence


def test_set_threads_does_not_fire_at_physical_core_count(result_factory) -> None:
    result = result_factory(config={"effective_threads": 4, "threads": 4})
    assert "SET_THREADS" not in _codes(result)


def test_ort_graph_optimization_rule(result_factory) -> None:
    disabled = result_factory(
        backend="onnx",
        extra={"graph_optimization_level": "ORT_DISABLE_ALL"},
    )
    recommendations = generate_recommendations(disabled)
    assert [rec.code for rec in recommendations] == ["ORT_GRAPH_OPT"]
    assert "ORT_DISABLE_ALL" in recommendations[0].evidence

    enabled = result_factory(
        backend="onnx",
        extra={"graph_optimization_level": "ORT_ENABLE_ALL"},
    )
    assert "ORT_GRAPH_OPT" not in _codes(enabled)


def test_reduce_batch_requires_superlinear_rss(result_factory) -> None:
    superlinear = result_factory(
        extra={
            "batch_sweep": [
                {"batch_size": 1, "rss_peak_mb": 100.0},
                {"batch_size": 8, "rss_peak_mb": 1800.0},
            ]
        }
    )
    recommendations = generate_recommendations(superlinear)
    assert [rec.code for rec in recommendations] == ["REDUCE_BATCH"]
    assert "18.00x" in recommendations[0].evidence
    assert "8.00x" in recommendations[0].evidence

    linear = result_factory(
        extra={
            "batch_sweep": [
                {"batch_size": 1, "rss_peak_mb": 100.0},
                {"batch_size": 8, "rss_peak_mb": 800.0},
            ]
        }
    )
    assert "REDUCE_BATCH" not in _codes(linear)


def test_remeasure_fires_on_high_cv(result_factory) -> None:
    result = result_factory(iteration_latencies_ms=[10.0, 20.0, 10.0, 20.0])
    recommendations = generate_recommendations(result)
    assert [rec.code for rec in recommendations] == ["REMEASURE"]
    assert "CV=0.333" in recommendations[0].evidence
    assert "threshold 0.15" in recommendations[0].evidence


def test_cache_blocking_fires_only_with_poor_counters(result_factory) -> None:
    poor = result_factory(
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
    recommendations = generate_recommendations(poor)
    assert [rec.code for rec in recommendations] == ["CACHE_BLOCKING"]
    assert "cache-misses=500" in recommendations[0].evidence
    assert "miss rate=0.5000" in recommendations[0].evidence
    assert "IPC=0.500" in recommendations[0].evidence

    good = result_factory()
    assert "CACHE_BLOCKING" not in _codes(good)


def test_perf_unavailable_is_reported_as_info(result_factory) -> None:
    result = result_factory(
        perf={"available": False, "reason": "perf binary missing", "counters": {}}
    )
    recommendations = generate_recommendations(result)
    assert [rec.code for rec in recommendations] == ["PERF_UNAVAILABLE"]
    assert recommendations[0].severity == "info"
    assert "perf binary missing" in recommendations[0].evidence


def test_recommendations_are_deterministically_ordered(result_factory) -> None:
    result = result_factory(
        iteration_latencies_ms=[10.0, 20.0, 10.0, 20.0],
        perf={
            "available": True,
            "reason": "",
            "counters": {
                "cache-misses": {"value": 500.0},
                "cache-references": {"value": 1000.0},
                "instructions": {"value": 500.0},
                "cycles": {"value": 1000.0},
            },
        },
    )
    assert _codes(result) == ["CACHE_BLOCKING", "REMEASURE"]
    assert _codes(result) == _codes(result)
