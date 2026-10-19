"""Pure hotspot calculations over :class:`RunResult` objects.

Nothing in this module reads files or touches the host. That keeps the
thresholds and attribution math unit-testable from synthetic results and makes
report output deterministic for a given run JSON.
"""

from __future__ import annotations

import statistics
from typing import Any

from tensortrace.profilers.schema import LayerTiming, RunResult
from tensortrace.system.sysfs import nodes_for_cpus

DEFAULT_MISS_RATE_THRESHOLD = 0.10
DEFAULT_IPC_THRESHOLD = 1.0
DEFAULT_CV_THRESHOLD = 0.15


def top_layers_by_latency(result: RunResult, n: int = 10) -> list[LayerTiming]:
    """Return the slowest layers by total measured time."""
    return sorted(result.layers, key=lambda layer: layer.total_ms, reverse=True)[:n]


def cumulative_attribution(result: RunResult, threshold: float = 0.80) -> dict[str, Any]:
    """Attribute time cumulatively until ``threshold`` of total layer time is covered.

    ``threshold`` is a fraction (0.80 means "which layers account for 80%"). The
    returned rows include cumulative percentages so reports can show the exact
    cutoff.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    ordered = sorted(result.layers, key=lambda layer: layer.total_ms, reverse=True)
    total = sum(layer.total_ms for layer in ordered)
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for layer in ordered:
        cumulative += layer.total_ms
        rows.append(
            {
                "name": layer.name,
                "op_type": layer.op_type,
                "total_ms": round(layer.total_ms, 6),
                "percent": round(100.0 * layer.total_ms / total, 4) if total else 0.0,
                "cumulative_percent": round(100.0 * cumulative / total, 4) if total else 0.0,
            }
        )
        if total and cumulative / total >= threshold:
            break
    return {
        "threshold_percent": round(threshold * 100.0, 4),
        "layers": rows,
        "layer_count": len(rows),
        "covered_percent": round(100.0 * cumulative / total, 4) if total else 0.0,
        "total_ms": round(total, 6),
    }


def _memory_row(layer: LayerTiming, denominator: int) -> dict[str, Any]:
    """Build one memory-hotspot row with its share of total output bytes."""
    return {
        "name": layer.name,
        "op_type": layer.op_type,
        "output_bytes": layer.output_bytes,
        "self_memory_bytes": layer.self_memory_bytes,
        "percent_of_output_bytes": round(
            100.0 * (layer.output_bytes or 0) / denominator, 4
        )
        if denominator
        else 0.0,
    }


def memory_hotspots(result: RunResult, n: int = 5) -> dict[str, list[dict[str, Any]]]:
    """Return layers ranked by output bytes and by self memory.

    Missing per-layer memory (common for framework operators that do not report
    allocation) is represented by an empty list rather than zero-filled rows.
    """
    denominator = sum(layer.output_bytes or 0 for layer in result.layers)
    by_output = sorted(
        (layer for layer in result.layers if layer.output_bytes is not None),
        key=lambda layer: layer.output_bytes or 0,
        reverse=True,
    )[:n]
    by_self = sorted(
        (layer for layer in result.layers if layer.self_memory_bytes is not None),
        key=lambda layer: layer.self_memory_bytes or 0,
        reverse=True,
    )[:n]
    return {
        "by_output_bytes": [_memory_row(layer, denominator) for layer in by_output],
        "by_self_memory": [
            _memory_row(layer, denominator) for layer in by_self
        ],
    }


def _counter_value(perf: dict[str, Any] | None, event: str) -> float | None:
    """Read one counter from either a RunResult.perf dict or a raw counter map."""
    if not perf:
        return None
    counters = perf.get("counters", perf)
    if not isinstance(counters, dict) or event not in counters:
        return None
    value: Any = counters[event]
    if isinstance(value, dict):
        value = value.get("value")
    elif isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cache_miss_rate(perf: dict[str, Any] | None) -> float | None:
    """Return ``cache-misses / cache-references``, or ``None`` when unavailable."""
    misses = _counter_value(perf, "cache-misses")
    references = _counter_value(perf, "cache-references")
    if misses is None or references is None or references <= 0:
        return None
    return round(misses / references, 6)


def ipc(perf: dict[str, Any] | None) -> float | None:
    """Return instructions per cycle, or ``None`` when counters are unavailable."""
    instructions = _counter_value(perf, "instructions")
    cycles = _counter_value(perf, "cycles")
    if instructions is None or cycles is None or cycles <= 0:
        return None
    return round(instructions / cycles, 6)


def cache_analysis(
    perf: dict[str, Any] | None,
    miss_threshold: float = DEFAULT_MISS_RATE_THRESHOLD,
    ipc_threshold: float = DEFAULT_IPC_THRESHOLD,
) -> dict[str, Any]:
    """Summarize cache miss rate and IPC and flag them against thresholds."""
    miss_rate = cache_miss_rate(perf)
    instructions_per_cycle = ipc(perf)
    flags: list[str] = []
    if miss_rate is not None and miss_rate > miss_threshold:
        flags.append("high_cache_miss_rate")
    if instructions_per_cycle is not None and instructions_per_cycle < ipc_threshold:
        flags.append("low_ipc")
    return {
        "available": bool(perf and perf.get("available")),
        "reason": None if perf and perf.get("available") else (perf or {}).get("reason"),
        "cache_misses": _counter_value(perf, "cache-misses"),
        "cache_references": _counter_value(perf, "cache-references"),
        "cache_miss_rate": miss_rate,
        "miss_rate_threshold": miss_threshold,
        "instructions": _counter_value(perf, "instructions"),
        "cycles": _counter_value(perf, "cycles"),
        "ipc": instructions_per_cycle,
        "ipc_threshold": ipc_threshold,
        "flags": flags,
    }


def variance_flag(
    result: RunResult, threshold: float = DEFAULT_CV_THRESHOLD
) -> dict[str, Any]:
    """Flag noisy measurements when the coefficient of variation exceeds ``threshold``."""
    values = list(result.iteration_latencies_ms)
    mean = statistics.fmean(values) if values else float(result.latency.get("mean_ms", 0.0))
    stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
    cv = (stddev / mean) if mean > 0 else None
    return {
        "count": len(values),
        "mean_ms": round(mean, 6),
        "stddev_ms": round(stddev, 6),
        "cv": round(cv, 6) if cv is not None else None,
        "threshold": threshold,
        "flagged": bool(cv is not None and cv > threshold),
    }


def _affinity_cpus(result: RunResult) -> list[int]:
    """Extract affinity CPUs from a result across schema variants."""
    system = result.system or {}
    cpu = system.get("cpu") or {}
    affinity = cpu.get("affinity")
    if isinstance(affinity, dict):
        cpus = affinity.get("cpus") or []
    elif isinstance(affinity, list):
        cpus = affinity
    else:
        cpus = []
    if not cpus:
        cpus = result.snapshot.get("affinity") or []
    return [int(cpu) for cpu in cpus if isinstance(cpu, int)]


def numa_placement_check(result: RunResult) -> dict[str, Any]:
    """Check whether affinity spans NUMA nodes and threads exceed physical cores."""
    system = result.system or {}
    cpu = system.get("cpu") or {}
    numa = system.get("numa") or {}
    affinity = _affinity_cpus(result)
    nodes = nodes_for_cpus(affinity, numa) if numa.get("available") else []
    thread_count = (
        result.config.get("effective_threads")
        or result.config.get("threads")
        or result.extra.get("thread_count")
    )
    physical_cores = cpu.get("physical_cores")
    logical_cores = cpu.get("logical_cores")
    return {
        "numa_available": bool(numa.get("available")),
        "numa_reason": numa.get("reason"),
        "affinity_cpus": affinity,
        "numa_nodes": nodes,
        "spans_multiple_numa_nodes": len(nodes) > 1,
        "thread_count": thread_count,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "exceeds_physical_cores": bool(
            thread_count is not None
            and physical_cores is not None
            and thread_count > physical_cores
        ),
    }
