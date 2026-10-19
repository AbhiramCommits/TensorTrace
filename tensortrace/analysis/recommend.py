"""Deterministic tuning recommendations derived from measured run data.

Each recommendation carries an ``evidence`` string built from the actual
numbers that triggered it. There is no hidden state and no randomness: the same
run JSON always produces the same ordered list.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tensortrace.analysis import hotspots
from tensortrace.profilers.schema import RunResult

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class Recommendation:
    """One actionable tuning recommendation with cited evidence."""

    severity: str
    code: str
    message: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recommendation:
        """Rebuild a recommendation from a dictionary."""
        return cls(
            severity=str(data.get("severity", "info")),
            code=str(data.get("code", "UNKNOWN")),
            message=str(data.get("message", "")),
            evidence=str(data.get("evidence", "")),
        )


def _batch_recommendation(result: RunResult) -> Recommendation | None:
    """Recommend a smaller batch when RSS grows faster than batch size."""
    sweep = result.extra.get("batch_sweep") or []
    if not isinstance(sweep, list) or len(sweep) < 2:
        return None
    usable = [
        row
        for row in sweep
        if isinstance(row, dict)
        and row.get("batch_size")
        and (row.get("rss_peak_mb") is not None or row.get("rss_mb") is not None)
    ]
    if len(usable) < 2:
        return None
    usable.sort(key=lambda row: float(row["batch_size"]))
    first, last = usable[0], usable[-1]
    first_batch = float(first["batch_size"])
    last_batch = float(last["batch_size"])
    first_rss = float(first.get("rss_peak_mb", first.get("rss_mb")))
    last_rss = float(last.get("rss_peak_mb", last.get("rss_mb")))
    if first_batch <= 0 or last_batch <= first_batch or first_rss <= 0:
        return None
    actual_ratio = last_rss / first_rss
    linear_ratio = last_batch / first_batch
    if actual_ratio <= linear_ratio * 1.20:
        return None
    return Recommendation(
        severity="warning",
        code="REDUCE_BATCH",
        message=(
            "Peak RSS grows super-linearly with batch size; reduce the batch size "
            "or add activation checkpointing before increasing it further."
        ),
        evidence=(
            f"batch {first_batch:g} -> {last_batch:g} increased peak RSS "
            f"{first_rss:.1f} -> {last_rss:.1f} MB ({actual_ratio:.2f}x, "
            f"linear would be {linear_ratio:.2f}x; threshold 1.20x linear)"
        ),
    )


def generate_recommendations(result: RunResult) -> list[Recommendation]:
    """Return ordered recommendations for one run result."""
    recommendations: list[Recommendation] = []
    numa = hotspots.numa_placement_check(result)
    if numa["numa_available"] and numa["spans_multiple_numa_nodes"]:
        recommendations.append(
            Recommendation(
                severity="warning",
                code="PIN_NUMA",
                message=(
                    "Pin the process and its worker threads to CPUs from a single NUMA "
                    "node so tensor memory and compute stay local."
                ),
                evidence=(
                    f"affinity CPUs {numa['affinity_cpus']} span NUMA nodes "
                    f"{numa['numa_nodes']}; thread_count={numa['thread_count']}"
                ),
            )
        )
    if numa["exceeds_physical_cores"]:
        recommendations.append(
            Recommendation(
                severity="warning",
                code="THREAD_OVERSUBSCRIBE",
                message=(
                    "Thread count exceeds physical cores; reduce intra-op threads to "
                    "avoid scheduler contention and SMT sharing."
                ),
                evidence=(
                    f"thread_count={numa['thread_count']} > physical_cores="
                    f"{numa['physical_cores']} (logical_cores={numa['logical_cores']})"
                ),
            )
        )
        recommendations.append(
            Recommendation(
                severity="warning",
                code="SET_THREADS",
                message=(
                    "Set OMP_NUM_THREADS and torch.set_num_threads (or ORT "
                    "intra_op_num_threads) to the physical core count and re-measure."
                ),
                evidence=(
                    f"measured thread_count={numa['thread_count']}, physical_cores="
                    f"{numa['physical_cores']}; set OMP_NUM_THREADS="
                    f"{numa['physical_cores']}"
                ),
            )
        )

    graph_level = result.extra.get("graph_optimization_level")
    if result.backend == "onnx" and graph_level is not None and graph_level != "ORT_ENABLE_ALL":
        recommendations.append(
            Recommendation(
                severity="warning",
                code="ORT_GRAPH_OPT",
                message=(
                    "Enable ONNX Runtime graph optimization "
                    "(SessionOptions.graph_optimization_level = ORT_ENABLE_ALL)."
                ),
                evidence=f"measured graph_optimization_level={graph_level!r}",
            )
        )

    batch_recommendation = _batch_recommendation(result)
    if batch_recommendation is not None:
        recommendations.append(batch_recommendation)

    variance = hotspots.variance_flag(result)
    if variance["flagged"]:
        recommendations.append(
            Recommendation(
                severity="warning",
                code="REMEASURE",
                message=(
                    "Iteration latency is unstable; re-measure on an idle host or increase "
                    "the iteration count before trusting small deltas."
                ),
                evidence=(
                    f"CV={variance['cv']:.3f} (stddev={variance['stddev_ms']:.3f} ms / "
                    f"mean={variance['mean_ms']:.3f} ms) over {variance['count']} iterations "
                    f"> threshold {variance['threshold']:.2f}"
                ),
            )
        )

    cache = hotspots.cache_analysis(result.perf)
    if cache["available"] and cache["flags"]:
        details: list[str] = []
        if "high_cache_miss_rate" in cache["flags"]:
            details.append(
                f"cache-misses={cache['cache_misses']:.0f}, "
                f"cache-references={cache['cache_references']:.0f}, "
                f"miss rate={cache['cache_miss_rate']:.4f} > "
                f"{cache['miss_rate_threshold']:.2f}"
            )
        if "low_ipc" in cache["flags"]:
            details.append(
                f"instructions={cache['instructions']:.0f}, cycles={cache['cycles']:.0f}, "
                f"IPC={cache['ipc']:.3f} < {cache['ipc_threshold']:.2f}"
            )
        recommendations.append(
            Recommendation(
                severity="warning",
                code="CACHE_BLOCKING",
                message=(
                    "Investigate cache blocking, tensor layout, and operator fusion; the "
                    "measured cache and instruction counters indicate poor locality."
                ),
                evidence="; ".join(details),
            )
        )
    if not cache["available"]:
        recommendations.append(
            Recommendation(
                severity="info",
                code="PERF_UNAVAILABLE",
                message=(
                    "Kernel counters were not collected. Run on Linux with perf installed "
                    "and perf_event_paranoid permitting user-space counting to enable "
                    "cache/IPC analysis."
                ),
                evidence=f"perf unavailable: {cache['reason'] or 'unknown reason'}",
            )
        )

    return sorted(
        recommendations,
        key=lambda rec: (_SEVERITY_ORDER.get(rec.severity, 9), rec.code, rec.message),
    )
