"""Metric-by-metric comparison used to gate CI on latency regressions."""

from __future__ import annotations

from typing import Any

from tensortrace.analysis import hotspots
from tensortrace.profilers.schema import RunResult


def _peak_rss_mb(result: RunResult) -> float | None:
    summary = result.time_series_summary or {}
    value = summary.get("rss_peak_mb")
    if value is not None:
        return float(value)
    memory = (result.snapshot or {}).get("memory") or {}
    peak_kb = memory.get("rss_peak_kb")
    return round(float(peak_kb) / 1024.0, 3) if peak_kb is not None else None


def _metric(
    name: str,
    baseline: float | None,
    candidate: float | None,
    *,
    higher_is_better: bool = False,
    gate: bool = False,
    unit: str = "",
    threshold_percent: float = 10.0,
) -> dict[str, Any]:
    if baseline is None or candidate is None:
        return {
            "metric": name,
            "baseline": baseline,
            "candidate": candidate,
            "delta": None,
            "percent_change": None,
            "status": "unavailable",
            "unit": unit,
            "gate": gate,
        }
    delta = candidate - baseline
    percent_change = (delta / baseline * 100.0) if baseline else None
    if percent_change is None:
        status = "unchanged"
    elif not higher_is_better:
        status = (
            "regression"
            if percent_change > threshold_percent
            else "improvement"
            if percent_change < -threshold_percent
            else "unchanged"
        )
    else:
        status = (
            "regression"
            if percent_change < -threshold_percent
            else "improvement"
            if percent_change > threshold_percent
            else "unchanged"
        )
    return {
        "metric": name,
        "baseline": round(baseline, 6),
        "candidate": round(candidate, 6),
        "delta": round(delta, 6),
        "percent_change": round(percent_change, 4) if percent_change is not None else None,
        "status": status,
        "unit": unit,
        "gate": gate,
    }


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    threshold: float = 0.10,
) -> dict[str, Any]:
    """Compare latency, throughput, memory, cache, and IPC metrics.

    ``threshold`` is a fraction (0.10 = 10%). Only mean-latency regression is a
    gating failure; other statuses are informational.
    """
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    baseline_latency = baseline.latency or {}
    candidate_latency = candidate.latency or {}
    baseline_cache = hotspots.cache_analysis(baseline.perf)
    candidate_cache = hotspots.cache_analysis(candidate.perf)
    limit = threshold * 100.0

    rows = [
        _metric(
            "mean_latency_ms",
            baseline_latency.get("mean_ms"),
            candidate_latency.get("mean_ms"),
            gate=True,
            unit="ms",
            threshold_percent=limit,
        ),
        _metric(
            "median_latency_ms",
            baseline_latency.get("median_ms"),
            candidate_latency.get("median_ms"),
            unit="ms",
            threshold_percent=limit,
        ),
        _metric(
            "p95_latency_ms",
            baseline_latency.get("p95_ms"),
            candidate_latency.get("p95_ms"),
            unit="ms",
            threshold_percent=limit,
        ),
        _metric(
            "throughput_ips",
            baseline_latency.get("throughput_ips"),
            candidate_latency.get("throughput_ips"),
            higher_is_better=True,
            unit="inferences/s",
            threshold_percent=limit,
        ),
        _metric(
            "peak_rss_mb",
            _peak_rss_mb(baseline),
            _peak_rss_mb(candidate),
            unit="MB",
            threshold_percent=limit,
        ),
        _metric(
            "cache_miss_rate",
            baseline_cache["cache_miss_rate"],
            candidate_cache["cache_miss_rate"],
            unit="ratio",
            threshold_percent=limit,
        ),
        _metric(
            "ipc",
            baseline_cache["ipc"],
            candidate_cache["ipc"],
            unit="instr/cycle",
            threshold_percent=limit,
        ),
    ]

    threshold_percent = threshold * 100.0
    for row in rows:
        if row["gate"]:
            row["regression_threshold_percent"] = threshold_percent
            if row["percent_change"] is not None:
                row["status"] = (
                    "regression"
                    if row["percent_change"] > threshold_percent
                    else "improvement"
                    if row["percent_change"] < -threshold_percent
                    else "unchanged"
                )
    gated = [row for row in rows if row["gate"]]
    regressed = [row["metric"] for row in gated if row["status"] == "regression"]
    return {
        "threshold": threshold,
        "threshold_percent": threshold_percent,
        "baseline": {
            "run_id": baseline.run_id,
            "backend": baseline.backend,
            "model": baseline.model,
            "threads": baseline.config.get("effective_threads"),
        },
        "candidate": {
            "run_id": candidate.run_id,
            "backend": candidate.backend,
            "model": candidate.model,
            "threads": candidate.config.get("effective_threads"),
        },
        "rows": rows,
        "regression": bool(regressed),
        "regressed_metrics": regressed,
    }
