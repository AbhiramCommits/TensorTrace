"""Hotspot detection and tuning recommendations."""

from __future__ import annotations

from typing import Any

from tensortrace.analysis import hotspots
from tensortrace.analysis.recommend import Recommendation, generate_recommendations
from tensortrace.profilers.schema import RunResult

__all__ = ["Recommendation", "analyze_result", "generate_recommendations", "hotspots"]


def analyze_result(result: RunResult) -> dict[str, Any]:
    """Compute every hotspot view and recommendation for a run result."""
    return {
        "top_layers": hotspots.top_layers_by_latency(result, 10),
        "cumulative": hotspots.cumulative_attribution(result, 0.80),
        "memory": hotspots.memory_hotspots(result, 5),
        "cache": hotspots.cache_analysis(result.perf),
        "variance": hotspots.variance_flag(result),
        "numa": hotspots.numa_placement_check(result),
        "recommendations": generate_recommendations(result),
    }
