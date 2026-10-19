"""Shared result schema for every inference backend.

The JSON emitted by TensorTrace is versioned; ``SCHEMA_VERSION`` changes when
fields are removed or their meaning changes. Reports and comparisons rely on
this schema rather than on backend-specific structures.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class LayerTiming:
    """Aggregated timing and memory for one layer or operation type."""

    name: str
    op_type: str
    mean_ms: float
    total_ms: float
    calls: int = 1
    output_bytes: int | None = None
    self_memory_bytes: int | None = None
    percent_of_total: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerTiming:
        """Rebuild a layer row from a JSON dictionary."""
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class RunResult:
    """Complete result of one measured inference run."""

    run_id: str
    backend: str
    model: str
    config: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    layers: list[LayerTiming] = field(default_factory=list)
    op_types: list[LayerTiming] = field(default_factory=list)
    operators: list[dict[str, Any]] = field(default_factory=list)
    iteration_latencies_ms: list[float] = field(default_factory=list)
    latency: dict[str, Any] = field(default_factory=dict)
    snapshot: dict[str, Any] = field(default_factory=dict)
    time_series: list[dict[str, Any]] = field(default_factory=list)
    time_series_summary: dict[str, Any] = field(default_factory=dict)
    perf: dict[str, Any] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full result to JSON-compatible primitives."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "backend": self.backend,
            "model": self.model,
            "config": self.config,
            "environment": self.environment,
            "layers": [layer.to_dict() for layer in self.layers],
            "op_types": [layer.to_dict() for layer in self.op_types],
            "operators": self.operators,
            "iteration_latencies_ms": self.iteration_latencies_ms,
            "latency": self.latency,
            "snapshot": self.snapshot,
            "time_series": self.time_series,
            "time_series_summary": self.time_series_summary,
            "perf": self.perf,
            "system": self.system,
            "warnings": self.warnings,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        """Rebuild a result from a JSON dictionary."""
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            run_id=data.get("run_id", ""),
            backend=data.get("backend", "unknown"),
            model=data.get("model", "unknown"),
            config=data.get("config", {}),
            environment=data.get("environment", {}),
            layers=[LayerTiming.from_dict(row) for row in data.get("layers", [])],
            op_types=[LayerTiming.from_dict(row) for row in data.get("op_types", [])],
            operators=data.get("operators", []),
            iteration_latencies_ms=data.get("iteration_latencies_ms", []),
            latency=data.get("latency", {}),
            snapshot=data.get("snapshot", {}),
            time_series=data.get("time_series", []),
            time_series_summary=data.get("time_series_summary", {}),
            perf=data.get("perf", {}),
            system=data.get("system", {}),
            warnings=data.get("warnings", []),
            extra=data.get("extra", {}),
        )

    def save(self, path: str | Path) -> Path:
        """Write the result as pretty-printed JSON and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        """Load a result from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def latency_stats(values_ms: list[float]) -> dict[str, Any]:
    """Compute mean, median, p95, standard deviation, and throughput."""
    if not values_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "stddev_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "throughput_ips": 0.0,
        }
    ordered = sorted(values_ms)
    mean = statistics.fmean(ordered)
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "count": len(ordered),
        "mean_ms": round(mean, 6),
        "median_ms": round(statistics.median(ordered), 6),
        "p95_ms": round(ordered[p95_index], 6),
        "stddev_ms": round(statistics.pstdev(ordered), 6) if len(ordered) > 1 else 0.0,
        "min_ms": round(ordered[0], 6),
        "max_ms": round(ordered[-1], 6),
        "throughput_ips": round(1000.0 / mean, 6) if mean > 0 else 0.0,
    }


def assign_percentages(layers: list[LayerTiming]) -> list[LayerTiming]:
    """Set ``percent_of_total`` in place using summed layer time as the denominator."""
    total = sum(layer.total_ms for layer in layers)
    if total <= 0:
        for layer in layers:
            layer.percent_of_total = 0.0
        return layers
    for layer in layers:
        layer.percent_of_total = round(100.0 * layer.total_ms / total, 4)
    return layers
