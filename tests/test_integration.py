"""Integration tests that execute real inference through the full pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tensortrace.profilers.runner as runner_module
from tensortrace.config import RunConfig
from tensortrace.profilers.runner import profile_run
from tensortrace.report.render import render_report

SCHEMA_KEYS = {
    "schema_version",
    "run_id",
    "backend",
    "model",
    "config",
    "environment",
    "layers",
    "op_types",
    "operators",
    "iteration_latencies_ms",
    "latency",
    "snapshot",
    "time_series",
    "time_series_summary",
    "perf",
    "system",
    "warnings",
    "extra",
}


@pytest.mark.slow
def test_profile_run_real_tiny_model(tmp_path: Path) -> None:
    config = RunConfig(
        model="mlp",
        backend="torch",
        batch_size=1,
        iterations=2,
        warmup=1,
        threads=1,
        output_dir=tmp_path,
        seed=42,
    )
    result = profile_run(config)
    payload = result.to_dict()
    assert set(payload) == SCHEMA_KEYS
    assert payload["schema_version"] == 1
    assert payload["backend"] == "torch"
    assert payload["model"] == "mlp"
    assert payload["iteration_latencies_ms"]
    assert payload["latency"]["mean_ms"] > 0
    assert payload["latency"]["count"] == 2
    assert payload["layers"]
    assert isinstance(payload["perf"]["available"], bool)
    assert payload["environment"]["packages"]["torch"]
    assert payload["system"]["cpu"]["model"]
    assert payload["time_series_summary"]["count"] >= 1
    json.dumps(payload)

    saved = result.save(tmp_path / f"{result.run_id}.json")
    assert saved.is_file()
    assert json.loads(saved.read_text())["run_id"] == result.run_id


def test_pipeline_degrades_gracefully_without_perf(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner_module, "perf_available", lambda: False)
    monkeypatch.setattr(
        runner_module, "perf_unavailable_reason", lambda: "test: perf disabled"
    )
    config = RunConfig(
        model="mlp",
        backend="torch",
        batch_size=1,
        iterations=1,
        warmup=0,
        threads=1,
        output_dir=tmp_path,
        seed=7,
    )
    result = profile_run(config)
    assert result.perf["available"] is False
    assert result.perf["reason"] == "test: perf disabled"
    assert result.latency["mean_ms"] > 0

    markdown_path, html_path = render_report(result, tmp_path)
    markdown = markdown_path.read_text()
    html = html_path.read_text()
    assert "perf stat unavailable: test: perf disabled" in markdown
    assert "PERF_UNAVAILABLE" in markdown
    assert "PERF_UNAVAILABLE" in html
    assert "unavailable" in html
