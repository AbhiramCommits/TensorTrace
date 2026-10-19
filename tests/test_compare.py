"""Tests for regression comparison and CI gating behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensortrace.analysis.compare import compare_runs
from tensortrace.cli import main


def _latency(mean: float) -> dict[str, float]:
    return {
        "count": 4,
        "mean_ms": mean,
        "median_ms": mean,
        "p95_ms": mean,
        "stddev_ms": 0.0,
        "min_ms": mean,
        "max_ms": mean,
        "throughput_ips": 1000.0 / mean,
    }


def _write(result, path: Path) -> Path:
    result.save(path)
    assert json.loads(path.read_text())["schema_version"] == 1
    return path


def test_no_regression_exits_zero(result_factory, tmp_path: Path, capsys) -> None:
    baseline = result_factory(run_id="base", latency=_latency(10.0))
    candidate = result_factory(run_id="candidate", latency=_latency(10.0))
    comparison = compare_runs(baseline, candidate)
    assert comparison["regression"] is False
    assert comparison["regressed_metrics"] == []
    assert comparison["rows"][0]["status"] == "unchanged"
    assert comparison["rows"][0]["percent_change"] == 0.0

    baseline_path = _write(baseline, tmp_path / "base.json")
    candidate_path = _write(candidate, tmp_path / "candidate.json")
    code = main(["compare", str(baseline_path), str(candidate_path)])
    output = capsys.readouterr().out
    assert code == 0
    assert "PASS" in output
    assert "base" in output and "candidate" in output


def test_25_percent_latency_regression_exits_nonzero(
    result_factory, tmp_path: Path, capsys
) -> None:
    baseline = result_factory(run_id="base", latency=_latency(10.0))
    candidate = result_factory(run_id="candidate", latency=_latency(12.5))
    comparison = compare_runs(baseline, candidate)
    assert comparison["regression"] is True
    assert comparison["regressed_metrics"] == ["mean_latency_ms"]
    mean_row = comparison["rows"][0]
    assert mean_row["delta"] == 2.5
    assert mean_row["percent_change"] == 25.0
    assert mean_row["status"] == "regression"

    baseline_path = _write(baseline, tmp_path / "base.json")
    candidate_path = _write(candidate, tmp_path / "candidate.json")
    code = main(["compare", str(baseline_path), str(candidate_path)])
    output = capsys.readouterr().out
    assert code == 1
    assert "REGRESSION" in output
    assert "mean_latency_ms" in output
    assert "25.00%" in output


def test_configurable_threshold_can_suppress_gate(
    result_factory, tmp_path: Path
) -> None:
    baseline = result_factory(run_id="base", latency=_latency(10.0))
    candidate = result_factory(run_id="candidate", latency=_latency(12.5))
    assert compare_runs(baseline, candidate, threshold=0.30)["regression"] is False

    baseline_path = _write(baseline, tmp_path / "base.json")
    candidate_path = _write(candidate, tmp_path / "candidate.json")
    assert (
        main(
            [
                "compare",
                str(baseline_path),
                str(candidate_path),
                "--threshold",
                "0.30",
            ]
        )
        == 0
    )


def test_non_gating_metric_regressions_do_not_fail(result_factory) -> None:
    baseline = result_factory(run_id="base", latency=_latency(10.0))
    candidate_latency = _latency(10.0)
    candidate_latency["p95_ms"] = 20.0
    candidate_latency["throughput_ips"] = 50.0
    candidate = result_factory(run_id="candidate", latency=candidate_latency)
    comparison = compare_runs(baseline, candidate)
    assert comparison["regression"] is False
    rows = {row["metric"]: row for row in comparison["rows"]}
    assert rows["mean_latency_ms"]["status"] == "unchanged"
    assert rows["p95_latency_ms"]["status"] == "regression"
    assert rows["throughput_ips"]["status"] == "regression"


def test_cache_and_memory_metrics_are_compared(result_factory) -> None:
    baseline = result_factory(
        perf={
            "available": True,
            "reason": "",
            "counters": {
                "cache-misses": {"value": 20.0},
                "cache-references": {"value": 1000.0},
                "instructions": {"value": 3000.0},
                "cycles": {"value": 1000.0},
            },
        },
        time_series_summary={"rss_peak_mb": 100.0},
    )
    candidate = result_factory(
        perf={
            "available": True,
            "reason": "",
            "counters": {
                "cache-misses": {"value": 400.0},
                "cache-references": {"value": 1000.0},
                "instructions": {"value": 500.0},
                "cycles": {"value": 1000.0},
            },
        },
        time_series_summary={"rss_peak_mb": 150.0},
    )
    comparison = compare_runs(baseline, candidate)
    rows = {row["metric"]: row for row in comparison["rows"]}
    assert rows["cache_miss_rate"]["baseline"] == 0.02
    assert rows["cache_miss_rate"]["candidate"] == 0.4
    assert rows["cache_miss_rate"]["status"] == "regression"
    assert rows["ipc"]["baseline"] == 3.0
    assert rows["ipc"]["candidate"] == 0.5
    assert rows["peak_rss_mb"]["percent_change"] == 50.0


def test_missing_metrics_are_marked_unavailable(result_factory) -> None:
    baseline = result_factory(perf={"available": False, "reason": "no perf", "counters": {}})
    candidate = result_factory(perf={"available": False, "reason": "no perf", "counters": {}})
    comparison = compare_runs(baseline, candidate)
    rows = {row["metric"]: row for row in comparison["rows"]}
    assert rows["cache_miss_rate"]["status"] == "unavailable"
    assert rows["ipc"]["status"] == "unavailable"


def test_negative_threshold_rejected(result_factory) -> None:
    with pytest.raises(ValueError):
        compare_runs(result_factory(), result_factory(), threshold=-0.1)
