"""Tests for perf CSV parsing and graceful perf degradation."""

from __future__ import annotations

import tensortrace.system.perf as perf_module
from tensortrace.system.perf import (
    PerfResult,
    parse_perf_stat_csv,
    perf_available,
    perf_unavailable_reason,
    run_perf_stat,
)


def test_parse_perf_csv_mixed_supported_and_unsupported(fixtures_dir) -> None:
    counters = parse_perf_stat_csv((fixtures_dir / "perf_stat.csv").read_text())
    assert counters["cycles"] == (None, "<not supported>", None)
    assert counters["instructions"] == (None, "<not supported>", None)
    assert counters["cache-references"] == (None, "<not supported>", None)
    assert counters["cache-misses"] == (None, "<not supported>", None)
    assert counters["page-faults"] == (725.0, "", 100.0)
    assert counters["context-switches"] == (31.0, "", 100.0)
    assert len(counters) == 6


def test_parse_perf_csv_not_counted_and_summary_rows() -> None:
    text = (
        "seconds time elapsed,0.123,,\n"
        "<not counted>,,task-clock,0,0.00,,\n"
        "1234,,cycles,100,99.50,,\n"
        "# comment\n"
        "malformed\n"
    )
    counters = parse_perf_stat_csv(text)
    assert counters == {
        "task-clock": (None, "<not counted>", None),
        "cycles": (1234.0, "", 99.5),
    }


def test_parse_perf_csv_empty() -> None:
    assert parse_perf_stat_csv("") == {}


def test_perf_result_to_dict_shape() -> None:
    result = PerfResult(
        available=True,
        counters={"cycles": (100.0, "", 99.0)},
        command=["perf", "stat"],
        returncode=0,
        raw="raw text",
    )
    payload = result.to_dict()
    assert payload["available"] is True
    assert payload["counters"]["cycles"] == {"value": 100.0, "unit": "", "run_pct": 99.0}
    assert payload["command"] == ["perf", "stat"]


def test_run_perf_stat_returns_unavailable_without_perf(monkeypatch) -> None:
    monkeypatch.setattr(perf_module, "perf_available", lambda: False)
    monkeypatch.setattr(perf_module, "perf_unavailable_reason", lambda: "no perf here")
    result = run_perf_stat(["true"])
    assert result.available is False
    assert result.reason == "no perf here"
    assert result.counters == {}


def test_perf_available_is_boolean_on_this_host() -> None:
    assert isinstance(perf_available(), bool)
    if not perf_available():
        assert isinstance(perf_unavailable_reason(), str)
        assert perf_unavailable_reason()


def test_probe_reports_missing_binary(monkeypatch) -> None:
    perf_module._probe_perf.cache_clear()
    monkeypatch.setattr(perf_module.shutil, "which", lambda _name: None)
    try:
        assert perf_available() is False
        assert "not found" in perf_unavailable_reason()
    finally:
        perf_module._probe_perf.cache_clear()


def test_parse_perf_csv_rejects_bad_numeric_value() -> None:
    counters = parse_perf_stat_csv("not-a-number,,cycles,0,0,,\n")
    assert counters == {}
