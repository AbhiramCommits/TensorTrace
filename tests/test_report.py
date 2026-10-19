"""Smoke tests for Markdown/HTML report generation."""

from __future__ import annotations

from pathlib import Path

from tensortrace.report.render import (
    render_comparison_markdown,
    render_html,
    render_markdown,
    render_report,
)


def _all_codes_result(result_factory, backend: str = "onnx"):
    return result_factory(
        backend=backend,
        config={
            "model": "mlp",
            "backend": backend,
            "batch_size": 1,
            "iterations": 4,
            "warmup": 1,
            "threads": 8,
            "effective_threads": 8,
            "output_dir": "reports",
            "seed": 42,
        },
        iteration_latencies_ms=[10.0, 20.0, 10.0, 20.0],
        snapshot={"affinity": [0, 4], "procfs_available": True},
        system={
            "host": {"system": "Linux", "release": "6.8"},
            "cpu": {
                "model": "Test CPU",
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
            "cache": {"available": True, "reason": None},
        },
        perf={
            "available": True,
            "reason": "",
            "counters": {
                "cache-misses": {"value": 500.0, "unit": "", "run_pct": 100.0},
                "cache-references": {"value": 1000.0, "unit": "", "run_pct": 100.0},
                "instructions": {"value": 500.0, "unit": "", "run_pct": 100.0},
                "cycles": {"value": 1000.0, "unit": "", "run_pct": 100.0},
                "page-faults": {"value": 725.0, "unit": "", "run_pct": 100.0},
            },
        },
        extra={
            "graph_optimization_level": "ORT_DISABLE_ALL",
            "batch_sweep": [
                {"batch_size": 1, "rss_peak_mb": 100.0},
                {"batch_size": 8, "rss_peak_mb": 1800.0},
            ],
        },
    )


def test_markdown_and_html_contain_model_and_every_code(result_factory) -> None:
    result = _all_codes_result(result_factory)
    markdown = render_markdown(result)
    html = render_html(result)
    expected_codes = {
        "PIN_NUMA",
        "THREAD_OVERSUBSCRIBE",
        "SET_THREADS",
        "ORT_GRAPH_OPT",
        "REDUCE_BATCH",
        "REMEASURE",
        "CACHE_BLOCKING",
    }
    for document in (markdown, html):
        assert "mlp" in document
        assert "{{" not in document
        assert "{%" not in document
        for code in expected_codes:
            assert code in document
        assert "Test CPU" in document
        assert "mean" in document.lower()


def test_report_writes_markdown_and_html_files(result_factory, tmp_path: Path) -> None:
    result = result_factory(run_id="run-2026-test")
    markdown_path, html_path = render_report(result, tmp_path)
    assert markdown_path == tmp_path / "run-2026-test.md"
    assert html_path == tmp_path / "run-2026-test.html"
    assert markdown_path.is_file()
    assert html_path.is_file()
    assert markdown_path.read_text().startswith("# TensorTrace report")
    assert "<html" in html_path.read_text()


def test_perf_unavailable_is_rendered_explicitly(result_factory) -> None:
    result = result_factory(
        perf={
            "available": False,
            "reason": "perf binary was not found on PATH",
            "counters": {},
        },
        snapshot={"procfs_available": False, "affinity": [0, 1]},
    )
    markdown = render_markdown(result)
    html = render_html(result)
    for document in (markdown, html):
        assert "PERF_UNAVAILABLE" in document
        assert "perf stat unavailable" in document
        assert "perf binary was not found on PATH" in document
        assert "cycles" in document
        assert "unavailable" in document


def test_clean_run_says_no_recommendations(result_factory) -> None:
    result = result_factory()
    markdown = render_markdown(result)
    html = render_html(result)
    assert "No recommendations triggered." in markdown
    assert "No recommendations triggered." in html


def test_comparison_markdown_marks_regression(result_factory) -> None:
    baseline = result_factory(
        run_id="base",
        latency={
            "mean_ms": 10.0,
            "median_ms": 10.0,
            "p95_ms": 10.0,
            "throughput_ips": 100.0,
        },
    )
    candidate = result_factory(
        run_id="candidate",
        latency={
            "mean_ms": 12.5,
            "median_ms": 12.5,
            "p95_ms": 12.5,
            "throughput_ips": 80.0,
        },
    )
    from tensortrace.analysis.compare import compare_runs

    markdown = render_comparison_markdown(compare_runs(baseline, candidate))
    assert "mean_latency_ms" in markdown
    assert "REGRESSION" in markdown
    assert "25.00%" in markdown
