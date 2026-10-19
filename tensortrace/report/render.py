"""Markdown and HTML report rendering for TensorTrace run results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from tensortrace.analysis import analyze_result
from tensortrace.profilers.schema import LayerTiming, RunResult
from tensortrace.system.perf import DEFAULT_EVENTS

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _fmt(value: Any, digits: int = 3) -> str:
    """Format a number for a table, using ``unavailable`` for missing values."""
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.{digits}f}"
        return f"{value:.{digits}f}"
    return str(value)


def _md_cell(value: Any) -> str:
    """Escape a value for a Markdown table cell."""
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Render a Markdown table as a list of lines."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(cell) for cell in row) + " |")
    return lines


def _environment_rows(result: RunResult) -> list[list[Any]]:
    environment = result.environment or {}
    system = result.system or {}
    host = system.get("host") or {}
    cpu = system.get("cpu") or {}
    numa = system.get("numa") or {}
    cache = system.get("cache") or {}
    perf = result.perf or {}
    packages = environment.get("packages") or {}
    affinity = cpu.get("affinity") or {}
    rows: list[list[Any]] = [
        ["Run ID", result.run_id],
        ["Platform", environment.get("platform")],
        ["Host OS", f"{host.get('system', 'unavailable')} {host.get('release', '')}".strip()],
        ["CPU model", cpu.get("model")],
        ["Physical / logical cores", f"{cpu.get('physical_cores')} / {cpu.get('logical_cores')}"],
        ["Affinity", f"mask={affinity.get('mask')} cpus={affinity.get('cpus')}"],
        ["NUMA", "available" if numa.get("available") else f"unavailable ({numa.get('reason')})"],
        ["Cache topology", "available" if cache.get("available") else "unavailable"],
        ["Python", environment.get("python_version")],
        [
            "perf stat",
            "available" if perf.get("available") else f"unavailable ({perf.get('reason')})",
        ],
        [
            "procfs",
            "available" if (result.snapshot or {}).get("procfs_available") else "unavailable",
        ],
        ["Measured at", environment.get("timestamp_utc")],
    ]
    for name in ("torch", "onnxruntime", "onnx", "numpy", "psutil"):
        if name in packages:
            rows.append([name, packages[name]])
    return rows


def _config_rows(result: RunResult) -> list[list[Any]]:
    config = result.config or {}
    extra = result.extra or {}
    model = extra.get("model") or {}
    rows: list[list[Any]] = [
        ["Backend", result.backend],
        ["Model", result.model],
        ["Batch size", config.get("batch_size")],
        ["Iterations", config.get("iterations")],
        ["Warmup iterations", config.get("warmup")],
        ["Threads requested", config.get("threads")],
        ["Threads effective", config.get("effective_threads")],
        ["Seed", config.get("seed")],
    ]
    if extra.get("input_shape"):
        rows.append(["Input shape", extra["input_shape"]])
    if model.get("parameter_count") is not None:
        rows.append(["Parameters", model["parameter_count"]])
    return rows


def _latency_rows(result: RunResult) -> list[list[Any]]:
    latency = result.latency or {}
    return [
        ["Iterations", latency.get("count")],
        ["Mean", f"{_fmt(latency.get('mean_ms'))} ms"],
        ["Median", f"{_fmt(latency.get('median_ms'))} ms"],
        ["p95", f"{_fmt(latency.get('p95_ms'))} ms"],
        ["Stddev", f"{_fmt(latency.get('stddev_ms'))} ms"],
        ["Min / max", f"{_fmt(latency.get('min_ms'))} / {_fmt(latency.get('max_ms'))} ms"],
        ["Throughput", f"{_fmt(latency.get('throughput_ips'))} inferences/s"],
    ]


def _perf_rows(result: RunResult) -> list[dict[str, Any]]:
    perf = result.perf or {}
    available = bool(perf.get("available"))
    reason = perf.get("reason") or ""
    counters = perf.get("counters") or {}
    events = list(DEFAULT_EVENTS) + [event for event in counters if event not in DEFAULT_EVENTS]
    rows: list[dict[str, Any]] = []
    for event in events:
        if not available:
            rows.append(
                {
                    "event": event,
                    "value": "unavailable",
                    "unit": "-",
                    "run_pct": None,
                    "status": reason or "perf unavailable",
                }
            )
            continue
        entry = counters.get(event)
        if not isinstance(entry, dict):
            rows.append(
                {
                    "event": event,
                    "value": "not recorded",
                    "unit": "-",
                    "run_pct": None,
                    "status": "not present in perf output",
                }
            )
            continue
        value = entry.get("value")
        unit = entry.get("unit") or ""
        run_pct = entry.get("run_pct")
        if value is None:
            rows.append(
                {
                    "event": event,
                    "value": "unavailable",
                    "unit": unit or "-",
                    "run_pct": None,
                    "status": unit or "not counted",
                }
            )
        else:
            rows.append(
                {
                    "event": event,
                    "value": _fmt(value, 0),
                    "unit": unit or "count",
                    "run_pct": run_pct,
                    "status": "measured",
                }
            )
    return rows


def _time_series_rows(result: RunResult) -> list[list[Any]]:
    summary = result.time_series_summary or {}
    memory = (result.snapshot or {}).get("memory") or {}
    return [
        ["Samples", summary.get("count")],
        ["Series duration", f"{_fmt(summary.get('duration_ms'))} ms"],
        ["RSS start / peak / end", (
            f"{_fmt(summary.get('rss_start_mb'))} / {_fmt(summary.get('rss_peak_mb'))} / "
            f"{_fmt(summary.get('rss_end_mb'))} MB"
        )],
        ["RSS source", memory.get("source")],
        ["CPU mean / peak / p95", (
            f"{_fmt(summary.get('cpu_mean_percent'))}% / "
            f"{_fmt(summary.get('cpu_peak_percent'))}% / "
            f"{_fmt(summary.get('cpu_p95_percent'))}%"
        )],
        ["Process CPU mean / peak", (
            f"{_fmt(summary.get('process_cpu_mean_percent'))}% / "
            f"{_fmt(summary.get('process_cpu_peak_percent'))}%"
        )],
    ]


def _recommendation_rows(result: RunResult) -> list[dict[str, Any]]:
    return [rec.to_dict() for rec in analyze_result(result)["recommendations"]]


def build_context(result: RunResult) -> dict[str, Any]:
    """Assemble every value the Markdown writer and HTML template need."""
    analysis = analyze_result(result)
    top_layers: list[LayerTiming] = analysis["top_layers"]
    return {
        "title": f"TensorTrace report - {result.backend}/{result.model}",
        "run_id": result.run_id,
        "result": result.to_dict(),
        "env_rows": _environment_rows(result),
        "config_rows": _config_rows(result),
        "latency_rows": _latency_rows(result),
        "top_layers": [layer.to_dict() for layer in top_layers],
        "cumulative": analysis["cumulative"],
        "memory": analysis["memory"],
        "cache": analysis["cache"],
        "variance": analysis["variance"],
        "numa": analysis["numa"],
        "perf_rows": _perf_rows(result),
        "perf_available": bool((result.perf or {}).get("available")),
        "perf_reason": (result.perf or {}).get("reason") or "",
        "time_series_rows": _time_series_rows(result),
        "recommendations": [rec.to_dict() for rec in analysis["recommendations"]],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def render_markdown(result: RunResult) -> str:
    """Render a complete Markdown report for one run."""
    context = build_context(result)
    lines: list[str] = [
        f"# {context['title']}",
        "",
        f"**Run ID:** `{result.run_id}`  ",
        f"**Generated:** {context['generated_at']}",
        "",
        "## Environment",
        "",
        *_table(["Field", "Value"], context["env_rows"]),
        "",
        "## Configuration",
        "",
        *_table(["Field", "Value"], context["config_rows"]),
        "",
        "## Latency summary",
        "",
        *_table(["Metric", "Value"], context["latency_rows"]),
        "",
        "## Top 10 layers by total latency",
        "",
        *_table(
            ["#", "Layer", "Operation", "Mean ms", "Total ms", "% of total", "Output bytes"],
            [
                [
                    index + 1,
                    layer["name"],
                    layer["op_type"],
                    _fmt(layer["mean_ms"]),
                    _fmt(layer["total_ms"]),
                    _fmt(layer["percent_of_total"], 1),
                    _fmt(layer["output_bytes"]),
                ]
                for index, layer in enumerate(context["top_layers"])
            ],
        ),
        "",
        "## Cumulative attribution",
        "",
        (
            f"Layers covering {_fmt(context['cumulative']['threshold_percent'], 0)}% of measured "
            f"layer time: {context['cumulative']['layer_count']} "
            f"(covered {_fmt(context['cumulative']['covered_percent'], 1)}%)."
        ),
        "",
        *_table(
            ["Layer", "Total ms", "%", "Cumulative %"],
            [
                [
                    row["name"],
                    _fmt(row["total_ms"]),
                    _fmt(row["percent"], 1),
                    _fmt(row["cumulative_percent"], 1),
                ]
                for row in context["cumulative"]["layers"]
            ],
        ),
        "",
        "## Memory hotspots (output bytes)",
        "",
        *_table(
            ["Layer", "Operation", "Output bytes", "% of output bytes"],
            [
                [
                    row["name"],
                    row["op_type"],
                    _fmt(row["output_bytes"]),
                    _fmt(row["percent_of_output_bytes"], 1),
                ]
                for row in context["memory"]["by_output_bytes"]
            ],
        ),
        "",
        "## Kernel counters (perf stat)",
        "",
        *(
            [f"perf stat unavailable: {context['perf_reason']}", ""]
            if not context["perf_available"]
            else []
        ),
        *_table(
            ["Event", "Value", "Unit", "Run %", "Status"],
            [
                [
                    row["event"],
                    row["value"],
                    row["unit"],
                    _fmt(row["run_pct"], 2),
                    row["status"],
                ]
                for row in context["perf_rows"]
            ],
        ),
        "",
        "## CPU and memory time series",
        "",
        *_table(["Metric", "Value"], context["time_series_rows"]),
        "",
        "## Hotspot flags",
        "",
        *_table(
            ["Check", "Measured", "Threshold", "Flagged"],
            [
                [
                    "Cache miss rate",
                    _fmt(context["cache"]["cache_miss_rate"], 4),
                    _fmt(context["cache"]["miss_rate_threshold"], 2),
                    "high_cache_miss_rate" in context["cache"]["flags"],
                ],
                [
                    "IPC",
                    _fmt(context["cache"]["ipc"], 3),
                    _fmt(context["cache"]["ipc_threshold"], 2),
                    "low_ipc" in context["cache"]["flags"],
                ],
                [
                    "Latency CV",
                    _fmt(context["variance"]["cv"], 3),
                    _fmt(context["variance"]["threshold"], 2),
                    context["variance"]["flagged"],
                ],
                [
                    "NUMA span",
                    context["numa"]["numa_nodes"],
                    "single node",
                    context["numa"]["spans_multiple_numa_nodes"],
                ],
                [
                    "Threads vs physical cores",
                    f"{context['numa']['thread_count']} / {context['numa']['physical_cores']}",
                    "<= physical cores",
                    context["numa"]["exceeds_physical_cores"],
                ],
            ],
        ),
        "",
        "## Recommendations",
        "",
    ]
    if not context["recommendations"]:
        lines.append("No recommendations triggered.")
    else:
        for index, rec in enumerate(context["recommendations"], start=1):
            lines.extend(
                [
                    f"{index}. **[{rec['severity'].upper()}] {rec['code']}** - {rec['message']}",
                    f"   - Evidence: {rec['evidence']}",
                ]
            )
    lines.append("")
    return "\n".join(lines)


def render_html(result: RunResult) -> str:
    """Render a complete HTML report for one run using the Jinja2 template."""
    environment = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template("report.html.j2")
    return template.render(**build_context(result))


def render_report(result: RunResult, output_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Write ``<run-id>.md`` and ``<run-id>.html`` and return both paths."""
    directory = Path(output_dir) if output_dir is not None else Path("reports")
    directory.mkdir(parents=True, exist_ok=True)
    run_id = result.run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    markdown_path = directory / f"{run_id}.md"
    html_path = directory / f"{run_id}.html"
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    html_path.write_text(render_html(result), encoding="utf-8")
    return markdown_path, html_path


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Render a compare result as a Markdown regression table."""
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    lines: list[str] = [
        "# TensorTrace regression comparison",
        "",
        f"**Baseline:** `{baseline['run_id']}` ({baseline['backend']}/{baseline['model']}, "
        f"threads={baseline['threads']})  ",
        f"**Candidate:** `{candidate['run_id']}` ({candidate['backend']}/{candidate['model']}, "
        f"threads={candidate['threads']})  ",
        f"**Mean-latency threshold:** {_fmt(comparison['threshold_percent'], 1)}%",
        "",
        *_table(
            ["Metric", "Baseline", "Candidate", "Delta", "% Change", "Status"],
            [
                [
                    row["metric"],
                    _fmt(row["baseline"]),
                    _fmt(row["candidate"]),
                    _fmt(row["delta"]),
                    (
                        f"{_fmt(row['percent_change'], 2)}%"
                        if row["percent_change"] is not None
                        else "unavailable"
                    ),
                    row["status"],
                ]
                for row in comparison["rows"]
            ],
        ),
        "",
    ]
    if comparison["regression"]:
        lines.append(
            "**Result: REGRESSION** - gated metric(s): "
            + ", ".join(comparison["regressed_metrics"])
        )
    else:
        lines.append("**Result: PASS** - no mean-latency regression beyond the threshold.")
    lines.append("")
    return "\n".join(lines)
