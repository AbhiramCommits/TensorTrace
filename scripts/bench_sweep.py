#!/usr/bin/env python3
"""Run the full TensorTrace benchmark matrix and write a single summary table.

Matrix: 2 models x 2 backends x {1 thread, physical core count}. Raw JSON
results are written to ``reports/sweep-runs`` and the table to
``reports/sweep-summary.md``.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from tensortrace.analysis.compare import compare_runs
from tensortrace.analysis.hotspots import cache_analysis
from tensortrace.analysis.recommend import generate_recommendations
from tensortrace.config import RunConfig
from tensortrace.profilers.runner import profile_run, save_result
from tensortrace.system.sysfs import collect_sysinfo

MODELS: tuple[str, ...] = ("resnet18", "mlp")
BACKENDS: tuple[str, ...] = ("torch", "onnx")


def physical_cores() -> int:
    """Return the physical core count, falling back to logical CPUs."""
    return psutil.cpu_count(logical=False) or os.cpu_count() or 1


def _relative(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def run_sweep(
    summary_path: Path,
    run_dir: Path,
    iterations: int,
    warmup: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Execute the matrix and return one record per configuration."""
    run_dir.mkdir(parents=True, exist_ok=True)
    thread_counts = sorted({1, physical_cores()})
    total = len(MODELS) * len(BACKENDS) * len(thread_counts)
    records: list[dict[str, Any]] = []
    index = 0
    for backend in BACKENDS:
        for model in MODELS:
            for threads in thread_counts:
                index += 1
                print(
                    f"[{index}/{total}] {backend}/{model} threads={threads}",
                    flush=True,
                )
                config = RunConfig(
                    model=model,
                    backend=backend,
                    batch_size=batch_size,
                    iterations=iterations,
                    warmup=warmup,
                    threads=threads,
                    output_dir=run_dir,
                    seed=seed,
                )
                result = profile_run(config)
                path = save_result(result, run_dir)
                cache = cache_analysis(result.perf)
                records.append(
                    {
                        "backend": backend,
                        "model": model,
                        "threads": result.config.get("effective_threads", threads),
                        "result": result,
                        "path": _relative(path),
                        "perf_available": result.perf.get("available", False),
                        "cache_miss_rate": cache["cache_miss_rate"],
                        "ipc": cache["ipc"],
                        "recommendations": [rec.code for rec in generate_recommendations(result)],
                    }
                )
                print(
                    f"    mean={result.latency['mean_ms']:.3f} ms "
                    f"p95={result.latency['p95_ms']:.3f} ms "
                    f"throughput={result.latency['throughput_ips']:.2f} inf/s -> {path}",
                    flush=True,
                )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    return records


def _fmt(value: float | None, digits: int = 3) -> str:
    """Format a numeric cell or ``unavailable``."""
    if value is None:
        return "unavailable"
    return f"{value:,.{digits}f}"


def render_summary(records: list[dict[str, Any]], sysinfo: dict[str, Any]) -> str:
    """Render the sweep records as one Markdown document."""
    generated = datetime.now(timezone.utc).isoformat()
    cpu = sysinfo.get("cpu") or {}
    numa = sysinfo.get("numa") or {}
    perf = sysinfo.get("perf") or {}
    thread_list = ", ".join(
        str(count) for count in sorted({record["threads"] for record in records})
    )
    lines: list[str] = [
        "# TensorTrace benchmark sweep",
        "",
        f"Generated: {generated}",
        "",
        "## Host",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| CPU | {cpu.get('model')} |",
        f"| Physical / logical cores | {cpu.get('physical_cores')} / {cpu.get('logical_cores')} |",
        (
            "| NUMA | available |"
            if numa.get("available")
            else f"| NUMA | unavailable ({numa.get('reason')}) |"
        ),
        (
            "| perf | available |"
            if perf.get("available")
            else f"| perf | unavailable ({perf.get('reason')}) |"
        ),
        f"| Models | {', '.join(MODELS)} |",
        f"| Backends | {', '.join(BACKENDS)} |",
        f"| Thread counts | {thread_list} |",
        "",
        "## Measurements",
        "",
        "| Backend | Model | Threads | Mean ms | Median ms | p95 ms | Throughput (inf/s) "
        "| Peak RSS MB | Cache miss rate / IPC | Top layer | Top layer % | Run JSON |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        result = record["result"]
        latency = result.latency
        summary = result.time_series_summary
        top = result.layers[0] if result.layers else None
        cache_cell = (
            f"{_fmt(record['cache_miss_rate'], 4)} / {_fmt(record['ipc'], 3)}"
            if record["cache_miss_rate"] is not None
            else "unavailable"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    record["backend"],
                    record["model"],
                    str(record["threads"]),
                    _fmt(latency.get("mean_ms")),
                    _fmt(latency.get("median_ms")),
                    _fmt(latency.get("p95_ms")),
                    _fmt(latency.get("throughput_ips"), 2),
                    _fmt(summary.get("rss_peak_mb"), 1),
                    cache_cell,
                    top.name if top else "unavailable",
                    _fmt(top.percent_of_total if top else None, 1),
                    f"`{record['path']}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 1-thread vs physical-core scaling",
            "",
            "| Backend | Model | 1 thread mean ms | Physical-core mean ms | Speedup | Mean change "
            "| Gate |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    by_key = {
        (record["backend"], record["model"], record["threads"]): record
        for record in records
    }
    for backend in BACKENDS:
        for model in MODELS:
            baseline = by_key.get((backend, model, 1))
            candidate = by_key.get((backend, model, physical_cores()))
            if baseline is None or candidate is None:
                continue
            comparison = compare_runs(
                baseline["result"], candidate["result"], threshold=0.10
            )
            mean_row = next(
                row for row in comparison["rows"] if row["metric"] == "mean_latency_ms"
            )
            base_mean = baseline["result"].latency["mean_ms"]
            candidate_mean = candidate["result"].latency["mean_ms"]
            speedup = base_mean / candidate_mean if candidate_mean else None
            gate = "REGRESSION" if comparison["regression"] else "PASS"
            lines.append(
                "| "
                + " | ".join(
                    [
                        backend,
                        model,
                        _fmt(base_mean),
                        _fmt(candidate_mean),
                        f"{speedup:.2f}x" if speedup is not None else "unavailable",
                        f"{mean_row['percent_change']:+.2f}%"
                        if mean_row["percent_change"] is not None
                        else "unavailable",
                        gate,
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Recommendations emitted", ""])
    for record in records:
        codes = ", ".join(record["recommendations"]) or "none"
        lines.append(
            f"- `{record['backend']}/{record['model']}` threads={record['threads']}: {codes}"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the benchmark sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/sweep-summary.md"))
    parser.add_argument("--run-dir", type=Path, default=Path("reports/sweep-runs"))
    parser.add_argument("--iters", type=int, default=10, dest="iterations")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    sysinfo = collect_sysinfo()
    records = run_sweep(
        summary_path=args.out,
        run_dir=args.run_dir,
        iterations=args.iterations,
        warmup=args.warmup,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    summary = render_summary(records, sysinfo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(summary, encoding="utf-8")
    print(f"summary: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
