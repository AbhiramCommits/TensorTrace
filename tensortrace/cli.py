"""Command-line interface for TensorTrace."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from tensortrace import __version__
from tensortrace.analysis.compare import compare_runs
from tensortrace.config import DEFAULT_BACKENDS, DEFAULT_MODELS, RunConfig
from tensortrace.profilers.runner import profile_run, save_result
from tensortrace.profilers.schema import RunResult
from tensortrace.report.render import render_comparison_markdown, render_report
from tensortrace.system.sysfs import collect_sysinfo


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    """Register the options shared by all ``run`` invocations."""
    parser.add_argument("--backend", choices=[*DEFAULT_BACKENDS, "both"], default="torch")
    parser.add_argument("--model", choices=DEFAULT_MODELS, default="resnet18")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10, dest="iterations")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("reports"))


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser and all subcommands."""
    parser = argparse.ArgumentParser(
        prog="tensortrace",
        description="Profile PyTorch and ONNX Runtime inference with Linux kernel counters.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run inference and write a JSON result")
    _add_common_run_args(run_parser)

    sub.add_parser("sysinfo", help="print host CPU, NUMA, cache, affinity, and perf info")

    report_parser = sub.add_parser("report", help="render Markdown and HTML for a run JSON")
    report_parser.add_argument("run_json", type=Path)

    compare_parser = sub.add_parser("compare", help="compare baseline and candidate runs")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="mean-latency regression threshold as a fraction (default: 0.10)",
    )
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    """Profile the selected backend(s), save JSON results, and print a summary."""
    base = RunConfig(
        model=args.model,
        backend="torch" if args.backend == "both" else args.backend,
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup=args.warmup,
        threads=args.threads,
        output_dir=args.out,
        seed=args.seed,
    )
    backends = list(DEFAULT_BACKENDS) if args.backend == "both" else [args.backend]
    for backend in backends:
        config = replace(base, backend=backend)
        result = profile_run(config)
        path = save_result(result, config.output_dir)
        latency = result.latency
        print(
            f"[{backend}/{config.model}] mean={latency['mean_ms']:.3f} ms "
            f"median={latency['median_ms']:.3f} ms p95={latency['p95_ms']:.3f} ms "
            f"throughput={latency['throughput_ips']:.2f} inferences/s -> {path}"
        )
        for warning in result.warnings:
            print(f"  warning: {warning}")
    return 0


def _cmd_sysinfo(_args: argparse.Namespace) -> int:
    """Print the host telemetry payload as JSON."""
    print(json.dumps(collect_sysinfo(), indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Render Markdown and HTML next to a run JSON file."""
    try:
        result = RunResult.load(args.run_json)
    except (OSError, ValueError) as exc:
        print(f"tensortrace report: cannot read {args.run_json}: {exc}", file=sys.stderr)
        return 2
    markdown_path, html_path = render_report(result, args.run_json.parent)
    print(f"markdown: {markdown_path}")
    print(f"html: {html_path}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Print a regression table and return 1 when the latency gate fails."""
    try:
        baseline = RunResult.load(args.baseline)
        candidate = RunResult.load(args.candidate)
    except (OSError, ValueError) as exc:
        print(f"tensortrace compare: cannot read input: {exc}", file=sys.stderr)
        return 2
    comparison = compare_runs(baseline, candidate, threshold=args.threshold)
    print(render_comparison_markdown(comparison))
    return 1 if comparison["regression"] else 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``tensortrace`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "sysinfo": _cmd_sysinfo,
        "report": _cmd_report,
        "compare": _cmd_compare,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
