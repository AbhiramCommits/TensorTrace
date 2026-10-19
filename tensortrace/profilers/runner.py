"""Run orchestration: framework timings surrounded by system telemetry and perf."""

from __future__ import annotations

import importlib
import json
import os
import platform
import socket
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tensortrace.config import RunConfig
from tensortrace.profilers.schema import RunResult
from tensortrace.system.perf import (
    DEFAULT_EVENTS,
    PerfResult,
    perf_available,
    perf_unavailable_reason,
)
from tensortrace.system.procfs import SystemSnapshot, TelemetrySampler, summarize_samples
from tensortrace.system.sysfs import collect_sysinfo


def profile_backend(config: RunConfig) -> RunResult:
    """Dispatch to the requested inference backend without any system wrapper."""
    if config.backend == "torch":
        from tensortrace.profilers.torch_profiler import profile_torch

        return profile_torch(config)
    if config.backend == "onnx":
        from tensortrace.profilers.onnx_profiler import profile_onnx

        return profile_onnx(config)
    raise ValueError(f"unknown backend {config.backend!r}; choose torch or onnx")


def collect_environment() -> dict[str, Any]:
    """Collect interpreter, host, and package versions for a run result."""
    packages: dict[str, str] = {}
    try:
        import torch

        packages["torch"] = torch.__version__
    except ImportError:  # pragma: no cover - torch is a required dependency
        pass
    try:
        import onnxruntime

        packages["onnxruntime"] = onnxruntime.__version__
    except ImportError:  # pragma: no cover - onnxruntime is a required dependency
        pass
    try:
        import psutil

        packages["psutil"] = psutil.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import numpy

        packages["numpy"] = numpy.__version__
    except ImportError:  # pragma: no cover
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "packages": packages,
    }


def run_id_for(config: RunConfig, output_dir: Path | None = None) -> str:
    """Build a filesystem-safe run ID, adding a suffix if the name already exists."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"run-{timestamp}-{config.backend}-{config.model}"
    if output_dir is None:
        return base
    candidate = base
    counter = 2
    while (output_dir / f"{candidate}.json").exists():
        candidate = f"{base}-{counter}"
        counter += 1
    return candidate


def save_result(result: RunResult, output_dir: Path) -> Path:
    """Persist a run result inside ``output_dir`` and return the written path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not result.run_id:
        result.run_id = run_id_for(
            RunConfig(
                model=result.model,
                backend=result.backend,
                batch_size=int(result.config.get("batch_size", 1)),
                iterations=int(result.config.get("iterations", 1)),
                warmup=int(result.config.get("warmup", 0)),
                threads=int(result.config.get("threads", 0)),
                output_dir=output_dir,
                seed=int(result.config.get("seed", 42)),
            ),
            output_dir,
        )
    return result.save(output_dir / f"{result.run_id}.json")


def _worker_command(config: RunConfig, destination: Path) -> list[str]:
    """Build the command line used for the perf-instrumented child process."""
    return [
        sys.executable,
        "-m",
        "tensortrace.profilers.worker",
        "--out",
        str(destination),
        "--config",
        json.dumps(config.to_dict()),
    ]


def _run_under_perf(config: RunConfig, output_dir: Path) -> tuple[RunResult, PerfResult]:
    """Run the backend in a child process wrapped by ``perf stat``."""
    from tensortrace.system.perf import run_perf_stat

    handle = tempfile.NamedTemporaryFile(
        prefix=".tensortrace-worker-", suffix=".json", dir=output_dir, delete=False
    )
    destination = Path(handle.name)
    handle.close()
    destination.unlink(missing_ok=True)
    timeout = max(600.0, config.iterations * 30.0 + 180.0)
    perf_result = run_perf_stat(
        _worker_command(config, destination), DEFAULT_EVENTS, timeout=timeout
    )
    if not destination.exists():
        # perf could start but the child failed; fall back so a result still exists.
        detail = perf_result.raw.strip().splitlines()
        reason = detail[-1] if detail else "worker produced no JSON result"
        fallback = profile_backend(config)
        return fallback, PerfResult(
            available=False,
            reason=f"perf-instrumented worker failed: {reason}",
            command=perf_result.command,
            returncode=perf_result.returncode,
            raw=perf_result.raw,
        )
    try:
        partial = RunResult.from_dict(json.loads(destination.read_text(encoding="utf-8")))
    finally:
        destination.unlink(missing_ok=True)
    return partial, perf_result


def _preload_dependencies(config: RunConfig) -> None:
    """Import framework libraries before telemetry starts.

    Imports allocate hundreds of megabytes in CPython; loading them first keeps
    peak-RSS figures attributable to the model and measured run rather than to
    interpreter startup.
    """
    try:
        modules = ["torch", "torchvision"]
        if config.backend == "onnx":
            modules.append("onnxruntime")
        for module in modules:
            importlib.import_module(module)
    except ImportError:  # pragma: no cover - dependencies are required
        pass


def profile_run(config: RunConfig) -> RunResult:
    """Run one backend, wrapping it with system snapshots, sampling, and perf counters.

    When ``perf stat`` is usable, the backend executes in a child process so the
    PMU window and the measured inference window are identical. Without perf,
    the backend runs in-process and the report records the unavailable reason.
    """
    if config.backend not in ("torch", "onnx"):
        raise ValueError(f"unknown backend {config.backend!r}; choose torch or onnx")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _preload_dependencies(config)

    snapshot = SystemSnapshot()
    sampler = TelemetrySampler(interval_ms=25)
    perf_ok = perf_available()
    snapshot.start()
    sampler.start()
    try:
        if perf_ok:
            result, perf_result = _run_under_perf(config, output_dir)
        else:
            result = profile_backend(config)
            perf_result = PerfResult(available=False, reason=perf_unavailable_reason())
    finally:
        samples = sampler.stop()
        snapshot.stop()

    result.run_id = run_id_for(config, output_dir)
    result.config = config.to_dict()
    result.backend = config.backend
    result.model = config.model
    result.environment = collect_environment()
    result.system = collect_sysinfo()
    result.snapshot = snapshot.diff()
    result.time_series = samples
    result.time_series_summary = summarize_samples(samples)
    result.perf = perf_result.to_dict()
    return result
