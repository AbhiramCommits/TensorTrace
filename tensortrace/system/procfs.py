"""Thin, testable readers for Linux ``/proc`` interfaces.

Every parser in this module is a pure function over captured text. The
``read_*`` helpers are deliberately tiny so tests can exercise parser logic
without touching a live kernel. On non-Linux hosts the readers return empty
dictionaries and callers record the metric as unavailable.
"""

from __future__ import annotations

import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

PROC_STAT_FIELDS: tuple[str, ...] = (
    "user",
    "nice",
    "system",
    "idle",
    "iowait",
    "irq",
    "softirq",
    "steal",
    "guest",
    "guest_nice",
)
# guest time is already included in user/nice, so it is excluded from totals.
_TOTAL_FIELDS: tuple[str, ...] = (
    "user",
    "nice",
    "system",
    "idle",
    "iowait",
    "irq",
    "softirq",
    "steal",
)
_CPU_LABEL = re.compile(r"^cpu\d+$")

_STATUS_KEYS: dict[str, str] = {
    "VmRSS": "VmRSS_kb",
    "VmHWM": "VmHWM_kb",
    "VmSize": "VmSize_kb",
    "Threads": "Threads",
    "voluntary_ctxt_switches": "voluntary_ctxt_switches",
    "nonvoluntary_ctxt_switches": "nonvoluntary_ctxt_switches",
}


def parse_proc_stat(text: str) -> dict[str, dict[str, int]]:
    """Parse ``/proc/stat`` CPU lines into per-CPU jiffy counters.

    The aggregate ``cpu`` line is returned alongside ``cpu0``, ``cpu1``, and so
    on. Unknown or short lines are ignored; missing trailing fields default to
    zero, which matches older kernels.
    """
    parsed: dict[str, dict[str, int]] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        label = parts[0]
        if label != "cpu" and not _CPU_LABEL.match(label):
            continue
        try:
            values = [int(value) for value in parts[1:]]
        except ValueError:
            continue
        parsed[label] = {
            name: values[index] if index < len(values) else 0
            for index, name in enumerate(PROC_STAT_FIELDS)
        }
    return parsed


def _cpu_delta(before: dict[str, int], after: dict[str, int]) -> tuple[int, int]:
    """Return ``(busy_jiffies, total_jiffies)`` between two CPU snapshots."""
    busy = 0
    total = 0
    for name in _TOTAL_FIELDS:
        delta = max(0, after.get(name, 0) - before.get(name, 0))
        total += delta
        if name not in ("idle", "iowait"):
            busy += delta
    return busy, total


def _cpu_sort_key(label: str) -> tuple[int, str]:
    """Sort ``cpu`` before ``cpu0`` and numerically thereafter."""
    if label == "cpu":
        return (-1, label)
    return (int(label[3:]), label)


def cpu_utilization(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Compute aggregate and per-CPU utilization percentages from jiffy deltas.

    Returns a dictionary with ``aggregate``, ``per_cpu``, ``busy_jiffies``,
    and ``total_jiffies``. A zero-length sampling window yields ``0.0`` rather
    than a division error.
    """
    per_cpu: dict[str, float] = {}
    busy_sum = 0
    total_sum = 0
    for label in sorted(after, key=_cpu_sort_key):
        if not _CPU_LABEL.match(label) or label not in before:
            continue
        busy, total = _cpu_delta(before[label], after[label])
        busy_sum += busy
        total_sum += total
        per_cpu[label] = round(100.0 * busy / total, 4) if total else 0.0

    if "cpu" in before and "cpu" in after:
        aggregate_busy, aggregate_total = _cpu_delta(before["cpu"], after["cpu"])
    else:
        aggregate_busy, aggregate_total = busy_sum, total_sum
    aggregate = round(100.0 * aggregate_busy / aggregate_total, 4) if aggregate_total else 0.0
    return {
        "aggregate": aggregate,
        "per_cpu": per_cpu,
        "busy_jiffies": aggregate_busy,
        "total_jiffies": aggregate_total,
    }


def parse_proc_status(text: str) -> dict[str, int]:
    """Parse the memory and scheduler fields TensorTrace uses from ``/proc/self/status``.

    Memory values are normalized to kilobytes. Context-switch counters are raw
    cumulative counts.
    """
    parsed: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator or key not in _STATUS_KEYS:
            continue
        fields = rest.strip().split()
        if not fields:
            continue
        try:
            parsed[_STATUS_KEYS[key]] = int(fields[0])
        except ValueError:
            continue
    return parsed


def parse_meminfo(text: str) -> dict[str, int]:
    """Parse ``/proc/meminfo`` values into a ``{name: kilobytes}`` mapping."""
    parsed: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        fields = rest.strip().split()
        if not fields:
            continue
        try:
            parsed[key.strip()] = int(fields[0])
        except ValueError:
            continue
    return parsed


def parse_cpuinfo(text: str) -> dict[str, str]:
    """Parse the first processor block of ``/proc/cpuinfo`` into a flat mapping."""
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if parsed:
                break
            continue
        key, separator, value = line.partition(":")
        if separator:
            parsed[key.strip()] = value.strip()
    return parsed


def _read_text(path: Path) -> str | None:
    """Read a procfs file, returning ``None`` when it does not exist."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def read_stat(path: Path = Path("/proc/stat")) -> dict[str, dict[str, int]]:
    """Read and parse ``/proc/stat``; return ``{}`` when unavailable."""
    text = _read_text(path)
    return parse_proc_stat(text) if text is not None else {}


def read_self_status(path: Path = Path("/proc/self/status")) -> dict[str, int]:
    """Read and parse ``/proc/self/status``; return ``{}`` when unavailable."""
    text = _read_text(path)
    return parse_proc_status(text) if text is not None else {}


def read_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    """Read and parse ``/proc/meminfo``; return ``{}`` when unavailable."""
    text = _read_text(path)
    return parse_meminfo(text) if text is not None else {}


def read_cpuinfo(path: Path = Path("/proc/cpuinfo")) -> dict[str, str]:
    """Read and parse ``/proc/cpuinfo``; return ``{}`` when unavailable."""
    text = _read_text(path)
    return parse_cpuinfo(text) if text is not None else {}


def procfs_available() -> bool:
    """Return whether the host exposes the Linux procfs entry points we need."""
    return Path("/proc/stat").is_file() and Path("/proc/self/status").is_file()


def cpu_model() -> str:
    """Return the host CPU model, using procfs first and platform fallbacks next."""
    info = read_cpuinfo()
    for key in ("model name", "Model", "cpu model", "Hardware"):
        value = info.get(key)
        if value:
            return value
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        return platform.processor() or platform.machine() or "unavailable"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - platform specific
        return "unavailable"


@dataclass
class Snapshot:
    """One point-in-time capture of kernel counters and process memory."""

    timestamp: float
    stat: dict[str, dict[str, int]]
    status: dict[str, int]
    meminfo: dict[str, int]
    affinity: list[int]
    rss_kb: int | None
    threads: int | None
    rss_source: str
    procfs_available: bool


@dataclass
class SystemSnapshot:
    """Capture system state before/after a workload and diff the two windows."""

    before: Snapshot | None = None
    after: Snapshot | None = None
    _process: psutil.Process = field(default_factory=psutil.Process, repr=False)

    def _capture(self) -> Snapshot:
        """Capture a single snapshot without raising on missing Linux files."""
        status = read_self_status()
        stat = read_stat()
        meminfo = read_meminfo()
        try:
            affinity = sorted(self._process.cpu_affinity())
        except (psutil.Error, AttributeError, NotImplementedError):
            affinity = []
        if not affinity:
            # macOS and some restricted containers return an empty affinity set.
            from tensortrace.system.sysfs import current_affinity

            affinity = current_affinity()
        rss_kb = status.get("VmRSS_kb")
        source = "procfs"
        if rss_kb is None:
            try:
                rss_kb = self._process.memory_info().rss // 1024
                source = "psutil"
            except psutil.Error:
                rss_kb = None
                source = "unavailable"
        threads = status.get("Threads")
        if threads is None:
            try:
                threads = self._process.num_threads()
            except psutil.Error:
                threads = None
        return Snapshot(
            timestamp=time.time(),
            stat=stat,
            status=status,
            meminfo=meminfo,
            affinity=affinity,
            rss_kb=rss_kb,
            threads=threads,
            rss_source=source,
            procfs_available=procfs_available(),
        )

    def start(self) -> Snapshot:
        """Capture the opening snapshot."""
        self.before = self._capture()
        return self.before

    def stop(self) -> Snapshot:
        """Capture the closing snapshot."""
        self.after = self._capture()
        return self.after

    def diff(self) -> dict[str, Any]:
        """Diff the opening and closing snapshots into report-friendly data."""
        if self.before is None or self.after is None:
            raise RuntimeError("SystemSnapshot.start() and stop() must be called before diff()")
        before, after = self.before, self.after

        if before.stat and after.stat:
            utilization = cpu_utilization(before.stat, after.stat)
            cpu = {
                "available": True,
                "reason": None,
                "aggregate_percent": utilization["aggregate"],
                "per_cpu_percent": utilization["per_cpu"],
            }
        else:
            cpu = {
                "available": False,
                "reason": "/proc/stat is not readable on this host",
                "aggregate_percent": None,
                "per_cpu_percent": {},
            }

        rss_values = [value for value in (before.rss_kb, after.rss_kb) if value is not None]
        context_switches = {
            key: after.status.get(key, 0) - before.status.get(key, 0)
            for key in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")
            if key in before.status and key in after.status
        }
        memory = {
            "source": after.rss_source,
            "rss_start_kb": before.rss_kb,
            "rss_end_kb": after.rss_kb,
            "rss_peak_kb": max(rss_values) if rss_values else None,
            "rss_delta_kb": (
                after.rss_kb - before.rss_kb
                if before.rss_kb is not None and after.rss_kb is not None
                else None
            ),
            "threads_start": before.threads,
            "threads_end": after.threads,
            "context_switches": context_switches,
            "mem_total_kb": after.meminfo.get("MemTotal"),
            "mem_available_kb": after.meminfo.get("MemAvailable"),
        }
        return {
            "procfs_available": before.procfs_available and after.procfs_available,
            "window_seconds": round(after.timestamp - before.timestamp, 6),
            "cpu": cpu,
            "memory": memory,
            "affinity": after.affinity or before.affinity,
        }


class TelemetrySampler(threading.Thread):
    """Background thread polling process RSS, thread count, and CPU during a run."""

    def __init__(
        self,
        interval_ms: int = 50,
        process: psutil.Process | None = None,
        proc_stat_path: Path = Path("/proc/stat"),
    ) -> None:
        super().__init__(name="tensortrace-telemetry", daemon=True)
        self.interval_s = max(interval_ms, 1) / 1000.0
        self.proc_stat_path = proc_stat_path
        self.samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._process = process or psutil.Process()
        self._previous_stat: dict[str, dict[str, int]] | None = None
        self._t0 = 0.0

    def _record(self) -> None:
        """Append one sample to the time series."""
        current_stat = read_stat(self.proc_stat_path)
        cpu_percent: float | None
        if current_stat and self._previous_stat is not None:
            cpu_percent = cpu_utilization(self._previous_stat, current_stat)["aggregate"]
        elif current_stat:
            cpu_percent = None
        else:
            cpu_percent = psutil.cpu_percent(interval=None)
        if current_stat:
            self._previous_stat = current_stat

        try:
            memory = self._process.memory_info()
            rss_kb = memory.rss // 1024
            threads = self._process.num_threads()
        except psutil.Error:
            rss_kb = None
            threads = None
        process_cpu = self._process.cpu_percent(interval=None)
        self.samples.append(
            {
                "t_ms": round((time.perf_counter() - self._t0) * 1000.0, 3),
                "timestamp": time.time(),
                "rss_kb": rss_kb,
                "rss_mb": round(rss_kb / 1024.0, 3) if rss_kb is not None else None,
                "threads": threads,
                "cpu_percent": cpu_percent,
                "process_cpu_percent": process_cpu,
            }
        )

    def run(self) -> None:
        """Poll until :meth:`stop` is called."""
        self._t0 = time.perf_counter()
        psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)
        self._previous_stat = read_stat(self.proc_stat_path) or None
        self._record()
        while not self._stop_event.wait(self.interval_s):
            self._record()

    def stop(self) -> list[dict[str, Any]]:
        """Signal the sampler, join it, take a final sample, and return the series."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=5)
        self._record()
        return self.samples


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a telemetry time series for report tables."""
    if not samples:
        return {
            "count": 0,
            "duration_ms": 0.0,
            "rss_start_mb": None,
            "rss_peak_mb": None,
            "rss_end_mb": None,
            "cpu_mean_percent": None,
            "cpu_peak_percent": None,
            "cpu_p95_percent": None,
            "process_cpu_mean_percent": None,
            "process_cpu_peak_percent": None,
        }
    rss = [sample["rss_mb"] for sample in samples if sample.get("rss_mb") is not None]
    cpu = [sample["cpu_percent"] for sample in samples if sample.get("cpu_percent") is not None]
    process_cpu = [
        sample["process_cpu_percent"]
        for sample in samples
        if sample.get("process_cpu_percent") is not None
    ]

    def _p95(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
        return round(ordered[index], 3)

    return {
        "count": len(samples),
        "duration_ms": samples[-1]["t_ms"] - samples[0]["t_ms"],
        "rss_start_mb": rss[0] if rss else None,
        "rss_peak_mb": round(max(rss), 3) if rss else None,
        "rss_end_mb": rss[-1] if rss else None,
        "cpu_mean_percent": round(sum(cpu) / len(cpu), 3) if cpu else None,
        "cpu_peak_percent": round(max(cpu), 3) if cpu else None,
        "cpu_p95_percent": _p95(cpu),
        "process_cpu_mean_percent": round(sum(process_cpu) / len(process_cpu), 3)
        if process_cpu
        else None,
        "process_cpu_peak_percent": round(max(process_cpu), 3) if process_cpu else None,
    }
