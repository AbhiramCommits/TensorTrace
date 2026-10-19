"""Readers and pure parsers for Linux ``/sys`` topology interfaces.

NUMA and cache data are optional: containers, VMs, and non-Linux hosts may not
expose them. Every public reader returns a dictionary containing ``available``
and ``reason`` so reports can say "unavailable" without fabricating values.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

import psutil

from tensortrace.system.procfs import cpu_model, read_meminfo

_SIZE_SUFFIXES = {"K": 1024, "M": 1024**2, "G": 1024**3}
_TYPE_CODES = {"data": "d", "instruction": "i", "unified": "u"}


def parse_cpu_list(text: str) -> list[int]:
    """Parse a Linux CPU list such as ``0-3,8,10-11`` into sorted integers."""
    cpus: set[int] = set()
    for token in text.strip().replace("\n", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, _, end_text = token.partition("-")
            try:
                start, end = int(start_text), int(end_text)
            except ValueError:
                continue
            if start <= end:
                cpus.update(range(start, end + 1))
            continue
        try:
            cpus.add(int(token))
        except ValueError:
            continue
    return sorted(cpus)


def format_cpu_list(cpus: list[int]) -> str:
    """Format integers as a compact Linux CPU list (``0-3,8``)."""
    if not cpus:
        return ""
    ordered = sorted(set(cpus))
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def format_affinity_mask(cpus: list[int]) -> str:
    """Render a CPU list as a hexadecimal affinity mask (Linux ``Cpus_allowed`` style)."""
    mask = 0
    for cpu in cpus:
        if cpu >= 0:
            mask |= 1 << cpu
    return f"{mask:x}" if mask else "0"


def parse_size_bytes(text: str) -> int | None:
    """Parse a sysfs cache size such as ``32K`` into bytes."""
    value = text.strip().upper()
    if not value:
        return None
    suffix = value[-1]
    multiplier = _SIZE_SUFFIXES.get(suffix)
    try:
        if multiplier is None:
            return int(value)
        return int(value[:-1]) * multiplier
    except ValueError:
        return None


def parse_node_meminfo(text: str) -> dict[str, int]:
    """Parse ``/sys/devices/system/node/nodeN/meminfo`` into kilobytes by field."""
    parsed: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0].startswith("Node"):
            continue
        key = parts[2].rstrip(":")
        try:
            parsed[key] = int(parts[3])
        except ValueError:
            continue
    return parsed


def _read(path: Path) -> str | None:
    """Read a sysfs file, returning ``None`` when it is absent or unreadable."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def numa_topology(root: Path = Path("/sys/devices/system/node")) -> dict[str, Any]:
    """Read NUMA nodes, their CPU lists, memory info, and distance matrices."""
    root = Path(root)
    if not root.is_dir():
        return {
            "available": False,
            "reason": f"{root} is not present on this host",
            "nodes": {},
        }
    nodes: dict[str, dict[str, Any]] = {}
    for node_dir in sorted(root.glob("node[0-9]*")):
        try:
            node_id = int(node_dir.name[4:])
        except ValueError:
            continue
        cpu_text = _read(node_dir / "cpulist") or ""
        meminfo_text = _read(node_dir / "meminfo") or ""
        distance_text = _read(node_dir / "distance") or ""
        distances: list[int] = []
        for token in distance_text.split():
            try:
                distances.append(int(token))
            except ValueError:
                continue
        nodes[node_dir.name] = {
            "node_id": node_id,
            "cpus": parse_cpu_list(cpu_text),
            "cpu_list": format_cpu_list(parse_cpu_list(cpu_text)),
            "meminfo": parse_node_meminfo(meminfo_text),
            "distance": distances,
        }
    if not nodes:
        return {
            "available": False,
            "reason": f"no node[0-9]* directories found under {root}",
            "nodes": {},
        }
    return {"available": True, "reason": None, "nodes": nodes}


def cache_topology(root: Path = Path("/sys/devices/system/cpu")) -> dict[str, Any]:
    """Read per-CPU cache levels, sizes, line sizes, and sharing maps.

    Duplicate shared caches (for example an L3 reported by every core) are
    collapsed by ``(level, type, shared_cpu_list)`` so the output stays compact.
    """
    root = Path(root)
    if not root.is_dir():
        return {
            "available": False,
            "reason": f"{root} is not present on this host",
            "caches": [],
            "levels": {},
        }
    cpu_dirs = sorted(
        (path for path in root.glob("cpu[0-9]*") if path.is_dir()),
        key=lambda path: int(path.name[3:]),
    )
    caches: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for cpu_dir in cpu_dirs:
        index_dirs = sorted(
            (path for path in (cpu_dir / "cache").glob("index[0-9]*") if path.is_dir()),
            key=lambda path: int(path.name[5:]),
        )
        for index_dir in index_dirs:
            level_text = _read(index_dir / "level") or ""
            type_text = _read(index_dir / "type") or ""
            shared_text = _read(index_dir / "shared_cpu_list") or ""
            try:
                level = int(level_text.strip())
            except ValueError:
                continue
            cache_type = type_text.strip() or "Unknown"
            key = (str(level), cache_type.lower(), shared_text.strip())
            if key in seen:
                continue
            seen.add(key)
            size_text = (_read(index_dir / "size") or "").strip()
            line_text = (_read(index_dir / "coherency_line_size") or "").strip()
            try:
                line_size = int(line_text) if line_text else None
            except ValueError:
                line_size = None
            caches.append(
                {
                    "cpu": int(cpu_dir.name[3:]),
                    "index": int(index_dir.name[5:]),
                    "level": level,
                    "type": cache_type,
                    "size_text": size_text,
                    "size_bytes": parse_size_bytes(size_text),
                    "line_size": line_size,
                    "shared_cpu_list": shared_text.strip(),
                    "shared_cpus": parse_cpu_list(shared_text),
                    "missing_fields": [
                        name
                        for name, present in (
                            ("size", bool(size_text)),
                            ("coherency_line_size", bool(line_text)),
                        )
                        if not present
                    ],
                }
            )
    if not caches:
        return {
            "available": False,
            "reason": f"no CPU cache index directories found under {root}",
            "caches": [],
            "levels": {},
        }

    levels: dict[str, dict[str, Any]] = {}
    for cache in caches:
        code = _TYPE_CODES.get(cache["type"].lower(), cache["type"].lower()[:1] or "u")
        name = f"L{cache['level']}{code}"
        entry = levels.setdefault(
            name,
            {
                "level": cache["level"],
                "type": cache["type"],
                "size_bytes": cache["size_bytes"],
                "line_size": cache["line_size"],
                "sharing": [],
            },
        )
        entry["sharing"].append(
            {
                "shared_cpu_list": cache["shared_cpu_list"],
                "cpus": cache["shared_cpus"],
            }
        )
    return {"available": True, "reason": None, "caches": caches, "levels": levels}


def current_affinity_info() -> dict[str, Any]:
    """Return the process CPU affinity list, mask, and the API used to read it."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        try:
            cpus = sorted(getter(0))
            return {
                "cpus": cpus,
                "mask": format_affinity_mask(cpus),
                "source": "os.sched_getaffinity",
            }
        except OSError:
            pass
    try:
        cpus = sorted(psutil.Process().cpu_affinity())
        if cpus:
            return {"cpus": cpus, "mask": format_affinity_mask(cpus), "source": "psutil"}
    except (psutil.Error, AttributeError, ValueError):
        pass
    cpus = list(range(os.cpu_count() or 1))
    return {"cpus": cpus, "mask": format_affinity_mask(cpus), "source": "os.cpu_count fallback"}


def current_affinity() -> list[int]:
    """Return the CPUs this process may run on.

    Prefers ``os.sched_getaffinity`` on Linux, then psutil, then every CPU
    reported by the OS. The fallback is recorded by :func:`current_affinity_info`.
    """
    return current_affinity_info()["cpus"]


def nodes_for_cpus(cpus: list[int], numa: dict[str, Any]) -> list[str]:
    """Return the NUMA node names that contain at least one of ``cpus``."""
    wanted = set(cpus)
    return sorted(
        name
        for name, node in numa.get("nodes", {}).items()
        if wanted.intersection(node.get("cpus", []))
    )


def _host_memory_kb() -> int | None:
    """Return total memory in KiB, using procfs first and psutil second."""
    meminfo = read_meminfo()
    if meminfo.get("MemTotal"):
        return meminfo["MemTotal"]
    try:
        return psutil.virtual_memory().total // 1024
    except psutil.Error:  # pragma: no cover - defensive
        return None


def collect_sysinfo() -> dict[str, Any]:
    """Collect the JSON payload printed by ``tensortrace sysinfo``."""
    affinity = current_affinity_info()
    numa = numa_topology()
    cache = cache_topology()
    from tensortrace.system import perf

    return {
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "linux": platform.system() == "Linux",
        },
        "cpu": {
            "model": cpu_model(),
            "logical_cores": os.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "affinity": affinity,
            "affinity_numa_nodes": nodes_for_cpus(affinity["cpus"], numa),
        },
        "memory": {"total_kb": _host_memory_kb()},
        "numa": numa,
        "cache": cache,
        "perf": {
            "available": perf.perf_available(),
            "reason": perf.perf_unavailable_reason(),
        },
        "procfs": {
            "available": (Path("/proc/stat").is_file() and Path("/proc/self/status").is_file()),
            "reason": None
            if (Path("/proc/stat").is_file() and Path("/proc/self/status").is_file())
            else "/proc/stat or /proc/self/status is not readable on this host",
        },
    }
