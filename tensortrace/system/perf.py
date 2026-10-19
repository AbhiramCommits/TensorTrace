"""``perf stat`` discovery, execution, and CSV parsing.

The parser is intentionally independent of the subprocess wrapper so captured
CSV text can be tested exactly. Missing binaries, restrictive
``perf_event_paranoid`` settings, and unsupported PMU events all degrade to an
explicit ``available``/``reason`` pair instead of an exception.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

DEFAULT_EVENTS: tuple[str, ...] = (
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "page-faults",
    "context-switches",
)

PerfCounter = tuple[float | None, str, float | None]


@dataclass
class PerfResult:
    """Outcome of a ``perf stat`` invocation."""

    available: bool
    reason: str = ""
    counters: dict[str, PerfCounter] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible primitives."""
        return {
            "available": self.available,
            "reason": self.reason,
            "counters": {
                event: {"value": value, "unit": unit, "run_pct": run_pct}
                for event, (value, unit, run_pct) in self.counters.items()
            },
            "command": self.command,
            "returncode": self.returncode,
            "raw": self.raw,
        }


@lru_cache(maxsize=1)
def _probe_perf() -> tuple[bool, str]:
    """Probe once for a usable ``perf`` binary and return ``(available, reason)``."""
    binary = shutil.which("perf")
    if binary is None:
        return False, "perf binary was not found on PATH"
    try:
        version = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if version.returncode != 0:
            detail = (version.stderr or version.stdout).strip().splitlines()
            return False, f"perf --version failed: {detail[-1] if detail else 'unknown error'}"
        # A permission probe catches perf_event_paranoid and container seccomp blocks.
        probe = subprocess.run(
            [binary, "stat", "-x,", "-e", "task-clock", "--", "true"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout).strip().splitlines()
            message = detail[-1] if detail else f"exit {probe.returncode}"
            return (
                False,
                "perf stat is not permitted in this environment "
                f"(check /proc/sys/kernel/perf_event_paranoid): {message}",
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"perf probe failed: {exc}"
    return True, ""


def perf_available() -> bool:
    """Return whether ``perf stat`` can be executed successfully on this host."""
    return _probe_perf()[0]


def perf_unavailable_reason() -> str:
    """Return a human-readable reason when perf is unusable (empty otherwise)."""
    return _probe_perf()[1]


def parse_perf_stat_csv(text: str) -> dict[str, PerfCounter]:
    """Parse ``perf stat -x,`` CSV output into ``{event: (value, unit, run_pct)}``.

    Rows marked ``<not supported>`` or ``<not counted>`` become
    ``(None, "<not supported>", None)`` so callers can distinguish "zero" from
    "the PMU cannot count this".
    """
    counters: dict[str, PerfCounter] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        value_text = parts[0].strip()
        unit = parts[1].strip()
        event = parts[2].strip()
        if not event or " " in event:
            # Summary rows such as "seconds time elapsed" are not counters.
            continue
        lowered = value_text.lower()
        if "not supported" in lowered:
            counters[event] = (None, "<not supported>", None)
            continue
        if "not counted" in lowered:
            counters[event] = (None, "<not counted>", None)
            continue
        run_pct: float | None = None
        if len(parts) >= 5 and parts[4].strip():
            try:
                run_pct = float(parts[4].strip())
            except ValueError:
                run_pct = None
        try:
            value = float(value_text.replace(",", ""))
        except ValueError:
            continue
        counters[event] = (value, unit, run_pct)
    return counters


def run_perf_stat(
    cmd_argv: list[str],
    events: tuple[str, ...] | list[str] = DEFAULT_EVENTS,
    timeout: float = 600.0,
) -> PerfResult:
    """Run ``cmd_argv`` under ``perf stat -x, -e <events>`` and parse the counters."""
    if not perf_available():
        return PerfResult(available=False, reason=perf_unavailable_reason())
    binary = shutil.which("perf")
    if binary is None:  # pragma: no cover - guarded by perf_available
        return PerfResult(available=False, reason="perf binary disappeared after probe")
    command = [binary, "stat", "-x,", "-e", ",".join(events), "--", *cmd_argv]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PerfResult(
            available=False,
            reason=f"perf stat failed to start: {exc}",
            command=command,
        )
    raw = (completed.stderr or "") + (completed.stdout or "")
    counters = parse_perf_stat_csv(raw)
    if completed.returncode != 0 and not counters:
        lines = raw.strip().splitlines()
        reason = lines[-1] if lines else f"perf exited with {completed.returncode}"
        return PerfResult(
            available=False,
            reason=reason,
            command=command,
            returncode=completed.returncode,
            raw=raw,
        )
    return PerfResult(
        available=True,
        reason="",
        counters=counters,
        command=command,
        returncode=completed.returncode,
        raw=raw,
    )
