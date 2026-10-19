"""Tests for pure /proc parsers and telemetry summarization."""

from __future__ import annotations

from pathlib import Path

from tensortrace.system.procfs import (
    cpu_model,
    cpu_utilization,
    parse_cpuinfo,
    parse_meminfo,
    parse_proc_stat,
    parse_proc_status,
    read_meminfo,
    read_self_status,
    read_stat,
    summarize_samples,
)


def _cpu(user: int, system: int, idle: int, **extra: int) -> dict[str, int]:
    values = {
        "user": user,
        "nice": 0,
        "system": system,
        "idle": idle,
        "iowait": 0,
        "irq": 0,
        "softirq": 0,
        "steal": 0,
        "guest": 0,
        "guest_nice": 0,
    }
    values.update(extra)
    return values


def test_parse_proc_stat_reads_per_cpu_jiffies(fixtures_dir: Path) -> None:
    parsed = parse_proc_stat((fixtures_dir / "proc_stat.txt").read_text())
    assert len(parsed) == 9
    assert set(parsed) == {"cpu", *[f"cpu{index}" for index in range(8)]}
    assert parsed["cpu"]["user"] == 1882886
    assert parsed["cpu0"]["softirq"] == 251685
    assert parsed["cpu0"]["guest"] == 0
    assert parsed["cpu0"]["guest_nice"] == 0


def test_parse_proc_stat_ignores_non_cpu_lines_and_short_rows() -> None:
    text = "intr 1 2 3\nctxt 99\ncpu 10 0 5 80 1 0 2 0 0 0\ncpu0 1 2 3\n"
    parsed = parse_proc_stat(text)
    assert set(parsed) == {"cpu", "cpu0"}
    assert parsed["cpu"]["steal"] == 0
    assert parsed["cpu0"]["user"] == 1
    assert parsed["cpu0"]["nice"] == 2
    assert parsed["cpu0"]["system"] == 3
    assert parsed["cpu0"]["idle"] == 0
    assert parsed["cpu0"]["softirq"] == 0


def test_cpu_utilization_math() -> None:
    before = {
        "cpu": _cpu(100, 100, 800),
        "cpu0": _cpu(50, 50, 400),
        "cpu1": _cpu(50, 50, 400),
    }
    after = {
        "cpu": _cpu(250, 175, 925),
        "cpu0": _cpu(100, 75, 475),
        "cpu1": _cpu(150, 100, 450),
    }
    result = cpu_utilization(before, after)
    assert result["per_cpu"] == {"cpu0": 50.0, "cpu1": 75.0}
    assert result["aggregate"] == 64.2857
    assert result["busy_jiffies"] == 225
    assert result["total_jiffies"] == 350


def test_cpu_utilization_zero_window_is_zero() -> None:
    snapshot = {"cpu": _cpu(10, 10, 10), "cpu0": _cpu(5, 5, 5)}
    result = cpu_utilization(snapshot, snapshot)
    assert result["aggregate"] == 0.0
    assert result["per_cpu"] == {"cpu0": 0.0}


def test_cpu_utilization_without_aggregate_line_uses_per_cpu_sums() -> None:
    before = {"cpu0": _cpu(0, 0, 100)}
    after = {"cpu0": _cpu(50, 50, 100)}
    result = cpu_utilization(before, after)
    assert result["aggregate"] == 100.0
    assert result["busy_jiffies"] == 100


def test_parse_proc_status_exact_values(fixtures_dir: Path) -> None:
    parsed = parse_proc_status((fixtures_dir / "proc_status.txt").read_text())
    assert parsed["VmRSS_kb"] == 1620
    assert parsed["VmHWM_kb"] == 1620
    assert parsed["VmSize_kb"] == 3472
    assert parsed["Threads"] == 1
    assert parsed["voluntary_ctxt_switches"] == 4
    assert parsed["nonvoluntary_ctxt_switches"] == 1


def test_parse_proc_status_ignores_unknown_and_bad_values() -> None:
    text = "Name:\tpython\nVmRSS:\t 1234 kB\nVmHWM:\tnot-a-number kB\nThreads:\t7\n"
    parsed = parse_proc_status(text)
    assert parsed == {"VmRSS_kb": 1234, "Threads": 7}


def test_parse_meminfo_exact_values(fixtures_dir: Path) -> None:
    parsed = parse_meminfo((fixtures_dir / "meminfo.txt").read_text())
    assert parsed["MemTotal"] == 8124776
    assert parsed["MemFree"] == 295524
    assert parsed["MemAvailable"] == 3702516
    assert parsed["Buffers"] == 972312


def test_parse_cpuinfo_first_block() -> None:
    text = (
        "processor\t: 0\n"
        "model name\t: Example CPU 9000\n"
        "cpu cores\t: 8\n"
        "\n"
        "processor\t: 1\n"
        "model name\t: Example CPU 9000\n"
    )
    parsed = parse_cpuinfo(text)
    assert parsed["processor"] == "0"
    assert parsed["model name"] == "Example CPU 9000"
    assert parsed["cpu cores"] == "8"


def test_cpu_model_is_nonempty() -> None:
    assert isinstance(cpu_model(), str)
    assert cpu_model().strip()


def test_reader_wrappers_use_files_and_handle_missing(tmp_path: Path) -> None:
    stat_file = tmp_path / "stat"
    stat_file.write_text("cpu 1 2 3 4 5 6 7 8 9 10\n")
    status_file = tmp_path / "status"
    status_file.write_text("VmRSS:\t42 kB\n")
    meminfo_file = tmp_path / "meminfo"
    meminfo_file.write_text("MemTotal:\t1024 kB\n")

    assert read_stat(stat_file)["cpu"]["user"] == 1
    assert read_self_status(status_file) == {"VmRSS_kb": 42}
    assert read_meminfo(meminfo_file) == {"MemTotal": 1024}
    assert read_stat(tmp_path / "missing") == {}
    assert read_self_status(tmp_path / "missing") == {}
    assert read_meminfo(tmp_path / "missing") == {}


def test_summarize_samples_math() -> None:
    samples = [
        {"t_ms": 0.0, "rss_mb": 100.0, "cpu_percent": 10.0, "process_cpu_percent": 50.0},
        {"t_ms": 100.0, "rss_mb": 200.0, "cpu_percent": 20.0, "process_cpu_percent": 60.0},
        {"t_ms": 200.0, "rss_mb": 150.0, "cpu_percent": 30.0, "process_cpu_percent": 70.0},
    ]
    summary = summarize_samples(samples)
    assert summary["count"] == 3
    assert summary["duration_ms"] == 200.0
    assert summary["rss_start_mb"] == 100.0
    assert summary["rss_peak_mb"] == 200.0
    assert summary["rss_end_mb"] == 150.0
    assert summary["cpu_mean_percent"] == 20.0
    assert summary["cpu_peak_percent"] == 30.0
    assert summary["cpu_p95_percent"] == 30.0
    assert summary["process_cpu_mean_percent"] == 60.0
    assert summary["process_cpu_peak_percent"] == 70.0


def test_summarize_empty_samples() -> None:
    summary = summarize_samples([])
    assert summary["count"] == 0
    assert summary["rss_peak_mb"] is None
    assert summary["cpu_mean_percent"] is None
