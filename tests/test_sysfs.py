"""Tests for /sys topology parsers and affinity helpers."""

from __future__ import annotations

from pathlib import Path

from tensortrace.system.sysfs import (
    cache_topology,
    current_affinity,
    current_affinity_info,
    format_affinity_mask,
    format_cpu_list,
    nodes_for_cpus,
    numa_topology,
    parse_cpu_list,
    parse_node_meminfo,
    parse_size_bytes,
)


def test_parse_cpu_list_ranges_and_singles() -> None:
    assert parse_cpu_list("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert parse_cpu_list("0-1\n4\n") == [0, 1, 4]
    assert parse_cpu_list("") == []
    assert parse_cpu_list("5-3,bad,7") == [7]


def test_format_cpu_list_round_trip() -> None:
    assert format_cpu_list([0, 1, 2, 3, 8, 10, 11]) == "0-3,8,10-11"
    assert format_cpu_list([4, 5]) == "4-5"
    assert format_cpu_list([]) == ""


def test_format_affinity_mask() -> None:
    assert format_affinity_mask([0, 1, 7]) == "83"
    assert format_affinity_mask([0]) == "1"
    assert format_affinity_mask([]) == "0"


def test_parse_size_bytes() -> None:
    assert parse_size_bytes("32K") == 32768
    assert parse_size_bytes("1024K") == 1048576
    assert parse_size_bytes("2M") == 2097152
    assert parse_size_bytes("1G") == 1073741824
    assert parse_size_bytes("64") == 64
    assert parse_size_bytes("") is None
    assert parse_size_bytes("bad") is None


def test_parse_node_meminfo(fixtures_dir: Path) -> None:
    text = (fixtures_dir / "sysfs/node/node0/meminfo").read_text()
    parsed = parse_node_meminfo(text)
    assert parsed == {"MemTotal": 16384000, "MemFree": 8192000, "MemUsed": 8192000}


def test_numa_topology_fixture(fixtures_dir: Path) -> None:
    topology = numa_topology(fixtures_dir / "sysfs/node")
    assert topology["available"] is True
    assert topology["reason"] is None
    assert list(topology["nodes"]) == ["node0", "node1"]
    assert topology["nodes"]["node0"]["cpus"] == [0, 1, 2, 3]
    assert topology["nodes"]["node1"]["cpu_list"] == "4-7"
    assert topology["nodes"]["node0"]["distance"] == [10, 21]
    assert topology["nodes"]["node1"]["meminfo"]["MemUsed"] == 12288000


def test_numa_topology_missing_root(tmp_path: Path) -> None:
    topology = numa_topology(tmp_path / "no-such-node")
    assert topology["available"] is False
    assert "not present" in topology["reason"]
    assert topology["nodes"] == {}


def test_cache_topology_fixture_deduplicates_shared_caches(fixtures_dir: Path) -> None:
    topology = cache_topology(fixtures_dir / "sysfs/cpu")
    assert topology["available"] is True
    assert len(topology["caches"]) == 4
    assert set(topology["levels"]) == {"L1d", "L1i", "L2u", "L3u"}
    l3 = topology["levels"]["L3u"]
    assert l3["size_bytes"] == 32768 * 1024
    assert l3["line_size"] == 64
    assert l3["sharing"][0]["shared_cpu_list"] == "0-7"
    l1 = topology["levels"]["L1d"]
    assert l1["size_bytes"] == 32768
    assert l1["sharing"] == [{"shared_cpu_list": "0", "cpus": [0]}]


def test_cache_topology_missing_root(tmp_path: Path) -> None:
    topology = cache_topology(tmp_path / "no-such-cpu")
    assert topology["available"] is False
    assert "not present" in topology["reason"]
    assert topology["caches"] == []
    assert topology["levels"] == {}


def test_nodes_for_cpus() -> None:
    topology = {
        "available": True,
        "nodes": {
            "node0": {"cpus": [0, 1, 2, 3]},
            "node1": {"cpus": [4, 5, 6, 7]},
        },
    }
    assert nodes_for_cpus([0, 1], topology) == ["node0"]
    assert nodes_for_cpus([0, 5], topology) == ["node0", "node1"]
    assert nodes_for_cpus([99], topology) == []


def test_current_affinity_returns_real_cpus() -> None:
    cpus = current_affinity()
    assert cpus
    assert all(isinstance(cpu, int) for cpu in cpus)
    info = current_affinity_info()
    assert info["cpus"] == cpus
    assert info["mask"] == format_affinity_mask(cpus)
    assert info["source"] in {
        "os.sched_getaffinity",
        "psutil",
        "os.cpu_count fallback",
    }
