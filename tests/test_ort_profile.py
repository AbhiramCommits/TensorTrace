"""Tests for ONNX Runtime profile JSON parsing and aggregation."""

from __future__ import annotations

import json
from pathlib import Path

from tensortrace.profilers.onnx_profiler import (
    parse_ort_profile,
    rows_to_layers,
    select_measured_rows,
    shape_bytes,
)


def _load(fixtures_dir: Path):
    return json.loads((fixtures_dir / "ort_profile.json").read_text())


def test_parse_ort_profile_exact_rows(fixtures_dir: Path) -> None:
    rows = parse_ort_profile(_load(fixtures_dir))
    assert len(rows) == 3
    assert [row["duration_us"] for row in rows] == [325.0, 28.0, 12.0]
    assert [row["run_index"] for row in rows] == [0, 1, 2]
    for row in rows:
        assert row["name"] == "fused /fc/Gemm_kernel_time"
        assert row["op_type"] == "FusedGemm"
        assert row["output_bytes"] == 32
        assert row["self_memory_bytes"] == 560
        assert row["node_index"] == "2"


def test_parse_ort_profile_accepts_list_and_ignores_session_events(fixtures_dir: Path) -> None:
    obj = _load(fixtures_dir)
    events = obj["traceEvents"] if isinstance(obj, dict) else obj
    rows = parse_ort_profile(events)
    assert len(rows) == 3
    assert all(row["name"] != "model_run" for row in rows)


def test_parse_ort_profile_no_node_events() -> None:
    assert (
        parse_ort_profile({"traceEvents": [{"cat": "Session", "name": "model_run", "ts": 1}]})
        == []
    )
    assert parse_ort_profile({"traceEvents": []}) == []
    assert parse_ort_profile("not-a-list") == []


def test_parse_ort_profile_skips_missing_duration() -> None:
    events = [
        {"cat": "Node", "name": "X", "ts": 1, "args": {"op_type": "X"}},
        {"cat": "Node", "name": "Y", "ts": 2, "dur": 5, "args": {"op_type": "Y"}},
    ]
    rows = parse_ort_profile(events)
    assert len(rows) == 1
    assert rows[0]["name"] == "Y"


def test_select_measured_rows_by_run_index(fixtures_dir: Path) -> None:
    rows = parse_ort_profile(_load(fixtures_dir))
    selected = select_measured_rows(rows, warmup=1, iterations=2)
    assert [row["duration_us"] for row in selected] == [28.0, 12.0]


def test_select_measured_rows_fallback_without_run_index() -> None:
    rows = [{"name": f"n{index}", "duration_ms": 1.0, "run_index": None} for index in range(12)]
    selected = select_measured_rows(rows, warmup=1, iterations=3)
    assert len(selected) == 9
    assert selected[0]["name"] == "n3"


def test_rows_to_layers_by_name(fixtures_dir: Path) -> None:
    rows = parse_ort_profile(_load(fixtures_dir))
    layers = rows_to_layers(rows, iterations=3)
    assert len(layers) == 1
    layer = layers[0]
    assert layer.name == "fused /fc/Gemm_kernel_time"
    assert layer.op_type == "FusedGemm"
    assert layer.total_ms == 0.365
    assert layer.mean_ms == 0.121667
    assert layer.calls == 3
    assert layer.output_bytes == 32
    assert layer.self_memory_bytes == 560
    assert layer.percent_of_total == 100.0


def test_rows_to_layers_by_op_type() -> None:
    rows = [
        {"name": "conv0", "op_type": "Conv", "duration_ms": 1.0},
        {"name": "conv1", "op_type": "Conv", "duration_ms": 3.0},
        {"name": "relu0", "op_type": "Relu", "duration_ms": 1.0},
    ]
    layers = rows_to_layers(rows, iterations=2, group_by="op_type")
    assert [layer.name for layer in layers] == ["Conv", "Relu"]
    assert layers[0].total_ms == 4.0
    assert layers[0].mean_ms == 2.0
    assert layers[0].percent_of_total == 80.0


def test_shape_bytes_formats() -> None:
    assert shape_bytes([{"float": [2, 4]}]) == 32
    assert shape_bytes([[1, 3, 4]]) == 48
    assert shape_bytes([{"int64": [2, 2]}]) == 32
    assert shape_bytes([{"float": [1, "batch"]}]) is None
    assert shape_bytes([]) is None
    assert shape_bytes(None) is None
