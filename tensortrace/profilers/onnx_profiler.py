"""ONNX Runtime inference profiler with ORT tracing JSON parsing."""

from __future__ import annotations

import json
import time
import warnings
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from tensortrace.config import RunConfig
from tensortrace.profilers.models import build_torch_model, make_input, model_metadata
from tensortrace.profilers.schema import LayerTiming, RunResult, assign_percentages, latency_stats
from tensortrace.profilers.torch_profiler import configure_torch_threads

_DTYPE_BYTES = {
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


def _shape_elements(shape: Any) -> int | None:
    """Return the element count of one shape, or ``None`` when a dim is symbolic."""
    if not isinstance(shape, list) or not shape:
        return None
    count = 1
    for dim in shape:
        if not isinstance(dim, int):
            return None
        count *= dim
    return count


def shape_bytes(type_shape: Any) -> int | None:
    """Estimate bytes from an ORT ``*_type_shape`` argument.

    ORT encodes tensors as ``[{"float": [1, 3, 224, 224]}]`` in newer releases
    and as ``[[1, 3, 224, 224]]`` in older ones. Unknown shapes return ``None``
    rather than a fabricated size.
    """
    if not isinstance(type_shape, list):
        return None
    total = 0
    found = False
    for item in type_shape:
        if isinstance(item, dict):
            for dtype, shape in item.items():
                elements = _shape_elements(shape)
                if elements is None:
                    return None
                total += elements * _DTYPE_BYTES.get(str(dtype).lower(), 4)
                found = True
        elif isinstance(item, list):
            elements = _shape_elements(item)
            if elements is None:
                return None
            total += elements * 4
            found = True
    return total if found else None


def parse_ort_profile(json_obj: Any) -> list[dict[str, Any]]:
    """Parse an ORT profiling JSON object into per-node latency rows.

    Only ``Node`` trace events are returned. ``run_index`` is derived from the
    timestamps of ``model_run`` session events so warmup rows can be removed
    after the fact.
    """
    if isinstance(json_obj, dict):
        events = json_obj.get("traceEvents", [])
    elif isinstance(json_obj, list):
        events = json_obj
    else:
        return []
    if not isinstance(events, list):
        return []

    run_starts = sorted(
        float(event["ts"])
        for event in events
        if isinstance(event, dict) and event.get("name") == "model_run" and "ts" in event
    )
    rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        args = event.get("args") or {}
        if event.get("cat") != "Node" and "op_type" not in args:
            continue
        duration_us = event.get("dur")
        if duration_us is None:
            continue
        try:
            duration_us = float(duration_us)
        except (TypeError, ValueError):
            continue
        op_name = str(args.get("op_name") or args.get("op_type") or "")
        name = str(event.get("name") or op_name or "unknown")
        op_type = str(args.get("op_type") or op_name or name.split("_")[0])
        timestamp = event.get("ts")
        run_index: int | None = None
        if run_starts and timestamp is not None:
            try:
                run_index = bisect_right(run_starts, float(timestamp)) - 1
            except (TypeError, ValueError):
                run_index = None
        memory = args.get("mem_bytes_requested_in_use")
        try:
            self_memory = int(memory) if memory is not None else None
        except (TypeError, ValueError):
            self_memory = None
        rows.append(
            {
                "name": name,
                "op_type": op_type,
                "duration_us": round(duration_us, 3),
                "duration_ms": round(duration_us / 1000.0, 6),
                "start_us": float(timestamp) if timestamp is not None else None,
                "thread": event.get("tid"),
                "run_index": run_index,
                "node_index": args.get("node_index"),
                "output_bytes": shape_bytes(args.get("output_type_shape")),
                "self_memory_bytes": self_memory,
            }
        )
    return rows


def select_measured_rows(
    rows: list[dict[str, Any]], warmup: int, iterations: int
) -> list[dict[str, Any]]:
    """Drop warmup node rows from a parsed ORT trace.

    Uses ``run_index`` when every row has one. Otherwise falls back to the
    common static-graph case where each run emits the same number of nodes.
    """
    if not rows:
        return []
    if all(row.get("run_index") is not None for row in rows):
        selected = [row for row in rows if int(row["run_index"]) >= warmup]
        if selected:
            return selected
    total_runs = warmup + iterations
    if total_runs > 0 and len(rows) % total_runs == 0:
        nodes_per_run = len(rows) // total_runs
        return rows[warmup * nodes_per_run :]
    return rows


def rows_to_layers(
    rows: list[dict[str, Any]], iterations: int, group_by: str = "name"
) -> list[LayerTiming]:
    """Aggregate node rows by name or op type into :class:`LayerTiming` rows."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_by) or row.get("name") or "unknown")].append(row)
    layers: list[LayerTiming] = []
    for key, items in grouped.items():
        total_ms = sum(float(item["duration_ms"]) for item in items)
        output_values = [
            int(item["output_bytes"]) for item in items if item.get("output_bytes") is not None
        ]
        memory_values = [
            int(item["self_memory_bytes"])
            for item in items
            if item.get("self_memory_bytes") is not None
        ]
        layers.append(
            LayerTiming(
                name=key,
                op_type=str(items[0].get("op_type") or key),
                mean_ms=round(total_ms / iterations, 6),
                total_ms=round(total_ms, 6),
                calls=len(items),
                output_bytes=max(output_values) if output_values else None,
                self_memory_bytes=max(memory_values) if memory_values else None,
            )
        )
    layers.sort(key=lambda layer: layer.total_ms, reverse=True)
    assign_percentages(layers)
    return layers


def export_onnx(
    config: RunConfig, cache_dir: Path = Path(".onnx-cache")
) -> tuple[Path, dict[str, Any]]:
    """Export a deterministic random-weight model to ONNX, reusing the cache when possible."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{config.model}-b{config.batch_size}-s{config.seed}.onnx"
    model, input_shape = build_torch_model(config.model, config.seed)
    cache_hit = target.exists()
    if not cache_hit:
        tensor = make_input(input_shape, config.batch_size, config.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(
                model,
                tensor,
                target,
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
    return target, {
        "model": model_metadata(model),
        "input_shape": [config.batch_size, *input_shape],
        "cache_hit": cache_hit,
    }


def profile_onnx(config: RunConfig, cache_dir: Path = Path(".onnx-cache")) -> RunResult:
    """Export and profile an ONNX Runtime inference session."""
    threads = configure_torch_threads(config)
    model_path, export_meta = export_onnx(config, cache_dir=cache_dir)
    model, input_shape = build_torch_model(config.model, config.seed)
    tensor = make_input(input_shape, config.batch_size, config.seed)
    array = tensor.detach().cpu().numpy().astype(np.float32)

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    warnings_list: list[str] = []

    for _ in range(config.warmup):
        session.run(None, {input_name: array})
    latencies: list[float] = []

    for _ in range(config.iterations):
        started = time.perf_counter_ns()
        session.run(None, {input_name: array})
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)

    rows: list[dict[str, Any]] = []
    try:
        profile_path = Path(session.end_profiling())
        try:
            profile_obj = json.loads(profile_path.read_text(encoding="utf-8"))
        finally:
            profile_path.unlink(missing_ok=True)
        all_rows = parse_ort_profile(profile_obj)
        rows = select_measured_rows(all_rows, config.warmup, config.iterations)
    except (OSError, ValueError, RuntimeError) as exc:
        warnings_list.append(f"ORT profiling trace could not be parsed: {exc}")

    layers = rows_to_layers(rows, config.iterations, group_by="name")
    op_types = rows_to_layers(rows, config.iterations, group_by="op_type")
    if not layers:
        warnings_list.append("no ORT node rows were captured; per-layer timings are empty")

    return RunResult(
        run_id="",
        backend="onnx",
        model=config.model,
        config=config.to_dict(),
        layers=layers,
        op_types=op_types,
        iteration_latencies_ms=[round(value, 6) for value in latencies],
        latency=latency_stats(latencies),
        warnings=warnings_list,
        extra={
            **export_meta,
            "onnx_path": str(model_path),
            "provider": "CPUExecutionProvider",
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "thread_count": threads,
            "model_parameters": model_metadata(model)["parameter_count"],
            "profiled_rows": len(rows),
        },
    )
