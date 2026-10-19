"""PyTorch inference profiler using forward hooks and ``torch.profiler``."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import torch
from torch import nn

from tensortrace.config import RunConfig
from tensortrace.profilers.models import build_torch_model, make_input, model_metadata
from tensortrace.profilers.schema import LayerTiming, RunResult, assign_percentages, latency_stats


def configure_torch_threads(config: RunConfig) -> int:
    """Apply the requested intra-op thread count and return the actual value."""
    torch.set_num_threads(config.effective_threads())
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Can only be set before parallel work starts; an earlier run won this race.
        pass
    return torch.get_num_threads()


def tensor_bytes(value: Any) -> int:
    """Recursively sum the bytes of tensors in a hook output."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    return 0


def extract_operators(profile: torch.profiler.profile, limit: int = 50) -> list[dict[str, Any]]:
    """Extract operator CPU time and self memory from a completed profiler session."""
    rows: list[dict[str, Any]] = []
    for event in profile.key_averages():
        if event.count == 0:
            continue
        self_cpu = float(event.self_cpu_time_total)
        self_memory = int(getattr(event, "self_cpu_memory_usage", 0) or 0)
        if self_cpu <= 0 and self_memory <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "self_cpu_time_us": round(self_cpu, 3),
                "cpu_time_us": round(float(event.cpu_time_total), 3),
                "self_cpu_memory_bytes": self_memory,
            }
        )
    rows.sort(key=lambda row: (-row["self_cpu_time_us"], row["name"]))
    return rows[:limit]


def profile_torch(config: RunConfig) -> RunResult:
    """Run a PyTorch model with per-leaf-module hooks and a torch profiler pass."""
    threads = configure_torch_threads(config)
    model, input_shape = build_torch_model(config.model, config.seed)
    tensor = make_input(input_shape, config.batch_size, config.seed)
    warnings: list[str] = []

    leaf_modules = {
        name: module
        for name, module in model.named_modules()
        if name and not any(True for _ in module.children())
    }
    module_names = {module: name for name, module in leaf_modules.items()}
    starts: dict[nn.Module, int] = {}
    current: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    def pre_hook(module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        starts[module] = time.perf_counter_ns()

    def post_hook(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        start = starts.pop(module, None)
        if start is None:
            return
        elapsed = time.perf_counter_ns() - start
        name = module_names[module]
        row = current[name]
        row[0] += elapsed
        row[1] += 1
        row[2] += tensor_bytes(output)
        row[3] += 1

    handles = [
        module.register_forward_pre_hook(pre_hook) for module in leaf_modules.values()
    ] + [module.register_forward_hook(post_hook) for module in leaf_modules.values()]

    aggregate: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    latencies: list[float] = []
    try:
        with torch.inference_mode():
            for _ in range(config.warmup):
                model(tensor)
            for _ in range(config.iterations):
                current.clear()
                started = time.perf_counter_ns()
                model(tensor)
                latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
                for name, (elapsed, calls, output_bytes, byte_calls) in current.items():
                    aggregate[name][0] += elapsed
                    aggregate[name][1] += calls
                    aggregate[name][2] += output_bytes
                    aggregate[name][3] += byte_calls
    finally:
        for handle in handles:
            handle.remove()

    layers: list[LayerTiming] = []
    for name, (elapsed, calls, output_bytes, byte_calls) in aggregate.items():
        module_type = type(leaf_modules[name]).__name__
        layers.append(
            LayerTiming(
                name=name,
                op_type=module_type,
                mean_ms=round(elapsed / 1_000_000.0 / config.iterations, 6),
                total_ms=round(elapsed / 1_000_000.0, 6),
                calls=calls,
                output_bytes=round(output_bytes / byte_calls) if byte_calls else None,
                self_memory_bytes=None,
            )
        )
    layers.sort(key=lambda layer: layer.total_ms, reverse=True)
    assign_percentages(layers)
    if not layers:
        warnings.append("no leaf-module hooks fired; per-layer timings are empty")

    operators: list[dict[str, Any]] = []
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            with torch.inference_mode():
                model(tensor)
        operators = extract_operators(profiler)
    except Exception as exc:  # torch.profiler can fail on exotic builds
        warnings.append(f"torch.profiler operator pass failed: {exc}")

    latency = latency_stats(latencies)
    return RunResult(
        run_id="",
        backend="torch",
        model=config.model,
        config=config.to_dict(),
        layers=layers,
        operators=operators,
        iteration_latencies_ms=[round(value, 6) for value in latencies],
        latency=latency,
        warnings=warnings,
        extra={
            "model": model_metadata(model),
            "input_shape": [config.batch_size, *input_shape],
            "thread_count": threads,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
    )
