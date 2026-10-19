# Changelog

All notable changes to TensorTrace are documented in this file.

## [0.1.0] - 2026-10-19

### Added

- PyTorch inference profiler with per-leaf-module hooks, output tensor byte
  accounting, `torch.profiler` operator CPU time, and self memory.
- ONNX Runtime profiler with deterministic export caching, ORT tracing, pure
  per-node JSON parsing, warmup-row filtering, and `op_type` aggregation.
- Linux telemetry layer for `/proc/stat`, `/proc/self/status`,
  `/proc/meminfo`, NUMA sysfs, CPU cache sysfs, and process affinity.
- `perf stat -x,` invocation with permission probing, `<not supported>` /
  `<not counted>` parsing, and explicit unavailable results.
- Before/after `SystemSnapshot`, background CPU/RSS sampling, and time-series
  summaries.
- Hotspot analysis: top layers, 80% cumulative attribution, memory hotspots,
  cache miss rate, IPC, coefficient of variation, and NUMA/thread placement.
- Deterministic recommendation rules with cited measured evidence:
  `PIN_NUMA`, `THREAD_OVERSUBSCRIBE`, `SET_THREADS`, `ORT_GRAPH_OPT`,
  `REDUCE_BATCH`, `REMEASURE`, `CACHE_BLOCKING`, and `PERF_UNAVAILABLE`.
- Markdown and Jinja2 HTML reports with environment, config, latency, layer,
  memory, perf, time-series, and recommendation sections.
- `tensortrace compare` regression table with a configurable mean-latency gate
  that exits non-zero for CI.
- `tensortrace sysinfo` JSON output with explicit unavailable states.
- Makefile targets (`setup`, `lint`, `test`, `profile`, `report`, `clean`) and
  `scripts/bench_sweep.py` for the 2 x 2 x 2 benchmark matrix.
- Pytest suite covering every parser, analysis rule, comparison gate, report
  rendering, graceful perf degradation, and a real tiny-model integration run.
- GitHub Actions CI for Python 3.10 and 3.11 with ruff, coverage, the slow
  integration test on 3.11, and report artifact upload.
- Project documentation: README with measured results, contributing guide, and
  MIT license.
