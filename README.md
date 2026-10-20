# TensorTrace

TensorTrace is a Linux-aware inference profiler for PyTorch and ONNX Runtime.
It measures what the framework sees (per-layer latency, operator time, tensor
bytes) and what the kernel sees (per-process RSS, CPU time series, context
switches, and `perf stat` counters) in the same measurement window, then turns
the combined data into concrete tuning recommendations and versioned Markdown +
HTML reports.

Most profilers stop at a framework trace. TensorTrace exists to answer the next
question: *is this layer slow because of the model, or because the process is
spread across NUMA nodes, oversubscribed on SMT threads, or missing cache?*
When the kernel counters are unavailable, it says so explicitly instead of
inventing a number.

## What it does

- Runs real inference with random-weight PyTorch models and the same models
  exported to ONNX Runtime.
- Collects per-leaf-module wall-clock latency and output tensor bytes with
  forward hooks, plus a `torch.profiler` operator pass with `record_shapes` and
  `profile_memory`.
- Parses ONNX Runtime profiling JSON into per-node rows and aggregates by
  `op_type` (pure function, fixture-tested).
- Samples process RSS, thread count, and CPU utilization in a background thread
  while the workload runs.
- Reads `/proc/stat`, `/proc/self/status`, `/proc/meminfo`, NUMA node topology,
  CPU cache topology, and process affinity. On hosts without these files every
  field is marked `unavailable`.
- Runs the workload under `perf stat -x,` when `perf` is usable, so hardware
  counters cover the same window as the framework timings. If perf is missing
  or blocked by `perf_event_paranoid`, the report records the reason.
- Computes 80% cumulative latency attribution, memory hotspots, cache miss rate,
  IPC, coefficient of variation, and NUMA/thread placement checks.
- Emits deterministic `Recommendation` objects whose evidence strings contain
  the measured numbers that triggered them.
- Renders `reports/<run-id>.md` and `reports/<run-id>.html`, and compares a
  candidate run against a baseline with a CI-gating mean-latency threshold.

## Architecture

```
tensortrace.cli
  ├── run      -> profilers.runner.profile_run
  │                 ├── system.procfs.SystemSnapshot / TelemetrySampler
  │                 ├── system.perf.run_perf_stat  (child process when available)
  │                 └── profilers.torch_profiler / profilers.onnx_profiler
  ├── sysinfo  -> system.sysfs.collect_sysinfo
  ├── report   -> report.render.render_report
  └── compare  -> analysis.compare.compare_runs -> report.render
```

| Module | Owns |
| --- | --- |
| `tensortrace/cli.py` | Argument parsing, command dispatch, exit codes |
| `tensortrace/config.py` | `RunConfig` validation and serialization |
| `tensortrace/profilers/models.py` | Deterministic ResNet-18 and MLP definitions |
| `tensortrace/profilers/torch_profiler.py` | Leaf-module hooks, torch profiler operators |
| `tensortrace/profilers/onnx_profiler.py` | ONNX export, ORT session, profile JSON parsing |
| `tensortrace/profilers/runner.py` | One-run orchestration and result assembly |
| `tensortrace/profilers/worker.py` | Child process used for the perf-instrumented window |
| `tensortrace/profilers/schema.py` | Versioned `LayerTiming` / `RunResult` schema and statistics |
| `tensortrace/system/procfs.py` | Pure `/proc` parsers, snapshots, sampling thread |
| `tensortrace/system/sysfs.py` | NUMA/cache topology, affinity, sysinfo payload |
| `tensortrace/system/perf.py` | perf discovery, invocation, CSV parsing |
| `tensortrace/analysis/hotspots.py` | Pure attribution, cache, variance, NUMA checks |
| `tensortrace/analysis/recommend.py` | Deterministic recommendation rules |
| `tensortrace/analysis/compare.py` | Baseline/candidate metric deltas and gate |
| `tensortrace/report/render.py` | Markdown writer and Jinja2 HTML renderer |

## Install

Python 3.11+ is required (the pinned numpy 2.4 and onnxruntime 1.31 have no
Python 3.10 wheels). The build host used Python 3.11.14.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

On a Linux CPU-only host, the CPU wheel index keeps the install smaller:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
pip install -e .
```

## Usage

### `tensortrace sysinfo`

```bash
tensortrace sysinfo
```

Real output from this macOS build host (Apple M3), including the explicit
`unavailable` entries:

```json
{
  "host": {
    "system": "Darwin",
    "release": "27.0.0",
    "machine": "arm64",
    "python": "3.11.14",
    "linux": false
  },
  "cpu": {
    "model": "Apple M3",
    "logical_cores": 8,
    "physical_cores": 8,
    "affinity": {
      "cpus": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
      ],
      "mask": "ff",
      "source": "os.cpu_count fallback"
    },
    "affinity_numa_nodes": []
  },
  "memory": {
    "total_kb": 16777216
  },
  "numa": {
    "available": false,
    "reason": "/sys/devices/system/node is not present on this host",
    "nodes": {}
  },
  "cache": {
    "available": false,
    "reason": "/sys/devices/system/cpu is not present on this host",
    "caches": [],
    "levels": {}
  },
  "perf": {
    "available": false,
    "reason": "perf binary was not found on PATH"
  },
  "procfs": {
    "available": false,
    "reason": "/proc/stat or /proc/self/status is not readable on this host"
  }
}
```

The full payload is committed as `reports/sysinfo.json`.

### `tensortrace run`

```bash
tensortrace run --backend torch --model resnet18 \
  --batch-size 1 --iters 10 --warmup 3 --threads 1 --out reports

tensortrace run --backend onnx --model mlp \
  --batch-size 1 --iters 10 --warmup 3 --threads 0 --out reports
```

`--threads 0` means "all logical CPUs available to the process" (8 here).
`--backend both` runs both frameworks in sequence. Real output from the Step 3
matrix:

```text
[torch/resnet18] mean=77.370 ms median=77.610 ms p95=99.232 ms throughput=12.92 inferences/s -> reports/run-20261019-083255-torch-resnet18.json
[torch/resnet18] mean=21.012 ms median=20.911 ms p95=22.205 ms throughput=47.59 inferences/s -> reports/run-20261019-083301-torch-resnet18.json
[torch/mlp] mean=0.120 ms median=0.064 ms p95=0.448 ms throughput=8337.97 inferences/s -> reports/run-20261019-083306-torch-mlp.json
[torch/mlp] mean=0.036 ms median=0.035 ms p95=0.038 ms throughput=28106.39 inferences/s -> reports/run-20261019-083309-torch-mlp.json
[onnx/resnet18] mean=77.621 ms median=79.011 ms p95=81.919 ms throughput=12.88 inferences/s -> reports/run-20261019-083313-onnx-resnet18.json
[onnx/resnet18] mean=60.832 ms median=63.489 ms p95=72.963 ms throughput=16.44 inferences/s -> reports/run-20261019-083343-onnx-resnet18.json
[onnx/mlp] mean=0.024 ms median=0.023 ms p95=0.029 ms throughput=42068.60 inferences/s -> reports/run-20261019-083326-onnx-mlp.json
[onnx/mlp] mean=0.035 ms median=0.033 ms p95=0.044 ms throughput=28929.58 inferences/s -> reports/run-20261019-083330-onnx-mlp.json
```

Every JSON contains the full config, environment, per-layer rows, per-iteration
latencies, CPU/memory time series, perf counters, and warnings.

Those lines are from the Step 3 separate-process matrix. The Step 7 sweep
re-ran the same matrix inside one process; its authoritative table is in
[Results](#results) and `reports/sweep-summary.md`. Absolute latencies shifted
because this host is not an isolated benchmark machine.

### `tensortrace report`

```bash
tensortrace report reports/run-20261019-083255-torch-resnet18.json
```

Real output:

```text
markdown: reports/run-20261019-083255-torch-resnet18.md
html: reports/run-20261019-083255-torch-resnet18.html
```

The Markdown and HTML reports include environment, config, latency summary, the
top-10 layer table, memory hotspots, the perf table (with `unavailable` rows),
the CPU/memory summary, hotspot flags, and every recommendation with its
evidence.

### `tensortrace compare`

```bash
tensortrace compare \
  reports/run-20261019-083255-torch-resnet18.json \
  reports/run-20261019-083301-torch-resnet18.json
```

Real output (exit code 0 because latency improved):

```text
# TensorTrace regression comparison

**Baseline:** `run-20261019-083255-torch-resnet18` (torch/resnet18, threads=1)
**Candidate:** `run-20261019-083301-torch-resnet18` (torch/resnet18, threads=8)
**Mean-latency threshold:** 10.0%

| Metric | Baseline | Candidate | Delta | % Change | Status |
| --- | --- | --- | --- | --- | --- |
| mean_latency_ms | 77.370 | 21.012 | -56.357 | -72.84% | improvement |
| median_latency_ms | 77.610 | 20.911 | -56.700 | -73.06% | improvement |
| p95_latency_ms | 99.232 | 22.205 | -77.027 | -77.62% | improvement |
| throughput_ips | 12.925 | 47.592 | 34.667 | 268.21% | improvement |
| peak_rss_mb | 322.172 | 341.906 | 19.734 | 6.13% | unchanged |
| cache_miss_rate | unavailable | unavailable | unavailable | unavailable | unavailable |
| ipc | unavailable | unavailable | unavailable | unavailable | unavailable |

**Result: PASS** - no mean-latency regression beyond the threshold.
```

The CI gate fails when mean latency regresses beyond 10% (override with
`--threshold`, a fraction). This ONNX MLP comparison exits 1:

```text
| mean_latency_ms | 0.024 | 0.035 | 0.011 | 45.42% | regression |
**Result: REGRESSION** - gated metric(s): mean_latency_ms
```

The complete comparison tables are committed as `reports/compare-*.md`.

## Results

The Step 7 sweep (`make profile`, committed as `reports/sweep-summary.md` with
raw JSON under `reports/sweep-runs/`) is the latest authoritative matrix. The
Step 3 JSONs remain committed alongside it. All numbers were produced by
commands in this README on the build host described in `ENVIRONMENT.md`.
Nothing is estimated.

### Host and software

| Item | Measured value |
| --- | --- |
| CPU | Apple M3 |
| Cores | 8 physical / 8 logical |
| Memory | 16 GiB (16,777,216 KiB) |
| OS | macOS 27.0.1, Darwin 27.0.0, arm64 |
| NUMA nodes | unavailable (macOS exposes no `/sys/devices/system/node`); not reported as zero |
| Python | 3.11.14 |
| torch | 2.14.1 |
| torchvision | 0.29.1 |
| onnx | 1.23.2 |
| onnxruntime | 1.31.0 |
| perf | unavailable (`perf binary was not found on PATH`) |

**Measurement conditions.** The host was not idle. The Step 3 matrix ran while
the 1-minute load average was about 12–14 because macOS storage-management
services were active on a nearly full disk. The Step 7 sweep ran later with a
1-minute load of 7.66, a 5-minute load of 16.90, and a 15-minute load of 16.17.
Absolute latencies shifted
between the two matrices: for example, torch/resnet18 at 1 thread measured
77.370 ms in Step 3 and 41.813 ms in the sweep. Both datasets are committed;
the sweep is used below. Peak RSS is higher in the sweep because all eight
configurations ran in one process and the allocator retained memory. Several
configurations still crossed the 15% coefficient-of-variation threshold and
emitted `REMEASURE`, including the 296.079 ms ONNX ResNet-18 8-thread outlier
from Step 3 (`reports/run-20261019-083322-onnx-resnet18.json`, CV 0.371).

### Latency, throughput, and peak RSS

Sweep values, batch size 1, 10 measured iterations, 3 warmup iterations. Peak
RSS is the process peak recorded by the sampling thread (psutil source on
macOS) and includes model construction/ONNX export in the measurement window.

| Backend | Model | Threads | Mean ms | Median ms | p95 ms | Stddev ms | Throughput (inf/s) | Peak RSS MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch | resnet18 | 1 | 41.813 | 36.831 | 58.822 | 9.021 | 23.92 | 343.2 |
| torch | resnet18 | 8 | 23.334 | 23.227 | 24.558 | 0.726 | 42.86 | 368.3 |
| torch | mlp | 1 | 0.036 | 0.036 | 0.037 | 0.0004 | 27,949.28 | 326.6 |
| torch | mlp | 8 | 0.036 | 0.036 | 0.041 | 0.0018 | 27,450.58 | 326.6 |
| onnx | resnet18 | 1 | 64.203 | 63.908 | 67.276 | 1.549 | 15.58 | 425.3 |
| onnx | resnet18 | 8 | 66.701 | 66.014 | 97.018 | 13.593 | 14.99 | 426.6 |
| onnx | mlp | 1 | 0.024 | 0.024 | 0.027 | 0.0021 | 41,782.44 | 343.3 |
| onnx | mlp | 8 | 0.092 | 0.043 | 0.543 | 0.1502 | 10,836.19 | 343.3 |

### Top 3 hottest layers

Sweep values. Percentages are share of summed leaf-module or node time for that
run.

| Backend / model / threads | Top 3 layers (% of total layer time) |
| --- | --- |
| torch / resnet18 / 1 | `conv1` 15.4%; `maxpool` 13.1%; `layer1.0.conv1` 7.7% |
| torch / resnet18 / 8 | `conv1` 11.1%; `layer1.1.conv1` 6.6%; `maxpool` 6.5% |
| torch / mlp / 1 | `fc1` 28.2%; `fc2` 27.8%; `fc3` 24.3% |
| torch / mlp / 8 | `fc1` 28.2%; `fc2` 27.5%; `fc3` 24.3% |
| onnx / resnet18 / 1 | `layer4.1/conv2` 7.4%; `layer4.1/conv1` 7.3%; `conv1` 7.2% |
| onnx / resnet18 / 8 | `layer3.1/conv1` 8.6%; `layer3.0/conv2` 8.3%; `conv1` 8.2% |
| onnx / mlp / 1 | `fc2` Gemm 47.2%; `fc1` Gemm 34.0%; `fc3` Gemm 18.9% |
| onnx / mlp / 8 | `fc1` Gemm 77.2%; `fc2` Gemm 13.5%; `fc3` Gemm 9.3% |

### Kernel counters: unavailable on this host

Every run records:

```json
"perf": {
  "available": false,
  "reason": "perf binary was not found on PATH",
  "counters": {}
}
```

Therefore cache-miss rate and IPC are **unavailable**, not zero, in every
report. macOS does not provide `/proc`, `/sys/devices/system/node`, or the Linux
`perf` toolchain, so this build cannot produce live cache/IPC numbers.

The parser was still validated against real perf output captured from an
Ubuntu 22.04 Linux container (perf 5.15.216) where the virtualized PMU returned
`<not supported>` for `cycles`, `instructions`, `cache-references`, and
`cache-misses`, while `page-faults` (725) and `context-switches` (31) counted.
The fixture is `tests/fixtures/perf_stat.csv`, and the parser test asserts both
the unsupported rows and the supported counts.

### 1-thread vs all-thread scaling

Sweep values. All-thread means 8 logical CPUs on this host. The sweep computes
the same `compare_runs` gate used by `tensortrace compare`, with a 10%
mean-latency threshold.

| Workload | 1 thread mean | 8 threads mean | Speedup | Mean change | Gate |
| --- | --- | --- | --- | --- | --- |
| torch / resnet18 | 41.813 ms | 23.334 ms | 1.79x | -44.19% | PASS |
| torch / mlp | 0.036 ms | 0.036 ms | 0.98x | +1.82% | PASS |
| onnx / resnet18 | 64.203 ms | 66.701 ms | 0.96x | +3.89% | PASS |
| onnx / mlp | 0.024 ms | 0.092 ms | 0.26x | +285.59% | REGRESSION (exit 1) |

The ONNX MLP result is a real small-workload threading regression: thread-pool
synchronization and allocator jitter cost more than the 20–50 µs of Gemm work.
Its 8-thread run had a 0.043 ms median but a 0.543 ms p95 and emitted
`REMEASURE` (CV 1.628). TensorTrace's gate caught the mean regression and
returned exit code 1. The torch MLP workload is also too small to benefit from
eight threads; its 0.98x result is effectively flat rather than a speedup.

### Recommendations actually emitted

Across all committed Step 3 and Step 7 run JSONs, the engine emitted:

| Code | Severity | Where | Evidence |
| --- | --- | --- | --- |
| `PERF_UNAVAILABLE` | info | every run | `perf unavailable: perf binary was not found on PATH` |
| `REMEASURE` | warning | sweep `run-20261019-084556-torch-resnet18` (1 thread) | `CV=0.216 (stddev=9.021 ms / mean=41.813 ms) over 10 iterations > threshold 0.15` |
| `REMEASURE` | warning | sweep `run-20261019-084559-onnx-resnet18` (8 threads) | `CV=0.204 (stddev=13.593 ms / mean=66.701 ms) over 10 iterations > threshold 0.15` |
| `REMEASURE` | warning | sweep `run-20261019-084559-onnx-mlp-2` (8 threads) | `CV=1.628 (stddev=0.150 ms / mean=0.092 ms) over 10 iterations > threshold 0.15` |
| `REMEASURE` | warning | Step 3 `run-20261019-083306-torch-mlp` (1 thread) | `CV=1.008 (stddev=0.121 ms / mean=0.120 ms) over 10 iterations > threshold 0.15` |
| `REMEASURE` | warning | Step 3 `run-20261019-083322-onnx-resnet18` (8 threads, unstable repeat) | `CV=0.371 (stddev=109.711 ms / mean=296.079 ms) over 10 iterations > threshold 0.15` |

No `PIN_NUMA`, `THREAD_OVERSUBSCRIBE`, or `SET_THREADS` fired because NUMA
topology was unavailable on the host and the thread count never exceeded the 8
physical cores. No `CACHE_BLOCKING` fired because cache miss rate and IPC were
unavailable. No `ORT_GRAPH_OPT` fired because ONNX Runtime was already using
`ORT_ENABLE_ALL`. No `REDUCE_BATCH` fired because these runs did not include a
multi-batch sweep. Those rules are covered by unit tests with synthetic results
that do trigger them.

## Tests and CI

Local commands and exact output:

```text
$ make lint
.venv/bin/python -m ruff check .
All checks passed!

$ make test
.venv/bin/python -m pytest -q --cov=tensortrace --cov-report=term-missing
........................................................................ [ 90%]
........                                                                 [100%]
TOTAL                                      1453    245    386    64    81%
80 passed in 15.66s
```

- **80 tests passing**, including the `@pytest.mark.slow` integration test that
  runs a real MLP through `profile_run`.
- **81% total coverage** (branch coverage enabled) across `tensortrace/`.
- **ruff clean** with `E`, `F`, `I`, `UP`, and `B` rules.
- The suite runs offline and without root. Parser tests use captured text;
  analysis tests use synthetic `RunResult` objects.
- `.github/workflows/ci.yml` runs on `ubuntu-latest` with Python 3.11 and
  3.12, CPU torch wheels, `ruff check .`, the full suite (including slow) on
  3.11, the fast suite on 3.12, and uploads `reports/` as an artifact.

Test files:

| File | Covers |
| --- | --- |
| `tests/test_procfs.py` | `/proc/stat`, `/proc/self/status`, `/proc/meminfo`, utilization math, sampling summary |
| `tests/test_sysfs.py` | CPU lists, masks, sizes, NUMA nodes, cache dedup, affinity |
| `tests/test_perf.py` | CSV parsing including `<not supported>` / `<not counted>`, unavailable path |
| `tests/test_ort_profile.py` | ORT node parsing, run-index selection, op-type aggregation, shape bytes |
| `tests/test_hotspots.py` | top layers, 80% attribution, memory, cache/IPC, CV, NUMA checks |
| `tests/test_recommend.py` | Every recommendation code fires only on its condition and cites numbers |
| `tests/test_compare.py` | PASS exit 0, 25% regression exit 1, threshold override, non-gating metrics |
| `tests/test_report.py` | Markdown/HTML smoke tests, every emitted code, no Jinja tokens, unavailable perf |
| `tests/test_integration.py` | Real `profile_run` schema/latency and graceful perf degradation |
| `tests/test_config.py` | `RunConfig` validation and round-trip serialization |

## Limitations

- **Linux-only full telemetry.** The profiler is designed for Linux hosts. On
  macOS (this build host), `/proc`, `/sys/devices/system/node`,
  `/sys/devices/system/cpu`, and `perf` are unavailable; reports record that
  explicitly. Affinity, RSS, and sampling fall back to psutil, and CPU
  utilization falls back to psutil's system-wide sampler.
- **perf permissions.** Linux hosts need a usable `perf` binary and a
  `perf_event_paranoid` setting that permits the events. Hardware events can
  also be unsupported in VMs and containers, as the captured fixture shows.
- **Random weights.** Models use random weights (`pretrained=False` or seeded
  initialization). TensorTrace measures latency and memory, not accuracy.
- **Peak RSS includes setup.** The sampler starts before model construction and
  ONNX export, so peak RSS is a whole-run process peak rather than pure
  inference activation memory.
- **Hook and profiler overhead.** Forward hooks add per-module timing overhead,
  and the torch profiler operator table comes from a separate single iteration
  so it does not inflate the reported per-iteration latencies.
- **ONNX Runtime profiling overhead.** ORT tracing is enabled for the measured
  iterations; node times are useful for attribution but carry the profiler's
  own overhead.
- **Shared-host noise.** The build host had a load average around 12–14 from
  unrelated services. The `REMEASURE` recommendation is the intended response,
  but small deltas on this host should not be treated as production truth.
- **CPU only.** The current profilers target CPU execution providers.
- **Topology fixtures.** The NUMA/cache unit-test fixtures are Linux-ABI-shaped
  samples because the Apple host and its LinuxKit container exposed no NUMA
  node files and omitted some cache attributes. Their provenance is documented
  in `tests/fixtures/README.md`.

## Roadmap

- Run the full integration suite on a native Linux CI runner with real
  `/proc`, NUMA sysfs, and permitted `perf` counters.
- Add batch-size sweeps to exercise the `REDUCE_BATCH` rule on real data.
- Export Chrome tracing / flame graphs alongside the JSON and HTML reports.
- Add GPU profiling paths (CUDA events, Nsight counters) and a GPU-specific
  recommendation set.
- Aggregate repeated runs into confidence intervals and track regressions over
  time rather than comparing single runs.
- Add a `tensortrace sweep` command built on `scripts/bench_sweep.py`.

## License

MIT. See `LICENSE`.
