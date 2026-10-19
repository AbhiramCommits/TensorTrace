# Build environment

Captured during Step 1 of the TensorTrace build. All values below were read
from the machine that produced the committed reports.

## Host

```text
$ uname -a
Darwin Abhirams-MacBook-Air.local 27.0.0 Darwin Kernel Version 27.0.0: Tue Aug 11 21:05:41 PDT 2026; root:xnu-13432.1.9~1/RELEASE_ARM64_T8122 arm64
```

- macOS product version: 27.0.1
- CPU model (`/proc/cpuinfo` is not present on Darwin; value below is from
  `sysctl -n machdep.cpu.brand_string`): **Apple M3**
- Physical cores: **8** (`sysctl -n hw.physicalcpu`)
- Logical cores: **8** (`sysctl -n hw.logicalcpu` / `os.cpu_count()`)
- Total memory: **17179869184 bytes (16 GiB)** (`sysctl -n hw.memsize`)

The build host is **not Linux**, so live `/proc`, `/sys/devices/system/node`,
`/sys/devices/system/cpu/*/cache`, and `perf` data are not available on it.
TensorTrace records those sources as `unavailable` rather than inventing
numbers. Parser fixtures were captured from an Ubuntu 22.04 Linux container
(`labbench/linux-host`, kernel 7.0.14-linuxkit, aarch64) running via Docker on
this same host.

## Python

```text
$ .venv/bin/python --version
Python 3.11.14
```

## Installed packages

| Package | Version |
| --- | --- |
| torch | 2.14.1 |
| torchvision | 0.29.1 |
| onnx | 1.23.2 |
| onnxruntime | 1.31.0 |
| psutil | 7.2.2 |
| numpy | 2.4.6 |
| Jinja2 | 3.1.6 |
| pytest | 9.1.1 |
| pytest-cov | 7.1.0 |
| ruff | 0.17.1 |

## Linux telemetry availability on the build host

| Source | Status |
| --- | --- |
| `/proc/stat`, `/proc/self/status`, `/proc/meminfo` | unavailable (Darwin host) |
| `/sys/devices/system/node/*` (NUMA) | unavailable (Darwin host) |
| `/sys/devices/system/cpu/*/cache/*` | unavailable (Darwin host) |
| `perf stat` binary | unavailable (no `perf` on PATH) |

TensorTrace falls back to psutil for process RSS, thread count, CPU sampling,
and affinity on non-Linux hosts, and labels kernel-specific metrics as
unavailable in every report.
