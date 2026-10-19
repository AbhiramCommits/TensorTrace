# TensorTrace benchmark sweep

Generated: 2026-10-19T08:45:59.771778+00:00

## Host

| Field | Value |
| --- | --- |
| CPU | Apple M3 |
| Physical / logical cores | 8 / 8 |
| NUMA | unavailable (/sys/devices/system/node is not present on this host) |
| perf | unavailable (perf binary was not found on PATH) |
| Models | resnet18, mlp |
| Backends | torch, onnx |
| Thread counts | 1, 8 |

## Measurements

| Backend | Model | Threads | Mean ms | Median ms | p95 ms | Throughput (inf/s) | Peak RSS MB | Cache miss rate / IPC | Top layer | Top layer % | Run JSON |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch | resnet18 | 1 | 41.813 | 36.831 | 58.822 | 23.92 | 343.2 | unavailable | conv1 | 15.4 | `reports/sweep-runs/run-20261019-084556-torch-resnet18.json` |
| torch | resnet18 | 8 | 23.334 | 23.227 | 24.558 | 42.86 | 368.3 | unavailable | conv1 | 11.1 | `reports/sweep-runs/run-20261019-084557-torch-resnet18.json` |
| torch | mlp | 1 | 0.036 | 0.036 | 0.037 | 27,949.28 | 326.6 | unavailable | fc1 | 28.2 | `reports/sweep-runs/run-20261019-084557-torch-mlp.json` |
| torch | mlp | 8 | 0.036 | 0.036 | 0.041 | 27,450.58 | 326.6 | unavailable | fc1 | 28.2 | `reports/sweep-runs/run-20261019-084557-torch-mlp-2.json` |
| onnx | resnet18 | 1 | 64.203 | 63.908 | 67.276 | 15.58 | 425.3 | unavailable | /layer4/layer4.1/conv2/Conv_kernel_time | 7.4 | `reports/sweep-runs/run-20261019-084558-onnx-resnet18.json` |
| onnx | resnet18 | 8 | 66.701 | 66.014 | 97.018 | 14.99 | 426.6 | unavailable | /layer3/layer3.1/conv1/Conv_kernel_time | 8.6 | `reports/sweep-runs/run-20261019-084559-onnx-resnet18.json` |
| onnx | mlp | 1 | 0.024 | 0.024 | 0.027 | 41,782.44 | 343.3 | unavailable | fused /fc2/Gemm_kernel_time | 47.2 | `reports/sweep-runs/run-20261019-084559-onnx-mlp.json` |
| onnx | mlp | 8 | 0.092 | 0.043 | 0.543 | 10,836.19 | 343.3 | unavailable | fused /fc1/Gemm_kernel_time | 77.2 | `reports/sweep-runs/run-20261019-084559-onnx-mlp-2.json` |

## 1-thread vs physical-core scaling

| Backend | Model | 1 thread mean ms | Physical-core mean ms | Speedup | Mean change | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| torch | resnet18 | 41.813 | 23.334 | 1.79x | -44.19% | PASS |
| torch | mlp | 0.036 | 0.036 | 0.98x | +1.82% | PASS |
| onnx | resnet18 | 64.203 | 66.701 | 0.96x | +3.89% | PASS |
| onnx | mlp | 0.024 | 0.092 | 0.26x | +285.59% | REGRESSION |

## Recommendations emitted

- `torch/resnet18` threads=1: REMEASURE, PERF_UNAVAILABLE
- `torch/resnet18` threads=8: PERF_UNAVAILABLE
- `torch/mlp` threads=1: PERF_UNAVAILABLE
- `torch/mlp` threads=8: PERF_UNAVAILABLE
- `onnx/resnet18` threads=1: PERF_UNAVAILABLE
- `onnx/resnet18` threads=8: REMEASURE, PERF_UNAVAILABLE
- `onnx/mlp` threads=1: PERF_UNAVAILABLE
- `onnx/mlp` threads=8: REMEASURE, PERF_UNAVAILABLE
