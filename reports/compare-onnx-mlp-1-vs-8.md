# TensorTrace regression comparison

**Baseline:** `run-20261019-083326-onnx-mlp` (onnx/mlp, threads=1)  
**Candidate:** `run-20261019-083330-onnx-mlp` (onnx/mlp, threads=8)  
**Mean-latency threshold:** 10.0%

| Metric | Baseline | Candidate | Delta | % Change | Status |
| --- | --- | --- | --- | --- | --- |
| mean_latency_ms | 0.024 | 0.035 | 0.011 | 45.42% | regression |
| median_latency_ms | 0.023 | 0.033 | 0.010 | 41.51% | regression |
| p95_latency_ms | 0.029 | 0.044 | 0.015 | 53.50% | regression |
| throughput_ips | 42,068.597 | 28,929.577 | -13,139.020 | -31.23% | regression |
| peak_rss_mb | 306.922 | 307.281 | 0.359 | 0.12% | unchanged |
| cache_miss_rate | unavailable | unavailable | unavailable | unavailable | unavailable |
| ipc | unavailable | unavailable | unavailable | unavailable | unavailable |

**Result: REGRESSION** - gated metric(s): mean_latency_ms

