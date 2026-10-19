# TensorTrace regression comparison

**Baseline:** `run-20261019-083313-onnx-resnet18` (onnx/resnet18, threads=1)  
**Candidate:** `run-20261019-083343-onnx-resnet18` (onnx/resnet18, threads=8)  
**Mean-latency threshold:** 10.0%

| Metric | Baseline | Candidate | Delta | % Change | Status |
| --- | --- | --- | --- | --- | --- |
| mean_latency_ms | 77.621 | 60.832 | -16.789 | -21.63% | improvement |
| median_latency_ms | 79.011 | 63.489 | -15.522 | -19.65% | improvement |
| p95_latency_ms | 81.919 | 72.963 | -8.956 | -10.93% | improvement |
| throughput_ips | 12.883 | 16.439 | 3.556 | 27.60% | improvement |
| peak_rss_mb | 401.031 | 389.328 | -11.703 | -2.92% | unchanged |
| cache_miss_rate | unavailable | unavailable | unavailable | unavailable | unavailable |
| ipc | unavailable | unavailable | unavailable | unavailable | unavailable |

**Result: PASS** - no mean-latency regression beyond the threshold.

