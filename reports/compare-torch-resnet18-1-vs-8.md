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

