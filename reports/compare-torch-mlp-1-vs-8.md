# TensorTrace regression comparison

**Baseline:** `run-20261019-083306-torch-mlp` (torch/mlp, threads=1)  
**Candidate:** `run-20261019-083309-torch-mlp` (torch/mlp, threads=8)  
**Mean-latency threshold:** 10.0%

| Metric | Baseline | Candidate | Delta | % Change | Status |
| --- | --- | --- | --- | --- | --- |
| mean_latency_ms | 0.120 | 0.036 | -0.084 | -70.33% | improvement |
| median_latency_ms | 0.064 | 0.035 | -0.029 | -45.33% | improvement |
| p95_latency_ms | 0.448 | 0.038 | -0.409 | -91.43% | improvement |
| throughput_ips | 8,337.975 | 28,106.388 | 19,768.413 | 237.09% | improvement |
| peak_rss_mb | 261.125 | 285.828 | 24.703 | 9.46% | unchanged |
| cache_miss_rate | unavailable | unavailable | unavailable | unavailable | unavailable |
| ipc | unavailable | unavailable | unavailable | unavailable | unavailable |

**Result: PASS** - no mean-latency regression beyond the threshold.

