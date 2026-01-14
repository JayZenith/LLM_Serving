# Benchmark Results

## Concurrency Sweep Results (Qwen2.5-1.5B, RTX 4090)

| Concurrency | TPS | Avg TTFT (ms) | P95 TTFT (ms) | P99 TTFT (ms) | P95 Latency (ms) | P99 Latency (ms) |
|-------------|-----|---------------|---------------|---------------|------------------|------------------|
| 1 | 213.5 | 6.5 | 6.8 | 13.1 | 469.3 | 500.7 |
| 2 | 402.7 | 9.0 | 9.6 | 10.6 | 507.3 | 513.1 |
| 4 | 699.0 | 11.7 | 19.7 | 22.2 | 560.6 | 563.5 |
| 8 | 1289.8 | 14.1 | 23.0 | 23.6 | 588.0 | 588.2 |
| 16 | 2367.8 | 21.7 | 32.7 | 33.4 | 560.6 | 561.0 |
| 32 | 2413.3 | 360.4 | 547.0 | 547.8 | 1119.0 | 1119.4 |

## Key Observations

- **Throughput Saturation at c=32**: TPS plateaus at ~2400 (only 1.9% increase from c=16)
- **P99 TTFT Spike**: 33ms at c=16 → 548ms at c=32 (16.4x increase = queueing)
- **Linear Scaling up to c=16**: 11.1x throughput improvement before saturation
- **100% Success Rate**: All 300 requests completed (0 failures, 0 rate-limited)
- **GPU Compute Bound**: Saturation indicates GPU reached maximum batch processing capacity

## Test Configuration

- **Model**: Qwen/Qwen2.5-1.5B (1.5B parameters)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **Model Memory**: 2.91 GiB
- **Max Model Length**: 2048 tokens
- **Max Tokens Generated**: 100 per request
- **Requests per Level**: 50
- **vLLM Version**: 0.13.0
