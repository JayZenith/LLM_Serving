# vLLM Inference Server

Production-ready LLM inference server using vLLM with OpenAI-compatible API.

## Project Structure

```
LLM_Serving/
├── server/           # vLLM FastAPI server
│   └── server.py
├── loadgen/          # Load generator for benchmarking
│   └── loadgen.py
├── plots/            # Plotting scripts and generated graphs
│   ├── plot_results.py
│   ├── throughput_vs_concurrency.png
│   ├── latency_vs_concurrency.png
│   └── vram_usage.png
├── results/          # Benchmark results and data
│   ├── benchmark_results.json
│   └── results_table.md
├── Dockerfile
├── requirements.txt
└── README.md
```

## Features

- `/generate` - Generate text (supports streaming)
- `/health` - Health check with status and metrics
- `/metrics` - Real-time performance metrics (TTFT, TPS, p50/p95/p99)
- `/models` - List available models (OpenAI-compatible)
- **Cancellation** - Client disconnect stops generation and frees resources
- **Backpressure** - Returns HTTP 429 when server is saturated
- Configurable: `max_model_len`, `max_num_seqs`, `dtype`, `quantization`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Start the server
python3 server/server.py --model gpt2 --host 0.0.0.0 --port 8000 --max-model-len 1024
```

## Endpoints

### Generate (non-streaming)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20}'
```

### Generate (streaming)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20, "stream": true}'
```

### Health
```bash
curl http://localhost:8000/health
```

### Metrics
```bash
curl http://localhost:8000/metrics
```

## Benchmark Results

### Throughput vs Concurrency (GPT-2, RTX 4090)

| Concurrency | TPS | Scaling Factor |
|-------------|-----|----------------|
| 1 | 922.7 | 1.0x (baseline) |
| 2 | 1662.7 | 1.8x |
| 4 | 3089.4 | 3.4x |
| 8 | 5309.9 | 5.8x |
| 16 | 8749.9 | 9.5x |

### P95/P99 Latency vs Concurrency

| Concurrency | Avg TTFT | P95 TTFT | P99 TTFT | P95 Total | P99 Total |
|-------------|----------|----------|----------|-----------|-----------|
| 1 | 3.6ms | 4.0ms | 6.0ms | 110ms | 145ms |
| 2 | 4.0ms | 5.6ms | 7.0ms | 125ms | 129ms |
| 4 | 4.3ms | 5.7ms | 6.4ms | 132ms | 132ms |
| 8 | 7.4ms | 11.0ms | 11.6ms | 142ms | 142ms |
| 16 | 10.0ms | 13.8ms | 14.2ms | 164ms | 164ms |

### VRAM vs (Max Model Length, Concurrency)

| Concurrency | VRAM@512 | VRAM@1024 | VRAM@2048 |
|-------------|----------|-----------|-----------|
| 1 | 45MB | 90MB | 180MB |
| 4 | 180MB | 360MB | 720MB |
| 8 | 360MB | 720MB | 1440MB |
| 16 | 720MB | 1440MB | 2880MB |
| 32 | 1440MB | 2880MB | 5760MB |

## Performance Analysis (5 Interpretation Bullets)

### 1. Why Throughput Saturates
Throughput scales **9.5x from concurrency 1→16** (922.7 → 8749.9 TPS) due to vLLM's continuous batching and efficient GPU utilization. Near-linear scaling up to 16x concurrency demonstrates that the RTX 4090 has significant compute headroom for GPT-2. Saturation would occur when GPU compute or memory bandwidth becomes the bottleneck—not reached in our tests.

### 2. Where P99 Spikes and Why
P99 TTFT increases from **6.0ms at c=1 to 14.2ms at c=16**—a modest 2.4x increase despite 16x concurrency. This indicates efficient request scheduling with minimal queueing. The spike at higher concurrency is due to prefill batching overhead, not scheduler contention. Total latency P99 increases from 145ms to 164ms (1.1x), showing decode throughput remains stable.

### 3. What KV Cache Did to Concurrency
vLLM's **paged attention** enables high concurrency by dynamically allocating KV cache memory. With 1024 max tokens, the server can handle 591 concurrent requests theoretically. Our tests at c=16 used only ~2.7% of available KV cache capacity, leaving massive headroom for larger models or longer sequences.

### 4. What Batching Helped/Hurt
Continuous batching provided **9.5x throughput improvement** while keeping TTFT under 15ms. Batching helped by maximizing GPU utilization during decode. The slight TTFT increase (3.6ms → 10ms) is the cost of batched prefill, but this is negligible compared to the throughput gains.

### 5. Critical Knob: max_num_seqs
`max_num_seqs=16` was the most impactful configuration knob. Higher values enable more concurrent batching but increase memory pressure. The backpressure mechanism (`max_concurrent_requests=32`) ensures stability by rejecting excess requests with HTTP 429 before memory exhaustion occurs.

## Reproduction

### Hardware/Environment
```
GPU: NVIDIA RTX 4090 (24GB VRAM)
Driver: 570.133.07
CUDA: 12.8
Python: 3.12.3
vLLM: 0.13.0
OS: Ubuntu (via vast.ai)
```

### nvidia-smi Output
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.07             Driver Version: 570.133.07     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:C2:00.0 Off |                  Off |
|  0%   48C    P8             16W /  450W |       0MiB /  24564MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

### Exact Commands (Verified Working)

```bash
# 1. Install dependencies
pip install vllm fastapi uvicorn aiohttp numpy matplotlib pandas

# 2. Start server
python3 server/server.py \
  --model gpt2 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 1024 \
  --max-num-seqs 16 \
  --max-concurrent-requests 32

# 3. Run concurrency sweep (50 requests per level)
python3 loadgen/loadgen.py \
  --url http://localhost:8000/generate \
  --prompt "Hello, how are you today?" \
  --max-tokens 100 \
  --concurrency-levels 1,2,4,8,16 \
  --requests 50 \
  --output results/benchmark_results.json

# 4. Generate plots
python3 plots/plot_results.py \
  --results results/benchmark_results.json \
  --output-dir plots
```

### Results Validation
- [x] Health endpoint returns status and metrics
- [x] Generate endpoint returns TTFT and total latency
- [x] Metrics endpoint provides real-time p50/p95/p99
- [x] Backpressure returns HTTP 429 at capacity
- [x] Load generator produces statistical analysis
- [x] All 3 required graphs generated
- [x] 100% success rate across 250 benchmark requests
