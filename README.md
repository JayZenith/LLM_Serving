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

### Option 1: pip install (recommended for development)
```bash
pip install -r requirements.txt
```

### Option 2: Docker (recommended for production)
```bash
# Build the image
docker build -t vllm-server:latest .

# Run with GPU support (requires nvidia-container-toolkit)
docker run --gpus all -p 8000:8000 vllm-server:latest

# Run with custom model
docker run --gpus all -p 8000:8000 vllm-server:latest \
  python3 server/server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-1.5B \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --max-concurrent-requests 32

# Run benchmark inside container
docker run --gpus all -it vllm-server:latest \
  python3 loadgen/loadgen.py \
  --url http://localhost:8000/generate \
  --concurrency-levels 1,2,4,8,16 \
  --requests 50
```

**Docker Prerequisites:**
- NVIDIA GPU with CUDA 12.x
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- Docker 20.10+ with GPU support

**Note:** The Docker setup was not tested during development. I don't have a local GPU, so I used a vast.ai cloud instance for all benchmarking. Since vast.ai instances run inside containers, Docker-in-Docker is not supported there. If you have a native Docker setup with GPU access, you can test and use the Dockerfile yourself.

## Usage

```bash
# Start the server
python3 server/server.py --model Qwen/Qwen2.5-1.5B --host 0.0.0.0 --port 8000 --max-model-len 2048
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

### Throughput vs Concurrency (Qwen2.5-1.5B, RTX 4090)

| Concurrency | TPS | Scaling Factor |
|-------------|-----|----------------|
| 1 | 213.5 | 1.0x (baseline) |
| 2 | 402.7 | 1.9x |
| 4 | 699.0 | 3.3x |
| 8 | 1289.8 | 6.0x |
| 16 | 2367.8 | 11.1x |
| 32 | 2413.3 | 11.3x (saturated) |

### P95/P99 Latency vs Concurrency

| Concurrency | Avg TTFT | P95 TTFT | P99 TTFT | P95 Total | P99 Total |
|-------------|----------|----------|----------|-----------|-----------|
| 1 | 6.5ms | 6.8ms | 13.1ms | 469ms | 501ms |
| 2 | 9.0ms | 9.6ms | 10.6ms | 507ms | 513ms |
| 4 | 11.7ms | 19.7ms | 22.2ms | 561ms | 564ms |
| 8 | 14.1ms | 23.0ms | 23.6ms | 588ms | 588ms |
| 16 | 21.7ms | 32.7ms | 33.4ms | 561ms | 561ms |
| 32 | 360.4ms | 547.0ms | 547.8ms | 1119ms | 1119ms |

### VRAM Usage

| Component | Memory |
|-----------|--------|
| Model weights (Qwen2.5-1.5B, bf16) | 2.91 GiB |
| KV cache capacity | 665,872 tokens |
| Max concurrent requests (2048 tokens each) | 325x |

## Performance Analysis (5 Interpretation Bullets)

### 1. Why Throughput Saturates
Throughput scales **11.1x from concurrency 1→16** (213.5 → 2367.8 TPS) due to vLLM's continuous batching. At **c=32, throughput plateaus** at 2413 TPS (only 1.9% gain)—this is the GPU compute saturation point where the RTX 4090 cannot process batches any faster. The 1.5B model is 12x larger than GPT-2, resulting in proportionally lower TPS but more realistic production behavior.

### 2. Where P99 Spikes and Why
P99 TTFT spikes dramatically at c=32: **33ms → 548ms (16.4x increase)**. This is the classic queueing effect—requests wait in the scheduler queue because GPU compute is fully utilized. At c=16, TTFT remains stable (33ms p99) because batch processing keeps up with incoming requests. The spike at c=32 indicates the optimal operating point is at or below c=16 for this model.

### 3. What KV Cache Did to Concurrency
vLLM's **paged attention** provides 665,872 tokens of KV cache, enabling 325 concurrent requests at 2048 tokens each. Our tests used only ~10% of KV capacity (32 × 2048 = 65,536 tokens), proving that **GPU compute, not memory, is the bottleneck** for Qwen2.5-1.5B. Larger models (7B+) would see memory become the limiting factor first.

### 4. What Batching Helped/Hurt
Continuous batching provided **11.1x throughput improvement** (c=1 to c=16) while keeping TTFT under 35ms. However, **batching hurts at saturation**: at c=32, the decode phase cannot keep up, causing requests to queue. Total latency doubles (561ms → 1119ms) as requests spend more time waiting than processing.

### 5. Critical Knob: max_num_seqs
`max_num_seqs=16` is the optimal setting for Qwen2.5-1.5B on RTX 4090. Setting it to 32 causes saturation and latency spikes. The **backpressure mechanism** (`max_concurrent_requests=32`) is essential—without it, unbounded queueing would cause OOM or unbounded latency growth. The 429 rejection rate at c=32 was 0% because we stayed within limits, but the latency spike shows we're at the edge.

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

# 2. Start server (Qwen2.5-1.5B)
python3 server/server.py \
  --model Qwen/Qwen2.5-1.5B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --max-concurrent-requests 32

# 3. Run concurrency sweep (50 requests per level)
python3 loadgen/loadgen.py \
  --url http://localhost:8000/generate \
  --prompt "Explain the theory of relativity in simple terms." \
  --max-tokens 100 \
  --concurrency-levels 1,2,4,8,16,32 \
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
- [x] 100% success rate across 300 benchmark requests
- [x] Saturation point identified at concurrency=32
