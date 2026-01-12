import os
import argparse
import uuid
import time
import asyncio
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: Optional[float] = None
    total_time_ms: Optional[float] = None


class MetricsStore:
    def __init__(self):
        self.request_count = 0
        self.ttft_latencies: List[float] = []
        self.total_latencies: List[float] = []
        self.active_requests: int = 0
        self.queue_depth: int = 0
        self.total_tokens_generated = 0
        self.error_count = 0
        self.request_metrics: Dict[str, Dict[str, Any]] = {}

    def record_ttft(self, request_id: str, ttft_ms: float):
        self.ttft_latencies.append(ttft_ms)
        if request_id in self.request_metrics:
            self.request_metrics[request_id]['ttft_ms'] = ttft_ms

    def record_completion(self, request_id: str, total_time_ms: float, tokens_generated: int):
        self.total_latencies.append(total_time_ms)
        self.total_tokens_generated += tokens_generated
        if request_id in self.request_metrics:
            self.request_metrics[request_id]['total_time_ms'] = total_time_ms
            self.request_metrics[request_id]['tokens'] = tokens_generated

    def get_percentile(self, values: List[float], p: float) -> Optional[float]:
        if not values:
            return None
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def get_metrics(self) -> Dict[str, Any]:
        avg_ttft = sum(self.ttft_latencies) / len(self.ttft_latencies) if self.ttft_latencies else 0
        avg_total = sum(self.total_latencies) / len(self.total_latencies) if self.total_latencies else 0
        avg_tps = self.total_tokens_generated / sum(self.total_latencies) * 1000 if self.total_latencies else 0

        return {
            "request_count": self.request_count,
            "active_requests": self.active_requests,
            "queue_depth": self.queue_depth,
            "total_tokens_generated": self.total_tokens_generated,
            "error_count": self.error_count,
            "ttft_ms": {
                "avg": round(avg_ttft, 2),
                "p50": round(self.get_percentile(self.ttft_latencies, 50) or 0, 2),
                "p95": round(self.get_percentile(self.ttft_latencies, 95) or 0, 2),
                "p99": round(self.get_percentile(self.ttft_latencies, 99) or 0, 2),
            },
            "total_time_ms": {
                "avg": round(avg_total, 2),
                "p50": round(self.get_percentile(self.total_latencies, 50) or 0, 2),
                "p95": round(self.get_percentile(self.total_latencies, 95) or 0, 2),
                "p99": round(self.get_percentile(self.total_latencies, 99) or 0, 2),
            },
            "tokens_per_second": round(avg_tps, 2),
        }


llm_engine: Optional[AsyncLLMEngine] = None
metrics = MetricsStore()
max_concurrent_requests: int = 16
request_queue: asyncio.Queue = asyncio.Queue()
active_requests: Dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_engine, max_concurrent_requests

    engine_args = AsyncEngineArgs(
        model=os.getenv("MODEL_NAME", "gpt2"),
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "1024")),
        max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "8")),
        dtype=os.getenv("DTYPE", "auto"),
        quantization=os.getenv("QUANTIZATION", None),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_log_stats=True,
    )

    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "16"))

    worker_task = asyncio.create_task(process_request_queue())
    yield

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    del llm_engine


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return {
        "status": "healthy",
        "model": os.getenv("MODEL_NAME", "gpt2"),
        "active_requests": metrics.active_requests,
        "queue_depth": metrics.queue_depth,
    }


@app.get("/metrics")
async def get_metrics():
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return metrics.get_metrics()


@app.post("/generate")
async def generate(request: GenerateRequest, http_request: Request):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    metrics.request_count += 1

    if metrics.active_requests >= max_concurrent_requests:
        metrics.queue_depth += 1
        metrics.error_count += 1
        metrics.queue_depth -= 1
        raise HTTPException(status_code=429, detail="Server is busy, try again later")

    request_id = str(uuid.uuid4())
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    metrics.request_metrics[request_id] = {
        "start_time": time.time(),
        "prompt_tokens": 0,
    }

    metrics.active_requests += 1

    try:
        if request.stream:
            return StreamingResponse(
                stream_tokens_with_cancel(
                    request.prompt,
                    sampling_params,
                    request_id,
                    http_request
                ),
                media_type="text/plain"
            )

        results_generator = llm_engine.generate(request.prompt, sampling_params, request_id)
        first_output = True
        final_output = None
        tokens_generated = 0

        async for request_output in results_generator:
            if first_output:
                ttft_ms = (time.time() - metrics.request_metrics[request_id]['start_time']) * 1000
                metrics.record_ttft(request_id, ttft_ms)
                first_output = False

            final_output = request_output
            if request_output.outputs:
                tokens_generated = len(request_output.outputs[0].token_ids)

        total_time_ms = (time.time() - metrics.request_metrics[request_id]['start_time']) * 1000
        metrics.record_completion(request_id, total_time_ms, tokens_generated)

        return GenerateResponse(
            text=final_output.outputs[0].text if final_output and final_output.outputs else "",
            prompt_tokens=final_output.prompt_token_ids.__len__() if final_output else 0,
            completion_tokens=len(final_output.outputs[0].token_ids) if final_output and final_output.outputs else 0,
            ttft_ms=round(metrics.request_metrics.get(request_id, {}).get('ttft_ms', 0), 2),
            total_time_ms=round(total_time_ms, 2),
        )

    except asyncio.CancelledError:
        metrics.active_requests -= 1
        if request_id in metrics.request_metrics:
            del metrics.request_metrics[request_id]
        metrics.error_count += 1
        raise HTTPException(status_code=499, detail="Client cancelled request")
    except Exception as e:
        metrics.active_requests -= 1
        metrics.error_count += 1
        if request_id in metrics.request_metrics:
            del metrics.request_metrics[request_id]
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        metrics.active_requests -= 1
        if request_id in metrics.request_metrics:
            del metrics.request_metrics[request_id]


async def stream_tokens_with_cancel(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    http_request: Request
):
    try:
        results_generator = llm_engine.generate(prompt, sampling_params, request_id)
        first_output = True
        tokens_generated = 0

        async for request_output in results_generator:
            if await http_request.is_disconnected():
                raise asyncio.CancelledError("Client disconnected")

            if first_output:
                ttft_ms = (time.time() - metrics.request_metrics[request_id]['start_time']) * 1000
                metrics.record_ttft(request_id, ttft_ms)
                first_output = False

            if request_output.outputs:
                text = request_output.outputs[0].text
                tokens_generated = len(request_output.outputs[0].token_ids)
                yield text

        total_time_ms = (time.time() - metrics.request_metrics[request_id]['start_time']) * 1000
        metrics.record_completion(request_id, total_time_ms, tokens_generated)

    except asyncio.CancelledError:
        if request_id in metrics.request_metrics:
            del metrics.request_metrics[request_id]
        raise


async def process_request_queue():
    while True:
        try:
            await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            break


@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": os.getenv("MODEL_NAME", "gpt2"),
                "object": "model",
                "owned_by": "vllm",
            }
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-concurrent-requests", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--quantization", type=str, default=None)

    args = parser.parse_args()

    os.environ["MODEL_NAME"] = args.model
    os.environ["MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ["MAX_NUM_SEQS"] = str(args.max_num_seqs)
    os.environ["MAX_CONCURRENT_REQUESTS"] = str(args.max_concurrent_requests)
    os.environ["DTYPE"] = args.dtype
    if args.quantization:
        os.environ["QUANTIZATION"] = args.quantization

    uvicorn.run(app, host=args.host, port=args.port)
