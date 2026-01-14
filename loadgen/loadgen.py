#!/usr/bin/env python3
"""
Load generator for vLLM benchmarking.
Performs concurrency and prompt-length sweeps to measure throughput and latency.
"""
import asyncio
import argparse
import json
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
import aiohttp
import numpy as np


@dataclass
class BenchmarkResult:
    concurrency: int
    prompt_tokens: int
    max_tokens: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    tokens_per_second: float
    total_time_s: float


async def generate_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
    timeout: float = 300.0
) -> Dict[str, Any]:
    start_time = time.time()

    try:
        async with session.post(
            url,
            json={"prompt": prompt, "max_tokens": max_tokens, "stream": False},
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status == 429:
                return {
                    "request_id": request_id,
                    "status": 429,
                    "error": "Rate limited"
                }

            data = await response.json()
            end_time = time.time()

            return {
                "request_id": request_id,
                "status": 200,
                "latency_ms": (end_time - start_time) * 1000,
                "ttft_ms": data.get("ttft_ms", None),
                "completion_tokens": data.get("completion_tokens", 0),
                "prompt_tokens": data.get("prompt_tokens", 0),
            }
    except Exception as e:
        end_time = time.time()
        return {
            "request_id": request_id,
            "status": 500,
            "error": str(e),
            "latency_ms": (end_time - start_time) * 1000,
        }


async def run_benchmark(
    url: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    num_requests: int = 100
) -> BenchmarkResult:
    """
    Run benchmark with specified concurrency level.
    """
    print(f"Running benchmark: concurrency={concurrency}, num_requests={num_requests}")

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(req_id: int) -> Dict[str, Any]:
            async with semaphore:
                result = await generate_request(session, url, prompt, max_tokens, req_id)
                return result

        start_time = time.time()
        results = await asyncio.gather(
            *[bounded_request(i) for i in range(num_requests)],
            return_exceptions=True
        )
        total_time = time.time() - start_time

    successful = [r for r in results if not isinstance(r, Exception) and isinstance(r, dict) and r.get("status") == 200]
    failed = [r for r in results if isinstance(r, Exception) or (isinstance(r, dict) and r.get("status") != 200)]

    ttfts = [r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]
    latencies = [r["latency_ms"] for r in successful]
    total_tokens = sum(r["completion_tokens"] for r in successful)

    return BenchmarkResult(
        concurrency=concurrency,
        prompt_tokens=len(prompt.split()),
        max_tokens=max_tokens,
        total_requests=num_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        avg_ttft_ms=statistics.mean(ttfts) if ttfts else 0,
        p50_ttft_ms=statistics.median(ttfts) if ttfts else 0,
        p95_ttft_ms=float(np.percentile(ttfts, 95)) if ttfts else 0.0,
        p99_ttft_ms=float(np.percentile(ttfts, 99)) if ttfts else 0.0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        p50_latency_ms=statistics.median(latencies) if latencies else 0,
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        p99_latency_ms=float(np.percentile(latencies, 99)) if latencies else 0.0,
        tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
        total_time_s=total_time,
    )


async def run_concurrency_sweep(
    url: str,
    prompt: str,
    max_tokens: int,
    concurrency_levels: List[int],
    num_requests_per_level: int = 100
) -> List[BenchmarkResult]:
    """
    Run benchmark across multiple concurrency levels.
    """
    results = []

    for concurrency in concurrency_levels:
        result = await run_benchmark(url, prompt, max_tokens, concurrency, num_requests_per_level)
        results.append(result)
        print(f"  TPS: {result.tokens_per_second:.2f}, p99 TTFT: {result.p99_ttft_ms:.2f}ms")

    return results


async def run_prompt_length_sweep(
    url: str,
    max_tokens: int,
    concurrency: int,
    prompt_lengths: List[int],
    num_requests_per_length: int = 50
) -> List[BenchmarkResult]:
    """
    Run benchmark across different prompt lengths.
    """
    results = []

    for prompt_len in prompt_lengths:
        prompt = " ".join(["hello"] * prompt_len)
        result = await run_benchmark(url, prompt, max_tokens, concurrency, num_requests_per_length)
        results.append(result)
        print(f"  Prompt tokens: {result.prompt_tokens}, TPS: {result.tokens_per_second:.2f}")

    return results


def save_results(results: List[BenchmarkResult], filename: str):
    """
    Save benchmark results to JSON file.
    """
    data = [
        {
            "concurrency": r.concurrency,
            "prompt_tokens": r.prompt_tokens,
            "max_tokens": r.max_tokens,
            "total_requests": r.total_requests,
            "successful_requests": r.successful_requests,
            "failed_requests": r.failed_requests,
            "avg_ttft_ms": r.avg_ttft_ms,
            "p50_ttft_ms": r.p50_ttft_ms,
            "p95_ttft_ms": r.p95_ttft_ms,
            "p99_ttft_ms": r.p99_ttft_ms,
            "avg_latency_ms": r.avg_latency_ms,
            "p50_latency_ms": r.p50_latency_ms,
            "p95_latency_ms": r.p95_latency_ms,
            "p99_latency_ms": r.p99_latency_ms,
            "tokens_per_second": r.tokens_per_second,
            "total_time_s": r.total_time_s,
        }
        for r in results
    ]

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {filename}")


def print_summary(results: List[BenchmarkResult]):
    """
    Print summary table of benchmark results.
    """
    print("\n" + "=" * 100)
    print(f"{'Concurrency':<12} {'TPS':<12} {'Avg TTFT':<12} {'p95 TTFT':<12} {'p99 TTFT':<12} {'p95 Lat':<12} {'p99 Lat':<12}")
    print("=" * 100)

    for r in results:
        print(
            f"{r.concurrency:<12} "
            f"{r.tokens_per_second:<12.2f} "
            f"{r.avg_ttft_ms:<12.2f} "
            f"{r.p95_ttft_ms:<12.2f} "
            f"{r.p99_ttft_ms:<12.2f} "
            f"{r.p95_latency_ms:<12.2f} "
            f"{r.p99_latency_ms:<12.2f}"
        )


async def main():
    parser = argparse.ArgumentParser(description="Load generator for vLLM benchmarking")
    parser.add_argument("--url", type=str, default="http://localhost:8000/generate", help="Server URL")
    parser.add_argument("--prompt", type=str, default="Hello, how are you today?", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--concurrency-levels", type=str, default="1,2,4,8,16,32", help="Comma-separated concurrency levels")
    parser.add_argument("--prompt-lengths", type=str, default="10,50,100,200,500", help="Comma-separated prompt token counts")
    parser.add_argument("--requests", type=int, default=100, help="Requests per concurrency level")
    parser.add_argument("--output", type=str, default="results/benchmark_results.json", help="Output JSON file")
    parser.add_argument("--mode", type=str, default="concurrency", choices=["concurrency", "prompt"], help="Sweep mode")

    args = parser.parse_args()

    concurrency_levels = [int(x) for x in args.concurrency_levels.split(",")]
    prompt_lengths = [int(x) for x in args.prompt_lengths.split(",")]

    print(f"Starting benchmark against {args.url}")
    print(f"Mode: {args.mode}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")

    if args.mode == "concurrency":
        results = await run_concurrency_sweep(
            args.url, args.prompt, args.max_tokens, concurrency_levels, args.requests
        )
    else:
        results = await run_prompt_length_sweep(
            args.url, args.max_tokens, concurrency_levels[0], prompt_lengths, args.requests
        )

    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
