#!/usr/bin/env python3
"""
Generate benchmark plots from results data.
Requires matplotlib and pandas.
"""
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any


def load_results(filename: str) -> List[Dict[str, Any]]:
    """Load benchmark results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_throughput_vs_concurrency(results: List[Dict[str, Any]], output_file: str):
    """Plot throughput vs concurrency."""
    df = pd.DataFrame(results)
    df = df[df['prompt_tokens'] == 8]  # Fixed prompt length for comparison

    plt.figure(figsize=(10, 6))
    plt.plot(df['concurrency'], df['tokens_per_second'], marker='o', linewidth=2)
    plt.xlabel('Concurrency Level')
    plt.ylabel('Tokens/Second')
    plt.title('Throughput vs Concurrency (8-token prompts, 100-token generations)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_latency_vs_concurrency(results: List[Dict[str, Any]], output_file: str):
    """Plot p95/p99 latency vs concurrency."""
    df = pd.DataFrame(results)
    df = df[df['prompt_tokens'] == 8]  # Fixed prompt length

    plt.figure(figsize=(10, 6))
    plt.plot(df['concurrency'], df['p95_ttft_ms'], marker='s', label='P95 TTFT', linewidth=2)
    plt.plot(df['concurrency'], df['p99_ttft_ms'], marker='^', label='P99 TTFT', linewidth=2)
    plt.xlabel('Concurrency Level')
    plt.ylabel('Time-to-First-Token (ms)')
    plt.title('TTFT Latency vs Concurrency (8-token prompts)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_vram_usage(results: List[Dict[str, Any]], output_file: str):
    """Plot estimated VRAM usage vs concurrency and model length."""
    # Estimated VRAM calculation (simplified)
    # KV cache ≈ num_layers * seq_len * hidden_size * 2 * num_requests * dtype_bytes
    # For GPT-2: 12 layers, 768 hidden, bfloat16 (2 bytes)
    def estimate_vram_mb(concurrency: int, max_model_len: int) -> float:
        kv_cache_per_request = 12 * max_model_len * 768 * 2 * 2  # 2 for KV pairs
        return (concurrency * kv_cache_per_request) / (1024 * 1024)

    concurrency_levels = [1, 2, 4, 8, 16, 32]
    model_lengths = [512, 1024, 2048]

    plt.figure(figsize=(12, 8))

    for max_len in model_lengths:
        vram_usage = [estimate_vram_mb(c, max_len) for c in concurrency_levels]
        plt.plot(concurrency_levels, vram_usage, marker='o', label=f'Max length: {max_len}', linewidth=2)

    plt.axhline(y=24*1024, color='red', linestyle='--', alpha=0.7, label='RTX 4090 VRAM (24GB)')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Estimated VRAM Usage (MB)')
    plt.title('VRAM Usage vs Concurrency and Max Model Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def generate_table(results: List[Dict[str, Any]], filename: str):
    """Generate markdown table of results."""
    df = pd.DataFrame(results)
    df = df[df['prompt_tokens'] == 8]  # Fixed prompt length

    table = "| Concurrency | TPS | Avg TTFT | P95 TTFT | P99 TTFT | P95 Total | P99 Total |\n"
    table += "|------------|-----|----------|----------|----------|-----------|-----------|\n"

    for _, row in df.iterrows():
        table += f"| {row['concurrency']:>10} | {row['tokens_per_second']:>3.1f} | {row['avg_ttft_ms']:>8.1f} | {row['p95_ttft_ms']:>8.1f} | {row['p99_ttft_ms']:>8.1f} | {row['p95_latency_ms']:>9.1f} | {row['p99_latency_ms']:>9.1f} |\n"

    with open(filename, 'w') as f:
        f.write(table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument("--results", type=str, default="benchmark_results.json", help="Results JSON file")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory")

    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        results = load_results(args.results)

        plot_throughput_vs_concurrency(results, f"{args.output_dir}/throughput_vs_concurrency.png")
        plot_latency_vs_concurrency(results, f"{args.output_dir}/latency_vs_concurrency.png")
        plot_vram_usage(results, f"{args.output_dir}/vram_usage.png")
        generate_table(results, f"{args.output_dir}/results_table.md")

        print(f"Plots generated in {args.output_dir}/")

    except FileNotFoundError:
        print(f"Results file {args.results} not found. Run loadgen.py first.")
        # Generate sample data for demonstration
        sample_results = [
            {"concurrency": 1, "prompt_tokens": 8, "tokens_per_second": 25.3, "avg_ttft_ms": 180.2, "p95_ttft_ms": 220.5, "p99_ttft_ms": 280.1, "p95_latency_ms": 1100.0, "p99_latency_ms": 1400.0},
            {"concurrency": 2, "prompt_tokens": 8, "tokens_per_second": 48.7, "avg_ttft_ms": 195.8, "p95_ttft_ms": 240.2, "p99_ttft_ms": 310.8, "p95_latency_ms": 1150.0, "p99_latency_ms": 1500.0},
            {"concurrency": 4, "prompt_tokens": 8, "tokens_per_second": 92.1, "avg_ttft_ms": 210.4, "p95_ttft_ms": 265.7, "p99_ttft_ms": 340.3, "p95_latency_ms": 1220.0, "p99_latency_ms": 1650.0},
            {"concurrency": 8, "prompt_tokens": 8, "tokens_per_second": 165.8, "avg_ttft_ms": 235.6, "p95_ttft_ms": 295.2, "p99_ttft_ms": 380.9, "p95_latency_ms": 1350.0, "p99_latency_ms": 1850.0},
            {"concurrency": 16, "prompt_tokens": 8, "tokens_per_second": 245.3, "avg_ttft_ms": 280.1, "p95_ttft_ms": 350.8, "p99_ttft_ms": 480.4, "p95_latency_ms": 1600.0, "p99_latency_ms": 2200.0},
            {"concurrency": 32, "prompt_tokens": 8, "tokens_per_second": 245.3, "avg_ttft_ms": 450.7, "p95_ttft_ms": 680.3, "p99_ttft_ms": 950.2, "p95_latency_ms": 2800.0, "p99_latency_ms": 4200.0},
        ]

        plot_throughput_vs_concurrency(sample_results, f"{args.output_dir}/throughput_vs_concurrency.png")
        plot_latency_vs_concurrency(sample_results, f"{args.output_dir}/latency_vs_concurrency.png")
        plot_vram_usage(sample_results, f"{args.output_dir}/vram_usage.png")
        generate_table(sample_results, f"{args.output_dir}/results_table.md")

        print(f"Sample plots generated in {args.output_dir}/ (run actual benchmarks to replace)")
