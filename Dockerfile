# Dockerfile for vLLM Inference Server
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY server/ ./server/
COPY loadgen/ ./loadgen/
COPY plots/ ./plots/

# Create results directory
RUN mkdir -p results

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python3", "server/server.py", "--host", "0.0.0.0", "--port", "8000", "--model", "Qwen/Qwen2.5-1.5B", "--max-model-len", "2048", "--max-num-seqs", "16", "--max-concurrent-requests", "32"]
