# vLLM FastAPI Server

OpenAI-compatible LLM inference server using vLLM.

## Features

- /generate - Generate text (supports streaming)
- /health - Health check endpoint
- /models - List available models
- Configurable: max_model_len, max_num_seqs, dtype, quantization

## Installation

pip install -r requirements.txt

## Usage

python3 server.py --model gpt2 --host 0.0.0.0 --port 8000 --max-model-len 1024

## Endpoints

### Generate (non-streaming)

curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d "{\"prompt\": \"Hello\", \"max_tokens\": 20}"

### Generate (streaming)

curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d "{\"prompt\": \"Hello\", \"max_tokens\": 20, \"stream\": true}"

### Health

curl http://localhost:8000/health

### Models

curl http://localhost:8000/models
