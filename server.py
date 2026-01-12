import os
import argparse
import uuid
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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


llm_engine: Optional[AsyncLLMEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_engine
    
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
    yield
    del llm_engine


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return {"status": "healthy", "model": os.getenv("MODEL_NAME", "gpt2")}


@app.post("/generate")
async def generate(request: GenerateRequest):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    
    request_id = str(uuid.uuid4())
    
    if request.stream:
        return StreamingResponse(
            stream_tokens(request.prompt, sampling_params, request_id),
            media_type="text/plain"
        )
    
    results_generator = llm_engine.generate(request.prompt, sampling_params, request_id)
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    
    return GenerateResponse(
        text=final_output.outputs[0].text,
        prompt_tokens=final_output.prompt_token_ids.__len__(),
        completion_tokens=len(final_output.outputs[0].token_ids),
    )


async def stream_tokens(prompt: str, sampling_params: SamplingParams, request_id: str):
    results_generator = llm_engine.generate(prompt, sampling_params, request_id)
    async for request_output in results_generator:
        if request_output.outputs:
            yield request_output.outputs[0].text


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
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--quantization", type=str, default=None)
    
    args = parser.parse_args()
    
    os.environ["MODEL_NAME"] = args.model
    os.environ["MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ["MAX_NUM_SEQS"] = str(args.max_num_seqs)
    os.environ["DTYPE"] = args.dtype
    if args.quantization:
        os.environ["QUANTIZATION"] = args.quantization
    
    uvicorn.run(app, host=args.host, port=args.port)
