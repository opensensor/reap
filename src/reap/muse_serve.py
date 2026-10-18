"""Minimal single-request, greedy OpenAI-compatible Muse text endpoint.

This initial server implements non-streaming completions and text chat. It uses
the exported tokenizer/chat template and does not claim tool or vision support.
"""

import argparse
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    prompt: str | None = None
    messages: list[Message] | None = None
    max_tokens: int = Field(default=64, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = 0
    top_p: float = 1
    n: int = 1
    stream: bool = False
    reasoning_effort: str = "low"


def parse_response(text):
    """Separate Muse's self/user channels, preserving a truncated body."""
    message = "<|start|>assistant" + text
    matches = re.findall(
        r"<\|start\|>assistant(?:\s+to=(self|user|commentary))?<\|message\|>(.*?)(?=<\|eom\|>|<\|eot\|>|<\|end_of_text\|>|$)",
        message,
        re.S,
    )
    content = "".join(body for channel, body in matches if channel != "self")
    reasoning = "".join(body for channel, body in matches if channel == "self")
    return {
        "role": "assistant",
        "content": content,
        **({"reasoning_content": reasoning} if reasoning else {}),
    }


def create_app(
    runtime, tokenizer, *, eos, model_name="Muse-Spark-Contribution128-W4A16-W8Attn"
):
    app = FastAPI(title="Muse Ascend bring-up")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "pid": os.getpid(),
            "model": model_name,
            "max_context": runtime.max_context,
            "execution": "sequential layer placement; greedy; single request",
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {"id": model_name, "object": "model", "created": 0, "owned_by": "local"}
            ],
        }

    def complete(request, chat):
        if request.model not in (None, model_name):
            raise HTTPException(404, "unknown model")
        if (
            request.stream
            or request.n != 1
            or request.temperature != 0
            or request.top_p != 1
        ):
            raise HTTPException(
                400,
                "initial runtime supports non-streaming greedy generation with n=1, temperature=0, top_p=1",
            )
        if request.reasoning_effort not in ("low", "medium", "high", "xhigh"):
            raise HTTPException(400, "invalid reasoning_effort")
        if chat:
            if not request.messages or request.prompt is not None:
                raise HTTPException(400, "chat requires messages and no prompt")
            if any(
                m.role not in ("system", "developer", "user", "assistant")
                for m in request.messages
            ):
                raise HTTPException(400, "unsupported message role")
            try:
                ids = tokenizer.apply_chat_template(
                    [m.model_dump() for m in request.messages],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                    reasoning_effort=request.reasoning_effort,
                )
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, str(exc)) from exc
        else:
            if not request.prompt or request.messages is not None:
                raise HTTPException(400, "completion requires a prompt and no messages")
            ids = tokenizer.encode(request.prompt, add_special_tokens=True)
        limit = request.max_completion_tokens or request.max_tokens
        if not ids or len(ids) + limit > runtime.max_context:
            raise HTTPException(
                400, "prompt plus output allowance exceeds configured context"
            )
        if not lock.acquire(blocking=False):
            raise HTTPException(503, "runtime is handling another request")
        try:
            tokens = runtime.generate(ids, limit, eos=eos)
        finally:
            # Caches contain request-specific state; release after success or failure.
            runtime.reset()
            lock.release()
        finish = "stop" if tokens and tokens[-1] in eos else "length"
        choice = {"index": 0, "finish_reason": finish}
        if chat:
            choice["message"] = parse_response(
                tokenizer.decode(tokens, skip_special_tokens=False)
            )
        else:
            choice["text"] = tokenizer.decode(tokens, skip_special_tokens=True)
            choice["logprobs"] = None
        return {
            "id": ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex,
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": len(tokens),
                "total_tokens": len(ids) + len(tokens),
            },
        }

    @app.post("/v1/completions")
    def completions(request: Request):
        return complete(request, False)

    @app.post("/v1/chat/completions")
    def chat_completions(request: Request):
        return complete(request, True)

    return app


def main():
    import torch
    import uvicorn
    from transformers import AutoTokenizer
    from reap.muse_runtime import MuseRuntime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["npu:0", "npu:1", "npu:2", "npu:3", "npu:4", "npu:5"],
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--reserve-gib", type=float, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--served-model-name")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if any(d.startswith("npu:") for d in args.devices):
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
        torch.npu.matmul.allow_hf32 = False
        if args.library is None:
            parser.error("NPU execution requires --library")
        torch.ops.load_library(str(args.library))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generation = json.loads((args.model / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos = [eos] if isinstance(eos, int) else eos
    runtime = MuseRuntime(
        args.model,
        args.devices,
        max_context=args.max_context,
        prefill_chunk=args.prefill_chunk,
        reserve_gib=args.reserve_gib,
    )
    uvicorn.run(
        create_app(
            runtime,
            tokenizer,
            eos=eos,
            model_name=args.served_model_name or args.model.name,
        ),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
