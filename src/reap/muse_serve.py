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
import secrets
import ipaddress
import socket

from fastapi import FastAPI, HTTPException, Request as HTTPRequest
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
    runtime,
    tokenizer,
    *,
    eos,
    model_name="Muse-Spark-Contribution128-W4A16-W8Attn",
    resident=None,
    resident_token=None,
):
    app = FastAPI(title="Muse Ascend bring-up")
    lock = threading.Lock()
    if resident is not None and (not resident_token or len(resident_token) < 32):
        raise ValueError("resident controls require a secret of at least 32 characters")

    @app.get("/health")
    def health():
        return {
            "status": "blocked" if resident and resident.failed else "ready",
            "pid": os.getpid(),
            "model": model_name,
            "max_context": runtime.max_context,
            "execution": "sequential layer placement; greedy; single request",
            "optimized": getattr(runtime, "optimized", False),
            "bounded_attention": getattr(runtime, "bounded_attention", False),
            "prefill_chunk": getattr(runtime, "prefill_chunk", None),
            "deterministic": getattr(runtime, "deterministic", False),
            "resident": resident.status() if resident else None,
        }

    if resident is not None:

        def authorize(request):
            try:
                local = ipaddress.ip_address(request.client.host).is_loopback
            except (ValueError, AttributeError):
                local = False
            if not local or not secrets.compare_digest(
                request.headers.get("authorization", ""), "Bearer " + resident_token
            ):
                raise HTTPException(
                    403, "resident controls require local authentication"
                )

        @app.get("/resident/status")
        def resident_status(request: HTTPRequest):
            authorize(request)
            return resident.status()

        @app.post("/resident/{action}")
        def resident_action(action: str, body: dict, request: HTTPRequest):
            authorize(request)
            if action not in ("load-native", "switch"):
                raise HTTPException(404, "unknown resident action")
            if not lock.acquire(blocking=False):
                raise HTTPException(503, "runtime is handling another request")
            try:
                if action == "load-native":
                    return resident.load_native(body)
                if set(body) - {"native", "source"}:
                    raise ValueError("unknown switch fields")
                return resident.switch(**body)
            except Exception as exc:
                raise HTTPException(400, str(exc)) from exc
            finally:
                runtime.reset()
                lock.release()

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
            if resident and resident.failed:
                raise HTTPException(503, "resident controls blocked; restart required")
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
    parser.add_argument("--additional-port", type=int, action="append", default=[])
    parser.add_argument("--served-model-name")
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--bounded-attention", action="store_true")
    parser.add_argument("--packed-cache", type=Path)
    parser.add_argument("--resident-token-file", type=Path)
    args = parser.parse_args()
    if args.bounded_attention and (not args.optimized or args.max_context < 1):
        parser.error("--bounded-attention requires --optimized and positive context")
    if args.resident_token_file and not args.optimized:
        parser.error("resident controls require --optimized")
    token = None
    if args.resident_token_file:
        token = args.resident_token_file.read_text().strip()
        if args.resident_token_file.stat().st_mode & 0o077 or len(token) < 32:
            parser.error(
                "resident token file must be private and contain a 32+ character secret"
            )
    torch.use_deterministic_algorithms(args.deterministic)
    torch.set_num_threads(8 if args.optimized else 2)
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
        optimized=args.optimized,
        bounded_attention=args.bounded_attention,
        packed_cache=args.packed_cache,
    )
    torch.set_num_threads(2)
    runtime.deterministic = args.deterministic
    resident = None
    if args.resident_token_file:
        from reap.muse_resident import Resident

        resident = Resident(runtime)
    config = uvicorn.Config(
        create_app(
            runtime,
            tokenizer,
            eos=eos,
            model_name=args.served_model_name or args.model.name,
            resident=resident,
            resident_token=token,
        ),
        host=args.host,
        port=args.port,
        workers=1,
        proxy_headers=False,
    )
    # Both listeners share one process, model allocation and inference lock.
    # Keep the existing endpoint while introducing a new primary port.
    sockets = []
    try:
        for port in dict.fromkeys([args.port, *args.additional_port]):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(listener)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.host, port))
            listener.listen(2048)
        uvicorn.Server(config).run(sockets=sockets)
    finally:
        for listener in sockets:
            listener.close()


if __name__ == "__main__":
    main()
