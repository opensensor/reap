"""Minimal single-request, greedy OpenAI-compatible Muse text endpoint.

This server implements streaming and non-streaming completions and text chat. It uses
the exported tokenizer/chat template, including function tools. Vision is unsupported.
"""

import argparse
import asyncio
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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from reap.muse_tools import Tool, ToolCall, active_tools, extract_calls, strip_calls


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def text_parts(cls, value):
        if isinstance(value, list):
            if any(
                not isinstance(p, dict)
                or p.get("type") != "text"
                or not isinstance(p.get("text"), str)
                for p in value
            ):
                raise ValueError("only text content is supported")
            return "".join(p["text"] for p in value)
        return value


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = False


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
    stream_options: StreamOptions | None = None
    reasoning_effort: str = "low"
    tools: list[Tool] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool = True


def parse_response(text, tools=None, *, final=True):
    """Separate Muse channels and translate completed native function calls."""
    message = "<|start|>assistant" + text
    matches = re.findall(
        r"<\|start\|>assistant(?:\s+to=([^\s<]+))?<\|message\|>(.*?)(?=<\|eom\|>|<\|eot\|>|<\|end_of_text\|>|$)",
        message,
        re.S,
    )
    content, reasoning, calls = [], [], []
    for channel, body in matches:
        if channel == "self":
            reasoning.append(body)
        else:
            if tools and final:
                calls.extend(extract_calls(body, tools))
            if channel in ("", "user", "commentary"):
                content.append(strip_calls(body) if tools else body)
    return {
        "role": "assistant",
        "content": "".join(content),
        **({"reasoning_content": "".join(reasoning)} if reasoning else {}),
        **({"tool_calls": calls} if calls else {}),
    }


def visible_stream_text(tokenizer, tokens, chat, final=False, tools=None):
    text = tokenizer.decode(
        tokens, skip_special_tokens=not chat, clean_up_tokenization_spaces=False
    )
    if not final:
        # A byte-level tokenizer can temporarily decode an incomplete Unicode
        # character. Muse's channel delimiters must also remain out of deltas.
        text = text.rstrip("\ufffd")
        if chat:
            for marker in ("<|start|>", "<|eom|>", "<|eot|>", "<|end_of_text|>"):
                for size in range(min(len(marker) - 1, len(text)), 0, -1):
                    if text.endswith(marker[:size]):
                        text = text[:-size]
                        break
    return parse_response(text, tools, final=final) if chat else {"text": text}


def response_usage(runtime, ids, tokens):
    # Snapshot while the generation lock is held: another request may replace
    # generation_stats before the response body or final SSE frame is emitted.
    cached = getattr(runtime, "generation_stats", {}).get("cached_prompt_tokens", 0)
    return {
        "prompt_tokens": len(ids),
        "completion_tokens": len(tokens),
        "total_tokens": len(ids) + len(tokens),
        "prompt_tokens_details": {"cached_tokens": cached},
    }


def streaming_response(
    runtime,
    tokenizer,
    ids,
    limit,
    eos,
    chat,
    model_name,
    lock,
    *,
    include_usage=False,
    tools=None,
    parallel_tool_calls=True,
):
    """Keep one device worker and request lock until completion or cancellation."""
    stopped = threading.Event()
    started = False

    async def events():
        nonlocal started
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def send(kind, value):
            if not stopped.is_set():
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, (kind, value))
                except RuntimeError:
                    stopped.set()

        def produce():
            result = None
            try:
                tokens = runtime.generate(
                    ids,
                    limit,
                    eos=eos,
                    on_token=lambda t: send("token", t),
                    cancel=stopped,
                )
                result = ("done", response_usage(runtime, ids, tokens))
            except InterruptedError:
                pass
            except Exception as exc:
                result = ("error", str(exc))
            finally:
                try:
                    if (
                        result is not None
                        and result[0] == "done"
                        and not stopped.is_set()
                    ):
                        getattr(runtime, "release_request_cache", runtime.reset)()
                    else:
                        runtime.reset()
                except Exception as exc:
                    result = ("error", str(exc))
                finally:
                    lock.release()
            if result is not None:
                send(*result)

        base = {
            "id": ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex,
            "object": "chat.completion.chunk" if chat else "text_completion",
            "created": int(time.time()),
            "model": model_name,
        }

        def frame(value):
            return "data: " + json.dumps(value) + "\n\n"

        def chunk(delta, finish=None):
            choice = {"index": 0, "finish_reason": finish}
            choice.update(
                {"delta": delta}
                if chat
                else {"text": delta.get("text", ""), "logprobs": None}
            )
            return {**base, "choices": [choice]}

        worker = threading.Thread(target=produce, name="muse-generation", daemon=True)
        started = True
        try:
            worker.start()
        except Exception:
            lock.release()
            raise
        tokens, emitted = [], {}
        try:
            if chat:
                yield frame(chunk({"role": "assistant"}))
            while True:
                kind, value = await queue.get()
                if kind == "error":
                    yield frame(
                        {"error": {"message": value, "type": "generation_error"}}
                    )
                    return
                if kind == "token":
                    tokens.append(value)
                visible = visible_stream_text(
                    tokenizer, tokens, chat, final=kind == "done", tools=tools
                )
                delta = {}
                for key in ("content", "reasoning_content") if chat else ("text",):
                    text = visible.get(key, "")
                    previous = emitted.get(key, "")
                    if not text.startswith(previous):
                        raise RuntimeError("tokenizer revised already streamed text")
                    if len(text) > len(previous):
                        delta[key] = text[len(previous) :]
                    emitted[key] = text
                if delta:
                    yield frame(chunk(delta))
                if kind == "done":
                    calls = visible.get("tool_calls", [])
                    if not parallel_tool_calls and len(calls) > 1:
                        raise ValueError(
                            "model produced multiple calls with parallel_tool_calls=false"
                        )
                    if calls:
                        yield frame(
                            chunk(
                                {
                                    "tool_calls": [
                                        {"index": i, **call}
                                        for i, call in enumerate(calls)
                                    ]
                                }
                            )
                        )
                    finish = (
                        "tool_calls"
                        if calls
                        else ("stop" if tokens and tokens[-1] in eos else "length")
                    )
                    yield frame(chunk({}, finish))
                    if include_usage:
                        yield frame(
                            {
                                **base,
                                "choices": [],
                                "usage": value,
                            }
                        )
                    yield "data: [DONE]\n\n"
                    return
        except Exception as exc:
            yield frame({"error": {"message": str(exc), "type": "stream_error"}})
        finally:
            stopped.set()

    class Response(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                stopped.set()
                # A disconnect while sending response headers can prevent the
                # iterator (and its worker/finally block) from starting at all.
                if not started:
                    lock.release()

    return Response(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
            "execution": getattr(
                runtime,
                "execution",
                "sequential layer placement; greedy; single request",
            ),
            "optimized": getattr(runtime, "optimized", False),
            "bounded_attention": getattr(runtime, "bounded_attention", False),
            "prefill_chunk": getattr(runtime, "prefill_chunk", None),
            "prefix_cache": getattr(runtime, "prefix_cache_enabled", False),
            "tensor_output": getattr(runtime, "tensor_output", False),
            "deterministic": getattr(runtime, "deterministic", False),
            "streaming": True,
            "tools": True,
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
        if request.n != 1 or request.temperature != 0 or request.top_p != 1:
            raise HTTPException(
                400,
                "runtime supports greedy generation with n=1, temperature=0, top_p=1",
            )
        if request.reasoning_effort not in ("low", "medium", "high", "xhigh"):
            raise HTTPException(400, "invalid reasoning_effort")
        try:
            tools = active_tools(request.tools, request.tool_choice)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not chat and (request.tools is not None or request.tool_choice is not None):
            raise HTTPException(400, "tool fields require chat completions")
        if chat:
            if not request.messages or request.prompt is not None:
                raise HTTPException(400, "chat requires messages and no prompt")
            if any(
                m.role not in ("system", "developer", "user", "assistant", "tool")
                for m in request.messages
            ):
                raise HTTPException(400, "unsupported message role")
            try:
                ids = tokenizer.apply_chat_template(
                    [
                        m.model_dump(exclude={"reasoning_content"}, exclude_none=True)
                        for m in request.messages
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                    reasoning_effort=request.reasoning_effort,
                    **({"tools": tools} if tools else {}),
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
        if resident and resident.failed:
            lock.release()
            raise HTTPException(503, "resident controls blocked; restart required")
        if request.stream:
            return streaming_response(
                runtime,
                tokenizer,
                ids,
                limit,
                eos,
                chat,
                model_name,
                lock,
                include_usage=bool(
                    request.stream_options and request.stream_options.include_usage
                ),
                tools=tools,
                parallel_tool_calls=request.parallel_tool_calls,
            )
        completed = False
        try:
            tokens = runtime.generate(ids, limit, eos=eos)
            usage = response_usage(runtime, ids, tokens)
            completed = True
        finally:
            try:
                if completed:
                    getattr(runtime, "release_request_cache", runtime.reset)()
                else:
                    runtime.reset()
            finally:
                lock.release()
        finish = "stop" if tokens and tokens[-1] in eos else "length"
        choice = {"index": 0, "finish_reason": finish}
        if chat:
            try:
                choice["message"] = parse_response(
                    tokenizer.decode(
                        tokens,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    tools,
                )
            except ValueError as exc:
                raise HTTPException(502, str(exc)) from exc
            calls = choice["message"].get("tool_calls", [])
            if not request.parallel_tool_calls and len(calls) > 1:
                raise HTTPException(
                    502, "model produced multiple calls with parallel_tool_calls=false"
                )
            if calls:
                choice["finish_reason"] = "tool_calls"
        else:
            choice["text"] = tokenizer.decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            choice["logprobs"] = None
        return {
            "id": ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex,
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [choice],
            "usage": usage,
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
    parser.add_argument("--tensor-experts", action="store_true")
    parser.add_argument("--tensor-attention", action="store_true")
    parser.add_argument("--tensor-output", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--bounded-attention", action="store_true")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--packed-cache", type=Path)
    parser.add_argument("--resident-token-file", type=Path)
    parser.add_argument("--native-manifest", type=Path)
    parser.add_argument("--resident-patch-file", type=Path)
    args = parser.parse_args()
    if args.bounded_attention and (not args.optimized or args.max_context < 1):
        parser.error("--bounded-attention requires --optimized and positive context")
    if args.resident_token_file and not args.optimized:
        parser.error("resident controls require --optimized")
    native_manifest = None
    if args.native_manifest:
        if not args.resident_token_file:
            parser.error("--native-manifest requires --resident-token-file")
        from reap.muse_native import check_manifest

        native_manifest = json.loads(args.native_manifest.read_text())
        check_manifest(native_manifest)
    source = None
    if args.resident_patch_file:
        if not args.resident_token_file:
            parser.error("--resident-patch-file requires --resident-token-file")
        source = args.resident_patch_file.read_text()
        compile(source, str(args.resident_patch_file), "exec")
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
        tensor_experts=args.tensor_experts,
        tensor_attention=args.tensor_attention,
        tensor_output=args.tensor_output,
        prefix_cache=args.prefix_cache,
    )
    torch.set_num_threads(1 if args.tensor_experts else 2)
    runtime.deterministic = args.deterministic
    resident = None
    if args.resident_token_file:
        from reap.muse_resident import Resident

        resident = Resident(runtime)
        if native_manifest:
            resident.load_native(native_manifest)
        if native_manifest or source:
            receipt = resident.switch(
                native=native_manifest["name"] if native_manifest else None,
                source=source,
            )
            print(
                json.dumps({"event": "startup_resident_activated", **receipt}),
                flush=True,
            )
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
        runtime.close()


if __name__ == "__main__":
    main()
