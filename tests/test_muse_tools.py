import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import pytest
import torch
from fastapi.testclient import TestClient

from reap.muse_serve import create_app, parse_response, Message
from reap.muse_tools import extract_calls
from reap.muse_tensor import make_pool
from tests.test_muse_serve import Runtime, Tokenizer

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "flags": {"type": "array"},
                    "enabled": {"type": "boolean"},
                },
            },
        },
    }
]
CALL = (
    '<atem:function_calls>\n<atem:invoke name="lookup">\n'
    '<atem:parameter name="query"> 123 & <x> </atem:parameter>\n'
    '<atem:parameter name="limit">2</atem:parameter>\n'
    '<atem:parameter name="flags">["a", "b"]</atem:parameter>\n'
    '<atem:parameter name="enabled">false</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)


@pytest.mark.parametrize("stream", [False, True])
def test_kilo_tools_auto_returns_normal_chat_and_preserves_schemas(stream):
    runtime, tokenizer = Runtime(), Tokenizer()
    client = TestClient(create_app(runtime, tokenizer, eos=[2]))
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "what are you?"}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": 3,
            "stream": stream,
        },
    )
    assert response.status_code == 200
    assert tokenizer.chat[1]["tools"] == TOOLS
    if stream:
        assert "answer" in response.text and "[DONE]" in response.text
    else:
        assert response.json()["choices"][0]["message"]["content"] == "answer"


@pytest.mark.parametrize("stream", [False, True])
def test_tool_calls_are_api_calls_not_visible_xml_and_history_roundtrips(stream):
    class ToolTokenizer(Tokenizer):
        def decode(self, ids, **kwargs):
            return " to=lookup<|message|>" + CALL + ("<|eot|>" if len(ids) > 1 else "")

    runtime, tokenizer = Runtime(), ToolTokenizer()
    client = TestClient(create_app(runtime, tokenizer, eos=[2]))
    body = {
        "messages": [{"role": "user", "content": "look up 123"}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 3,
        "stream": stream,
    }
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
        calls = next(
            e["choices"][0]["delta"]["tool_calls"]
            for e in events
            if e["choices"][0]["delta"].get("tool_calls")
        )
        assert calls[0].pop("index") == 0
        assert not any(e["choices"][0]["delta"].get("content") for e in events)
    else:
        choice = response.json()["choices"][0]
        assert (
            choice["finish_reason"] == "tool_calls"
            and choice["message"]["content"] == ""
        )
        calls = choice["message"]["tool_calls"]
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "query": " 123 & <x> ",
        "limit": 2,
        "flags": ["a", "b"],
        "enabled": False,
    }
    body["messages"] += [
        {"role": "assistant", "content": None, "tool_calls": calls},
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": "result"},
    ]
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    history = tokenizer.chat[0]
    assert isinstance(history[-2]["tool_calls"][0]["function"]["arguments"], dict)
    assert history[-1]["tool_call_id"] == calls[0]["id"]


def test_tool_none_forced_choices_and_argument_validation():
    tokenizer = Tokenizer()
    client = TestClient(create_app(Runtime(), tokenizer, eos=[2]))
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": TOOLS,
        "max_tokens": 3,
    }
    assert (
        client.post(
            "/v1/chat/completions", json={**body, "tool_choice": "none"}
        ).status_code
        == 200
    )
    assert "tools" not in tokenizer.chat[1]
    for choice in ("required", {"type": "function", "function": {"name": "lookup"}}):
        assert (
            client.post(
                "/v1/chat/completions", json={**body, "tool_choice": choice}
            ).status_code
            == 400
        )
    with pytest.raises(ValueError):
        Message(
            role="assistant",
            tool_calls=[{"id": "a", "function": {"name": "lookup", "arguments": "[]"}}],
        )
    with pytest.raises(ValueError, match="unavailable"):
        extract_calls(CALL.replace('name="lookup"', 'name="missing"'), TOOLS)
    for i in range(1, len("<atem:function_calls>")):
        assert (
            parse_response(
                " to=user<|message|>hello" + "<atem:function_calls>"[:i],
                TOOLS,
                final=False,
            )["content"]
            == "hello"
        )


@pytest.mark.parametrize("failure_phase", [None, "up", "down"])
def test_pipelined_experts_drain_failures_and_reuse_device_routes(
    monkeypatch, failure_phase
):
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(device=lambda _: nullcontext()), raising=False
    )
    scope = {}
    exec(
        (
            Path(__file__).parents[1]
            / "scripts/muse_resident_patches/layer_decode_graphs.py"
        ).read_text(),
        scope,
    )
    fault = [failure_phase]

    class Graph:
        def __init__(self, rank, phase):
            self.rank, self.phase = rank, phase
            self.routes = torch.zeros(8, dtype=torch.int32)

        def run(self, value, routes):
            if self.rank == 2 and fault[0] == self.phase:
                raise RuntimeError("injected rank failure")
            if self.phase == "down":
                assert routes is graphs[self.rank, "up"].routes
            self.routes.copy_(routes)
            return (
                value[:, self.rank : self.rank + 1] + routes[:, None]
                if self.phase == "up"
                else value[:, self.rank : self.rank + 1] * 2
            )

    graphs = {(i, p): Graph(i, p) for i in range(3) for p in ("up", "down")}
    pool = make_pool(["cpu"] * 3)
    experts = SimpleNamespace(
        ranks=[SimpleNamespace(device="cpu") for _ in range(3)],
        graphs=graphs,
        pool=pool,
        prepare_graphs=lambda *args: None,
    )
    try:
        x = torch.arange(24).reshape(8, 3).float()
        ids = torch.arange(8, dtype=torch.int32)
        if failure_phase:
            with pytest.raises(RuntimeError, match="injected"):
                scope["pipelined_project"](experts, x, ids)
            fault[0] = None
        for ids in (ids, ids.flip(0)):
            actual = scope["pipelined_project"](experts, x, ids)
            torch.testing.assert_close(actual, (x + ids[:, None]) * 2, rtol=0, atol=0)
    finally:
        pool.shutdown()
