import asyncio
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from reap.muse_serve import create_app, streaming_response, visible_stream_text
from tests.test_muse_serve import Runtime, Tokenizer


def test_stream_contract_matches_nonstream_and_releases_request():
    runtime = Runtime()
    client = TestClient(create_app(runtime, Tokenizer(), eos=[2]))
    for chat in (False, True):
        path = "/v1/chat/completions" if chat else "/v1/completions"
        body = (
            {"messages": [{"role": "user", "content": "hello"}]}
            if chat
            else {"prompt": "hello"}
        )
        body["max_tokens"] = 3
        expected = client.post(path, json=body).json()
        response = client.post(
            path,
            json={**body, "stream": True, "stream_options": {"include_usage": True}},
        )
        assert response.status_code == 200
        lines = [s[6:] for s in response.text.splitlines() if s.startswith("data: ")]
        assert lines[-1] == "[DONE]"
        events = [json.loads(s) for s in lines[:-1]]
        assert events[-1]["usage"] == expected["usage"]
        assert events[-2]["choices"][0]["finish_reason"] == "stop"
        if chat:
            for field in ("content", "reasoning_content"):
                assert "".join(
                    e["choices"][0]["delta"].get(field, "")
                    for e in events
                    if e["choices"]
                ) == expected["choices"][0]["message"].get(field, "")
        else:
            assert (
                "".join(e["choices"][0]["text"] for e in events if e["choices"])
                == expected["choices"][0]["text"]
            )
        assert client.post(path, json=body).status_code == 200


class PieceTokenizer:
    def decode(self, tokens, **kwargs):
        return "".join(tokens)


def test_split_headers_delimiters_and_unicode_do_not_leak():
    raw = " to=self<|message|>think<|eom|><|start|>assistant to=user<|message|>answer<|eot|>"
    emitted = {"content": "", "reasoning_content": ""}
    for end in range(1, len(raw) + 1):
        current = visible_stream_text(PieceTokenizer(), list(raw[:end]), True)
        for key in emitted:
            text = current.get(key, "")
            assert text.startswith(emitted[key]), (end, key, text, emitted)
            assert "<|" not in text
            emitted[key] = text
    assert emitted == {"content": "answer", "reasoning_content": "think"}
    assert visible_stream_text(PieceTokenizer(), ["a", "\ufffd"], False)["text"] == "a"
    assert visible_stream_text(PieceTokenizer(), ["a", "é"], False)["text"] == "aé"


def test_first_token_arrives_before_completion_and_disconnect_cancels():
    class SlowRuntime:
        reset_done = threading.Event()
        cancelled = threading.Event()

        def generate(self, ids, limit, eos, on_token, cancel):
            on_token("hello")
            assert cancel.wait(3), "stream close did not cancel generation"
            self.cancelled.set()
            raise InterruptedError()

        def reset(self):
            self.reset_done.set()

    async def exercise():
        runtime = SlowRuntime()
        lock = threading.Lock()
        lock.acquire()
        response = streaming_response(
            runtime, PieceTokenizer(), [1], 4, [], False, "muse", lock
        )
        iterator = response.body_iterator
        first = await asyncio.wait_for(anext(iterator), 2)
        assert "hello" in first
        assert lock.locked() and not runtime.reset_done.is_set()
        await iterator.aclose()
        assert await asyncio.to_thread(runtime.reset_done.wait, 2)
        assert runtime.cancelled.is_set() and not lock.locked()

    asyncio.run(exercise())


def test_stream_failure_is_an_error_and_releases_lock():
    class Failed(Runtime):
        def generate(self, *args, **kwargs):
            raise FloatingPointError("nonfinite logits")

    client = TestClient(create_app(Failed(), Tokenizer(), eos=[2]))
    for _ in range(2):
        result = client.post(
            "/v1/completions", json={"prompt": "hello", "max_tokens": 3, "stream": True}
        )
        assert "nonfinite logits" in result.text
        assert "[DONE]" not in result.text


def test_text_parts_and_reasoning_history_without_accepting_images():
    client = TestClient(create_app(Runtime(), Tokenizer(), eos=[2]))
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
                "reasoning_content": "earlier reasoning",
            }
        ],
        "max_tokens": 3,
    }
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    body["messages"][0]["content"] = [
        {"type": "image_url", "image_url": {"url": "invalid"}}
    ]
    assert client.post("/v1/chat/completions", json=body).status_code == 422


def test_disconnect_before_iterator_starts_releases_lock():
    async def exercise():
        lock = threading.Lock()
        lock.acquire()
        response = streaming_response(
            Runtime(), PieceTokenizer(), [1], 3, [], False, "muse", lock
        )

        async def send(message):
            raise OSError("disconnected before response headers")

        async def receive():
            return {"type": "http.disconnect"}

        with pytest.raises(Exception):
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
            )
        assert not lock.locked()

    asyncio.run(exercise())
