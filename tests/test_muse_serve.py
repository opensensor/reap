import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from reap.muse_serve import create_app, parse_response


class Runtime:
    max_context = 20
    resets = 0

    def generate(self, ids, limit, eos, *, on_token=None, cancel=None):
        self.call = (ids, limit, eos)
        if on_token:
            for token in [3, 2]:
                on_token(token)
        return [3, 2]

    def reset(self):
        self.resets += 1


class Tokenizer:
    def encode(self, text, **kwargs):
        return [1, 4, 5]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"]
        self.chat = (messages, kwargs)
        return [1, 5, 4]

    def decode(self, ids, skip_special_tokens, **kwargs):
        return (
            "answer"
            if skip_special_tokens
            else " to=self<|message|>thought<|eom|><|start|>assistant to=user<|message|>answer<|eot|>"
        )


def test_endpoint_contract_context_and_request_isolation():
    runtime, tokenizer = Runtime(), Tokenizer()
    client = TestClient(create_app(runtime, tokenizer, eos=[2], model_name="muse"))
    assert client.get("/health").status_code == 200
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "muse",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
        },
    )
    assert response.status_code == 200
    value = response.json()
    assert value["choices"][0]["message"] == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "thought",
    }
    assert value["choices"][0]["finish_reason"] == "stop"
    assert value["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert runtime.resets == 1
    assert runtime.call == ([1, 5, 4], 3, [2])
    for changes in ({"temperature": 1}, {"max_tokens": 25}, {"n": 2}):
        assert (
            client.post(
                "/v1/completions", json={"prompt": "code", "max_tokens": 3, **changes}
            ).status_code
            == 400
        )
    assert (
        client.post("/v1/completions", json={"prompt": "code", "tools": []}).status_code
        == 422
    )
    assert (
        client.post("/v1/completions", json={"prompt": "code", "max_tokens": 3}).json()[
            "choices"
        ][0]["text"]
        == "answer"
    )


def test_channels_and_partial_header():
    assert parse_response(" to=user<|message|>partial")["content"] == "partial"
    assert parse_response(" to=self<|message|>unfinished") == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "unfinished",
    }
    assert parse_response(" to=") == {"role": "assistant", "content": ""}


def test_real_tokenizer_chat_result_is_token_ids():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message.content }} {% endfor %}"
    )
    runtime = Runtime()
    client = TestClient(create_app(runtime, tokenizer, eos=[2]))
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 3},
    )
    assert response.status_code == 200
    assert runtime.call[0] == [1]
