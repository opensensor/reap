"""Serving integration with real bounded KV and deterministic CPU logits."""

import json
from threading import Event
from types import SimpleNamespace

import pytest

from reap.muse_lookup import decode
from reap.muse_runtime import MuseRuntime
from tests.test_muse_lookup import Runtime, assert_cache_equal


class ServingRuntime(Runtime):
    generate = MuseRuntime.generate
    _generate = MuseRuntime._generate
    reset = MuseRuntime.reset
    release_request_cache = MuseRuntime.release_request_cache
    optimized = True
    prefix_cache_enabled = True
    c = SimpleNamespace(
        vocab_size=32, layer_types=["full_attention", "sliding_attention"]
    )

    def __init__(self, lookup=7):
        self.layers = [None, None]
        self.experimental_lookup_tokens = lookup
        self.generation_stats = {}
        self.events = []
        self.progress = lambda event: self.events.append(json.loads(event))
        super().__init__()


@pytest.mark.parametrize("suffix", [[], [1], [1, 8]])
def test_prefix_reuse_changed_suffix_and_streaming_without_second_prefill(suffix):
    prompt = list(range(1, 9))
    runtime = ServingRuntime()
    runtime.generate(prompt, max_new_tokens=3)
    runtime.release_request_cache()
    assert runtime.prefix_snapshot is not None
    runtime.calls.clear()
    tokens = []
    actual = runtime.generate(
        prompt + suffix, max_new_tokens=18, on_token=tokens.append
    )
    reference = ServingRuntime(0)
    expected = reference.generate(prompt + suffix, max_new_tokens=18)
    assert actual == expected == tokens
    assert_cache_equal(runtime, reference)
    assert runtime.generation_stats["cached_prompt_tokens"] == 8
    assert runtime.generation_stats["lookup"]["completion_tokens"] == len(actual)
    prefill = [call for call in runtime.calls if not call[2]]
    assert prefill == ([(8, suffix, False)] if suffix else [])
    assert [
        event["tokens"]
        for event in runtime.events
        if event["event"] == "decode_progress"
    ] == [16]
    assert runtime.prefix_snapshot.tokens == tuple(prompt)


@pytest.mark.parametrize("eos", [(2,), (4,), (9,), ()])
def test_streaming_only_emits_verified_tokens_and_eos(eos):
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 1]
    runtime, reference = ServingRuntime(), ServingRuntime(0)
    tokens = []
    actual = runtime.generate(
        prompt, max_new_tokens=10, eos=eos, on_token=tokens.append
    )
    assert actual == reference.generate(prompt, max_new_tokens=10, eos=eos) == tokens
    assert_cache_equal(runtime, reference)
    if eos == (4,):
        assert runtime.generation_stats["lookup"]["accepted_draft_tokens"] == 2


def test_rejected_proposal_never_streamed():
    runtime = ServingRuntime()
    tokens = []
    actual = runtime.generate([1, 2, 9, 1], max_new_tokens=5, on_token=tokens.append)
    assert actual == tokens == [2, 3, 4, 5, 6]
    assert runtime.generation_stats["lookup"]["proposed_tokens"] > 0
    assert runtime.generation_stats["lookup"]["accepted_draft_tokens"] == 0


def test_changed_prefix_recomputes_and_uses_actual_history():
    runtime, reference = ServingRuntime(), ServingRuntime(0)
    runtime.history_dependent = reference.history_dependent = True
    runtime.generate([0] * 8, max_new_tokens=3)
    runtime.release_request_cache()
    runtime.calls.clear()
    prompt = [1] + [0] * 10
    actual = runtime.generate(prompt, max_new_tokens=12)
    assert actual == reference.generate(prompt, max_new_tokens=12)
    assert runtime.generation_stats["cached_prompt_tokens"] == 0
    assert runtime.calls[:2] == [(0, prompt[:8], False), (8, prompt[8:], False)]
    assert_cache_equal(runtime, reference)


def assert_cleared(runtime):
    assert runtime.seen == 0
    assert runtime.prefix_snapshot is None
    assert runtime.generation_stats == {}
    assert all(
        cache.key is None and cache._speculative_transaction is None
        for cache in runtime.cache
    )


@pytest.mark.parametrize("cancel_after", [0, 1, 2, 4])
def test_cancellation_drops_existing_and_pending_prefix(cancel_after):
    runtime = ServingRuntime()
    prompt = list(range(1, 9)) + [1]
    runtime.generate(prompt, max_new_tokens=1)
    runtime.release_request_cache()
    cancel = Event()
    tokens = []
    if not cancel_after:
        cancel.set()

    def emit(token):
        tokens.append(token)
        if len(tokens) == cancel_after:
            cancel.set()

    with pytest.raises(InterruptedError, match="cancelled"):
        runtime.generate(prompt, max_new_tokens=4, on_token=emit, cancel=cancel)
    assert tokens == list(range(2, 2 + cancel_after))
    assert_cleared(runtime)


@pytest.mark.parametrize("phase", ["prefill", "verification", "callback"])
def test_failure_never_publishes_prefix(phase):
    runtime = ServingRuntime()
    prompt = list(range(1, 9)) + [1]
    runtime.generate(prompt, max_new_tokens=1)
    runtime.release_request_cache()
    runtime.fail_after_position = {"prefill": 8, "verification": 9}.get(phase)

    def emit(_):
        if phase == "callback":
            raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="failed|failure"):
        runtime.generate(prompt, max_new_tokens=10, on_token=emit)
    assert_cleared(runtime)


def test_cancellation_inside_verification_aborts_before_draft_callbacks(monkeypatch):
    runtime = ServingRuntime()
    prompt = list(range(1, 9)) + [1]
    cancel = Event()
    tokens = []
    forward = runtime.forward

    def interrupted_forward(ids, *, all_logits=False):
        logits = forward(ids, all_logits=all_logits)
        if all_logits and len(ids) > 1:
            cancel.set()
        return logits

    monkeypatch.setattr(runtime, "forward", interrupted_forward)
    with pytest.raises(InterruptedError, match="cancelled"):
        runtime.generate(
            prompt, max_new_tokens=10, on_token=tokens.append, cancel=cancel
        )
    assert tokens == [2]
    assert_cleared(runtime)


def test_decode_reuses_existing_prefill_and_snapshot():
    runtime = ServingRuntime()
    prompt = list(range(1, 9))
    runtime.generate(prompt, max_new_tokens=1)
    prefix = runtime.prefix_snapshot
    runtime.release_request_cache()
    runtime.cache = prefix.restore()
    runtime.seen = len(prompt)
    runtime.calls.clear()
    tokens = []
    result = decode(
        runtime, prompt, prefix.logits, max_new_tokens=4, on_token=tokens.append
    )
    assert result["tokens"] == tokens == [9, 10, 11, 12]
    assert all(call[2] for call in runtime.calls)
    assert runtime.prefix_snapshot is prefix


def test_default_generation_does_not_enter_lookup(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("default generation called lookup")

    monkeypatch.setattr("reap.muse_lookup.decode", forbidden)
    runtime = ServingRuntime(0)
    assert runtime.generate([1, 2], max_new_tokens=3) == [3, 4, 5]
    assert "lookup" not in runtime.generation_stats


@pytest.mark.parametrize(
    "value,optimized,bounded",
    [
        (True, True, True),
        (-1, True, True),
        (8, True, True),
        (1, False, True),
        (7, True, False),
    ],
)
def test_invalid_option_rejected_before_checkpoint_io(
    tmp_path, value, optimized, bounded
):
    with pytest.raises(ValueError, match="lookup"):
        MuseRuntime(
            tmp_path / "nonexistent",
            experimental_lookup_tokens=value,
            optimized=optimized,
            bounded_attention=bounded,
        )
