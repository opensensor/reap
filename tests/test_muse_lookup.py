"""Exact algorithm tests with deterministic logits and real bounded CPU KV."""

from types import SimpleNamespace

import pytest
import torch

from reap.muse_attention import AppendCache, bounded_attention
from reap.muse_lookup import PromptLookup, generate


class Runtime:
    bounded_attention = True
    prefill_chunk = 8
    max_context = 256
    c = SimpleNamespace(vocab_size=32)

    def __init__(self):
        self.calls = []
        self.fail_after_position = None
        self.history_dependent = False
        self.reset()

    def reset(self):
        self.cache = [AppendCache(), AppendCache()]
        self.seen = 0
        self.prefix_snapshot = None

    @torch.inference_mode()
    def forward(self, ids, *, all_logits=False):
        self.calls.append((self.seen, list(ids), all_logits))
        x = torch.tensor(ids).float().reshape(1, 1, -1, 1).expand(-1, -1, -1, 4) * 0.01
        for index, (cache, window) in enumerate(zip(self.cache, (None, 4))):
            bounded_attention(
                x,
                x,
                x,
                start=self.seen,
                cache=cache,
                window=window,
                prefill_chunk=self.prefill_chunk,
                max_context=self.max_context,
            )
            if (
                index == 0
                and self.fail_after_position is not None
                and self.seen >= self.fail_after_position
            ):
                raise RuntimeError("injected model failure")
        # Every row predicts the next number, independent of batching. Real
        # target models still need their own batched numerical qualification.
        logits = torch.full((1, len(ids), self.c.vocab_size), -10.0)
        for row, token in enumerate(ids):
            if self.history_dependent:
                prefix = self.cache[0].key[0, 0, : self.seen + row + 1, 0]
                next_token = (
                    int(torch.round(prefix / 0.01).long().sum()) % self.c.vocab_size
                )
            else:
                next_token = (token + 1) % self.c.vocab_size
            logits[0, row, next_token] = 10.0
        self.seen += len(ids)
        return logits if all_logits else logits[:, -1:]


def sequential(runtime, prompt, count, eos=()):
    for offset in range(0, len(prompt), runtime.prefill_chunk):
        logits = runtime.forward(prompt[offset : offset + runtime.prefill_chunk])
    result = []
    for index in range(count):
        token = int(logits[0, -1].argmax())
        result.append(token)
        if token in eos or index == count - 1:
            break
        logits = runtime.forward([token])
    return result


def assert_cache_equal(actual, expected):
    assert actual.seen == expected.seen
    for cache, reference in zip(actual.cache, expected.cache):
        assert (cache.seen, cache.length) == (reference.seen, reference.length)
        assert cache._speculative_transaction is None
        torch.testing.assert_close(
            cache.key[:, :, : cache.length],
            reference.key[:, :, : reference.length],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cache.value[:, :, : cache.length],
            reference.value[:, :, : reference.length],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "prompt,draft,matched",
    [
        ([1, 2, 9, 1], 3, 0),
        ([1, 2, 3, 4, 9, 1], 4, 2),
        ([1, 2, 3, 4, 5, 6, 7, 8, 1], 3, 3),
    ],
)
def test_first_middle_and_full_acceptance_reuse_correct_logit_row(
    prompt, draft, matched
):
    runtime, reference = Runtime(), Runtime()
    result = generate(
        runtime,
        prompt,
        max_new_tokens=10,
        max_proposal_tokens=draft,
        min_ngram=1,
        max_ngram=1,
    )
    assert result["tokens"] == sequential(reference, prompt, 10)
    first = result["steps"][0]
    assert first["matched"] == matched
    assert first["committed_inputs"] == matched + 1
    assert first["block_rows"] > 1
    assert_cache_equal(runtime, reference)
    stats = result["stats"]
    assert stats["committed_input_tokens"] == len(result["tokens"]) - 1
    assert stats["verification_calls"] == sum(
        step["block_rows"] > 1 for step in result["steps"]
    )
    assert stats["accepted_draft_tokens"] == sum(
        step["matched"] for step in result["steps"]
    )
    assert stats["proposed_tokens"] == sum(
        len(step["proposed"]) for step in result["steps"]
    )
    assert all(value >= 0 for key, value in stats.items() if key.endswith("seconds"))


@pytest.mark.parametrize("limit", range(1, 12))
def test_length_cap_preserves_last_uncached_token(limit):
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 1]
    runtime, reference = Runtime(), Runtime()
    result = generate(runtime, prompt, max_new_tokens=limit, min_ngram=1, max_ngram=2)
    assert result["tokens"] == sequential(reference, prompt, limit)
    assert len(result["tokens"]) == limit
    assert runtime.seen == len(prompt) + limit - 1
    assert_cache_equal(runtime, reference)


@pytest.mark.parametrize(
    "prompt,eos",
    [
        ([1], (2,)),
        ([1, 2, 3, 4, 5, 1], (4,)),
        ([1, 2, 9, 1], (9,)),
    ],
)
def test_anchor_accepted_draft_and_rejected_draft_eos(prompt, eos):
    runtime, reference = Runtime(), Runtime()
    result = generate(
        runtime, prompt, max_new_tokens=15, eos=eos, min_ngram=1, max_ngram=1
    )
    assert result["tokens"] == sequential(reference, prompt, 15, eos)
    assert result["tokens"][-1] in eos
    assert_cache_equal(runtime, reference)
    if eos == (4,):
        step = result["steps"][0]
        assert step["proposed"] == [3, 4, 5]
        assert step["matched"] == 2 and step["committed_inputs"] == 2
        assert step["stopped_on_draft_eos"] is True
    if eos == (9,):
        assert result["steps"][0]["matched"] == 0
        assert result["tokens"] == list(range(2, 10))


def test_no_lookup_uses_single_row_fallback_without_transactions():
    runtime, reference = Runtime(), Runtime()
    prompt = [20]
    result = generate(runtime, prompt, max_new_tokens=6)
    assert result["tokens"] == sequential(reference, prompt, 6)
    assert result["stats"]["fallback_calls"] == 5
    assert result["stats"]["verification_calls"] == 0
    assert result["stats"]["acceptance_rate"] is None
    assert_cache_equal(runtime, reference)


def test_maximum_block_and_rolled_sliding_cache_match_baseline():
    runtime, reference = Runtime(), Runtime()
    prompt = list(range(1, 16)) * 2 + [1]
    result = generate(runtime, prompt, max_new_tokens=18, min_ngram=1, max_ngram=2)
    assert result["tokens"] == sequential(reference, prompt, 18)
    assert max(step["block_rows"] for step in result["steps"]) == 8
    assert any(step["matched"] == 7 for step in result["steps"])
    assert_cache_equal(runtime, reference)
    # Continuing from the terminal uncached token is also identical.
    torch.testing.assert_close(
        runtime.forward([result["tokens"][-1]]),
        reference.forward([result["tokens"][-1]]),
        rtol=0,
        atol=0,
    )


def test_rejected_proposals_never_enter_history(monkeypatch):
    recorded = []
    original = PromptLookup.append

    def append(self, token):
        recorded.append(token)
        original(self, token)

    monkeypatch.setattr(PromptLookup, "append", append)
    prompt = [1, 2, 9, 1]
    result = generate(Runtime(), prompt, max_new_tokens=7, min_ngram=1, max_ngram=1)
    assert recorded == prompt + result["tokens"]


@pytest.mark.parametrize("prompt", [[0] * 20, [1, 2, 9, 1, 20]])
def test_logits_conditioned_on_entire_committed_prefix_match_baseline(prompt):
    runtime, reference = Runtime(), Runtime()
    runtime.history_dependent = reference.history_dependent = True
    result = generate(runtime, prompt, max_new_tokens=15, min_ngram=1, max_ngram=1)
    assert result["tokens"] == sequential(reference, prompt, 15)
    assert result["stats"]["verification_calls"] > 0
    assert_cache_equal(runtime, reference)


def test_index_prefers_long_suffix_and_has_bounded_storage():
    lookup = PromptLookup(min_ngram=1, max_ngram=2, max_entries=20)
    for token in [1, 2, 3, 7, 1, 2, 4, 8, 1, 2]:
        lookup.append(token)
    assert lookup.propose(2) == [4, 8]
    for token in range(100):
        lookup.append(token)
        assert len(lookup.index) <= 20
        assert all(len(locations) <= 4 for locations in lookup.index.values())
    assert lookup.propose(0) == []


def test_overlapping_repeat_lookup_uses_only_known_continuation():
    lookup = PromptLookup(min_ngram=1, max_ngram=1)
    for token in [1] * 5:
        lookup.append(token)
    assert lookup.propose(7) == [1] * 4


def test_verification_failure_resets_partial_runtime():
    runtime = Runtime()
    prompt = [1, 2, 3, 4, 5, 1]
    runtime.fail_after_position = len(prompt)
    with pytest.raises(RuntimeError, match="injected model failure"):
        generate(runtime, prompt, max_new_tokens=10, min_ngram=1, max_ngram=1)
    assert runtime.seen == 0
    assert all(
        cache.key is None and cache._speculative_transaction is None
        for cache in runtime.cache
    )


def test_requires_reset_runtime_without_destroying_existing_state():
    runtime = Runtime()
    runtime.forward([1, 2])
    caches = runtime.cache
    with pytest.raises(ValueError, match="reset bounded-KV"):
        generate(runtime, [1], max_new_tokens=5)
    assert runtime.cache is caches and runtime.seen == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"max_proposal_tokens": 8},
        {"min_ngram": 0},
        {"eos": [99]},
    ],
)
def test_invalid_configuration_fails_before_prefill(kwargs):
    runtime = Runtime()
    with pytest.raises(ValueError):
        generate(runtime, [1], **kwargs)
    assert runtime.calls == [] and runtime.seen == 0


@pytest.mark.parametrize("proposal_cap", range(8))
def test_lookup_only_uses_qualified_verification_shapes(proposal_cap):
    runtime, reference = Runtime(), Runtime()
    prompt = list(range(1, 16)) * 2 + [1]
    result = generate(
        runtime,
        prompt,
        max_new_tokens=18,
        max_proposal_tokens=proposal_cap,
        min_ngram=1,
        max_ngram=1,
        eos=(9,),
    )
    assert result["tokens"] == sequential(reference, prompt, 18, (9,))
    assert all(step["block_rows"] in (1, 2, 4, 8) for step in result["steps"])
    assert all(len(step["proposed"]) <= proposal_cap for step in result["steps"])
    assert_cache_equal(runtime, reference)
