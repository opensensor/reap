from types import SimpleNamespace

import pytest
import torch

from reap.muse_attention import AppendCache, bounded_attention
from reap.muse_speculative import benchmark


class FakeRuntime:
    """Use real rolling KV storage and predictions that depend on full history."""

    def __init__(self, *, perturb_blocks=False, fail_block=False):
        self.c = SimpleNamespace(
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=4,
            vocab_size=17,
        )
        self.layers = [SimpleNamespace(device="cpu") for _ in self.c.layer_types]
        self.max_context = 128
        self.prefill_chunk = 8
        self.bounded_attention = True
        self.perturb_blocks, self.fail_block = perturb_blocks, fail_block
        self.full_allocations = set()
        self.reset()

    def reset(self):
        self.cache = [AppendCache(), AppendCache()]
        self.seen = 0
        self.prefix_snapshot = None

    def forward(self, ids, *, all_logits=False):
        if self.fail_block and len(ids) > 1 and self.seen:
            raise RuntimeError("injected verification failure")
        full, sliding = self.cache
        previous = [] if full.key is None else full.key[0, 0, : full.length, 0].tolist()
        if sliding.key is not None:
            assert sliding.key[0, 0, : sliding.length, 0].tolist() == previous[-3:]
        history = previous + list(ids)
        logits = torch.zeros(1, len(ids), self.c.vocab_size)
        for index in range(len(ids)):
            target = int(sum(history[: self.seen + index + 1])) % self.c.vocab_size
            logits[0, index, target] = 1
            if self.perturb_blocks and len(ids) > 1:
                logits[0, index, (target + 1) % self.c.vocab_size] = 2
        key = (
            torch.tensor(ids, dtype=torch.float16)
            .view(1, 1, -1, 1)
            .expand(-1, -1, -1, 4)
        )
        for cache, window in zip(self.cache, (None, self.c.sliding_window)):
            bounded_attention(
                torch.zeros_like(key),
                key,
                key,
                start=self.seen,
                cache=cache,
                window=window,
                max_context=self.max_context,
                prefill_chunk=self.prefill_chunk,
            )
        self.full_allocations.add(full.key.data_ptr())
        self.seen += len(ids)
        return logits if all_logits else logits[:, -1:]


@pytest.mark.parametrize("prompt", [[1, 2], [1, 2, 3, 4, 5, 6, 7]])
def test_oracle_restores_rolling_cache_and_retains_one_full_allocation(prompt):
    runtime = FakeRuntime()
    result = benchmark(runtime, prompt, max_tokens=8, repeats=1)
    assert result["completed"]
    assert not result["draft_cost_included"] and not result["rejection_rollback_tested"]
    assert len(result["records"]) == 5
    assert all(
        row["all_logits_exact"] and row["argmax_exact"] for row in result["records"]
    )
    assert result["past_sliding_window"] == (len(prompt) >= 4)
    assert {row["forward_calls"] for row in result["records"]} == {1, 2, 4, 8}
    assert len(runtime.full_allocations) == 1
    assert runtime.seen == 0 and all(cache.key is None for cache in runtime.cache)


def test_oracle_reports_batched_disagreements_without_claiming_acceptance():
    result = benchmark(
        FakeRuntime(perturb_blocks=True), [1, 2], max_tokens=8, repeats=1
    )
    for row in result["records"]:
        assert row["argmax_exact"] == (row["block_rows"] == 1)
        assert row["all_logits_exact"] == row["argmax_exact"]
        assert len(row["argmax_mismatch_positions"]) == (
            0 if row["block_rows"] == 1 else 8
        )


def test_oracle_resets_after_failed_block():
    runtime = FakeRuntime(fail_block=True)
    with pytest.raises(RuntimeError, match="injected"):
        benchmark(runtime, [1, 2], max_tokens=8, repeats=1)
    assert runtime.seen == 0 and all(cache.key is None for cache in runtime.cache)


def test_oracle_rejects_live_request_without_resetting_it():
    runtime = FakeRuntime()
    runtime.forward([1, 2])
    with pytest.raises(ValueError, match="idle, reset"):
        benchmark(runtime, [3, 4], max_tokens=8)
    assert runtime.seen == 2 and all(cache.seen == 2 for cache in runtime.cache)


@pytest.mark.parametrize(
    "options",
    [
        {"repeats": True},
        {"repeats": 11},
        {"max_tokens": 65},
        {"max_tokens": 7},
        {"block_sizes": [1, True]},
        {"block_sizes": [1, 32]},
        {"block_sizes": [1, 2, 2]},
        {"prompt_ids": [True]},
        {"prompt_ids": [-1]},
        {"prompt_ids": [17]},
        {"prompt_ids": []},
        {"prompt_ids": [1] * 128},
    ],
)
def test_oracle_rejects_invalid_or_unbounded_options(options):
    arguments = {"prompt_ids": [1, 2], "max_tokens": 8, **options}
    with pytest.raises(ValueError):
        benchmark(FakeRuntime(), **arguments)
