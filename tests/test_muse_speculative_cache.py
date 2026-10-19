"""CPU causal attention and rollback tests; no model weights or device needed."""

import pytest
import torch

from reap.muse_attention import AppendCache, bounded_attention, grouped_attention
from reap.muse_speculative_cache import SpeculativeCacheTransaction


class Runtime:
    prefill_chunk = 5
    max_context = 96
    windows = (None, 4, 1)

    def __init__(self):
        self.cache = [AppendCache() for _ in self.windows]
        self.seen = 0
        self.fail = None
        self.operator = None
        generator = torch.Generator().manual_seed(188)
        self.embeddings = torch.randn(100, 4, generator=generator) * 0.1

    def forward(self, ids):
        x = self.embeddings[ids].reshape(1, 1, len(ids), 4)
        for index, (cache, window) in enumerate(zip(self.cache, self.windows)):

            def op(*args, **kwargs):
                if self.fail == (index, "operator"):
                    raise RuntimeError("injected attention failure")
                return grouped_attention(*args, **kwargs)

            x = x + bounded_attention(
                x,
                x * 0.5,
                x * 0.7,
                start=self.seen,
                cache=cache,
                window=window,
                max_context=self.max_context,
                prefill_chunk=self.prefill_chunk,
                op=self.operator or op,
            )
            if self.fail == (index, "after"):
                raise RuntimeError("injected layer failure")
        if self.fail == "head":
            raise RuntimeError("injected output head failure")
        self.seen += len(ids)
        return x


def prefill(runtime, length):
    for start in range(0, length, runtime.prefill_chunk):
        runtime.forward(list(range(start, min(length, start + runtime.prefill_chunk))))


def capture(runtime):
    return [
        (
            cache.seen,
            cache.length,
            cache.key,
            cache.value,
            cache.key[:, :, : cache.length].clone() if cache.key is not None else None,
            cache.value[:, :, : cache.length].clone()
            if cache.value is not None
            else None,
        )
        for cache in runtime.cache
    ]


def assert_restored(runtime, saved):
    for cache, (seen, length, key, value, valid_key, valid_value) in zip(
        runtime.cache, saved
    ):
        assert cache.seen == seen and cache.length == length
        assert cache.key is key and cache.value is value
        assert cache._speculative_transaction is None
        if key is not None:
            torch.testing.assert_close(
                cache.key[:, :, :length], valid_key, rtol=0, atol=0
            )
            torch.testing.assert_close(
                cache.value[:, :, :length], valid_value, rtol=0, atol=0
            )
    assert runtime.seen == saved[0][0]


@pytest.mark.parametrize("prefix", [0, 2, 13])
@pytest.mark.parametrize("accepted", range(6))
def test_every_accepted_prefix_matches_sequential_after_rollover(prefix, accepted):
    runtime, reference = Runtime(), Runtime()
    prefill(runtime, prefix)
    prefill(reference, prefix)
    saved = capture(runtime)
    tape = list(range(50, 55))
    with SpeculativeCacheTransaction(
        runtime.cache, proposal_tokens=5, runtime=runtime
    ) as tx:
        actual = runtime.forward(tape)
        # Sliding prefixes remain intact until commit, even past their window.
        for cache, original in zip(runtime.cache, saved):
            if original[2] is not None:
                assert cache.key is original[2]
                torch.testing.assert_close(
                    cache.key[:, :, : original[1]], original[4], rtol=0, atol=0
                )
            assert cache.length == original[1] + 5
        tx.commit(accepted)
    sequential = [reference.forward([token]) for token in tape[:accepted]]
    if accepted:
        torch.testing.assert_close(
            actual[:, :, :accepted], torch.cat(sequential, dim=-2)
        )
    if accepted == 0:
        assert_restored(runtime, saved)
    for cache, expected, original in zip(runtime.cache, reference.cache, saved):
        assert cache.seen == expected.seen == prefix + accepted
        assert cache.length == expected.length
        if original[2] is not None:
            assert cache.key is original[2] and cache.value is original[3]
        if cache.key is not None:
            torch.testing.assert_close(
                cache.key[:, :, : cache.length], expected.key[:, :, : expected.length]
            )
            torch.testing.assert_close(
                cache.value[:, :, : cache.length],
                expected.value[:, :, : expected.length],
            )
    torch.testing.assert_close(runtime.forward([91]), reference.forward([91]))


@pytest.mark.parametrize(
    "failure", [(0, "operator"), (1, "operator"), (1, "after"), (2, "after"), "head"]
)
def test_partial_layer_or_head_failure_aborts_without_prefix_damage(failure):
    runtime, reference = Runtime(), Runtime()
    prefill(runtime, 13)
    prefill(reference, 13)
    saved = capture(runtime)
    runtime.fail = failure
    with (
        pytest.raises(RuntimeError, match="injected"),
        SpeculativeCacheTransaction(runtime.cache, proposal_tokens=5, runtime=runtime),
    ):
        runtime.forward([50, 51, 52, 53, 54])
    assert_restored(runtime, saved)
    runtime.fail = None
    torch.testing.assert_close(runtime.forward([91]), reference.forward([91]))


def test_default_exit_and_explicit_abort_restore_outer_seen():
    runtime = Runtime()
    prefill(runtime, 13)
    saved = capture(runtime)
    for explicit in (False, True):
        with SpeculativeCacheTransaction(
            runtime.cache, proposal_tokens=3, runtime=runtime
        ) as tx:
            runtime.forward([50, 51, 52])
            assert runtime.seen == 16
            if explicit:
                tx.abort()
        assert_restored(runtime, saved)


def test_pending_transaction_rejects_nested_and_second_forward():
    runtime = Runtime()
    prefill(runtime, 13)
    with SpeculativeCacheTransaction(
        runtime.cache, proposal_tokens=3, runtime=runtime
    ) as tx:
        with (
            pytest.raises(RuntimeError, match="nested or pending"),
            SpeculativeCacheTransaction(
                runtime.cache, proposal_tokens=3, runtime=runtime
            ),
        ):
            pass
        runtime.forward([50, 51, 52])
        pending = capture(runtime)
        with pytest.raises(RuntimeError, match="only one append"):
            runtime.forward([53, 54, 55])
        for cache, saved in zip(runtime.cache, pending):
            assert (cache.seen, cache.length) == saved[:2]
            torch.testing.assert_close(
                cache.key[:, :, : cache.length], saved[4], rtol=0, atol=0
            )
        tx.commit(2)
    assert runtime.seen == 15


def test_rejects_resolution_during_attention_and_incomplete_commit():
    runtime = Runtime()
    prefill(runtime, 13)
    with SpeculativeCacheTransaction(
        runtime.cache, proposal_tokens=3, runtime=runtime
    ) as tx:
        with pytest.raises(RuntimeError, match="incomplete"):
            tx.commit(1)

        def op(*args, **kwargs):
            with pytest.raises(RuntimeError, match="during attention"):
                tx.commit(1)
            with pytest.raises(RuntimeError, match="during attention"):
                tx.abort()
            return grouped_attention(*args, **kwargs)

        runtime.operator = op
        runtime.forward([50, 51, 52])
        tx.commit(1)
    assert runtime.seen == 14


@pytest.mark.parametrize("accepted", [-1, 6, 1.0, True])
def test_invalid_commit_count_aborts(accepted):
    runtime = Runtime()
    prefill(runtime, 13)
    saved = capture(runtime)
    with (
        pytest.raises(ValueError, match="accepted input count"),
        SpeculativeCacheTransaction(
            runtime.cache, proposal_tokens=5, runtime=runtime
        ) as tx,
    ):
        runtime.forward([50, 51, 52, 53, 54])
        tx.commit(accepted)
    assert_restored(runtime, saved)


def test_no_full_context_clone_and_ordinary_path_unchanged(monkeypatch):
    runtime = Runtime()
    prefill(runtime, 13)
    sizes = []
    original = torch.Tensor.clone

    def record(tensor, *args, **kwargs):
        sizes.append(tensor.shape[-2])
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", record)
    with SpeculativeCacheTransaction(
        runtime.cache, proposal_tokens=5, runtime=runtime
    ) as tx:
        runtime.forward([50, 51, 52, 53, 54])
        assert sizes == []  # No transaction copies before a decision.
        tx.commit(3)
    assert sizes and max(sizes) <= 3  # Window - 1; never the full cache.
    runtime.forward([91])
    assert [cache.length for cache in runtime.cache] == [17, 3, 0]
    assert all(cache._speculative_transaction is None for cache in runtime.cache)


def test_failed_compaction_copy_restores_all_sliding_prefixes(monkeypatch):
    runtime = Runtime()
    runtime.windows = (None, 4, 4)
    prefill(runtime, 13)
    saved = capture(runtime)
    original = torch.Tensor.copy_
    calls = 0

    def fail_once(tensor, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected compaction copy failure")
        return original(tensor, *args, **kwargs)

    with SpeculativeCacheTransaction(
        runtime.cache, proposal_tokens=5, runtime=runtime
    ) as tx:
        runtime.forward([50, 51, 52, 53, 54])
        monkeypatch.setattr(torch.Tensor, "copy_", fail_once)
        with pytest.raises(RuntimeError, match="compaction copy"):
            tx.commit(3)
    assert tx.state == "aborted"
    assert_restored(runtime, saved)


def test_runtime_capacity_and_append_size_guards_leave_caches_untouched():
    runtime = Runtime()
    prefill(runtime, 13)
    saved = capture(runtime)
    with (
        pytest.raises(ValueError, match="prefill chunk"),
        SpeculativeCacheTransaction(runtime.cache, proposal_tokens=6, runtime=runtime),
    ):
        pass
    with (
        pytest.raises(ValueError, match="match transaction block"),
        SpeculativeCacheTransaction(runtime.cache, proposal_tokens=3, runtime=runtime),
    ):
        runtime.forward([50])
    assert_restored(runtime, saved)


def test_cache_only_transaction_requires_no_runtime():
    cache = AppendCache()
    x = torch.ones(1, 1, 3, 4)
    with SpeculativeCacheTransaction([cache], proposal_tokens=3) as tx:
        bounded_attention(x, x, x, start=0, cache=cache, window=2, prefill_chunk=3)
        tx.commit(2)
    assert cache.seen == 2 and cache.length == 1


def test_unadapted_resident_attention_is_rejected_before_mutation(monkeypatch):
    from reap import muse_attention

    runtime = Runtime()
    prefill(runtime, 13)
    saved = capture(runtime)
    monkeypatch.setattr(
        muse_attention, "bounded_attention", lambda *args, **kwargs: None
    )
    with (
        pytest.raises(RuntimeError, match="installed attention"),
        SpeculativeCacheTransaction(runtime.cache, proposal_tokens=3, runtime=runtime),
    ):
        pass
    assert_restored(runtime, saved)
