import pytest
import torch
from reap.muse_attention import AppendCache, bounded_attention, grouped_attention


def reference(q, k, v, allowed):
    repeats = q.shape[1] // k.shape[1]
    k, v = [x.repeat_interleave(repeats, dim=1) for x in (k, v)]
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    probs = torch.softmax(
        scores.masked_fill(~allowed, torch.finfo(q.dtype).min),
        dim=-1,
        dtype=torch.float32,
    ).to(q.dtype)
    return probs @ v


@pytest.mark.parametrize("window", [None, 1, 4])
@pytest.mark.parametrize("injected", [False, True])
def test_bounded_cache_matches_causal_reference_across_appends(window, injected):
    torch.manual_seed(17)
    q = torch.randn(1, 16, 17, 16).half() * 0.2
    k = torch.randn(1, 2, 17, 16).half() * 0.2
    v = torch.randn_like(k)
    positions = torch.arange(17)
    allowed = positions[None, :] <= positions[:, None]
    if window:
        allowed &= positions[None, :] > positions[:, None] - window
    expected = reference(q, k, v, allowed)
    cache = AppendCache()
    calls = []

    def op(query, key, value, **kw):
        length = kw["actual_seq_lengths"][0]
        calls.append((query.shape[-2], length, key.shape[-2]))
        assert (
            query.shape[-2] * 8 <= 32 and key.is_contiguous() and value.is_contiguous()
        )
        return grouped_attention(query, key, value, **kw)

    parts = []
    start = 0
    pointer = None
    for count in (5, 1, 3, 5, 3):
        parts.append(
            bounded_attention(
                q[:, :, start : start + count],
                k[:, :, start : start + count],
                v[:, :, start : start + count],
                start=start,
                cache=cache,
                window=window,
                max_context=32,
                prefill_chunk=5,
                op=op if injected else None,
            )
        )
        if pointer is not None:
            assert cache.key.data_ptr() == pointer
        pointer = cache.key.data_ptr()
        start += count
        assert cache.length == (start if window is None else min(start, window - 1))
        # Excluded capacity must never affect valid outputs.
        cache.key[:, :, cache.length :].fill_(float("nan"))
        cache.value[:, :, cache.length :].fill_(float("nan"))
    # Chunk shapes change FP16 accumulation rounding by up to one ULP.
    torch.testing.assert_close(torch.cat(parts, dim=-2), expected, rtol=1e-3, atol=1e-5)
    assert (not injected or len(calls) > 5) and cache.seen == 17
    with pytest.raises(ValueError, match="contiguous"):
        bounded_attention(
            q[:, :, :1],
            k[:, :, :1],
            v[:, :, :1],
            start=0,
            cache=cache,
            window=window,
            max_context=32,
            op=op,
        )


def test_bounded_context_guard_precedes_operator():
    q = torch.zeros(1, 16, 1, 16).half()
    k = torch.zeros(1, 2, 1, 16).half()
    with pytest.raises(ValueError, match="context"):
        bounded_attention(
            q, k, k, start=65536, max_context=65536, op=lambda *a, **kw: None
        )
