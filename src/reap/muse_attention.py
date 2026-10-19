"""Bounded grouped attention with in-place FP16 KV appends and causal chunking."""

from dataclasses import dataclass, field
import torch


@dataclass
class AppendCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    seen: int = 0
    length: int = 0
    _speculative_transaction: object | None = field(
        default=None, repr=False, compare=False
    )


def bounded_attention(
    q,
    k,
    v,
    *,
    start,
    cache=None,
    window=None,
    max_context=65536,
    prefill_chunk=64,
    op=None,
):
    """Append ordinary KV, or defer compaction inside an explicit transaction."""
    kwargs = dict(
        start=start,
        cache=cache,
        window=window,
        max_context=max_context,
        prefill_chunk=prefill_chunk,
        op=op,
    )
    transaction = None if cache is None else cache._speculative_transaction
    if transaction is None:
        return _bounded_attention(q, k, v, **kwargs)
    with transaction.append(cache, start=start, count=q.shape[-2], window=window):
        return _bounded_attention(q, k, v, **kwargs)


# Resident replacements must preserve the hooks and delayed compaction before
# advertising this capability. The transaction checks the currently installed
# module function, so an older replacement fails before it can overwrite KV.
bounded_attention.supports_speculative_cache_transactions = True


def _bounded_attention(
    q,
    k,
    v,
    *,
    start,
    cache=None,
    window=None,
    max_context=65536,
    prefill_chunk=64,
    op=None,
):
    """Bound grouped attention workspace while batching short-prefix queries.

    Full-attention storage is allocated once per request, then appended in place.
    actual_seq_lengths excludes uninitialized capacity. Sliding caches retain
    only the previous window minus one tokens, plus space for one input chunk.
    """
    if (
        q.ndim != 4
        or k.ndim != 4
        or k.shape != v.shape
        or q.shape[0] != 1
        or k.shape[0] != 1
    ):
        raise ValueError("bounded attention requires single-sequence BNSD tensors")
    count, heads, kv_heads, dim = q.shape[-2], q.shape[1], k.shape[1], q.shape[-1]
    if (
        count < 1
        or count > prefill_chunk
        or q.shape[-2:] != k.shape[-2:]
        or heads % kv_heads
        or dim > 128
    ):
        raise ValueError("unsupported bounded attention geometry")
    if start < 0 or start + count > max_context:
        raise ValueError("bounded attention context exceeded")
    if window is not None and window < 1:
        raise ValueError("invalid sliding window")
    use_grouped = op is None
    if op is None:
        op = grouped_attention
    previous = 0
    if cache is not None:
        if cache.seen != start:
            raise ValueError("cache positions are not contiguous")
        capacity = (
            max_context
            if window is None
            else min(max_context, window - 1 + prefill_chunk)
        )
        if cache.key is None:
            cache.key = torch.empty(
                (1, kv_heads, capacity, dim), dtype=k.dtype, device=k.device
            )
            cache.value = torch.empty_like(cache.key)
        previous = cache.length
        length = previous + count
        if length > cache.key.shape[-2]:
            raise ValueError("KV cache capacity exceeded")
        cache.key[:, :, previous:length].copy_(k)
        cache.value[:, :, previous:length].copy_(v)
        key, value = cache.key, cache.value
    else:
        length = count
        key, value = k.contiguous(), v.contiguous()
    # Copy each valid KV prefix at most once, without Hq/Hkv replication.
    # Score/softmax memory is bounded by the small query subchunk.
    if use_grouped:
        key = key[:, :, :length].contiguous()
        value = value[:, :, :length].contiguous()
    repeats = heads // kv_heads
    query_chunk = min(16, 32 // repeats)
    if query_chunk < 1:
        raise ValueError("GQA ratio exceeds the bounded score budget")
    if use_grouped:
        # At 96K, retain the proven four-query Muse tile. Short prefixes can
        # batch more queries within the same score-element budget, avoiding
        # dozens of tiny BMM/mask/softmax launches per prefill layer. Cap the
        # budget independently of the configured context and keep injected
        # operators on their existing, conservative geometry.
        score_budget = 32 * min(max_context, 98304)
        query_chunk = min(prefill_chunk, max(1, score_budget // (repeats * length)))
    outputs = []
    if count == 1 and use_grouped:
        # The cache retains only valid past positions (at most window-1), and
        # this append adds exactly the current position. Every key is allowed;
        # avoid constructing and applying an all-false decode mask.
        outputs.append(
            op(
                q.contiguous(),
                key,
                value,
                atten_mask=None,
                actual_seq_lengths=[length],
                num_heads=heads,
                num_key_value_heads=kv_heads,
                scale_value=dim**-0.5,
            )
        )
    else:
        key_positions = torch.arange(key.shape[-2], device=q.device) + start - previous
        for first in range(0, count, query_chunk):
            end = min(count, first + query_chunk)
            positions = torch.arange(start + first, start + end, device=q.device)
            allowed = key_positions[None, :] <= positions[:, None]
            if window is not None:
                allowed &= key_positions[None, :] > positions[:, None] - window
            # The mask covers the physical capacity; actual_seq_lengths bounds the
            # valid prefix, including when no cache is supplied.
            out = op(
                q[:, :, first:end].contiguous(),
                key,
                value,
                atten_mask=(~allowed)[None, None].contiguous(),
                actual_seq_lengths=[length],
                num_heads=heads,
                num_key_value_heads=kv_heads,
                input_layout="BNSD",
                scale_value=dim**-0.5,
                inner_precise=0,
            )
            outputs.append(out)
    if cache is not None:
        keep = (
            min(length, window - 1)
            if window is not None and cache._speculative_transaction is None
            else length
        )
        if keep < length and keep:
            cache.key[:, :, :keep].copy_(
                cache.key[:, :, length - keep : length].clone()
            )
            cache.value[:, :, :keep].copy_(
                cache.value[:, :, length - keep : length].clone()
            )
        cache.length = keep
        cache.seen = start + count
    return outputs[0] if count == 1 and use_grouped else torch.cat(outputs, dim=-2)


def grouped_attention(
    query,
    key,
    value,
    *,
    atten_mask,
    actual_seq_lengths,
    num_heads,
    num_key_value_heads,
    scale_value,
    **_,
):
    """Evaluate GQA without repeated KV heads or vendor fused-attention kernels."""
    length = actual_seq_lengths[0]
    count, dim = query.shape[-2:]
    grouped = query.reshape(1, num_key_value_heads, -1, dim)
    scores = (grouped @ key[:, :, :length].transpose(-1, -2)).reshape(
        1, num_heads, count, length
    ) * scale_value
    if atten_mask is not None:
        scores = scores.masked_fill(
            atten_mask[..., :length], torch.finfo(query.dtype).min
        )
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return (
        probabilities.reshape(1, num_key_value_heads, -1, length) @ value[:, :, :length]
    ).reshape_as(query)
