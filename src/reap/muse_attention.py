"""Bounded 310P fused GQA with in-place FP16 KV appends and causal chunking."""

from dataclasses import dataclass
import torch


@dataclass
class AppendCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    seen: int = 0
    length: int = 0


def flash_attention(
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
    """Process at most 32 GQA query rows per call, as required by 300I Duo.

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
        raise ValueError("fused attention requires single-sequence BNSD tensors")
    count, heads, kv_heads, dim = q.shape[-2], q.shape[1], k.shape[1], q.shape[-1]
    if (
        count < 1
        or count > prefill_chunk
        or q.shape[-2:] != k.shape[-2:]
        or heads % kv_heads
        or dim > 128
    ):
        raise ValueError("unsupported fused attention geometry")
    if start < 0 or start + count > max_context or max_context > 65536:
        raise ValueError("fused attention context exceeded")
    if window is not None and window < 1:
        raise ValueError("invalid sliding window")
    if op is None:
        import torch_npu

        op = torch_npu.npu_incre_flash_attention
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
    # No Hq/Hkv replication and no materialized Hq*Q*K score/softmax tensors.
    repeats = heads // kv_heads
    query_chunk = min(16, 32 // repeats)
    if query_chunk < 1:
        raise ValueError("GQA ratio exceeds the 310P fused-kernel limit")
    outputs = []
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
        keep = min(length, window - 1) if window is not None else length
        if keep < length and keep:
            cache.key[:, :, :keep].copy_(
                cache.key[:, :, length - keep : length].clone()
            )
            cache.value[:, :, :keep].copy_(
                cache.value[:, :, length - keep : length].clone()
            )
        cache.length = keep
        cache.seen = start + count
    return torch.cat(outputs, dim=-2)
