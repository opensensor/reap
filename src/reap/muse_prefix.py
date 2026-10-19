"""One reusable prompt prefix with bounded sliding-cache snapshots."""

from dataclasses import dataclass

import torch

from reap.muse_attention import AppendCache


@dataclass
class PrefixSnapshot:
    tokens: tuple
    logits: torch.Tensor
    entries: list

    @classmethod
    def capture(cls, tokens, logits, caches, layer_types):
        entries = []
        for cache, kind in zip(caches, layer_types, strict=True):
            if cache.seen != len(tokens) or cache.key is None:
                raise ValueError("prefix snapshot requires a complete prompt boundary")
            sliding = kind == "sliding_attention"
            # Full-attention prefixes are append-only. Sliding buffers move
            # during decode, so retain only their valid window in a separate
            # allocation. Never duplicate the full-context cache capacity.
            key, value = cache.key, cache.value
            capacity = key.shape[-2]
            if sliding:
                key = key[:, :, : cache.length].clone()
                value = value[:, :, : cache.length].clone()
            entries.append((key, value, cache.seen, cache.length, capacity, sliding))
        return cls(tuple(tokens), logits, entries)

    def matches(self, tokens, chunk):
        count = len(self.tokens)
        # A partial chunk can be reused only for an identical prompt. Shared
        # prefixes end on the original chunk boundary so GEMM/attention shapes
        # for the suffix remain identical to an uncached request.
        return (
            len(tokens) >= count
            and (count % chunk == 0 or len(tokens) == count)
            and tuple(tokens[:count]) == self.tokens
        )

    def restore(self):
        caches = []
        for key, value, seen, length, capacity, sliding in self.entries:
            if sliding:
                shape = (*key.shape[:-2], capacity, key.shape[-1])
                active_key = torch.empty(shape, dtype=key.dtype, device=key.device)
                active_value = torch.empty_like(active_key)
                active_key[:, :, :length].copy_(key)
                active_value[:, :, :length].copy_(value)
                key, value = active_key, active_value
            caches.append(AppendCache(key, value, seen, length))
        return caches


def snapshot_reserve_bytes(config, index, max_context):
    """Account for old and replacement sliding snapshots briefly coexisting."""
    if config.layer_types[index] != "sliding_attention":
        return 0
    tokens = min(max_context, config.sliding_window - 1)
    return 2 * tokens * 2 * config.num_key_value_heads * config.head_dim * 2
