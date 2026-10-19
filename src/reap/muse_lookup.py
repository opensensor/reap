"""Experimental greedy prompt-lookup decoding on an exclusively owned runtime.

This is not a serving replacement. The caller must supply a reset bounded-KV
runtime and hold its request lock. Drafts come only from prompt/accepted token
history; there is no learned draft model. CPU algorithm tests do not establish
hardware greedy equivalence: batched FP16/INT8 verification can change logits
relative to single-row decoding. Qualify that numerical path before serving.
"""

from collections import OrderedDict, deque
import time

import torch

from reap.muse_attention import AppendCache
from reap.muse_speculative_cache import SpeculativeCacheTransaction


class PromptLookup:
    """Incremental bounded n-gram index; only short suffixes are copied per step."""

    def __init__(self, *, min_ngram=2, max_ngram=4, max_entries=65536):
        if (
            type(min_ngram) is not int
            or type(max_ngram) is not int
            or not 1 <= min_ngram <= max_ngram <= 16
            or type(max_entries) is not int
            or not 1 <= max_entries <= 1_000_000
        ):
            raise ValueError("invalid prompt-lookup index geometry")
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.max_entries = max_entries
        self.history = []
        self.index = OrderedDict()

    def append(self, token):
        # Only index occurrences with at least one known continuation token.
        end = len(self.history)
        self.history.append(token)
        for size in range(self.min_ngram, min(self.max_ngram, end) + 1):
            key = tuple(self.history[end - size : end])
            locations = self.index.get(key)
            if locations is None:
                locations = self.index[key] = deque(maxlen=4)
            else:
                self.index.move_to_end(key)
            locations.append(end)
            if len(self.index) > self.max_entries:
                self.index.popitem(last=False)

    def propose(self, max_tokens):
        if type(max_tokens) is not int or not 0 <= max_tokens <= 7:
            raise ValueError("lookup draft length must be between zero and seven")
        if max_tokens == 0:
            return []
        end = len(self.history)
        for size in range(min(self.max_ngram, end), self.min_ngram - 1, -1):
            locations = self.index.get(tuple(self.history[end - size : end]))
            if not locations:
                continue
            # Prefer a recent occurrence with a full block; otherwise use the
            # longest available continuation among four retained occurrences.
            location = locations[0]
            for candidate in reversed(locations):
                if end - candidate >= max_tokens:
                    location = candidate
                    break
            return self.history[location : min(location + max_tokens, end)]
        return []


def _synchronize_cache_devices(runtime):
    # forward() returns CPU logits, but commit only enqueues device KV copies.
    # Fence those copies so the reported commit cost is not charged to the next
    # verification. This prototype measures the synchronization overhead too.
    devices = {cache.key.device for cache in runtime.cache if cache.key is not None}
    for device in sorted(devices, key=str):
        if device.type == "npu":
            torch.npu.synchronize(device)
        elif device.type == "cuda":
            torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(
    runtime,
    prompt_ids,
    *,
    max_new_tokens=32,
    eos=(),
    max_proposal_tokens=7,
    min_ngram=2,
    max_ngram=4,
    max_index_entries=65536,
):
    """Return tokens and measured draft/verify/commit/fallback evidence.

    Each block contains one already verified anchor plus 1, 3, or 7 proposed
    tokens, matching the qualified B2/B4/B8 verification shapes. Commit the
    anchor and matching proposal prefix. The retained row's
    logits predict the replacement token on rejection or the bonus token on
    full acceptance. EOS/the final output token remains uncached, matching the
    ordinary runtime generator. The runtime is reset if generation fails.
    """
    if (
        not isinstance(prompt_ids, (list, tuple))
        or not prompt_ids
        or any(
            type(token) is not int or not 0 <= token < runtime.c.vocab_size
            for token in prompt_ids
        )
        or type(max_new_tokens) is not int
        or max_new_tokens < 1
        or len(prompt_ids) + max_new_tokens > runtime.max_context
        or type(max_proposal_tokens) is not int
        or not 0 <= max_proposal_tokens <= 7
    ):
        raise ValueError("invalid lookup prompt, generation limit, or draft length")
    eos = frozenset(eos)
    if any(
        type(token) is not int or not 0 <= token < runtime.c.vocab_size for token in eos
    ):
        raise ValueError("invalid EOS token")
    if (
        not runtime.bounded_attention
        or runtime.seen != 0
        or getattr(runtime, "prefix_snapshot", None) is not None
        or not runtime.cache
        or any(
            not isinstance(cache, AppendCache)
            or cache.key is not None
            or cache.value is not None
            or cache.seen != 0
            or cache.length != 0
            or cache._speculative_transaction is not None
            for cache in runtime.cache
        )
    ):
        raise ValueError(
            "lookup generation requires an exclusively owned reset bounded-KV runtime"
        )
    lookup = PromptLookup(
        min_ngram=min_ngram, max_ngram=max_ngram, max_entries=max_index_entries
    )
    stats = dict(
        draft_seconds=0.0,
        verification_seconds=0.0,
        acceptance_seconds=0.0,
        commit_seconds=0.0,
        fallback_seconds=0.0,
        proposed_tokens=0,
        accepted_draft_tokens=0,
        verification_calls=0,
        fallback_calls=0,
        committed_input_tokens=0,
    )
    outputs, steps = [], []
    started = time.perf_counter()
    try:
        for offset in range(0, len(prompt_ids), runtime.prefill_chunk):
            logits = runtime.forward(
                prompt_ids[offset : offset + runtime.prefill_chunk]
            )
        stats["prefill_seconds"] = time.perf_counter() - started
        decode_started = time.perf_counter()
        draft_started = time.perf_counter()
        for token in prompt_ids:
            lookup.append(token)
        stats["draft_seconds"] += time.perf_counter() - draft_started

        while len(outputs) < max_new_tokens:
            anchor = int(logits[0, -1].argmax())
            outputs.append(anchor)
            draft_started = time.perf_counter()
            lookup.append(anchor)
            if anchor in eos or len(outputs) == max_new_tokens:
                stats["draft_seconds"] += time.perf_counter() - draft_started
                break
            # Reserve one final/bonus output slot so the last emitted token is
            # never committed merely to reach the requested length limit.
            budget = min(
                max_proposal_tokens,
                runtime.prefill_chunk - 1,
                max_new_tokens - len(outputs) - 1,
            )
            proposed = lookup.propose(budget)
            size = max((size for size in (1, 3, 7) if size <= len(proposed)), default=0)
            proposed = proposed[:size]
            # Keep the qualified block shape even if a proposal contains EOS.
            # Acceptance stops at EOS and commit excludes it and its suffix.
            stats["draft_seconds"] += time.perf_counter() - draft_started
            block = [anchor, *proposed]
            position = runtime.seen
            if not proposed:
                fallback_started = time.perf_counter()
                logits = runtime.forward(block, all_logits=True)
                elapsed = time.perf_counter() - fallback_started
                stats["fallback_seconds"] += elapsed
                stats["fallback_calls"] += 1
                stats["committed_input_tokens"] += 1
                steps.append(
                    {
                        "start": position,
                        "block_rows": 1,
                        "proposed": [],
                        "matched": 0,
                        "committed_inputs": 1,
                        "fallback_seconds": elapsed,
                    }
                )
                continue

            with SpeculativeCacheTransaction(
                runtime.cache, proposal_tokens=len(block), runtime=runtime
            ) as transaction:
                verify_started = time.perf_counter()
                verified = runtime.forward(block, all_logits=True)
                verify_elapsed = time.perf_counter() - verify_started
                stats["verification_seconds"] += verify_elapsed
                stats["verification_calls"] += 1
                stats["proposed_tokens"] += len(proposed)
                acceptance_started = time.perf_counter()
                predictions = verified[0].argmax(dim=-1).tolist()
                matched = 0
                for index, token in enumerate(proposed):
                    if token != predictions[index]:
                        break
                    matched += 1
                    if token in eos:
                        break
                stopped = matched > 0 and proposed[matched - 1] in eos
                committed = 1 + matched - int(stopped)
                # Do not retain logits conditioned on a rejected suffix.
                logits = verified[:, committed - 1 : committed]
                stats["acceptance_seconds"] += time.perf_counter() - acceptance_started
                commit_started = time.perf_counter()
                transaction.commit(committed)
                _synchronize_cache_devices(runtime)
                commit_elapsed = time.perf_counter() - commit_started
                stats["commit_seconds"] += commit_elapsed
            stats["accepted_draft_tokens"] += matched
            stats["committed_input_tokens"] += committed
            draft_started = time.perf_counter()
            for token in proposed[:matched]:
                outputs.append(token)
                lookup.append(token)
            stats["draft_seconds"] += time.perf_counter() - draft_started
            steps.append(
                {
                    "start": position,
                    "block_rows": len(block),
                    "proposed": proposed,
                    "matched": matched,
                    "committed_inputs": committed,
                    "verification_seconds": verify_elapsed,
                    "commit_seconds": commit_elapsed,
                    "stopped_on_draft_eos": stopped,
                }
            )
            if stopped:
                break

        if runtime.seen != len(prompt_ids) + len(outputs) - 1 or any(
            cache.seen != runtime.seen for cache in runtime.cache
        ):
            raise RuntimeError("lookup generation left inconsistent cache positions")
        stats["decode_seconds"] = time.perf_counter() - decode_started
        stats["total_seconds"] = time.perf_counter() - started
        stats["completion_tokens"] = len(outputs)
        stats["prompt_tokens"] = len(prompt_ids)
        stats["acceptance_rate"] = (
            stats["accepted_draft_tokens"] / stats["proposed_tokens"]
            if stats["proposed_tokens"]
            else None
        )
        return {
            "tokens": outputs,
            "stats": stats,
            "steps": steps,
            "kind": "experimental_greedy_prompt_lookup",
            "supported_block_rows": [1, 2, 4, 8],
            "hardware_greedy_equivalence_proven": False,
        }
    except Exception:
        runtime.reset()
        raise
