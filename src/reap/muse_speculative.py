"""Measure a perfect-draft verification ceiling on an exclusively owned runtime.

This is a callable benchmark, not a resident replacement patch. The caller must
hold the server's request lock (or own an offline runtime), reset its request and
prefix caches, and persist the returned evidence. It does not install hooks,
load weights, generate real draft proposals, or implement rejection rollback.
"""

import statistics
import time

import torch

from reap.muse_prefix import PrefixSnapshot


def _synchronize(runtime):
    devices = {str(layer.device) for layer in runtime.layers}
    for layer in runtime.layers:
        if hasattr(layer, "tensor_experts"):
            devices.update(str(rank.device) for rank in layer.tensor_experts.ranks)
    if hasattr(runtime, "output_head"):
        devices.update(str(rank.device) for rank in runtime.output_head.ranks)
    for name in sorted(devices):
        device = torch.device(name)
        if device.type == "npu":
            torch.npu.synchronize(device)
        elif device.type == "cuda":
            torch.cuda.synchronize(device)


def _difference(actual, reference):
    difference = (actual.float() - reference.float()).abs()
    expected = reference.argmax(-1)
    predicted = actual.argmax(-1)
    mismatches = (predicted != expected).nonzero(as_tuple=False).flatten().tolist()
    return {
        "all_logits_exact": torch.equal(actual, reference),
        "max_abs_logit_difference": float(difference.max()),
        "mean_abs_logit_difference": float(difference.mean()),
        "argmax_exact": not mismatches,
        "argmax_mismatch_positions": mismatches,
        "predicted_token_ids": predicted.tolist(),
    }


@torch.inference_mode()
def benchmark(
    runtime, prompt_ids, *, block_sizes=(1, 2, 4, 8), repeats=3, max_tokens=16
):
    """Compare fixed oracle input tokens sequentially and in causal blocks.

    B input rows model one pending token plus K=B-1 speculative proposals. Each
    forward returns all B next-token logits. B/T is only a perfect-acceptance,
    zero-draft-cost ceiling, not achieved speculative decoding throughput.

    Restore the exact same prompt boundary outside every timed interval. Full
    attention snapshots share the append-only allocation; only live sliding KV
    is copied. Changing matrix shapes can change floating-point results, so the
    report records every argmax mismatch and full-logit differences explicitly.
    """
    if not isinstance(prompt_ids, (list, tuple)) or not isinstance(
        block_sizes, (list, tuple)
    ):
        raise ValueError("oracle prompt and block sizes must be lists or tuples")
    prompt_ids, block_sizes = list(prompt_ids), tuple(block_sizes)
    steps = max_tokens
    if (
        not prompt_ids
        or any(
            type(token) is not int or not 0 <= token < runtime.c.vocab_size
            for token in prompt_ids
        )
        or not block_sizes
        or any(type(size) is not int for size in block_sizes)
        or len(set(block_sizes)) != len(block_sizes)
        or 1 not in block_sizes
        or type(steps) is not int
        or not 1 <= steps <= 64
        or type(repeats) is not int
        or not 1 <= repeats <= 10
        or any(
            type(size) is not int
            or size < 1
            or size > min(runtime.prefill_chunk, 16)
            or steps % size
            for size in block_sizes
        )
        or len(prompt_ids) + steps > runtime.max_context
    ):
        raise ValueError("invalid oracle prompt, block sizes, repetitions, or context")
    if not runtime.bounded_attention:
        raise ValueError("oracle probe requires bounded AppendCache attention")
    if (
        runtime.seen
        or getattr(runtime, "prefix_snapshot", None) is not None
        or any(cache.key is not None or cache.seen for cache in runtime.cache)
    ):
        raise ValueError(
            "oracle probe requires an idle, reset runtime and exclusive ownership"
        )

    records = []
    snapshot = None

    def restore():
        # Drop the preceding active sliding allocation before restoring another.
        runtime.cache = []
        runtime.cache = snapshot.restore()
        runtime.seen = len(prompt_ids)

    def evaluate(tape, size):
        chunks = []
        for offset in range(0, len(tape), size):
            # Muse already returns owned CPU logits. Retain these tensors until
            # timing finishes, avoiding additional comparisons/copies in the loop.
            chunks.append(
                runtime.forward(tape[offset : offset + size], all_logits=True)
            )
        return chunks

    try:
        prefill_started = time.perf_counter()
        for offset in range(0, len(prompt_ids), runtime.prefill_chunk):
            logits = runtime.forward(
                prompt_ids[offset : offset + runtime.prefill_chunk]
            )
        _synchronize(runtime)
        prefill_seconds = time.perf_counter() - prefill_started
        snapshot = PrefixSnapshot.capture(
            prompt_ids, logits, runtime.cache, runtime.c.layer_types
        )

        # Construct a teacher-generated tape without charging its cost to the
        # fictional draft. Its measured sequential passes are bracketed below.
        tape, reference_chunks = [], []
        for _ in range(steps):
            token = int(logits[0, -1].argmax())
            tape.append(token)
            logits = runtime.forward([token], all_logits=True)
            reference_chunks.append(logits)
        _synchronize(runtime)
        reference = torch.cat(reference_chunks, dim=1)[0].cpu().clone()
        del reference_chunks

        arms = [("sequential_before", 1)]
        arms.extend((f"block_{size}", size) for size in block_sizes if size != 1)
        arms.append(("sequential_after", 1))
        for arm, size in arms:
            restore()
            evaluate(tape, size)  # Shape-specific warmup, excluded from timing.
            _synchronize(runtime)
            for repeat in range(repeats):
                restore()
                _synchronize(runtime)
                started = time.perf_counter()
                chunks = evaluate(tape, size)
                _synchronize(runtime)
                elapsed = time.perf_counter() - started
                actual = torch.cat(chunks, dim=1)[0].cpu()
                records.append(
                    {
                        "arm": arm,
                        "repeat": repeat,
                        "block_rows": size,
                        "speculative_proposals": size - 1,
                        "verified_input_tokens": steps,
                        "forward_calls": steps // size,
                        "seconds": elapsed,
                        "seconds_per_block": elapsed / (steps // size),
                        "oracle_tokens_per_second": steps / elapsed,
                        **_difference(actual, reference),
                    }
                )
                if runtime.seen != len(prompt_ids) + steps or any(
                    cache.seen != runtime.seen for cache in runtime.cache
                ):
                    raise RuntimeError(
                        "oracle evaluation left inconsistent cache positions"
                    )

        sequential_seconds = statistics.mean(
            statistics.median(r["seconds"] for r in records if r["arm"] == arm)
            for arm in ("sequential_before", "sequential_after")
        )
        summaries = []
        for arm, size in arms:
            samples = [r for r in records if r["arm"] == arm]
            seconds = statistics.median(r["seconds"] for r in samples)
            summaries.append(
                {
                    "arm": arm,
                    "block_rows": size,
                    "median_seconds": seconds,
                    "oracle_speedup_over_bracketed_sequential": sequential_seconds
                    / seconds,
                    "all_logits_exact": all(r["all_logits_exact"] for r in samples),
                    "argmax_exact": all(r["argmax_exact"] for r in samples),
                }
            )
        return {
            "completed": True,
            "kind": "oracle_verification_ceiling",
            "draft_cost_included": False,
            "acceptance_cost_included": False,
            "snapshot_restore_cost_included": False,
            "rejection_rollback_tested": False,
            "prompt_ids": prompt_ids,
            "prompt_tokens": len(prompt_ids),
            "sliding_window": runtime.c.sliding_window,
            "past_sliding_window": len(prompt_ids) >= runtime.c.sliding_window,
            "prefill_seconds": prefill_seconds,
            "input_token_ids": tape,
            "reference_next_token_ids": reference.argmax(-1).tolist(),
            "records": records,
            "summary": summaries,
        }
    finally:
        # The caller supplied an empty runtime. Return it to that state even
        # when a block failed, without preserving any experimental prefix.
        runtime.reset()
        snapshot = None
