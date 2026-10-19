"""Opt-in KV transactions for one causal verification block.

The caller must exclusively own the runtime, including its request lock. This
module does not make concurrent generation safe or intercept runtime.reset().
An abort restores logical metadata; speculative suffix bytes remain invalid.
Sliding compaction is deferred until commit, so a failed layer never overwrites
the committed prefix. Commit backups are bounded by sliding windows, not full
context capacity. Fatal device errors still require resetting the runtime.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import threading

from reap import muse_attention
from reap.muse_attention import AppendCache


@dataclass
class _Entry:
    cache: AppendCache
    key: object
    value: object
    seen: int
    length: int
    window: int | None = None
    started: bool = False
    in_flight: bool = False
    completed: bool = False


class SpeculativeCacheTransaction:
    """Verify B inputs once, then retain exactly A inputs, where 0 <= A <= B.

    Example::

        with SpeculativeCacheTransaction(
            runtime.cache, proposal_tokens=len(ids), runtime=runtime
        ) as transaction:
            logits = runtime.forward(ids, all_logits=True)
            transaction.commit(accepted_input_count)

    Leaving the context without commit, including after a partial-layer failure,
    aborts. With ``runtime=None`` only caches are managed; the caller must manage
    its own outer position. B includes any pending anchor input, not only draft
    proposals. A new transaction is needed for every verification block.
    """

    def __init__(self, caches, *, proposal_tokens, runtime=None):
        if type(proposal_tokens) is not int or proposal_tokens < 1:
            raise ValueError("proposal_tokens must be a positive integer")
        self.caches = tuple(caches)
        if (
            not self.caches
            or any(not isinstance(cache, AppendCache) for cache in self.caches)
            or len({id(cache) for cache in self.caches}) != len(self.caches)
        ):
            raise ValueError("transaction requires distinct AppendCache objects")
        self.proposal_tokens = proposal_tokens
        self.runtime = runtime
        self.state = "new"
        self._entries = {}
        self._thread = None
        self.start = None

    def __enter__(self):
        if self.state != "new":
            raise RuntimeError("transaction cannot be entered twice")
        if not getattr(
            muse_attention.bounded_attention,
            "supports_speculative_cache_transactions",
            False,
        ):
            raise RuntimeError(
                "installed attention does not support cache transactions"
            )
        start = self.caches[0].seen
        for cache in self.caches:
            if cache._speculative_transaction is not None:
                raise RuntimeError("nested or pending cache transaction")
            if (
                cache.seen != start
                or cache.seen < 0
                or not 0 <= cache.length <= cache.seen
                or (cache.key is None) != (cache.value is None)
                or (cache.key is None and cache.length != 0)
                or (
                    cache.key is not None
                    and (
                        cache.key.shape != cache.value.shape
                        or cache.key.ndim != 4
                        or cache.length > cache.key.shape[-2]
                    )
                )
            ):
                raise ValueError("inconsistent cache at transaction boundary")
        self.start = start
        if self.runtime is not None:
            self._check_runtime(start)
            if self.proposal_tokens > self.runtime.prefill_chunk:
                raise ValueError("verification block exceeds prefill chunk")
            if start + self.proposal_tokens > self.runtime.max_context:
                raise ValueError("verification block exceeds context capacity")
        self._thread = threading.get_ident()
        self._entries = {
            id(cache): _Entry(cache, cache.key, cache.value, cache.seen, cache.length)
            for cache in self.caches
        }
        for cache in self.caches:
            cache._speculative_transaction = self
        self.state = "active"
        return self

    def _check_runtime(self, expected_seen):
        if (
            len(self.runtime.cache) != len(self.caches)
            or any(a is not b for a, b in zip(self.runtime.cache, self.caches))
            or self.runtime.seen != expected_seen
        ):
            raise RuntimeError("runtime cache or position changed outside transaction")

    def _check_active(self):
        if self.state != "active":
            raise RuntimeError("transaction is not active")
        if threading.get_ident() != self._thread:
            raise RuntimeError("transaction requires exclusive same-thread ownership")

    @contextmanager
    def append(self, cache, *, start, count, window):
        """Internal attention hook; mark exceptions without losing rollback data."""
        self._check_active()
        entry = self._entries.get(id(cache))
        if entry is None or cache._speculative_transaction is not self:
            raise RuntimeError("cache does not belong to this transaction")
        if entry.started:
            raise RuntimeError("only one append per cache is allowed before resolution")
        if start != self.start or count != self.proposal_tokens:
            raise ValueError("verification append does not match transaction block")
        if window is not None and entry.length > window - 1:
            raise ValueError("sliding cache was not compacted at transaction boundary")
        entry.window = window
        entry.started = entry.in_flight = True
        try:
            yield
            if cache.seen != start + count or cache.length != entry.length + count:
                raise RuntimeError("attention did not preserve speculative KV prefix")
            entry.completed = True
        finally:
            entry.in_flight = False

    def _require_idle(self):
        self._check_active()
        if any(entry.in_flight for entry in self._entries.values()):
            raise RuntimeError("cannot resolve transaction during attention")

    def _restore_metadata(self):
        for entry in self._entries.values():
            cache = entry.cache
            cache.key, cache.value = entry.key, entry.value
            cache.seen, cache.length = entry.seen, entry.length
        if self.runtime is not None:
            self.runtime.seen = self.start

    def _finish(self, state):
        for cache in self.caches:
            cache._speculative_transaction = None
        self.state = state

    def abort(self):
        """Discard suffixes without copying the full-attention cache."""
        self._require_idle()
        # Replacing a runtime's cache list while pending is unsupported. Do not
        # silently clobber newly installed caches or hide an external reset.
        if self.runtime is not None:
            seen = self.runtime.seen
            if seen not in (self.start, self.start + self.proposal_tokens):
                raise RuntimeError("runtime position changed outside transaction")
            self._check_runtime(seen)
        self._restore_metadata()
        self._finish("aborted")

    def commit(self, accepted_input_count):
        """Retain an accepted prefix, then compact only sliding KV storage."""
        self._require_idle()
        if type(accepted_input_count) is not int or not (
            0 <= accepted_input_count <= self.proposal_tokens
        ):
            raise ValueError("accepted input count must be between zero and B")
        if not all(entry.completed for entry in self._entries.values()):
            raise RuntimeError("cannot commit an incomplete verification forward")
        if self.runtime is not None:
            self._check_runtime(self.start + self.proposal_tokens)
        if accepted_input_count == 0:
            self._restore_metadata()
            self._finish("committed")
            return

        # Stage all overlap-safe sources and original sliding prefixes before
        # modifying any committed bytes. Full-attention banks are never cloned.
        copies = []
        for entry in self._entries.values():
            length = entry.length + accepted_input_count
            keep = min(length, entry.window - 1) if entry.window is not None else length
            if 0 < keep < length:
                cache = entry.cache
                copies.append(
                    (
                        entry,
                        cache.key[:, :, length - keep : length].clone(),
                        cache.value[:, :, length - keep : length].clone(),
                        cache.key[:, :, : entry.length].clone(),
                        cache.value[:, :, : entry.length].clone(),
                    )
                )
        try:
            for entry, key, value, _, _ in copies:
                entry.cache.key[:, :, : key.shape[-2]].copy_(key)
                entry.cache.value[:, :, : value.shape[-2]].copy_(value)
        except Exception:
            # These copies can themselves fail after a fatal device error;
            # callers must reset the runtime in that case, not reuse its cache.
            for entry, _, _, key, value in copies:
                entry.cache.key[:, :, : entry.length].copy_(key)
                entry.cache.value[:, :, : entry.length].copy_(value)
            self.abort()
            raise
        for entry in self._entries.values():
            length = entry.length + accepted_input_count
            entry.cache.length = (
                min(length, entry.window - 1) if entry.window is not None else length
            )
            entry.cache.seen = self.start + accepted_input_count
        if self.runtime is not None:
            self.runtime.seen = self.start + accepted_input_count
        self._finish("committed")

    def __exit__(self, exc_type, exc, traceback):
        if self.state == "active":
            self.abort()
        return False
