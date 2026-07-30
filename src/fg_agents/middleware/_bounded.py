"""A size-bounded, LRU-evicting dict for per-session middleware state.

Middleware that tracks state keyed by ``session_id`` (loop guards, rate-limit
buckets) accumulates one entry per session ever seen. On a long-lived server
that is an unbounded memory leak. This dict caps the entry count and evicts the
least-recently-used key on overflow. The tracked state is best-effort — a
loop-guard counter, a token bucket — so evicting a long-idle session's entry at
worst resets that state the next time the session is touched, never a
correctness problem.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")

# Far more than any realistic concurrent-session count, small enough that the
# retained state can never dominate memory.
DEFAULT_MAX_ENTRIES = 100_000


class BoundedLRU(OrderedDict[_K, _V]):
    """An ``OrderedDict`` that keeps at most ``max_entries`` keys, evicting the
    least-recently-used on overflow. Reads and writes mark a key most-recent."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        super().__init__()
        self._max = max(1, max_entries)

    def __getitem__(self, key: _K) -> _V:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):  # type: ignore[override]
        # dict.get bypasses __getitem__, so override it too — otherwise a
        # caller that reads a hot key only via .get() (e.g. the rate limiter,
        # which mutates the bucket in place) never refreshes its recency and a
        # busy key can be evicted ahead of an idle one.
        if key in self:
            return self[key]  # refreshes recency via __getitem__
        return default

    def __setitem__(self, key: _K, value: _V) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._max:
            self.popitem(last=False)  # drop the least-recently-used
