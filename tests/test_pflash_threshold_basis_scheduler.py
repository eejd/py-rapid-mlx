# SPDX-License-Identifier: Apache-2.0
"""Scheduler and cache integration for ``--pflash-threshold-basis uncached``.

The pure threshold arithmetic lives in ``test_pflash_threshold_basis.py``.
These tests pin the cache contract: the basis decision uses
``MemoryAwarePrefixCache.probe`` (no LRU / statistics / copy / decompress
side effects). A probe that would suppress compression is confirmed by exactly
one real fetch; if that fetch misses after eviction, admission recomputes and
compresses rather than trusting stale probe metadata.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402

from rapid_mlx.memory_cache import (  # noqa: E402
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    PrefixProbe,
)
from rapid_mlx.pflash import PFlashConfig  # noqa: E402
from rapid_mlx.request import Request, SamplingParams  # noqa: E402
from rapid_mlx.scheduler import (  # noqa: E402
    Scheduler,
    SchedulerConfig,
    _pflash_cached_prefix_tokens,
)


class _Array:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


class _KV:
    """Plain KV layer double: trimmable, no recurrent state."""

    def __init__(self, nbytes: int = 1000) -> None:
        self.keys = _Array(nbytes // 2)
        self.values = _Array(nbytes // 2)
        self.offset = 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        return n


class _DummyTokenizer:
    eos_token_id = None

    def encode(self, prompt):
        return list(prompt) if not isinstance(prompt, str) else [ord(c) for c in prompt]

    def decode(self, token_ids):
        return "".join(chr(t) for t in token_ids)


def _config(**overrides) -> PFlashConfig:
    values = dict(
        mode="auto",
        threshold=1000,
        keep_ratio=0.5,
        min_keep_tokens=16,
        sink_tokens=8,
        tail_tokens=8,
        block_size=8,
        query_window=16,
        stride_blocks=8,
        skip_when_tools=True,
        threshold_basis="uncached",
    )
    values.update(overrides)
    return PFlashConfig(**values).validate()


def _scheduler(pflash_config: PFlashConfig) -> Scheduler:
    return Scheduler(
        model=object(),
        tokenizer=_DummyTokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            pflash_config=pflash_config,
        ),
    )


def _fake_cache(
    matched: int,
    match_type: str = "prefix",
    *,
    fetch_matched: int | None = None,
) -> MagicMock:
    """Cache double with independently configurable probe/fetch reuse."""
    fake = MagicMock()
    fake._entries = {}
    fake._last_match_type = "miss"
    fake.probe.return_value = PrefixProbe(matched, match_type if matched else "miss")

    def _fetch(prompt):
        actual = matched if fetch_matched is None else fetch_matched
        if actual:
            fake._last_match_type = match_type
            return ["cache"], list(prompt[actual:])
        fake._last_match_type = "miss"
        return None, list(prompt)

    fake.fetch.side_effect = _fetch
    return fake


def _request(n: int, **kwargs) -> Request:
    return Request(f"req-{n}", list(range(n)), SamplingParams(max_tokens=4), **kwargs)


# --------------------------------------------------------------------------
# MemoryAwarePrefixCache.probe: no side effects, same answer as fetch
# --------------------------------------------------------------------------


class TestProbe:
    @pytest.fixture
    def cache(self):
        return MemoryAwarePrefixCache(
            MagicMock(), MemoryCacheConfig(max_memory_mb=1, max_entries=10)
        )

    def test_probe_reports_exact_prefix_and_miss_without_touching_state(self, cache):
        cache.store([1, 2, 3], [_KV()])
        cache.store([9, 9, 9, 9], [_KV()])
        before = (
            cache._stats.hits,
            cache._stats.misses,
            cache._stats.tokens_saved,
            list(cache._entries),
            cache._last_match_type,
        )

        assert cache.probe([1, 2, 3]) == PrefixProbe(3, "exact")
        assert cache.probe([1, 2, 3, 4, 5]) == PrefixProbe(3, "prefix")
        assert cache.probe([7, 8]) == PrefixProbe(0, "miss")
        assert cache.probe([]) == PrefixProbe(0, "miss")

        after = (
            cache._stats.hits,
            cache._stats.misses,
            cache._stats.tokens_saved,
            list(cache._entries),
            cache._last_match_type,
        )
        assert after == before  # no hit/miss counted, LRU order and marker untouched

    def test_probe_does_not_consume_or_copy_what_fetch_then_returns(self, cache):
        kv = [_KV()]
        cache.store([1, 2, 3], kv)
        assert cache.probe([1, 2, 3, 4]).matched_tokens == 3
        assert cache._entries[(1, 2, 3)].cache is kv  # not copied by the probe

        result, remaining = cache.fetch([1, 2, 3, 4])
        assert result is not None and result is not kv  # fetch still deep-copies
        assert remaining == [4]
        assert cache._stats.hits == 1
        assert cache._last_match_type == "prefix"

    def test_probe_matches_fetch_on_a_divergent_sequence(self, cache):
        cache.store([1, 2, 3, 4, 5, 10, 11, 12], [_KV()])
        requested = [1, 2, 3, 4, 5, 20, 21]
        probe = cache.probe(requested)
        _, remaining = cache.fetch(requested)
        assert probe.match_type == "lcp"
        assert probe.matched_tokens == len(requested) - len(remaining) == 5

    def test_probe_matches_rewindable_pooling_lcp_without_mutation(self, cache):
        from mlx_lm.models.cache import CacheList, RotatingKVCache

        from rapid_mlx.models.deepseek_v4_cache import PoolingCache

        rotating = RotatingKVCache(max_size=128)
        kv = mx.zeros((1, 1, 8, 4))
        rotating.update_and_fetch(kv, kv)
        pooling = PoolingCache(ratio=4)
        pooled_kv, _, _ = pooling.accumulate_windows(
            mx.zeros((1, 7, 4)), mx.zeros((1, 7, 2)), 0
        )
        pooling.update_and_fetch(pooled_kv)
        cache.store(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [CacheList(rotating, pooling)],
        )
        requested = [1, 2, 3, 4, 5, 6, 20]
        stored_pooling = next(iter(cache._entries.values())).cache[0].caches[1]
        before = (stored_pooling.remainder, stored_pooling.buf_kv)

        probe = cache.probe(requested)

        assert probe == PrefixProbe(6, "lcp")
        assert (stored_pooling.remainder, stored_pooling.buf_kv) == before
        _, remaining = cache.fetch(requested)
        assert remaining == [20]

    def test_probe_rejects_pooling_lcp_beyond_rewind_window(self, cache):
        from mlx_lm.models.cache import CacheList, RotatingKVCache

        from rapid_mlx.models.deepseek_v4_cache import PoolingCache

        rotating = RotatingKVCache(max_size=128)
        kv = mx.zeros((1, 1, 8, 4))
        rotating.update_and_fetch(kv, kv)
        pooling = PoolingCache(ratio=4)
        pooled_kv, _, _ = pooling.accumulate_windows(
            mx.zeros((1, 7, 4)), mx.zeros((1, 7, 2)), 0
        )
        pooling.update_and_fetch(pooled_kv)
        cache.store(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [CacheList(rotating, pooling)],
        )
        requested = [1, 20, 21]

        assert cache.probe(requested) == PrefixProbe(0, "miss")
        result, remaining = cache.fetch(requested)
        assert result is None
        assert remaining == requested


# --------------------------------------------------------------------------
# Scheduler: the basis decision probes, admission fetches once
# --------------------------------------------------------------------------


class TestSchedulerUncachedBasis:
    def test_warm_hit_skips_compression_and_fetches_exactly_once(self):
        scheduler = _scheduler(_config())
        fake = _fake_cache(matched=1500)
        scheduler.memory_aware_cache = fake
        request = _request(2000)

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is False
        assert request.pflash_metadata["reason"] == "cached_prefix"
        assert request.pflash_metadata["cached_prefix_tokens"] == 1500
        fake.probe.assert_called_once()
        fake.fetch.assert_called_once()
        assert request.cached_tokens == 1500
        assert request.cache_hit_type == "prefix"
        assert request.prompt_token_ids == list(range(2000))

    def test_miss_compresses_and_never_fetches(self):
        scheduler = _scheduler(_config())
        fake = _fake_cache(matched=0)
        scheduler.memory_aware_cache = fake
        request = _request(2000)

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is True
        assert request.pflash_metadata["cached_prefix_tokens"] == 0
        fake.probe.assert_called_once()
        fake.fetch.assert_not_called()
        assert request.cache_hit_type == "miss"

    def test_eviction_between_probe_and_fetch_recomputes_and_compresses(self):
        scheduler = _scheduler(_config())
        fake = _fake_cache(matched=1500, fetch_matched=0)
        scheduler.memory_aware_cache = fake
        request = _request(2000)

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is True
        assert request.pflash_metadata["cached_prefix_tokens"] == 0
        fake.probe.assert_called_once()
        fake.fetch.assert_called_once()
        assert request.cache_hit_type == "miss"

    def test_large_uncached_remainder_still_compresses_and_never_fetches(self):
        scheduler = _scheduler(_config())
        fake = _fake_cache(matched=500)  # 1500 uncached >= threshold 1000
        scheduler.memory_aware_cache = fake
        request = _request(2000)

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is True
        assert request.pflash_metadata["cached_prefix_tokens"] == 500
        fake.probe.assert_called_once()
        fake.fetch.assert_not_called()

    def test_structural_fallback_skips_probe_and_fetches_exactly_once(self):
        # Tool prompts are ineligible before threshold arithmetic, so admission
        # skips the O(prompt) probe and performs only the normal cache fetch.
        scheduler = _scheduler(_config(skip_when_tools=True))
        fake = _fake_cache(matched=500)
        scheduler.memory_aware_cache = fake
        request = _request(2000, has_tools=True)

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is False
        assert request.pflash_metadata["reason"] == "tools"
        fake.probe.assert_not_called()
        fake.fetch.assert_called_once()
        assert request.cached_tokens == 500

    def test_budget_fallback_after_probe_fetches_exactly_once(self):
        # sink + tail exceed the keep budget → ``insufficient_middle_budget``.
        scheduler = _scheduler(
            _config(keep_ratio=0.01, min_keep_tokens=16, sink_tokens=8, tail_tokens=16)
        )
        fake = _fake_cache(matched=500)
        scheduler.memory_aware_cache = fake
        request = _request(2000)

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is False
        assert request.pflash_metadata["reason"] == "insufficient_middle_budget"
        fake.probe.assert_called_once()
        fake.fetch.assert_called_once()

    def test_total_basis_and_short_prompts_never_probe(self):
        total = _scheduler(_config(threshold_basis="total"))
        fake = _fake_cache(matched=1500)
        total.memory_aware_cache = fake
        total.add_request(_request(2000))
        fake.probe.assert_not_called()  # upstream behaviour: compressed, no cache
        fake.fetch.assert_not_called()

        short = _scheduler(_config())
        fake_short = _fake_cache(matched=0)
        short.memory_aware_cache = fake_short
        short.add_request(_request(500))  # below threshold
        fake_short.probe.assert_not_called()
        fake_short.fetch.assert_called_once()

    def test_helper_guards(self):
        cfg = _config()
        cache = _fake_cache(matched=700)
        tokens = list(range(1200))
        assert _pflash_cached_prefix_tokens(cfg, cache, None, tokens) == 700
        # block-aware cache in use → total semantics
        assert _pflash_cached_prefix_tokens(cfg, cache, object(), tokens) == 0
        # no memory-aware cache / mode always / short prompt / no probe API
        assert _pflash_cached_prefix_tokens(cfg, None, None, tokens) == 0
        assert (
            _pflash_cached_prefix_tokens(_config(mode="always"), cache, None, tokens)
            == 0
        )
        assert _pflash_cached_prefix_tokens(cfg, cache, None, tokens[:999]) == 0
        assert _pflash_cached_prefix_tokens(cfg, object(), None, tokens) == 0
        # never reports more than the prompt holds
        cache.probe.return_value = PrefixProbe(5000, "prefix")
        assert _pflash_cached_prefix_tokens(cfg, cache, None, tokens) == 1200
