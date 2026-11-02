from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from django_llm_tally import InMemoryCache, LLMTally, cached_llm_call


def test_cache_stores_input_settings_and_returns_same_output() -> None:
    call_count = 0

    def generate(prompt: str, temperature: float = 0.7) -> str:
        nonlocal call_count
        call_count += 1
        return f"response-{call_count}:{prompt}:{temperature}"

    cache = InMemoryCache(default_ttl=60.0, default_max_requests=None)

    res1 = cached_llm_call(
        generate,
        "Hello world",
        temperature=0.2,
        cache=cache,
        ttl=60.0,
        settings={"feature": "greeting", "model": "gpt-4o-mini"},
    )
    res2 = cached_llm_call(
        generate,
        "Hello world",
        temperature=0.2,
        cache=cache,
        ttl=60.0,
        settings={"feature": "greeting", "model": "gpt-4o-mini"},
    )
    assert res1 == res2 == "response-1:Hello world:0.2"
    assert call_count == 1
    assert cache.size == 1

    # Different settings -> cache miss -> generates new output
    res3 = cached_llm_call(
        generate,
        "Hello world",
        temperature=0.2,
        cache=cache,
        ttl=60.0,
        settings={"feature": "greeting", "model": "gpt-4o"},
    )
    assert res3 == "response-2:Hello world:0.2"
    assert call_count == 2


def test_cache_vanishes_after_ttl_lifetime_on_fetch() -> None:
    call_count = 0

    def generate(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"gen-{call_count}:{prompt}"

    cache = InMemoryCache()
    out1 = cached_llm_call(generate, "ping", cache=cache, ttl=0.05)
    assert out1 == "gen-1:ping"
    assert call_count == 1

    # Immediately fetched before TTL -> returns cached
    out2 = cached_llm_call(generate, "ping", cache=cache, ttl=0.05)
    assert out2 == "gen-1:ping"
    assert call_count == 1

    # Wait for TTL to expire; on next fetch it vanishes itself and generates a new response
    time.sleep(0.07)
    out3 = cached_llm_call(generate, "ping", cache=cache, ttl=0.05)
    assert out3 == "gen-2:ping"
    assert call_count == 2


def test_cache_kills_itself_after_x_requests() -> None:
    call_count = 0

    def generate(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"run-{call_count}:{prompt}"

    cache = InMemoryCache()

    # Initial call: populates cache with max_requests=2
    first = cached_llm_call(generate, "item", cache=cache, ttl=300.0, max_requests=2)
    assert first == "run-1:item"
    assert call_count == 1
    assert cache.size == 1

    # Cached fetch #1 -> returns cached output, entry still alive (1 < 2)
    hit1 = cached_llm_call(generate, "item", cache=cache, ttl=300.0, max_requests=2)
    assert hit1 == "run-1:item"
    assert call_count == 1
    assert cache.size == 1

    # Cached fetch #2 -> returns cached output and immediately kills itself (2 == max_requests)
    hit2 = cached_llm_call(generate, "item", cache=cache, ttl=300.0, max_requests=2)
    assert hit2 == "run-1:item"
    assert call_count == 1
    assert cache.size == 0  # Entry vanished itself after 2 requests!

    # Next fetch -> cache miss -> generates fresh response "run-2:item"
    fresh = cached_llm_call(generate, "item", cache=cache, ttl=300.0, max_requests=2)
    assert fresh == "run-2:item"
    assert call_count == 2


def test_tally_cache_decorator_sync_and_async(tmp_path: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as td:
        base = tmp_path or Path(td)
        tally = LLMTally(
            log_path=base / "logs.jsonl",
            dashboard_path=base / "dashboard.html",
            auto_update_dashboard=False,
        )
        calls = 0

        @tally.cache(ttl=60.0, max_requests=1, config={"mode": "fast"})
        async def async_ask(q: str) -> str:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.001)
            return f"ans-{calls}:{q}"

        async def _run() -> None:
            r1 = await async_ask("hello")
            r2 = await async_ask("hello")  # 1st cached hit -> kills itself because max_requests=1
            r3 = await async_ask("hello")  # Regenerates!
            assert r1 == "ans-1:hello"
            assert r2 == "ans-1:hello"
            assert r3 == "ans-2:hello"
            assert calls == 2

        asyncio.run(_run())


class TestCacheSuite(unittest.TestCase):
    def test_stores_input_settings(self) -> None:
        test_cache_stores_input_settings_and_returns_same_output()

    def test_ttl_eviction(self) -> None:
        test_cache_vanishes_after_ttl_lifetime_on_fetch()

    def test_max_requests_eviction(self) -> None:
        test_cache_kills_itself_after_x_requests()

    def test_decorator_sync_async(self) -> None:
        test_tally_cache_decorator_sync_and_async()
