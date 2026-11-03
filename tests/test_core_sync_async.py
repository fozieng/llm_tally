from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import List

from llm_tally import AILogHelper, LLMTally


def test_end_to_end_sync_and_async_pipeline(tmp_path: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as td:
        base = tmp_path or Path(td)
        sleeps: List[float] = []
        async_sleeps: List[float] = []

        async def fake_async_sleep(d: float) -> None:
            async_sleeps.append(d)

        helper = AILogHelper(
            default_model="gpt-4o-mini",
            log_path=base / "logs.jsonl",
            dashboard_path=base / "dashboard.html",
            sleep_func=sleeps.append,
            async_sleep_func=fake_async_sleep,
        )
        assert isinstance(helper, LLMTally)

        sync_calls = 0

        @helper.track(
            feature="sync_chat",
            model="gpt-4o",
            cache=True,
            ttl=60.0,
            max_requests=2,
            config={"temperature": 0.1},
        )
        def sync_llm(prompt: str) -> dict:
            nonlocal sync_calls
            sync_calls += 1
            if sync_calls == 1:
                # First ever attempt fails with 500 -> triggers 0.5s retry
                err = RuntimeError("HTTP 500 Internal Server Error")
                setattr(err, "status_code", 500)
                raise err
            return {
                "reply": f"sync:{prompt}:{sync_calls}",
                "usage": {"input_tokens": 500, "output_tokens": 200},
            }

        # Initial call: Attempt 1 fails (500), Attempt 2 succeeds after 0.5s retry and caches result
        res1 = sync_llm("hello")
        assert res1["reply"] == "sync:hello:2"
        assert sleeps == [0.5]

        # Cached hit #1
        res2 = sync_llm("hello")
        assert res2 == res1
        assert sync_calls == 2

        # Cached hit #2 -> reaches max_requests=2 and kills itself
        res3 = sync_llm("hello")
        assert res3 == res1
        assert sync_calls == 2

        # Next call -> cache entry killed itself, so executes sync_llm again!
        res4 = sync_llm("hello")
        assert res4["reply"] == "sync:hello:3"
        assert sync_calls == 3

        # Test async pipeline as well
        async_calls = 0

        @helper.track(
            feature="async_summarizer",
            model="gemini-2.5-flash",
            cache=True,
            ttl=60.0,
            max_requests=1,
        )
        async def async_llm(text: str) -> dict:
            nonlocal async_calls
            async_calls += 1
            if async_calls == 1:
                raise TimeoutError("Async network timeout")
            return {
                "summary": f"async:{text}:{async_calls}",
                "usage": {"input_tokens": 800, "output_tokens": 100},
            }

        async def _run_async() -> None:
            a1 = await async_llm("article")
            assert a1["summary"] == "async:article:2"
            assert async_sleeps == [0.5]

            # 1st cached hit (kills itself after 1 hit)
            a2 = await async_llm("article")
            assert a2 == a1
            assert async_calls == 2

            # Regenerates on next call
            a3 = await async_llm("article")
            assert a3["summary"] == "async:article:3"
            assert async_calls == 3

        asyncio.run(_run_async())

        report = helper.get_report()
        assert report["overview"]["total_calls"] == 7
        assert report["overview"]["cached_calls"] == 3
        assert report["overview"]["total_retries"] == 2


class TestCorePipelineSuite(unittest.TestCase):
    def test_sync_and_async_pipeline(self) -> None:
        test_end_to_end_sync_and_async_pipeline()
