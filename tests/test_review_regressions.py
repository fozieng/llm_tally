import asyncio
import unittest
from types import SimpleNamespace

from llm_tally import (
    InMemoryCache, LLMTally, RetryConfig, RetryExhaustedError,
    cached_llm_call, execute_with_retry_async, execute_with_retry_sync,
)


class TestReviewRegressions(unittest.TestCase):
    def test_factory_closures_have_independent_caches(self):
        for tracked in (False, True):
            with self.subTest(tracked=tracked):
                tally = LLMTally(persist_logs=False, auto_update_dashboard=False)
                cache = InMemoryCache()

                def factory(prefix):
                    calls = 0

                    def answer(prompt):
                        nonlocal calls
                        calls += 1
                        return f"{prefix}{prompt}:{calls}"

                    if tracked:
                        return tally.track(feature="chat", cache=True)(answer)
                    return lambda prompt: cached_llm_call(answer, prompt, cache=cache)

                first, second = factory("A: "), factory("B: ")
                self.assertEqual(first("hello"), "A: hello:1")
                self.assertEqual(second("hello"), "B: hello:1")
                # Changing captured counters must not invalidate the cached result.
                self.assertEqual(first("hello"), "A: hello:1")
                self.assertEqual(second("hello"), "B: hello:1")

    def test_async_factory_closures_have_independent_caches(self):
        async def scenario():
            tally = LLMTally(persist_logs=False, auto_update_dashboard=False)

            def factory(prefix):
                @tally.track(feature="chat", cache=True)
                async def answer(prompt):
                    return prefix + prompt
                return answer

            first, second = factory("A: "), factory("B: ")
            for _ in range(2):
                self.assertEqual(await first("hello"), "A: hello")
                self.assertEqual(await second("hello"), "B: hello")
        asyncio.run(scenario())

    def test_returned_429_retries_sync_and_async_with_headers(self):
        for asynchronous in (False, True):
            for dictionary in (False, True):
                with self.subTest(asynchronous=asynchronous, dictionary=dictionary):
                    calls, waits = [], []
                    response = {"status_code": 429, "headers": {"Retry-After": "2"}}
                    if not dictionary:
                        response = SimpleNamespace(**response)

                    def call():
                        calls.append(1)
                        return response if len(calls) == 1 else "ok"

                    async def async_call():
                        return call()

                    async def sleep(delay):
                        waits.append(delay)

                    config = RetryConfig(retries=1, delays=(0,), max_retry_after=1)
                    if asynchronous:
                        result, failures = asyncio.run(execute_with_retry_async(
                            async_call, config=config, sleep_func=sleep))
                    else:
                        result, failures = execute_with_retry_sync(
                            call, config=config, sleep_func=waits.append)
                    self.assertEqual(result, "ok")
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(waits, [1.0])
                    self.assertEqual([f.status_code for f in failures], [429])

    def test_returned_429_exhaustion_and_disabled_retries(self):
        response = {"status_code": 429}
        with self.assertRaises(RetryExhaustedError) as caught:
            execute_with_retry_sync(lambda: response, config=RetryConfig(retries=2, delays=(0,)))
        self.assertEqual([a.status_code for a in caught.exception.attempts], [429] * 3)
        for config in (RetryConfig(enabled=False), RetryConfig(retry_on_rate_limit=False)):
            result, failures = execute_with_retry_sync(lambda: response, config=config)
            self.assertIs(result, response)
            self.assertEqual(failures, [])
