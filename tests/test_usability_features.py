"""
Regression tests for the 0.2.0 usability fixes: opt-in caching, call-time model detection,
estimated-token flag, preview redaction, per-call metadata, record sinks, dashboard throttling,
HTTP 429 / Retry-After retries, pricing files, `None` constructor arguments, abandoned and
context-managed streams, and the CLI.
"""

import asyncio
import contextlib
import gc
import io
import json
import random
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace as NS

from llm_tally import LLMTally, RetryConfig, RetryExhaustedError, execute_with_retry_sync
from llm_tally.cli import main as cli_main
from llm_tally.retry import extract_retry_after, is_retryable_exception


class RateLimitError(Exception):
    """Mimics `openai.RateLimitError` / `anthropic.RateLimitError` (status 429)."""

    def __init__(self, message: str = "Rate limit reached", headers=None) -> None:
        super().__init__(message)
        self.status_code = 429
        self.response = NS(status_code=429, headers=headers or {})


class TestUsabilityFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.base = Path(self._td.name)
        warnings.simplefilter("ignore")

    def tearDown(self) -> None:
        self._td.cleanup()

    def make_tally(self, **kwargs) -> LLMTally:
        kwargs.setdefault("log_path", self.base / "logs.jsonl")
        kwargs.setdefault("dashboard_path", self.base / "dashboard.html")
        kwargs.setdefault("auto_update_dashboard", False)
        kwargs.setdefault("sleep_func", lambda _: None)
        return LLMTally(**kwargs)

    # --- Constructor defaults --------------------------------------------------
    def test_none_constructor_arguments_mean_defaults(self) -> None:
        tally = LLMTally(
            default_model=None,
            log_path=None,
            dashboard_path=None,
            default_cache_enabled=None,
            default_retry_enabled=None,
            default_retries=None,
            retry_delays=None,
            auto_update_dashboard=None,
        )
        self.assertEqual(tally.default_model, "gpt-4o-mini")
        self.assertEqual(tally.log_path, Path(".llm_tally/logs.jsonl"))
        self.assertEqual(tally.dashboard_path, Path(".llm_tally/dashboard.html"))
        self.assertFalse(tally.default_cache_enabled)
        self.assertTrue(tally.default_retry_enabled)
        self.assertEqual(tally.default_retries, 3)
        self.assertEqual(tally.retry_delays, (0.5, 1.0, 1.5))
        self.assertTrue(tally.auto_update_dashboard)
        with self.assertRaises(ValueError):
            LLMTally(default_retries=-1)

    def test_caching_is_opt_in(self) -> None:
        tally = self.make_tally()

        @tally.track()
        def creative(prompt: str) -> str:
            return f"{prompt}-{random.random()}"

        self.assertNotEqual(creative("poem"), creative("poem"))

        @tally.track(cache=True)
        def deterministic(prompt: str) -> str:
            return f"{prompt}-{random.random()}"

        self.assertEqual(deterministic("q"), deterministic("q"))

    # --- Model & token accuracy -------------------------------------------------
    def test_call_time_model_kwarg_is_used_for_pricing(self) -> None:
        tally = self.make_tally()

        @tally.track()
        def ask(prompt: str, model: str = "gpt-4o") -> dict:
            return {"usage": {"input_tokens": 1_000_000, "output_tokens": 0}}

        ask("hi", model="claude-opus-4-5")
        rec = tally.get_records()[-1]
        self.assertEqual(rec.model, "claude-opus-4-5")
        self.assertAlmostEqual(rec.cost_usd, 5.0)

        @tally.track(model="gpt-4o-mini")
        def pinned(prompt: str, model: str = "x") -> dict:
            return {"usage": {"input_tokens": 1_000_000, "output_tokens": 0}}

        pinned("hi", model="claude-opus-4-5")
        self.assertEqual(tally.get_records()[-1].model, "gpt-4o-mini")

    def test_estimated_tokens_are_flagged(self) -> None:
        tally = self.make_tally()

        @tally.track()
        def plain(prompt: str) -> str:
            return "a plain string with no usage metadata"

        @tally.track()
        def real(prompt: str) -> dict:
            return {"usage": {"input_tokens": 10, "output_tokens": 5}}

        plain("hi")
        real("hi")
        est, exact = tally.get_records()
        self.assertTrue(est.estimated_tokens)
        self.assertFalse(exact.estimated_tokens)
        overview = tally.get_report()["overview"]
        self.assertEqual(overview["estimated_calls"], 1)
        self.assertEqual(overview["estimated_cost_usd"], round(est.cost_usd, 8))

    # --- Privacy ---------------------------------------------------------------
    def test_log_previews_false_and_redact(self) -> None:
        tally = self.make_tally(log_previews=False, auto_update_dashboard=True)

        @tally.track()
        def secret(prompt: str) -> str:
            return "reply about 123-45-6789"

        secret("my SSN is 123-45-6789")
        raw = (self.base / "logs.jsonl").read_text()
        self.assertNotIn("123-45-6789", raw)
        self.assertNotIn("123-45-6789", (self.base / "dashboard.html").read_text())

        redacting = self.make_tally(
            log_path=self.base / "r.jsonl",
            redact=lambda text: text.replace("123-45-6789", "[SSN]"),
        )
        redacting.run(secret, "my SSN is 123-45-6789")
        rec = redacting.get_records()[-1]
        self.assertIn("[SSN]", rec.input_preview)
        self.assertNotIn("123-45-6789", rec.input_preview + rec.output_preview)

    # --- Metadata & sinks -------------------------------------------------------
    def test_metadata_from_instance_context_decorator_and_run(self) -> None:
        tally = self.make_tally(metadata={"env": "test"})

        @tally.track(metadata={"prompt_version": "v2"})
        def ask(prompt: str) -> dict:
            return {"usage": {"input_tokens": 1000, "output_tokens": 100}}

        def undecorated(prompt: str) -> dict:
            return {"usage": {"input_tokens": 1000, "output_tokens": 100}}

        with tally.context(user_id="alice", tenant="acme"):
            ask("a")
            with tally.context(user_id="bob"):
                ask("b")
        tally.run(undecorated, "c", metadata={"user_id": "carol"})

        metas = [r.metadata for r in tally.get_records()]
        self.assertEqual(
            metas[0], {"env": "test", "user_id": "alice", "tenant": "acme", "prompt_version": "v2"}
        )
        self.assertEqual(metas[1]["user_id"], "bob")
        self.assertEqual(metas[2]["user_id"], "carol")
        self.assertNotIn("tenant", metas[2])

        by_user = {row["value"]: row for row in tally.get_report()["metadata_costs"]["user_id"]}
        self.assertEqual(set(by_user), {"alice", "bob", "carol"})
        self.assertEqual(by_user["alice"]["total_calls"], 1)

    def test_context_metadata_is_isolated_per_asyncio_task(self) -> None:
        tally = self.make_tally()

        @tally.track()
        async def ask(prompt: str) -> dict:
            await asyncio.sleep(0.01)
            return {"usage": {"input_tokens": 1, "output_tokens": 1}}

        async def handle(user: str) -> None:
            with tally.context(user_id=user):
                await ask(user)

        async def main() -> None:
            await asyncio.gather(*(handle(u) for u in ("u1", "u2", "u3")))

        asyncio.run(main())
        records = tally.get_records()
        self.assertEqual(len(records), 3)
        for rec in records:
            self.assertIn(f'"{rec.metadata["user_id"]}"', rec.input_preview)

    def test_on_record_sink_gets_one_final_record_per_call_even_without_file(self) -> None:
        received = []
        tally = self.make_tally(persist_logs=False, on_record=received.append)

        @tally.track()
        def ask(prompt: str) -> dict:
            return {"usage": {"input_tokens": 5, "output_tokens": 5}}

        @tally.track()
        def stream(prompt: str):
            yield "hello "
            yield "world"

        ask("x")
        self.assertEqual(list(stream("y")), ["hello ", "world"])
        self.assertEqual(len(received), 2)
        self.assertEqual(received[1].status, "success")
        self.assertEqual(received[1].output_preview, "hello world")
        self.assertFalse((self.base / "logs.jsonl").exists())

        def broken_sink(_record) -> None:
            raise RuntimeError("sink down")

        tally.add_sink(broken_sink)
        self.assertEqual(ask("z")["usage"]["input_tokens"], 5)  # a failing sink never breaks the call

    def test_custom_cache_backend_is_used(self) -> None:
        class DictCache:
            def __init__(self) -> None:
                self.data = {}

            def fetch(self, key):
                if key in self.data:
                    return True, self.data[key], "hit"
                return False, None, "miss"

            def store(self, key, input_settings, output, ttl=None, max_requests=None, use_defaults=True):
                self.data[key] = output

            def clear(self) -> None:
                self.data.clear()

        backend = DictCache()
        tally = self.make_tally(cache_backend=backend)

        @tally.track(cache=True)
        def ask(prompt: str) -> str:
            return f"{prompt}-{random.random()}"

        self.assertEqual(ask("q"), ask("q"))
        self.assertEqual(len(backend.data), 1)
        tally.clear_cache()
        self.assertEqual(backend.data, {})

    # --- Dashboard throttling -----------------------------------------------------
    def test_dashboard_updates_are_throttled_and_flushed(self) -> None:
        tally = self.make_tally(auto_update_dashboard=True, dashboard_update_interval=60.0)
        dash = self.base / "dashboard.html"

        @tally.track()
        def ask(feature_marker: str) -> dict:
            return {"usage": {"input_tokens": 1, "output_tokens": 1}}

        ask("first_call_marker")
        self.assertTrue(dash.exists())
        first_mtime = dash.stat().st_mtime_ns
        for i in range(20):
            ask(f"call_{i}")
        self.assertEqual(dash.stat().st_mtime_ns, first_mtime)  # no rewrite per call
        self.assertIsNotNone(tally.flush())
        self.assertIn("call_19", dash.read_text())
        self.assertIsNone(tally.flush())  # nothing pending

    def test_dashboard_days_window(self) -> None:
        tally = self.make_tally()

        @tally.track(feature="recent_feature")
        def ask() -> dict:
            return {"usage": {"input_tokens": 1, "output_tokens": 1}}

        ask()
        old = tally.get_records()[0].to_dict()
        old.update(id="old1", feature="ancient_feature", timestamp="2020-01-01T00:00:00+00:00")
        with open(self.base / "logs.jsonl", "a") as fh:
            fh.write(json.dumps(old) + "\n")
        self.assertEqual(len(tally.get_records()), 2)
        self.assertEqual(len(tally.get_records(days=7)), 1)
        self.assertNotIn("ancient_feature", tally.render_dashboard(days=7))
        self.assertIn("ancient_feature", tally.render_dashboard())

    # --- Retries -------------------------------------------------------------------
    def test_429_is_retried_and_honors_retry_after(self) -> None:
        sleeps = []
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RateLimitError(headers={"Retry-After": "4"})
            if calls["n"] == 2:
                raise RateLimitError(headers={"retry-after-ms": "2500"})
            return "ok"

        result, failures = execute_with_retry_sync(flaky, config=RetryConfig(), sleep_func=sleeps.append)
        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [4.0, 2.5])
        self.assertEqual([f.status_code for f in failures], [429, 429])

        capped = RetryConfig(max_retry_after=1.0)
        self.assertEqual(capped.compute_wait(0, RateLimitError(headers={"Retry-After": "3600"})), 1.0)

    def test_rate_limit_retry_can_be_disabled(self) -> None:
        tally = self.make_tally(retry_on_rate_limit=False)
        calls = {"n": 0}

        @tally.track()
        def limited() -> str:
            calls["n"] += 1
            raise RateLimitError()

        with self.assertRaises(RateLimitError):
            limited()
        self.assertEqual(calls["n"], 1)
        self.assertEqual(is_retryable_exception(RateLimitError(), retry_on_rate_limit=False), (False, 429))

    def test_rate_limit_detected_by_class_name_and_exhausts(self) -> None:
        class TooManyRequests(Exception):
            pass

        self.assertTrue(is_retryable_exception(TooManyRequests("slow down"))[0])
        tally = self.make_tally()

        @tally.track()
        def always_limited() -> str:
            raise TooManyRequests("slow down")

        with self.assertRaises(RetryExhaustedError) as ctx:
            always_limited()
        self.assertEqual(len(ctx.exception.attempts), 4)

    def test_jitter_only_adds_wait(self) -> None:
        cfg = RetryConfig(jitter=0.5)
        waits = [cfg.compute_wait(1) for _ in range(200)]
        self.assertTrue(all(1.0 <= w <= 1.5 for w in waits))
        self.assertGreater(len(set(waits)), 1)

    def test_extract_retry_after_variants(self) -> None:
        self.assertEqual(extract_retry_after(NS(retry_after=7)), 7.0)
        self.assertIsNone(extract_retry_after(ValueError("x")))
        self.assertIsNone(extract_retry_after(RateLimitError(headers={"Retry-After": "soon"})))
        future = RateLimitError(headers={"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"})
        self.assertGreater(extract_retry_after(future), 1000)

    # --- Pricing file ----------------------------------------------------------------
    def test_pricing_file(self) -> None:
        price_file = self.base / "prices.json"
        price_file.write_text(
            json.dumps({"house-model": {"input_cost_per_1m": 2.0, "output_cost_per_1m": 8.0}})
        )
        tally = self.make_tally(pricing_file=price_file)

        @tally.track(model="house-model")
        def ask() -> dict:
            return {"usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}}

        ask()
        self.assertAlmostEqual(tally.get_records()[-1].cost_usd, 10.0)

        bad = self.base / "bad.json"
        bad.write_text(json.dumps({"m": {"input_cost_per_1m": 1}}))
        with self.assertRaises(ValueError):
            tally.load_pricing(bad)

    # --- Streams ----------------------------------------------------------------------
    def test_abandoned_stream_is_logged_as_incomplete(self) -> None:
        tally = self.make_tally()

        @tally.track()
        def long_stream():
            yield "a" * 400
            yield "b" * 400

        s = long_stream()
        next(iter(s))
        del s
        gc.collect()
        rec = tally.get_records()[-1]
        self.assertEqual(rec.status, "incomplete")
        self.assertGreater(rec.output_tokens, 0)
        self.assertTrue(rec.estimated_tokens)

    def test_stream_manager_whose_enter_returns_the_stream(self) -> None:
        """Mimics Anthropic `client.messages.stream()` -> `MessageStreamManager`."""
        events = []
        snapshot = NS(model="claude-sonnet-4-5", usage=NS(input_tokens=1000, output_tokens=200))

        class MessageStream:
            current_message_snapshot = snapshot

            def __iter__(self):
                return iter(["x", "y"])

            @property
            def text_stream(self):
                return iter(["x", "y"])

        class MessageStreamManager:
            def __enter__(self):
                events.append("enter")
                return MessageStream()

            def __exit__(self, *exc):
                events.append("exit")
                return False

        tally = self.make_tally()

        @tally.track()
        def open_stream():
            return MessageStreamManager()

        with open_stream() as stream:
            self.assertEqual(list(stream), ["x", "y"])
        self.assertEqual(events, ["enter", "exit"])
        rec = tally.get_records()[-1]
        self.assertEqual(rec.status, "success")
        self.assertEqual(rec.output_preview, "xy")

        # Iterating `stream.text_stream` bypasses the proxy; usage comes from the snapshot
        with open_stream() as stream:
            self.assertEqual("".join(stream.text_stream), "xy")
        rec = tally.get_records()[-1]
        self.assertEqual((rec.model, rec.input_tokens, rec.output_tokens), ("claude-sonnet-4-5", 1000, 200))
        self.assertFalse(rec.estimated_tokens)
        self.assertAlmostEqual(rec.cost_usd, 0.006)

    def test_async_stream_manager_whose_aenter_returns_the_stream(self) -> None:
        class AsyncMessageStream:
            def __init__(self) -> None:
                self._items = iter(["p", "q"])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration:
                    raise StopAsyncIteration

        class AsyncMessageStreamManager:
            async def __aenter__(self):
                return AsyncMessageStream()

            async def __aexit__(self, *exc):
                return False

        tally = self.make_tally()

        @tally.track()
        def open_stream():
            return AsyncMessageStreamManager()

        async def consume():
            async with open_stream() as stream:
                return [chunk async for chunk in stream]

        self.assertEqual(asyncio.run(consume()), ["p", "q"])
        self.assertEqual(tally.get_records()[-1].output_preview, "pq")

    # --- Privacy scope (errors, attempts, config) -----------------------------------
    def test_log_previews_false_hides_error_text_but_keeps_types(self) -> None:
        class BadRequest(Exception):
            status_code = 400

        tally = self.make_tally(log_previews=False)

        @tally.track()
        def bad(prompt: str) -> str:
            raise BadRequest(f"Invalid prompt: {prompt}")

        @tally.track(retries=1)
        def flaky(prompt: str) -> str:
            raise ConnectionError(f"reset while sending {prompt}")

        with self.assertRaises(BadRequest):
            bad("SECRET user text")
        with self.assertRaises(RetryExhaustedError):
            flaky("SECRET again")
        raw = (self.base / "logs.jsonl").read_text()
        self.assertNotIn("SECRET", raw)
        failed_bad, failed_flaky = tally.get_records()
        self.assertTrue(failed_bad.error_message.startswith("BadRequest: "))
        self.assertEqual(failed_bad.attempt_errors[0]["error_type"], "BadRequest")
        self.assertEqual(failed_bad.attempt_errors[0]["status_code"], 400)
        self.assertEqual(len(failed_flaky.attempt_errors), 2)

    def test_redact_applies_to_errors_attempts_and_config(self) -> None:
        tally = self.make_tally(redact=lambda s: s.replace("SECRET", "[x]"))

        @tally.track(config={"system": "SECRET system prompt", "nested": ["SECRET"], "temperature": 0})
        def bad(prompt: str) -> str:
            raise ValueError(f"cannot handle {prompt}")

        with self.assertRaises(ValueError):
            bad("SECRET user text")
        self.assertNotIn("SECRET", (self.base / "logs.jsonl").read_text())
        rec = tally.get_records()[-1]
        self.assertEqual(rec.config, {"system": "[x] system prompt", "nested": ["[x]"], "temperature": 0})
        self.assertIn("[x] user text", rec.error_message)

    def test_redact_that_raises_keeps_the_record(self) -> None:
        tally = self.make_tally(redact=lambda s: 1 / 0)

        @tally.track()
        def ok(prompt: str) -> dict:
            return {"usage": {"input_tokens": 10, "output_tokens": 5}}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ok("hi")
        rec = tally.get_records()[-1]
        self.assertEqual(rec.input_preview, "[redaction failed]")
        self.assertEqual(rec.input_tokens, 10)
        self.assertTrue(any("redact function raised" in str(w.message) for w in caught))

    # --- Append-only, incremental, rotating log ---------------------------------------
    def test_log_is_append_only_and_reads_incrementally(self) -> None:
        from llm_tally import JSONLLogger

        tally = self.make_tally()

        @tally.track()
        def stream(prompt: str):
            yield "a"
            yield "b"

        @tally.track()
        def plain(prompt: str) -> str:
            return "x"

        plain("1")
        list(stream("2"))
        log = self.base / "logs.jsonl"
        self.assertEqual(len(log.read_text().splitlines()), 3)  # plain + placeholder + final
        records = tally.get_records()
        self.assertEqual([r.status for r in records], ["success", "success"])
        self.assertEqual(records[1].output_preview, "ab")

        # Another process/logger appending is picked up incrementally
        other = JSONLLogger(log)
        extra = records[0]
        extra.id = "external1"
        other.append(extra)
        self.assertEqual(len(tally.get_records()), 3)
        self.assertGreater(tally.logger._cache_offset, 0)

        # Mutating returned records never corrupts the read cache
        tally.get_records()[0].config["poisoned"] = True
        self.assertNotIn("poisoned", tally.get_records()[0].config)

        # compact() keeps one line per record and the cache resets cleanly
        self.assertEqual(tally.logger.compact(), 3)
        self.assertEqual(len(log.read_text().splitlines()), 3)
        self.assertEqual(len(tally.get_records()), 3)

    def test_log_rotation(self) -> None:
        tally = self.make_tally(max_log_mb=0.002, log_backups=2)  # ~2 KB per file

        @tally.track()
        def ask(prompt: str) -> dict:
            return {"usage": {"input_tokens": 1, "output_tokens": 1}}

        for i in range(40):
            ask(f"prompt {i} " + "x" * 50)
        log = self.base / "logs.jsonl"
        self.assertTrue(Path(f"{log}.1").exists())
        self.assertTrue(Path(f"{log}.2").exists())
        self.assertFalse(Path(f"{log}.3").exists())
        self.assertLessEqual(log.stat().st_size, 2100)
        current = tally.get_records()
        self.assertGreater(len(current), 0)
        self.assertLess(len(current), 40)
        self.assertIn("prompt 39", current[-1].input_preview)

    def test_dashboard_throttle_stretches_with_build_time(self) -> None:
        tally = self.make_tally(auto_update_dashboard=True, dashboard_update_interval=0.01)

        @tally.track()
        def ask(prompt: str) -> str:
            return "x"

        ask("first")
        dash = self.base / "dashboard.html"
        first_mtime = dash.stat().st_mtime_ns
        tally._last_dashboard_build_seconds = 30.0  # pretend rebuilds are very slow
        import time as _time

        _time.sleep(0.05)
        ask("second")
        self.assertEqual(dash.stat().st_mtime_ns, first_mtime)
        tally.flush()
        self.assertIn("second", dash.read_text())

    # --- Stream manager failing on enter -----------------------------------------------
    def test_stream_manager_enter_failure_is_logged_as_failed(self) -> None:
        class MessageStreamManager:
            def __enter__(self):
                raise ConnectionError("upstream reset")

            def __exit__(self, *exc):
                return False

        class AsyncMessageStreamManager:
            async def __aenter__(self):
                raise TimeoutError("connect timeout")

            async def __aexit__(self, *exc):
                return False

        tally = self.make_tally()

        @tally.track()
        def open_sync():
            return MessageStreamManager()

        @tally.track()
        def open_async():
            return AsyncMessageStreamManager()

        with self.assertRaises(ConnectionError):
            with open_sync():
                pass
        rec = tally.get_records()[-1]
        self.assertEqual((rec.status, rec.error_message), ("failed", "ConnectionError: upstream reset"))

        async def consume():
            async with open_async():
                pass

        with self.assertRaises(TimeoutError):
            asyncio.run(consume())
        rec = tally.get_records()[-1]
        self.assertEqual((rec.status, rec.error_message), ("failed", "TimeoutError: connect timeout"))
        self.assertEqual(len(tally.get_records()), 2)

    # --- Thread pools ---------------------------------------------------------------------
    def test_wrap_context_carries_metadata_into_thread_pools(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        tally = self.make_tally()

        @tally.track()
        def work(i: int) -> dict:
            return {"usage": {"input_tokens": 1, "output_tokens": 1}}

        with tally.context(user_id="alice"), ThreadPoolExecutor(4) as pool:
            list(pool.map(tally.wrap_context(work), range(8)))
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(work, range(2)))  # outside any context -> no metadata
        metas = [r.metadata for r in tally.get_records()]
        self.assertEqual(metas[:8], [{"user_id": "alice"}] * 8)
        self.assertEqual(metas[8:], [{}, {}])

    # --- Async sinks ------------------------------------------------------------------
    def test_async_sinks_do_not_delay_calls_and_flush_drains_them(self) -> None:
        import threading
        import time as _time

        received = []
        sink_threads = set()

        def slow_sink(record) -> None:
            _time.sleep(0.05)
            sink_threads.add(threading.current_thread().name)
            received.append(record.id)

        tally = self.make_tally(persist_logs=False, on_record=slow_sink, async_sinks=True)

        @tally.track()
        def fast() -> str:
            return "x"

        start = _time.perf_counter()
        for _ in range(5):
            fast()
        self.assertLess(_time.perf_counter() - start, 0.1)
        tally.flush()
        self.assertEqual(len(received), 5)
        self.assertEqual(sink_threads, {"llm-tally-sinks"})

    def test_async_sink_queue_full_drops_with_warning(self) -> None:
        import threading

        gate = threading.Event()
        tally = self.make_tally(
            persist_logs=False, on_record=lambda r: gate.wait(5), async_sinks=True, sink_queue_size=1
        )

        @tally.track()
        def fast() -> str:
            return "x"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(5):
                fast()
        gate.set()
        self.assertTrue(any("sink queue is full" in str(w.message) for w in caught))
        self.assertTrue(tally._drain_sinks(5))

    # --- CLI ----------------------------------------------------------------------
    def test_cli_report_and_dashboard(self) -> None:
        tally = self.make_tally()

        @tally.track(feature="cli_feature", metadata={"user_id": "dana"})
        def ask() -> dict:
            return {"usage": {"input_tokens": 1000, "output_tokens": 100}}

        ask()
        log = str(self.base / "logs.jsonl")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli_main(["report", "--log", log]), 0)
        self.assertIn("cli_feature", out.getvalue())
        self.assertIn("Top user_id: dana", out.getvalue())

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli_main(["report", "--log", log, "--json"]), 0)
        self.assertEqual(json.loads(out.getvalue())["overview"]["total_calls"], 1)

        html_out = self.base / "cli.html"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main(["dashboard", "--log", log, "--out", str(html_out)]), 0)
        self.assertIn("cli_feature", html_out.read_text())

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main(["report", "--log", str(self.base / "missing.jsonl")]), 0)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli_main(["compact", "--log", log]), 0)
        self.assertIn("1 calls", out.getvalue())


if __name__ == "__main__":
    unittest.main()
