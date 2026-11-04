from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterator, List

from llm_tally import (
    InMemoryCache,
    LLMTally,
    PricingRegistry,
    build_cache_key,
    extract_token_usage,
)


class DummyHTTPError(Exception):
    def __init__(self, msg: str, status_code: int) -> None:
        super().__init__(msg)
        self.status_code = status_code


class TestAll15BugRegressions(unittest.TestCase):
    # ------------------------------------------------------------------
    # Bug #1: {1: "a"} vs {"1": "a"} and (1, 2) vs [1, 2] and True vs 1
    # ------------------------------------------------------------------
    def test_bug_01_distinct_cache_keys_for_types(self) -> None:
        k_int_dict, _ = build_cache_key("fn", ({1: "a"},), {})
        k_str_dict, _ = build_cache_key("fn", ({"1": "a"},), {})
        self.assertNotEqual(k_int_dict, k_str_dict)

        k_tuple, _ = build_cache_key("fn", ((1, 2),), {})
        k_list, _ = build_cache_key("fn", ([1, 2],), {})
        self.assertNotEqual(k_tuple, k_list)

        k_bool, _ = build_cache_key("fn", (True,), {})
        k_int, _ = build_cache_key("fn", (1,), {})
        self.assertNotEqual(k_bool, k_int)

    # ------------------------------------------------------------------
    # Bug #2: Set with mixed types ({1, "a"}) crashes with TypeError
    # ------------------------------------------------------------------
    def test_bug_02_mixed_type_sets_do_not_crash(self) -> None:
        k1, _ = build_cache_key("fn", ({1, "a", (2, "b")},), {})
        k2, _ = build_cache_key("fn", ({"a", 1, (2, "b")},), {})
        self.assertEqual(k1, k2)

        k_frozenset, _ = build_cache_key("fn", (frozenset({1, "a"}),), {})
        self.assertNotEqual(k1, k_frozenset)

    # ------------------------------------------------------------------
    # Bug #3: Self-referential objects & decorated instance methods (`self`)
    # ------------------------------------------------------------------
    def test_bug_03_self_referential_objects_and_instance_methods(self) -> None:
        # 1. Self-referential list & dict should not raise RecursionError
        cyclic_list: List[Any] = [1, 2]
        cyclic_list.append(cyclic_list)
        k_cycle, _ = build_cache_key("fn", (cyclic_list,), {})
        self.assertTrue(len(k_cycle) == 64)

        # 2. Decorated instance method where `self` holds state & cyclic ref:
        #    - Same state (`Bot("pirate")`) hits the cache on repeated calls.
        #    - Different state (`Bot("pirate")` vs `Bot("lawyer")`) gets distinct cache entries!
        with tempfile.TemporaryDirectory() as td:
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
            )
            api_calls = 0

            class Bot:
                def __init__(self, persona: str) -> None:
                    self.persona = persona
                    self.self_ref = self  # cyclic attribute on self!

                @tally.track(feature="bot_reply", model="gpt-4o-mini", cache=True)
                def reply(self, prompt: str) -> str:
                    nonlocal api_calls
                    api_calls += 1
                    return f"[{self.persona}] {prompt}"

            pirate = Bot("pirate")
            self.assertEqual(pirate.reply("hi"), "[pirate] hi")
            self.assertEqual(pirate.reply("hi"), "[pirate] hi")  # Cache hit!
            self.assertEqual(api_calls, 1)

            lawyer = Bot("lawyer")
            self.assertEqual(lawyer.reply("hi"), "[lawyer] hi")  # Distinct instance state -> miss!
            self.assertEqual(api_calls, 2)

    # ------------------------------------------------------------------
    # Bug #4: Implicit `__context__` chaining & CancelledError not retried
    # ------------------------------------------------------------------
    def test_bug_04_implicit_context_and_cancelled_error_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sleeps: List[float] = []
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
                sleep_func=sleeps.append,
            )
            attempts = 0

            @tally.track(feature="implicit_context", model="gpt-4o-mini")
            def raises_value_error_inside_except() -> str:
                nonlocal attempts
                attempts += 1
                try:
                    raise ConnectionError("initial network error caught internally")
                except ConnectionError:
                    # Implicit __context__ is ConnectionError, but actual error is ValueError!
                    raise ValueError("bad user input")

            with self.assertRaises(ValueError):
                raises_value_error_inside_except()

            self.assertEqual(attempts, 1)
            self.assertEqual(sleeps, [])

            # Also verify asyncio.CancelledError inside except ConnectionError is never retried
            async_attempts = 0

            @tally.track(feature="cancel_test", model="gpt-4o-mini")
            async def cancelled_task() -> str:
                nonlocal async_attempts
                async_attempts += 1
                try:
                    raise ConnectionError("network error")
                except ConnectionError:
                    raise asyncio.CancelledError()

            async def _run_cancel() -> None:
                with self.assertRaises(asyncio.CancelledError):
                    await cancelled_task()

            asyncio.run(_run_cancel())
            self.assertEqual(async_attempts, 1)

    # ------------------------------------------------------------------
    # Bug #5: SystemExit(503) is not treated as HTTP 503
    # ------------------------------------------------------------------
    def test_bug_05_system_exit_503_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sleeps: List[float] = []
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
                sleep_func=sleeps.append,
            )
            attempts = 0

            @tally.track(feature="sys_exit", model="gpt-4o-mini")
            def exit_503() -> str:
                nonlocal attempts
                attempts += 1
                raise SystemExit(503)

            with self.assertRaises(SystemExit) as ctx:
                exit_503()

            self.assertEqual(ctx.exception.code, 503)
            self.assertEqual(attempts, 1)
            self.assertEqual(sleeps, [])

    # ------------------------------------------------------------------
    # Bug #6: retry=False returns 503 response object without raising
    # ------------------------------------------------------------------
    def test_bug_06_retry_false_does_not_raise_on_503_response_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
            )

            @tally.track(feature="no_retry_503", model="gpt-4o-mini", retry=False, cache=False)
            def return_503_dict() -> dict:
                return {"status_code": 503, "error": "maintenance"}

            res = return_503_dict()
            self.assertEqual(res, {"status_code": 503, "error": "maintenance"})

    # ------------------------------------------------------------------
    # Bug #7: Anthropic input_tokens + cache_read_input_tokens + cache_creation_input_tokens
    # ------------------------------------------------------------------
    def test_bug_07_anthropic_cost_calculation_with_cached_and_creation_tokens(self) -> None:
        registry = PricingRegistry()
        # 100 new input tokens ($3/1M = $0.0003) + 1,000,000 cached read tokens ($0.30/1M = $0.30) -> $0.3003
        cost = registry.calculate_cost(
            "claude-3-5-sonnet",
            input_tokens=100,
            output_tokens=0,
            cached_input_tokens=1_000_000,
        )
        self.assertAlmostEqual(cost, 0.3003, places=7)

        # With cache_creation_input_tokens (billed at 1.25x = $3.75/1M): 100,000 creation tokens = $0.375
        resp = SimpleNamespace(
            model="claude-3-5-sonnet",
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=0,
                cache_read_input_tokens=1_000_000,
                cache_creation_input_tokens=100_000,
            ),
        )
        u = extract_token_usage(resp)
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.cached_input_tokens, 1_000_000)
        self.assertEqual(u.cache_creation_input_tokens, 100_000)

        total_cost = registry.calculate_cost(
            "claude-3-5-sonnet",
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cached_input_tokens=u.cached_input_tokens,
            cache_creation_input_tokens=u.cache_creation_input_tokens,
        )
        self.assertAlmostEqual(total_cost, 0.3003 + 0.375, places=7)

    # ------------------------------------------------------------------
    # Bug #8: OpenAI Responses API `input_tokens_details.cached_tokens`
    # ------------------------------------------------------------------
    def test_bug_08_openai_responses_api_input_tokens_details(self) -> None:
        resp = SimpleNamespace(
            model="gpt-4o",
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=200,
                input_tokens_details=SimpleNamespace(cached_tokens=800),
            ),
        )
        u = extract_token_usage(resp)
        self.assertEqual(u.input_tokens, 1000)
        self.assertEqual(u.output_tokens, 200)
        self.assertEqual(u.cached_input_tokens, 800)

    # ------------------------------------------------------------------
    # Bug #9: Custom models with `/` in the name match properly
    # ------------------------------------------------------------------
    def test_bug_09_custom_models_with_slash_match(self) -> None:
        registry = PricingRegistry()
        registry.register("meta-llama/llama-3-70b", input_cost_per_1m=0.88, output_cost_per_1m=0.99)
        resolved = registry.resolve("meta-llama/llama-3-70b")
        self.assertEqual(resolved.model, "meta-llama/llama-3-70b")
        self.assertEqual(resolved.input_cost_per_1m, 0.88)
        self.assertEqual(resolved.output_cost_per_1m, 0.99)

    # ------------------------------------------------------------------
    # Bug #10: o3-pro, gpt-5, claude-opus-4-5, claude-haiku-4-5, gemini-3 & unknown warning
    # ------------------------------------------------------------------
    def test_bug_10_model_pricing_accuracy_and_unknown_warning(self) -> None:
        registry = PricingRegistry()

        # o3-pro must NOT match o3
        o3_pro = registry.resolve("o3-pro")
        o3 = registry.resolve("o3")
        self.assertEqual(o3_pro.model, "o3-pro")
        self.assertEqual(o3_pro.input_cost_per_1m, 20.00)
        self.assertEqual(o3.input_cost_per_1m, 2.00)

        # gpt-5 must NOT match gpt-4
        gpt5 = registry.resolve("gpt-5")
        self.assertEqual(gpt5.model, "gpt-5")
        self.assertEqual(gpt5.input_cost_per_1m, 1.25)

        # Claude 4.5 Opus ($5/$25) and Claude 4.5 Haiku ($1/$5)
        opus45 = registry.resolve("claude-opus-4-5-20251101")
        self.assertEqual(opus45.input_cost_per_1m, 5.00)
        self.assertEqual(opus45.output_cost_per_1m, 25.00)

        haiku45 = registry.resolve("claude-haiku-4-5")
        self.assertEqual(haiku45.input_cost_per_1m, 1.00)
        self.assertEqual(haiku45.output_cost_per_1m, 5.00)

        # Gemini 3 Pro & Flash
        gem3 = registry.resolve("gemini-3-pro-preview")
        self.assertEqual(gem3.input_cost_per_1m, 2.00)
        gem3_flash = registry.resolve("gemini-3-flash")
        self.assertEqual(gem3_flash.input_cost_per_1m, 0.50)

        # Unknown model emits a UserWarning instead of silently falling back
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            unknown = registry.resolve("nonexistent-model-999")
            self.assertEqual(unknown.input_cost_per_1m, 0.50)
            self.assertTrue(
                any(issubclass(w.category, UserWarning) and "nonexistent-model-999" in str(w.message) for w in caught_warnings)
            )

    # ------------------------------------------------------------------
    # Bug #11: Non-JSON config & unwritable log path never break LLM calls
    # ------------------------------------------------------------------
    def test_bug_11_non_json_config_and_unwritable_log_never_raise(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # 1. Non-JSON-serializable object in config works without raising
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
            )

            @tally.track(
                feature="non_json_cfg",
                model="gpt-4o-mini",
                config={"custom_obj": object(), "callback": lambda x: x},
            )
            def call_ok() -> dict:
                return {"ok": True, "usage": {"input_tokens": 10, "output_tokens": 10}}

            self.assertEqual(call_ok()["ok"], True)

            # 2. Unwritable log path (e.g. pointing to a directory as a file) emits warning, never breaks call
            bad_log_dir = Path(td) / "dir_as_file"
            bad_log_dir.mkdir()
            tally_bad_io = LLMTally(
                log_path=bad_log_dir,
                dashboard_path=Path(td) / "dash.html",
            )

            @tally_bad_io.track(feature="io_resilient", model="gpt-4o-mini")
            def succeed_despite_io() -> str:
                return "paid-llm-response"

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                self.assertEqual(succeed_despite_io(), "paid-llm-response")

    # ------------------------------------------------------------------
    # Bug #12: Generators & streams are not cached as exhausted iterators
    # ------------------------------------------------------------------
    def test_bug_12_generators_and_streams_work_on_repeated_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
            )

            @tally.track(feature="stream_chat", model="gpt-4o-mini", cache=True)
            def stream_tokens(prompt: str) -> Iterator[str]:
                yield "Hello "
                yield prompt
                yield "!"

            first_run = list(stream_tokens("world"))
            second_run = list(stream_tokens("world"))
            self.assertEqual(first_run, ["Hello ", "world", "!"])
            self.assertEqual(second_run, ["Hello ", "world", "!"])

            records = tally.get_records()
            self.assertEqual(len(records), 2)
            self.assertGreater(records[0].output_tokens, 0)

            # Also verify async generators
            @tally.track(feature="async_stream_chat", model="gpt-4o-mini", cache=True)
            async def async_stream_tokens(prompt: str) -> AsyncIterator[str]:
                yield "Async "
                yield prompt

            async def _consume() -> None:
                out1 = [c async for c in async_stream_tokens("one")]
                out2 = [c async for c in async_stream_tokens("one")]
                self.assertEqual(out1, ["Async ", "one"])
                self.assertEqual(out2, ["Async ", "one"])

            asyncio.run(_consume())

    # ------------------------------------------------------------------
    # Bug #13: Mutating a returned cached object does not corrupt the cache
    # ------------------------------------------------------------------
    def test_bug_13_mutating_cached_result_does_not_corrupt_subsequent_hits(self) -> None:
        cache = InMemoryCache()
        cache.store("k1", {}, {"items": ["a", "b"]})

        hit1, val1, _ = cache.fetch("k1")
        self.assertTrue(hit1)
        val1["items"].append("MUTATED")
        val1["hacked"] = True

        hit2, val2, _ = cache.fetch("k1")
        self.assertTrue(hit2)
        self.assertEqual(val2, {"items": ["a", "b"]})
        self.assertNotIn("hacked", val2)

    # ------------------------------------------------------------------
    # Bug #14: Cache purges expired entries & enforces max_size
    # ------------------------------------------------------------------
    def test_bug_14_cache_purges_expired_and_bounds_max_size(self) -> None:
        cache = InMemoryCache(default_ttl=0.03, max_size=5)
        for i in range(5):
            cache.store(f"exp-{i}", {}, f"val-{i}", ttl=0.03)
        self.assertEqual(cache.size, 5)

        # Wait for TTL expiration; checking cache.size or storing a new item purges all 5 expired items
        time.sleep(0.05)
        self.assertEqual(cache.size, 0)

        # Now insert 10 long-lived items into a cache with max_size=5
        for i in range(10):
            cache.store(f"live-{i}", {}, f"val-{i}", ttl=300.0)
        self.assertEqual(cache.size, 5)
        # Only the 5 newest items (live-5 .. live-9) remain
        self.assertIsNone(cache.get_entry("live-0"))
        self.assertIsNotNone(cache.get_entry("live-9"))

    # ------------------------------------------------------------------
    # Bug #15: Mixed failures (500, 500, then 400) preserve retry history
    # ------------------------------------------------------------------
    def test_bug_15_mixed_500_then_400_failure_logs_retries_and_attempt_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
                sleep_func=lambda _: None,
            )
            step = 0

            @tally.track(feature="mixed_fail", model="gpt-4o-mini")
            def fail_500_500_400() -> str:
                nonlocal step
                step += 1
                if step <= 2:
                    raise DummyHTTPError(f"Server error #{step}", status_code=500)
                raise DummyHTTPError("Bad request on attempt 3", status_code=400)

            with self.assertRaises(DummyHTTPError) as ctx:
                fail_500_500_400()

            self.assertEqual(ctx.exception.status_code, 400)
            records = tally.get_records()
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec.status, "failed")
            self.assertEqual(rec.retries, 2)
            self.assertEqual(len(rec.attempt_errors), 3)
            self.assertEqual(rec.attempt_errors[0]["status_code"], 500)
            self.assertEqual(rec.attempt_errors[1]["status_code"], 500)
            self.assertEqual(rec.attempt_errors[2]["status_code"], 400)


    # ------------------------------------------------------------------
    # Issue A: Custom objects without custom `__repr__` must not collide
    # ------------------------------------------------------------------
    def test_issue_A_custom_objects_without_repr_have_distinct_cache_keys(self) -> None:
        class Prompt:
            def __init__(self, text: str) -> None:
                self.text = text

        class SlottedPrompt:
            __slots__ = ("text",)

            def __init__(self, text: str) -> None:
                self.text = text

        with tempfile.TemporaryDirectory() as td:
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
            )

            @tally.track(feature="qa", model="gpt-4o-mini", cache=True)
            def ask(p: Any) -> str:
                return f"Answer for {p.text}"

            ans_france = ask(Prompt("capital of France?"))
            ans_japan = ask(Prompt("capital of Japan?"))
            self.assertEqual(ans_france, "Answer for capital of France?")
            self.assertEqual(ans_japan, "Answer for capital of Japan?")

            ans_slot_1 = ask(SlottedPrompt("first"))
            ans_slot_2 = ask(SlottedPrompt("second"))
            self.assertEqual(ans_slot_1, "Answer for first")
            self.assertEqual(ans_slot_2, "Answer for second")

    # ------------------------------------------------------------------
    # Issue B: Memory address in `__repr__` & truncated `...` array reprs
    # ------------------------------------------------------------------
    def test_issue_B_memory_address_repr_and_ndarray_buffer_hashing(self) -> None:
        # 1. Object without __dict__ whose __repr__ includes a memory address
        class CExtensionLike:
            __slots__ = ()

            def __repr__(self) -> str:
                return f"<CExtensionLike object at 0x{id(self):x}>"

        k1, _ = build_cache_key("fn", (CExtensionLike(),), {})
        k2, _ = build_cache_key("fn", (CExtensionLike(),), {})
        self.assertEqual(k1, k2)

        # 2. Array-like object whose __repr__ truncates interior values with "..."
        class FakeNdArray:
            def __init__(self, data: bytes) -> None:
                self._data = data
                self.shape = (len(data),)
                self.dtype = "uint8"

            def tobytes(self) -> bytes:
                return self._data

            def __repr__(self) -> str:
                return "array([0, 1, ..., 255])"

        arr1 = FakeNdArray(b"\x00\x01\xAA\xFF")
        arr2 = FakeNdArray(b"\x00\x01\xBB\xFF")
        ka1, _ = build_cache_key("fn", (arr1,), {})
        ka2, _ = build_cache_key("fn", (arr2,), {})
        self.assertNotEqual(ka1, ka2)

    # ------------------------------------------------------------------
    # Issue C: A stream that fails partway is logged as "failed"
    # ------------------------------------------------------------------
    def test_issue_C_stream_that_fails_partway_is_logged_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
            )

            @tally.track(feature="broken_stream", model="gpt-4o-mini")
            def partial_stream() -> Iterator[str]:
                yield "chunk-1 "
                raise RuntimeError("mid-stream disconnect")

            s = partial_stream()
            self.assertEqual(next(s), "chunk-1 ")
            with self.assertRaises(RuntimeError):
                next(s)

            records = tally.get_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "failed")
            self.assertIn("mid-stream disconnect", records[0].error_message or "")

    # ------------------------------------------------------------------
    # Issue D: A stream nobody iterates is still logged
    # ------------------------------------------------------------------
    def test_issue_D_stream_nobody_iterates_is_still_logged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
            )

            @tally.track(feature="unconsumed_stream", model="gpt-4o-mini")
            def make_stream(prompt: str) -> Iterator[str]:
                yield prompt

            _unused = make_stream("hello world")
            records = tally.get_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].feature, "unconsumed_stream")
            self.assertGreater(records[0].input_tokens, 0)

    # ------------------------------------------------------------------
    # Issue E: Anthropic stream `message_start.message.usage` + `message_delta.usage`
    # ------------------------------------------------------------------
    def test_issue_E_anthropic_stream_chunks_merge_input_and_output_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
            )

            @tally.track(feature="anthropic_stream", model="claude-3-5-sonnet")
            def stream_claude() -> Iterator[Any]:
                yield SimpleNamespace(
                    type="message_start",
                    message=SimpleNamespace(
                        model="claude-3-5-sonnet",
                        usage=SimpleNamespace(
                            input_tokens=250,
                            output_tokens=1,
                            cache_read_input_tokens=1000,
                            cache_creation_input_tokens=0,
                        ),
                    ),
                )
                yield SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(text="Hello from Claude"),
                )
                yield SimpleNamespace(
                    type="message_delta",
                    usage=SimpleNamespace(output_tokens=85),
                )

            chunks = list(stream_claude())
            self.assertEqual(len(chunks), 3)

            records = tally.get_records()
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec.input_tokens, 250)
            self.assertEqual(rec.cached_input_tokens, 1000)
            self.assertEqual(rec.output_tokens, 85)
            self.assertGreater(rec.cost_usd, 0.0)

    # ------------------------------------------------------------------
    # Issue F: SDK stream preserves `.close()`, `with ... as s:`, `.response`
    #          and built-in iterators (`iter(list)`, `map`) are not wrapped
    # ------------------------------------------------------------------
    def test_issue_F_sdk_stream_methods_preserved_and_builtin_iterators_unwrapped(self) -> None:
        class FakeSDKStream:
            def __init__(self) -> None:
                self.response = {"headers": {"x-request-id": "req-123"}}
                self.closed = False
                self._items = iter(["a", "b"])

            def __iter__(self) -> "FakeSDKStream":
                return self

            def __next__(self) -> str:
                return next(self._items)

            def close(self) -> None:
                self.closed = True

            def __enter__(self) -> "FakeSDKStream":
                return self

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
                self.close()
                return False

        with tempfile.TemporaryDirectory() as td:
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
            )

            @tally.track(feature="sdk_stream", model="gpt-4o-mini")
            def get_sdk_stream() -> FakeSDKStream:
                return FakeSDKStream()

            with get_sdk_stream() as stream:
                self.assertEqual(stream.response["headers"]["x-request-id"], "req-123")
                collected = list(stream)
                self.assertEqual(collected, ["a", "b"])
            self.assertTrue(stream.closed)

            # Built-in list_iterator and map should not be wrapped in _SyncStreamProxy
            @tally.track(feature="list_iter", model="gpt-4o-mini")
            def get_list_iter() -> Iterator[int]:
                return iter([10, 20, 30])

            it = get_list_iter()
            self.assertEqual(type(it).__name__, "list_iterator")
            self.assertEqual(list(it), [10, 20, 30])
            # Second call still returns a fresh iterator (not cached as exhausted!)
            self.assertEqual(list(get_list_iter()), [10, 20, 30])

    # ------------------------------------------------------------------
    # Issue G: Bedrock, Vertex, fine-tuned `ft:`, and modality model IDs
    # ------------------------------------------------------------------
    def test_issue_G_bedrock_vertex_finetuned_and_modality_model_ids(self) -> None:
        registry = PricingRegistry()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(
                registry.resolve("anthropic.claude-3-5-sonnet-20241022-v2:0").model,
                "claude-3-5-sonnet",
            )
            self.assertEqual(
                registry.resolve("us.anthropic.claude-sonnet-4-20250514-v1:0").model,
                "claude-sonnet-4",
            )
            self.assertEqual(
                registry.resolve("claude-3-5-sonnet-v2@20241022").model,
                "claude-3-5-sonnet",
            )
            self.assertEqual(
                registry.resolve("gpt-4o-audio-preview").input_cost_per_1m,
                2.50,
            )
            self.assertEqual(
                registry.resolve("gpt-5-chat-latest").input_cost_per_1m,
                1.25,
            )
            self.assertEqual(
                registry.resolve("gpt-5.1").input_cost_per_1m,
                1.25,
            )
            self.assertEqual(
                registry.resolve("gpt-4-32k").input_cost_per_1m,
                60.00,
            )
            self.assertEqual(
                registry.resolve("ft:gpt-4o-mini-2024-07-18:my-org:custom:123").model,
                "gpt-4o-mini",
            )
            self.assertEqual(len(caught), 0)

    # ------------------------------------------------------------------
    # Issue H: Custom-registered Claude model uses Anthropic cache billing rule
    # ------------------------------------------------------------------
    def test_issue_H_custom_registered_claude_model_uses_anthropic_billing_rule(self) -> None:
        registry = PricingRegistry()
        # Registered with default provider="custom"
        registry.register("proxy-claude-sonnet", input_cost_per_1m=3.00, output_cost_per_1m=15.00)
        cost = registry.calculate_cost(
            "proxy-claude-sonnet",
            input_tokens=100,
            output_tokens=0,
            cached_input_tokens=1_000_000,
        )
        self.assertAlmostEqual(cost, 0.3003, places=7)

        # Even if model name doesn't contain "claude", an Anthropic response with cache_read_input_tokens sets input_excludes_cached=True
        registry.register("my-gateway-model", input_cost_per_1m=3.00, output_cost_per_1m=15.00)
        resp = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=0,
                cache_read_input_tokens=1_000_000,
            )
        )
        u = extract_token_usage(resp)
        self.assertTrue(u.input_excludes_cached)
        cost2 = registry.calculate_cost(
            "my-gateway-model",
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cached_input_tokens=u.cached_input_tokens,
            input_excludes_cached=u.input_excludes_cached,
        )
        self.assertAlmostEqual(cost2, 0.3003, places=7)

    # ------------------------------------------------------------------
    # Issue I: Unknown model under `-W error` does not raise after a paid call
    # ------------------------------------------------------------------
    def test_issue_I_unknown_model_under_warnings_as_errors_does_not_fail_call(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
            )

            @tally.track(feature="strict_warnings", model="totally-unknown-2027-model")
            def paid_call() -> dict:
                return {"answer": "success", "usage": {"input_tokens": 1000, "output_tokens": 500}}

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                res = paid_call()
                self.assertEqual(res["answer"], "success")

            records = tally.get_records()
            self.assertEqual(len(records), 1)
            self.assertGreater(records[0].cost_usd, 0.0)

    # ------------------------------------------------------------------
    # Issue J: Result holding an unpicklable lock still clones mutable fields
    # ------------------------------------------------------------------
    def test_issue_J_result_with_unpicklable_lock_still_clones_mutable_data(self) -> None:
        import threading

        class SDKResponseWithLock:
            def __init__(self, text: str) -> None:
                self._lock = threading.Lock()
                self.data = {"text": text, "items": ["x", "y"]}

        cache = InMemoryCache()
        original = SDKResponseWithLock("immutable-answer")
        cache.store("lock-key", {}, original)

        hit1, res1, _ = cache.fetch("lock-key")
        self.assertTrue(hit1)
        # Mutate the returned object's nested dict & list
        res1.data["text"] = "CORRUPTED"
        res1.data["items"].append("HACKED")

        hit2, res2, _ = cache.fetch("lock-key")
        self.assertTrue(hit2)
        self.assertEqual(res2.data["text"], "immutable-answer")
        self.assertEqual(res2.data["items"], ["x", "y"])

    # ------------------------------------------------------------------
    # Issue M: Dashboard single-pass placeholder replacement
    # ------------------------------------------------------------------
    def test_issue_M_dashboard_placeholders_in_prompt_and_title_do_not_corrupt_json(self) -> None:
        import json as _json
        import re as _re

        from llm_tally.dashboard import render_dashboard_html

        with tempfile.TemporaryDirectory() as td:
            tally = LLMTally(
                log_path=Path(td) / "logs.jsonl",
                dashboard_path=Path(td) / "dashboard.html",
                auto_update_dashboard=False,
            )

            @tally.track(feature="feat_{{DASHBOARD_JS}}", model="gpt-4o-mini")
            def ask(prompt: str) -> str:
                return f"Echo: {prompt} and {{{{REPORT_JSON}}}} </script>"

            ask("What does {{DASHBOARD_JS}} or {{DASHBOARD_CSS}} or {{TITLE}} do?")
            html_out = render_dashboard_html(
                tally.get_records(),
                title="My Title {{REPORT_JSON}} & {{DASHBOARD_JS}} <script>",
            )

            # Extract the embedded JSON from `const REPORT_DATA = ...;`
            match = _re.search(
                r"const REPORT_DATA = (\{.*?\});\n",
                html_out,
                flags=_re.DOTALL,
            )
            self.assertIsNotNone(match)
            parsed = _json.loads(match.group(1))
            self.assertEqual(len(parsed["recent_calls"]), 1)
            rec = parsed["recent_calls"][0]
            self.assertIn("{{DASHBOARD_JS}}", rec["input_preview"])
            self.assertIn("{{REPORT_JSON}}", rec["output_preview"])
            self.assertIn("feat_{{DASHBOARD_JS}}", rec["feature"])
            # Ensure the dashboard JS was injected only once, not pasted into the JSON
            self.assertEqual(html_out.count("function renderSummaryAndKpis("), 1)
            # Title placeholders must be literal HTML-escaped text, not expanded to JSON/JS
            self.assertIn(
                "<title>My Title {{REPORT_JSON}} &amp; {{DASHBOARD_JS}} &lt;script&gt;</title>",
                html_out,
            )

    # ------------------------------------------------------------------
    # Issue N: Concurrent multiprocess streaming calls do not lose log records
    # ------------------------------------------------------------------
    def test_issue_N_multiprocess_stream_updates_preserve_all_records(self) -> None:
        import multiprocessing

        with tempfile.TemporaryDirectory() as td:
            log_file = Path(td) / "mp_logs.jsonl"
            ctx = multiprocessing.get_context("spawn")
            procs = [
                ctx.Process(target=_mp_stream_worker, args=(str(log_file), wid, 120))
                for wid in range(4)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=60)
                self.assertEqual(p.exitcode, 0)

            from llm_tally.tracker import JSONLLogger

            logger = JSONLLogger(log_file)
            records = logger.read_all()
            self.assertEqual(len(records), 480)
            self.assertTrue(all(r.output_tokens > 0 for r in records))
            # The log is append-only: each stream writes a placeholder line and a final line
            raw_lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(raw_lines), 960)
            # `compact()` rewrites it to one line per record without losing anything
            self.assertEqual(logger.compact(), 480)
            raw_lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(raw_lines), 480)
            self.assertEqual(len(JSONLLogger(log_file).read_all()), 480)


def _mp_stream_worker(log_path: str, worker_id: int, count: int) -> None:
    """Top-level multiprocessing worker that runs `count` tracked streaming calls."""
    tally = LLMTally(
        log_path=log_path,
        auto_update_dashboard=False,
    )

    class _WorkerStream:
        def __init__(self, idx: int) -> None:
            self._items = [f"w{worker_id}-{idx}-chunk1 ", f"w{worker_id}-{idx}-chunk2"]
            self._pos = 0

        def __iter__(self) -> "_WorkerStream":
            return self

        def __next__(self) -> str:
            if self._pos >= len(self._items):
                raise StopIteration
            val = self._items[self._pos]
            self._pos += 1
            return val

    @tally.track(feature=f"worker_{worker_id}", model="gpt-4o-mini", retry=False)
    def stream_call(idx: int) -> _WorkerStream:
        return _WorkerStream(idx)

    for i in range(count):
        s = stream_call(i)
        list(s)

