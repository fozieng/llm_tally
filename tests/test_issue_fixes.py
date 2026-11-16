"""Regression tests for the issues listed in the code review (issues.md)."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_tally import LLMTally, PricingRegistry, RetryExhaustedError  # noqa: E402
from llm_tally.cli import main as cli_main  # noqa: E402
from llm_tally.tracker import CallRecord, analyze_logs, extract_token_usage  # noqa: E402


def usage(i: int, o: int) -> dict:
    return {"content": "x", "usage": {"input_tokens": i, "output_tokens": o}}


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.base = Path(self._td.name)
        warnings.simplefilter("ignore")

    def tearDown(self) -> None:
        warnings.resetwarnings()
        self._td.cleanup()

    def make(self, **kwargs) -> LLMTally:
        kwargs.setdefault("log_path", self.base / "logs.jsonl")
        kwargs.setdefault("dashboard_path", self.base / "dashboard.html")
        kwargs.setdefault("auto_update_dashboard", False)
        kwargs.setdefault("sleep_func", lambda _: None)
        return LLMTally(**kwargs)


class ModelAttributionTests(Base):
    def test_M1_no_model_is_unknown_and_unpriced_not_gpt_4o_mini(self) -> None:
        tally = self.make()

        @tally.track(feature="chat")
        def ask(prompt: str) -> str:
            return "Paris"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ask("capital of France?")
            ask("capital of Spain?")
        rec = tally.get_records()[-1]
        self.assertEqual(rec.model, "unknown")
        self.assertEqual(rec.cost_usd, 0.0)
        self.assertTrue(rec.unpriced)
        self.assertTrue(rec.estimated_tokens)
        relevant = [w for w in caught if "could not tell which model" in str(w.message)]
        self.assertEqual(len(relevant), 1, "warn once per model, not per call")
        self.assertEqual(Path(relevant[0].filename).name, "test_issue_fixes.py")

    def test_M1_explicit_default_model_still_works(self) -> None:
        tally = self.make(default_model="gpt-4o")

        @tally.track()
        def ask(prompt: str) -> dict:
            return usage(1_000_000, 0)

        ask("hi")
        rec = tally.get_records()[-1]
        self.assertEqual(rec.model, "gpt-4o")
        self.assertAlmostEqual(rec.cost_usd, 2.5)
        self.assertFalse(rec.unpriced)

    def test_M2_model_from_default_parameter_and_positional_argument(self) -> None:
        tally = self.make()

        @tally.track()
        def ask(prompt: str, model: str = "claude-sonnet-4-5") -> dict:
            return usage(1_000_000, 0)

        ask("hi")
        ask("hi", "gpt-4o")
        ask("hi", model="gpt-4o-mini")
        self.assertEqual(
            [r.model for r in tally.get_records()], ["claude-sonnet-4-5", "gpt-4o", "gpt-4o-mini"]
        )
        self.assertAlmostEqual(tally.get_records()[0].cost_usd, 3.0)

    def test_M2_method_with_model_attribute_style_call(self) -> None:
        tally = self.make()

        class Bot:
            @tally.track()
            def ask(self, prompt: str, model: str = "gpt-4o") -> dict:
                return usage(10, 10)

        Bot().ask("hi")
        self.assertEqual(tally.get_records()[-1].model, "gpt-4o")

    def test_M4_langchain_message_model(self) -> None:
        tally = self.make()

        @tally.track()
        def lc(prompt: str):
            return NS(
                content="hello",
                usage_metadata={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
                response_metadata={"model_name": "claude-sonnet-4-5-20250929"},
            )

        lc("p")
        rec = tally.get_records()[-1]
        self.assertEqual(rec.model, "claude-sonnet-4-5-20250929")
        self.assertFalse(rec.unpriced)
        self.assertEqual(rec.output_preview, "hello")


class CostTests(Base):
    def test_C1_unknown_model_is_flagged_and_reported(self) -> None:
        tally = self.make()

        @tally.track(model="claude-opus-9")
        def ask(prompt: str) -> dict:
            return usage(1_000_000, 1_000_000)

        ask("hi")
        rec = tally.get_records()[-1]
        self.assertEqual(rec.cost_usd, 0.0)
        self.assertTrue(rec.unpriced)
        overview = tally.get_report()["overview"]
        self.assertEqual(overview["unpriced_calls"], 1)
        self.assertEqual(overview["unpriced_models"], {"claude-opus-9": 1})

    def test_C1_fallback_pricing_is_opt_in_and_still_flagged(self) -> None:
        tally = self.make(fallback_pricing=(1.0, 2.0))

        @tally.track(model="claude-opus-9")
        def ask(prompt: str) -> dict:
            return usage(1_000_000, 1_000_000)

        ask("hi")
        rec = tally.get_records()[-1]
        self.assertAlmostEqual(rec.cost_usd, 3.0)
        self.assertTrue(rec.unpriced)

    def test_C3_registering_a_model_reprices_past_calls(self) -> None:
        tally = self.make()

        @tally.track(model="house-model")
        def ask(prompt: str) -> dict:
            return usage(1_000_000, 1_000_000)

        ask("hi")
        self.assertEqual(tally.get_report()["spend_summary"]["all_time_usd"], 0.0)
        tally.register_model("house-model", input_cost_per_1m=1.0, output_cost_per_1m=4.0)
        report = tally.get_report()
        self.assertAlmostEqual(report["spend_summary"]["all_time_usd"], 5.0)
        self.assertEqual(report["overview"]["unpriced_calls"], 0)
        rec = tally.get_records()[-1]
        self.assertFalse(rec.unpriced)
        self.assertAlmostEqual(rec.cost_usd, 5.0)

    def test_C3_cli_pricing_file_reprices(self) -> None:
        tally = self.make()

        @tally.track(model="house-model")
        def ask(prompt: str) -> dict:
            return usage(1_000_000, 0)

        ask("hi")
        prices = self.base / "prices.json"
        prices.write_text(json.dumps({"house-model": {"input_cost_per_1m": 7, "output_cost_per_1m": 7}}))
        out = self.base / "report.json"
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli_main(["report", "--log", str(self.base / "logs.jsonl"), "--json",
                      "--pricing-file", str(prices)])
        report = json.loads(buf.getvalue())
        self.assertAlmostEqual(report["spend_summary"]["all_time_usd"], 7.0)
        del out

    def test_C5_C6_model_id_resolution(self) -> None:
        reg = PricingRegistry()
        self.assertEqual(reg.lookup("gpt-4-1106-preview").input_cost_per_1m, 10.0)
        self.assertEqual(reg.lookup("gpt-4-0125-preview").output_cost_per_1m, 30.0)
        self.assertEqual(reg.lookup("gpt-4o-2024-05-13").input_cost_per_1m, 5.0)
        self.assertEqual(reg.lookup("gpt-4o-2024-08-06").input_cost_per_1m, 2.5)
        self.assertIsNone(reg.lookup("gpt-4o-transcribe"))
        self.assertIsNone(reg.lookup("gpt-4o-mini-tts"))
        self.assertEqual(reg.lookup("gpt-4o-audio-preview-2024-12-17").model, "gpt-4o-audio")
        free = reg.lookup("google/gemini-2.0-flash-exp:free")
        self.assertEqual((free.input_cost_per_1m, free.output_cost_per_1m), (0.0, 0.0))
        for bedrock in (
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "apac.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ):
            self.assertEqual(reg.lookup(bedrock).model, "claude-sonnet-4-5", bedrock)
        self.assertIsNone(reg.lookup("unknown"))
        self.assertIsNone(reg.lookup(None))

    def test_C7_gemini_thinking_tokens_are_billed(self) -> None:
        u = extract_token_usage(
            NS(model_version="gemini-2.5-pro", usage_metadata=NS(
                prompt_token_count=1000, candidates_token_count=100, thoughts_token_count=5000,
                tool_use_prompt_token_count=20, cached_content_token_count=None))
        )
        self.assertEqual((u.input_tokens, u.output_tokens), (1020, 5100))

    def test_C8_openai_responses_api_stream(self) -> None:
        tally = self.make()

        @tally.track(feature="resp_stream")
        def stream(prompt: str):
            def gen():
                yield NS(type="response.output_text.delta", delta="Hello ")
                yield NS(type="response.output_text.delta", delta="world")
                yield NS(type="response.completed", response=NS(
                    model="gpt-4o-2024-08-06",
                    usage=NS(input_tokens=1000, output_tokens=2000, total_tokens=3000)))
            return gen()

        for _ in stream("hi"):
            pass
        rec = tally.get_records()[-1]
        self.assertEqual(rec.model, "gpt-4o-2024-08-06")
        self.assertEqual((rec.input_tokens, rec.output_tokens), (1000, 2000))
        self.assertFalse(rec.estimated_tokens)
        self.assertEqual(rec.output_preview, "Hello world")


class DashboardTests(Base):
    @staticmethod
    def _rec(i: int, **kw) -> CallRecord:
        base = dict(id=f"r{i}", timestamp=datetime.now(timezone.utc).isoformat(), feature="f",
                    function_name="fn", model="gpt-4o", config={}, status="success",
                    cache_hit=False, latency_ms=100.0, input_tokens=1, output_tokens=1,
                    cached_input_tokens=0, cost_usd=0.0, retries=0)
        base.update(kw)
        return CallRecord(**base)

    def test_D1_top_feature_adds_up_configs(self) -> None:
        recs = [
            self._rec(1, feature="A", config={"t": 0}, cost_usd=3.0),
            self._rec(2, feature="A", config={"t": 1}, cost_usd=3.0),
            self._rec(3, feature="B", cost_usd=4.0),
        ]
        report = analyze_logs(recs)
        self.assertEqual(report["overview"]["top_cost_feature"], "A")
        self.assertEqual(report["feature_totals"][0]["total_cost_usd"], 6.0)
        self.assertEqual(report["feature_totals"][0]["configs"], 2)

    def test_D2_latency_ignores_cache_hits_and_failures(self) -> None:
        recs = [self._rec(0, latency_ms=200.0)]
        recs += [self._rec(i, latency_ms=0.1, cache_hit=True, status="cached") for i in range(1, 10)]
        recs += [self._rec(10, latency_ms=5000.0, status="failed")]
        row = analyze_logs(recs)["latency_stats"][0]
        self.assertEqual(row["avg_latency_ms"], 200.0)
        self.assertEqual(row["live_calls"], 1)
        self.assertEqual(row["total_calls"], 11)

    def test_D3_recent_calls_limit_and_total(self) -> None:
        recs = [self._rec(i) for i in range(30)]
        report = analyze_logs(recs, recent_limit=10)
        self.assertEqual(len(report["recent_calls"]), 10)
        self.assertEqual(report["recent_calls_total"], 30)

    def test_D8_days_use_the_given_time_zone(self) -> None:
        tz = timezone(timedelta(hours=-8))
        now = datetime(2026, 1, 10, 20, 0, tzinfo=tz)  # 04:00 UTC on Jan 11
        recs = [self._rec(1, timestamp=datetime(2026, 1, 11, 3, 0, tzinfo=timezone.utc).isoformat(),
                          cost_usd=1.0)]
        report = analyze_logs(recs, now=now)
        self.assertEqual(report["spend_summary"]["today_usd"], 1.0)
        self.assertEqual(report["weekly_daily_spend"][-1]["date"], "2026-01-10")
        self.assertEqual(report["timezone"], "UTC-08:00")

    def test_D12_stream_placeholder_and_early_close(self) -> None:
        tally = self.make()

        @tally.track(feature="s", model="gpt-4o")
        def s(prompt: str):
            def gen():
                for _ in range(100):
                    yield {"content": "tok "}
            return gen()

        it = s("hello")
        self.assertEqual(tally.get_records()[-1].status, "streaming")
        next(it)
        it.close()
        rec = tally.get_records()[-1]
        self.assertEqual(rec.status, "incomplete")
        self.assertEqual(rec.output_tokens, 1)

    def test_D13_dashboard_file_is_readable_by_others(self) -> None:
        tally = self.make()

        @tally.track(model="gpt-4o")
        def ask(prompt: str) -> dict:
            return usage(1, 1)

        ask("x")
        path = tally.generate_dashboard()
        mode = stat.S_IMODE(os.stat(path).st_mode)
        self.assertEqual(mode, 0o644)

    def test_dashboard_contains_unpriced_notice_data(self) -> None:
        tally = self.make()

        @tally.track(model="mystery-model")
        def ask(prompt: str) -> dict:
            return usage(1, 1)

        ask("x")
        html_text = tally.render_dashboard()
        self.assertIn('"unpriced_calls": 1', html_text)
        self.assertIn('"pricing_as_of"', html_text)


class UsabilityTests(Base):
    def test_U1_run_passes_declared_model_and_metadata_to_the_function(self) -> None:
        tally = self.make()
        seen = {}

        def create(*, model: str, messages: list, metadata=None):
            seen.update(model=model, metadata=metadata)
            return {"model": model, "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}

        tally.run(create, model="gpt-4o", messages=[], metadata={"user_id": "u1"})
        self.assertEqual(seen, {"model": "gpt-4o", "metadata": {"user_id": "u1"}})
        rec = tally.get_records()[-1]
        self.assertEqual(rec.model, "gpt-4o")
        self.assertAlmostEqual(rec.cost_usd, 2.5)
        self.assertEqual(rec.metadata, {})

        tally.run(create, model="gpt-4o", messages=[], metadata={"x": "1"},
                  tally_options={"metadata": {"tenant": "acme"}, "feature": "chat"})
        rec = tally.get_records()[-1]
        self.assertEqual(rec.metadata, {"tenant": "acme"})
        self.assertEqual(rec.feature, "chat")
        with self.assertRaises(TypeError):
            tally.run(create, model="gpt-4o", messages=[], tally_options={"nope": 1})

    def test_U1_run_bound_method_and_kwargs_function(self) -> None:
        tally = self.make()

        class Completions:
            def create(self, *, model: str, messages: list):
                return {"model": model, "usage": {"input_tokens": 1, "output_tokens": 1}}

        tally.run(Completions().create, model="gpt-4o", messages=[])
        self.assertEqual(tally.get_records()[-1].model, "gpt-4o")

        def passthrough(**kw):
            return {"got": kw, "usage": {"input_tokens": 1, "output_tokens": 1}}

        out = tally.run(passthrough, model="gpt-4o", feature="x")
        self.assertEqual(out["got"], {"model": "gpt-4o"})

    def test_U1_cached_call_and_cache_decorator_with_colliding_names(self) -> None:
        tally = self.make()
        calls = []

        def f(prompt: str, ttl: int = 0, model: str = "gpt-4o"):
            calls.append((prompt, ttl, model))
            return usage(1, 1)

        tally.cached_call(f, "a", ttl=5, model="gpt-4o-mini")
        self.assertEqual(calls[-1], ("a", 5, "gpt-4o-mini"))

        cached = tally.cache(ttl=60)(f)
        cached("b", ttl=7)
        cached("b", ttl=7)
        self.assertEqual(calls[-1], ("b", 7, "gpt-4o"))
        self.assertEqual(len(calls), 2)

    def test_U3_U4_previews_mask_secrets_and_skip_self(self) -> None:
        tally = self.make()

        @tally.track(model="gpt-4o")
        def call(prompt: str, api_key: str = "", max_tokens: int = 5, headers=None):
            return NS(choices=[NS(message=NS(content="the answer"))],
                      usage=NS(prompt_tokens=1, completion_tokens=1))

        call("hi", api_key="sk-live-SECRET", max_tokens=5, headers={"Authorization": "Bearer abc"})
        rec = tally.get_records()[-1]
        self.assertNotIn("SECRET", rec.input_preview)
        self.assertNotIn("Bearer", rec.input_preview)
        self.assertIn('"max_tokens": 5', rec.input_preview)
        self.assertEqual(rec.output_preview, "the answer")

        class Bot:
            @tally.track(model="gpt-4o")
            def ask(self, prompt: str) -> dict:
                return usage(1, 1)

        Bot().ask("just the prompt")
        self.assertEqual(tally.get_records()[-1].input_preview, "just the prompt")

    def test_U6_deep_self_never_breaks_the_call(self) -> None:
        tally = self.make()

        class Node:
            def __init__(self, nxt=None):
                self.nxt = nxt

        head = None
        for _ in range(5000):
            head = Node(head)

        class Bot:
            def __init__(self) -> None:
                self.state = head
                self.calls = 0

            @tally.track(model="gpt-4o")
            def ask(self, prompt: str) -> dict:
                self.calls += 1
                return usage(1, 1)

            @tally.track(model="gpt-4o", cache=True)
            def ask_cached(self, prompt: str) -> dict:
                self.calls += 1
                return usage(1, 1)

        bot = Bot()
        bot.ask("hi")
        bot.ask_cached("hi")
        self.assertEqual(bot.calls, 2)
        self.assertEqual(len(tally.get_records()), 2)

    def test_U2_cli_reports_missing_log(self) -> None:
        import contextlib
        import io

        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = cli_main(["report", "--log", str(self.base / "missing.jsonl")])
        self.assertEqual(code, 0)
        self.assertIn("no log file", err.getvalue())


class RetryTests(Base):
    class RateLimitError(Exception):
        status_code = 429

    def test_R2_no_retries_raises_the_original_exception(self) -> None:
        tally = self.make(default_retries=0)
        err_cls = self.RateLimitError

        @tally.track(model="gpt-4o")
        def boom():
            raise err_cls("slow down")

        with self.assertRaises(err_cls) as ctx:
            boom()
        self.assertEqual(len(ctx.exception.llm_tally_attempts), 1)
        self.assertEqual(tally.get_records()[-1].status, "failed")

    def test_R2_raise_original_error_option(self) -> None:
        tally = self.make(raise_original_error=True)
        err_cls = self.RateLimitError

        @tally.track(model="gpt-4o")
        def boom():
            raise err_cls("slow down")

        with self.assertRaises(err_cls) as ctx:
            boom()
        self.assertEqual(len(ctx.exception.llm_tally_attempts), 4)
        rec = tally.get_records()[-1]
        self.assertEqual(rec.retries, 3)
        self.assertEqual(len(rec.attempt_errors), 4)

        default = self.make(log_path=self.base / "other.jsonl")

        @default.track(model="gpt-4o")
        def boom2():
            raise err_cls("slow down")

        with self.assertRaises(RetryExhaustedError):
            boom2()

    def test_R3_returned_payload_with_status_field_is_not_retried(self) -> None:
        tally = self.make()
        calls = []

        @tally.track(model="gpt-4o")
        def job():
            calls.append(1)
            return {"status": 503, "detail": "app payload"}

        self.assertEqual(job()["status"], 503)
        self.assertEqual(len(calls), 1)

        @tally.track(model="gpt-4o")
        def http_like():
            calls.append(1)
            return NS(status=503, headers={})

        with self.assertRaises(RetryExhaustedError):
            http_like()


class ProductionTests(Base):
    def test_S3_rotation_keeps_report_totals(self) -> None:
        # Enough backups to hold every record: totals then never drop at a rotation
        # (with fewer backups the oldest file is deleted by design)
        tally = self.make(max_log_mb=0.002, log_backups=20)

        @tally.track(model="gpt-4o")
        def ask(prompt: str) -> dict:
            return usage(1, 1)

        seen = []
        for i in range(30):
            ask(f"prompt {i} " + "x" * 40)
            seen.append(tally.get_report()["overview"]["total_calls"])
        self.assertTrue(Path(f"{self.base / 'logs.jsonl'}.1").exists())
        self.assertEqual(seen, sorted(seen), "totals must not drop after a rotation")
        self.assertEqual(seen[-1], 30)

    def test_S4_sinks_receive_records_when_log_is_unwritable(self) -> None:
        ro = self.base / "ro"
        ro.mkdir()
        os.chmod(ro, 0o500)
        try:
            got = []
            tally = self.make(log_path=ro / "sub" / "logs.jsonl", on_record=got.append)

            @tally.track(model="gpt-4o")
            def ask(prompt: str) -> dict:
                return usage(1, 1)

            ask("x")
            self.assertEqual(len(got), 1)
        finally:
            os.chmod(ro, 0o700)

    def test_S2_large_log_dashboard_builds_in_background(self) -> None:
        tally = self.make(auto_update_dashboard=True, dashboard_update_interval=0.0)
        tally._last_dashboard_build_seconds = 1.0  # pretend rebuilds are slow
        started = threading.Event()
        real_generate = tally.generate_dashboard

        def slow_generate(*a, **k):
            started.set()
            self.assertIsNot(threading.current_thread(), threading.main_thread())
            return real_generate(*a, **k)

        tally.generate_dashboard = slow_generate  # type: ignore[assignment]

        @tally.track(model="gpt-4o")
        def ask(prompt: str) -> dict:
            return usage(1, 1)

        ask("x")
        self.assertTrue(started.wait(5))
        # Let the background build finish before tearDown deletes its directory
        timer = tally._dashboard_timer
        if timer is not None:
            timer.join(5)
        for t in threading.enumerate():
            if isinstance(t, threading.Timer) and t is not threading.current_thread():
                t.join(5)

    def test_async_calls_still_work(self) -> None:
        tally = self.make()

        @tally.track()
        async def ask(prompt: str, model: str = "gpt-4o") -> dict:
            return usage(1_000_000, 0)

        asyncio.run(ask("hi"))
        rec = tally.get_records()[-1]
        self.assertEqual(rec.model, "gpt-4o")
        self.assertEqual(rec.input_preview, "hi")


if __name__ == "__main__":
    unittest.main()
