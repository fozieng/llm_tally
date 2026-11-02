from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from django_llm_tally import (
    LLMTally,
    PricingRegistry,
    RetryExhaustedError,
    extract_token_usage,
)


def test_builtin_pricing_includes_openai_anthropic_and_gemini() -> None:
    registry = PricingRegistry()

    # OpenAI
    gpt4o = registry.resolve("gpt-4o-2024-08-06")
    assert gpt4o.provider == "openai"
    assert gpt4o.model == "gpt-4o"

    # Anthropic
    claude = registry.resolve("claude-3-7-sonnet-20250219")
    assert claude.provider == "anthropic"
    assert claude.model == "claude-3-7-sonnet"

    # Gemini
    gemini = registry.resolve("models/gemini-2.5-flash")
    assert gemini.provider == "gemini"
    assert gemini.model == "gemini-2.5-flash"

    # Custom model registration
    registry.register("custom-llama-70b", input_cost_per_1m=0.80, output_cost_per_1m=2.40)
    cost = registry.calculate_cost("custom-llama-70b", input_tokens=500_000, output_tokens=250_000)
    assert abs(cost - (0.40 + 0.60)) < 1e-7


def test_token_extraction_across_openai_anthropic_and_gemini_shapes() -> None:
    # OpenAI style
    openai_resp = SimpleNamespace(
        model="gpt-4o",
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500),
    )
    u_oai = extract_token_usage(openai_resp)
    assert u_oai.input_tokens == 1000
    assert u_oai.output_tokens == 500
    assert u_oai.detected_model == "gpt-4o"
    assert not u_oai.estimated

    # Anthropic style
    anth_resp = SimpleNamespace(
        model="claude-3-5-sonnet",
        usage=SimpleNamespace(input_tokens=2000, output_tokens=300, cache_read_input_tokens=500),
    )
    u_anth = extract_token_usage(anth_resp)
    assert u_anth.input_tokens == 2000
    assert u_anth.output_tokens == 300
    assert u_anth.cached_input_tokens == 500

    # Gemini style
    gemini_resp = SimpleNamespace(
        model_version="gemini-2.5-pro",
        usage_metadata=SimpleNamespace(
            prompt_token_count=4000,
            candidates_token_count=1000,
            cached_content_token_count=0,
        ),
    )
    u_gem = extract_token_usage(gemini_resp)
    assert u_gem.input_tokens == 4000
    assert u_gem.output_tokens == 1000
    assert u_gem.detected_model == "gemini-2.5-pro"


def test_cost_analysis_and_html_dashboard_answers_all_questions(tmp_path: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as td:
        base = tmp_path or Path(td)
        log_file = base / "logs.jsonl"
        dash_file = base / "dashboard.html"

        tally = LLMTally(
            default_model="gpt-4o-mini",
            log_path=log_file,
            dashboard_path=dash_file,
            sleep_func=lambda _: None,  # Instant retries for fast tests
        )

        @tally.track(
            feature="document_analyzer",
            model="claude-opus-4",
            config={"depth": "deep"},
        )
        def expensive_feature(doc: str) -> dict:
            return {"summary": "done", "usage": {"input_tokens": 100_000, "output_tokens": 20_000}}

        @tally.track(
            feature="quick_tagger",
            model="gemini-2.5-flash",
            config={"depth": "shallow"},
        )
        def cheap_feature(text: str) -> dict:
            return {"tags": ["ai"], "usage": {"input_tokens": 1_000, "output_tokens": 100}}

        @tally.track(
            feature="unreliable_webhook",
            model="gpt-4o",
        )
        def failing_feature() -> dict:
            raise ConnectionError("Gateway unreachable")

        expensive_feature("contract.pdf")
        cheap_feature("short note")
        cheap_feature("short note")  # Cache hit!

        failed_caught = False
        try:
            failing_feature()
        except RetryExhaustedError:
            failed_caught = True
        assert failed_caught

        report = tally.get_report()

        # 1. How much money did I spend this week?
        assert report["spend_summary"]["this_week_usd"] > 0
        assert len(report["weekly_daily_spend"]) == 7

        # 2. Which feature costs the most?
        assert report["overview"]["top_cost_feature"] == "document_analyzer"
        assert report["feature_costs"][0]["feature"] == "document_analyzer"

        # 3. Which calls fail the most?
        assert "unreliable_webhook" in report["overview"]["most_failing_call"]
        assert report["failure_stats"][0]["feature"] == "unreliable_webhook"
        assert report["failure_stats"][0]["failed_calls"] == 1
        assert report["failure_stats"][0]["total_retries"] == 3

        # 4. Which calls are slow?
        assert len(report["latency_stats"]) == 3

        # 5. HTML Dashboard generated & contains all required sections + search
        assert dash_file.exists()
        html_text = dash_file.read_text(encoding="utf-8")
        assert "How much money did I spend this week?" in html_text
        assert "Which feature costs the most?" in html_text
        assert "Which calls fail the most?" in html_text
        assert "Which calls are slow?" in html_text
        assert 'id="searchInput"' in html_text
        assert "document_analyzer" in html_text
        assert "unreliable_webhook" in html_text


class TestCostAndDashboardSuite(unittest.TestCase):
    def test_pricing(self) -> None:
        test_builtin_pricing_includes_openai_anthropic_and_gemini()

    def test_token_extraction(self) -> None:
        test_token_extraction_across_openai_anthropic_and_gemini_shapes()

    def test_dashboard_and_analytics(self) -> None:
        test_cost_analysis_and_html_dashboard_answers_all_questions()
