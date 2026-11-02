"""
`django-llm-tally`: Zero-dependency LLM helper for Python & Django.

Features:
1. Self-evicting local in-memory cache (`cached_llm_call`, `LLMTally.cached_call`, `@tally.cache`)
   that stores `(input + settings) -> output`, vanishes after `ttl` seconds on fetch,
   and kills itself after `max_requests` fetches.
2. Automatic retry on HTTP 500/5xx or network issues (`LLMTally.retry_call`, `@tally.track`)
   with default delays of [0.5s, 1.0s, 1.5s], raising `RetryExhaustedError` with all
   attempt errors if all attempts fail.
3. Per-feature & per-config cost analysis across OpenAI, Anthropic, Google Gemini, and
   custom models, persisted to JSONL with an interactive static HTML dashboard.
"""

from .cache import CacheConfig, CacheEntry, InMemoryCache, build_cache_key, cached_llm_call
from .core import AILogHelper, LLMTally, get_default_tally
from .dashboard import render_dashboard_html, write_dashboard_file
from .django_integration import get_django_llm_tally_settings, make_django_dashboard_view
from .pricing import DEFAULT_PRICING, ModelPricing, PricingRegistry
from .retry import (
    DEFAULT_RETRY_DELAYS,
    AttemptFailure,
    HTTPStatus5xxError,
    RetryConfig,
    RetryExhaustedError,
    execute_with_retry_async,
    execute_with_retry_sync,
    is_retryable_exception,
)
from .tracker import CallRecord, JSONLLogger, TokenUsage, analyze_logs, extract_token_usage

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LLMTally",
    "AILogHelper",
    "get_default_tally",
    "CacheConfig",
    "CacheEntry",
    "InMemoryCache",
    "build_cache_key",
    "cached_llm_call",
    "DEFAULT_RETRY_DELAYS",
    "RetryConfig",
    "AttemptFailure",
    "HTTPStatus5xxError",
    "RetryExhaustedError",
    "execute_with_retry_sync",
    "execute_with_retry_async",
    "is_retryable_exception",
    "DEFAULT_PRICING",
    "ModelPricing",
    "PricingRegistry",
    "TokenUsage",
    "CallRecord",
    "JSONLLogger",
    "extract_token_usage",
    "analyze_logs",
    "render_dashboard_html",
    "write_dashboard_file",
    "get_django_llm_tally_settings",
    "make_django_dashboard_view",
]

