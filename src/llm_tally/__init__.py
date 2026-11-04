"""
`llm-tally`: Zero-dependency LLM helper for Python.

Features:
1. Self-evicting local in-memory cache (`cached_llm_call`, `LLMTally.cached_call`, `@tally.cache`)
   that stores `(input + settings) -> output`, vanishes after `ttl` seconds on fetch,
   and kills itself after `max_requests` fetches.
2. Automatic retry on HTTP 500/5xx, 429 rate limits, or network issues (`LLMTally.retry_call`,
   `@tally.track`) with default delays of [0.5s, 1.0s, 1.5s] (honoring `Retry-After`), raising
   `RetryExhaustedError` with all attempt errors if all attempts fail.
3. Per-feature & per-config cost analysis across OpenAI, Anthropic, Google Gemini, and
   custom models, persisted to JSONL (and/or custom `on_record` sinks) with an interactive
   static HTML dashboard. Tag calls with `metadata=` or `tally.context(user_id=...)`.
"""

from .cache import CacheConfig, CacheEntry, InMemoryCache, build_cache_key, cached_llm_call
from .core import AILogHelper, LLMTally, get_default_tally
from .dashboard import render_dashboard_html, write_dashboard_file
from .pricing import DEFAULT_PRICING, ModelPricing, PricingRegistry
from .retry import (
    DEFAULT_RETRY_DELAYS,
    AttemptFailure,
    HTTPStatus5xxError,
    RetryConfig,
    RetryExhaustedError,
    execute_with_retry_async,
    execute_with_retry_sync,
    extract_retry_after,
    is_retryable_exception,
)
from .tracker import (
    CallRecord,
    JSONLLogger,
    TokenUsage,
    analyze_logs,
    extract_token_usage,
    filter_records_since,
)

__version__ = "0.2.0"

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
    "extract_retry_after",
    "DEFAULT_PRICING",
    "ModelPricing",
    "PricingRegistry",
    "TokenUsage",
    "CallRecord",
    "JSONLLogger",
    "extract_token_usage",
    "analyze_logs",
    "filter_records_since",
    "render_dashboard_html",
    "write_dashboard_file",
]
