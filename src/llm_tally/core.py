"""
Core `LLMTally` (alias `AILogHelper`) Manager Class.

Unifies:
- Feature 1: Self-evicting local in-memory cache (saves input + settings -> output,
  vanishes after `ttl` lifetime on fetch, kills itself after `max_requests` fetches,
  and bounds memory via `max_size`).
- Feature 2: Automatic retry on HTTP 500/5xx or network issues (default: 3 retries at
  0.5s, 1.0s, and 1.5s), raising `RetryExhaustedError` with all attempt errors when exhausted.
- Feature 3: Per-feature & per-config cost analysis across OpenAI, Anthropic, Gemini,
  and custom models, persisted to JSONL with an auto-updated interactive HTML dashboard.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

from .cache import InMemoryCache, build_cache_key, is_llm_stream
from .dashboard import render_dashboard_html, write_dashboard_file
from .pricing import ModelPricing, PricingRegistry
from .retry import (
    DEFAULT_RETRY_DELAYS,
    AttemptFailure,
    RetryConfig,
    RetryExhaustedError,
    execute_with_retry_async,
    execute_with_retry_sync,
)
from .tracker import (
    CallRecord,
    JSONLLogger,
    TokenUsage,
    analyze_logs,
    extract_token_usage,
    format_preview,
)

F = TypeVar("F", bound=Callable[..., Any])


def _extract_chunk_text(chunk: Any) -> str:
    """Extract text content from common streaming chunk formats for fallback token estimation."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="ignore")
    if isinstance(chunk, dict):
        if isinstance(chunk.get("content"), str):
            return chunk["content"]
        if isinstance(chunk.get("text"), str):
            return chunk["text"]
        delta_d = chunk.get("delta")
        if isinstance(delta_d, dict):
            if isinstance(delta_d.get("content"), str):
                return delta_d["content"]
            if isinstance(delta_d.get("text"), str):
                return delta_d["text"]
        return ""
    # OpenAI ChatCompletionChunk: chunk.choices[0].delta.content
    choices = getattr(chunk, "choices", None)
    if isinstance(choices, (list, tuple)) and choices:
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None)
        if isinstance(content, str):
            return content
    # Anthropic / Gemini chunk.text or chunk.delta.text
    txt = getattr(chunk, "text", None)
    if isinstance(txt, str):
        return txt
    delta = getattr(chunk, "delta", None)
    delta_txt = getattr(delta, "text", None)
    if isinstance(delta_txt, str):
        return delta_txt
    return ""


class _StreamUsageAccumulator:
    """Accumulates token counts and text across streaming events (OpenAI, Anthropic, Gemini)."""

    def __init__(self) -> None:
        self.has_structured_usage: bool = False
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cached_input_tokens: int = 0
        self.cache_creation_input_tokens: int = 0
        self.detected_model: Optional[str] = None
        self.input_excludes_cached: Optional[bool] = None
        self.collected_text: List[str] = []

    def observe_chunk(self, chunk: Any) -> None:
        u = extract_token_usage(chunk, args=(), kwargs={})
        if u.detected_model and not self.detected_model:
            self.detected_model = u.detected_model

        if not u.estimated and (
            u.input_tokens > 0
            or u.output_tokens > 0
            or u.cached_input_tokens > 0
            or u.cache_creation_input_tokens > 0
        ):
            self.has_structured_usage = True
            if u.input_tokens > 0:
                self.input_tokens = u.input_tokens
            if u.output_tokens > 0:
                self.output_tokens = u.output_tokens
            if u.cached_input_tokens > 0:
                self.cached_input_tokens = u.cached_input_tokens
            if u.cache_creation_input_tokens > 0:
                self.cache_creation_input_tokens = u.cache_creation_input_tokens
            if u.input_excludes_cached is not None:
                self.input_excludes_cached = u.input_excludes_cached

        piece = _extract_chunk_text(chunk)
        if piece:
            self.collected_text.append(piece)

    def build_final_usage(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[TokenUsage, str]:
        joined_text = "".join(self.collected_text)
        if self.has_structured_usage:
            return (
                TokenUsage(
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                    cached_input_tokens=self.cached_input_tokens,
                    cache_creation_input_tokens=self.cache_creation_input_tokens,
                    estimated=False,
                    detected_model=self.detected_model,
                    input_excludes_cached=self.input_excludes_cached,
                ),
                joined_text,
            )
        fallback = extract_token_usage(joined_text, args=args, kwargs=kwargs)
        if self.detected_model and not fallback.detected_model:
            fallback.detected_model = self.detected_model
        return fallback, joined_text


class _SyncStreamProxy:
    """
    Transparent proxy around a synchronous generator or SDK stream object.
    Preserves `.close()`, `with ... as stream:`, `.response`, and custom SDK attributes,
    while updating the pre-logged `CallRecord` on completion or mid-stream failure.
    """

    def __init__(
        self,
        stream: Any,
        tally: "LLMTally",
        record_id: str,
        timestamp: str,
        start_perf: float,
        feature: str,
        func_short: str,
        chosen_model: str,
        model: Optional[str],
        cfg_dict: Dict[str, Any],
        failed_attempts: List[AttemptFailure],
        explicit_input_tokens: Optional[int],
        explicit_output_tokens: Optional[int],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        self._stream = stream
        self._iterator: Optional[Any] = None
        self._tally = tally
        self._record_id = record_id
        self._timestamp = timestamp
        self._start_perf = start_perf
        self._feature = feature
        self._func_short = func_short
        self._chosen_model = chosen_model
        self._model = model
        self._cfg_dict = cfg_dict
        self._failed_attempts = failed_attempts
        self._explicit_input_tokens = explicit_input_tokens
        self._explicit_output_tokens = explicit_output_tokens
        self._args = args
        self._kwargs = kwargs
        self._acc = _StreamUsageAccumulator()
        self._finalized = False

    def __iter__(self) -> "_SyncStreamProxy":
        if self._iterator is None:
            self._iterator = iter(self._stream)
        return self

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self._stream)
        try:
            chunk = next(self._iterator)
            self._acc.observe_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize(status="success", error_message=None)
            raise
        except BaseException as exc:
            self._finalize(status="failed", error_message=f"{type(exc).__name__}: {exc}")
            raise

    def close(self) -> Any:
        try:
            close_fn = getattr(self._stream, "close", None)
            if callable(close_fn):
                return close_fn()
        finally:
            self._finalize(status="success", error_message=None)

    def __enter__(self) -> "_SyncStreamProxy":
        enter_fn = getattr(self._stream, "__enter__", None)
        if callable(enter_fn):
            enter_fn()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if exc_val is not None:
            self._finalize(status="failed", error_message=f"{type(exc_val).__name__}: {exc_val}")
        else:
            self._finalize(status="success", error_message=None)
        exit_fn = getattr(self._stream, "__exit__", None)
        if callable(exit_fn):
            return exit_fn(exc_type, exc_val, exc_tb)
        close_fn = getattr(self._stream, "close", None)
        if callable(close_fn):
            close_fn()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _finalize(self, status: str, error_message: Optional[str]) -> None:
        if self._finalized:
            return
        self._finalized = True
        latency_ms = (time.perf_counter() - self._start_perf) * 1000.0
        usage, joined_text = self._acc.build_final_usage(self._args, self._kwargs)
        in_tok = (
            self._explicit_input_tokens
            if self._explicit_input_tokens is not None
            else usage.input_tokens
        )
        out_tok = (
            self._explicit_output_tokens
            if self._explicit_output_tokens is not None
            else usage.output_tokens
        )
        final_model = str(
            self._model
            or self._cfg_dict.get("model")
            or usage.detected_model
            or self._chosen_model
        )
        cost_usd = self._tally._safe_calculate_cost(
            model=final_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=usage.cached_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            input_excludes_cached=usage.input_excludes_cached,
        )
        self._tally._update_recorded_call(
            record_id=self._record_id,
            timestamp=self._timestamp,
            feature=self._feature,
            function_name=self._func_short,
            model=final_model,
            config=self._cfg_dict,
            status=status,
            cache_hit=False,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=usage.cached_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cost_usd=cost_usd,
            retries=len(self._failed_attempts),
            error_message=error_message,
            attempt_errors=[a.to_dict() for a in self._failed_attempts],
            args=self._args,
            kwargs=self._kwargs,
            response=joined_text or "<stream>",
        )


class _AsyncStreamProxy:
    """
    Transparent proxy around an async generator or async SDK stream object.
    Preserves `.aclose()`, `.close()`, `async with ... as stream:`, `.response`, and custom attributes.
    """

    def __init__(
        self,
        stream: Any,
        tally: "LLMTally",
        record_id: str,
        timestamp: str,
        start_perf: float,
        feature: str,
        func_short: str,
        chosen_model: str,
        model: Optional[str],
        cfg_dict: Dict[str, Any],
        failed_attempts: List[AttemptFailure],
        explicit_input_tokens: Optional[int],
        explicit_output_tokens: Optional[int],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        self._stream = stream
        self._aiterator: Optional[Any] = None
        self._tally = tally
        self._record_id = record_id
        self._timestamp = timestamp
        self._start_perf = start_perf
        self._feature = feature
        self._func_short = func_short
        self._chosen_model = chosen_model
        self._model = model
        self._cfg_dict = cfg_dict
        self._failed_attempts = failed_attempts
        self._explicit_input_tokens = explicit_input_tokens
        self._explicit_output_tokens = explicit_output_tokens
        self._args = args
        self._kwargs = kwargs
        self._acc = _StreamUsageAccumulator()
        self._finalized = False

    def __aiter__(self) -> "_AsyncStreamProxy":
        if self._aiterator is None:
            self._aiterator = self._stream.__aiter__()
        return self

    async def __anext__(self) -> Any:
        if self._aiterator is None:
            self._aiterator = self._stream.__aiter__()
        try:
            chunk = await self._aiterator.__anext__()
            self._acc.observe_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize(status="success", error_message=None)
            raise
        except BaseException as exc:
            self._finalize(status="failed", error_message=f"{type(exc).__name__}: {exc}")
            raise

    async def aclose(self) -> Any:
        try:
            aclose_fn = getattr(self._stream, "aclose", None)
            if callable(aclose_fn):
                return await aclose_fn()
            close_fn = getattr(self._stream, "close", None)
            if callable(close_fn):
                res = close_fn()
                if inspect.isawaitable(res):
                    return await res
                return res
        finally:
            self._finalize(status="success", error_message=None)

    def close(self) -> Any:
        try:
            close_fn = getattr(self._stream, "close", None)
            if callable(close_fn):
                return close_fn()
        finally:
            self._finalize(status="success", error_message=None)

    async def __aenter__(self) -> "_AsyncStreamProxy":
        aenter_fn = getattr(self._stream, "__aenter__", None)
        if callable(aenter_fn):
            await aenter_fn()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if exc_val is not None:
            self._finalize(status="failed", error_message=f"{type(exc_val).__name__}: {exc_val}")
        else:
            self._finalize(status="success", error_message=None)
        aexit_fn = getattr(self._stream, "__aexit__", None)
        if callable(aexit_fn):
            return await aexit_fn(exc_type, exc_val, exc_tb)
        await self.aclose()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _finalize(self, status: str, error_message: Optional[str]) -> None:
        if self._finalized:
            return
        self._finalized = True
        latency_ms = (time.perf_counter() - self._start_perf) * 1000.0
        usage, joined_text = self._acc.build_final_usage(self._args, self._kwargs)
        in_tok = (
            self._explicit_input_tokens
            if self._explicit_input_tokens is not None
            else usage.input_tokens
        )
        out_tok = (
            self._explicit_output_tokens
            if self._explicit_output_tokens is not None
            else usage.output_tokens
        )
        final_model = str(
            self._model
            or self._cfg_dict.get("model")
            or usage.detected_model
            or self._chosen_model
        )
        cost_usd = self._tally._safe_calculate_cost(
            model=final_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=usage.cached_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            input_excludes_cached=usage.input_excludes_cached,
        )
        self._tally._update_recorded_call(
            record_id=self._record_id,
            timestamp=self._timestamp,
            feature=self._feature,
            function_name=self._func_short,
            model=final_model,
            config=self._cfg_dict,
            status=status,
            cache_hit=False,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=usage.cached_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cost_usd=cost_usd,
            retries=len(self._failed_attempts),
            error_message=error_message,
            attempt_errors=[a.to_dict() for a in self._failed_attempts],
            args=self._args,
            kwargs=self._kwargs,
            response=joined_text or "<stream>",
        )


class LLMTally:
    """
    Main manager for caching, retrying, and tracking LLM calls & costs.
    """

    def __init__(
        self,
        default_model: str = "gpt-4o-mini",
        log_path: str | Path = ".llm_tally/logs.jsonl",
        dashboard_path: str | Path = ".llm_tally/dashboard.html",
        default_cache_enabled: bool = True,
        default_ttl: Optional[float] = 300.0,
        default_max_requests: Optional[int] = None,
        default_max_cache_size: Optional[int] = 1000,
        default_retry_enabled: bool = True,
        default_retries: int = 3,
        retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
        auto_update_dashboard: bool = True,
        custom_pricing: Optional[Dict[str, ModelPricing]] = None,
        sleep_func: Optional[Callable[[float], Any]] = None,
        async_sleep_func: Optional[Callable[[float], Any]] = None,
    ) -> None:
        self.default_model: str = str(default_model)
        self.log_path: Path = Path(log_path)
        self.dashboard_path: Path = Path(dashboard_path)
        self.default_cache_enabled: bool = bool(default_cache_enabled)
        self.default_ttl: Optional[float] = default_ttl
        self.default_max_requests: Optional[int] = default_max_requests
        self.default_max_cache_size: Optional[int] = default_max_cache_size
        self.default_retry_enabled: bool = bool(default_retry_enabled)
        self.default_retries: int = int(default_retries)
        self.retry_delays: Tuple[float, ...] = tuple(float(d) for d in retry_delays)
        self.auto_update_dashboard: bool = bool(auto_update_dashboard)

        self.cache_store = InMemoryCache(
            default_ttl=self.default_ttl,
            default_max_requests=self.default_max_requests,
            max_size=self.default_max_cache_size,
        )
        self.pricing = PricingRegistry(custom_pricing=custom_pricing)
        self.logger = JSONLLogger(log_path=self.log_path)
        self._sleep_func = sleep_func or time.sleep
        self._async_sleep_func = async_sleep_func or asyncio.sleep

    # -------------------------------------------------------------------------
    # Model Pricing Management
    # -------------------------------------------------------------------------
    def register_model(
        self,
        model: str,
        input_cost_per_1m: float,
        output_cost_per_1m: float,
        provider: str = "custom",
        cached_input_cost_per_1m: Optional[float] = None,
        cache_creation_cost_per_1m: Optional[float] = None,
    ) -> ModelPricing:
        """
        Register or override pricing (in USD per 1M tokens) for any LLM model.
        """
        return self.pricing.register(
            model=model,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
            provider=provider,
            cached_input_cost_per_1m=cached_input_cost_per_1m,
            cache_creation_cost_per_1m=cache_creation_cost_per_1m,
        )

    def _safe_calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        input_excludes_cached: Optional[bool] = None,
    ) -> float:
        """
        Compute cost for a completed LLM call. If warnings are configured as errors
        (`-W error` or `pytest` `filterwarnings = ["error"]`), ensures a warning on an
        unknown model never turns a successful paid LLM call into an exception.
        """
        try:
            return self.pricing.calculate_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                input_excludes_cached=input_excludes_cached,
                warn_on_unknown=True,
            )
        except Warning:
            return self.pricing.calculate_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                input_excludes_cached=input_excludes_cached,
                warn_on_unknown=False,
            )
        except Exception:
            return 0.0

    # -------------------------------------------------------------------------
    # Feature 1: Explicit Self-Evicting In-Memory Cache Function & Decorator
    # -------------------------------------------------------------------------
    def cached_call(
        self,
        func: Callable[..., Any],
        *args: Any,
        ttl: Optional[float] = None,
        max_requests: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        feature: Optional[str] = None,
        track_cost: bool = True,
        retry: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Feature 1 Explicit Function: Execute `func(*args, **kwargs)` using the local
        in-memory cache.
        """
        if not track_cost and not retry:
            effective_ttl = self.default_ttl if ttl is None else ttl
            effective_max_req = self.default_max_requests if max_requests is None else max_requests
            func_name = self._func_identifier(func)
            settings_meta = {"config": config or {}, "model": model or self.default_model}
            key, snapshot = build_cache_key(
                func_name, args, kwargs, settings=settings_meta, func=func
            )

            if inspect.iscoroutinefunction(func):

                async def _async_only_cache() -> Any:
                    hit, cached_val, _ = self.cache_store.fetch(key)
                    if hit:
                        return cached_val
                    res = await func(*args, **kwargs)
                    self.cache_store.store(
                        key=key,
                        input_settings=snapshot,
                        output=res,
                        ttl=effective_ttl,
                        max_requests=effective_max_req,
                        use_defaults=False,
                    )
                    return res

                return _async_only_cache()

            hit, cached_val, _ = self.cache_store.fetch(key)
            if hit:
                return cached_val
            res = func(*args, **kwargs)
            self.cache_store.store(
                key=key,
                input_settings=snapshot,
                output=res,
                ttl=effective_ttl,
                max_requests=effective_max_req,
                use_defaults=False,
            )
            return res

        return self.run(
            func,
            *args,
            feature=feature or getattr(func, "__name__", "cached_call"),
            model=model,
            config=config,
            cache=True,
            ttl=ttl,
            max_requests=max_requests,
            retry=retry,
            **kwargs,
        )

    def cache(
        self,
        ttl: Optional[float] = None,
        max_requests: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        feature: Optional[str] = None,
        model: Optional[str] = None,
        track_cost: bool = True,
        retry: bool = False,
    ) -> Callable[[F], F]:
        """
        Feature 1 Decorator: Wrap a function with the self-evicting local in-memory cache.
        """

        def _decorator(func: F) -> F:
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return await self.cached_call(
                        func,
                        *args,
                        ttl=ttl,
                        max_requests=max_requests,
                        config=config,
                        feature=feature or getattr(func, "__name__", "cached_call"),
                        model=model,
                        track_cost=track_cost,
                        retry=retry,
                        **kwargs,
                    )

                return _async_wrapper  # type: ignore[return-value]

            @functools.wraps(func)
            def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.cached_call(
                    func,
                    *args,
                    ttl=ttl,
                    max_requests=max_requests,
                    config=config,
                    feature=feature or getattr(func, "__name__", "cached_call"),
                    model=model,
                    track_cost=track_cost,
                    retry=retry,
                    **kwargs,
                )

            return _sync_wrapper  # type: ignore[return-value]

        return _decorator

    # -------------------------------------------------------------------------
    # Feature 2: Explicit Retry Function
    # -------------------------------------------------------------------------
    def retry_call(
        self,
        func: Callable[..., Any],
        *args: Any,
        retries: Optional[int] = None,
        retry_delays: Optional[Sequence[float]] = None,
        extra_exceptions: Tuple[Type[BaseException], ...] = (),
        **kwargs: Any,
    ) -> Any:
        """
        Feature 2 Explicit Function: Execute `func(*args, **kwargs)` with automatic
        retries on HTTP 500/5xx or network issues.
        """
        cfg = RetryConfig(
            enabled=True,
            retries=self.default_retries if retries is None else retries,
            delays=self.retry_delays if retry_delays is None else tuple(retry_delays),
            extra_exceptions=extra_exceptions,
        )
        if inspect.iscoroutinefunction(func):

            async def _run_async() -> Any:
                result, _ = await execute_with_retry_async(
                    func,
                    *args,
                    config=cfg,
                    sleep_func=self._async_sleep_func,
                    **kwargs,
                )
                return result

            return _run_async()

        result, _ = execute_with_retry_sync(
            func,
            *args,
            config=cfg,
            sleep_func=self._sleep_func,
            **kwargs,
        )
        return result

    # -------------------------------------------------------------------------
    # Unified Runner & Decorator (Features 1 + 2 + 3)
    # -------------------------------------------------------------------------
    def track(
        self,
        feature: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        cache: Optional[bool] = None,
        ttl: Optional[float] = None,
        max_requests: Optional[int] = None,
        retry: Optional[bool] = None,
        retries: Optional[int] = None,
        retry_delays: Optional[Sequence[float]] = None,
        extra_exceptions: Tuple[Type[BaseException], ...] = (),
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> Callable[[F], F]:
        """
        Unified decorator combining:
        - Feature 1: Self-evicting in-memory cache (`cache`, `ttl`, `max_requests`)
        - Feature 2: 500/network retry (`retry`, `retries`, `retry_delays`)
        - Feature 3: Per-feature & per-config cost analytics + HTML dashboard updates
        """

        def _decorator(func: F) -> F:
            feat_name = feature or getattr(func, "__name__", "default")

            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def _async_wrapped(*args: Any, **kwargs: Any) -> Any:
                    return await self._execute_async(
                        func=func,
                        args=args,
                        kwargs=kwargs,
                        feature=feat_name,
                        model=model,
                        config=config,
                        cache=cache,
                        ttl=ttl,
                        max_requests=max_requests,
                        retry=retry,
                        retries=retries,
                        retry_delays=retry_delays,
                        extra_exceptions=extra_exceptions,
                        explicit_input_tokens=input_tokens,
                        explicit_output_tokens=output_tokens,
                    )

                return _async_wrapped  # type: ignore[return-value]

            @functools.wraps(func)
            def _sync_wrapped(*args: Any, **kwargs: Any) -> Any:
                return self._execute_sync(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    feature=feat_name,
                    model=model,
                    config=config,
                    cache=cache,
                    ttl=ttl,
                    max_requests=max_requests,
                    retry=retry,
                    retries=retries,
                    retry_delays=retry_delays,
                    extra_exceptions=extra_exceptions,
                    explicit_input_tokens=input_tokens,
                    explicit_output_tokens=output_tokens,
                )

            return _sync_wrapped  # type: ignore[return-value]

        return _decorator

    def run(
        self,
        func: Callable[..., Any],
        *args: Any,
        feature: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        cache: Optional[bool] = None,
        ttl: Optional[float] = None,
        max_requests: Optional[int] = None,
        retry: Optional[bool] = None,
        retries: Optional[int] = None,
        retry_delays: Optional[Sequence[float]] = None,
        extra_exceptions: Tuple[Type[BaseException], ...] = (),
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute `func(*args, **kwargs)` directly with caching, retries, and cost tracking."""
        feat_name = feature or getattr(func, "__name__", "default")
        if inspect.iscoroutinefunction(func):
            return self._execute_async(
                func=func,
                args=args,
                kwargs=kwargs,
                feature=feat_name,
                model=model,
                config=config,
                cache=cache,
                ttl=ttl,
                max_requests=max_requests,
                retry=retry,
                retries=retries,
                retry_delays=retry_delays,
                extra_exceptions=extra_exceptions,
                explicit_input_tokens=input_tokens,
                explicit_output_tokens=output_tokens,
            )
        return self._execute_sync(
            func=func,
            args=args,
            kwargs=kwargs,
            feature=feat_name,
            model=model,
            config=config,
            cache=cache,
            ttl=ttl,
            max_requests=max_requests,
            retry=retry,
            retries=retries,
            retry_delays=retry_delays,
            extra_exceptions=extra_exceptions,
            explicit_input_tokens=input_tokens,
            explicit_output_tokens=output_tokens,
        )

    # -------------------------------------------------------------------------
    # Internal Sync & Async Execution Pipelines
    # -------------------------------------------------------------------------
    @staticmethod
    def _func_identifier(func: Callable[..., Any]) -> str:
        mod = getattr(func, "__module__", "__main__")
        qual = getattr(func, "__qualname__", getattr(func, "__name__", "callable"))
        return f"{mod}.{qual}"

    def _create_stream_proxy(
        self,
        stream: Any,
        start_perf: float,
        feature: str,
        func_short: str,
        chosen_model: str,
        model: Optional[str],
        cfg_dict: Dict[str, Any],
        failed_attempts: List[AttemptFailure],
        explicit_input_tokens: Optional[int],
        explicit_output_tokens: Optional[int],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        Log an initial `CallRecord` immediately (so un-iterated streams are never missed)
        and return a transparent `_SyncStreamProxy` or `_AsyncStreamProxy` that updates
        the record in-place as the stream is consumed or closed.
        """
        initial_usage = extract_token_usage("", args=args, kwargs=kwargs)
        in_tok = (
            explicit_input_tokens
            if explicit_input_tokens is not None
            else initial_usage.input_tokens
        )
        out_tok = explicit_output_tokens if explicit_output_tokens is not None else 0
        final_model = str(model or cfg_dict.get("model") or chosen_model)
        initial_cost = self._safe_calculate_cost(
            model=final_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
        latency_ms = (time.perf_counter() - start_perf) * 1000.0
        rec = self._record_call(
            feature=feature,
            function_name=func_short,
            model=final_model,
            config=cfg_dict,
            status="success",
            cache_hit=False,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=0,
            cache_creation_input_tokens=0,
            cost_usd=initial_cost,
            retries=len(failed_attempts),
            error_message=None,
            attempt_errors=[a.to_dict() for a in failed_attempts],
            args=args,
            kwargs=kwargs,
            response="<stream>",
        )
        rec_id = rec.id if rec is not None else uuid.uuid4().hex[:12]
        rec_ts = rec.timestamp if rec is not None else datetime.now(timezone.utc).isoformat()

        if inspect.isasyncgen(stream) or hasattr(stream, "__anext__") or hasattr(stream, "__aiter__"):
            return _AsyncStreamProxy(
                stream=stream,
                tally=self,
                record_id=rec_id,
                timestamp=rec_ts,
                start_perf=start_perf,
                feature=feature,
                func_short=func_short,
                chosen_model=str(chosen_model),
                model=model,
                cfg_dict=cfg_dict,
                failed_attempts=failed_attempts,
                explicit_input_tokens=explicit_input_tokens,
                explicit_output_tokens=explicit_output_tokens,
                args=args,
                kwargs=kwargs,
            )
        return _SyncStreamProxy(
            stream=stream,
            tally=self,
            record_id=rec_id,
            timestamp=rec_ts,
            start_perf=start_perf,
            feature=feature,
            func_short=func_short,
            chosen_model=str(chosen_model),
            model=model,
            cfg_dict=cfg_dict,
            failed_attempts=failed_attempts,
            explicit_input_tokens=explicit_input_tokens,
            explicit_output_tokens=explicit_output_tokens,
            args=args,
            kwargs=kwargs,
        )

    def _execute_sync(
        self,
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        feature: str,
        model: Optional[str],
        config: Optional[Dict[str, Any]],
        cache: Optional[bool],
        ttl: Optional[float],
        max_requests: Optional[int],
        retry: Optional[bool],
        retries: Optional[int],
        retry_delays: Optional[Sequence[float]],
        extra_exceptions: Tuple[Type[BaseException], ...],
        explicit_input_tokens: Optional[int],
        explicit_output_tokens: Optional[int],
    ) -> Any:
        use_cache = self.default_cache_enabled if cache is None else cache
        effective_ttl = self.default_ttl if ttl is None else ttl
        effective_max_req = self.default_max_requests if max_requests is None else max_requests
        use_retry = self.default_retry_enabled if retry is None else retry
        effective_retries = self.default_retries if retries is None else retries
        effective_delays = self.retry_delays if retry_delays is None else tuple(retry_delays)
        chosen_model = model or (config or {}).get("model") or self.default_model
        cfg_dict = dict(config or {})

        func_id = self._func_identifier(func)
        func_short = getattr(func, "__qualname__", getattr(func, "__name__", "callable"))
        cache_settings = {"feature": feature, "model": chosen_model, "config": cfg_dict}
        cache_key, snapshot = build_cache_key(
            func_id, args, kwargs, settings=cache_settings, func=func
        )

        start_perf = time.perf_counter()

        # 1. Check In-Memory Cache
        if use_cache:
            hit, cached_output, _ = self.cache_store.fetch(cache_key)
            if hit:
                latency_ms = (time.perf_counter() - start_perf) * 1000.0
                self._record_call(
                    feature=feature,
                    function_name=func_short,
                    model=str(chosen_model),
                    config=cfg_dict,
                    status="cached",
                    cache_hit=True,
                    latency_ms=latency_ms,
                    input_tokens=0,
                    output_tokens=0,
                    cached_input_tokens=0,
                    cache_creation_input_tokens=0,
                    cost_usd=0.0,
                    retries=0,
                    error_message=None,
                    attempt_errors=[],
                    args=args,
                    kwargs=kwargs,
                    response=cached_output,
                )
                return cached_output

        # 2. Execute with Retry Engine
        retry_cfg = RetryConfig(
            enabled=use_retry,
            retries=effective_retries,
            delays=effective_delays,
            extra_exceptions=extra_exceptions,
        )
        try:
            result, failed_attempts = execute_with_retry_sync(
                func,
                *args,
                config=retry_cfg,
                sleep_func=self._sleep_func,
                **kwargs,
            )
        except RetryExhaustedError as exc:
            latency_ms = (time.perf_counter() - start_perf) * 1000.0
            self._record_call(
                feature=feature,
                function_name=func_short,
                model=str(chosen_model),
                config=cfg_dict,
                status="failed",
                cache_hit=False,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                cached_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=0.0,
                retries=max(0, len(exc.attempts) - 1),
                error_message=str(exc),
                attempt_errors=[a.to_dict() for a in exc.attempts],
                args=args,
                kwargs=kwargs,
                response=None,
            )
            raise
        except BaseException as exc:
            latency_ms = (time.perf_counter() - start_perf) * 1000.0
            prior_attempts: List[AttemptFailure] = getattr(exc, "_llm_tally_attempts", [])
            self._record_call(
                feature=feature,
                function_name=func_short,
                model=str(chosen_model),
                config=cfg_dict,
                status="failed",
                cache_hit=False,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                cached_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=0.0,
                retries=max(0, len(prior_attempts) - 1),
                error_message=f"{type(exc).__name__}: {exc}",
                attempt_errors=[a.to_dict() for a in prior_attempts],
                args=args,
                kwargs=kwargs,
                response=None,
            )
            raise

        # 3. Handle Generators / SDK Streams without caching exhausted iterators
        if is_llm_stream(result):
            return self._create_stream_proxy(
                stream=result,
                start_perf=start_perf,
                feature=feature,
                func_short=func_short,
                chosen_model=str(chosen_model),
                model=model,
                cfg_dict=cfg_dict,
                failed_attempts=failed_attempts,
                explicit_input_tokens=explicit_input_tokens,
                explicit_output_tokens=explicit_output_tokens,
                args=args,
                kwargs=kwargs,
            )

        latency_ms = (time.perf_counter() - start_perf) * 1000.0

        # 4. Store in Cache if enabled (InMemoryCache.store skips any single-use iterator)
        if use_cache:
            self.cache_store.store(
                key=cache_key,
                input_settings=snapshot,
                output=result,
                ttl=effective_ttl,
                max_requests=effective_max_req,
                use_defaults=False,
            )

        # 5. Extract Token Usage & Compute Cost Safely
        usage = extract_token_usage(result, args=args, kwargs=kwargs)
        in_tok = explicit_input_tokens if explicit_input_tokens is not None else usage.input_tokens
        out_tok = (
            explicit_output_tokens if explicit_output_tokens is not None else usage.output_tokens
        )
        final_model = str(
            model or cfg_dict.get("model") or usage.detected_model or self.default_model
        )
        cost_usd = self._safe_calculate_cost(
            model=final_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=usage.cached_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            input_excludes_cached=usage.input_excludes_cached,
        )

        self._record_call(
            feature=feature,
            function_name=func_short,
            model=final_model,
            config=cfg_dict,
            status="success",
            cache_hit=False,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=usage.cached_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cost_usd=cost_usd,
            retries=len(failed_attempts),
            error_message=None,
            attempt_errors=[a.to_dict() for a in failed_attempts],
            args=args,
            kwargs=kwargs,
            response=result,
        )
        return result

    async def _execute_async(
        self,
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        feature: str,
        model: Optional[str],
        config: Optional[Dict[str, Any]],
        cache: Optional[bool],
        ttl: Optional[float],
        max_requests: Optional[int],
        retry: Optional[bool],
        retries: Optional[int],
        retry_delays: Optional[Sequence[float]],
        extra_exceptions: Tuple[Type[BaseException], ...],
        explicit_input_tokens: Optional[int],
        explicit_output_tokens: Optional[int],
    ) -> Any:
        use_cache = self.default_cache_enabled if cache is None else cache
        effective_ttl = self.default_ttl if ttl is None else ttl
        effective_max_req = self.default_max_requests if max_requests is None else max_requests
        use_retry = self.default_retry_enabled if retry is None else retry
        effective_retries = self.default_retries if retries is None else retries
        effective_delays = self.retry_delays if retry_delays is None else tuple(retry_delays)
        chosen_model = model or (config or {}).get("model") or self.default_model
        cfg_dict = dict(config or {})

        func_id = self._func_identifier(func)
        func_short = getattr(func, "__qualname__", getattr(func, "__name__", "callable"))
        cache_settings = {"feature": feature, "model": chosen_model, "config": cfg_dict}
        cache_key, snapshot = build_cache_key(
            func_id, args, kwargs, settings=cache_settings, func=func
        )

        start_perf = time.perf_counter()

        # 1. Check In-Memory Cache
        if use_cache:
            hit, cached_output, _ = self.cache_store.fetch(cache_key)
            if hit:
                latency_ms = (time.perf_counter() - start_perf) * 1000.0
                self._record_call(
                    feature=feature,
                    function_name=func_short,
                    model=str(chosen_model),
                    config=cfg_dict,
                    status="cached",
                    cache_hit=True,
                    latency_ms=latency_ms,
                    input_tokens=0,
                    output_tokens=0,
                    cached_input_tokens=0,
                    cache_creation_input_tokens=0,
                    cost_usd=0.0,
                    retries=0,
                    error_message=None,
                    attempt_errors=[],
                    args=args,
                    kwargs=kwargs,
                    response=cached_output,
                )
                return cached_output

        # 2. Execute with Async Retry Engine
        retry_cfg = RetryConfig(
            enabled=use_retry,
            retries=effective_retries,
            delays=effective_delays,
            extra_exceptions=extra_exceptions,
        )
        try:
            result, failed_attempts = await execute_with_retry_async(
                func,
                *args,
                config=retry_cfg,
                sleep_func=self._async_sleep_func,
                **kwargs,
            )
        except RetryExhaustedError as exc:
            latency_ms = (time.perf_counter() - start_perf) * 1000.0
            self._record_call(
                feature=feature,
                function_name=func_short,
                model=str(chosen_model),
                config=cfg_dict,
                status="failed",
                cache_hit=False,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                cached_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=0.0,
                retries=max(0, len(exc.attempts) - 1),
                error_message=str(exc),
                attempt_errors=[a.to_dict() for a in exc.attempts],
                args=args,
                kwargs=kwargs,
                response=None,
            )
            raise
        except BaseException as exc:
            latency_ms = (time.perf_counter() - start_perf) * 1000.0
            prior_attempts: List[AttemptFailure] = getattr(exc, "_llm_tally_attempts", [])
            self._record_call(
                feature=feature,
                function_name=func_short,
                model=str(chosen_model),
                config=cfg_dict,
                status="failed",
                cache_hit=False,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                cached_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=0.0,
                retries=max(0, len(prior_attempts) - 1),
                error_message=f"{type(exc).__name__}: {exc}",
                attempt_errors=[a.to_dict() for a in prior_attempts],
                args=args,
                kwargs=kwargs,
                response=None,
            )
            raise

        # 3. Handle Generators / SDK Streams without caching exhausted iterators
        if is_llm_stream(result):
            return self._create_stream_proxy(
                stream=result,
                start_perf=start_perf,
                feature=feature,
                func_short=func_short,
                chosen_model=str(chosen_model),
                model=model,
                cfg_dict=cfg_dict,
                failed_attempts=failed_attempts,
                explicit_input_tokens=explicit_input_tokens,
                explicit_output_tokens=explicit_output_tokens,
                args=args,
                kwargs=kwargs,
            )

        latency_ms = (time.perf_counter() - start_perf) * 1000.0

        # 4. Store in Cache if enabled
        if use_cache:
            self.cache_store.store(
                key=cache_key,
                input_settings=snapshot,
                output=result,
                ttl=effective_ttl,
                max_requests=effective_max_req,
                use_defaults=False,
            )

        # 5. Extract Token Usage & Compute Cost Safely
        usage = extract_token_usage(result, args=args, kwargs=kwargs)
        in_tok = explicit_input_tokens if explicit_input_tokens is not None else usage.input_tokens
        out_tok = (
            explicit_output_tokens if explicit_output_tokens is not None else usage.output_tokens
        )
        final_model = str(
            model or cfg_dict.get("model") or usage.detected_model or self.default_model
        )
        cost_usd = self._safe_calculate_cost(
            model=final_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=usage.cached_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            input_excludes_cached=usage.input_excludes_cached,
        )

        self._record_call(
            feature=feature,
            function_name=func_short,
            model=final_model,
            config=cfg_dict,
            status="success",
            cache_hit=False,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=usage.cached_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cost_usd=cost_usd,
            retries=len(failed_attempts),
            error_message=None,
            attempt_errors=[a.to_dict() for a in failed_attempts],
            args=args,
            kwargs=kwargs,
            response=result,
        )
        return result

    def _record_call(
        self,
        feature: str,
        function_name: str,
        model: str,
        config: Dict[str, Any],
        status: str,
        cache_hit: bool,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        cost_usd: float,
        retries: int,
        error_message: Optional[str],
        attempt_errors: List[Dict[str, Any]],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        response: Any,
        cache_creation_input_tokens: int = 0,
    ) -> Optional[CallRecord]:
        try:
            record = CallRecord(
                id=uuid.uuid4().hex[:12],
                timestamp=datetime.now(timezone.utc).isoformat(),
                feature=feature,
                function_name=function_name,
                model=model,
                config=config,
                status=status,
                cache_hit=cache_hit,
                latency_ms=round(latency_ms, 2),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                cost_usd=cost_usd,
                retries=retries,
                error_message=error_message,
                attempt_errors=attempt_errors,
                input_preview=format_preview({"args": args, "kwargs": kwargs}),
                output_preview=format_preview(response),
                cache_creation_input_tokens=cache_creation_input_tokens,
            )
            self.logger.append(record)
            if self.auto_update_dashboard:
                self.generate_dashboard()
            return record
        except Exception as log_exc:
            try:
                warnings.warn(
                    f"llm-tally failed to write telemetry log or dashboard: {log_exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Warning:
                pass
            return None

    def _update_recorded_call(
        self,
        record_id: str,
        timestamp: str,
        feature: str,
        function_name: str,
        model: str,
        config: Dict[str, Any],
        status: str,
        cache_hit: bool,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        cost_usd: float,
        retries: int,
        error_message: Optional[str],
        attempt_errors: List[Dict[str, Any]],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        response: Any,
        cache_creation_input_tokens: int = 0,
    ) -> Optional[CallRecord]:
        try:
            record = CallRecord(
                id=record_id,
                timestamp=timestamp,
                feature=feature,
                function_name=function_name,
                model=model,
                config=config,
                status=status,
                cache_hit=cache_hit,
                latency_ms=round(latency_ms, 2),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                cost_usd=cost_usd,
                retries=retries,
                error_message=error_message,
                attempt_errors=attempt_errors,
                input_preview=format_preview({"args": args, "kwargs": kwargs}),
                output_preview=format_preview(response),
                cache_creation_input_tokens=cache_creation_input_tokens,
            )
            self.logger.update(record)
            if self.auto_update_dashboard:
                self.generate_dashboard()
            return record
        except Exception as log_exc:
            try:
                warnings.warn(
                    f"llm-tally failed to update stream telemetry log: {log_exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Warning:
                pass
            return None

    # -------------------------------------------------------------------------
    # Feature 3: Analytics & Dashboard Generation
    # -------------------------------------------------------------------------
    def get_records(self) -> List[CallRecord]:
        """Read all logged call records from the JSONL file."""
        return self.logger.read_all()

    def get_report(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Return the full analytics dictionary answering:
        - `spend_summary["this_week_usd"]`: How much money did I spend this week?
        - `feature_costs`: Which feature (per config) costs the most?
        - `failure_stats`: Which calls fail the most?
        - `latency_stats` / `slowest_calls`: Which calls are slow?
        """
        records = self.logger.read_all()
        return analyze_logs(records, now=now)

    def render_dashboard(
        self,
        title: str = "llm-tally — Cost & Reliability Dashboard",
    ) -> str:
        """Render the self-contained HTML dashboard as a string."""
        records = self.logger.read_all()
        return render_dashboard_html(records, title=title)

    def generate_dashboard(
        self,
        dashboard_path: Optional[str | Path] = None,
        title: str = "llm-tally — Cost & Reliability Dashboard",
    ) -> Path:
        """Write/update the static HTML dashboard file on disk and return its path."""
        target = Path(dashboard_path) if dashboard_path is not None else self.dashboard_path
        records = self.logger.read_all()
        return write_dashboard_file(records, dashboard_path=target, title=title)

    def clear_cache(self) -> None:
        """Clear all entries from the in-memory cache."""
        self.cache_store.clear()

    def clear_logs(self) -> None:
        """Delete the JSONL call log file and regenerate an empty dashboard."""
        self.logger.clear()
        if self.auto_update_dashboard and self.dashboard_path.exists():
            self.generate_dashboard()


# Alias for convenience
AILogHelper = LLMTally

_DEFAULT_INSTANCE: Optional[LLMTally] = None


def get_default_tally() -> LLMTally:
    """Return or initialize a default singleton `LLMTally` instance."""
    global _DEFAULT_INSTANCE
    if _DEFAULT_INSTANCE is None:
        _DEFAULT_INSTANCE = LLMTally()
    return _DEFAULT_INSTANCE
