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
import atexit
import contextlib
import contextvars
import functools
import inspect
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
import warnings
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .cache import InMemoryCache, build_cache_key, is_llm_stream
from .dashboard import render_dashboard_html, write_dashboard_file
from .pricing import UNKNOWN_MODEL, ModelPricing, PricingRegistry
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
    filter_records_since,
    format_preview,
    reprice_records,
)

F = TypeVar("F", bound=Callable[..., Any])
RecordSink = Callable[[CallRecord], Any]

# No model is assumed by default: a call whose model can't be determined is logged as
# "unknown" and flagged `unpriced` instead of being priced as some arbitrary model.
DEFAULT_MODEL: Optional[str] = None
# `LLM_TALLY_DIR` moves the default log and dashboard location (e.g. to an absolute path, so
# every process writes to the same place no matter which directory it was started from).
DEFAULT_DIR = os.environ.get("LLM_TALLY_DIR") or ".llm_tally"
DEFAULT_LOG_PATH = os.path.join(DEFAULT_DIR, "logs.jsonl")
DEFAULT_DASHBOARD_PATH = os.path.join(DEFAULT_DIR, "dashboard.html")
DEFAULT_DASHBOARD_TITLE = "llm-tally — Cost & Reliability Dashboard"

logger = logging.getLogger("llm_tally")

# Inline dashboard rebuilds above this log size (or slower than this) move to a background thread
_INLINE_DASHBOARD_MAX_LOG_BYTES = 2 * 1024 * 1024
_INLINE_DASHBOARD_MAX_SECONDS = 0.05

# Metadata set via `tally.context(...)`; contextvars keep it per-thread and per-asyncio-task
_CONTEXT_METADATA: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "llm_tally_context_metadata", default={}
)

# Instances with a throttled, not-yet-written dashboard are flushed at interpreter exit
_LIVE_TALLIES: "weakref.WeakSet[LLMTally]" = weakref.WeakSet()


@atexit.register
def _flush_all_dashboards() -> None:
    for tally in list(_LIVE_TALLIES):
        try:
            tally.flush()
        except Exception:
            pass


def _reset_tallies_after_fork() -> None:
    for tally in list(_LIVE_TALLIES):
        try:
            tally._reset_after_fork()
        except Exception:
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_tallies_after_fork)


_SIGNATURES: "weakref.WeakKeyDictionary[Any, Optional[inspect.Signature]]" = weakref.WeakKeyDictionary()
_SIGNATURE_LOCK = threading.Lock()

# Argument names whose values are never written to previews (matched case-insensitively)
_SECRET_ARG_RE = re.compile(
    r"^(?:.*api[_-]?key.*|.*secret.*|.*passw(?:or)?d.*|.*credential.*|authorization|"
    r"(?:access|auth|bearer|refresh|id|session)[_-]?token|token|cookie|x-api-key)$",
    re.IGNORECASE,
)
_REDACTED = "[redacted]"


def _signature_of(func: Callable[..., Any]) -> Optional[inspect.Signature]:
    """`inspect.signature` of the underlying function, cached per function object."""
    target = getattr(func, "__func__", func)
    try:
        with _SIGNATURE_LOCK:
            if target in _SIGNATURES:
                return _SIGNATURES[target]
    except TypeError:  # not weak-referenceable / not hashable
        pass
    try:
        sig: Optional[inspect.Signature] = inspect.signature(target)
    except (TypeError, ValueError):
        sig = None
    try:
        with _SIGNATURE_LOCK:
            _SIGNATURES[target] = sig
    except TypeError:
        pass
    return sig


def _bound_arguments(
    func: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    apply_defaults: bool = True,
) -> Optional[Tuple[inspect.Signature, Dict[str, Any]]]:
    """Bind a call to `func`'s signature, or None if that isn't possible."""
    sig = _signature_of(func)
    if sig is None:
        return None
    call_args = tuple(args)
    if inspect.ismethod(func):
        call_args = (func.__self__,) + call_args
    try:
        bound = sig.bind(*call_args, **kwargs)
    except TypeError:
        return None
    if apply_defaults:
        bound.apply_defaults()
    return sig, dict(bound.arguments)


def _declared_params(func: Callable[..., Any]) -> frozenset:
    """Names of the parameters `func` explicitly declares (not `*args` / `**kwargs`)."""
    sig = _signature_of(func)
    if sig is None:
        return frozenset()
    return frozenset(
        name
        for name, param in sig.parameters.items()
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )


def _call_time_model(
    func: Optional[Callable[..., Any]],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Optional[str]:
    """
    Return the model the wrapped function was called with: a `model` argument passed by
    keyword or position, or the function's own `model=` default.
    """
    val = kwargs.get("model")
    if isinstance(val, str) and val.strip():
        return val.strip()
    if func is not None:
        bound = _bound_arguments(func, args, kwargs)
        if bound is not None:
            val = bound[1].get("model")
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _mask_secrets(value: Any, _depth: int = 0) -> Any:
    """Replace values stored under secret-looking keys (api_key, password, ...) in dicts."""
    if _depth > 4:
        return value
    if isinstance(value, dict):
        return {
            k: (_REDACTED if isinstance(k, str) and _SECRET_ARG_RE.match(k) else _mask_secrets(v, _depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_mask_secrets(v, _depth + 1) for v in value]
    return value


def _call_inputs(
    func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    The arguments the caller actually passed, by parameter name, for previews and token
    estimates: drops `self` / `cls`, flattens `*args` / `**kwargs`, and masks secret-looking
    arguments.
    """
    bound = _bound_arguments(func, args, kwargs, apply_defaults=False)
    if bound is None:
        inputs: Dict[str, Any] = {}
        if args:
            inputs["args"] = list(args)
        inputs.update(kwargs)
        return _mask_secrets(inputs)
    sig, arguments = bound
    params = list(sig.parameters.values())
    inputs = {}
    for idx, param in enumerate(params):
        if param.name not in arguments:
            continue
        val = arguments[param.name]
        if idx == 0 and param.name in ("self", "cls"):
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            if val:
                inputs[param.name] = list(val)
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            inputs.update(val)
        else:
            inputs[param.name] = val
    return _mask_secrets(inputs)


def _input_preview_value(inputs: Dict[str, Any]) -> Any:
    """Show a lone text argument as itself instead of as a JSON object."""
    text_values = [v for v in inputs.values() if isinstance(v, str)]
    if len(inputs) == 1 and len(text_values) == 1:
        return text_values[0]
    return inputs


def _response_text(value: Any) -> Any:
    """Best-effort answer text from common SDK response shapes (falls back to `value`)."""
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    try:
        if isinstance(value, dict):
            for key in ("content", "text", "output_text"):
                if isinstance(value.get(key), str):
                    return value[key]
            return value
        # OpenAI Responses API
        out_text = getattr(value, "output_text", None)
        if isinstance(out_text, str) and out_text:
            return out_text
        # OpenAI Chat Completions
        choices = getattr(value, "choices", None)
        if isinstance(choices, (list, tuple)) and choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            text = getattr(choices[0], "text", None)
            if isinstance(text, str):
                return text
        # Anthropic Messages / LangChain messages
        content = getattr(value, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, (list, tuple)):
            parts = [getattr(block, "text", None) for block in content]
            texts = [t for t in parts if isinstance(t, str)]
            if texts:
                return "".join(texts)
        # Gemini (`.text` is a property that can raise when there is no text part)
        text = getattr(value, "text", None)
        if isinstance(text, str):
            return text
    except Exception:
        pass
    return value


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
    # OpenAI Responses API `response.output_text.delta` events carry the text in `delta`
    if isinstance(delta, str) and str(getattr(chunk, "type", "")).endswith("output_text.delta"):
        return delta
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

    def observe_snapshot(self, stream: Any) -> None:
        """
        If no chunk carried usage (e.g. the caller iterated Anthropic's `stream.text_stream`
        instead of the stream itself), read usage from the SDK's accumulated message snapshot.
        """
        if self.has_structured_usage:
            return
        for attr in ("current_message_snapshot", "current_completion_snapshot"):
            try:
                snapshot = getattr(stream, attr, None)
            except Exception:
                continue
            if snapshot is not None:
                self._observe_usage(extract_token_usage(snapshot, args=(), kwargs={}))
                return

    def observe_chunk(self, chunk: Any) -> None:
        self._observe_usage(extract_token_usage(chunk, args=(), kwargs={}))
        piece = _extract_chunk_text(chunk)
        if piece:
            self.collected_text.append(piece)

    def _observe_usage(self, u: TokenUsage) -> None:
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

    def build_final_usage(self, inputs: Dict[str, Any]) -> Tuple[TokenUsage, str]:
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
        fallback = extract_token_usage(joined_text, inputs=inputs)
        if self.detected_model and not fallback.detected_model:
            fallback.detected_model = self.detected_model
        return fallback, joined_text


class _SyncStreamProxy:
    """
    Transparent proxy around a synchronous generator or SDK stream object.
    Preserves `.close()`, `with ... as stream:`, `.response`, and custom SDK attributes,
    while updating the pre-logged `CallRecord` on completion or mid-stream failure.
    A proxy that is garbage-collected before the stream finishes is logged as `incomplete`.
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
        inputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._stream = stream
        # The original context manager when `__enter__` returned a different stream object
        self._cm: Any = None
        self._metadata = dict(metadata or {})
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
        self._inputs = inputs
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
            self._finalize_on_close()

    def __enter__(self) -> "_SyncStreamProxy":
        enter_fn = getattr(self._stream, "__enter__", None)
        if callable(enter_fn):
            try:
                entered = enter_fn()
            except BaseException as exc:
                # SDK stream managers send the request on enter; log the real failure
                self._finalize(status="failed", error_message=f"{type(exc).__name__}: {exc}")
                raise
            # e.g. Anthropic `MessageStreamManager.__enter__()` returns the real `MessageStream`
            if entered is not None and entered is not self._stream:
                self._cm = self._stream
                self._stream = entered
                self._iterator = None
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if exc_val is not None:
            self._finalize(status="failed", error_message=f"{type(exc_val).__name__}: {exc_val}")
        else:
            self._finalize(status="success", error_message=None)
        target = self._cm if self._cm is not None else self._stream
        exit_fn = getattr(target, "__exit__", None)
        if callable(exit_fn):
            return exit_fn(exc_type, exc_val, exc_tb)
        close_fn = getattr(target, "close", None)
        if callable(close_fn):
            close_fn()
        return False

    def __getattr__(self, name: str) -> Any:
        # Guard against recursion when `__init__` never ran (e.g. during `__del__`)
        if "_stream" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["_stream"], name)

    def __del__(self) -> None:
        if self.__dict__.get("_finalized", True):
            return
        try:
            self._finalize(
                status="incomplete",
                error_message="Stream was abandoned before it was fully consumed or closed",
            )
        except Exception:
            pass

    def _finalize_on_close(self) -> None:
        """
        `close()` before the stream was exhausted means the caller stopped early: log it as
        `incomplete`, unless the SDK's final snapshot shows the full response arrived.
        """
        if self._finalized:
            return
        self._acc.observe_snapshot(self._stream)
        if self._acc.has_structured_usage:
            self._finalize(status="success", error_message=None)
        else:
            self._finalize(
                status="incomplete",
                error_message="Stream was closed before it was fully consumed",
            )

    def _finalize(self, status: str, error_message: Optional[str]) -> None:
        if self._finalized:
            return
        self._finalized = True
        latency_ms = (time.perf_counter() - self._start_perf) * 1000.0
        self._acc.observe_snapshot(self._stream)
        usage, joined_text = self._acc.build_final_usage(self._inputs)
        estimated = usage.estimated and not (
            self._explicit_input_tokens is not None and self._explicit_output_tokens is not None
        )
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
        cost_usd, priced = self._tally._safe_calculate_cost(
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
            inputs=self._inputs,
            response=joined_text or "<stream>",
            estimated_tokens=estimated,
            metadata=self._metadata,
            unpriced=not priced,
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
        inputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._stream = stream
        # The original async context manager when `__aenter__` returned a different stream object
        self._cm: Any = None
        self._metadata = dict(metadata or {})
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
        self._inputs = inputs
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
            self._finalize_on_close()

    def close(self) -> Any:
        try:
            close_fn = getattr(self._stream, "close", None)
            if callable(close_fn):
                return close_fn()
        finally:
            self._finalize_on_close()

    async def __aenter__(self) -> "_AsyncStreamProxy":
        aenter_fn = getattr(self._stream, "__aenter__", None)
        if callable(aenter_fn):
            try:
                entered = await aenter_fn()
            except BaseException as exc:
                self._finalize(status="failed", error_message=f"{type(exc).__name__}: {exc}")
                raise
            if entered is not None and entered is not self._stream:
                self._cm = self._stream
                self._stream = entered
                self._aiterator = None
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if exc_val is not None:
            self._finalize(status="failed", error_message=f"{type(exc_val).__name__}: {exc_val}")
        else:
            self._finalize(status="success", error_message=None)
        target = self._cm if self._cm is not None else self._stream
        aexit_fn = getattr(target, "__aexit__", None)
        if callable(aexit_fn):
            return await aexit_fn(exc_type, exc_val, exc_tb)
        await self.aclose()
        return False

    def __getattr__(self, name: str) -> Any:
        # Guard against recursion when `__init__` never ran (e.g. during `__del__`)
        if "_stream" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["_stream"], name)

    def __del__(self) -> None:
        if self.__dict__.get("_finalized", True):
            return
        try:
            self._finalize(
                status="incomplete",
                error_message="Stream was abandoned before it was fully consumed or closed",
            )
        except Exception:
            pass

    def _finalize_on_close(self) -> None:
        """
        `close()` before the stream was exhausted means the caller stopped early: log it as
        `incomplete`, unless the SDK's final snapshot shows the full response arrived.
        """
        if self._finalized:
            return
        self._acc.observe_snapshot(self._stream)
        if self._acc.has_structured_usage:
            self._finalize(status="success", error_message=None)
        else:
            self._finalize(
                status="incomplete",
                error_message="Stream was closed before it was fully consumed",
            )

    def _finalize(self, status: str, error_message: Optional[str]) -> None:
        if self._finalized:
            return
        self._finalized = True
        latency_ms = (time.perf_counter() - self._start_perf) * 1000.0
        self._acc.observe_snapshot(self._stream)
        usage, joined_text = self._acc.build_final_usage(self._inputs)
        estimated = usage.estimated and not (
            self._explicit_input_tokens is not None and self._explicit_output_tokens is not None
        )
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
        cost_usd, priced = self._tally._safe_calculate_cost(
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
            inputs=self._inputs,
            response=joined_text or "<stream>",
            estimated_tokens=estimated,
            metadata=self._metadata,
            unpriced=not priced,
        )


class LLMTally:
    """
    Main manager for caching, retrying, and tracking LLM calls & costs.
    """

    def __init__(
        self,
        default_model: Optional[str] = None,
        log_path: Optional[Union[str, Path]] = None,
        dashboard_path: Optional[Union[str, Path]] = None,
        default_cache_enabled: Optional[bool] = None,
        default_ttl: Optional[float] = 300.0,
        default_max_requests: Optional[int] = None,
        default_max_cache_size: Optional[int] = 1000,
        default_retry_enabled: Optional[bool] = None,
        default_retries: Optional[int] = None,
        retry_delays: Optional[Sequence[float]] = None,
        auto_update_dashboard: Optional[bool] = None,
        custom_pricing: Optional[Dict[str, ModelPricing]] = None,
        sleep_func: Optional[Callable[[float], Any]] = None,
        async_sleep_func: Optional[Callable[[float], Any]] = None,
        *,
        dashboard_update_interval: float = 2.0,
        dashboard_days: Optional[float] = None,
        log_previews: bool = True,
        redact: Optional[Callable[[str], str]] = None,
        persist_logs: bool = True,
        on_record: Optional[Union[RecordSink, Sequence[RecordSink]]] = None,
        cache_backend: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        retry_on_rate_limit: bool = True,
        retry_jitter: float = 0.0,
        max_retry_after: float = 60.0,
        pricing_file: Optional[Union[str, Path]] = None,
        max_log_mb: Optional[float] = None,
        log_backups: int = 3,
        async_sinks: bool = False,
        sink_queue_size: int = 10_000,
        fallback_pricing: Optional[Tuple[float, float]] = None,
        raise_original_error: bool = False,
        dashboard_recent_calls: int = 500,
    ) -> None:
        """
        Every positional argument accepts `None` to mean "use the default":
        `default_model=None` (no model is assumed; calls whose model can't be detected are logged
        as `"unknown"` and flagged unpriced), `log_path=".llm_tally/logs.jsonl"`,
        `dashboard_path=".llm_tally/dashboard.html"`, `default_cache_enabled=False`,
        `default_retry_enabled=True`, `default_retries=3`, `retry_delays=(0.5, 1.0, 1.5)`,
        `auto_update_dashboard=True`. For `default_ttl`, `default_max_requests`, and
        `default_max_cache_size`, `None` means "no limit".

        Keyword-only options:
        - `dashboard_update_interval`: write the auto-updated dashboard at most once per this
          many seconds (pending updates are flushed at exit or via `flush()`); `0` = every call.
          The interval automatically stretches to 10x the last rebuild time so a large log never
          spends more than ~10% of a core on dashboard rebuilds.
        - `dashboard_days`: only include the last N days of logs in the dashboard.
        - `log_previews=False` stops input/output text and error message text from being written
          to logs and the dashboard (exception types and status codes are kept).
        - `redact`: applied to previews, error messages, retry-attempt messages, and string values
          in `config` before they are stored (e.g. to mask emails or keys). If it raises, the text
          is stored as `"[redaction failed]"` and the record is still kept.
        - `persist_logs=False` disables the JSONL file (use `on_record` sinks instead).
        - `max_log_mb` / `log_backups`: rotate the JSONL file to `<log>.1` ... `<log>.N` when it
          would exceed this size. Reports and the dashboard read the current file and backups.
        - `on_record`: callable(s) receiving every finished `CallRecord` (ship to a DB, OTel, ...).
        - `async_sinks=True` runs sinks on a background thread (bounded by `sink_queue_size`;
          records are dropped with a warning if it fills up) so slow sinks never delay LLM calls.
        - `cache_backend`: any object implementing `InMemoryCache`'s `fetch`/`store`/`clear`.
        - `metadata`: tags added to every record (e.g. `{"env": "prod"}`).
        - `retry_on_rate_limit`, `retry_jitter`, `max_retry_after`: HTTP 429 handling and backoff.
        - `pricing_file`: JSON file of model prices loaded on top of the built-in catalog.
        - `fallback_pricing=(input_per_1m, output_per_1m)`: price unknown models at this rate
          instead of $0. Those calls are still flagged `unpriced`.
        - `raise_original_error=True`: when all retries fail, re-raise the SDK's own exception
          (with `exc.llm_tally_attempts`) instead of `RetryExhaustedError`. With no retries
          configured the original exception is always raised.
        - `dashboard_recent_calls`: how many of the latest calls the dashboard's search table holds.

        `LLM_TALLY_DIR` (environment variable) changes the default `.llm_tally` directory.
        """
        self.default_model: Optional[str] = str(default_model) if default_model else DEFAULT_MODEL
        self.log_path: Path = Path(log_path if log_path is not None else DEFAULT_LOG_PATH)
        self.dashboard_path: Path = Path(
            dashboard_path if dashboard_path is not None else DEFAULT_DASHBOARD_PATH
        )
        self.default_cache_enabled: bool = (
            False if default_cache_enabled is None else bool(default_cache_enabled)
        )
        self.default_ttl: Optional[float] = default_ttl
        self.default_max_requests: Optional[int] = default_max_requests
        self.default_max_cache_size: Optional[int] = default_max_cache_size
        self.default_retry_enabled: bool = (
            True if default_retry_enabled is None else bool(default_retry_enabled)
        )
        self.default_retries: int = 3 if default_retries is None else int(default_retries)
        if self.default_retries < 0:
            raise ValueError(f"default_retries must be >= 0, got {self.default_retries}")
        self.retry_delays: Tuple[float, ...] = tuple(
            float(d) for d in (DEFAULT_RETRY_DELAYS if retry_delays is None else retry_delays)
        )
        self.auto_update_dashboard: bool = (
            True if auto_update_dashboard is None else bool(auto_update_dashboard)
        )
        self.dashboard_update_interval: float = max(0.0, float(dashboard_update_interval))
        self.dashboard_days: Optional[float] = dashboard_days
        self.log_previews: bool = bool(log_previews)
        self.redact: Optional[Callable[[str], str]] = redact
        self.persist_logs: bool = bool(persist_logs)
        if on_record is None:
            self.sinks: List[RecordSink] = []
        elif callable(on_record):
            self.sinks = [on_record]
        else:
            self.sinks = list(on_record)
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.retry_on_rate_limit: bool = bool(retry_on_rate_limit)
        self.retry_jitter: float = max(0.0, float(retry_jitter))
        self.max_retry_after: float = float(max_retry_after)
        self.raise_original_error: bool = bool(raise_original_error)
        self.fallback_pricing: Optional[Tuple[float, float]] = (
            (float(fallback_pricing[0]), float(fallback_pricing[1]))
            if fallback_pricing is not None
            else None
        )
        self.dashboard_recent_calls: int = max(0, int(dashboard_recent_calls))
        self._warned_models: set = set()
        self._warned_lock = threading.Lock()

        self.cache_store = (
            cache_backend
            if cache_backend is not None
            else InMemoryCache(
                default_ttl=self.default_ttl,
                default_max_requests=self.default_max_requests,
                max_size=self.default_max_cache_size,
            )
        )
        self.pricing = PricingRegistry(custom_pricing=custom_pricing)
        if pricing_file is not None:
            self.pricing.load_file(pricing_file)
        self.logger = JSONLLogger(
            log_path=self.log_path,
            max_bytes=int(max_log_mb * 1024 * 1024) if max_log_mb else None,
            backups=log_backups,
        )
        self._sleep_func = sleep_func or time.sleep
        self._async_sleep_func = async_sleep_func or asyncio.sleep

        self._dashboard_lock = threading.Lock()
        self._last_dashboard_write = float("-inf")
        self._dashboard_dirty = False
        self._dashboard_timer: Optional[threading.Timer] = None
        self._last_dashboard_build_seconds = 0.0

        self.async_sinks: bool = bool(async_sinks)
        self._sink_queue_size = max(1, int(sink_queue_size))
        self._sink_queue: "Optional[queue.Queue[CallRecord]]" = None
        self._sink_pid: Optional[int] = None
        self._sink_start_lock = threading.Lock()
        _LIVE_TALLIES.add(self)

    # -------------------------------------------------------------------------
    # Per-call Metadata, Sinks & Retry Config Helpers
    # -------------------------------------------------------------------------
    @contextlib.contextmanager
    def context(self, **metadata: Any) -> Iterator[Dict[str, Any]]:
        """
        Attach metadata (e.g. `user_id`, `tenant`, `request_id`) to every call made inside
        the block, including calls made through decorators. Nested blocks merge; safe across
        threads and asyncio tasks:

            with tally.context(user_id=user.id, prompt_version="v3"):
                answer = ask_llm(question)
        """
        merged = {**_CONTEXT_METADATA.get(), **metadata}
        token = _CONTEXT_METADATA.set(merged)
        try:
            yield merged
        finally:
            _CONTEXT_METADATA.reset(token)

    def _resolve_metadata(self, explicit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge metadata: instance defaults < `tally.context(...)` < decorator / `run(metadata=...)`."""
        return {**self.metadata, **_CONTEXT_METADATA.get(), **(explicit or {})}

    @staticmethod
    def wrap_context(func: F) -> F:
        """
        Carry the current `tally.context(...)` metadata into another thread. `contextvars`
        are not inherited by `ThreadPoolExecutor` / `threading.Thread` workers, so wrap the
        function you submit:

            with tally.context(user_id="alice"):
                pool.map(tally.wrap_context(ask_llm), prompts)
        """
        captured = _CONTEXT_METADATA.get()

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _async_bound(*args: Any, **kwargs: Any) -> Any:
                token = _CONTEXT_METADATA.set({**_CONTEXT_METADATA.get(), **captured})
                try:
                    return await func(*args, **kwargs)
                finally:
                    _CONTEXT_METADATA.reset(token)

            return _async_bound  # type: ignore[return-value]

        @functools.wraps(func)
        def _bound(*args: Any, **kwargs: Any) -> Any:
            token = _CONTEXT_METADATA.set({**_CONTEXT_METADATA.get(), **captured})
            try:
                return func(*args, **kwargs)
            finally:
                _CONTEXT_METADATA.reset(token)

        return _bound  # type: ignore[return-value]

    def add_sink(self, sink: RecordSink) -> None:
        """Register a callable that receives every finished `CallRecord`."""
        self.sinks.append(sink)

    def _deliver_to_sinks(self, record: CallRecord) -> None:
        for sink in list(self.sinks):
            try:
                sink(record)
            except Exception as sink_exc:
                self._warn(f"llm-tally record sink {sink!r} raised: {sink_exc}")

    def _emit_to_sinks(self, record: CallRecord) -> None:
        if not self.sinks:
            return
        if not self.async_sinks:
            self._deliver_to_sinks(record)
            return
        sink_queue = self._ensure_sink_worker()
        try:
            sink_queue.put_nowait(record)
        except queue.Full:
            self._warn(
                f"llm-tally sink queue is full ({self._sink_queue_size} records); "
                f"dropped record {record.id}. Sinks are slower than your LLM traffic."
            )

    def _ensure_sink_worker(self) -> "queue.Queue[CallRecord]":
        """Start (or, after `fork()`, restart) the background sink thread."""
        with self._sink_start_lock:
            if self._sink_queue is None or self._sink_pid != os.getpid():
                sink_queue: "queue.Queue[CallRecord]" = queue.Queue(maxsize=self._sink_queue_size)
                worker = threading.Thread(
                    target=self._sink_worker_loop,
                    args=(sink_queue,),
                    name="llm-tally-sinks",
                    daemon=True,
                )
                self._sink_queue = sink_queue
                self._sink_pid = os.getpid()
                worker.start()
            return self._sink_queue

    def _sink_worker_loop(self, sink_queue: "queue.Queue[CallRecord]") -> None:
        while True:
            record = sink_queue.get()
            try:
                self._deliver_to_sinks(record)
            finally:
                sink_queue.task_done()

    def _drain_sinks(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds for queued sink deliveries. Returns True if drained."""
        sink_queue = self._sink_queue
        if sink_queue is None or self._sink_pid != os.getpid():
            return True
        deadline = time.monotonic() + timeout
        while sink_queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True

    # -------------------------------------------------------------------------
    # Privacy: previews, error text, and config redaction
    # -------------------------------------------------------------------------
    def _safe_redact(self, text: str) -> str:
        if self.redact is None:
            return text
        try:
            return str(self.redact(text))
        except Exception as exc:
            self._warn(
                f"llm-tally redact function raised {type(exc).__name__}: {exc}; "
                "stored '[redaction failed]' instead of the text"
            )
            return "[redaction failed]"

    def _preview(self, value: Any) -> str:
        if not self.log_previews:
            return ""
        return format_preview(value, redact=self._safe_redact if self.redact else None)

    def _scrub_error(self, message: Optional[str]) -> Optional[str]:
        """Hide (log_previews=False) and/or redact error text, keeping the exception type."""
        if message is None:
            return None
        if not self.log_previews:
            head, sep, _ = message.partition(":")
            if sep:
                message = f"{head}: [message hidden: log_previews=False]"
        return self._safe_redact(message)

    def _scrub_attempts(self, attempts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.log_previews and self.redact is None:
            return attempts
        scrubbed = []
        for attempt in attempts:
            item = dict(attempt)
            if isinstance(item.get("message"), str):
                item["message"] = (
                    "[message hidden: log_previews=False]"
                    if not self.log_previews
                    else self._safe_redact(item["message"])
                )
            scrubbed.append(item)
        return scrubbed

    def _redact_obj(self, value: Any) -> Any:
        """Apply `redact` to every string inside `config`-like dicts / lists."""
        if self.redact is None:
            return value
        if isinstance(value, str):
            return self._safe_redact(value)
        if isinstance(value, dict):
            return {k: self._redact_obj(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact_obj(v) for v in value]
        return value

    def _retry_config(
        self,
        enabled: bool,
        retries: int,
        delays: Tuple[float, ...],
        extra_exceptions: Tuple[Type[BaseException], ...],
    ) -> RetryConfig:
        return RetryConfig(
            enabled=enabled,
            retries=retries,
            delays=delays,
            extra_exceptions=extra_exceptions,
            retry_on_rate_limit=self.retry_on_rate_limit,
            jitter=self.retry_jitter,
            max_retry_after=self.max_retry_after,
            raise_original=self.raise_original_error,
        )

    @staticmethod
    def _route_options(
        func: Callable[..., Any],
        options: Dict[str, Any],
        kwargs: Dict[str, Any],
        tally_options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Decide which keyword arguments of `run()` / `cached_call()` / `retry_call()` belong to
        `func`. An option whose name `func` declares as a parameter (e.g. `model` or `metadata`
        on `client.chat.completions.create`) is passed through to `func` instead of being
        consumed. `model` is also still used for pricing, and is passed through to functions
        that take `**kwargs`. `tally_options={...}` always sets llm-tally options.
        Mutates `kwargs`; returns the options llm-tally should use.
        """
        sig = _signature_of(func)
        declared = _declared_params(func)
        has_var_kw = sig is not None and any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        resolved = dict(options)
        for name, value in options.items():
            if value is None or value == () or name in kwargs:
                continue
            if name in declared or (name == "model" and has_var_kw):
                kwargs[name] = value
                if name != "model":
                    resolved[name] = () if name == "extra_exceptions" else None
        if tally_options:
            unknown = set(tally_options) - set(options)
            if unknown:
                raise TypeError(f"unknown tally_options: {sorted(unknown)}")
            resolved.update(tally_options)
        return resolved

    def load_pricing(self, path: Union[str, Path]) -> int:
        """Load model prices from a JSON file (see `PricingRegistry.load_file`)."""
        return self.pricing.load_file(path)

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
        warn: bool = True,
    ) -> Tuple[float, bool]:
        """
        Return `(cost_usd, priced)` for a completed call. A model with no known price is not
        guessed: it costs 0 (or `fallback_pricing`, if configured) with `priced=False`, and a
        warning is logged once per model. Never raises, so telemetry can't break a paid call.
        """
        try:
            pricing = self.pricing.lookup(model)
        except Exception as exc:
            self._warn(f"llm-tally pricing lookup failed for model {model!r}: {exc}")
            pricing = None
        priced = pricing is not None
        if pricing is None:
            if warn:
                self._warn_unpriced(model)
            if self.fallback_pricing is None:
                return 0.0, False
            pricing = ModelPricing(
                model=str(model),
                provider="custom",
                input_cost_per_1m=float(self.fallback_pricing[0]),
                output_cost_per_1m=float(self.fallback_pricing[1]),
            )
        try:
            cost = pricing.compute_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                input_excludes_cached=input_excludes_cached,
            )
        except Exception as exc:
            self._warn(f"llm-tally could not compute cost for model {model!r}: {exc}")
            return 0.0, False
        return cost, priced

    def _warn_unpriced(self, model: str) -> None:
        key = str(model)
        with self._warned_lock:
            if key in self._warned_models:
                return
            self._warned_models.add(key)
        if key == UNKNOWN_MODEL:
            message = (
                "llm-tally could not tell which model a call used, so it is logged as "
                "'unknown' with no cost. Pass `model=` to the decorator, call the function with "
                "`model=...`, or return the SDK response object (which reports its model)."
            )
        else:
            message = (
                f"llm-tally has no price for model {key!r}; its calls are logged as unpriced "
                f"(cost {'from fallback_pricing' if self.fallback_pricing else '$0'}) and flagged on "
                f"the dashboard. Set a price with `tally.register_model({key!r}, input_cost_per_1m=..., "
                "output_cost_per_1m=...)` or `pricing_file=`; earlier calls are re-priced in reports."
            )
        # Apps with logging configured get it in their logs; otherwise a UserWarning that
        # points at the calling line
        if logger.hasHandlers():
            logger.warning(message)
            return
        try:
            warnings.warn(message, UserWarning, stacklevel=5)
        except Warning:
            pass

    def _cache_key_for(
        self,
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        feature: str,
        model: str,
        cfg_dict: Dict[str, Any],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Build the cache key, or return `(None, {})` (no caching for this call) if the inputs
        can't be hashed, e.g. a very deep object graph on `self`. Never raises.
        """
        settings = {"feature": feature, "model": model, "config": cfg_dict}
        try:
            return build_cache_key(
                self._func_identifier(func), args, kwargs, settings=settings, func=func
            )
        except (RecursionError, Exception) as exc:
            self._warn(
                f"llm-tally could not build a cache key for {self._func_identifier(func)} "
                f"({type(exc).__name__}); the call runs without caching"
            )
            return None, {}

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
        metadata: Optional[Dict[str, Any]] = None,
        tally_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Feature 1 Explicit Function: Execute `func(*args, **kwargs)` using the local
        in-memory cache (always on here, regardless of `default_cache_enabled`).
        With `track_cost=False` and `retry=False` nothing is logged; otherwise this is
        `run(..., cache=True)`. Options that `func` itself declares are passed to `func`
        (see `run`).
        """
        opts = self._route_options(
            func,
            {"ttl": ttl, "max_requests": max_requests, "config": config, "model": model,
             "feature": feature, "metadata": metadata},
            kwargs,
            tally_options,
        )
        return self._cached_call_impl(
            func,
            args,
            kwargs,
            ttl=opts["ttl"],
            max_requests=opts["max_requests"],
            config=opts["config"],
            model=opts["model"],
            feature=opts["feature"],
            track_cost=track_cost,
            retry=retry,
            metadata=opts["metadata"],
        )

    def _cached_call_impl(
        self,
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        ttl: Optional[float],
        max_requests: Optional[int],
        config: Optional[Dict[str, Any]],
        model: Optional[str],
        feature: Optional[str],
        track_cost: bool,
        retry: bool,
        metadata: Optional[Dict[str, Any]],
    ) -> Any:
        if not track_cost and not retry:
            effective_ttl = self.default_ttl if ttl is None else ttl
            effective_max_req = self.default_max_requests if max_requests is None else max_requests
            func_name = self._func_identifier(func)
            settings_meta = {
                "config": config or {},
                "model": model or _call_time_model(func, args, kwargs) or self.default_model,
            }
            try:
                key: Optional[str]
                key, snapshot = build_cache_key(
                    func_name, args, kwargs, settings=settings_meta, func=func
                )
            except (RecursionError, Exception) as exc:
                self._warn(
                    f"llm-tally could not build a cache key for {func_name} "
                    f"({type(exc).__name__}); the call runs without caching"
                )
                return func(*args, **kwargs)

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

        return self._dispatch(
            func,
            args,
            kwargs,
            feature=feature or getattr(func, "__name__", "cached_call"),
            model=model,
            config=config,
            cache=True,
            ttl=ttl,
            max_requests=max_requests,
            retry=retry,
            metadata=metadata,
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable[[F], F]:
        """
        Feature 1 Decorator: Wrap a function with the self-evicting local in-memory cache.
        """

        def _decorator(func: F) -> F:
            options = dict(
                ttl=ttl,
                max_requests=max_requests,
                config=config,
                feature=feature or getattr(func, "__name__", "cached_call"),
                model=model,
                track_cost=track_cost,
                retry=retry,
                metadata=metadata,
            )
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return await self._cached_call_impl(func, args, kwargs, **options)

                return _async_wrapper  # type: ignore[return-value]

            @functools.wraps(func)
            def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return self._cached_call_impl(func, args, kwargs, **options)

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
        tally_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Feature 2 Explicit Function: Execute `func(*args, **kwargs)` with automatic
        retries on HTTP 500/5xx or network issues. Options that `func` declares are passed
        to `func` (see `run`).
        """
        opts = self._route_options(
            func,
            {"retries": retries, "retry_delays": retry_delays, "extra_exceptions": extra_exceptions},
            kwargs,
            tally_options,
        )
        retries, retry_delays = opts["retries"], opts["retry_delays"]
        extra_exceptions = tuple(opts["extra_exceptions"] or ())
        cfg = self._retry_config(
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
        metadata: Optional[Dict[str, Any]] = None,
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
                        metadata=metadata,
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
                    metadata=metadata,
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
        metadata: Optional[Dict[str, Any]] = None,
        tally_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute `func(*args, **kwargs)` directly with caching, retries, and cost tracking.

        Keyword arguments that `func` declares are passed to `func`, never swallowed. So
        `tally.run(client.chat.completions.create, model="gpt-4o", messages=..., metadata={...})`
        sends `model` and `metadata` to the SDK (and prices the call as `gpt-4o`). Use
        `tally_options={"metadata": {...}, "feature": ...}` to set llm-tally options whose
        names collide with `func`'s parameters.
        """
        opts = self._route_options(
            func,
            {"feature": feature, "model": model, "config": config, "cache": cache, "ttl": ttl,
             "max_requests": max_requests, "retry": retry, "retries": retries,
             "retry_delays": retry_delays, "extra_exceptions": extra_exceptions,
             "input_tokens": input_tokens, "output_tokens": output_tokens, "metadata": metadata},
            kwargs,
            tally_options,
        )
        return self._dispatch(
            func,
            args,
            kwargs,
            feature=opts["feature"],
            model=opts["model"],
            config=opts["config"],
            cache=opts["cache"],
            ttl=opts["ttl"],
            max_requests=opts["max_requests"],
            retry=opts["retry"],
            retries=opts["retries"],
            retry_delays=opts["retry_delays"],
            extra_exceptions=tuple(opts["extra_exceptions"] or ()),
            input_tokens=opts["input_tokens"],
            output_tokens=opts["output_tokens"],
            metadata=opts["metadata"],
        )

    def _dispatch(
        self,
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Run `func(*args, **kwargs)` through the sync or async pipeline (kwargs go to `func`)."""
        feat_name = feature or getattr(func, "__name__", "default")
        execute = self._execute_async if inspect.iscoroutinefunction(func) else self._execute_sync
        return execute(
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
            metadata=metadata,
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
        inputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Log an initial `CallRecord` immediately (so un-iterated streams are never missed)
        and return a transparent `_SyncStreamProxy` or `_AsyncStreamProxy` that updates
        the record in-place as the stream is consumed or closed.
        """
        initial_usage = extract_token_usage("", inputs=inputs)
        in_tok = (
            explicit_input_tokens
            if explicit_input_tokens is not None
            else initial_usage.input_tokens
        )
        out_tok = explicit_output_tokens if explicit_output_tokens is not None else 0
        final_model = str(model or cfg_dict.get("model") or chosen_model)
        initial_cost, priced = self._safe_calculate_cost(
            model=final_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            warn=False,
        )
        latency_ms = (time.perf_counter() - start_perf) * 1000.0
        rec = self._record_call(
            feature=feature,
            function_name=func_short,
            model=final_model,
            config=cfg_dict,
            # Replaced by the final record once the stream finishes, fails, or is abandoned
            status="streaming",
            cache_hit=False,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=0,
            cache_creation_input_tokens=0,
            cost_usd=initial_cost,
            unpriced=not priced,
            retries=len(failed_attempts),
            error_message=None,
            attempt_errors=[a.to_dict() for a in failed_attempts],
            inputs=inputs,
            response="<stream>",
            metadata=metadata,
            # Sinks get the finished record from the proxy, not this placeholder
            emit=False,
        )
        rec_id = rec.id if rec is not None else uuid.uuid4().hex[:12]
        rec_ts = rec.timestamp if rec is not None else datetime.now(timezone.utc).isoformat()

        is_async_stream = (
            inspect.isasyncgen(stream)
            or hasattr(stream, "__anext__")
            or hasattr(stream, "__aiter__")
            or (hasattr(stream, "__aenter__") and not hasattr(stream, "__enter__"))
        )
        if is_async_stream:
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
                inputs=inputs,
                metadata=metadata,
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
            inputs=inputs,
            metadata=metadata,
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        use_cache = self.default_cache_enabled if cache is None else cache
        effective_ttl = self.default_ttl if ttl is None else ttl
        effective_max_req = self.default_max_requests if max_requests is None else max_requests
        use_retry = self.default_retry_enabled if retry is None else retry
        effective_retries = self.default_retries if retries is None else retries
        effective_delays = self.retry_delays if retry_delays is None else tuple(retry_delays)
        call_model = _call_time_model(func, args, kwargs)
        chosen_model = (
            model
            or (config or {}).get("model")
            or call_model
            or self.default_model
            or UNKNOWN_MODEL
        )
        meta = self._resolve_metadata(metadata)
        cfg_dict = dict(config or {})
        inputs = _call_inputs(func, args, kwargs)

        func_short = getattr(func, "__qualname__", getattr(func, "__name__", "callable"))
        cache_key, snapshot = (
            self._cache_key_for(func, args, kwargs, feature, chosen_model, cfg_dict)
            if use_cache
            else (None, {})
        )
        use_cache = cache_key is not None

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
                    inputs=inputs,
                    response=cached_output,
                    metadata=meta,
                )
                return cached_output

        # 2. Execute with Retry Engine
        retry_cfg = self._retry_config(
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
                inputs=inputs,
                response=None,
                metadata=meta,
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
                inputs=inputs,
                response=None,
                metadata=meta,
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
                inputs=inputs,
                metadata=meta,
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
        usage = extract_token_usage(result, inputs=inputs)
        in_tok = explicit_input_tokens if explicit_input_tokens is not None else usage.input_tokens
        out_tok = (
            explicit_output_tokens if explicit_output_tokens is not None else usage.output_tokens
        )
        final_model = str(
            model
            or cfg_dict.get("model")
            or usage.detected_model
            or call_model
            or self.default_model
            or UNKNOWN_MODEL
        )
        estimated = usage.estimated and not (
            explicit_input_tokens is not None and explicit_output_tokens is not None
        )
        cost_usd, priced = self._safe_calculate_cost(
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
            inputs=inputs,
            response=result,
            metadata=meta,
            estimated_tokens=estimated,
            unpriced=not priced,
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        use_cache = self.default_cache_enabled if cache is None else cache
        effective_ttl = self.default_ttl if ttl is None else ttl
        effective_max_req = self.default_max_requests if max_requests is None else max_requests
        use_retry = self.default_retry_enabled if retry is None else retry
        effective_retries = self.default_retries if retries is None else retries
        effective_delays = self.retry_delays if retry_delays is None else tuple(retry_delays)
        call_model = _call_time_model(func, args, kwargs)
        chosen_model = (
            model
            or (config or {}).get("model")
            or call_model
            or self.default_model
            or UNKNOWN_MODEL
        )
        meta = self._resolve_metadata(metadata)
        cfg_dict = dict(config or {})
        inputs = _call_inputs(func, args, kwargs)

        func_short = getattr(func, "__qualname__", getattr(func, "__name__", "callable"))
        cache_key, snapshot = (
            self._cache_key_for(func, args, kwargs, feature, chosen_model, cfg_dict)
            if use_cache
            else (None, {})
        )
        use_cache = cache_key is not None

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
                    inputs=inputs,
                    response=cached_output,
                    metadata=meta,
                )
                return cached_output

        # 2. Execute with Async Retry Engine
        retry_cfg = self._retry_config(
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
                inputs=inputs,
                response=None,
                metadata=meta,
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
                inputs=inputs,
                response=None,
                metadata=meta,
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
                inputs=inputs,
                metadata=meta,
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
        usage = extract_token_usage(result, inputs=inputs)
        in_tok = explicit_input_tokens if explicit_input_tokens is not None else usage.input_tokens
        out_tok = (
            explicit_output_tokens if explicit_output_tokens is not None else usage.output_tokens
        )
        final_model = str(
            model
            or cfg_dict.get("model")
            or usage.detected_model
            or call_model
            or self.default_model
            or UNKNOWN_MODEL
        )
        estimated = usage.estimated and not (
            explicit_input_tokens is not None and explicit_output_tokens is not None
        )
        cost_usd, priced = self._safe_calculate_cost(
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
            inputs=inputs,
            response=result,
            metadata=meta,
            estimated_tokens=estimated,
            unpriced=not priced,
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
        inputs: Dict[str, Any],
        response: Any,
        cache_creation_input_tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        estimated_tokens: bool = False,
        emit: bool = True,
        unpriced: bool = False,
    ) -> Optional[CallRecord]:
        return self._persist_record(
            record_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            update=False,
            emit=emit,
            feature=feature,
            function_name=function_name,
            model=model,
            config=config,
            status=status,
            cache_hit=cache_hit,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cost_usd=cost_usd,
            retries=retries,
            error_message=error_message,
            attempt_errors=attempt_errors,
            inputs=inputs,
            response=response,
            cache_creation_input_tokens=cache_creation_input_tokens,
            metadata=metadata,
            estimated_tokens=estimated_tokens,
            unpriced=unpriced,
        )

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
        inputs: Dict[str, Any],
        response: Any,
        cache_creation_input_tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        estimated_tokens: bool = False,
        unpriced: bool = False,
    ) -> Optional[CallRecord]:
        return self._persist_record(
            record_id=record_id,
            timestamp=timestamp,
            update=True,
            emit=True,
            feature=feature,
            function_name=function_name,
            model=model,
            config=config,
            status=status,
            cache_hit=cache_hit,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cost_usd=cost_usd,
            retries=retries,
            error_message=error_message,
            attempt_errors=attempt_errors,
            inputs=inputs,
            response=response,
            cache_creation_input_tokens=cache_creation_input_tokens,
            metadata=metadata,
            estimated_tokens=estimated_tokens,
            unpriced=unpriced,
        )

    def _persist_record(
        self,
        record_id: str,
        timestamp: str,
        update: bool,
        emit: bool,
        inputs: Dict[str, Any],
        response: Any,
        metadata: Optional[Dict[str, Any]],
        latency_ms: float,
        **fields: Any,
    ) -> Optional[CallRecord]:
        """
        Build a `CallRecord`, append it (or update it in place for streams), refresh the
        dashboard if due, and hand it to the sinks. Telemetry failures only warn.
        """
        try:
            fields["error_message"] = self._scrub_error(fields.get("error_message"))
            fields["attempt_errors"] = self._scrub_attempts(list(fields.get("attempt_errors") or []))
            fields["config"] = self._redact_obj(dict(fields.get("config") or {}))
            record = CallRecord(
                id=record_id,
                timestamp=timestamp,
                latency_ms=round(latency_ms, 2),
                input_preview=self._preview(_input_preview_value(inputs)),
                output_preview=self._preview(_response_text(response)),
                metadata=dict(metadata or {}),
                **fields,
            )
        except Exception as build_exc:
            self._warn(f"llm-tally failed to build telemetry record: {build_exc}")
            return None

        logged = False
        if self.persist_logs:
            try:
                if update:
                    self.logger.update(record)
                else:
                    self.logger.append(record)
                logged = True
            except Exception as log_exc:
                verb = "update stream telemetry log" if update else "write telemetry log"
                self._warn(f"llm-tally failed to {verb}: {log_exc}")
            if logged:
                try:
                    self._maybe_update_dashboard()
                except Exception as dash_exc:
                    self._warn(f"llm-tally failed to write dashboard: {dash_exc}")
        # Sinks are independent of the local file: a read-only filesystem must not lose records
        if emit:
            self._emit_to_sinks(record)
        if self.persist_logs and not logged and not self.sinks:
            return None
        return record

    @staticmethod
    def _warn(message: str) -> None:
        try:
            warnings.warn(message, RuntimeWarning, stacklevel=4)
        except Warning:
            pass

    def _maybe_update_dashboard(self) -> None:
        """
        Regenerate the dashboard, at most once per `dashboard_update_interval` seconds.
        A skipped update is written by a background timer when the interval ends.
        """
        if not self.auto_update_dashboard:
            return
        with self._dashboard_lock:
            now = time.monotonic()
            interval = self.dashboard_update_interval
            if interval > 0:
                # Never spend more than ~10% of the time rebuilding a large dashboard
                interval = max(interval, 10.0 * self._last_dashboard_build_seconds)
            remaining = interval - (now - self._last_dashboard_write)
            if interval > 0 and remaining > 0:
                self._dashboard_dirty = True
                if self._dashboard_timer is None:
                    timer = threading.Timer(remaining, self._flush_from_timer)
                    timer.daemon = True
                    self._dashboard_timer = timer
                    timer.start()
                return
            if not self._inline_dashboard_ok():
                # A large log takes a while to rebuild: never block the caller (or the event loop)
                self._dashboard_dirty = True
                if self._dashboard_timer is None:
                    timer = threading.Timer(0.0, self._flush_from_timer)
                    timer.daemon = True
                    self._dashboard_timer = timer
                    timer.start()
                return
            self._last_dashboard_write = now
            self._dashboard_dirty = False
        self.generate_dashboard()

    def _inline_dashboard_ok(self) -> bool:
        """Rebuild inline only while it is cheap (small log and fast previous builds)."""
        if self._last_dashboard_build_seconds > _INLINE_DASHBOARD_MAX_SECONDS:
            return False
        try:
            size = os.path.getsize(self.log_path)
        except OSError:
            return True
        return size <= _INLINE_DASHBOARD_MAX_LOG_BYTES

    def _reset_after_fork(self) -> None:
        """A forked child inherits lock state but not threads: start clean."""
        self._dashboard_lock = threading.Lock()
        self._dashboard_timer = None
        self._sink_start_lock = threading.Lock()
        self._warned_lock = threading.Lock()

    def _flush_from_timer(self) -> None:
        with self._dashboard_lock:
            self._dashboard_timer = None
        try:
            self.flush()
        except Exception as exc:
            self._warn(f"llm-tally failed to write dashboard: {exc}")

    def flush(self, sink_timeout: float = 5.0) -> Optional[Path]:
        """
        Deliver queued `async_sinks` records (waiting up to `sink_timeout` seconds) and write any
        dashboard update skipped by throttling. Returns the dashboard path, or None if no update
        was pending. Runs automatically after the throttle interval and at interpreter exit.
        """
        self._drain_sinks(sink_timeout)
        with self._dashboard_lock:
            if not self._dashboard_dirty:
                return None
            self._dashboard_dirty = False
            self._last_dashboard_write = time.monotonic()
        if not self.log_path.exists():
            # Log was cleared or its directory removed; don't recreate it with an empty dashboard
            return None
        return self.generate_dashboard()

    # -------------------------------------------------------------------------
    # Feature 3: Analytics & Dashboard Generation
    # -------------------------------------------------------------------------
    def _load_records(self, days: Optional[float], now: Optional[datetime] = None, copy: bool = False) -> List[CallRecord]:
        """Records from the log and its rotated backups, with newly priced models re-priced."""
        records = self.logger.read_all(copy_records=copy, include_backups=True)
        return reprice_records(filter_records_since(records, days, now=now), self.pricing)

    def get_records(self, days: Optional[float] = None) -> List[CallRecord]:
        """
        Read logged call records (current log file plus rotated backups; only the last `days`
        days if given). Calls to models that were unpriced when logged but have a price now are
        returned re-priced.
        """
        return self._load_records(days, copy=True)

    def get_report(
        self,
        now: Optional[datetime] = None,
        days: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Return the full analytics dictionary answering:
        - `spend_summary["this_week_usd"]`: How much money did I spend this week?
        - `feature_costs`: Which feature (per config) costs the most?
        - `failure_stats`: Which calls fail the most?
        - `latency_stats` / `slowest_calls`: Which calls are slow?
        - `metadata_costs`: Which user / tenant / prompt version (any metadata key) costs the most?
        """
        records = self._load_records(days, now=now)
        return analyze_logs(records, now=now, recent_limit=self.dashboard_recent_calls)

    def render_dashboard(
        self,
        title: str = DEFAULT_DASHBOARD_TITLE,
        days: Optional[float] = None,
    ) -> str:
        """Render the self-contained HTML dashboard as a string."""
        window = self.dashboard_days if days is None else days
        records = self._load_records(window)
        return render_dashboard_html(records, title=title, recent_limit=self.dashboard_recent_calls)

    def generate_dashboard(
        self,
        dashboard_path: Optional[Union[str, Path]] = None,
        title: str = DEFAULT_DASHBOARD_TITLE,
        days: Optional[float] = None,
    ) -> Path:
        """Write/update the static HTML dashboard file on disk and return its path."""
        target = Path(dashboard_path) if dashboard_path is not None else self.dashboard_path
        window = self.dashboard_days if days is None else days
        started = time.perf_counter()
        records = self._load_records(window)
        path = write_dashboard_file(
            records, dashboard_path=target, title=title, recent_limit=self.dashboard_recent_calls
        )
        self._last_dashboard_build_seconds = time.perf_counter() - started
        return path

    def clear_cache(self) -> None:
        """Clear all entries from the cache."""
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
