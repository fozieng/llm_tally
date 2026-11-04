"""
Feature 2: Resilient Retry Engine for HTTP 500/5xx, 429 Rate Limits & Network Failures.

Default behavior:
- On failed attempts caused by an HTTP 500/5xx status, an HTTP 429 rate limit, or a
  network/timeout issue, retries 3 times with delays of 0.5s, 1.0s, and 1.5s
  (4 total attempts: 1 initial + 3 retries).
- A server-provided `Retry-After` / `retry-after-ms` header (or `retry_after` attribute)
  extends the wait, capped at `max_retry_after` seconds.
- Optional `jitter` adds up to `jitter * delay` random extra wait to spread out retries.
- If all attempts fail, raises `RetryExhaustedError` containing the complete list of
  errors from every attempt.
"""

from __future__ import annotations

import asyncio
import random
import socket
import time
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

DEFAULT_RETRY_DELAYS: Tuple[float, ...] = (0.5, 1.0, 1.5)

_NON_RETRYABLE_BASE_EXCEPTIONS = (
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
    asyncio.CancelledError,
)

_NETWORK_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "NetworkError",
        "RemoteProtocolError",
        "ConnectionClosedError",
        "ServerDisconnectedError",
        "ClientConnectorError",
    }
)

_SERVER_ERROR_EXCEPTION_NAMES = frozenset(
    {
        "InternalServerError",
        "ServiceUnavailableError",
        "BadGatewayError",
        "GatewayTimeoutError",
        "ServerError",
        "OverloadedError",
    }
)

_RATE_LIMIT_EXCEPTION_NAMES = frozenset(
    {
        "RateLimitError",
        "RateLimitExceeded",
        "TooManyRequests",
        "TooManyRequestsError",
        "ResourceExhausted",
    }
)


@dataclass
class RetryConfig:
    """Configuration for automatic retries on 500/5xx, 429 rate limits, and network failures."""

    enabled: bool = True
    retries: int = 3
    delays: Tuple[float, ...] = DEFAULT_RETRY_DELAYS
    extra_exceptions: Tuple[Type[BaseException], ...] = ()
    retry_on_rate_limit: bool = True
    jitter: float = 0.0
    max_retry_after: float = 60.0

    def get_delay_for_retry(self, retry_index: int) -> float:
        """
        Return the delay in seconds before retry `retry_index` (0-based: 0 -> 0.5s, 1 -> 1.0s, 2 -> 1.5s).
        If `retries` exceeds the length of `delays`, scales using the last delay step.
        """
        if not self.delays:
            return 0.0
        if retry_index < len(self.delays):
            return float(self.delays[retry_index])
        return float(self.delays[-1])

    def compute_wait(self, retry_index: int, exc: Optional[BaseException] = None) -> float:
        """
        Return the actual wait before retry `retry_index`: the configured delay, extended by a
        server `Retry-After` hint from `exc` (capped at `max_retry_after`), plus optional jitter.
        """
        delay = self.get_delay_for_retry(retry_index)
        if exc is not None:
            hint = extract_retry_after(exc)
            if hint is not None:
                delay = max(delay, min(hint, float(self.max_retry_after)))
        if self.jitter > 0 and delay > 0:
            delay += random.uniform(0.0, delay * float(self.jitter))
        return delay


@dataclass
class AttemptFailure:
    """Detailed record of a single failed attempt."""

    attempt: int  # 1-based attempt number
    error_type: str
    message: str
    status_code: Optional[int] = None
    delay_before: float = 0.0
    duration_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    exception: Optional[BaseException] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt,
            "error_type": self.error_type,
            "message": self.message,
            "status_code": self.status_code,
            "delay_before": self.delay_before,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp,
        }


class HTTPStatus5xxError(RuntimeError):
    """Raised internally when a function returns a response object with status_code >= 500."""

    def __init__(self, status_code: int, response: Any = None, message: Optional[str] = None) -> None:
        self.status_code = status_code
        self.response = response
        msg = message or f"HTTP {status_code} Server Error returned by response"
        super().__init__(msg)


class RetryExhaustedError(Exception):
    """
    Raised when the initial call and all retry attempts fail.
    Contains `attempts` (structured `AttemptFailure` list) and `errors` (all attempt exceptions).
    """

    def __init__(self, attempts: Sequence[AttemptFailure], func_name: str = "llm_call") -> None:
        self.func_name = func_name
        self.attempts: List[AttemptFailure] = list(attempts)
        self.errors: List[BaseException] = [
            a.exception for a in self.attempts if a.exception is not None
        ]
        summary_lines = [
            f"All {len(self.attempts)} attempts failed for '{func_name}':"
        ]
        for item in self.attempts:
            status_part = f" [HTTP {item.status_code}]" if item.status_code else ""
            wait_part = f" (after {item.delay_before}s wait)" if item.delay_before > 0 else " (initial attempt)"
            summary_lines.append(
                f"  - Attempt {item.attempt}{wait_part}{status_part}: {item.error_type}: {item.message}"
            )
        super().__init__("\n".join(summary_lines))


def extract_status_code(exc_or_obj: Any) -> Optional[int]:
    """
    Extract an HTTP status code from an exception or response object if present.
    Ignores control-flow exceptions like `SystemExit(503)` whose `.code` is an exit code.
    """
    if isinstance(exc_or_obj, _NON_RETRYABLE_BASE_EXCEPTIONS):
        return None

    for attr in ("status_code", "http_status", "status"):
        val = getattr(exc_or_obj, attr, None)
        if isinstance(val, int) and not isinstance(val, bool) and 100 <= val <= 599:
            return val

    # Only read `.code` on urllib HTTPError (avoid catching SystemExit.code or unrelated error codes)
    if isinstance(exc_or_obj, urllib.error.HTTPError):
        code_val = getattr(exc_or_obj, "code", None)
        if isinstance(code_val, int) and not isinstance(code_val, bool) and 100 <= code_val <= 599:
            return code_val

    response = getattr(exc_or_obj, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            val = getattr(response, attr, None)
            if isinstance(val, int) and not isinstance(val, bool) and 100 <= val <= 599:
                return val

    if isinstance(exc_or_obj, dict):
        for key in ("status_code", "http_status", "status"):
            val = exc_or_obj.get(key)
            if isinstance(val, int) and not isinstance(val, bool) and 100 <= val <= 599:
                return val
    return None


def _header_value(headers: Any, name: str) -> Optional[str]:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    try:
        val = getter(name)
        if val is None and isinstance(headers, dict):
            lowered = {str(k).lower(): v for k, v in headers.items()}
            val = lowered.get(name)
    except Exception:
        return None
    return None if val is None else str(val)


def extract_retry_after(exc: BaseException) -> Optional[float]:
    """
    Return the server-requested wait in seconds from `exc.retry_after`, or from the
    `retry-after-ms` / `retry-after` header of `exc.response` (seconds or HTTP-date).
    """
    attr = getattr(exc, "retry_after", None)
    if isinstance(attr, (int, float)) and not isinstance(attr, bool) and attr >= 0:
        return float(attr)

    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)

    ms_val = _header_value(headers, "retry-after-ms")
    if ms_val is not None:
        try:
            return max(0.0, float(ms_val) / 1000.0)
        except ValueError:
            pass

    raw = _header_value(headers, "retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, IndexError):
        return None


def is_retryable_exception(
    exc: BaseException,
    extra_exceptions: Tuple[Type[BaseException], ...] = (),
    retry_on_rate_limit: bool = True,
) -> Tuple[bool, Optional[int]]:
    """
    Determine whether `exc` represents an HTTP 500/5xx error, an HTTP 429 rate limit
    (when `retry_on_rate_limit` is True), or a network/timeout issue.

    Returns:
        Tuple of `(should_retry: bool, status_code: Optional[int])`.
    """
    # Never retry control-flow or cancellation exceptions
    if isinstance(exc, _NON_RETRYABLE_BASE_EXCEPTIONS):
        return False, None

    if extra_exceptions and isinstance(exc, extra_exceptions):
        return True, extract_status_code(exc)

    status_code = extract_status_code(exc)
    if status_code is not None:
        if 500 <= status_code <= 599:
            return True, status_code
        if status_code == 429:
            return retry_on_rate_limit, status_code
        # Explicit 4xx client errors (400, 401, 403, 404, 422) should not be retried
        if 400 <= status_code < 500:
            return False, status_code

    if any(base.__name__ in _RATE_LIMIT_EXCEPTION_NAMES for base in type(exc).__mro__):
        return retry_on_rate_limit, status_code

    # Standard Python network & timeout exceptions
    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            socket.timeout,
            socket.gaierror,
            urllib.error.URLError,
            asyncio.TimeoutError,
            HTTPStatus5xxError,
        ),
    ):
        return True, status_code

    # Check class name for SDK-specific network/timeout or 5xx exceptions (OpenAI, Anthropic, Gemini, httpx, requests)
    cls_name = type(exc).__name__
    if cls_name in _NETWORK_EXCEPTION_NAMES or cls_name in _SERVER_ERROR_EXCEPTION_NAMES:
        return True, status_code

    for base in type(exc).__mro__:
        if base.__name__ in _NETWORK_EXCEPTION_NAMES or base.__name__ in _SERVER_ERROR_EXCEPTION_NAMES:
            return True, status_code

    # Inspect ONLY explicit chained cause (`raise X from cause`), NEVER implicit `__context__`
    cause = exc.__cause__
    if cause is not None and cause is not exc:
        cause_retryable, cause_status = is_retryable_exception(
            cause, extra_exceptions, retry_on_rate_limit
        )
        if cause_retryable:
            return True, status_code or cause_status

    # Check if message explicitly indicates an HTTP 500/502/503/504 or network failure
    msg_lower = str(exc).lower()
    if any(
        token in msg_lower
        for token in (
            "500 internal server error",
            "502 bad gateway",
            "503 service unavailable",
            "504 gateway timeout",
            "status_code=500",
            "status_code: 500",
            "http 500",
            "connection reset",
            "connection refused",
            "connection timed out",
            "network is unreachable",
            "temporary failure in name resolution",
        )
    ):
        if status_code is None and "500" in msg_lower:
            status_code = 500
        return True, status_code

    return False, status_code


def _check_response_for_5xx(result: Any) -> None:
    """If the returned object itself carries a 5xx status_code, raise HTTPStatus5xxError to trigger retry."""
    status = extract_status_code(result)
    if status is not None and 500 <= status <= 599:
        raise HTTPStatus5xxError(status_code=status, response=result)


def execute_with_retry_sync(
    func: Callable[..., Any],
    *args: Any,
    config: Optional[RetryConfig] = None,
    sleep_func: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> Tuple[Any, List[AttemptFailure]]:
    """
    Execute a synchronous callable with retry logic on 5xx and network errors.

    Returns:
        Tuple of `(result, failed_attempts_before_success)`.

    Raises:
        `RetryExhaustedError` if the initial call and all configured retries fail.
        Non-retryable exceptions (e.g., `ValueError` or HTTP 400) are re-raised immediately
        with `_llm_tally_attempts` attached so prior retry history is preserved.
    """
    cfg = config or RetryConfig()

    # If retries are disabled, call `func` directly without altering 5xx response objects
    if not cfg.enabled:
        return func(*args, **kwargs), []

    max_retries = max(0, cfg.retries)
    total_attempts = 1 + max_retries
    func_name = getattr(func, "__qualname__", getattr(func, "__name__", "llm_call"))
    failures: List[AttemptFailure] = []
    next_delay = 0.0

    for attempt_idx in range(total_attempts):
        attempt_num = attempt_idx + 1
        delay_before = 0.0 if attempt_idx == 0 else next_delay
        if delay_before > 0:
            sleep_func(delay_before)

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            _check_response_for_5xx(result)
            return result, failures
        except BaseException as exc:
            if isinstance(exc, _NON_RETRYABLE_BASE_EXCEPTIONS):
                raise
            duration_ms = (time.perf_counter() - start) * 1000.0
            should_retry, status_code = is_retryable_exception(
                exc, cfg.extra_exceptions, cfg.retry_on_rate_limit
            )
            failure = AttemptFailure(
                attempt=attempt_num,
                error_type=type(exc).__name__,
                message=str(exc),
                status_code=status_code,
                delay_before=delay_before,
                duration_ms=duration_ms,
                exception=exc,
            )
            failures.append(failure)

            if not should_retry:
                try:
                    setattr(exc, "_llm_tally_attempts", list(failures))
                except Exception:
                    pass
                raise
            if attempt_num >= total_attempts:
                raise RetryExhaustedError(attempts=failures, func_name=func_name) from exc
            next_delay = cfg.compute_wait(attempt_idx, exc)

    raise RetryExhaustedError(attempts=failures, func_name=func_name)


async def execute_with_retry_async(
    func: Callable[..., Any],
    *args: Any,
    config: Optional[RetryConfig] = None,
    sleep_func: Callable[[float], Any] = asyncio.sleep,
    **kwargs: Any,
) -> Tuple[Any, List[AttemptFailure]]:
    """
    Execute an asynchronous callable with retry logic on 5xx and network errors.

    Returns:
        Tuple of `(result, failed_attempts_before_success)`.

    Raises:
        `RetryExhaustedError` if the initial call and all configured retries fail.
    """
    cfg = config or RetryConfig()

    # If retries are disabled, call `func` directly without altering 5xx response objects
    if not cfg.enabled:
        return await func(*args, **kwargs), []

    max_retries = max(0, cfg.retries)
    total_attempts = 1 + max_retries
    func_name = getattr(func, "__qualname__", getattr(func, "__name__", "llm_call"))
    failures: List[AttemptFailure] = []
    next_delay = 0.0

    for attempt_idx in range(total_attempts):
        attempt_num = attempt_idx + 1
        delay_before = 0.0 if attempt_idx == 0 else next_delay
        if delay_before > 0:
            await sleep_func(delay_before)

        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            _check_response_for_5xx(result)
            return result, failures
        except BaseException as exc:
            if isinstance(exc, _NON_RETRYABLE_BASE_EXCEPTIONS):
                raise
            duration_ms = (time.perf_counter() - start) * 1000.0
            should_retry, status_code = is_retryable_exception(
                exc, cfg.extra_exceptions, cfg.retry_on_rate_limit
            )
            failure = AttemptFailure(
                attempt=attempt_num,
                error_type=type(exc).__name__,
                message=str(exc),
                status_code=status_code,
                delay_before=delay_before,
                duration_ms=duration_ms,
                exception=exc,
            )
            failures.append(failure)

            if not should_retry:
                try:
                    setattr(exc, "_llm_tally_attempts", list(failures))
                except Exception:
                    pass
                raise
            if attempt_num >= total_attempts:
                raise RetryExhaustedError(attempts=failures, func_name=func_name) from exc
            next_delay = cfg.compute_wait(attempt_idx, exc)

    raise RetryExhaustedError(attempts=failures, func_name=func_name)
