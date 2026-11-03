from __future__ import annotations

import asyncio
import unittest
from typing import List

from llm_tally import (
    DEFAULT_RETRY_DELAYS,
    RetryConfig,
    RetryExhaustedError,
    execute_with_retry_async,
    execute_with_retry_sync,
)


class DummyServer500Error(Exception):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class DummyClient400Error(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_default_retry_delays_are_0_5_1_0_1_5() -> None:
    assert DEFAULT_RETRY_DELAYS == (0.5, 1.0, 1.5)


def test_retry_on_500_and_network_errors_collects_all_errors_when_all_fail() -> None:
    recorded_sleeps: List[float] = []
    attempt_counter = 0

    def failing_call() -> str:
        nonlocal attempt_counter
        attempt_counter += 1
        if attempt_counter == 1:
            raise DummyServer500Error("Internal Server Error on attempt 1", status_code=500)
        if attempt_counter == 2:
            raise ConnectionError("Connection reset by peer on attempt 2")
        if attempt_counter == 3:
            raise TimeoutError("Read timed out on attempt 3")
        raise DummyServer500Error("Service Unavailable on attempt 4", status_code=503)

    raised_err: RetryExhaustedError | None = None
    try:
        execute_with_retry_sync(
            failing_call,
            config=RetryConfig(retries=3, delays=(0.5, 1.0, 1.5)),
            sleep_func=recorded_sleeps.append,
        )
    except RetryExhaustedError as exc:
        raised_err = exc

    assert raised_err is not None
    assert attempt_counter == 4
    assert recorded_sleeps == [0.5, 1.0, 1.5]
    assert len(raised_err.attempts) == 4
    assert len(raised_err.errors) == 4

    msg = str(raised_err)
    assert "Attempt 1 (initial attempt) [HTTP 500]: DummyServer500Error: Internal Server Error on attempt 1" in msg
    assert "Attempt 2 (after 0.5s wait): ConnectionError: Connection reset by peer on attempt 2" in msg
    assert "Attempt 3 (after 1.0s wait): TimeoutError: Read timed out on attempt 3" in msg
    assert "Attempt 4 (after 1.5s wait) [HTTP 503]: DummyServer500Error: Service Unavailable on attempt 4" in msg


def test_retry_succeeds_on_third_attempt() -> None:
    recorded_sleeps: List[float] = []
    attempts = 0

    def eventually_succeeds() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise DummyServer500Error(f"500 failure #{attempts}", status_code=500)
        return "recovered-ok"

    result, failures = execute_with_retry_sync(
        eventually_succeeds,
        config=RetryConfig(retries=3, delays=(0.5, 1.0, 1.5)),
        sleep_func=recorded_sleeps.append,
    )
    assert result == "recovered-ok"
    assert attempts == 3
    assert recorded_sleeps == [0.5, 1.0]
    assert len(failures) == 2


def test_retry_triggers_when_response_object_has_status_code_500() -> None:
    recorded_sleeps: List[float] = []
    attempts = 0

    def returns_500_response() -> dict:
        nonlocal attempts
        attempts += 1
        return {"status_code": 500, "body": f"fail #{attempts}"}

    raised_err: RetryExhaustedError | None = None
    try:
        execute_with_retry_sync(
            returns_500_response,
            config=RetryConfig(retries=3, delays=(0.5, 1.0, 1.5)),
            sleep_func=recorded_sleeps.append,
        )
    except RetryExhaustedError as exc:
        raised_err = exc

    assert raised_err is not None
    assert attempts == 4
    assert recorded_sleeps == [0.5, 1.0, 1.5]
    assert all(a.status_code == 500 for a in raised_err.attempts)


def test_non_retryable_400_error_fails_immediately_without_retries() -> None:
    recorded_sleeps: List[float] = []
    attempts = 0

    def bad_request() -> str:
        nonlocal attempts
        attempts += 1
        raise DummyClient400Error("Invalid API key / Bad Request", status_code=400)

    caught = False
    try:
        execute_with_retry_sync(
            bad_request,
            config=RetryConfig(retries=3, delays=(0.5, 1.0, 1.5)),
            sleep_func=recorded_sleeps.append,
        )
    except DummyClient400Error:
        caught = True

    assert caught
    assert attempts == 1
    assert recorded_sleeps == []


def test_async_retry_engine() -> None:
    recorded_sleeps: List[float] = []
    attempts = 0

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)

    async def flaky_async() -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise ConnectionError(f"Network blip {attempts}")
        return "async-ok"

    async def _run() -> None:
        res, failures = await execute_with_retry_async(
            flaky_async,
            config=RetryConfig(retries=3, delays=(0.5, 1.0, 1.5)),
            sleep_func=fake_sleep,
        )
        assert res == "async-ok"
        assert len(failures) == 2
        assert recorded_sleeps == [0.5, 1.0]

    asyncio.run(_run())


class TestRetrySuite(unittest.TestCase):
    def test_defaults(self) -> None:
        test_default_retry_delays_are_0_5_1_0_1_5()

    def test_all_attempts_fail(self) -> None:
        test_retry_on_500_and_network_errors_collects_all_errors_when_all_fail()

    def test_succeeds_after_retry(self) -> None:
        test_retry_succeeds_on_third_attempt()

    def test_response_500_status(self) -> None:
        test_retry_triggers_when_response_object_has_status_code_500()

    def test_non_retryable_400(self) -> None:
        test_non_retryable_400_error_fails_immediately_without_retries()

    def test_async_retries(self) -> None:
        test_async_retry_engine()
