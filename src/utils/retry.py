from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")

RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
    "quota",
    "resource exhausted",
)


def is_rate_limit_error(error: Exception) -> bool:
    error_text = f"{error.__class__.__name__}: {error}".lower()
    return any(marker in error_text for marker in RATE_LIMIT_MARKERS)


def _compute_backoff_seconds(
    attempt: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> float:
    exponential = min(max_delay_seconds, base_delay_seconds * (2 ** (attempt - 1)))
    jitter = random.uniform(0, min(1.0, exponential / 4 if exponential > 0 else 0.1))
    return exponential + jitter


def run_with_retry(
    func: Callable[..., T],
    *args,
    max_retries: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    **kwargs,
) -> T:
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            is_last_attempt = attempt >= max_retries
            if not is_rate_limit_error(error) or is_last_attempt:
                raise

            sleep_seconds = _compute_backoff_seconds(
                attempt=attempt,
                base_delay_seconds=base_delay_seconds,
                max_delay_seconds=max_delay_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError("Falha inesperada no mecanismo de retry")


async def run_with_retry_async(
    func: Callable[..., Awaitable[T]],
    *args,
    max_retries: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    **kwargs,
) -> T:
    for attempt in range(1, max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as error:
            is_last_attempt = attempt >= max_retries
            if not is_rate_limit_error(error) or is_last_attempt:
                raise

            sleep_seconds = _compute_backoff_seconds(
                attempt=attempt,
                base_delay_seconds=base_delay_seconds,
                max_delay_seconds=max_delay_seconds,
            )
            await asyncio.sleep(sleep_seconds)

    raise RuntimeError("Falha inesperada no mecanismo de retry async")
