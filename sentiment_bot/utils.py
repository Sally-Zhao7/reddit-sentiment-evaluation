"""Small helpers shared by the evaluator and the Reddit bot: basic logging
setup and a retry-with-backoff function for talking to the Reddit API.

Kept deliberately small - this is not a resilience framework, just enough
to fix the specific problem the original 2024 bot had: a fixed 60-second
sleep on every error, even when Reddit explicitly asked for a 9-minute
break (see the captured run log in legacy/original_2024_bot.py).
"""
from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Plain stdlib logging - one readable line per event, no JSON/structured layer."""
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger("sentiment_bot")


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------

_WAIT_PATTERN = re.compile(
    r"(?:take a break for|try again in)\s+(?P<amount>\d+)\s+(?P<unit>second|minute|hour)s?",
    re.IGNORECASE,
)


def extract_wait_seconds(message: str) -> Optional[float]:
    """Parse a wait time out of a Reddit rate-limit message, if it names one.

    >>> extract_wait_seconds('Take a break for 9 minutes before trying again.')
    540.0
    >>> extract_wait_seconds("some unrelated error")
    """
    if not message:
        return None
    match = _WAIT_PATTERN.search(message)
    if not match:
        return None
    amount = float(match.group("amount"))
    unit = match.group("unit").lower()
    multiplier = {"second": 1.0, "minute": 60.0, "hour": 3600.0}[unit]
    return amount * multiplier


def is_retryable_error(exc: BaseException) -> bool:
    """Transient errors worth retrying: rate limits, server errors, network hiccups.

    Matched on the exception's type name and message (not
    `isinstance(exc, praw.exceptions.APIException)`) so this module - and
    its tests - don't need praw installed.
    """
    text = f"{type(exc).__module__}.{type(exc).__name__} {exc}".lower()
    markers = ("ratelimit", "timeout", "connectionerror", "requestexception", "server error", "503", "502", "500")
    return any(marker in text for marker in markers)


@dataclass
class RetryConfig:
    max_retries: int = 5
    base_delay_seconds: float = 5.0
    max_delay_seconds: float = 600.0


def _backoff_delay(attempt: int, config: RetryConfig, rng: random.Random) -> float:
    """Exponential backoff (base * 2**attempt, capped) with +/-20% jitter."""
    delay = min(config.base_delay_seconds * (2 ** attempt), config.max_delay_seconds)
    jitter = delay * 0.2
    return max(0.0, delay + rng.uniform(-jitter, jitter))


def retry_call(
    fn: Callable[[], T],
    *,
    config: Optional[RetryConfig] = None,
    is_retryable: Callable[[BaseException], bool] = is_retryable_error,
    sleep: Callable[[float], None] = time.sleep,
    rng: Optional[random.Random] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> T:
    """Call `fn()`, retrying transient errors with backoff.

    If the error message names an explicit wait (Reddit's rate-limit errors
    do - "take a break for N minutes"), that wait is used instead of the
    exponential backoff. Re-raises the last exception once retries are
    exhausted, or immediately if `is_retryable` returns False.
    """
    config = config or RetryConfig()
    rng = rng or random.Random()

    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - intentionally broad, re-raised below
            if not is_retryable(exc) or attempt >= config.max_retries:
                raise
            wait_seconds = extract_wait_seconds(str(exc))
            if wait_seconds is None:
                wait_seconds = _backoff_delay(attempt, config, rng)
            wait_seconds = min(wait_seconds, config.max_delay_seconds)
            if on_retry:
                on_retry(attempt, exc, wait_seconds)
            sleep(wait_seconds)
            attempt += 1
