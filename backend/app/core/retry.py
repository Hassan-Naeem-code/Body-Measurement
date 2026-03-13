"""
Retry decorator with exponential backoff for transient failures.

Use for: model loading, network calls (torch.hub), database connections.
Do NOT retry: input validation errors (HTTPException, ValueError).
"""

import functools
import logging
import time
from typing import Tuple, Type

logger = logging.getLogger(__name__)

# Exceptions that indicate transient failures worth retrying
RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    RuntimeError,
    ConnectionError,
    TimeoutError,
    OSError,
    IOError,
)


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = RETRYABLE_EXCEPTIONS,
):
    """
    Decorator that retries a function on transient failures with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds (doubles each retry).
        max_delay: Maximum delay cap in seconds.
        retryable_exceptions: Tuple of exception types to retry on.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            "Retry %d/%d for %s after error: %s (waiting %.1fs)",
                            attempt + 1,
                            max_retries,
                            func.__qualname__,
                            str(e),
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d retries exhausted for %s: %s",
                            max_retries,
                            func.__qualname__,
                            str(e),
                        )
            raise last_exception

        return wrapper

    return decorator
