"""Retry logic with exponential backoff.

Handles rate limits (429), transient server errors (5xx) and timeouts.
Uses a fake "flaky" function so the retry/backoff behavior is fully testable offline.
"""

import random
import time
from functools import wraps


class RateLimitError(Exception):
    pass


class TransientServerError(Exception):
    pass


def retry_with_backoff(max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 30.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (RateLimitError, TransientServerError) as exc:
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay += random.uniform(0, delay * 0.1)  # jitter
                    print(f"[retry] attempt {attempt + 1} failed ({exc}); sleeping {delay:.2f}s")
                    time.sleep(delay)
        return wrapper
    return decorator


# a fake LLM call that fails twice before succeeding
_attempt_counter = {"count": 0}

@retry_with_backoff(max_retries=4, base_delay=0.2)
def flaky_llm_call(prompt: str) -> str:
    _attempt_counter["count"] += 1
    if _attempt_counter["count"] < 3:
        raise RateLimitError("429 Too Many Requests")
    return f"Response to: {prompt}"


if __name__ == "__main__":
    print(flaky_llm_call("Hello, model"))
