"""Dependency injection: shared FastAPI `Depends()` functions - API-key
auth and rate limiting - applied to every protected route via a
`dependencies=[...]` list instead of repeating the same checks inside
each handler.
"""

from fastapi import Depends, Header, HTTPException, status

from sessions import RateLimiter, SessionStore

API_KEY = "demo-key"  # hardcoded key

session_store = SessionStore()
rate_limiter = RateLimiter(max_requests=5, window_seconds=10)


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key")
    return x_api_key


def enforce_rate_limit(api_key: str = Depends(require_api_key)) -> None:
    if not rate_limiter.allow(api_key):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
