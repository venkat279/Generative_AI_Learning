"""Session and state management: an in-memory stand-in for what a production 
deployment would put in Redis, keyed by (tenant_id, conversation_id) so 
different tenants never share history (multi-tenant session isolation), 
with idle sessions swept out after a TTL (session expiry and cleanup) 
- the same expiry pattern Module 11's long-term memory uses, but 
applied to HTTP session state instead of remembered facts.
"""

import time

SESSION_TTL_SECONDS = 60 * 30


class SessionStore:
    def __init__(self, ttl_seconds: float = SESSION_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[tuple[str, str], dict] = {}

    def get_history(self, tenant_id: str, conversation_id: str) -> list[dict]:
        self._cleanup_expired()
        session = self._sessions.get((tenant_id, conversation_id))
        return list(session["messages"]) if session else []

    def append_turn(self, tenant_id: str, conversation_id: str, user_message: str, assistant_message: str) -> None:
        key = (tenant_id, conversation_id)
        session = self._sessions.setdefault(key, {"messages": []})
        session["messages"].append({"role": "user", "content": user_message})
        session["messages"].append({"role": "assistant", "content": assistant_message})
        session["last_used"] = time.time()

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [key for key, s in self._sessions.items() if now - s.get("last_used", now) > self.ttl_seconds]
        for key in expired:
            del self._sessions[key]

    def active_session_count(self) -> int:
        self._cleanup_expired()
        return len(self._sessions)


class RateLimiter:
    """Fixed-window rate limiting: at most `max_requests` calls per
    caller (keyed by API key) within a rolling `window_seconds`."""

    def __init__(self, max_requests: int = 5, window_seconds: float = 10):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        hits = [t for t in self._hits.get(key, []) if now - t < self.window_seconds]
        hits.append(now)
        self._hits[key] = hits
        return len(hits) <= self.max_requests
