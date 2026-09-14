"""
In-memory thread-safe TTL cache to eliminate redundant Supabase network roundtrips.
"""
import time
from typing import Any, Optional, Dict, Tuple
import threading
import logging

logger = logging.getLogger(__name__)

_cache: Dict[str, Tuple[float, Any]] = {}
_lock = threading.Lock()

def get_cache(key: str) -> Optional[Any]:
    with _lock:
        item = _cache.get(key)
        if item is None:
            return None
        expires_at, value = item
        if time.time() > expires_at:
            del _cache[key]
            return None
        return value

def set_cache(key: str, value: Any, ttl_seconds: float = 30.0) -> None:
    with _lock:
        _cache[key] = (time.time() + ttl_seconds, value)

def invalidate_cache(key: str) -> None:
    with _lock:
        if key in _cache:
            del _cache[key]

def invalidate_user_cache(user_id: str) -> None:
    with _lock:
        keys_to_del = [k for k in _cache if user_id in k]
        for k in keys_to_del:
            del _cache[k]

def clear_cache() -> None:
    with _lock:
        _cache.clear()
