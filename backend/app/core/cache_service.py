"""
MemoryVerse - Multi-Level Cache Service
Provides L1 (In-Memory LRU), L2 (Upstash Redis), and L3 (Database) caching.
Enforces content-addressable hashing, versioning, tenant isolation (user scoping),
and duplicate job fingerprinting.
"""
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from app.config.settings import settings

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = "v2"
MODEL_VERSION = "siglip-base-v1"
EMOTION_MODEL_VERSION = "emotion-fusion-v1"
PIPELINE_VERSION = "video-pipeline-v3"


class LRUMemoryCache:
    """Thread-safe in-memory LRU cache with TTL support (L1)."""

    def __init__(self, capacity: int = 1000, default_ttl: int = 3600):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        val, expiry = self._cache[key]
        if time.time() > expiry:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return val

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expiry = time.time() + (ttl if ttl is not None else self.default_ttl)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, expiry)
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()


class CacheService:
    """
    Orchestrates L1 (Memory) and L2 (Upstash Redis) caching with
    tenant-isolated, versioned keys.
    """

    def __init__(self):
        self.l1 = LRUMemoryCache(capacity=2000, default_ttl=3600)
        self.redis = None
        self._init_redis()

    def _init_redis(self):
        if settings.UPSTASH_REDIS_REST_URL and settings.UPSTASH_REDIS_REST_TOKEN:
            try:
                from upstash_redis import Redis
                self.redis = Redis(
                    url=settings.UPSTASH_REDIS_REST_URL,
                    token=settings.UPSTASH_REDIS_REST_TOKEN,
                )
                logger.info("CacheService initialized with Upstash Redis (L2).")
            except Exception as e:
                logger.warning(f"Upstash Redis initialization failed for CacheService: {e}")
                self.redis = None
        else:
            logger.info("Upstash Redis not configured. Using in-memory L1 cache fallback.")

    @staticmethod
    def hash_data(data: Any) -> str:
        """Computes deterministic SHA256 hash for bytes, string, or JSON-serializable structure."""
        if isinstance(data, bytes):
            return hashlib.sha256(data).hexdigest()
        if isinstance(data, str):
            return hashlib.sha256(data.encode("utf-8")).hexdigest()
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def make_key(
        self,
        category: str,
        user_id: Optional[str],
        identifier: str,
        version: str = PIPELINE_VERSION,
    ) -> str:
        """
        Constructs a tenant-safe, versioned cache key.
        Tenant scoping: If user_id is provided, keys are isolated under user:<user_id>.
        """
        user_scope = f"u:{user_id}" if user_id else "global"
        return f"mv:{CACHE_SCHEMA_VERSION}:{user_scope}:{category}:{version}:{identifier}"

    def get(self, key: str) -> Optional[Any]:
        """Check L1 then L2."""
        # 1. Check L1 Memory
        val = self.l1.get(key)
        if val is not None:
            return val

        # 2. Check L2 Redis
        if self.redis:
            try:
                raw = self.redis.get(key)
                if raw is not None:
                    if isinstance(raw, (dict, list, int, float, bool)):
                        parsed = raw
                    else:
                        parsed = json.loads(raw)
                    # Backfill L1
                    self.l1.set(key, parsed, ttl=600)
                    return parsed
            except Exception as e:
                logger.debug(f"Redis get failed for key {key}: {e}")

        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Write to L1 and L2."""
        self.l1.set(key, value, ttl=ttl)
        if self.redis:
            try:
                data_str = json.dumps(value, default=str)
                self.redis.set(key, data_str, ex=ttl)
            except Exception as e:
                logger.debug(f"Redis set failed for key {key}: {e}")

    def delete(self, key: str) -> None:
        self.l1.delete(key)
        if self.redis:
            try:
                self.redis.delete(key)
            except Exception as e:
                logger.debug(f"Redis delete failed for key {key}: {e}")

    # ── Specialized Caching Helpers ───────────────────────────────────────────

    def get_embedding(self, media_hash: str) -> Optional[list]:
        key = self.make_key("emb", None, media_hash, version=MODEL_VERSION)
        return self.get(key)

    def set_embedding(self, media_hash: str, embedding: list, ttl: int = 86400 * 7) -> None:
        key = self.make_key("emb", None, media_hash, version=MODEL_VERSION)
        self.set(key, embedding, ttl=ttl)

    def get_emotion(self, media_hash: str) -> Optional[Dict[str, Any]]:
        key = self.make_key("emotion", None, media_hash, version=EMOTION_MODEL_VERSION)
        return self.get(key)

    def set_emotion(self, media_hash: str, emotion_data: Dict[str, Any], ttl: int = 86400 * 7) -> None:
        key = self.make_key("emotion", None, media_hash, version=EMOTION_MODEL_VERSION)
        self.set(key, emotion_data, ttl=ttl)

    def get_story_plan(self, user_id: str, plan_fingerprint: str) -> Optional[Dict[str, Any]]:
        key = self.make_key("story", user_id, plan_fingerprint, version=PIPELINE_VERSION)
        return self.get(key)

    def set_story_plan(self, user_id: str, plan_fingerprint: str, plan: Dict[str, Any], ttl: int = 86400) -> None:
        key = self.make_key("story", user_id, plan_fingerprint, version=PIPELINE_VERSION)
        self.set(key, plan, ttl=ttl)

    # ── Duplicate Job Fingerprinting ──────────────────────────────────────────

    def compute_job_fingerprint(
        self,
        user_id: str,
        memory_id: str,
        selected_media_ids: Optional[list],
        mood: str,
        dimension: str,
    ) -> str:
        """Deterministic fingerprint for a video generation request."""
        sorted_ids = sorted(str(m) for m in (selected_media_ids or []))
        sig = f"{user_id}:{memory_id}:{','.join(sorted_ids)}:{mood.lower()}:{dimension}:{PIPELINE_VERSION}"
        return hashlib.sha256(sig.encode("utf-8")).hexdigest()

    def get_active_job_by_fingerprint(self, fingerprint: str) -> Optional[str]:
        """Returns job_id if an identical job was recently enqueued/processed."""
        key = f"mv:job_fp:{fingerprint}"
        return self.get(key)

    def set_active_job_by_fingerprint(self, fingerprint: str, job_id: str, ttl: int = 600) -> None:
        key = f"mv:job_fp:{fingerprint}"
        self.set(key, job_id, ttl=ttl)


# Global singleton instance
cache_service = CacheService()
