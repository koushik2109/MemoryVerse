"""
pgvector Database Adapter
Executes multi-vector similarity search queries via Supabase / PostgreSQL pgvector RPCs.
Supports 512-dim (CLIP), 768-dim (SigLIP), 1024-dim (BGE-M3), and 1536-dim (OpenAI).
"""
import os
import logging
from typing import List, Dict, Any, Optional, cast

logger = logging.getLogger(__name__)


class PgVectorAdapter:
    """
    Client adapter for executing vector similarity searches against Supabase pgvector.
    """

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        self.url = supabase_url or os.getenv("SUPABASE_URL", "")
        self.key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None and self.url and self.key:
            try:
                from supabase import create_client
                self._client = create_client(self.url, self.key)
            except Exception as e:
                logger.debug(f"Supabase client initialization deferred: {e}")
        return self._client

    def match_by_clip(self, query_vector: List[float], match_count: int = 10) -> List[Dict[str, Any]]:
        """Invokes RPC match_media_by_clip (512-dim)."""
        client = self._get_client()
        if not client:
            return []
        try:
            res = client.rpc("match_media_by_clip", {"query_embedding": query_vector, "match_count": match_count}).execute()
            return cast(List[Dict[str, Any]], res.data or [])
        except Exception as e:
            logger.debug(f"RPC match_media_by_clip error: {e}")
            return []

    def match_by_siglip(self, query_vector: List[float], match_count: int = 10) -> List[Dict[str, Any]]:
        """Invokes RPC match_media_by_siglip (768-dim)."""
        client = self._get_client()
        if not client:
            return []
        try:
            res = client.rpc("match_media_by_siglip", {"query_embedding": query_vector, "match_count": match_count}).execute()
            return cast(List[Dict[str, Any]], res.data or [])
        except Exception as e:
            logger.debug(f"RPC match_media_by_siglip error: {e}")
            return []

    def match_by_bge(self, query_vector: List[float], match_count: int = 10) -> List[Dict[str, Any]]:
        """Invokes RPC match_media_by_bge (1024-dim)."""
        client = self._get_client()
        if not client:
            return []
        try:
            res = client.rpc("match_media_by_bge", {"query_embedding": query_vector, "match_count": match_count}).execute()
            return cast(List[Dict[str, Any]], res.data or [])
        except Exception as e:
            logger.debug(f"RPC match_media_by_bge error: {e}")
            return []
