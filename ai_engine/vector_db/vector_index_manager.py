"""
Vector Index Manager
Generates and verifies IVFFlat and HNSW cosine similarity index definitions
for multi-dimensional vector columns in PostgreSQL / pgvector.
"""
from typing import Dict, Any, List


class VectorIndexManager:
    """
    Manages vector index definitions and tuning parameters for pgvector.
    """

    DIMENSIONS = {
        "clip": 512,
        "siglip": 768,
        "bge": 1024,
        "openai": 1536,
    }

    @staticmethod
    def get_ivfflat_index_sql(table_name: str, column_name: str, index_name: str, lists: int = 100) -> str:
        """Returns SQL statement to create an IVFFlat cosine similarity index."""
        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table_name} USING ivfflat ({column_name} vector_cosine_ops) "
            f"WITH (lists = {lists});"
        )

    @staticmethod
    def get_hnsw_index_sql(table_name: str, column_name: str, index_name: str, m: int = 16, ef_construction: int = 64) -> str:
        """Returns SQL statement to create an HNSW cosine similarity index."""
        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table_name} USING hnsw ({column_name} vector_cosine_ops) "
            f"WITH (m = {m}, ef_construction = {ef_construction});"
        )

    @staticmethod
    def get_all_recommended_indexes() -> List[str]:
        """Returns all recommended indexes for the MemoryVerse multi-vector schema."""
        return [
            VectorIndexManager.get_ivfflat_index_sql("media_embeddings", "clip_embedding", "idx_media_clip_ivfflat"),
            VectorIndexManager.get_ivfflat_index_sql("media_embeddings", "image_embedding", "idx_media_siglip_ivfflat"),
            VectorIndexManager.get_ivfflat_index_sql("media_embeddings", "text_embedding", "idx_media_bge_ivfflat"),
            VectorIndexManager.get_ivfflat_index_sql("media_embeddings", "embedding", "idx_media_embedding_ivfflat"),
        ]
