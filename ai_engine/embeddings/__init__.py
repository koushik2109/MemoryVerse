"""
Multimodal Embeddings Package
Exports BaseEmbedder, ClipEmbedder (512-dim), SiglipEmbedder (768-dim), and BgeEmbedder (1024-dim).
"""
from ai_engine.embeddings.base_embedder import BaseEmbedder
from ai_engine.embeddings.clip_embedder import ClipEmbedder
from ai_engine.embeddings.siglip_embedder import SiglipEmbedder
from ai_engine.embeddings.bge_embedder import BgeEmbedder

__all__ = ["BaseEmbedder", "ClipEmbedder", "SiglipEmbedder", "BgeEmbedder"]
