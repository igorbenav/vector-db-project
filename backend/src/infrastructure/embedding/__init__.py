"""Embedding infrastructure for text-to-vector conversion."""

from .service import EmbeddingService, get_embedding_service

__all__ = ["EmbeddingService", "get_embedding_service"]
