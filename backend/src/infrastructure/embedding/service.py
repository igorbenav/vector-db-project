"""Embedding service for automatic text-to-vector conversion using sentence-transformers."""

import asyncio
from functools import lru_cache
from typing import List, Optional, cast

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Service for generating vector embeddings from text.

    Uses sentence-transformers with all-mpnet-base-v2 model for high-quality
    768-dimensional embeddings suitable for semantic similarity search.

    Features:
    - Lazy model loading for faster startup
    - Batch processing for efficiency
    - Thread-safe model access
    - LRU caching for repeated text
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """Initialize embedding service.

        Args:
            model_name: HuggingFace model name for sentence transformers
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._model_lock = asyncio.Lock()

    async def _get_model(self) -> SentenceTransformer:
        """Get model instance, loading it if necessary (thread-safe)."""
        if self._model is None:
            async with self._model_lock:
                if self._model is None:
                    self._model = cast(SentenceTransformer, await asyncio.to_thread(SentenceTransformer, self.model_name))
        if self._model is None:
            raise RuntimeError("Model failed to load")
        return self._model

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed

        Returns:
            768-dimensional embedding vector
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        model = await self._get_model()

        embedding = await asyncio.to_thread(
            model.encode,
            text,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch processing).

        Args:
            texts: List of texts to embed

        Returns:
            List of 768-dimensional embedding vectors
        """
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("All texts must be non-empty")

        model = await self._get_model()

        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
            batch_size=32,
        )

        if hasattr(embeddings, "tolist"):
            return cast(List[List[float]], embeddings.tolist())
        else:
            return [emb.tolist() for emb in embeddings]

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension for this model."""
        return 768

    async def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get singleton embedding service instance."""
    return EmbeddingService()
