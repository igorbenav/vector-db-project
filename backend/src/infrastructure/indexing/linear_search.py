"""Linear search vector index implementation."""

import math
from typing import Any, Dict, List, Optional, cast

from .base import ChunkVector, IndexStats, IndexType, SearchResult, VectorIndex


class LinearSearchIndex(VectorIndex):
    """Linear search vector index using brute-force cosine similarity.

    This is the simplest vector indexing algorithm that compares the query
    vector against every vector in the index. While it's the slowest for large
    datasets, it guarantees exact results and has minimal memory overhead.

    Characteristics:
    - Time Complexity (Search): O(n * d) where n = vectors, d = dimension
    - Space Complexity: O(n * d) for storing vectors
    - Accuracy: 100% (exact results)
    - Build Time: O(1) (no build phase needed)

    Best for:
    - Small datasets (< 10,000 vectors)
    - Applications requiring exact results
    - Testing and benchmarking other algorithms
    """

    @property
    def index_type(self) -> IndexType:
        """Return the type of this index."""
        return IndexType.LINEAR_SEARCH

    async def add_vector(self, vector: ChunkVector) -> None:
        """Add a vector to the index.

        Args:
            vector: The vector to add to the index
        """
        self._validate_embedding(vector.embedding)
        self.vectors.append(vector)
        self._built = True

    async def add_vectors(self, vectors: List[ChunkVector]) -> None:
        """Add multiple vectors to the index efficiently.

        Args:
            vectors: List of vectors to add to the index
        """
        for vector in vectors:
            self._validate_embedding(vector.embedding)

        self.vectors.extend(vectors)
        self._built = True

    async def remove_vector(self, chunk_id: int) -> bool:
        """Remove a vector from the index.

        Args:
            chunk_id: ID of the chunk to remove

        Returns:
            True if the vector was found and removed, False otherwise
        """
        for i, vector in enumerate(self.vectors):
            if vector.chunk_id == chunk_id:
                del self.vectors[i]
                return True
        return False

    async def search(
        self, query_embedding: List[float], k: int, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for the k most similar vectors using linear search.

        Args:
            query_embedding: The query vector
            k: Number of nearest neighbors to return
            metadata_filter: Optional metadata filter criteria

        Returns:
            List of search results sorted by similarity (descending)
        """
        self._validate_embedding(query_embedding)

        if not self.vectors:
            return []

        similarities = []

        for vector in self.vectors:
            if metadata_filter and not self._matches_metadata_filter(vector.metadata, metadata_filter):
                continue

            similarity = self._cosine_similarity(query_embedding, vector.embedding)

            similarities.append({"vector": vector, "similarity_score": similarity})

        similarities.sort(key=lambda x: cast(float, x["similarity_score"]), reverse=True)
        top_k = similarities[:k]

        results = []
        for item in top_k:
            vector = cast(ChunkVector, item["vector"])
            score = cast(float, item["similarity_score"])
            results.append(
                SearchResult(
                    chunk_id=vector.chunk_id,
                    document_id=vector.document_id,
                    content=vector.content,
                    similarity_score=score,
                    metadata=vector.metadata,
                )
            )

        return results

    async def build_index(self) -> None:
        """Build/rebuild the index from the current vectors.

        For linear search, no building is required - just mark as built.
        """
        self._built = True

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Formula: cos(θ) = (A · B) / (||A|| ||B||)

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0

        similarity = dot_product / (magnitude1 * magnitude2)

        return max(0.0, similarity)

    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        return IndexStats(
            total_vectors=len(self.vectors),
            embedding_dimension=self.dimension,
            algorithm_params={"algorithm": "brute_force", "exact_search": True, "build_required": False},
        )
