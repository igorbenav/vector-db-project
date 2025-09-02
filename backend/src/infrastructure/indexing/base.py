"""Abstract base classes for vector indexing algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class IndexType(str, Enum):
    """Supported index types."""

    LINEAR_SEARCH = "linear_search"
    LSH = "lsh"
    IVF = "ivf"
    HNSW = "hnsw"


@dataclass
class ChunkVector:
    """Represents a chunk with its vector embedding."""

    chunk_id: int
    document_id: int
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Represents a search result with similarity score."""

    chunk_id: int
    document_id: int
    content: str
    similarity_score: float
    metadata: Dict[str, Any]


@dataclass
class IndexStats:
    """Statistics about an index."""

    total_vectors: int
    embedding_dimension: int
    index_size_bytes: Optional[int] = None
    build_time_ms: Optional[float] = None
    last_updated: Optional[str] = None
    algorithm_params: Optional[Dict[str, Any]] = None


class VectorIndex(ABC):
    """Abstract base class for vector indexing algorithms.

    This defines the interface that all vector indexing implementations must follow.
    Each implementation provides different trade-offs between search speed, memory usage,
    and result accuracy.
    """

    def __init__(self, dimension: int, **kwargs):
        """Initialize the vector index.

        Args:
            dimension: The dimension of the vectors to be indexed
            **kwargs: Algorithm-specific parameters
        """
        self.dimension = dimension
        self.vectors: List[ChunkVector] = []
        self._built = False

    @property
    @abstractmethod
    def index_type(self) -> IndexType:
        """Return the type of this index."""
        pass

    @abstractmethod
    async def add_vector(self, vector: ChunkVector) -> None:
        """Add a vector to the index.

        Args:
            vector: The vector to add to the index
        """
        pass

    @abstractmethod
    async def add_vectors(self, vectors: List[ChunkVector]) -> None:
        """Add multiple vectors to the index efficiently.

        Args:
            vectors: List of vectors to add to the index
        """
        pass

    @abstractmethod
    async def remove_vector(self, chunk_id: int) -> bool:
        """Remove a vector from the index.

        Args:
            chunk_id: ID of the chunk to remove

        Returns:
            True if the vector was found and removed, False otherwise
        """
        pass

    @abstractmethod
    async def search(
        self, query_embedding: List[float], k: int, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for the k most similar vectors.

        Args:
            query_embedding: The query vector
            k: Number of nearest neighbors to return
            metadata_filter: Optional metadata filter criteria

        Returns:
            List of search results sorted by similarity (descending)
        """
        pass

    @abstractmethod
    async def build_index(self) -> None:
        """Build/rebuild the index from the current vectors.

        This may be needed after adding many vectors or for algorithms
        that require a separate build phase.
        """
        pass

    async def clear(self) -> None:
        """Clear all vectors from the index."""
        self.vectors.clear()
        self._built = False

    def get_stats(self) -> IndexStats:
        """Get statistics about the current index."""
        return IndexStats(total_vectors=len(self.vectors), embedding_dimension=self.dimension)

    def is_built(self) -> bool:
        """Check if the index is built and ready for search."""
        return self._built

    def _validate_embedding(self, embedding: List[float]) -> None:
        """Validate that an embedding has the correct dimension.

        Args:
            embedding: The embedding to validate

        Raises:
            ValueError: If the embedding dimension is incorrect
        """
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding dimension {len(embedding)} does not match index dimension {self.dimension}")

    def _matches_metadata_filter(self, chunk_metadata: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
        """Check if chunk metadata matches the provided filter.

        Args:
            chunk_metadata: Metadata from the chunk
            metadata_filter: Filter criteria

        Returns:
            True if chunk matches the filter
        """
        for key, expected_value in metadata_filter.items():
            chunk_value = chunk_metadata.get(key)

            # TODO: support operators like gt, lt, contains, etc.
            if chunk_value != expected_value:
                return False

        return True
