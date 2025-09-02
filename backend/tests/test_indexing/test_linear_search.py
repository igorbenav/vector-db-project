"""Tests for LinearSearchIndex vector indexing algorithm."""

import pytest
from typing import List

from src.infrastructure.indexing.linear_search import LinearSearchIndex
from src.infrastructure.indexing.base import ChunkVector, IndexType


class TestLinearSearchIndex:
    """Test suite for LinearSearchIndex."""

    @pytest.fixture
    def index(self) -> LinearSearchIndex:
        """Create a LinearSearchIndex instance for testing."""
        return LinearSearchIndex(dimension=3)

    @pytest.fixture
    def sample_vectors(self) -> List[ChunkVector]:
        """Create sample vectors for testing."""
        return [
            ChunkVector(
                chunk_id=1,
                document_id=1,
                content="First chunk",
                embedding=[1.0, 0.0, 0.0],
                metadata={"category": "test", "priority": 1},
            ),
            ChunkVector(
                chunk_id=2,
                document_id=1,
                content="Second chunk",
                embedding=[0.0, 1.0, 0.0],
                metadata={"category": "test", "priority": 2},
            ),
            ChunkVector(
                chunk_id=3,
                document_id=2,
                content="Third chunk",
                embedding=[0.0, 0.0, 1.0],
                metadata={"category": "other", "priority": 1},
            ),
            ChunkVector(
                chunk_id=4,
                document_id=2,
                content="Fourth chunk",
                embedding=[0.7071, 0.7071, 0.0],  # 45 degrees from first two
                metadata={"category": "test", "priority": 3},
            ),
        ]

    def test_index_properties(self, index: LinearSearchIndex):
        """Test index properties and metadata."""
        assert index.index_type == IndexType.LINEAR_SEARCH
        assert index.dimension == 3
        assert not index.is_built()

    @pytest.mark.asyncio
    async def test_add_single_vector(self, index: LinearSearchIndex):
        """Test adding a single vector to the index."""
        vector = ChunkVector(chunk_id=1, document_id=1, content="Test chunk", embedding=[1.0, 0.0, 0.0], metadata={})

        await index.add_vector(vector)

        assert len(index.vectors) == 1
        assert index.vectors[0] == vector
        assert index.is_built()

    @pytest.mark.asyncio
    async def test_add_multiple_vectors(self, index: LinearSearchIndex, sample_vectors: List[ChunkVector]):
        """Test adding multiple vectors to the index."""
        await index.add_vectors(sample_vectors)

        assert len(index.vectors) == 4
        assert index.is_built()

        stats = index.get_stats()
        assert stats.total_vectors == 4
        assert stats.embedding_dimension == 3

    @pytest.mark.asyncio
    async def test_remove_vector(self, index: LinearSearchIndex, sample_vectors: List[ChunkVector]):
        """Test removing a vector from the index."""
        await index.add_vectors(sample_vectors)

        # Remove existing vector
        removed = await index.remove_vector(chunk_id=2)
        assert removed is True
        assert len(index.vectors) == 3

        # Try to remove non-existent vector
        removed = await index.remove_vector(chunk_id=999)
        assert removed is False
        assert len(index.vectors) == 3

    @pytest.mark.asyncio
    async def test_clear_index(self, index: LinearSearchIndex, sample_vectors: List[ChunkVector]):
        """Test clearing all vectors from the index."""
        await index.add_vectors(sample_vectors)
        assert len(index.vectors) == 4

        await index.clear()
        assert len(index.vectors) == 0
        assert not index.is_built()

    @pytest.mark.asyncio
    async def test_cosine_similarity_calculation(self, index: LinearSearchIndex):
        """Test cosine similarity calculation."""
        # Test identical vectors (should be 1.0)
        sim = index._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

        # Test orthogonal vectors (should be 0.0)
        sim = index._cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert abs(sim - 0.0) < 1e-6

        # Test 45-degree vectors (should be ~0.707)
        sim = index._cosine_similarity([1.0, 0.0, 0.0], [0.7071, 0.7071, 0.0])
        assert abs(sim - 0.7071) < 1e-3

        # Test zero vector handling
        sim = index._cosine_similarity([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert sim == 0.0

    @pytest.mark.asyncio
    async def test_search_empty_index(self, index: LinearSearchIndex):
        """Test search behavior on empty index."""
        results = await index.search(query_embedding=[1.0, 0.0, 0.0], k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_basic(self, index: LinearSearchIndex, sample_vectors: List[ChunkVector]):
        """Test basic search functionality."""
        await index.add_vectors(sample_vectors)

        # Search for vector most similar to [1.0, 0.0, 0.0]
        results = await index.search(query_embedding=[1.0, 0.0, 0.0], k=2)

        assert len(results) == 2
        # First result should be the identical vector (chunk_id=1)
        assert results[0].chunk_id == 1
        assert abs(results[0].similarity_score - 1.0) < 1e-6

        # Results should be sorted by similarity (descending)
        assert results[0].similarity_score >= results[1].similarity_score

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, index: LinearSearchIndex, sample_vectors: List[ChunkVector]):
        """Test search with metadata filtering."""
        await index.add_vectors(sample_vectors)

        # Search with category filter
        results = await index.search(query_embedding=[1.0, 0.0, 0.0], k=10, metadata_filter={"category": "test"})

        # Should only return vectors with category="test" (chunks 1, 2, 4)
        assert len(results) == 3
        for result in results:
            assert result.metadata["category"] == "test"

        # Search with priority filter
        results = await index.search(query_embedding=[1.0, 0.0, 0.0], k=10, metadata_filter={"priority": 1})

        # Should only return vectors with priority=1 (chunks 1, 3)
        assert len(results) == 2
        chunk_ids = {result.chunk_id for result in results}
        assert chunk_ids == {1, 3}

    @pytest.mark.asyncio
    async def test_search_k_parameter(self, index: LinearSearchIndex, sample_vectors: List[ChunkVector]):
        """Test that k parameter limits results correctly."""
        await index.add_vectors(sample_vectors)

        # Request more results than available
        results = await index.search(query_embedding=[1.0, 0.0, 0.0], k=10)
        assert len(results) == 4  # Should return all 4 vectors

        # Request fewer results than available
        results = await index.search(query_embedding=[1.0, 0.0, 0.0], k=2)
        assert len(results) == 2  # Should return exactly 2 vectors

    @pytest.mark.asyncio
    async def test_dimension_validation(self, index: LinearSearchIndex):
        """Test that dimension validation works correctly."""
        # Test correct dimension
        vector = ChunkVector(chunk_id=1, document_id=1, content="Test", embedding=[1.0, 0.0, 0.0], metadata={})
        await index.add_vector(vector)  # Should not raise

        # Test incorrect dimension
        with pytest.raises(ValueError, match="Embedding dimension"):
            vector_wrong = ChunkVector(
                chunk_id=2,
                document_id=1,
                content="Test",
                embedding=[1.0, 0.0],  # Wrong dimension (2 instead of 3)
                metadata={},
            )
            await index.add_vector(vector_wrong)

        # Test search with wrong dimension
        with pytest.raises(ValueError, match="Embedding dimension"):
            await index.search(
                query_embedding=[1.0, 0.0],  # Wrong dimension
                k=1,
            )

    @pytest.mark.asyncio
    async def test_build_index(self, index: LinearSearchIndex, sample_vectors: List[ChunkVector]):
        """Test index building (should be instant for linear search)."""
        await index.add_vectors(sample_vectors)

        # Should already be built after adding vectors
        assert index.is_built()

        # Building again should work
        await index.build_index()
        assert index.is_built()

    @pytest.mark.asyncio
    async def test_search_result_format(self, index: LinearSearchIndex, sample_vectors: List[ChunkVector]):
        """Test that search results have the correct format."""
        await index.add_vectors(sample_vectors)

        results = await index.search(query_embedding=[1.0, 0.0, 0.0], k=1)

        assert len(results) == 1
        result = results[0]

        # Check all required fields are present
        assert hasattr(result, "chunk_id")
        assert hasattr(result, "document_id")
        assert hasattr(result, "content")
        assert hasattr(result, "similarity_score")
        assert hasattr(result, "metadata")

        # Check field types
        assert isinstance(result.chunk_id, int)
        assert isinstance(result.document_id, int)
        assert isinstance(result.content, str)
        assert isinstance(result.similarity_score, float)
        assert isinstance(result.metadata, dict)

        # Check similarity score is in valid range
        assert 0.0 <= result.similarity_score <= 1.0
