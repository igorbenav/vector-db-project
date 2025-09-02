"""Tests for IndexManager service."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.indexing.manager import IndexManager
from src.infrastructure.indexing.base import ChunkVector, IndexType


class TestIndexManager:
    """Test suite for IndexManager."""

    @pytest.fixture
    def index_manager(self) -> IndexManager:
        """Create an IndexManager instance for testing."""
        return IndexManager()

    @pytest.fixture
    def mock_db(self) -> AsyncSession:
        """Create a mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def sample_chunk_vectors(self) -> list[ChunkVector]:
        """Create sample chunk vectors for testing."""
        return [
            ChunkVector(
                chunk_id=1, document_id=1, content="First chunk", embedding=[1.0, 0.0, 0.0], metadata={"category": "test"}
            ),
            ChunkVector(
                chunk_id=2, document_id=1, content="Second chunk", embedding=[0.0, 1.0, 0.0], metadata={"category": "test"}
            ),
        ]

    @pytest.mark.asyncio
    async def test_get_or_create_index_new(self, index_manager: IndexManager, mock_db: AsyncSession):
        """Test creating a new index for a library."""
        library_id = 1

        # Mock the database loading
        index_manager._load_library_vectors = AsyncMock()

        index = await index_manager.get_or_create_index(
            library_id=library_id, index_type=IndexType.LINEAR_SEARCH, embedding_dimension=3, db=mock_db
        )

        assert index is not None
        assert index.dimension == 3
        assert index.index_type == IndexType.LINEAR_SEARCH
        assert library_id in index_manager._indexes
        assert index_manager._index_types[library_id] == IndexType.LINEAR_SEARCH

        # Should have called _load_library_vectors
        index_manager._load_library_vectors.assert_called_once_with(library_id, index, mock_db)

    @pytest.mark.asyncio
    async def test_get_or_create_index_existing(self, index_manager: IndexManager, mock_db: AsyncSession):
        """Test getting an existing index for a library."""
        library_id = 1

        # Create index first
        index_manager._load_library_vectors = AsyncMock()
        first_index = await index_manager.get_or_create_index(library_id=library_id, embedding_dimension=3, db=mock_db)

        # Get the same index again
        second_index = await index_manager.get_or_create_index(library_id=library_id, embedding_dimension=3)

        # Should return the same instance
        assert first_index is second_index
        # Should only load vectors once
        assert index_manager._load_library_vectors.call_count == 1

    @pytest.mark.asyncio
    async def test_search_without_db(self, index_manager: IndexManager):
        """Test search when database session is not provided."""
        library_id = 1

        # Create empty index
        await index_manager.get_or_create_index(library_id=library_id, embedding_dimension=3)

        results = await index_manager.search(library_id=library_id, query_embedding=[1.0, 0.0, 0.0], k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_add_remove_chunk_vector(self, index_manager: IndexManager, sample_chunk_vectors: list[ChunkVector]):
        """Test adding and removing chunk vectors from index."""
        library_id = 1

        # Create index first
        await index_manager.get_or_create_index(library_id=library_id, embedding_dimension=3)

        # Add vector
        vector = sample_chunk_vectors[0]
        await index_manager.add_chunk_vector(library_id, vector)

        index = index_manager._indexes[library_id]
        assert len(index.vectors) == 1
        assert index.vectors[0] == vector

        # Remove vector
        removed = await index_manager.remove_chunk_vector(library_id, vector.chunk_id)
        assert removed is True
        assert len(index.vectors) == 0

        # Try to remove non-existent vector
        removed = await index_manager.remove_chunk_vector(library_id, 999)
        assert removed is False

    @pytest.mark.asyncio
    async def test_remove_from_nonexistent_library(self, index_manager: IndexManager):
        """Test removing vector from library that doesn't have an index."""
        removed = await index_manager.remove_chunk_vector(999, 1)
        assert removed is False

    def test_get_index_stats_existing(self, index_manager: IndexManager):
        """Test getting statistics for an existing index."""
        library_id = 1

        # Create and populate index
        index_manager._indexes[library_id] = MagicMock()
        index_manager._indexes[library_id].get_stats.return_value = MagicMock(total_vectors=5, embedding_dimension=128)
        index_manager._indexes[library_id].index_type = IndexType.LINEAR_SEARCH
        index_manager._indexes[library_id].time_complexity_search = "O(n * d)"
        index_manager._indexes[library_id].space_complexity = "O(n * d)"
        index_manager._indexes[library_id].is_built.return_value = True

        stats = index_manager.get_index_stats(library_id)

        assert stats is not None
        assert stats["index_type"] == "linear_search"
        assert stats["total_vectors"] == 5
        assert stats["embedding_dimension"] == 128
        assert stats["is_built"] is True

    def test_get_index_stats_nonexistent(self, index_manager: IndexManager):
        """Test getting statistics for a non-existent index."""
        stats = index_manager.get_index_stats(999)
        assert stats is None

    @pytest.mark.asyncio
    async def test_rebuild_library_index(self, index_manager: IndexManager, mock_db: AsyncSession):
        """Test rebuilding an index for a library."""
        library_id = 1

        # Create index first
        index_manager._load_library_vectors = AsyncMock()
        await index_manager.get_or_create_index(library_id=library_id, embedding_dimension=3, db=mock_db)

        # Add some vectors to simulate existing data
        index = index_manager._indexes[library_id]
        await index.add_vector(ChunkVector(1, 1, "test", [1.0, 0.0, 0.0], {}))

        # Rebuild should clear and reload
        await index_manager.rebuild_library_index(library_id, mock_db)

        # Should have called _load_library_vectors twice (once on creation, once on rebuild)
        assert index_manager._load_library_vectors.call_count == 2

    @pytest.mark.asyncio
    async def test_unsupported_index_type(self, index_manager: IndexManager):
        """Test creating index with unsupported type."""
        with pytest.raises(ValueError, match="Index type .* not implemented"):
            await index_manager.get_or_create_index(
                library_id=1,
                index_type=IndexType.LSH,  # Not implemented yet
                embedding_dimension=3,
            )
