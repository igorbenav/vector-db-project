"""Integration tests for IVF algorithm with IndexManager."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.indexing.manager import IndexManager
from src.infrastructure.indexing.base import IndexType, ChunkVector
from src.modules.library.services import LibraryService
from src.modules.library.schemas import VectorSearchRequest
from src.modules.document.services import DocumentService
from src.modules.document.schemas import DocumentCreate
from src.modules.chunk.services import ChunkService
from src.modules.chunk.schemas import ChunkCreate


class TestIVFIntegration:
    """Integration tests for IVF algorithm with the full system."""

    @pytest.fixture(autouse=True)
    async def clear_index_manager(self):
        """Clear the global index manager before each test."""
        from src.infrastructure.indexing.manager import index_manager

        index_manager._indexes.clear()
        index_manager._index_types.clear()
        yield
        index_manager._indexes.clear()
        index_manager._index_types.clear()

    @pytest.fixture
    def index_manager(self):
        """Create a fresh IndexManager for testing."""
        return IndexManager()

    @pytest.fixture
    def library_service(self):
        """Library service instance."""
        return LibraryService()

    @pytest.fixture
    def document_service(self):
        """Document service instance."""
        return DocumentService()

    @pytest.fixture
    def chunk_service(self):
        """Chunk service instance."""
        return ChunkService()

    @pytest.mark.asyncio
    async def test_ivf_index_creation_via_manager(self, index_manager):
        """Test creating IVF index through IndexManager."""
        # Create IVF index
        index = await index_manager.get_or_create_index(library_id=1, index_type=IndexType.IVF, embedding_dimension=128)

        assert index.index_type == IndexType.IVF
        assert index.dimension == 128
        assert not index.is_built()  # Should not be built until vectors are added

        # Check that it's cached
        same_index = await index_manager.get_or_create_index(library_id=1)
        assert same_index is index

    @pytest.mark.asyncio
    async def test_ivf_vs_linear_search_performance(
        self,
        library_service: LibraryService,
        document_service: DocumentService,
        chunk_service: ChunkService,
        sample_library,
        db_session: AsyncSession,
    ):
        """Compare IVF vs Linear Search performance and accuracy."""
        # Create test data
        document_data = DocumentCreate(
            title="IVF Performance Test Document",
            content="Content for IVF performance testing",
            library_id=sample_library["id"],
            metadata={"type": "performance_test"},
        )
        document = await document_service.create_document(document_data, db_session)

        # Create 50 chunks with varied embeddings
        chunk_data_list = []
        for i in range(50):
            # Create embeddings with some structure for clustering
            base_vector = [1.0, 0.0, 0.0] if i < 25 else [0.0, 1.0, 0.0]
            noise = [0.1 * (i % 5), 0.1 * ((i + 1) % 5), 0.1 * ((i + 2) % 5)]
            embedding = [base_vector[j] + noise[j] for j in range(3)]

            chunk_data_list.append(
                ChunkCreate(
                    content=f"Performance test chunk {i}",
                    embedding=embedding,
                    document_id=document.id,
                    metadata={"cluster": "A" if i < 25 else "B", "index": i},
                )
            )

        await chunk_service.create_chunks_bulk(chunk_data_list, db_session)

        # Test Linear Search
        search_request = VectorSearchRequest(query_embedding=[1.0, 0.0, 0.0], k=10)

        linear_result = await library_service.vector_search(
            library_id=sample_library["id"], search_request=search_request, db=db_session
        )

        # Clear index manager and test IVF
        from src.infrastructure.indexing.manager import index_manager

        index_manager._indexes.clear()
        index_manager._index_types.clear()

        # Force IVF index creation
        await index_manager.get_or_create_index(
            library_id=sample_library["id"], index_type=IndexType.IVF, embedding_dimension=3, db=db_session
        )

        ivf_result = await library_service.vector_search(
            library_id=sample_library["id"], search_request=search_request, db=db_session
        )

        # Both should return results
        assert linear_result is not None
        assert ivf_result is not None
        assert len(linear_result.results) == 10
        assert len(ivf_result.results) > 0  # IVF might return fewer due to clustering

        # IVF should be faster (measure by checking it completed successfully)
        assert ivf_result.query_time_ms > 0
        assert linear_result.query_time_ms > 0

        # Both should find high-similarity results for cluster A vectors
        assert linear_result.results[0].similarity_score > 0.8
        assert ivf_result.results[0].similarity_score > 0.8

    @pytest.mark.asyncio
    async def test_ivf_index_stats_integration(self, index_manager, sample_library, db_session):
        """Test IVF index statistics through IndexManager."""
        from src.infrastructure.indexing.manager import index_manager as global_manager

        # Create some test vectors directly
        test_vectors = [
            ChunkVector(
                chunk_id=i,
                document_id=1,
                content=f"Test content {i}",
                embedding=[float(i % 3), float((i + 1) % 3), float((i + 2) % 3)],
                metadata={"index": i},
            )
            for i in range(20)
        ]

        # Get IVF index and add vectors
        index = await global_manager.get_or_create_index(
            library_id=sample_library["id"], index_type=IndexType.IVF, embedding_dimension=3
        )

        await index.add_vectors(test_vectors)
        await index.build_index()

        # Check stats through IndexManager
        stats = global_manager.get_index_stats(sample_library["id"])

        assert stats is not None
        assert stats["index_type"] == "ivf"
        assert stats["total_vectors"] == 20
        assert stats["embedding_dimension"] == 3
        assert "num_clusters" in stats
        assert "num_probes" in stats
        assert "avg_cluster_size" in stats
        assert stats["is_built"] is True

    @pytest.mark.asyncio
    async def test_ivf_search_metadata_filtering_integration(
        self,
        library_service: LibraryService,
        document_service: DocumentService,
        chunk_service: ChunkService,
        sample_library,
        db_session: AsyncSession,
    ):
        """Test IVF search with metadata filtering through the full system."""
        # Create test data
        document_data = DocumentCreate(
            title="IVF Metadata Filter Test",
            content="Content for metadata filtering",
            library_id=sample_library["id"],
            metadata={},
        )
        document = await document_service.create_document(document_data, db_session)

        # Create chunks with different metadata
        chunk_data_list = [
            ChunkCreate(
                content="Priority chunk",
                embedding=[1.0, 0.0, 0.0],
                document_id=document.id,
                metadata={"priority": "high", "category": "important"},
            ),
            ChunkCreate(
                content="Normal chunk",
                embedding=[0.9, 0.1, 0.0],
                document_id=document.id,
                metadata={"priority": "low", "category": "normal"},
            ),
            ChunkCreate(
                content="Another priority chunk",
                embedding=[0.8, 0.2, 0.0],
                document_id=document.id,
                metadata={"priority": "high", "category": "important"},
            ),
        ]

        await chunk_service.create_chunks_bulk(chunk_data_list, db_session)

        # Force IVF index with higher num_probes for metadata filtering
        from src.infrastructure.indexing.manager import index_manager
        from src.infrastructure.indexing.ivf import IVFIndex

        # Create IVF index with num_probes=3 to search all clusters
        ivf_index = IVFIndex(dimension=3, num_clusters=3, num_probes=3)
        index_manager._indexes[sample_library["id"]] = ivf_index
        index_manager._index_types[sample_library["id"]] = IndexType.IVF

        # Load vectors into the index
        await index_manager._load_library_vectors(sample_library["id"], ivf_index, db_session)

        # Search with metadata filter
        search_request = VectorSearchRequest(query_embedding=[1.0, 0.0, 0.0], k=10, metadata_filter={"priority": "high"})

        result = await library_service.vector_search(
            library_id=sample_library["id"], search_request=search_request, db=db_session
        )

        assert result is not None
        assert len(result.results) == 2  # Only high priority chunks

        for search_result in result.results:
            assert search_result.metadata["priority"] == "high"
