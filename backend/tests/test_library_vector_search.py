"""Integration tests for library vector search functionality."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.library.services import LibraryService
from src.modules.library.schemas import VectorSearchRequest
from src.modules.document.services import DocumentService
from src.modules.document.schemas import DocumentCreate
from src.modules.chunk.services import ChunkService
from src.modules.chunk.schemas import ChunkCreate
from src.infrastructure.indexing.manager import index_manager


class TestLibraryVectorSearch:
    """Integration tests for vector search in libraries."""

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

    @pytest.fixture(autouse=True)
    async def clear_index_manager(self):
        """Clear the global index manager before each test."""
        index_manager._indexes.clear()
        index_manager._index_types.clear()
        yield
        # Clear after test as well for cleanup
        index_manager._indexes.clear()
        index_manager._index_types.clear()

    @pytest.mark.asyncio
    async def test_vector_search_empty_library(self, library_service: LibraryService, sample_library, db_session: AsyncSession):
        """Test vector search on an empty library."""
        search_request = VectorSearchRequest(query_embedding=[1.0, 0.0, 0.0], k=5)

        result = await library_service.vector_search(
            library_id=sample_library["id"], search_request=search_request, db=db_session
        )

        assert result is not None
        assert result.results == []
        assert result.total_chunks_searched == 0
        assert result.query_time_ms > 0

    @pytest.mark.asyncio
    async def test_vector_search_with_chunks(
        self,
        library_service: LibraryService,
        document_service: DocumentService,
        chunk_service: ChunkService,
        sample_library,
        db_session: AsyncSession,
    ):
        """Test vector search with actual chunks in the library."""
        # Create a document in the library
        document_data = DocumentCreate(
            title="Test Document", content="This is test content", library_id=sample_library["id"], metadata={"type": "test"}
        )
        document = await document_service.create_document(document_data, db_session)

        # Create chunks with different embeddings
        chunk_data_list = [
            ChunkCreate(
                content="First chunk content",
                embedding=[1.0, 0.0, 0.0],  # Perfect match for our query
                document_id=document.id,
                metadata={"category": "primary"},
            ),
            ChunkCreate(
                content="Second chunk content",
                embedding=[0.0, 1.0, 0.0],  # Orthogonal to query
                document_id=document.id,
                metadata={"category": "secondary"},
            ),
            ChunkCreate(
                content="Third chunk content",
                embedding=[0.7071, 0.7071, 0.0],  # 45 degrees from query
                document_id=document.id,
                metadata={"category": "primary"},
            ),
        ]

        created_chunks = await chunk_service.create_chunks_bulk(chunk_data_list, db_session)
        assert len(created_chunks) == 3

        # Perform vector search
        search_request = VectorSearchRequest(query_embedding=[1.0, 0.0, 0.0], k=3)

        result = await library_service.vector_search(
            library_id=sample_library["id"], search_request=search_request, db=db_session
        )

        assert result is not None
        assert len(result.results) == 3
        assert result.total_chunks_searched == 3
        assert result.query_time_ms > 0

        # Results should be sorted by similarity (descending)
        similarities = [r.similarity_score for r in result.results]
        assert similarities == sorted(similarities, reverse=True)

        # First result should be the perfect match
        assert abs(result.results[0].similarity_score - 1.0) < 1e-6
        assert result.results[0].content == "First chunk content"

    @pytest.mark.asyncio
    async def test_vector_search_with_metadata_filter(
        self,
        library_service: LibraryService,
        document_service: DocumentService,
        chunk_service: ChunkService,
        sample_library,
        db_session: AsyncSession,
    ):
        """Test vector search with metadata filtering."""
        # Create document and chunks
        document_data = DocumentCreate(
            title="Test Document", content="This is test content", library_id=sample_library["id"], metadata={}
        )
        document = await document_service.create_document(document_data, db_session)

        chunk_data_list = [
            ChunkCreate(
                content="Primary chunk",
                embedding=[1.0, 0.0, 0.0],
                document_id=document.id,
                metadata={"category": "primary", "priority": 1},
            ),
            ChunkCreate(
                content="Secondary chunk",
                embedding=[0.9, 0.1, 0.0],
                document_id=document.id,
                metadata={"category": "secondary", "priority": 1},
            ),
            ChunkCreate(
                content="Another primary chunk",
                embedding=[0.8, 0.2, 0.0],
                document_id=document.id,
                metadata={"category": "primary", "priority": 2},
            ),
        ]

        await chunk_service.create_chunks_bulk(chunk_data_list, db_session)

        # Search with metadata filter
        search_request = VectorSearchRequest(query_embedding=[1.0, 0.0, 0.0], k=10, metadata_filter={"category": "primary"})

        result = await library_service.vector_search(
            library_id=sample_library["id"], search_request=search_request, db=db_session
        )

        assert result is not None
        assert len(result.results) == 2  # Only primary category chunks

        for search_result in result.results:
            assert search_result.metadata["category"] == "primary"

    @pytest.mark.asyncio
    async def test_vector_search_k_parameter(
        self,
        library_service: LibraryService,
        document_service: DocumentService,
        chunk_service: ChunkService,
        sample_library,
        db_session: AsyncSession,
    ):
        """Test that k parameter correctly limits results."""
        # Create document and chunks
        document_data = DocumentCreate(
            title="Test Document", content="This is test content", library_id=sample_library["id"], metadata={}
        )
        document = await document_service.create_document(document_data, db_session)

        # Create 5 chunks
        chunk_data_list = [
            ChunkCreate(
                content=f"Chunk {i}", embedding=[1.0, float(i) * 0.1, 0.0], document_id=document.id, metadata={"index": i}
            )
            for i in range(5)
        ]

        await chunk_service.create_chunks_bulk(chunk_data_list, db_session)

        # Request fewer results than available
        search_request = VectorSearchRequest(query_embedding=[1.0, 0.0, 0.0], k=3)

        result = await library_service.vector_search(
            library_id=sample_library["id"], search_request=search_request, db=db_session
        )

        assert result is not None
        assert len(result.results) == 3  # Should return exactly 3 results
        assert result.total_chunks_searched == 5  # But searched all 5 chunks

    @pytest.mark.asyncio
    async def test_vector_search_nonexistent_library(self, library_service: LibraryService, db_session: AsyncSession):
        """Test vector search on non-existent library."""
        search_request = VectorSearchRequest(query_embedding=[1.0, 0.0, 0.0], k=5)

        result = await library_service.vector_search(
            library_id=999,  # Non-existent library
            search_request=search_request,
            db=db_session,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_vector_search_dimension_mismatch(
        self,
        library_service: LibraryService,
        document_service: DocumentService,
        chunk_service: ChunkService,
        sample_library,
        db_session: AsyncSession,
    ):
        """Test vector search with mismatched embedding dimensions."""
        # Create document and chunk with 3D embedding
        document_data = DocumentCreate(
            title="Test Document", content="This is test content", library_id=sample_library["id"], metadata={}
        )
        document = await document_service.create_document(document_data, db_session)

        chunk_data = ChunkCreate(
            content="Test chunk",
            embedding=[1.0, 0.0, 0.0],  # 3D embedding
            document_id=document.id,
            metadata={},
        )

        await chunk_service.create_chunk(chunk_data, db_session)

        # Search with 2D query embedding
        search_request = VectorSearchRequest(
            query_embedding=[1.0, 0.0],  # 2D query
            k=5,
        )

        # This should raise a ValueError due to dimension mismatch
        with pytest.raises(ValueError, match="Embedding dimension"):
            await library_service.vector_search(library_id=sample_library["id"], search_request=search_request, db=db_session)

    @pytest.mark.asyncio
    async def test_vector_search_multiple_documents(
        self,
        library_service: LibraryService,
        document_service: DocumentService,
        chunk_service: ChunkService,
        sample_library,
        db_session: AsyncSession,
    ):
        """Test vector search across multiple documents in a library."""
        # Create two documents
        doc1_data = DocumentCreate(
            title="Document 1", content="First document", library_id=sample_library["id"], metadata={"doc_type": "type1"}
        )
        doc2_data = DocumentCreate(
            title="Document 2", content="Second document", library_id=sample_library["id"], metadata={"doc_type": "type2"}
        )

        doc1 = await document_service.create_document(doc1_data, db_session)
        doc2 = await document_service.create_document(doc2_data, db_session)

        # Create chunks in both documents
        chunk_data_list = [
            ChunkCreate(
                content="Chunk from doc 1", embedding=[1.0, 0.0, 0.0], document_id=doc1.id, metadata={"source": "doc1"}
            ),
            ChunkCreate(
                content="Chunk from doc 2", embedding=[0.0, 1.0, 0.0], document_id=doc2.id, metadata={"source": "doc2"}
            ),
        ]

        await chunk_service.create_chunks_bulk(chunk_data_list, db_session)

        # Search should find chunks from both documents
        search_request = VectorSearchRequest(query_embedding=[1.0, 0.0, 0.0], k=10)

        result = await library_service.vector_search(
            library_id=sample_library["id"], search_request=search_request, db=db_session
        )

        assert result is not None
        assert len(result.results) == 2
        assert result.total_chunks_searched == 2

        # Check that chunks from both documents are returned
        document_ids = {r.document_id for r in result.results}
        assert document_ids == {doc1.id, doc2.id}
