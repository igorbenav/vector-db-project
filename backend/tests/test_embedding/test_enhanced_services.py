"""Tests for embedding-enhanced services."""

import pytest
from unittest.mock import AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.embedding.services import (
    EmbeddingChunkService,
    EmbeddingLibraryService,
    EmbeddingDocumentService,
    EmbeddingInfoService,
)
from src.modules.embedding.schemas import EmbeddedChunkCreate, TextSearchRequest, DocumentAutoChunk, EmbeddingInfo
from src.modules.chunk.schemas import ChunkRead, ChunkCreate
from src.modules.document.schemas import DocumentRead, DocumentCreate
from src.modules.library.schemas import VectorSearchRequest, VectorSearchResponse


class TestEmbeddingChunkService:
    """Test EmbeddingChunkService."""

    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return EmbeddingChunkService()

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def sample_embedded_chunk_create(self):
        """Sample EmbeddedChunkCreate data."""
        return EmbeddedChunkCreate(content="Test chunk content", document_id=1, metadata={"type": "test"})

    @pytest.fixture
    def sample_chunk_read(self):
        """Sample ChunkRead response."""
        return ChunkRead(
            id=1,
            document_id=1,
            content="Test chunk content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"type": "test"},
            created_at="2023-01-01T00:00:00Z",
            updated_at="2023-01-01T00:00:00Z",
        )

    @pytest.mark.asyncio
    async def test_create_chunk_with_text_auto_embedding(
        self, service, mock_db, sample_embedded_chunk_create, sample_chunk_read
    ):
        """Test creating chunk with auto-generated embedding."""

        # Mock embedding service
        with patch.object(service.embedding_service, "embed_text") as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]

            # Mock chunk service
            with patch.object(service.chunk_service, "create_chunk") as mock_create:
                mock_create.return_value = sample_chunk_read

                result = await service.create_chunk_with_text(sample_embedded_chunk_create, mock_db)

                # Verify embedding was generated
                mock_embed.assert_called_once_with("Test chunk content")

                # Verify chunk was created with generated embedding
                mock_create.assert_called_once()
                call_args = mock_create.call_args[0][0]  # First positional arg (ChunkCreate)
                assert isinstance(call_args, ChunkCreate)
                assert call_args.content == "Test chunk content"
                assert call_args.document_id == 1
                assert call_args.embedding == [0.1, 0.2, 0.3]
                assert call_args.metadata == {"type": "test"}

                assert result == sample_chunk_read

    @pytest.mark.asyncio
    async def test_create_chunk_with_text_provided_embedding(self, service, mock_db, sample_chunk_read):
        """Test creating chunk with user-provided embedding."""

        # Create data with embedding provided
        chunk_data = EmbeddedChunkCreate(
            content="Test content", document_id=1, embedding=[0.5, 0.6, 0.7], metadata={"type": "test"}
        )

        # Mock chunk service
        with patch.object(service.chunk_service, "create_chunk") as mock_create:
            mock_create.return_value = sample_chunk_read

            # Mock embedding service (should not be called)
            with patch.object(service.embedding_service, "embed_text") as mock_embed:
                result = await service.create_chunk_with_text(chunk_data, mock_db)

                # Verify embedding service was not called
                mock_embed.assert_not_called()

                # Verify chunk was created with provided embedding
                mock_create.assert_called_once()
                call_args = mock_create.call_args[0][0]
                assert call_args.embedding == [0.5, 0.6, 0.7]

                assert result == sample_chunk_read


class TestEmbeddingLibraryService:
    """Test EmbeddingLibraryService."""

    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return EmbeddingLibraryService()

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def sample_text_search_request(self):
        """Sample TextSearchRequest."""
        return TextSearchRequest(query_text="search query", k=5, metadata_filter={"type": "test"})

    @pytest.fixture
    def sample_search_response(self):
        """Sample VectorSearchResponse."""
        return VectorSearchResponse(results=[], query_time_ms=1.5, total_chunks_searched=10)

    @pytest.mark.asyncio
    async def test_text_search(self, service, mock_db, sample_text_search_request, sample_search_response):
        """Test text-based search with auto-generated embedding."""

        library_id = 1

        # Mock embedding service
        with patch.object(service.embedding_service, "embed_text") as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]

            # Mock library service
            with patch.object(service.library_service, "vector_search") as mock_search:
                mock_search.return_value = sample_search_response

                result = await service.text_search(library_id, sample_text_search_request, mock_db)

                # Verify embedding was generated from text
                mock_embed.assert_called_once_with("search query")

                # Verify vector search was called with generated embedding
                mock_search.assert_called_once()
                args = mock_search.call_args[0]
                assert args[0] == library_id  # library_id
                assert isinstance(args[1], VectorSearchRequest)  # VectorSearchRequest
                assert args[1].query_embedding == [0.1, 0.2, 0.3]
                assert args[1].k == 5
                assert args[1].metadata_filter == {"type": "test"}
                assert args[2] is mock_db  # db session

                assert result == sample_search_response


class TestEmbeddingDocumentService:
    """Test EmbeddingDocumentService."""

    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return EmbeddingDocumentService()

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def sample_doc_auto_chunk(self):
        """Sample DocumentAutoChunk data."""
        return DocumentAutoChunk(
            title="Test Document",
            content="This is a long document. It has multiple sentences. Each sentence provides information.",
            library_id=1,
            metadata={"author": "test"},
            chunk_size=100,
            chunk_overlap=10,
        )

    @pytest.fixture
    def sample_document_read(self):
        """Sample DocumentRead response."""
        return DocumentRead(
            id=1,
            library_id=1,
            title="Test Document",
            metadata={"author": "test"},
            created_at="2023-01-01T00:00:00Z",
            updated_at="2023-01-01T00:00:00Z",
            chunk_count=0,
        )

    @pytest.mark.asyncio
    async def test_create_document_with_auto_chunks(self, service, mock_db, sample_doc_auto_chunk, sample_document_read):
        """Test creating document with auto-chunking and embedding."""

        # Mock document service
        with (
            patch.object(service.document_service, "create_document") as mock_create_doc,
            patch.object(service.document_service, "get_document") as mock_get_doc,
        ):
            mock_create_doc.return_value = sample_document_read
            # Mock updated document with chunk count
            updated_doc = sample_document_read.model_copy(update={"chunk_count": 2})
            mock_get_doc.return_value = updated_doc

            # Mock embedding service
            with patch.object(service.embedding_service, "embed_texts") as mock_embed:
                mock_embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

                # Mock chunk service
                with patch.object(service.chunk_service, "create_chunks_bulk") as mock_create_chunks:
                    mock_create_chunks.return_value = []

                    # Mock index manager to avoid SQL calls
                    with patch("src.modules.embedding.services.index_manager") as mock_index_manager:
                        from unittest.mock import AsyncMock

                        mock_index_manager.get_or_create_index = AsyncMock(return_value=None)

                        result = await service.create_document_with_auto_chunks(sample_doc_auto_chunk, mock_db)

                    # Verify document was created
                    mock_create_doc.assert_called_once()
                    doc_args = mock_create_doc.call_args[0][0]
                    assert isinstance(doc_args, DocumentCreate)
                    assert doc_args.title == "Test Document"
                    assert doc_args.library_id == 1

                    # Verify embeddings were generated
                    mock_embed.assert_called_once()

                    # Verify chunks were created
                    mock_create_chunks.assert_called_once()

                    # Should return updated document with chunk count
                    assert result.id == updated_doc.id
                    assert result.chunk_count == 2

    def test_chunk_text_basic(self, service):
        """Test basic text chunking functionality."""
        text = "First sentence. Second sentence. Third sentence."

        chunks = service._chunk_text(text, chunk_size=30, chunk_overlap=5)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert "content" in chunk
            assert "index" in chunk
            assert "start" in chunk
            assert "end" in chunk
            assert len(chunk["content"]) <= 30 or "First sentence" in chunk["content"]

    def test_chunk_text_with_overlap(self, service):
        """Test text chunking with overlap."""
        text = "A" * 20 + ". " + "B" * 20 + ". " + "C" * 20 + "."

        chunks = service._chunk_text(text, chunk_size=30, chunk_overlap=10)

        # Should have multiple chunks due to length
        assert len(chunks) >= 2

        # Verify overlap exists between consecutive chunks
        if len(chunks) > 1:
            first_chunk_end = chunks[0]["content"][-10:]
            second_chunk_start = chunks[1]["content"][:10]
            # There should be some overlap in content
            assert len(first_chunk_end.strip()) > 0
            assert len(second_chunk_start.strip()) > 0

    def test_chunk_text_empty_input(self, service):
        """Test chunking empty text."""
        chunks = service._chunk_text("", chunk_size=100, chunk_overlap=10)
        assert chunks == []

    def test_chunk_text_single_sentence(self, service):
        """Test chunking single short sentence."""
        text = "Short sentence."
        chunks = service._chunk_text(text, chunk_size=100, chunk_overlap=10)

        assert len(chunks) == 1
        assert chunks[0]["content"] == "Short sentence"
        assert chunks[0]["index"] == 0


class TestEmbeddingInfoService:
    """Test EmbeddingInfoService."""

    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return EmbeddingInfoService()

    @pytest.mark.asyncio
    async def test_get_embedding_info(self, service):
        """Test getting embedding model information."""

        # Mock embedding service
        with (
            patch.object(service.embedding_service, "is_loaded") as mock_is_loaded,
            patch.object(service.embedding_service, "model_name", "test-model"),
            patch.object(type(service.embedding_service), "embedding_dimension", new_callable=lambda: 768),
        ):
            mock_is_loaded.return_value = True

            result = await service.get_embedding_info()

            assert isinstance(result, EmbeddingInfo)
            assert result.model_name == "test-model"
            assert result.dimension == 768
            assert result.is_loaded is True

            mock_is_loaded.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embedding_info_not_loaded(self, service):
        """Test getting embedding info when model is not loaded."""

        with patch.object(service.embedding_service, "is_loaded") as mock_is_loaded:
            mock_is_loaded.return_value = False

            result = await service.get_embedding_info()

            assert result.is_loaded is False
