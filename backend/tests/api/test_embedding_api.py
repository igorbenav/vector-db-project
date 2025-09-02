"""Tests for embedding API endpoints."""

import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport

from src.interfaces.main import app


@pytest.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    mock_service = AsyncMock()
    mock_service.embed_text.return_value = [0.1] * 768
    mock_service.embed_texts.return_value = [[0.1] * 768, [0.2] * 768]
    mock_service.is_loaded.return_value = True
    mock_service.model_name = "all-mpnet-base-v2"
    mock_service.embedding_dimension = 768
    return mock_service


class TestEmbeddingInfoEndpoint:
    """Test the embedding info endpoint."""

    @pytest.mark.asyncio
    async def test_get_embedding_info_success(self, async_client, mock_embedding_service):
        """Test successful embedding info retrieval."""

        from src.modules.embedding.schemas import EmbeddingInfo
        from src.modules.embedding.services import EmbeddingInfoService

        with patch.object(EmbeddingInfoService, "get_embedding_info") as mock_get_info:
            mock_get_info.return_value = EmbeddingInfo(model_name="all-mpnet-base-v2", dimension=768, is_loaded=True)

            response = await async_client.get("/api/v1/embedding/info")

            assert response.status_code == 200
            data = response.json()
            assert data["model_name"] == "all-mpnet-base-v2"
            assert data["dimension"] == 768
            assert data["is_loaded"] is True


class TestCreateChunkWithEmbeddings:
    """Test the create chunk with embeddings endpoint."""

    @pytest.mark.asyncio
    async def test_create_chunk_with_auto_embeddings_success(self, async_client, mock_embedding_service):
        """Test successful chunk creation with auto-generated embeddings."""

        chunk_data = {"content": "This is a test chunk content", "document_id": 1, "metadata": {"type": "test"}}

        from src.modules.chunk.schemas import ChunkRead
        from src.modules.embedding.services import EmbeddingChunkService
        from datetime import datetime

        with patch.object(EmbeddingChunkService, "create_chunk_with_text") as mock_create:
            mock_create.return_value = ChunkRead(
                id=1,
                document_id=1,
                content="This is a test chunk content",
                embedding=[0.1] * 768,
                metadata={"type": "test"},
                created_at=datetime.fromisoformat("2023-01-01T00:00:00+00:00"),
                updated_at=datetime.fromisoformat("2023-01-01T00:00:00+00:00"),
            )

            response = await async_client.post("/api/v1/embedding/chunks", json=chunk_data)

            assert response.status_code == 201
            data = response.json()
            assert data["content"] == chunk_data["content"]
            assert data["document_id"] == chunk_data["document_id"]
            assert data["metadata"] == chunk_data["metadata"]
            assert "embedding" in data

    @pytest.mark.asyncio
    async def test_create_chunk_with_provided_embeddings(self, async_client):
        """Test chunk creation with user-provided embeddings."""

        chunk_data = {"content": "Test content", "document_id": 1, "embedding": [0.5, 0.6, 0.7], "metadata": {"type": "manual"}}

        from src.modules.chunk.schemas import ChunkRead
        from src.modules.embedding.services import EmbeddingChunkService
        from datetime import datetime

        with patch.object(EmbeddingChunkService, "create_chunk_with_text") as mock_create:
            mock_create.return_value = ChunkRead(
                id=1,
                document_id=1,
                content="Test content",
                embedding=[0.5, 0.6, 0.7],
                metadata={"type": "manual"},
                created_at=datetime.fromisoformat("2023-01-01T00:00:00+00:00"),
                updated_at=datetime.fromisoformat("2023-01-01T00:00:00+00:00"),
            )

            response = await async_client.post("/api/v1/embedding/chunks", json=chunk_data)

            assert response.status_code == 201
            data = response.json()
            assert data["embedding"] == [0.5, 0.6, 0.7]

    @pytest.mark.asyncio
    async def test_create_chunk_validation_error(self, async_client):
        """Test chunk creation with validation errors."""

        # Empty content should fail validation
        chunk_data = {"content": "", "document_id": 1}

        response = await async_client.post("/api/v1/embedding/chunks", json=chunk_data)

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_create_chunk_missing_fields(self, async_client):
        """Test chunk creation with missing required fields."""

        chunk_data = {
            "content": "Test content"
            # Missing document_id
        }

        response = await async_client.post("/api/v1/embedding/chunks", json=chunk_data)

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_create_chunk_service_error(self, async_client):
        """Test chunk creation with service error."""

        chunk_data = {"content": "Test content", "document_id": 1}

        from src.modules.embedding.services import EmbeddingChunkService

        with patch.object(EmbeddingChunkService, "create_chunk_with_text") as mock_create:
            mock_create.side_effect = ValueError("Document not found")

            response = await async_client.post("/api/v1/embedding/chunks", json=chunk_data)

            assert response.status_code == 400
            assert "Document not found" in response.json()["detail"]


class TestTextSearchEndpoint:
    """Test the text search endpoint."""

    @pytest.mark.asyncio
    async def test_text_search_success(self, async_client):
        """Test successful text search."""

        search_data = {"query_text": "search for documents about AI", "k": 5, "metadata_filter": {"type": "article"}}

        from src.modules.embedding.services import EmbeddingLibraryService
        from src.modules.library.schemas import VectorSearchResponse, SearchResult

        with patch.object(EmbeddingLibraryService, "text_search") as mock_search:
            mock_search.return_value = VectorSearchResponse(
                results=[
                    SearchResult(
                        chunk_id=1,
                        document_id=1,
                        content="AI and machine learning content",
                        similarity_score=0.95,
                        metadata={"type": "article"},
                    )
                ],
                query_time_ms=2.5,
                total_chunks_searched=100,
            )

            response = await async_client.post("/api/v1/embedding/libraries/1/text-search", json=search_data)

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "query_time_ms" in data
            assert "total_chunks_searched" in data

    @pytest.mark.asyncio
    async def test_text_search_validation_errors(self, async_client):
        """Test text search with validation errors."""

        # Empty query text
        search_data = {"query_text": "", "k": 5}

        response = await async_client.post("/api/v1/embedding/libraries/1/text-search", json=search_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_text_search_invalid_k_value(self, async_client):
        """Test text search with invalid k value."""

        # k value too large
        search_data = {
            "query_text": "valid query",
            "k": 200,  # Max is 100
        }

        response = await async_client.post("/api/v1/embedding/libraries/1/text-search", json=search_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_text_search_with_metadata_filter(self, async_client):
        """Test text search with metadata filter."""

        search_data = {"query_text": "search query", "k": 3, "metadata_filter": {"type": "technical", "priority": "high"}}

        from src.modules.embedding.services import EmbeddingLibraryService
        from src.modules.library.schemas import VectorSearchResponse

        with patch.object(EmbeddingLibraryService, "text_search") as mock_search:
            mock_search.return_value = VectorSearchResponse(results=[], query_time_ms=1.5, total_chunks_searched=50)

            response = await async_client.post("/api/v1/embedding/libraries/1/text-search", json=search_data)

            assert response.status_code == 200
            # Verify the service was called with correct parameters
            mock_search.assert_called_once()


class TestDocumentAutoChunkEndpoint:
    """Test the document auto-chunk endpoint."""

    @pytest.mark.asyncio
    async def test_create_document_with_auto_chunks_success(self, async_client):
        """Test successful document creation with auto-chunking."""

        doc_data = {
            "title": "Long Document",
            "content": "This is a very long document. It contains multiple sentences. Each sentence has important information. "
            "The document should be chunked automatically.",
            "library_id": 1,
            "metadata": {"author": "test_user"},
            "chunk_size": 100,
            "chunk_overlap": 20,
        }

        from src.modules.embedding.services import EmbeddingDocumentService
        from src.modules.document.schemas import DocumentRead
        from datetime import datetime

        with patch.object(EmbeddingDocumentService, "create_document_with_auto_chunks") as mock_create:
            mock_create.return_value = DocumentRead(
                id=1,
                library_id=1,
                title="Long Document",
                metadata={"author": "test_user"},
                created_at=datetime.fromisoformat("2023-01-01T00:00:00+00:00"),
                updated_at=datetime.fromisoformat("2023-01-01T00:00:00+00:00"),
                chunk_count=3,
            )

            response = await async_client.post("/api/v1/embedding/documents/auto-chunk", json=doc_data)

            assert response.status_code == 201
            data = response.json()
            assert data["title"] == doc_data["title"]
            assert data["library_id"] == doc_data["library_id"]
            assert data["metadata"] == doc_data["metadata"]

    @pytest.mark.asyncio
    async def test_create_document_default_chunk_params(self, async_client):
        """Test document creation with default chunking parameters."""

        doc_data = {"title": "Test Document", "content": "Short content for testing.", "library_id": 1}

        from src.modules.embedding.services import EmbeddingDocumentService
        from src.modules.document.schemas import DocumentRead
        from datetime import datetime

        with patch.object(EmbeddingDocumentService, "create_document_with_auto_chunks") as mock_create:
            mock_create.return_value = DocumentRead(
                id=1,
                library_id=1,
                title="Test Document",
                metadata={},
                created_at=datetime.fromisoformat("2023-01-01T00:00:00+00:00"),
                updated_at=datetime.fromisoformat("2023-01-01T00:00:00+00:00"),
                chunk_count=1,
            )

            response = await async_client.post("/api/v1/embedding/documents/auto-chunk", json=doc_data)

            assert response.status_code == 201
            # Should use default chunk_size=500, chunk_overlap=50

    @pytest.mark.asyncio
    async def test_create_document_validation_errors(self, async_client):
        """Test document creation with validation errors."""

        # Missing required fields
        doc_data = {
            "title": "",  # Empty title should fail
            "content": "Valid content",
            "library_id": 1,
        }

        response = await async_client.post("/api/v1/embedding/documents/auto-chunk", json=doc_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_document_invalid_chunk_params(self, async_client):
        """Test document creation with invalid chunking parameters."""

        doc_data = {
            "title": "Test Document",
            "content": "Valid content",
            "library_id": 1,
            "chunk_size": 50,  # Too small (min 100)
            "chunk_overlap": 250,  # Too large (max 200)
        }

        response = await async_client.post("/api/v1/embedding/documents/auto-chunk", json=doc_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_document_service_error(self, async_client):
        """Test document creation with service error."""

        doc_data = {
            "title": "Test Document",
            "content": "Valid content",
            "library_id": 999,  # Non-existent library
        }

        from src.modules.embedding.services import EmbeddingDocumentService

        with patch.object(EmbeddingDocumentService, "create_document_with_auto_chunks") as mock_create:
            mock_create.side_effect = ValueError("Library not found")

            response = await async_client.post("/api/v1/embedding/documents/auto-chunk", json=doc_data)

            assert response.status_code == 400
            assert "Library not found" in response.json()["detail"]


class TestEmbeddingEndpointIntegration:
    """Integration tests for embedding endpoints."""

    @pytest.mark.asyncio
    async def test_embedding_endpoints_exist(self, async_client):
        """Test that all embedding endpoints are properly registered."""

        # Test that endpoints return proper method not allowed for wrong HTTP methods
        response = await async_client.get("/api/v1/embedding/chunks")
        assert response.status_code == 405  # Method not allowed (POST expected)

        response = await async_client.get("/api/v1/embedding/libraries/1/text-search")
        assert response.status_code == 405  # Method not allowed (POST expected)

        response = await async_client.get("/api/v1/embedding/documents/auto-chunk")
        assert response.status_code == 405  # Method not allowed (POST expected)

    @pytest.mark.asyncio
    async def test_embedding_endpoints_content_type(self, async_client):
        """Test that endpoints require JSON content type."""

        # Test with missing content-type header
        response = await async_client.post(
            "/api/v1/embedding/chunks", content="not json", headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422  # Unprocessable entity

    @pytest.mark.asyncio
    async def test_embedding_api_documentation(self, async_client):
        """Test that embedding endpoints appear in OpenAPI documentation."""

        response = await async_client.get("/openapi.json")
        assert response.status_code == 200

        openapi_spec = response.json()
        paths = openapi_spec["paths"]

        # Verify embedding endpoints are documented
        assert "/api/v1/embedding/info" in paths
        assert "/api/v1/embedding/chunks" in paths
        assert "/api/v1/embedding/libraries/{library_id}/text-search" in paths
        assert "/api/v1/embedding/documents/auto-chunk" in paths
