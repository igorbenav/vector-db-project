"""API tests for Document endpoints."""

from typing import Any, Dict
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.chunk.services import ChunkService
from src.modules.chunk.schemas import ChunkCreate
from src.infrastructure.indexing.manager import index_manager


class TestDocumentAPI:
    """API tests for document endpoints."""

    @pytest.fixture(autouse=True)
    async def clear_index_manager(self):
        """Clear the global index manager before each test."""
        index_manager._indexes.clear()
        index_manager._index_types.clear()
        yield
        index_manager._indexes.clear()
        index_manager._index_types.clear()

    @pytest.mark.asyncio
    async def test_create_document_success(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test successful document creation."""
        document_data = {
            "title": "Test Document",
            "library_id": sample_library["id"],
            "metadata": {"type": "test", "pages": 10},
        }

        response = await client.post("/api/v1/document/", json=document_data)

        assert response.status_code == 201
        data = response.json()
        assert data["title"] == document_data["title"]
        assert data["library_id"] == document_data["library_id"]
        assert data["metadata"] == document_data["metadata"]
        assert data["chunk_count"] == 0
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_create_document_minimal_data(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test document creation with minimal required data."""
        document_data = {"title": "Minimal Document", "library_id": sample_library["id"]}

        response = await client.post("/api/v1/document/", json=document_data)

        assert response.status_code == 201
        data = response.json()
        assert data["title"] == document_data["title"]
        assert data["library_id"] == document_data["library_id"]
        assert data["metadata"] == {}
        assert data["chunk_count"] == 0

    @pytest.mark.asyncio
    async def test_create_document_invalid_library(self, client: AsyncClient):
        """Test document creation with non-existent library."""
        document_data = {"title": "Test Document", "library_id": 999, "metadata": {"type": "test"}}

        response = await client.post("/api/v1/document/", json=document_data)

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_create_document_validation_errors(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test document creation validation errors."""
        # Test missing title
        response = await client.post("/api/v1/document/", json={"library_id": sample_library["id"]})
        assert response.status_code == 422

        # Test empty title
        response = await client.post("/api/v1/document/", json={"title": "", "library_id": sample_library["id"]})
        assert response.status_code == 422

        # Test missing library_id
        response = await client.post("/api/v1/document/", json={"title": "Test"})
        assert response.status_code == 422

        # Test invalid metadata type
        response = await client.post(
            "/api/v1/document/", json={"title": "Test", "library_id": sample_library["id"], "metadata": "not a dict"}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_document_success(self, client: AsyncClient, test_document: Dict[str, Any]):
        """Test successful document retrieval."""
        response = await client.get(f"/api/v1/document/{test_document['id']}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_document["id"]
        assert data["title"] == test_document["title"]
        assert data["library_id"] == test_document["library_id"]
        assert data["metadata"] == test_document["metadata"]
        assert data["chunk_count"] == 0

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, client: AsyncClient):
        """Test document retrieval with non-existent ID."""
        response = await client.get("/api/v1/document/999")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_get_document_invalid_id(self, client: AsyncClient):
        """Test document retrieval with invalid ID."""
        response = await client.get("/api/v1/document/invalid")

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_documents_empty(self, client: AsyncClient):
        """Test getting documents when none exist."""
        response = await client.get("/api/v1/document/")

        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["total_count"] == 0
        assert data["page"] == 1
        assert data["items_per_page"] == 50

    @pytest.mark.asyncio
    async def test_get_documents_with_data(self, client: AsyncClient, test_document: Dict[str, Any]):
        """Test getting documents with existing data."""
        response = await client.get("/api/v1/document/")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["total_count"] == 1
        assert data["data"][0]["id"] == test_document["id"]
        assert data["data"][0]["title"] == test_document["title"]

    @pytest.mark.asyncio
    async def test_get_documents_by_library(
        self, client: AsyncClient, test_document: Dict[str, Any], sample_library: Dict[str, Any]
    ):
        """Test getting documents filtered by library."""
        # Create a document in sample_library (different from test_document's library)
        document_data = {
            "title": "Library Specific Document",
            "library_id": sample_library["id"],
            "metadata": {"type": "specific"},
        }
        response = await client.post("/api/v1/document/", json=document_data)
        assert response.status_code == 201
        created_doc = response.json()

        # Get documents from sample_library only
        response = await client.get(f"/api/v1/document/?library_id={sample_library['id']}")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == created_doc["id"]
        assert data["data"][0]["library_id"] == sample_library["id"]

    @pytest.mark.asyncio
    async def test_get_documents_pagination(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test document pagination."""
        # Create multiple documents
        for i in range(3):
            document_data = {"title": f"Test Document {i}", "library_id": sample_library["id"]}
            await client.post("/api/v1/document/", json=document_data)

        # Test first page with page size 2
        response = await client.get("/api/v1/document/?page=1&items_per_page=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert data["total_count"] == 3
        assert data["page"] == 1
        assert data["items_per_page"] == 2

        # Test second page
        response = await client.get("/api/v1/document/?page=2&items_per_page=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["page"] == 2

    @pytest.mark.asyncio
    async def test_get_documents_invalid_pagination(self, client: AsyncClient):
        """Test document pagination with invalid parameters."""
        # Test invalid page number
        response = await client.get("/api/v1/document/?page=0")
        assert response.status_code == 422

        # Test invalid items_per_page
        response = await client.get("/api/v1/document/?items_per_page=0")
        assert response.status_code == 422

        # Test negative values
        response = await client.get("/api/v1/document/?page=-1")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_update_document_success(self, client: AsyncClient, test_document: Dict[str, Any]):
        """Test successful document update."""
        update_data = {"title": "Updated Document Title", "metadata": {"updated": True, "version": 2}}

        response = await client.put(f"/api/v1/document/{test_document['id']}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_document["id"]
        assert data["title"] == update_data["title"]
        assert data["metadata"] == update_data["metadata"]

    @pytest.mark.asyncio
    async def test_update_document_partial(self, client: AsyncClient, test_document: Dict[str, Any]):
        """Test partial document update."""
        update_data = {"title": "Partially Updated Title"}

        response = await client.put(f"/api/v1/document/{test_document['id']}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == update_data["title"]
        assert data["metadata"] == test_document["metadata"]  # Should remain unchanged

    @pytest.mark.asyncio
    async def test_update_document_not_found(self, client: AsyncClient):
        """Test updating non-existent document."""
        update_data = {"title": "Updated Title"}

        response = await client.put("/api/v1/document/999", json=update_data)

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_document_validation_errors(self, client: AsyncClient, test_document: Dict[str, Any]):
        """Test document update validation errors."""
        # Test empty title
        response = await client.put(f"/api/v1/document/{test_document['id']}", json={"title": ""})
        assert response.status_code == 422

        # Test invalid metadata type
        response = await client.put(f"/api/v1/document/{test_document['id']}", json={"metadata": "not a dict"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_document_success(self, client: AsyncClient, test_document: Dict[str, Any]):
        """Test successful document deletion."""
        response = await client.delete(f"/api/v1/document/{test_document['id']}")

        assert response.status_code == 204
        assert response.content == b""

        # Verify document is deleted
        get_response = await client.get(f"/api/v1/document/{test_document['id']}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, client: AsyncClient):
        """Test deleting non-existent document."""
        response = await client.delete("/api/v1/document/999")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document_invalid_id(self, client: AsyncClient):
        """Test deleting document with invalid ID."""
        response = await client.delete("/api/v1/document/invalid")

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_document_chunk_count_accuracy(
        self, client: AsyncClient, test_document: Dict[str, Any], db_session: AsyncSession
    ):
        """Test that document chunk count is accurate after adding chunks."""
        chunk_service = ChunkService()

        # Create chunks in the document
        chunk_data_list = [
            ChunkCreate(
                content=f"Test chunk {i}",
                embedding=[float(i), 0.0, 0.0],
                document_id=test_document["id"],
                metadata={"index": i},
            )
            for i in range(3)
        ]

        await chunk_service.create_chunks_bulk(chunk_data_list, db_session)

        # Check document chunk count
        response = await client.get(f"/api/v1/document/{test_document['id']}")

        assert response.status_code == 200
        data = response.json()
        assert data["chunk_count"] == 3

    @pytest.mark.asyncio
    async def test_concurrent_document_operations(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test concurrent document operations to validate thread safety."""
        import asyncio

        # Create multiple documents concurrently
        async def create_document(name: str):
            document_data = {
                "title": f"Concurrent Document {name}",
                "library_id": sample_library["id"],
                "metadata": {"concurrent": True},
            }
            try:
                response = await client.post("/api/v1/document/", json=document_data)
                return response.status_code, response.json() if response.status_code == 201 else response.text
            except Exception as e:
                return 500, str(e)

        # Create 5 documents concurrently
        tasks = [create_document(f"Test{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Debug: Print any failures
        for i, (status_code, data) in enumerate(results):
            if status_code != 201:
                print(f"Document Test{i} failed with status {status_code}: {data}")

        # All should succeed
        for status_code, data in results:
            assert status_code == 201, f"Expected 201, got {status_code}: {data}"
            assert "id" in data

        # Verify all documents exist in the library
        response = await client.get(f"/api/v1/document/?library_id={sample_library['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] >= 5
