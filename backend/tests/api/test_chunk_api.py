"""API tests for Chunk endpoints."""

from typing import Any, Dict
import pytest
from httpx import AsyncClient

from src.infrastructure.indexing.manager import index_manager


class TestChunkAPI:
    """API tests for chunk endpoints."""

    @pytest.fixture(autouse=True)
    async def clear_index_manager(self):
        """Clear the global index manager before each test."""
        index_manager._indexes.clear()
        index_manager._index_types.clear()
        yield
        index_manager._indexes.clear()
        index_manager._index_types.clear()

    @pytest.mark.asyncio
    async def test_create_chunk_success(self, client: AsyncClient, test_document: Dict[str, Any]):
        """Test successful chunk creation."""
        chunk_data = {
            "content": "Test chunk content for vector search",
            "embedding": [1.0, 0.5, 0.0, 0.3, 0.8],
            "document_id": test_document["id"],
            "metadata": {"section": "introduction", "position": 0},
        }

        response = await client.post("/api/v1/chunk/", json=chunk_data)

        assert response.status_code == 201
        data = response.json()
        assert data["content"] == chunk_data["content"]
        assert data["embedding"] == chunk_data["embedding"]
        assert data["document_id"] == chunk_data["document_id"]
        assert data["metadata"] == chunk_data["metadata"]
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_create_chunk_minimal_data(self, client: AsyncClient, test_document: Dict[str, Any]):
        """Test chunk creation with minimal required data."""
        chunk_data = {"content": "Minimal chunk content", "embedding": [0.1, 0.2, 0.3], "document_id": test_document["id"]}

        response = await client.post("/api/v1/chunk/", json=chunk_data)

        assert response.status_code == 201
        data = response.json()
        assert data["content"] == chunk_data["content"]
        assert data["embedding"] == chunk_data["embedding"]
        assert data["document_id"] == chunk_data["document_id"]
        assert data["metadata"] == {}

    @pytest.mark.asyncio
    async def test_create_chunk_invalid_document(self, client: AsyncClient):
        """Test chunk creation with non-existent document."""
        chunk_data = {
            "content": "Test chunk content",
            "embedding": [1.0, 0.0, 0.0],
            "document_id": 999,
            "metadata": {"type": "test"},
        }

        response = await client.post("/api/v1/chunk/", json=chunk_data)

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_create_chunk_validation_errors(self, client: AsyncClient, test_document: Dict[str, Any]):
        """Test chunk creation validation errors."""
        # Test missing content
        response = await client.post("/api/v1/chunk/", json={"embedding": [1.0, 0.0, 0.0], "document_id": test_document["id"]})
        assert response.status_code == 422

        # Test empty content
        response = await client.post(
            "/api/v1/chunk/", json={"content": "", "embedding": [1.0, 0.0, 0.0], "document_id": test_document["id"]}
        )
        assert response.status_code == 422

        # Test missing embedding
        response = await client.post("/api/v1/chunk/", json={"content": "Test content", "document_id": test_document["id"]})
        assert response.status_code == 422

        # Test empty embedding
        response = await client.post(
            "/api/v1/chunk/", json={"content": "Test content", "embedding": [], "document_id": test_document["id"]}
        )
        assert response.status_code == 422

        # Test missing document_id
        response = await client.post("/api/v1/chunk/", json={"content": "Test content", "embedding": [1.0, 0.0, 0.0]})
        assert response.status_code == 422

        # Test oversized embedding
        large_embedding = [1.0] * 5000  # Exceeds 4096 limit
        response = await client.post(
            "/api/v1/chunk/", json={"content": "Test content", "embedding": large_embedding, "document_id": test_document["id"]}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_chunk_success(self, client: AsyncClient, test_chunk: Dict[str, Any]):
        """Test successful chunk retrieval."""
        response = await client.get(f"/api/v1/chunk/{test_chunk['id']}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_chunk["id"]
        assert data["content"] == test_chunk["content"]
        assert data["embedding"] == test_chunk["embedding"]
        assert data["document_id"] == test_chunk["document_id"]
        assert data["metadata"] == test_chunk["metadata"]

    @pytest.mark.asyncio
    async def test_get_chunk_not_found(self, client: AsyncClient):
        """Test chunk retrieval with non-existent ID."""
        response = await client.get("/api/v1/chunk/999")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_get_chunk_invalid_id(self, client: AsyncClient):
        """Test chunk retrieval with invalid ID."""
        response = await client.get("/api/v1/chunk/invalid")

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_chunks_empty(self, client: AsyncClient):
        """Test getting chunks when none exist."""
        response = await client.get("/api/v1/chunk/")

        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["total_count"] == 0
        assert data["page"] == 1
        assert data["items_per_page"] == 50

    @pytest.mark.asyncio
    async def test_get_chunks_with_data(self, client: AsyncClient, test_chunk: Dict[str, Any]):
        """Test getting chunks with existing data."""
        response = await client.get("/api/v1/chunk/")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["total_count"] == 1
        assert data["data"][0]["id"] == test_chunk["id"]
        assert data["data"][0]["content"] == test_chunk["content"]

    @pytest.mark.asyncio
    async def test_get_chunks_by_document(self, client: AsyncClient, test_chunk: Dict[str, Any], test_document: Dict[str, Any]):
        """Test getting chunks filtered by document."""
        # Create another chunk in the same document
        chunk_data = {
            "content": "Second chunk in document",
            "embedding": [0.0, 1.0, 0.0],
            "document_id": test_document["id"],
            "metadata": {"section": "body"},
        }
        response = await client.post("/api/v1/chunk/", json=chunk_data)
        assert response.status_code == 201

        # Get chunks from this document only
        response = await client.get(f"/api/v1/chunk/?document_id={test_document['id']}")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2  # Original test_chunk + new chunk
        for chunk in data["data"]:
            assert chunk["document_id"] == test_document["id"]

    @pytest.mark.asyncio
    async def test_get_chunks_pagination(self, client: AsyncClient, test_document: Dict[str, Any]):
        """Test chunk pagination."""
        # Create multiple chunks
        for i in range(3):
            chunk_data = {
                "content": f"Test chunk {i}",
                "embedding": [float(i), 0.0, 0.0],
                "document_id": test_document["id"],
                "metadata": {"index": i},
            }
            await client.post("/api/v1/chunk/", json=chunk_data)

        # Test first page with page size 2
        response = await client.get("/api/v1/chunk/?page=1&items_per_page=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert data["total_count"] == 3
        assert data["page"] == 1
        assert data["items_per_page"] == 2

        # Test second page
        response = await client.get("/api/v1/chunk/?page=2&items_per_page=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["page"] == 2

    @pytest.mark.asyncio
    async def test_update_chunk_success(self, client: AsyncClient, test_chunk: Dict[str, Any]):
        """Test successful chunk update."""
        update_data = {
            "content": "Updated chunk content",
            "embedding": [0.9, 0.1, 0.0, 0.5, 0.3],
            "metadata": {"updated": True, "version": 2},
        }

        response = await client.put(f"/api/v1/chunk/{test_chunk['id']}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_chunk["id"]
        assert data["content"] == update_data["content"]
        assert data["embedding"] == update_data["embedding"]
        assert data["metadata"] == update_data["metadata"]

    @pytest.mark.asyncio
    async def test_update_chunk_partial(self, client: AsyncClient, test_chunk: Dict[str, Any]):
        """Test partial chunk update."""
        update_data = {"content": "Partially updated content"}

        response = await client.put(f"/api/v1/chunk/{test_chunk['id']}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == update_data["content"]
        assert data["embedding"] == test_chunk["embedding"]  # Should remain unchanged
        assert data["metadata"] == test_chunk["metadata"]  # Should remain unchanged

    @pytest.mark.asyncio
    async def test_update_chunk_not_found(self, client: AsyncClient):
        """Test updating non-existent chunk."""
        update_data = {"content": "Updated content"}

        response = await client.put("/api/v1/chunk/999", json=update_data)

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_chunk_validation_errors(self, client: AsyncClient, test_chunk: Dict[str, Any]):
        """Test chunk update validation errors."""
        # Test empty content
        response = await client.put(f"/api/v1/chunk/{test_chunk['id']}", json={"content": ""})
        assert response.status_code == 422

        # Test empty embedding
        response = await client.put(f"/api/v1/chunk/{test_chunk['id']}", json={"embedding": []})
        assert response.status_code == 422

        # Test oversized embedding
        large_embedding = [1.0] * 5000
        response = await client.put(f"/api/v1/chunk/{test_chunk['id']}", json={"embedding": large_embedding})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_chunk_success(self, client: AsyncClient, test_chunk: Dict[str, Any]):
        """Test successful chunk deletion."""
        response = await client.delete(f"/api/v1/chunk/{test_chunk['id']}")

        assert response.status_code == 204
        assert response.content == b""

        # Verify chunk is deleted
        get_response = await client.get(f"/api/v1/chunk/{test_chunk['id']}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_chunk_not_found(self, client: AsyncClient):
        """Test deleting non-existent chunk."""
        response = await client.delete("/api/v1/chunk/999")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_chunk_invalid_id(self, client: AsyncClient):
        """Test deleting chunk with invalid ID."""
        response = await client.delete("/api/v1/chunk/invalid")

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_concurrent_chunk_operations(self, client: AsyncClient, test_document: Dict[str, Any]):
        """Test concurrent chunk operations to validate thread safety."""
        import asyncio

        # Create multiple chunks concurrently
        async def create_chunk(index: int):
            chunk_data = {
                "content": f"Concurrent chunk content {index}",
                "embedding": [float(index), 0.0, 0.0],
                "document_id": test_document["id"],
                "metadata": {"concurrent": True, "index": index},
            }
            try:
                response = await client.post("/api/v1/chunk/", json=chunk_data)
                return response.status_code, response.json() if response.status_code == 201 else response.text
            except Exception as e:
                return 500, str(e)

        # Create 5 chunks concurrently
        tasks = [create_chunk(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Debug: Print any failures
        for i, (status_code, data) in enumerate(results):
            if status_code != 201:
                print(f"Chunk {i} failed with status {status_code}: {data}")

        # All should succeed
        for status_code, data in results:
            assert status_code == 201, f"Expected 201, got {status_code}: {data}"
            assert "id" in data

        # Verify all chunks exist in the document
        response = await client.get(f"/api/v1/chunk/?document_id={test_document['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] >= 5
