"""API tests for Library endpoints."""

from typing import Any, Dict
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.document.services import DocumentService
from src.modules.document.schemas import DocumentCreate
from src.modules.chunk.services import ChunkService
from src.modules.chunk.schemas import ChunkCreate
from src.infrastructure.indexing.manager import index_manager


class TestLibraryAPI:
    """API tests for library endpoints."""

    @pytest.fixture(autouse=True)
    async def clear_index_manager(self):
        """Clear the global index manager before each test."""
        index_manager._indexes.clear()
        index_manager._index_types.clear()
        yield
        index_manager._indexes.clear()
        index_manager._index_types.clear()

    @pytest.mark.asyncio
    async def test_create_library_success(self, client: AsyncClient):
        """Test successful library creation."""
        library_data = {
            "name": "Test Library",
            "description": "A test library for API testing",
            "metadata": {"category": "test", "priority": 1},
        }

        response = await client.post("/api/v1/library/", json=library_data)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == library_data["name"]
        assert data["description"] == library_data["description"]
        assert data["metadata"] == library_data["metadata"]
        assert data["document_count"] == 0
        assert data["chunk_count"] == 0
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_create_library_minimal_data(self, client: AsyncClient):
        """Test library creation with minimal required data."""
        library_data = {"name": "Minimal Library"}

        response = await client.post("/api/v1/library/", json=library_data)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == library_data["name"]
        assert data["description"] is None
        assert data["metadata"] == {}
        assert data["document_count"] == 0
        assert data["chunk_count"] == 0

    @pytest.mark.asyncio
    async def test_create_library_validation_errors(self, client: AsyncClient):
        """Test library creation validation errors."""
        # Test missing name
        response = await client.post("/api/v1/library/", json={})
        assert response.status_code == 422

        # Test empty name
        response = await client.post("/api/v1/library/", json={"name": ""})
        assert response.status_code == 422

        # Test invalid metadata type
        response = await client.post("/api/v1/library/", json={"name": "Test", "metadata": "not a dict"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_library_success(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test successful library retrieval."""
        response = await client.get(f"/api/v1/library/{sample_library['id']}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_library["id"]
        assert data["name"] == sample_library["name"]
        assert data["description"] == sample_library["description"]
        assert data["metadata"] == sample_library["metadata"]
        assert data["document_count"] == 0
        assert data["chunk_count"] == 0

    @pytest.mark.asyncio
    async def test_get_library_not_found(self, client: AsyncClient):
        """Test library retrieval with non-existent ID."""
        response = await client.get("/api/v1/library/999")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_get_library_invalid_id(self, client: AsyncClient):
        """Test library retrieval with invalid ID."""
        response = await client.get("/api/v1/library/invalid")

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_libraries_empty(self, client: AsyncClient):
        """Test getting libraries when none exist."""
        response = await client.get("/api/v1/library/")

        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["total_count"] == 0
        assert data["page"] == 1
        assert data["items_per_page"] == 50

    @pytest.mark.asyncio
    async def test_get_libraries_with_data(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test getting libraries with existing data."""
        response = await client.get("/api/v1/library/")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["total_count"] == 1
        assert data["data"][0]["id"] == sample_library["id"]
        assert data["data"][0]["name"] == sample_library["name"]

    @pytest.mark.asyncio
    async def test_get_libraries_pagination(self, client: AsyncClient):
        """Test library pagination."""
        # Create multiple libraries
        for i in range(3):
            library_data = {"name": f"Test Library {i}"}
            await client.post("/api/v1/library/", json=library_data)

        # Test first page with page size 2
        response = await client.get("/api/v1/library/?page=1&items_per_page=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert data["total_count"] == 3
        assert data["page"] == 1
        assert data["items_per_page"] == 2

        # Test second page
        response = await client.get("/api/v1/library/?page=2&items_per_page=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["page"] == 2

    @pytest.mark.asyncio
    async def test_get_libraries_invalid_pagination(self, client: AsyncClient):
        """Test library pagination with invalid parameters."""
        # Test invalid page number
        response = await client.get("/api/v1/library/?page=0")
        assert response.status_code == 422

        # Test invalid items_per_page
        response = await client.get("/api/v1/library/?items_per_page=0")
        assert response.status_code == 422

        # Test negative values
        response = await client.get("/api/v1/library/?page=-1")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_update_library_success(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test successful library update."""
        update_data = {
            "name": "Updated Library Name",
            "description": "Updated description",
            "metadata": {"updated": True, "version": 2},
        }

        response = await client.put(f"/api/v1/library/{sample_library['id']}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_library["id"]
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]
        assert data["metadata"] == update_data["metadata"]

    @pytest.mark.asyncio
    async def test_update_library_partial(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test partial library update."""
        update_data = {"name": "Partially Updated Name"}

        response = await client.put(f"/api/v1/library/{sample_library['id']}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["description"] == sample_library["description"]  # Should remain unchanged
        assert data["metadata"] == sample_library["metadata"]  # Should remain unchanged

    @pytest.mark.asyncio
    async def test_update_library_not_found(self, client: AsyncClient):
        """Test updating non-existent library."""
        update_data = {"name": "Updated Name"}

        response = await client.put("/api/v1/library/999", json=update_data)

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_library_validation_errors(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test library update validation errors."""
        # Test empty name
        response = await client.put(f"/api/v1/library/{sample_library['id']}", json={"name": ""})
        assert response.status_code == 422

        # Test invalid metadata type
        response = await client.put(f"/api/v1/library/{sample_library['id']}", json={"metadata": "not a dict"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_library_success(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test successful library deletion."""
        response = await client.delete(f"/api/v1/library/{sample_library['id']}")

        assert response.status_code == 204
        assert response.content == b""

        # Verify library is deleted
        get_response = await client.get(f"/api/v1/library/{sample_library['id']}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_library_not_found(self, client: AsyncClient):
        """Test deleting non-existent library."""
        response = await client.delete("/api/v1/library/999")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_library_invalid_id(self, client: AsyncClient):
        """Test deleting library with invalid ID."""
        response = await client.delete("/api/v1/library/invalid")

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_vector_search_empty_library(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test vector search on empty library."""
        search_data = {"query_embedding": [1.0, 0.0, 0.0], "k": 5}

        response = await client.post(f"/api/v1/library/{sample_library['id']}/search", json=search_data)

        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total_chunks_searched"] == 0
        assert data["query_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_vector_search_with_chunks(
        self, client: AsyncClient, sample_library: Dict[str, Any], db_session: AsyncSession
    ):
        """Test vector search with actual chunks."""
        # Create test data
        document_service = DocumentService()
        chunk_service = ChunkService()

        document_data = DocumentCreate(
            title="Search Test Document",
            content="Content for search testing",
            library_id=sample_library["id"],
            metadata={"type": "test"},
        )
        document = await document_service.create_document(document_data, db_session)

        chunk_data_list = [
            ChunkCreate(
                content="First search chunk", embedding=[1.0, 0.0, 0.0], document_id=document.id, metadata={"priority": 1}
            ),
            ChunkCreate(
                content="Second search chunk", embedding=[0.0, 1.0, 0.0], document_id=document.id, metadata={"priority": 2}
            ),
        ]

        await chunk_service.create_chunks_bulk(chunk_data_list, db_session)

        # Perform search
        search_data = {"query_embedding": [1.0, 0.0, 0.0], "k": 2}

        response = await client.post(f"/api/v1/library/{sample_library['id']}/search", json=search_data)

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["total_chunks_searched"] == 2
        assert data["query_time_ms"] > 0

        # Check results are sorted by similarity
        similarities = [r["similarity_score"] for r in data["results"]]
        assert similarities == sorted(similarities, reverse=True)

        # First result should be the perfect match
        assert abs(data["results"][0]["similarity_score"] - 1.0) < 1e-6
        assert data["results"][0]["content"] == "First search chunk"

    @pytest.mark.asyncio
    async def test_vector_search_with_metadata_filter(
        self, client: AsyncClient, sample_library: Dict[str, Any], db_session: AsyncSession
    ):
        """Test vector search with metadata filtering."""
        # Create test data
        document_service = DocumentService()
        chunk_service = ChunkService()

        document_data = DocumentCreate(
            title="Filter Test Document", content="Content for filter testing", library_id=sample_library["id"], metadata={}
        )
        document = await document_service.create_document(document_data, db_session)

        chunk_data_list = [
            ChunkCreate(
                content="Primary chunk", embedding=[1.0, 0.0, 0.0], document_id=document.id, metadata={"category": "primary"}
            ),
            ChunkCreate(
                content="Secondary chunk",
                embedding=[0.9, 0.1, 0.0],
                document_id=document.id,
                metadata={"category": "secondary"},
            ),
        ]

        await chunk_service.create_chunks_bulk(chunk_data_list, db_session)

        # Search with metadata filter
        search_data = {"query_embedding": [1.0, 0.0, 0.0], "k": 10, "metadata_filter": {"category": "primary"}}

        response = await client.post(f"/api/v1/library/{sample_library['id']}/search", json=search_data)

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["metadata"]["category"] == "primary"
        assert data["results"][0]["content"] == "Primary chunk"

    @pytest.mark.asyncio
    async def test_vector_search_not_found(self, client: AsyncClient):
        """Test vector search on non-existent library."""
        search_data = {"query_embedding": [1.0, 0.0, 0.0], "k": 5}

        response = await client.post("/api/v1/library/999/search", json=search_data)

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_vector_search_validation_errors(self, client: AsyncClient, sample_library: Dict[str, Any]):
        """Test vector search validation errors."""
        # Test missing query_embedding
        response = await client.post(f"/api/v1/library/{sample_library['id']}/search", json={"k": 5})
        assert response.status_code == 422

        # Test missing k - should use default value of 10 and succeed
        response = await client.post(
            f"/api/v1/library/{sample_library['id']}/search", json={"query_embedding": [1.0, 0.0, 0.0]}
        )
        assert response.status_code == 200

        # Test invalid k value
        response = await client.post(
            f"/api/v1/library/{sample_library['id']}/search", json={"query_embedding": [1.0, 0.0, 0.0], "k": 0}
        )
        assert response.status_code == 422

        # Test empty embedding
        response = await client.post(f"/api/v1/library/{sample_library['id']}/search", json={"query_embedding": [], "k": 5})
        assert response.status_code == 422

        # Test invalid metadata_filter type
        response = await client.post(
            f"/api/v1/library/{sample_library['id']}/search",
            json={"query_embedding": [1.0, 0.0, 0.0], "k": 5, "metadata_filter": "not a dict"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_vector_search_k_parameter_limit(
        self, client: AsyncClient, sample_library: Dict[str, Any], db_session: AsyncSession
    ):
        """Test that k parameter correctly limits search results."""
        # Create test data with multiple chunks
        document_service = DocumentService()
        chunk_service = ChunkService()

        document_data = DocumentCreate(
            title="K Limit Test Document", content="Content for k limit testing", library_id=sample_library["id"], metadata={}
        )
        document = await document_service.create_document(document_data, db_session)

        # Create 5 chunks
        chunk_data_list = [
            ChunkCreate(
                content=f"Test chunk {i}", embedding=[1.0, float(i) * 0.1, 0.0], document_id=document.id, metadata={"index": i}
            )
            for i in range(5)
        ]

        await chunk_service.create_chunks_bulk(chunk_data_list, db_session)

        # Search with k=3
        search_data = {"query_embedding": [1.0, 0.0, 0.0], "k": 3}

        response = await client.post(f"/api/v1/library/{sample_library['id']}/search", json=search_data)

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3  # Should return exactly 3 results
        assert data["total_chunks_searched"] == 5  # But searched all 5 chunks

    @pytest.mark.asyncio
    async def test_library_counts_with_documents_and_chunks(
        self, client: AsyncClient, sample_library: Dict[str, Any], db_session: AsyncSession
    ):
        """Test that library counts are correct after adding documents and chunks."""
        document_service = DocumentService()
        chunk_service = ChunkService()

        # Create 2 documents
        doc1_data = DocumentCreate(title="Document 1", content="First document", library_id=sample_library["id"], metadata={})
        doc2_data = DocumentCreate(title="Document 2", content="Second document", library_id=sample_library["id"], metadata={})

        doc1 = await document_service.create_document(doc1_data, db_session)
        doc2 = await document_service.create_document(doc2_data, db_session)

        # Create 3 chunks total (2 in doc1, 1 in doc2)
        chunk_data_list = [
            ChunkCreate(content="Chunk 1 in doc 1", embedding=[1.0, 0.0, 0.0], document_id=doc1.id, metadata={}),
            ChunkCreate(content="Chunk 2 in doc 1", embedding=[0.0, 1.0, 0.0], document_id=doc1.id, metadata={}),
            ChunkCreate(content="Chunk 1 in doc 2", embedding=[0.0, 0.0, 1.0], document_id=doc2.id, metadata={}),
        ]

        await chunk_service.create_chunks_bulk(chunk_data_list, db_session)

        # Check library counts
        response = await client.get(f"/api/v1/library/{sample_library['id']}")

        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 2
        assert data["chunk_count"] == 3

    @pytest.mark.asyncio
    async def test_concurrent_library_operations(self, client: AsyncClient):
        """Test concurrent library operations to validate thread safety."""
        import asyncio

        # Create multiple libraries concurrently
        async def create_library(name: str):
            library_data = {"name": f"Concurrent Library {name}"}
            try:
                response = await client.post("/api/v1/library/", json=library_data)
                return response.status_code, response.json() if response.status_code == 201 else response.text
            except Exception as e:
                return 500, str(e)

        # Create 5 libraries concurrently
        tasks = [create_library(f"Test{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Debug: Print any failures
        successful_results = []
        for i, (status_code, data) in enumerate(results):
            if status_code != 201:
                print(f"Library Test{i} failed with status {status_code}: {data}")
            else:
                successful_results.append(data)

        # All should succeed
        for status_code, data in results:
            assert status_code == 201, f"Expected 201, got {status_code}: {data}"
            assert "id" in data

        # Verify all libraries exist
        response = await client.get("/api/v1/library/")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] >= 5
