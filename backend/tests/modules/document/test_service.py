"""Tests for document service."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.document.services import DocumentService
from src.modules.document.schemas import DocumentCreate, DocumentUpdate


@pytest.fixture
def document_service():
    """Create document service instance."""
    return DocumentService()


@pytest.mark.asyncio
async def test_create_document(document_service: DocumentService, db_session: AsyncSession, test_library: dict):
    """Test creating a new document."""
    document_data = DocumentCreate(
        library_id=test_library["id"], title="Test Document", metadata={"source": "test", "pages": 10}
    )

    result = await document_service.create_document(document_data=document_data, db=db_session)

    assert result.library_id == test_library["id"]
    assert result.title == "Test Document"
    assert result.metadata == {"source": "test", "pages": 10}
    assert result.chunk_count == 0
    assert result.id is not None


@pytest.mark.asyncio
async def test_create_document_library_not_found(document_service: DocumentService, db_session: AsyncSession):
    """Test creating document with non-existent library."""
    document_data = DocumentCreate(library_id=99999, title="Test Document")

    result = await document_service.create_document(document_data=document_data, db=db_session)
    assert result is None


@pytest.mark.asyncio
async def test_get_document(document_service: DocumentService, db_session: AsyncSession, test_document: dict):
    """Test getting a specific document."""
    result = await document_service.get_document(document_id=test_document["id"], db=db_session)

    assert result.id == test_document["id"]
    assert result.library_id == test_document["library_id"]
    assert result.title == test_document["title"]
    assert result.metadata == test_document["metadata"]
    assert result.chunk_count == 0


@pytest.mark.asyncio
async def test_get_document_with_chunks(
    document_service: DocumentService, db_session: AsyncSession, test_document: dict, test_chunk: dict, test_chunk_2: dict
):
    """Test getting a document with chunk count."""
    result = await document_service.get_document(document_id=test_document["id"], db=db_session)

    assert result.id == test_document["id"]
    assert result.chunk_count == 2


@pytest.mark.asyncio
async def test_get_document_not_found(document_service: DocumentService, db_session: AsyncSession):
    """Test getting non-existent document."""
    result = await document_service.get_document(document_id=99999, db=db_session)
    assert result is None


@pytest.mark.asyncio
async def test_get_documents_by_library(
    document_service: DocumentService, db_session: AsyncSession, test_library: dict, test_document: dict, test_document_2: dict
):
    """Test getting documents by library with pagination."""
    result = await document_service.get_documents_by_library(
        library_id=test_library["id"], db=db_session, page=1, items_per_page=10
    )

    assert "data" in result
    assert "total_count" in result
    assert result["total_count"] >= 2
    assert len(result["data"]) >= 2

    # Check that all documents belong to the library
    for doc in result["data"]:
        assert doc["library_id"] == test_library["id"]


@pytest.mark.asyncio
async def test_get_documents_by_library_not_found(document_service: DocumentService, db_session: AsyncSession):
    """Test getting documents from non-existent library."""
    result = await document_service.get_documents_by_library(library_id=99999, db=db_session, page=1, items_per_page=10)

    assert result["data"] == []
    assert result["total_count"] == 0
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_get_documents_by_library_pagination(
    document_service: DocumentService, db_session: AsyncSession, test_library: dict, test_document: dict, test_document_2: dict
):
    """Test pagination for documents by library."""
    result = await document_service.get_documents_by_library(
        library_id=test_library["id"], db=db_session, page=1, items_per_page=1
    )

    assert len(result["data"]) == 1
    assert result["total_count"] >= 2
    assert result["has_more"] is True


@pytest.mark.asyncio
async def test_get_documents(
    document_service: DocumentService, db_session: AsyncSession, test_document: dict, test_document_2: dict
):
    """Test getting all documents with pagination."""
    result = await document_service.get_documents(db=db_session, page=1, items_per_page=10)

    assert "data" in result
    assert "total_count" in result
    assert result["total_count"] >= 2
    assert len(result["data"]) >= 2


@pytest.mark.asyncio
async def test_update_document(document_service: DocumentService, db_session: AsyncSession, test_document: dict):
    """Test updating a document."""
    update_data = DocumentUpdate(title="Updated Document", metadata={"source": "updated", "pages": 20})

    result = await document_service.update_document(document_id=test_document["id"], update_data=update_data, db=db_session)

    assert result.title == "Updated Document"
    assert result.metadata == {"source": "updated", "pages": 20}
    assert result.id == test_document["id"]
    assert result.library_id == test_document["library_id"]


@pytest.mark.asyncio
async def test_update_document_partial(document_service: DocumentService, db_session: AsyncSession, test_document: dict):
    """Test partial update of a document."""
    update_data = DocumentUpdate(title="Partially Updated")

    result = await document_service.update_document(document_id=test_document["id"], update_data=update_data, db=db_session)

    assert result.title == "Partially Updated"
    assert result.metadata == test_document["metadata"]  # Should remain unchanged
    assert result.id == test_document["id"]


@pytest.mark.asyncio
async def test_delete_document(document_service: DocumentService, db_session: AsyncSession, test_document: dict):
    """Test deleting a document."""
    result = await document_service.delete_document(document_id=test_document["id"], db=db_session)

    assert result is True

    # Verify document is deleted
    deleted_document = await document_service.get_document(document_id=test_document["id"], db=db_session)
    assert deleted_document is None


@pytest.mark.asyncio
async def test_delete_document_cascades_to_chunks(
    document_service: DocumentService, db_session: AsyncSession, test_document: dict, test_chunk: dict
):
    """Test that deleting a document cascades to chunks."""
    from src.modules.chunk.services import ChunkService

    chunk_service = ChunkService()

    # Verify entities exist before deletion
    document = await document_service.get_document(document_id=test_document["id"], db=db_session)
    chunk = await chunk_service.get_chunk(chunk_id=test_chunk["id"], db=db_session)

    assert document is not None
    assert chunk is not None

    # Delete document
    await document_service.delete_document(document_id=test_document["id"], db=db_session)

    # Verify cascade deletion
    deleted_document = await document_service.get_document(document_id=test_document["id"], db=db_session)
    deleted_chunk = await chunk_service.get_chunk(chunk_id=test_chunk["id"], db=db_session)

    assert deleted_document is None
    assert deleted_chunk is None


@pytest.mark.asyncio
async def test_create_document_with_minimal_data(
    document_service: DocumentService, db_session: AsyncSession, test_library: dict
):
    """Test creating a document with minimal required data."""
    document_data = DocumentCreate(library_id=test_library["id"], title="Minimal Document")

    result = await document_service.create_document(document_data=document_data, db=db_session)

    assert result.library_id == test_library["id"]
    assert result.title == "Minimal Document"
    assert result.metadata == {}
    assert result.chunk_count == 0


@pytest.mark.asyncio
async def test_get_documents_empty_result(document_service: DocumentService, db_session: AsyncSession):
    """Test getting documents when none exist."""
    result = await document_service.get_documents(db=db_session, page=1, items_per_page=10)

    assert result["data"] == []
    assert result["total_count"] == 0
    assert result["has_more"] is False
