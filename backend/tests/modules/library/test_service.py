"""Tests for library service."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.library.services import LibraryService
from src.modules.library.schemas import LibraryCreate, LibraryUpdate


@pytest.fixture
def library_service():
    """Create library service instance."""
    return LibraryService()


@pytest.mark.asyncio
async def test_create_library(library_service: LibraryService, db_session: AsyncSession):
    """Test creating a new library."""
    library_data = LibraryCreate(
        name="Test Library", description="A test library", metadata={"type": "test", "features": ["search"]}
    )

    result = await library_service.create_library(library_data=library_data, db=db_session)

    assert result.name == "Test Library"
    assert result.description == "A test library"
    assert result.metadata == {"type": "test", "features": ["search"]}
    assert result.document_count == 0
    assert result.chunk_count == 0
    assert result.id is not None


@pytest.mark.asyncio
async def test_get_library(library_service: LibraryService, db_session: AsyncSession, test_library: dict):
    """Test getting a specific library."""
    result = await library_service.get_library(library_id=test_library["id"], db=db_session)

    assert result.id == test_library["id"]
    assert result.name == test_library["name"]
    assert result.description == test_library["description"]
    assert result.metadata == test_library["metadata"]
    assert result.document_count == 0
    assert result.chunk_count == 0


@pytest.mark.asyncio
async def test_get_library_with_counts(
    library_service: LibraryService, db_session: AsyncSession, test_library: dict, test_document: dict, test_chunk: dict
):
    """Test getting a library with document and chunk counts."""
    result = await library_service.get_library(library_id=test_library["id"], db=db_session)

    assert result.id == test_library["id"]
    assert result.document_count == 1
    assert result.chunk_count == 1


@pytest.mark.asyncio
async def test_get_library_not_found(library_service: LibraryService, db_session: AsyncSession):
    """Test getting non-existent library."""
    result = await library_service.get_library(library_id=99999, db=db_session)
    assert result is None


@pytest.mark.asyncio
async def test_get_libraries_pagination(
    library_service: LibraryService, db_session: AsyncSession, test_library: dict, test_library_2: dict
):
    """Test getting libraries with pagination."""
    result = await library_service.get_libraries(db=db_session, page=1, items_per_page=10)

    assert "data" in result
    assert "total_count" in result
    assert "has_more" in result
    assert result["total_count"] >= 2
    assert len(result["data"]) >= 2


@pytest.mark.asyncio
async def test_get_libraries_with_limit(
    library_service: LibraryService, db_session: AsyncSession, test_library: dict, test_library_2: dict
):
    """Test getting libraries with pagination limit."""
    result = await library_service.get_libraries(db=db_session, page=1, items_per_page=1)

    assert "data" in result
    assert len(result["data"]) == 1
    assert result["total_count"] >= 2
    assert result["has_more"] is True


@pytest.mark.asyncio
async def test_update_library(library_service: LibraryService, db_session: AsyncSession, test_library: dict):
    """Test updating a library."""
    update_data = LibraryUpdate(
        name="Updated Library", description="Updated description", metadata={"type": "updated", "new_feature": True}
    )

    result = await library_service.update_library(library_id=test_library["id"], update_data=update_data, db=db_session)

    assert result.name == "Updated Library"
    assert result.description == "Updated description"
    assert result.metadata == {"type": "updated", "new_feature": True}
    assert result.id == test_library["id"]


@pytest.mark.asyncio
async def test_update_library_partial(library_service: LibraryService, db_session: AsyncSession, test_library: dict):
    """Test partial update of a library."""
    update_data = LibraryUpdate(name="Partially Updated")

    result = await library_service.update_library(library_id=test_library["id"], update_data=update_data, db=db_session)

    assert result.name == "Partially Updated"
    assert result.description == test_library["description"]  # Should remain unchanged
    assert result.id == test_library["id"]


@pytest.mark.asyncio
async def test_delete_library(library_service: LibraryService, db_session: AsyncSession, test_library: dict):
    """Test deleting a library."""
    result = await library_service.delete_library(library_id=test_library["id"], db=db_session)

    assert result is True

    # Verify library is deleted
    deleted_library = await library_service.get_library(library_id=test_library["id"], db=db_session)
    assert deleted_library is None


@pytest.mark.asyncio
async def test_delete_library_cascades_to_documents_and_chunks(
    library_service: LibraryService, db_session: AsyncSession, test_library: dict, test_document: dict, test_chunk: dict
):
    """Test that deleting a library cascades to documents and chunks."""
    from src.modules.document.services import DocumentService
    from src.modules.chunk.services import ChunkService

    document_service = DocumentService()
    chunk_service = ChunkService()

    # Verify entities exist before deletion
    library = await library_service.get_library(library_id=test_library["id"], db=db_session)
    document = await document_service.get_document(document_id=test_document["id"], db=db_session)
    chunk = await chunk_service.get_chunk(chunk_id=test_chunk["id"], db=db_session)

    assert library is not None
    assert document is not None
    assert chunk is not None

    # Delete library
    await library_service.delete_library(library_id=test_library["id"], db=db_session)

    # Verify cascade deletion
    deleted_library = await library_service.get_library(library_id=test_library["id"], db=db_session)
    deleted_document = await document_service.get_document(document_id=test_document["id"], db=db_session)
    deleted_chunk = await chunk_service.get_chunk(chunk_id=test_chunk["id"], db=db_session)

    assert deleted_library is None
    assert deleted_document is None
    assert deleted_chunk is None


@pytest.mark.asyncio
async def test_create_library_with_minimal_data(library_service: LibraryService, db_session: AsyncSession):
    """Test creating a library with minimal required data."""
    library_data = LibraryCreate(name="Minimal Library")

    result = await library_service.create_library(library_data=library_data, db=db_session)

    assert result.name == "Minimal Library"
    assert result.description is None
    assert result.metadata == {}
    assert result.document_count == 0
    assert result.chunk_count == 0


@pytest.mark.asyncio
async def test_get_libraries_empty_result(library_service: LibraryService, db_session: AsyncSession):
    """Test getting libraries when none exist."""
    result = await library_service.get_libraries(db=db_session, page=1, items_per_page=10)

    assert "data" in result
    assert result["data"] == []
    assert result["total_count"] == 0
    assert result["has_more"] is False
