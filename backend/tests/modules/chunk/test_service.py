"""Tests for chunk service."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.chunk.services import ChunkService
from src.modules.chunk.schemas import ChunkCreate, ChunkUpdate


@pytest.fixture
def chunk_service():
    """Create chunk service instance."""
    return ChunkService()


@pytest.mark.asyncio
async def test_create_chunk(chunk_service: ChunkService, db_session: AsyncSession, test_document: dict):
    """Test creating a new chunk."""
    chunk_data = ChunkCreate(
        document_id=test_document["id"],
        content="This is a test chunk for vector search.",
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5] * 100,  # 500-dim vector
        metadata={"position": 0, "section": "intro"},
    )

    result = await chunk_service.create_chunk(chunk_data=chunk_data, db=db_session)

    assert result.document_id == test_document["id"]
    assert result.content == "This is a test chunk for vector search."
    assert len(result.embedding) == 500
    assert result.metadata == {"position": 0, "section": "intro"}
    assert result.id is not None


@pytest.mark.asyncio
async def test_create_chunk_document_not_found(chunk_service: ChunkService, db_session: AsyncSession):
    """Test creating chunk with non-existent document."""
    chunk_data = ChunkCreate(document_id=99999, content="Test content", embedding=[0.1] * 100)

    result = await chunk_service.create_chunk(chunk_data=chunk_data, db=db_session)
    assert result is None


@pytest.mark.asyncio
async def test_create_chunks_bulk(chunk_service: ChunkService, db_session: AsyncSession, test_document: dict):
    """Test creating multiple chunks in bulk."""
    chunks_data = [
        ChunkCreate(
            document_id=test_document["id"], content="First chunk content", embedding=[0.1] * 100, metadata={"position": 0}
        ),
        ChunkCreate(
            document_id=test_document["id"], content="Second chunk content", embedding=[0.2] * 100, metadata={"position": 1}
        ),
        ChunkCreate(
            document_id=test_document["id"], content="Third chunk content", embedding=[0.3] * 100, metadata={"position": 2}
        ),
    ]

    result = await chunk_service.create_chunks_bulk(chunks_data=chunks_data, db=db_session)

    assert len(result) == 3
    for i, chunk in enumerate(result):
        assert chunk.document_id == test_document["id"]
        assert f"{['First', 'Second', 'Third'][i]} chunk content" in chunk.content
        assert chunk.metadata["position"] == i


@pytest.mark.asyncio
async def test_create_chunks_bulk_empty_list(chunk_service: ChunkService, db_session: AsyncSession):
    """Test creating bulk chunks with empty list."""
    result = await chunk_service.create_chunks_bulk(chunks_data=[], db=db_session)
    assert result == []


@pytest.mark.asyncio
async def test_create_chunks_bulk_invalid_document(chunk_service: ChunkService, db_session: AsyncSession):
    """Test creating bulk chunks with non-existent document."""
    chunks_data = [ChunkCreate(document_id=99999, content="Test content", embedding=[0.1] * 100)]

    result = await chunk_service.create_chunks_bulk(chunks_data=chunks_data, db=db_session)
    assert result == []


@pytest.mark.asyncio
async def test_get_chunk(chunk_service: ChunkService, db_session: AsyncSession, test_chunk: dict):
    """Test getting a specific chunk."""
    result = await chunk_service.get_chunk(chunk_id=test_chunk["id"], db=db_session)

    assert result.id == test_chunk["id"]
    assert result.document_id == test_chunk["document_id"]
    assert result.content == test_chunk["content"]
    assert result.embedding == test_chunk["embedding"]
    assert result.metadata == test_chunk["metadata"]


@pytest.mark.asyncio
async def test_get_chunk_not_found(chunk_service: ChunkService, db_session: AsyncSession):
    """Test getting non-existent chunk."""
    result = await chunk_service.get_chunk(chunk_id=99999, db=db_session)
    assert result is None


@pytest.mark.asyncio
async def test_get_chunks_by_document(
    chunk_service: ChunkService, db_session: AsyncSession, test_document: dict, test_chunk: dict, test_chunk_2: dict
):
    """Test getting chunks by document with pagination."""
    result = await chunk_service.get_chunks_by_document(
        document_id=test_document["id"], db=db_session, page=1, items_per_page=10
    )

    assert "data" in result
    assert "total_count" in result
    assert result["total_count"] >= 2
    assert len(result["data"]) >= 2

    # Check that all chunks belong to the document
    for chunk in result["data"]:
        assert chunk["document_id"] == test_document["id"]


@pytest.mark.asyncio
async def test_get_chunks_by_document_not_found(chunk_service: ChunkService, db_session: AsyncSession):
    """Test getting chunks from non-existent document."""
    result = await chunk_service.get_chunks_by_document(document_id=99999, db=db_session, page=1, items_per_page=10)

    assert result["data"] == []
    assert result["total_count"] == 0
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_get_chunks_by_document_pagination(
    chunk_service: ChunkService, db_session: AsyncSession, test_document: dict, test_chunk: dict, test_chunk_2: dict
):
    """Test pagination for chunks by document."""
    result = await chunk_service.get_chunks_by_document(
        document_id=test_document["id"], db=db_session, page=1, items_per_page=1
    )

    assert len(result["data"]) == 1
    assert result["total_count"] >= 2
    assert result["has_more"] is True


@pytest.mark.asyncio
async def test_get_chunks(chunk_service: ChunkService, db_session: AsyncSession, test_chunk: dict, test_chunk_2: dict):
    """Test getting all chunks with pagination."""
    result = await chunk_service.get_chunks(db=db_session, page=1, items_per_page=10)

    assert "data" in result
    assert "total_count" in result
    assert result["total_count"] >= 2
    assert len(result["data"]) >= 2


@pytest.mark.asyncio
async def test_update_chunk(chunk_service: ChunkService, db_session: AsyncSession, test_chunk: dict):
    """Test updating a chunk."""
    update_data = ChunkUpdate(
        content="Updated chunk content", embedding=[0.9] * 500, metadata={"position": 99, "section": "updated"}
    )

    result = await chunk_service.update_chunk(chunk_id=test_chunk["id"], update_data=update_data, db=db_session)

    assert result.content == "Updated chunk content"
    assert result.embedding == [0.9] * 500
    assert result.metadata == {"position": 99, "section": "updated"}
    assert result.id == test_chunk["id"]
    assert result.document_id == test_chunk["document_id"]


@pytest.mark.asyncio
async def test_update_chunk_partial(chunk_service: ChunkService, db_session: AsyncSession, test_chunk: dict):
    """Test partial update of a chunk."""
    update_data = ChunkUpdate(content="Partially updated content")

    result = await chunk_service.update_chunk(chunk_id=test_chunk["id"], update_data=update_data, db=db_session)

    assert result.content == "Partially updated content"
    assert result.embedding == test_chunk["embedding"]  # Should remain unchanged
    assert result.metadata == test_chunk["metadata"]  # Should remain unchanged
    assert result.id == test_chunk["id"]


@pytest.mark.asyncio
async def test_delete_chunk(chunk_service: ChunkService, db_session: AsyncSession, test_chunk: dict):
    """Test deleting a chunk."""
    result = await chunk_service.delete_chunk(chunk_id=test_chunk["id"], db=db_session)

    assert result is True

    # Verify chunk is deleted
    deleted_chunk = await chunk_service.get_chunk(chunk_id=test_chunk["id"], db=db_session)
    assert deleted_chunk is None


@pytest.mark.asyncio
async def test_delete_chunks_by_document(
    chunk_service: ChunkService, db_session: AsyncSession, test_document: dict, test_chunk: dict, test_chunk_2: dict
):
    """Test deleting all chunks in a document."""
    # Verify chunks exist before deletion
    chunk1 = await chunk_service.get_chunk(chunk_id=test_chunk["id"], db=db_session)
    chunk2 = await chunk_service.get_chunk(chunk_id=test_chunk_2["id"], db=db_session)

    assert chunk1 is not None
    assert chunk2 is not None

    # Delete all chunks in document
    result = await chunk_service.delete_chunks_by_document(document_id=test_document["id"], db=db_session)

    assert result is True

    # Verify all chunks are deleted
    deleted_chunk1 = await chunk_service.get_chunk(chunk_id=test_chunk["id"], db=db_session)
    deleted_chunk2 = await chunk_service.get_chunk(chunk_id=test_chunk_2["id"], db=db_session)

    assert deleted_chunk1 is None
    assert deleted_chunk2 is None


@pytest.mark.asyncio
async def test_get_chunks_by_ids(chunk_service: ChunkService, db_session: AsyncSession, test_chunk: dict, test_chunk_2: dict):
    """Test getting multiple chunks by their IDs."""
    chunk_ids = [test_chunk["id"], test_chunk_2["id"]]

    result = await chunk_service.get_chunks_by_ids(chunk_ids=chunk_ids, db=db_session)

    assert len(result) == 2
    result_ids = [chunk.id for chunk in result]
    assert test_chunk["id"] in result_ids
    assert test_chunk_2["id"] in result_ids


@pytest.mark.asyncio
async def test_get_chunks_by_ids_empty_list(chunk_service: ChunkService, db_session: AsyncSession):
    """Test getting chunks with empty ID list."""
    result = await chunk_service.get_chunks_by_ids(chunk_ids=[], db=db_session)
    assert result == []


@pytest.mark.asyncio
async def test_get_chunks_by_ids_non_existent(chunk_service: ChunkService, db_session: AsyncSession):
    """Test getting chunks with non-existent IDs."""
    result = await chunk_service.get_chunks_by_ids(chunk_ids=[99999, 99998], db=db_session)
    assert result == []


@pytest.mark.asyncio
async def test_create_chunk_with_minimal_data(chunk_service: ChunkService, db_session: AsyncSession, test_document: dict):
    """Test creating a chunk with minimal required data."""
    chunk_data = ChunkCreate(document_id=test_document["id"], content="Minimal chunk", embedding=[0.1] * 100)

    result = await chunk_service.create_chunk(chunk_data=chunk_data, db=db_session)

    assert result.document_id == test_document["id"]
    assert result.content == "Minimal chunk"
    assert len(result.embedding) == 100
    assert result.metadata == {}


@pytest.mark.asyncio
async def test_get_chunks_empty_result(chunk_service: ChunkService, db_session: AsyncSession):
    """Test getting chunks when none exist."""
    result = await chunk_service.get_chunks(db=db_session, page=1, items_per_page=10)

    assert result["data"] == []
    assert result["total_count"] == 0
    assert result["has_more"] is False
