"""Chunk API endpoints."""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from ....infrastructure.database.session import async_session
from ....modules.chunk.schemas import (
    ChunkCreate,
    ChunkRead,
    ChunkUpdate,
)
from ....modules.chunk.services import ChunkService
from ....modules.common.exceptions import ValidationError
from ....modules.common.utils.error_handler import handle_exception

router = APIRouter(prefix="/chunk", tags=["Chunks"])


def get_chunk_service() -> ChunkService:
    """Dependency to get chunk service instance."""
    return ChunkService()


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create New Chunk",
    description="""
    Creates a new chunk within a document.

    Chunks are the fundamental units of content in the vector database.
    Each chunk contains text content and its vector embedding for similarity search.

    - **content**: Text content of the chunk
    - **embedding**: Vector embedding representation
    - **document_id**: ID of the document this chunk belongs to
    - **metadata**: Optional additional metadata for the chunk
    """,
    responses={
        201: {"description": "Chunk created successfully"},
        400: {"description": "Invalid chunk data"},
        404: {"description": "Document not found"},
    },
    response_description="The created chunk",
)
async def create_chunk(
    chunk_data: ChunkCreate,
    chunk_service: ChunkService = Depends(get_chunk_service),
    db: AsyncSession = Depends(async_session),
) -> ChunkRead:
    """Create a new chunk."""
    try:
        result = await chunk_service.create_chunk(chunk_data, db)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        return result
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get(
    "/{chunk_id}",
    summary="Get Chunk Details",
    description="""
    Retrieves details for a specific chunk by ID.

    Returns the chunk content, embedding, and metadata.
    """,
    responses={
        200: {"description": "Chunk details"},
        404: {"description": "Chunk not found"},
    },
)
async def get_chunk(
    chunk_id: int,
    chunk_service: ChunkService = Depends(get_chunk_service),
    db: AsyncSession = Depends(async_session),
) -> ChunkRead:
    """Get a specific chunk by ID."""
    try:
        result = await chunk_service.get_chunk(chunk_id, db)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
        return result
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get(
    "/",
    summary="List Chunks",
    description="""
    Retrieves a paginated list of chunks, optionally filtered by document.

    Returns chunks ordered by creation time.

    - **document_id**: Optional document ID to filter chunks
    - **page**: Page number (1-indexed, default: 1)
    - **items_per_page**: Number of chunks per page (default: 50, max: 100)
    """,
    responses={
        200: {"description": "Paginated list of chunks"},
        400: {"description": "Invalid pagination parameters"},
        404: {"description": "Document not found (when document_id provided)"},
    },
)
async def get_chunks(
    document_id: Annotated[Optional[int], Query(description="Filter by document ID")] = None,
    page: Annotated[int, Query(ge=1, description="Page number (1-indexed)")] = 1,
    items_per_page: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 50,
    chunk_service: ChunkService = Depends(get_chunk_service),
    db: AsyncSession = Depends(async_session),
):
    """Get chunks with pagination and optional document filtering."""
    try:
        if document_id:
            return await chunk_service.get_chunks_by_document(document_id, db, page, items_per_page)
        return await chunk_service.get_chunks(db, page, items_per_page)
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.put(
    "/{chunk_id}",
    response_model=ChunkRead,
    summary="Update Chunk",
    description="""Update an existing chunk's content, metadata, or embeddings.
    
    This endpoint allows you to modify chunk properties. When content is updated,
    embeddings should also be provided or they will become stale.
    
    - **chunk_id**: ID of the chunk to update
    - **content**: New text content (optional)
    - **embedding**: New vector embedding (optional, should match content)
    - **metadata**: New metadata dictionary (optional)
    
    **Note**: Updating chunk content without updating embeddings may lead
    to inconsistent search results.
    """,
    responses={
        200: {"description": "Chunk updated successfully"},
        400: {"description": "Invalid update data or embedding dimension mismatch"},
        404: {"description": "Chunk not found"},
    },
)
async def update_chunk(
    chunk_id: int,
    update_data: ChunkUpdate,
    chunk_service: ChunkService = Depends(get_chunk_service),
    db: AsyncSession = Depends(async_session),
) -> ChunkRead:
    """Update a chunk."""
    try:
        result = await chunk_service.update_chunk(chunk_id, update_data, db)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
        return result
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.delete(
    "/{chunk_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Chunk",
    description="""Delete a specific chunk from the vector database.
    
    This operation permanently removes the chunk from both the database
    and the vector index. This action cannot be undone.
    
    - **chunk_id**: ID of the chunk to delete
    
    **Note**: Deleting chunks may affect document completeness.
    Consider deleting the entire document instead if appropriate.
    """,
    responses={
        204: {"description": "Chunk deleted successfully"},
        404: {"description": "Chunk not found"},
    },
)
async def delete_chunk(
    chunk_id: int,
    chunk_service: ChunkService = Depends(get_chunk_service),
    db: AsyncSession = Depends(async_session),
):
    """Delete a chunk."""
    try:
        success = await chunk_service.delete_chunk(chunk_id, db)
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
