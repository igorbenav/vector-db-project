"""Library API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from ....infrastructure.database.session import async_session
from ....infrastructure.indexing.base import IndexType
from ....infrastructure.indexing.manager import index_manager
from ....modules.common.exceptions import ValidationError
from ....modules.common.utils.error_handler import handle_exception
from ....modules.library.schemas import (
    LibraryCreate,
    LibraryRead,
    LibraryUpdate,
    VectorSearchRequest,
    VectorSearchResponse,
)
from ....modules.library.services import LibraryService

router = APIRouter(prefix="/library", tags=["Libraries"])


def get_library_service() -> LibraryService:
    """Dependency to get library service instance."""
    return LibraryService()


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create New Library",
    description="""
    Creates a new library for organizing documents and chunks.

    A library serves as a container for documents and provides isolation
    between different document collections for vector search operations.

    - **name**: Unique name for the library
    - **description**: Optional description of the library's purpose
    - **metadata**: Optional additional metadata for the library
    """,
    responses={
        201: {"description": "Library created successfully"},
        400: {"description": "Invalid library data"},
        409: {"description": "Library name already exists"},
    },
    response_description="The created library with document and chunk counts",
)
async def create_library(
    library_data: LibraryCreate,
    library_service: LibraryService = Depends(get_library_service),
    db: AsyncSession = Depends(async_session),
) -> LibraryRead:
    """Create a new library."""
    try:
        result = await library_service.create_library(library_data, db)
        if not result:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create library")
        return result
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get(
    "/{library_id}",
    summary="Get Library Details",
    description="""
    Retrieves details for a specific library by ID.

    Returns the library metadata along with current document and chunk counts.
    This provides a quick overview of the library's contents without loading
    all the actual documents.
    """,
    responses={
        200: {"description": "Library details with counts"},
        404: {"description": "Library not found"},
    },
)
async def get_library(
    library_id: int,
    library_service: LibraryService = Depends(get_library_service),
    db: AsyncSession = Depends(async_session),
) -> LibraryRead:
    """Get a specific library by ID."""
    try:
        result = await library_service.get_library(library_id, db)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found")
        return result
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get(
    "/",
    summary="List Libraries",
    description="""
    Retrieves a paginated list of all libraries.

    Returns libraries ordered by last update time (most recent first).
    Each library includes document and chunk counts for quick overview.

    - **page**: Page number (1-indexed, default: 1)
    - **items_per_page**: Number of libraries per page (default: 50, max: 100)
    """,
    responses={
        200: {"description": "Paginated list of libraries"},
        400: {"description": "Invalid pagination parameters"},
    },
)
async def get_libraries(
    page: Annotated[int, Query(ge=1, description="Page number (1-indexed)")] = 1,
    items_per_page: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 50,
    library_service: LibraryService = Depends(get_library_service),
    db: AsyncSession = Depends(async_session),
):
    """Get libraries with pagination."""
    try:
        return await library_service.get_libraries(db, page, items_per_page)
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.put(
    "/{library_id}",
    response_model=LibraryRead,
    summary="Update Library",
    description="""Update an existing library's name and description.
    
    This endpoint allows you to modify library metadata without affecting
    its documents, chunks, or vector index configuration.
    
    - **library_id**: ID of the library to update
    - **name**: New name for the library (optional)
    - **description**: New description (optional)
    
    **Note**: The library's index algorithm cannot be changed after creation.
    """,
    responses={
        200: {"description": "Library updated successfully"},
        400: {"description": "Invalid update data"},
        404: {"description": "Library not found"},
    },
)
async def update_library(
    library_id: int,
    update_data: LibraryUpdate,
    library_service: LibraryService = Depends(get_library_service),
    db: AsyncSession = Depends(async_session),
) -> LibraryRead:
    """Update a library."""
    try:
        result = await library_service.update_library(library_id, update_data, db)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found")
        return result
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found")
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.delete(
    "/{library_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Library",
    description="""Delete a library and all its associated documents and chunks.
    
    This operation permanently removes the entire library including:
    - All documents within the library
    - All chunks belonging to those documents
    - The library's vector index
    
    This action cannot be undone and will affect all content in the library.
    
    - **library_id**: ID of the library to delete
    """,
    responses={
        204: {"description": "Library deleted successfully"},
        404: {"description": "Library not found"},
    },
)
async def delete_library(
    library_id: int,
    library_service: LibraryService = Depends(get_library_service),
    db: AsyncSession = Depends(async_session),
):
    """Delete a library and all its documents and chunks."""
    try:
        success = await library_service.delete_library(library_id, db)
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found")
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found")
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get(
    "/{library_id}/stats",
    summary="Get Library Statistics",
    description="""Get detailed statistics for a library including vector index performance.
    
    Returns comprehensive information about the library's contents and
    vector index performance metrics:
    
    - Document and chunk counts
    - Vector index algorithm and configuration
    - Index performance statistics
    - Storage and memory usage information
    
    - **library_id**: ID of the library to get statistics for
    """,
    responses={
        200: {"description": "Library statistics returned successfully"},
        404: {"description": "Library not found"},
    },
)
async def get_library_stats(
    library_id: int,
    library_service: LibraryService = Depends(get_library_service),
):
    """Get library indexing statistics."""

    stats = index_manager.get_index_stats(library_id)
    if not stats:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library index not found")
    return stats


@router.post(
    "/{library_id}/rebuild-index",
    summary="Rebuild Vector Index",
    description="""Manually rebuild the vector index for a library with specified algorithm.
    
    This operation recreates the vector index from scratch using all
    current chunks in the library. Useful for:
    
    - Recovering from index corruption
    - Switching between index algorithms (LINEAR_SEARCH, IVF)
    - Applying algorithm optimizations
    - Ensuring index consistency after bulk operations
    
    - **library_id**: ID of the library to rebuild index for
    - **index_type**: Index algorithm (LINEAR_SEARCH or IVF)
    
    **Warning**: This operation may take time for large libraries
    and will temporarily affect search performance.
    """,
    responses={
        200: {"description": "Index rebuilt successfully"},
        400: {"description": "Invalid index type"},
        404: {"description": "Library not found"},
        500: {"description": "Index rebuild failed"},
    },
)
async def rebuild_index(
    library_id: int,
    index_type: str,
    library_service: LibraryService = Depends(get_library_service),
    db: AsyncSession = Depends(async_session),
):
    """Rebuild library index with specified algorithm (for testing)."""
    try:
        index_type_enum = IndexType(index_type)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid index type: {index_type}")

    if library_id in index_manager._indexes:
        del index_manager._indexes[library_id]

    await index_manager.get_or_create_index(library_id=library_id, index_type=index_type_enum, embedding_dimension=768, db=db)

    return {"message": f"Index rebuilt with {index_type} algorithm"}


@router.post(
    "/{library_id}/search",
    summary="Vector Search in Library",
    description="""
    Performs k-Nearest Neighbor vector search within a specific library.

    This endpoint searches for the most similar chunks to your query embedding
    using cosine similarity. Results are ranked by similarity score.

    - **query_embedding**: Vector embedding of your search query
    - **k**: Number of most similar results to return (max: 100)
    - **metadata_filter**: Optional metadata filters to narrow search scope

    The search uses Linear Search algorithm (O(n*d) complexity) for exact results.
    For large libraries, consider using more advanced indexing algorithms.
    """,
    responses={
        200: {"description": "Search results with similarity scores"},
        400: {"description": "Invalid search parameters"},
        404: {"description": "Library not found"},
    },
    response_description="Search results ranked by similarity with timing metrics",
)
async def search_library(
    library_id: int,
    search_request: VectorSearchRequest,
    library_service: LibraryService = Depends(get_library_service),
    db: AsyncSession = Depends(async_session),
) -> VectorSearchResponse:
    """Perform k-NN vector search within a library."""
    try:
        result = await library_service.vector_search(library_id, search_request, db)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found")
        return result
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
