"""Document API endpoints."""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from ....infrastructure.database.session import async_session
from ....modules.common.exceptions import ValidationError
from ....modules.common.utils.error_handler import handle_exception
from ....modules.document.schemas import (
    DocumentCreate,
    DocumentRead,
    DocumentUpdate,
)
from ....modules.document.services import DocumentService

router = APIRouter(prefix="/document", tags=["Documents"])


def get_document_service() -> DocumentService:
    """Dependency to get document service instance."""
    return DocumentService()


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create New Document",
    description="""
    Creates a new document within a library.

    Documents serve as logical groupings of chunks within a library,
    representing coherent units of content like files or articles.

    - **title**: Title or name of the document
    - **library_id**: ID of the library this document belongs to
    - **metadata**: Optional additional metadata for the document
    """,
    responses={
        201: {"description": "Document created successfully"},
        400: {"description": "Invalid document data"},
        404: {"description": "Library not found"},
    },
    response_description="The created document with chunk count",
)
async def create_document(
    document_data: DocumentCreate,
    document_service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(async_session),
) -> DocumentRead:
    """Create a new document."""
    try:
        result = await document_service.create_document(document_data, db)
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


@router.get(
    "/{document_id}",
    summary="Get Document Details",
    description="""
    Retrieves details for a specific document by ID.

    Returns the document metadata along with current chunk count.
    This provides a quick overview of the document's contents without loading
    all the actual chunks.
    """,
    responses={
        200: {"description": "Document details with chunk count"},
        404: {"description": "Document not found"},
    },
)
async def get_document(
    document_id: int,
    document_service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(async_session),
) -> DocumentRead:
    """Get a specific document by ID."""
    try:
        result = await document_service.get_document(document_id, db)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        return result
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get(
    "/",
    summary="List Documents",
    description="""
    Retrieves a paginated list of documents, optionally filtered by library.

    Returns documents ordered by last update time (most recent first).
    Each document includes chunk count for quick overview.

    - **library_id**: Optional library ID to filter documents
    - **page**: Page number (1-indexed, default: 1)
    - **items_per_page**: Number of documents per page (default: 50, max: 100)
    """,
    responses={
        200: {"description": "Paginated list of documents"},
        400: {"description": "Invalid pagination parameters"},
        404: {"description": "Library not found (when library_id provided)"},
    },
)
async def get_documents(
    library_id: Annotated[Optional[int], Query(description="Filter by library ID")] = None,
    page: Annotated[int, Query(ge=1, description="Page number (1-indexed)")] = 1,
    items_per_page: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 50,
    document_service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(async_session),
):
    """Get documents with pagination and optional library filtering."""
    try:
        if library_id:
            return await document_service.get_documents_by_library(library_id, db, page, items_per_page)
        return await document_service.get_documents(db, page, items_per_page)
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.put(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Update Document",
    description="""Update an existing document's metadata and title.
    
    This endpoint allows you to modify document properties without affecting
    its chunks or embeddings. Only the document metadata and title can be updated.
    
    - **document_id**: ID of the document to update
    - **title**: New title for the document (optional)
    - **metadata**: New metadata dictionary (optional)
    """,
    responses={
        200: {"description": "Document updated successfully"},
        400: {"description": "Invalid update data"},
        404: {"description": "Document not found"},
    },
)
async def update_document(
    document_id: int,
    update_data: DocumentUpdate,
    document_service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(async_session),
) -> DocumentRead:
    """Update a document."""
    try:
        result = await document_service.update_document(document_id, update_data, db)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        return result
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Document",
    description="""Delete a document and all its associated chunks.
    
    This operation permanently removes the document and all its chunks
    from the database and rebuilds the vector index to ensure search
    consistency. This action cannot be undone.
    
    - **document_id**: ID of the document to delete
    
    The operation will:
    1. Remove all chunks belonging to the document
    2. Delete the document record
    3. Rebuild the library's vector index
    """,
    responses={
        204: {"description": "Document deleted successfully"},
        404: {"description": "Document not found"},
    },
)
async def delete_document(
    document_id: int,
    document_service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(async_session),
):
    """Delete a document and all its chunks."""
    try:
        success = await document_service.delete_document(document_id, db)
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    except Exception as e:
        http_exc = handle_exception(e)
        if http_exc:
            raise http_exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
