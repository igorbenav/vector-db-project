"""Library management service for vector database operations."""

import time
from typing import Any, Optional, cast

from fastcrud.paginated.response import paginated_response
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from ...infrastructure.indexing.manager import index_manager
from ..chunk.models import Chunk
from ..document.models import Document
from .crud import library_crud
from .models import Library
from .schemas import LibraryCreate, LibraryRead, LibraryUpdate, SearchResult, VectorSearchRequest, VectorSearchResponse


class LibraryService:
    """Service for managing libraries in the vector database.

    Provides high-level operations for library management including
    creating libraries, retrieving with metadata counts, updating
    library information, and managing library lifecycle.

    Each library serves as a container for documents and provides
    isolation between different document collections for vector search.
    """

    async def create_library(
        self,
        library_data: LibraryCreate,
        db: AsyncSession,
    ) -> Optional[LibraryRead]:
        """Create a new library.

        Args:
            library_data: Library creation data
            db: Database session

        Returns:
            Created library data with counts
        """

        class LibraryCreateInternal(BaseModel):
            name: str
            description: str | None = None
            extra_metadata: dict | None = None

        library_internal = LibraryCreateInternal(
            name=library_data.name, description=library_data.description, extra_metadata=library_data.metadata
        )

        created_library = cast(Any, await library_crud.create(db=db, object=library_internal))

        library_data_dict = created_library.__dict__.copy()
        library_data_dict["metadata"] = library_data_dict.pop("extra_metadata", {})
        library_data_dict["document_count"] = 0
        library_data_dict["chunk_count"] = 0

        return LibraryRead(**library_data_dict)

    async def get_library(
        self,
        library_id: int,
        db: AsyncSession,
    ) -> Optional[LibraryRead]:
        """Get a specific library with document and chunk counts.

        Args:
            library_id: Library ID to retrieve
            db: Database session

        Returns:
            Library data with counts
        """
        stmt = await library_crud.select(id=library_id)

        stmt = (
            stmt.add_columns(
                func.count(func.distinct(Document.id)).label("document_count"), func.count(Chunk.id).label("chunk_count")
            )
            .outerjoin(Document, Library.id == Document.library_id)
            .outerjoin(Chunk, Document.id == Chunk.document_id)
            .group_by(Library.id)
        )

        result = await db.execute(stmt)
        row = result.first()

        if not row:
            return None

        library_data_dict = {
            "id": row.id,
            "name": row.name,
            "description": row.description,
            "metadata": row.extra_metadata or {},
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "document_count": row.document_count,
            "chunk_count": row.chunk_count,
        }

        return LibraryRead(**library_data_dict)

    async def get_libraries(
        self,
        db: AsyncSession,
        page: int = 1,
        items_per_page: int = 50,
    ) -> dict[str, Any]:
        """Get libraries with pagination and counts.

        Args:
            db: Database session
            page: Page number (1-indexed)
            items_per_page: Number of libraries per page

        Returns:
            Paginated response with libraries and counts
        """
        offset = (page - 1) * items_per_page
        stmt = await library_crud.select(sort_columns="updated_at", sort_orders="desc")
        stmt = (
            stmt.add_columns(
                func.count(func.distinct(Document.id)).label("document_count"), func.count(Chunk.id).label("chunk_count")
            )
            .outerjoin(Document, Library.id == Document.library_id)
            .outerjoin(Chunk, Document.id == Chunk.document_id)
            .group_by(Library.id)
            .offset(offset)
            .limit(items_per_page)
        )

        result = await db.execute(stmt)
        rows = result.fetchall()
        total_count = await library_crud.count(db=db)

        libraries = []
        for row in rows:
            library_data_dict = {
                "id": row.id,
                "name": row.name,
                "description": row.description,
                "metadata": row.extra_metadata or {},
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "document_count": row.document_count,
                "chunk_count": row.chunk_count,
            }
            libraries.append(library_data_dict)

        crud_data = {"data": libraries, "total_count": total_count}

        return paginated_response(crud_data, page, items_per_page)

    async def update_library(
        self,
        library_id: int,
        update_data: LibraryUpdate,
        db: AsyncSession,
    ) -> Optional[LibraryRead]:
        """Update a library.

        Args:
            library_id: Library ID to update
            update_data: Update data
            db: Database session

        Returns:
            Updated library data with counts
        """
        update_dict = update_data.model_dump(exclude_unset=True)
        if "metadata" in update_dict:
            update_dict["extra_metadata"] = update_dict.pop("metadata")

        await library_crud.update(db=db, id=library_id, object=update_dict)

        return await self.get_library(library_id, db)

    async def delete_library(
        self,
        library_id: int,
        db: AsyncSession,
    ) -> bool:
        """Delete a library and all its documents and chunks.

        Args:
            library_id: Library ID to delete
            db: Database session

        Returns:
            True if deletion was successful
        """
        await library_crud.delete(db=db, id=library_id)
        return True

    async def vector_search(
        self,
        library_id: int,
        search_request: VectorSearchRequest,
        db: AsyncSession,
    ) -> Optional[VectorSearchResponse]:
        """Perform k-NN vector search within a library.

        Delegates to the IndexManager which handles the actual search algorithm.

        Args:
            library_id: Library ID to search within
            search_request: Search parameters including query embedding and k
            db: Database session

        Returns:
            Search results with similarity scores and metadata
        """
        start_time = time.perf_counter()

        library_exists = await library_crud.exists(db=db, id=library_id)
        if not library_exists:
            return None

        search_results = await index_manager.search(
            library_id=library_id,
            query_embedding=search_request.query_embedding,
            k=search_request.k,
            metadata_filter=search_request.metadata_filter,
            db=db,
        )

        if search_results is None:
            return None

        end_time = time.perf_counter()
        query_time_ms = (end_time - start_time) * 1000

        index_stats = index_manager.get_index_stats(library_id)
        total_chunks_searched = index_stats.get("total_vectors", 0) if index_stats else 0

        converted_results = [
            SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                similarity_score=result.similarity_score,
                metadata=result.metadata,
            )
            for result in search_results
        ]

        return VectorSearchResponse(
            results=converted_results, query_time_ms=query_time_ms, total_chunks_searched=total_chunks_searched
        )
