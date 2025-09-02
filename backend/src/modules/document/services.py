"""Document management service for vector database operations."""

from typing import Any, Optional, cast

from fastcrud.paginated.response import paginated_response
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from ..chunk.models import Chunk
from ..library.crud import library_crud
from .crud import document_crud
from .models import Document
from .schemas import DocumentCreate, DocumentRead, DocumentUpdate


class DocumentService:
    """Service for managing documents in the vector database.

    Provides high-level operations for document management including
    creating documents within libraries, retrieving with chunk counts,
    updating document information, and managing document lifecycle.

    Documents serve as logical groupings of chunks within a library,
    representing coherent units of content like files or articles.
    """

    async def create_document(
        self,
        document_data: DocumentCreate,
        db: AsyncSession,
    ) -> Optional[DocumentRead]:
        """Create a new document within a library.

        Args:
            document_data: Document creation data
            db: Database session

        Returns:
            Created document data with counts
        """
        library_exists = await library_crud.exists(db=db, id=document_data.library_id)
        if not library_exists:
            return None

        class DocumentCreateInternal(BaseModel):
            library_id: int
            title: str
            extra_metadata: dict | None = None

        document_internal = DocumentCreateInternal(
            library_id=document_data.library_id, title=document_data.title, extra_metadata=document_data.metadata
        )

        created_document = cast(Any, await document_crud.create(db=db, object=document_internal))

        document_data_dict = created_document.__dict__.copy()
        document_data_dict["metadata"] = document_data_dict.pop("extra_metadata", {})
        document_data_dict["chunk_count"] = 0

        return DocumentRead(**document_data_dict)

    async def get_document(
        self,
        document_id: int,
        db: AsyncSession,
    ) -> Optional[DocumentRead]:
        """Get a specific document with chunk count.

        Args:
            document_id: Document ID to retrieve
            db: Database session

        Returns:
            Document data with chunk count
        """
        stmt = await document_crud.select(id=document_id)

        stmt = (
            stmt.add_columns(func.count(Chunk.id).label("chunk_count"))
            .outerjoin(Chunk, Document.id == Chunk.document_id)
            .group_by(Document.id)
        )

        result = await db.execute(stmt)
        row = result.first()

        if not row:
            return None

        document_data_dict = {
            "id": row.id,
            "library_id": row.library_id,
            "title": row.title,
            "metadata": row.extra_metadata or {},
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "chunk_count": row.chunk_count,
        }

        return DocumentRead(**document_data_dict)

    async def get_documents_by_library(
        self,
        library_id: int,
        db: AsyncSession,
        page: int = 1,
        items_per_page: int = 50,
    ) -> dict[str, Any]:
        """Get documents in a library with pagination and counts.

        Args:
            library_id: Library ID to get documents from
            db: Database session
            page: Page number (1-indexed)
            items_per_page: Number of documents per page

        Returns:
            Paginated response with documents and counts
        """
        library_exists = await library_crud.exists(db=db, id=library_id)
        if not library_exists:
            return paginated_response({"data": [], "total_count": 0}, page, items_per_page)

        offset = (page - 1) * items_per_page
        stmt = await document_crud.select(library_id=library_id, sort_columns="updated_at", sort_orders="desc")

        stmt = (
            stmt.add_columns(func.count(Chunk.id).label("chunk_count"))
            .outerjoin(Chunk, Document.id == Chunk.document_id)
            .group_by(Document.id)
            .offset(offset)
            .limit(items_per_page)
        )

        result = await db.execute(stmt)
        rows = result.fetchall()

        total_count = await document_crud.count(db=db, library_id=library_id)

        documents = []
        for row in rows:
            document_data_dict = {
                "id": row.id,
                "library_id": row.library_id,
                "title": row.title,
                "metadata": row.extra_metadata or {},
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "chunk_count": row.chunk_count,
            }
            documents.append(document_data_dict)

        crud_data = {"data": documents, "total_count": total_count}

        return paginated_response(crud_data, page, items_per_page)

    async def get_documents(
        self,
        db: AsyncSession,
        page: int = 1,
        items_per_page: int = 50,
    ) -> dict[str, Any]:
        """Get all documents with pagination and counts.

        Args:
            db: Database session
            page: Page number (1-indexed)
            items_per_page: Number of documents per page

        Returns:
            Paginated response with documents and counts
        """
        offset = (page - 1) * items_per_page

        stmt = await document_crud.select(sort_columns="updated_at", sort_orders="desc")
        stmt = (
            stmt.add_columns(func.count(Chunk.id).label("chunk_count"))
            .outerjoin(Chunk, Document.id == Chunk.document_id)
            .group_by(Document.id)
            .offset(offset)
            .limit(items_per_page)
        )

        result = await db.execute(stmt)
        rows = result.fetchall()

        total_count = await document_crud.count(db=db)

        documents = []
        for row in rows:
            document_data_dict = {
                "id": row.id,
                "library_id": row.library_id,
                "title": row.title,
                "metadata": row.extra_metadata or {},
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "chunk_count": row.chunk_count,
            }
            documents.append(document_data_dict)

        crud_data = {"data": documents, "total_count": total_count}

        return paginated_response(crud_data, page, items_per_page)

    async def update_document(
        self,
        document_id: int,
        update_data: DocumentUpdate,
        db: AsyncSession,
    ) -> Optional[DocumentRead]:
        """Update a document.

        Args:
            document_id: Document ID to update
            update_data: Update data
            db: Database session

        Returns:
            Updated document data with counts
        """
        update_dict = update_data.model_dump(exclude_unset=True)
        if "metadata" in update_dict:
            update_dict["extra_metadata"] = update_dict.pop("metadata")

        await document_crud.update(db=db, id=document_id, object=update_dict)

        return await self.get_document(document_id, db)

    async def delete_document(
        self,
        document_id: int,
        db: AsyncSession,
    ) -> bool:
        """Delete a document and all its chunks.

        Args:
            document_id: Document ID to delete
            db: Database session

        Returns:
            True if deletion was successful
        """
        document = await self.get_document(document_id, db)
        if not document:
            return False

        library_id = document.library_id

        await document_crud.delete(db=db, id=document_id)

        from ...infrastructure.indexing.manager import index_manager

        if library_id in index_manager._indexes:
            await index_manager.rebuild_library_index(library_id, db)

        return True
