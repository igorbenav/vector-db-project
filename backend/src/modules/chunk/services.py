"""Chunk management service for vector database operations."""

from datetime import datetime, timezone
from typing import Any, List, Optional, cast

from fastcrud.paginated.response import paginated_response
from pydantic import BaseModel
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ...infrastructure.indexing.base import ChunkVector
from ...infrastructure.indexing.manager import index_manager
from ..document.crud import document_crud
from .crud import chunk_crud
from .models import Chunk
from .schemas import ChunkCreate, ChunkRead, ChunkUpdate


class ChunkService:
    """Service for managing chunks in the vector database.

    Provides high-level operations for chunk management including
    creating chunks within documents, retrieving chunks, updating
    chunk information, and managing chunk lifecycle.

    Chunks are the atomic units of content with vector embeddings
    that enable similarity search functionality.
    """

    async def create_chunk(
        self,
        chunk_data: ChunkCreate,
        db: AsyncSession,
    ) -> Optional[ChunkRead]:
        """Create a new chunk within a document.

        Args:
            chunk_data: Chunk creation data
            db: Database session

        Returns:
            Created chunk data
        """
        document_exists = await document_crud.exists(db=db, id=chunk_data.document_id)
        if not document_exists:
            return None

        class ChunkCreateInternal(BaseModel):
            document_id: int
            content: str
            embedding: List[float]
            extra_metadata: dict | None = None

        chunk_internal = ChunkCreateInternal(
            document_id=chunk_data.document_id,
            content=chunk_data.content,
            embedding=chunk_data.embedding,
            extra_metadata=chunk_data.metadata,
        )

        created_chunk = cast(Any, await chunk_crud.create(db=db, object=chunk_internal))

        document = await document_crud.get(db=db, id=chunk_data.document_id)
        if document:
            chunk_vector = ChunkVector(
                chunk_id=created_chunk.id,
                document_id=created_chunk.document_id,
                content=created_chunk.content,
                embedding=created_chunk.embedding,
                metadata=created_chunk.extra_metadata or {},
            )
            await index_manager.add_chunk_vector(document["library_id"], chunk_vector)

        chunk_data_dict = created_chunk.__dict__.copy()
        chunk_data_dict["metadata"] = chunk_data_dict.pop("extra_metadata", {})

        return ChunkRead(**chunk_data_dict)

    async def create_chunks_bulk(
        self,
        chunks_data: List[ChunkCreate],
        db: AsyncSession,
    ) -> List[ChunkRead]:
        """Create multiple chunks in bulk using efficient bulk insert.

        Args:
            chunks_data: List of chunk creation data
            db: Database session

        Returns:
            List of created chunks
        """
        if not chunks_data:
            return []

        document_ids = list(set(chunk.document_id for chunk in chunks_data))
        for doc_id in document_ids:
            document_exists = await document_crud.exists(db=db, id=doc_id)
            if not document_exists:
                return []

        chunks_to_insert = []
        now = datetime.now(timezone.utc)
        for chunk_data in chunks_data:
            chunk_dict = {
                "document_id": chunk_data.document_id,
                "content": chunk_data.content,
                "embedding": chunk_data.embedding,
                "extra_metadata": chunk_data.metadata or {},
                "created_at": now,
                "updated_at": now,
            }
            chunks_to_insert.append(chunk_dict)

        stmt = insert(Chunk).returning(Chunk)
        result = await db.execute(stmt, chunks_to_insert)
        await db.commit()

        created_chunks = []
        chunk_vectors = []
        library_id = None

        for row in result.fetchall():
            chunk_obj = row.Chunk
            chunk_data_dict = {
                "id": chunk_obj.id,
                "document_id": chunk_obj.document_id,
                "content": chunk_obj.content,
                "embedding": chunk_obj.embedding,
                "metadata": chunk_obj.extra_metadata or {},
                "created_at": chunk_obj.created_at,
                "updated_at": chunk_obj.updated_at,
            }
            created_chunks.append(ChunkRead(**chunk_data_dict))

            if library_id is None:
                document = await document_crud.get(db=db, id=chunk_obj.document_id)
                if document:
                    library_id = document["library_id"]

            chunk_vector = ChunkVector(
                chunk_id=chunk_obj.id,
                document_id=chunk_obj.document_id,
                content=chunk_obj.content,
                embedding=chunk_obj.embedding,
                metadata=chunk_obj.extra_metadata or {},
            )
            chunk_vectors.append(chunk_vector)

        if library_id and chunk_vectors:
            for chunk_vector in chunk_vectors:
                await index_manager.add_chunk_vector(library_id, chunk_vector)

        return created_chunks

    async def get_chunk(
        self,
        chunk_id: int,
        db: AsyncSession,
    ) -> Optional[ChunkRead]:
        """Get a specific chunk.

        Args:
            chunk_id: Chunk ID to retrieve
            db: Database session

        Returns:
            Chunk data
        """
        chunk = await chunk_crud.get(db=db, id=chunk_id)
        if not chunk:
            return None

        chunk_data_dict = chunk.copy()
        chunk_data_dict["metadata"] = chunk_data_dict.pop("extra_metadata", {})

        return ChunkRead(**chunk_data_dict)

    async def get_chunks_by_document(
        self,
        document_id: int,
        db: AsyncSession,
        page: int = 1,
        items_per_page: int = 50,
    ) -> dict[str, Any]:
        """Get chunks in a document with pagination.

        Args:
            document_id: Document ID to get chunks from
            db: Database session
            page: Page number (1-indexed)
            items_per_page: Number of chunks per page

        Returns:
            Paginated response with chunks
        """
        document_exists = await document_crud.exists(db=db, id=document_id)
        if not document_exists:
            return paginated_response({"data": [], "total_count": 0}, page, items_per_page)

        offset = (page - 1) * items_per_page

        result = await chunk_crud.get_multi(
            db=db,
            document_id=document_id,
            limit=items_per_page,
            offset=offset,
            sort_columns="created_at",
            sort_orders="asc",
        )

        chunks_data = result.get("data", [])
        total_count = result.get("total_count", 0)

        chunks = []
        if isinstance(chunks_data, list):
            for chunk in chunks_data:
                chunk_data_dict = chunk.copy()
                chunk_data_dict["metadata"] = chunk_data_dict.pop("extra_metadata", {})
                chunks.append(chunk_data_dict)

        crud_data = {"data": chunks, "total_count": total_count}

        return paginated_response(crud_data, page, items_per_page)

    async def get_chunks(
        self,
        db: AsyncSession,
        page: int = 1,
        items_per_page: int = 50,
    ) -> dict[str, Any]:
        """Get all chunks with pagination.

        Args:
            db: Database session
            page: Page number (1-indexed)
            items_per_page: Number of chunks per page

        Returns:
            Paginated response with chunks
        """
        offset = (page - 1) * items_per_page

        result = await chunk_crud.get_multi(
            db=db,
            limit=items_per_page,
            offset=offset,
            sort_columns="created_at",
            sort_orders="asc",
        )

        chunks_data = result.get("data", [])
        total_count = result.get("total_count", 0)

        chunks = []
        if isinstance(chunks_data, list):
            for chunk in chunks_data:
                chunk_data_dict = chunk.copy()
                chunk_data_dict["metadata"] = chunk_data_dict.pop("extra_metadata", {})
                chunks.append(chunk_data_dict)

        crud_data = {"data": chunks, "total_count": total_count}

        return paginated_response(crud_data, page, items_per_page)

    async def update_chunk(
        self,
        chunk_id: int,
        update_data: ChunkUpdate,
        db: AsyncSession,
    ) -> Optional[ChunkRead]:
        """Update a chunk.

        Args:
            chunk_id: Chunk ID to update
            update_data: Update data
            db: Database session

        Returns:
            Updated chunk data
        """
        update_dict = update_data.model_dump(exclude_unset=True)
        if "metadata" in update_dict:
            update_dict["extra_metadata"] = update_dict.pop("metadata")

        await chunk_crud.update(db=db, id=chunk_id, object=update_dict)

        return await self.get_chunk(chunk_id, db)

    async def delete_chunk(
        self,
        chunk_id: int,
        db: AsyncSession,
    ) -> bool:
        """Delete a chunk.

        Args:
            chunk_id: Chunk ID to delete
            db: Database session

        Returns:
            True if deletion was successful
        """
        chunk = await chunk_crud.get(db=db, id=chunk_id)
        if chunk:
            document = await document_crud.get(db=db, id=chunk["document_id"])
            if document:
                await index_manager.remove_chunk_vector(document["library_id"], chunk_id)

        await chunk_crud.delete(db=db, id=chunk_id)
        return True

    async def delete_chunks_by_document(
        self,
        document_id: int,
        db: AsyncSession,
    ) -> bool:
        """Delete all chunks in a document.

        Args:
            document_id: Document ID to delete chunks from
            db: Database session

        Returns:
            True if deletion was successful
        """
        chunks_result = await chunk_crud.get_multi(db=db, document_id=document_id)
        chunks_data = chunks_result.get("data", [])

        if chunks_data:
            document = await document_crud.get(db=db, id=document_id)
            if document:
                for chunk in chunks_data:  # type: ignore
                    await index_manager.remove_chunk_vector(document["library_id"], chunk["id"])

        await chunk_crud.delete(db=db, document_id=document_id, allow_multiple=True)
        return True

    async def get_chunks_by_ids(
        self,
        chunk_ids: List[int],
        db: AsyncSession,
    ) -> List[ChunkRead]:
        """Get multiple chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve
            db: Database session

        Returns:
            List of chunks
        """
        if not chunk_ids:
            return []

        chunks = []
        for chunk_id in chunk_ids:
            chunk = await self.get_chunk(chunk_id, db)
            if chunk:
                chunks.append(chunk)

        return chunks
