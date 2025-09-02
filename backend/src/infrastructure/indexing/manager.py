"""Index management service for coordinating vector indexes across libraries."""

import asyncio
from typing import Any, Dict, List, Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession

from ...modules.chunk.crud import chunk_crud
from ...modules.chunk.models import Chunk
from ...modules.document.models import Document
from .base import ChunkVector, IndexType, SearchResult, VectorIndex
from .ivf import IVFIndex
from .linear_search import LinearSearchIndex


class IndexManager:
    """Manages vector indexes for libraries.

    This service coordinates the creation, updating, and querying of vector indexes
    for each library. It handles:
    - Per-library index instances
    - Lazy loading of indexes (build on first search)
    - Index updates when chunks are added/removed
    - Algorithm selection per library
    """

    def __init__(self):
        """Initialize the index manager."""
        self._indexes: Dict[int, Union[LinearSearchIndex, IVFIndex]] = {}
        self._index_types: Dict[int, IndexType] = {}
        self._locks: Dict[int, asyncio.Lock] = {}

    async def get_or_create_index(
        self,
        library_id: int,
        index_type: IndexType = IndexType.LINEAR_SEARCH,
        embedding_dimension: int = 1536,
        db: Optional[AsyncSession] = None,
    ) -> VectorIndex:
        """Get existing index or create a new one for the library.

        Args:
            library_id: Library ID to get index for
            index_type: Type of index to create if not exists
            embedding_dimension: Dimension of embeddings
            db: Database session for loading vectors

        Returns:
            Vector index instance for the library
        """
        if library_id not in self._locks:
            self._locks[library_id] = asyncio.Lock()

        async with self._locks[library_id]:
            if library_id not in self._indexes:
                if index_type == IndexType.LINEAR_SEARCH:
                    self._indexes[library_id] = LinearSearchIndex(dimension=embedding_dimension)
                elif index_type == IndexType.IVF:
                    num_clusters = max(10, min(100, int(embedding_dimension**0.5)))
                    self._indexes[library_id] = IVFIndex(dimension=embedding_dimension, num_clusters=num_clusters, num_probes=1)
                else:
                    raise ValueError(f"Index type {index_type} not implemented yet")
                self._index_types[library_id] = index_type

                if db:
                    await self._load_library_vectors(library_id, self._indexes[library_id], db)

        return self._indexes[library_id]

    async def search(
        self,
        library_id: int,
        query_embedding: List[float],
        k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
        db: Optional[AsyncSession] = None,
    ) -> Optional[List[SearchResult]]:
        """Perform vector search within a library.

        Args:
            library_id: Library ID to search within
            query_embedding: Query vector
            k: Number of results to return
            metadata_filter: Optional metadata filter
            db: Database session

        Returns:
            Search results or None if library not found
        """
        if library_id in self._indexes:
            index = self._indexes[library_id]
        else:
            await self.get_or_create_index(library_id=library_id, embedding_dimension=len(query_embedding), db=db)
            index = self._indexes[library_id]

        return await index.search(query_embedding=query_embedding, k=k, metadata_filter=metadata_filter)

    async def add_chunk_vector(self, library_id: int, chunk_vector: ChunkVector) -> None:
        """Add a new chunk vector to the library's index.

        Args:
            library_id: Library ID
            chunk_vector: The vector to add
        """
        if library_id in self._indexes and library_id in self._locks:
            async with self._locks[library_id]:
                await self._indexes[library_id].add_vector(chunk_vector)

    async def remove_chunk_vector(self, library_id: int, chunk_id: int) -> bool:
        """Remove a chunk vector from the library's index.

        Args:
            library_id: Library ID
            chunk_id: Chunk ID to remove

        Returns:
            True if vector was found and removed
        """
        if library_id in self._indexes and library_id in self._locks:
            async with self._locks[library_id]:
                return await self._indexes[library_id].remove_vector(chunk_id)
        return False

    async def rebuild_library_index(self, library_id: int, db: AsyncSession) -> None:
        """Rebuild the index for a library from database.

        Args:
            library_id: Library ID to rebuild index for
            db: Database session
        """
        if library_id in self._indexes and library_id in self._locks:
            async with self._locks[library_id]:
                index = self._indexes[library_id]
                await index.clear()
                await self._load_library_vectors(library_id, index, db)
                await index.build_index()

    def get_index_stats(self, library_id: int) -> Optional[Dict[str, Any]]:
        """Get statistics for a library's index.

        Args:
            library_id: Library ID

        Returns:
            Index statistics or None if index doesn't exist
        """
        if library_id not in self._indexes:
            return None

        index = self._indexes[library_id]
        stats = index.get_stats()

        result = {
            "index_type": index.index_type.value,
            "total_vectors": stats.total_vectors,
            "embedding_dimension": stats.embedding_dimension,
            "is_built": index.is_built(),
        }

        if stats.algorithm_params:
            result.update(stats.algorithm_params)

        return result

    async def _load_library_vectors(self, library_id: int, index: VectorIndex, db: AsyncSession) -> None:
        """Load all vectors for a library from the database into the index.

        Args:
            library_id: Library ID to load vectors for
            index: Index instance to load vectors into
            db: Database session
        """
        stmt = await chunk_crud.select()
        stmt = stmt.join(Document, Chunk.document_id == Document.id).where(Document.library_id == library_id)

        result = await db.execute(stmt)
        chunks = result.fetchall()

        chunk_vectors = []
        for chunk in chunks:
            chunk_vector = ChunkVector(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                embedding=chunk.embedding,
                metadata=chunk.extra_metadata or {},
            )
            chunk_vectors.append(chunk_vector)

        if chunk_vectors:
            await index.add_vectors(chunk_vectors)

        await index.build_index()


index_manager = IndexManager()
