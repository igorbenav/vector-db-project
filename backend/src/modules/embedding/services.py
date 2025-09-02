"""Services that add embedding capabilities on top of existing core services."""

import re
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ...infrastructure.embedding import get_embedding_service
from ...infrastructure.indexing.base import IndexType
from ...infrastructure.indexing.manager import index_manager
from ..chunk.schemas import ChunkCreate, ChunkRead
from ..chunk.services import ChunkService
from ..document.schemas import DocumentCreate, DocumentRead
from ..document.services import DocumentService
from ..library.schemas import VectorSearchRequest, VectorSearchResponse
from ..library.services import LibraryService
from .schemas import DocumentAutoChunk, EmbeddedChunkCreate, EmbeddingInfo, TextSearchRequest


class EmbeddingChunkService:
    """Chunk service with automatic embedding generation."""

    def __init__(self):
        self.chunk_service = ChunkService()
        self.embedding_service = get_embedding_service()

    async def create_chunk_with_text(self, chunk_data: EmbeddedChunkCreate, db: AsyncSession) -> ChunkRead:
        """Create chunk with auto-generated embeddings from text."""

        if chunk_data.embedding is None:
            embedding = await self.embedding_service.embed_text(chunk_data.content)
        else:
            embedding = chunk_data.embedding

        create_data = ChunkCreate(
            content=chunk_data.content,
            document_id=chunk_data.document_id,
            embedding=embedding,
            metadata=chunk_data.metadata or {},
        )

        result = await self.chunk_service.create_chunk(create_data, db)
        if result is None:
            raise ValueError("Failed to create chunk")
        return result


class EmbeddingLibraryService:
    """Library service with text-based search."""

    def __init__(self):
        self.library_service = LibraryService()
        self.embedding_service = get_embedding_service()

    async def text_search(
        self, library_id: int, search_request: TextSearchRequest, db: AsyncSession
    ) -> Optional[VectorSearchResponse]:
        """Search library using text query (auto-generates embedding)."""

        query_embedding = await self.embedding_service.embed_text(search_request.query_text)

        vector_request = VectorSearchRequest(
            query_embedding=query_embedding, k=search_request.k, metadata_filter=search_request.metadata_filter
        )

        return await self.library_service.vector_search(library_id, vector_request, db)


class EmbeddingDocumentService:
    """Document service with automatic chunking and embedding."""

    def __init__(self):
        self.document_service = DocumentService()
        self.chunk_service = ChunkService()
        self.embedding_service = get_embedding_service()

    async def create_document_with_auto_chunks(self, doc_data: DocumentAutoChunk, db: AsyncSession) -> DocumentRead:
        """Create document with automatic chunking and embedding."""
        await index_manager.get_or_create_index(
            library_id=doc_data.library_id, index_type=IndexType.LINEAR_SEARCH, embedding_dimension=768, db=db
        )

        document_create = DocumentCreate(title=doc_data.title, library_id=doc_data.library_id, metadata=doc_data.metadata or {})

        document = await self.document_service.create_document(document_create, db)
        if document is None:
            raise ValueError("Failed to create document")

        chunks = self._chunk_text(doc_data.content, doc_data.chunk_size, doc_data.chunk_overlap)

        if chunks:
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = await self.embedding_service.embed_texts(chunk_texts)

            chunk_creates = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_metadata = {"chunk_index": chunk["index"], "char_start": chunk["start"], "char_end": chunk["end"]}
                if doc_data.metadata:
                    chunk_metadata.update(doc_data.metadata)

                chunk_creates.append(
                    ChunkCreate(
                        content=chunk["content"],
                        document_id=document.id,
                        embedding=embedding,
                        metadata=chunk_metadata,
                    )
                )

            await self.chunk_service.create_chunks_bulk(chunk_creates, db)

        updated_document = await self.document_service.get_document(document.id, db)
        if updated_document is None:
            raise ValueError(f"Failed to reload document {document.id}")
        return updated_document

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[dict]:
        """Split text into overlapping chunks."""

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""
        char_start = 0
        chunk_index = 0

        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    char_end = char_start + len(current_chunk)
                    chunks.append({"content": current_chunk, "index": chunk_index, "start": char_start, "end": char_end})

                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    overlap_text = current_chunk[overlap_start:]
                    char_start = char_start + overlap_start
                    chunk_index += 1

                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            char_end = char_start + len(current_chunk)
            chunks.append({"content": current_chunk, "index": chunk_index, "start": char_start, "end": char_end})

        return chunks


class EmbeddingInfoService:
    """Service for embedding model information."""

    def __init__(self):
        self.embedding_service = get_embedding_service()

    async def get_embedding_info(self) -> EmbeddingInfo:
        """Get information about the embedding model."""

        is_loaded = await self.embedding_service.is_loaded()

        return EmbeddingInfo(
            model_name=self.embedding_service.model_name,
            dimension=self.embedding_service.embedding_dimension,
            is_loaded=is_loaded,
        )
