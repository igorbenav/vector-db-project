"""FastAPI dependencies for use in API endpoints."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...infrastructure.database import async_session
from ...modules.chunk.services import ChunkService
from ...modules.document.services import DocumentService
from ...modules.embedding.services import (
    EmbeddingChunkService,
    EmbeddingDocumentService,
    EmbeddingInfoService,
    EmbeddingLibraryService,
)
from ...modules.library.services import LibraryService

DbSession = Annotated[AsyncSession, Depends(async_session)]


def get_library_service() -> LibraryService:
    """Dependency for providing a LibraryService instance."""
    return LibraryService()


def get_document_service() -> DocumentService:
    """Dependency for providing a DocumentService instance."""
    return DocumentService()


def get_chunk_service() -> ChunkService:
    """Dependency for providing a ChunkService instance."""
    return ChunkService()


def get_embedding_chunk_service() -> EmbeddingChunkService:
    """Dependency for providing an EmbeddingChunkService instance."""
    return EmbeddingChunkService()


def get_embedding_library_service() -> EmbeddingLibraryService:
    """Dependency for providing an EmbeddingLibraryService instance."""
    return EmbeddingLibraryService()


def get_embedding_document_service() -> EmbeddingDocumentService:
    """Dependency for providing an EmbeddingDocumentService instance."""
    return EmbeddingDocumentService()


def get_embedding_info_service() -> EmbeddingInfoService:
    """Dependency for providing an EmbeddingInfoService instance."""
    return EmbeddingInfoService()
