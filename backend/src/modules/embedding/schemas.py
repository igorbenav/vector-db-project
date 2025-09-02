"""Schemas for embedding-enhanced operations."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EmbeddedChunkCreate(BaseModel):
    """Schema for creating chunk with auto-generated embeddings."""

    content: str = Field(min_length=1, max_length=10000, description="Text content to embed")
    document_id: int = Field(description="ID of the document this chunk belongs to")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional chunk metadata")

    embedding: Optional[List[float]] = Field(default=None, description="Optional manual embedding override")


class TextSearchRequest(BaseModel):
    """Schema for text-based search (auto-generates embeddings)."""

    query_text: str = Field(min_length=1, max_length=1000, description="Text query to search for")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class DocumentAutoChunk(BaseModel):
    """Schema for creating document with automatic chunking and embedding."""

    title: str = Field(min_length=1, max_length=500, description="Document title")
    content: str = Field(min_length=1, description="Full document content to chunk and embed")
    library_id: int = Field(description="ID of the library this document belongs to")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional document metadata")

    chunk_size: int = Field(default=500, ge=100, le=2000, description="Target size for each chunk (chars)")
    chunk_overlap: int = Field(default=50, ge=0, le=200, description="Overlap between consecutive chunks")


class EmbeddingInfo(BaseModel):
    """Information about the embedding model."""

    model_name: str = Field(description="Name of the embedding model")
    dimension: int = Field(description="Embedding dimension")
    is_loaded: bool = Field(description="Whether the model is loaded in memory")
