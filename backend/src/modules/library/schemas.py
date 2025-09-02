"""Pydantic schemas for library entities."""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..common.schemas import TimestampSchema


class LibraryBase(BaseModel):
    """Base schema for library data."""

    name: Annotated[str, Field(min_length=1, max_length=255, description="Library name")]
    description: Optional[str] = Field(default=None, max_length=1000, description="Library description")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional library metadata")


class LibraryCreate(LibraryBase):
    """Schema for creating a new library."""

    pass


class LibraryUpdate(BaseModel):
    """Schema for updating an existing library."""

    name: Optional[Annotated[str, Field(min_length=1, max_length=255)]] = None
    description: Optional[str] = Field(default=None, max_length=1000)
    metadata: Optional[Dict[str, Any]] = None


class LibraryRead(TimestampSchema, LibraryBase):
    """Schema for reading library data."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    document_count: Optional[int] = Field(default=0, description="Number of documents in library")
    chunk_count: Optional[int] = Field(default=0, description="Total number of chunks in library")


class LibraryListResponse(BaseModel):
    """Schema for paginated library list response."""

    libraries: List[LibraryRead]
    total: int
    has_more: bool
    offset: int
    limit: int


class VectorSearchRequest(BaseModel):
    """Schema for vector search request."""

    query_embedding: Annotated[List[float], Field(description="Query vector embedding")]
    k: Annotated[int, Field(ge=1, le=100, description="Number of nearest neighbors to return")] = 10
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters")

    @field_validator("query_embedding")
    @classmethod
    def validate_query_embedding(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("Query embedding cannot be empty")
        if len(v) > 4096:
            raise ValueError("Query embedding dimension too large (max 4096)")
        return v


class SearchResult(BaseModel):
    """Schema for individual search result."""

    chunk_id: int
    document_id: int
    content: str
    similarity_score: float = Field(description="Cosine similarity score (0-1)")
    metadata: Dict[str, Any]


class VectorSearchResponse(BaseModel):
    """Schema for vector search response."""

    results: List[SearchResult]
    query_time_ms: float = Field(description="Query execution time in milliseconds")
    total_chunks_searched: int = Field(description="Total number of chunks searched")


class TextSearchRequest(BaseModel):
    """Schema for text-based search (auto-generates embeddings)."""

    query_text: str = Field(min_length=1, max_length=1000, description="Text query to search for")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
