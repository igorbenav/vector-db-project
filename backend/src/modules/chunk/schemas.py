"""Pydantic schemas for chunk entities."""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..common.schemas import TimestampSchema


class ChunkBase(BaseModel):
    """Base schema for chunk data."""

    content: Annotated[str, Field(min_length=1, max_length=10000, description="Text content of the chunk")]
    embedding: Annotated[List[float], Field(description="Vector embedding representation")]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional chunk metadata")

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("Embedding cannot be empty")
        if len(v) > 4096:
            raise ValueError("Embedding dimension too large (max 4096)")
        return v


class ChunkCreate(ChunkBase):
    """Schema for creating a new chunk."""

    document_id: int = Field(description="ID of the document this chunk belongs to")


class ChunkCreateWithText(BaseModel):
    """Schema for creating chunk with auto-generated embeddings from text."""

    content: Annotated[str, Field(min_length=1, max_length=10000, description="Text content to embed")]
    document_id: int = Field(description="ID of the document this chunk belongs to")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional chunk metadata")

    embedding: Optional[List[float]] = Field(default=None, description="Optional manual embedding override")


class ChunkUpdate(BaseModel):
    """Schema for updating an existing chunk."""

    content: Optional[Annotated[str, Field(min_length=1, max_length=10000)]] = None
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            if not v:
                raise ValueError("Embedding cannot be empty")
            if len(v) > 4096:
                raise ValueError("Embedding dimension too large (max 4096)")
        return v


class ChunkRead(TimestampSchema, ChunkBase):
    """Schema for reading chunk data."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int


class ChunkListResponse(BaseModel):
    """Schema for paginated chunk list response."""

    chunks: List[ChunkRead]
    total: int
    has_more: bool
    offset: int
    limit: int
