"""Pydantic schemas for document entities."""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..common.schemas import TimestampSchema


class DocumentBase(BaseModel):
    """Base schema for document data."""

    title: Annotated[str, Field(min_length=1, max_length=255, description="Document title")]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional document metadata")


class DocumentCreate(DocumentBase):
    """Schema for creating a new document."""

    library_id: int = Field(description="ID of the library this document belongs to")


class DocumentUpdate(BaseModel):
    """Schema for updating an existing document."""

    title: Optional[Annotated[str, Field(min_length=1, max_length=255)]] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentRead(TimestampSchema, DocumentBase):
    """Schema for reading document data."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    library_id: int
    chunk_count: Optional[int] = Field(default=0, description="Number of chunks in document")


class DocumentListResponse(BaseModel):
    """Schema for paginated document list response."""

    documents: List[DocumentRead]
    total: int
    has_more: bool
    offset: int
    limit: int
