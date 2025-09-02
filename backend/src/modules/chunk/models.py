"""SQLAlchemy models for chunk entities."""

from typing import Any, Dict, List, Optional

from sqlalchemy import ARRAY, JSON, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from ...infrastructure.database.models import TimestampMixin
from ...infrastructure.database.session import Base


class Chunk(Base, TimestampMixin):
    """Chunk model for storing text content with vector embeddings.

    A chunk represents a piece of text with its associated
    vector embedding for similarity search. Chunks are the
    fundamental units for vector database operations.
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, init=False)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[List[float]] = mapped_column(ARRAY(Float))
    extra_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=None)
