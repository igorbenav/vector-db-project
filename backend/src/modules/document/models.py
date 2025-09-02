"""SQLAlchemy models for document entities."""

from typing import Any, Dict, Optional

from sqlalchemy import JSON, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ...infrastructure.database.models import TimestampMixin
from ...infrastructure.database.session import Base


class Document(Base, TimestampMixin):
    """Document model for organizing chunks within a library.

    A document represents a logical grouping of text chunks
    that belong together, such as pages from a single file
    or sections from an article.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, init=False)
    library_id: Mapped[int] = mapped_column(Integer, ForeignKey("libraries.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    extra_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=None)
