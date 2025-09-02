"""SQLAlchemy models for library entities."""

from typing import Any, Dict, Optional

from sqlalchemy import JSON, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ...infrastructure.database.models import TimestampMixin
from ...infrastructure.database.session import Base


class Library(Base, TimestampMixin):
    """Library model for organizing document collections.

    A library represents a collection of documents that can be
    searched and indexed together. Libraries provide isolation
    and organization for different document sets.
    """

    __tablename__ = "libraries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, init=False)
    name: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[Optional[str]] = mapped_column(String(1000), default=None)
    extra_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=None)
