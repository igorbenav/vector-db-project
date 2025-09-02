"""Script to create database tables from SQLAlchemy models."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.database.session import create_tables  # noqa: E402
import logging

logger = logging.getLogger(__name__)


async def main() -> None:
    """Create database tables."""
    logger.info("Creating database tables...")

    try:
        await create_tables()
        logger.info("✅ Database tables created successfully!")
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
