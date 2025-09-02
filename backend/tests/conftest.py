"""Test configuration and fixtures for vector database project."""

import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient, ASGITransport
import os
import pytest
import sys
from pathlib import Path

# mypy: disable-error-code="import-untyped"
from testcontainers.postgres import PostgresContainer
from testcontainers.core.docker_client import DockerClient

from src.infrastructure.database.session import Base
from src.interfaces.main import app
from src.infrastructure.config.settings import get_settings
from src.modules.library.models import Library
from src.modules.document.models import Document
from src.modules.chunk.models import Chunk

# Set test environment variables
os.environ["SQLITE_URI"] = ":memory:"
os.environ["SQLITE_ASYNC_PREFIX"] = "sqlite+aiosqlite:///"
os.environ["SECRET_KEY"] = "test_secret_key_for_tests"

TEST_DATABASE_URL = get_settings().DATABASE_URL

backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))


def is_docker_running() -> bool:
    """Check if Docker daemon is running."""
    try:
        DockerClient()
        return True
    except Exception:
        return False


@pytest_asyncio.fixture(scope="session")
async def pg_container():
    """Create a PostgreSQL container for testing."""
    if not is_docker_running():
        pytest.skip("Docker is required, but not running")

    with PostgresContainer() as pg:
        yield pg


@pytest_asyncio.fixture(scope="function")
async def test_db_url(pg_container):
    """Create a proper asyncpg URL for PostgreSQL."""
    host = pg_container.get_container_host_ip()

    port_to_expose = 5432
    if hasattr(pg_container, "port_to_expose"):
        port_to_expose = pg_container.port_to_expose

    port = pg_container.get_exposed_port(port_to_expose)
    db = "test"
    user = "test"
    password = "test"

    if hasattr(pg_container, "POSTGRES_USER"):
        user = pg_container.POSTGRES_USER
    if hasattr(pg_container, "POSTGRES_PASSWORD"):
        password = pg_container.POSTGRES_PASSWORD
    if hasattr(pg_container, "POSTGRES_DB"):
        db = pg_container.POSTGRES_DB

    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


@pytest_asyncio.fixture(scope="function")
async def test_db_engine(test_db_url):
    """Create a SQLAlchemy engine for testing."""
    engine = create_async_engine(test_db_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_db(test_db_engine):
    """Create a test database session."""
    async_session = sessionmaker(test_db_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:  # type: ignore
        yield session


@pytest_asyncio.fixture(scope="function")
async def db_session(test_db):
    """Alias for test_db fixture to maintain compatibility with existing tests."""
    yield test_db


@pytest_asyncio.fixture(scope="function")
async def client(test_db_engine):
    """Create a test client with proper database session isolation for concurrent operations."""
    app.dependency_overrides = {}

    # Create a session factory for concurrent requests
    from sqlalchemy.orm import sessionmaker

    test_session_factory = sessionmaker(test_db_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        """Each request gets its own isolated database session."""
        async with test_session_factory() as session:
            yield session

    from src.infrastructure.database.session import async_session

    app.dependency_overrides[async_session] = override_get_db

    os.environ["POSTGRES_SERVER"] = "localhost"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides = {}


# Test fixtures for vector database entities
@pytest_asyncio.fixture
async def test_library(db_session: AsyncSession):
    """Create a test library."""
    library = Library(
        name="Test Library",
        description="A test library for unit testing",
        extra_metadata={"type": "test", "features": ["search", "analytics"]},
    )
    db_session.add(library)
    await db_session.commit()
    return {
        "id": library.id,
        "name": library.name,
        "description": library.description,
        "metadata": library.extra_metadata,
        "created_at": library.created_at,
        "updated_at": library.updated_at,
    }


@pytest_asyncio.fixture
async def test_library_2(db_session: AsyncSession):
    """Create a second test library for testing updates."""
    library = Library(name="Second Library", description="Another test library", extra_metadata={"type": "secondary"})
    db_session.add(library)
    await db_session.commit()
    return {
        "id": library.id,
        "name": library.name,
        "description": library.description,
        "metadata": library.extra_metadata,
        "created_at": library.created_at,
        "updated_at": library.updated_at,
    }


@pytest_asyncio.fixture
async def test_document(db_session: AsyncSession, test_library: dict):
    """Create a test document."""
    document = Document(library_id=test_library["id"], title="Test Document", extra_metadata={"source": "test", "pages": 10})
    db_session.add(document)
    await db_session.commit()
    return {
        "id": document.id,
        "library_id": document.library_id,
        "title": document.title,
        "metadata": document.extra_metadata,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }


@pytest_asyncio.fixture
async def test_document_2(db_session: AsyncSession, test_library: dict):
    """Create a second test document."""
    document = Document(library_id=test_library["id"], title="Second Document", extra_metadata={"source": "test2", "pages": 5})
    db_session.add(document)
    await db_session.commit()
    return {
        "id": document.id,
        "library_id": document.library_id,
        "title": document.title,
        "metadata": document.extra_metadata,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }


@pytest_asyncio.fixture
async def test_chunk(db_session: AsyncSession, test_document: dict):
    """Create a test chunk."""
    chunk = Chunk(
        document_id=test_document["id"],
        content="This is a test chunk with some sample content for vector search.",
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5] * 100,  # 500-dim vector
        extra_metadata={"position": 0, "section": "intro"},
    )
    db_session.add(chunk)
    await db_session.commit()
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "content": chunk.content,
        "embedding": chunk.embedding,
        "metadata": chunk.extra_metadata,
        "created_at": chunk.created_at,
        "updated_at": chunk.updated_at,
    }


@pytest_asyncio.fixture
async def test_chunk_2(db_session: AsyncSession, test_document: dict):
    """Create a second test chunk."""
    chunk = Chunk(
        document_id=test_document["id"],
        content="This is another test chunk with different content for testing.",
        embedding=[0.6, 0.7, 0.8, 0.9, 1.0] * 100,  # 500-dim vector
        extra_metadata={"position": 1, "section": "body"},
    )
    db_session.add(chunk)
    await db_session.commit()
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "content": chunk.content,
        "embedding": chunk.embedding,
        "metadata": chunk.extra_metadata,
        "created_at": chunk.created_at,
        "updated_at": chunk.updated_at,
    }


@pytest_asyncio.fixture
async def sample_library(db_session: AsyncSession):
    """Create a sample library for vector search tests."""
    library = Library(
        name="Sample Library",
        description="A sample library for vector search testing",
        extra_metadata={"type": "sample", "search_enabled": True},
    )
    db_session.add(library)
    await db_session.commit()
    return {
        "id": library.id,
        "name": library.name,
        "description": library.description,
        "metadata": library.extra_metadata,
        "created_at": library.created_at,
        "updated_at": library.updated_at,
    }
