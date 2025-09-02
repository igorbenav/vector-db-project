import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..infrastructure.app_factory import create_application, lifespan_factory
from ..infrastructure.config.settings import get_settings
from ..interfaces.api import router as api_router
from ..interfaces.ui import router as ui_router

settings = get_settings()


@asynccontextmanager
async def simple_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Simple lifespan for vector database."""
    default_lifespan = lifespan_factory(settings)
    async with default_lifespan(app):
        yield


app = create_application(
    router=api_router,
    settings=settings,
    lifespan=simple_lifespan,
    title="Vector Database API",
    summary="REST API for vector database with k-NN search",
    description="""
    # Vector Database API

    This API provides vector database capabilities with:

    * 📊 **Vector Operations**: Store and search vector embeddings
    * 🔍 **k-NN Search**: Fast similarity search with multiple algorithms
    * 📚 **Document Management**: Organize chunks into documents and libraries
    * 🏗️ **Clean Architecture**: Modular design with services and repositories

    ## Features

    - CRUD operations for libraries, documents, and chunks
    - Multiple indexing algorithms for vector search
    - Metadata filtering capabilities
    - RESTful API design
    """,
    version="0.1.0",
)

app.include_router(ui_router)

static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTMX interface."""
    index_file = os.path.join(templates_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(
            content="""
        <html>
            <head><title>Vector Database</title></head>
            <body>
                <h1>Vector Database API</h1>
                <p>The HTMX interface is being set up. Please check back soon!</p>
                <p><a href="/docs">View API Documentation</a></p>
            </body>
        </html>
        """
        )
