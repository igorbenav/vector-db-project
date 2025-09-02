import json
from asyncio import Event
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Dict, List, Optional

import anyio
import fastapi
from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from .config.settings import (
    DatabaseSettings,
    EnvironmentOption,
    EnvironmentSettings,
    Settings,
    get_settings,
)
from .database.session import create_tables


async def set_threadpool_tokens(number_of_tokens: int = 100) -> None:
    """Configure the number of threadpool tokens for anyio."""
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = number_of_tokens


def lifespan_factory(
    settings: Settings,
    create_tables_on_startup: bool = True,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Factory to create a lifespan async context manager for a FastAPI app.

    Note: setup_early_instrumentation() should be called before using this factory
    to ensure proper instrumentation timing.

    Args:
        settings: Application settings
        create_tables_on_startup: Whether to create database tables on startup

    Returns:
        An async context manager for FastAPI's lifespan
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        initialization_complete = Event()
        app.state.initialization_complete = initialization_complete

        await set_threadpool_tokens()

        try:
            if isinstance(settings, DatabaseSettings) and create_tables_on_startup:
                await create_tables()

            initialization_complete.set()
            yield

        finally:
            pass

    return lifespan


def create_application(
    router: APIRouter,
    settings: Optional[Settings] = None,
    lifespan: Optional[Callable[[FastAPI], AbstractAsyncContextManager[None]]] = None,
    create_tables_on_startup: Optional[bool] = None,
    enable_cors: Optional[bool] = None,
    cors_origins: Optional[List[str]] = None,
    enable_docs_in_production: Optional[bool] = None,
    docs_production_dependency: Optional[Callable[..., Any]] = None,
    enable_gzip: Optional[bool] = None,
    openapi_prefix: Optional[str] = None,
    # OpenAPI metadata
    title: Optional[str] = None,
    summary: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
    terms_of_service: Optional[str] = None,
    contact: Optional[Dict[str, str]] = None,
    license_info: Optional[Dict[str, str]] = None,
    openapi_tags: Optional[List[Dict[str, Any]]] = None,
    # Documentation URLs
    docs_url: Optional[str] = None,
    redoc_url: Optional[str] = None,
    openapi_url: Optional[str] = None,
    **kwargs: Any,
) -> FastAPI:
    """Creates and configures a FastAPI application based on the provided settings.

    Args:
        router: The APIRouter containing the routes for the application
        settings: Application settings (uses get_settings() if None)
        lifespan: Optional lifespan function for the FastAPI app. If None, uses the default
            lifespan_factory which handles database, cache, rate limiter, and instrumentation.
        create_tables_on_startup: Whether to create database tables on startup.
            Defaults to settings.CREATE_TABLES_ON_STARTUP if None.
        enable_cors: Whether to enable CORS middleware.
            Defaults to settings.CORS_ENABLED if None.
        cors_origins: List of allowed origins for CORS.
            Defaults to settings.CORS_ORIGINS if None.
        enable_docs_in_production: Whether to enable API docs in production.
            Defaults to settings.ENABLE_DOCS_IN_PRODUCTION if None.
        docs_production_dependency: Dependency to protect docs in production.
        enable_gzip: Whether to enable GZip compression middleware.
            Defaults to settings.GZIP_ENABLED if None.
        openapi_prefix: Prefix to apply to OpenAPI paths.
            Defaults to settings.OPENAPI_PREFIX if None.
        title: The title of the API.
        summary: A short summary of the API.
        description: A detailed description of the API (supports Markdown).
        version: The version of the API.
        terms_of_service: URL to the Terms of Service for the API.
        contact: Contact information for the API (dict with name, url, email).
        license_info: License information for the API (dict with name, url or identifier).
        openapi_tags: List of dictionaries containing tag metadata.
        docs_url: URL for Swagger UI docs (set to None to disable).
        redoc_url: URL for ReDoc docs (set to None to disable).
        openapi_url: URL for OpenAPI schema (set to None to disable OpenAPI and docs).
        **kwargs: Additional keyword arguments passed to FastAPI constructor

    Returns:
        A configured FastAPI application
    """

    if settings is None:
        settings = get_settings()

    _create_tables_on_startup = True
    if create_tables_on_startup is not None:
        _create_tables_on_startup = create_tables_on_startup
    elif hasattr(settings, "CREATE_TABLES_ON_STARTUP"):
        _create_tables_on_startup = settings.CREATE_TABLES_ON_STARTUP

    _enable_cors = True
    if enable_cors is not None:
        _enable_cors = enable_cors
    elif hasattr(settings, "CORS_ENABLED"):
        _enable_cors = settings.CORS_ENABLED

    _cors_origins: List[str] = ["*"]
    if cors_origins is not None:
        _cors_origins = cors_origins
    elif hasattr(settings, "CORS_ORIGINS_LIST"):
        _cors_origins = settings.CORS_ORIGINS_LIST

    _enable_docs_in_production = False
    if enable_docs_in_production is not None:
        _enable_docs_in_production = enable_docs_in_production
    elif hasattr(settings, "ENABLE_DOCS_IN_PRODUCTION"):
        _enable_docs_in_production = settings.ENABLE_DOCS_IN_PRODUCTION

    _enable_gzip = True
    if enable_gzip is not None:
        _enable_gzip = enable_gzip
    elif hasattr(settings, "GZIP_ENABLED"):
        _enable_gzip = settings.GZIP_ENABLED

    _openapi_prefix = ""
    if openapi_prefix is not None:
        _openapi_prefix = openapi_prefix
    elif hasattr(settings, "OPENAPI_PREFIX"):
        _openapi_prefix = settings.OPENAPI_PREFIX

    metadata: Dict[str, Any] = {"openapi_prefix": _openapi_prefix}

    if title is not None:
        metadata["title"] = title
    elif hasattr(settings, "API_TITLE") and settings.API_TITLE:
        metadata["title"] = settings.API_TITLE
    elif hasattr(settings, "APP_NAME"):
        metadata["title"] = settings.APP_NAME

    if summary is not None:
        metadata["summary"] = summary
    elif hasattr(settings, "API_SUMMARY") and settings.API_SUMMARY:
        metadata["summary"] = settings.API_SUMMARY

    if description is not None:
        metadata["description"] = description
    elif hasattr(settings, "API_DESCRIPTION") and settings.API_DESCRIPTION:
        metadata["description"] = settings.API_DESCRIPTION
    elif hasattr(settings, "APP_DESCRIPTION"):
        metadata["description"] = settings.APP_DESCRIPTION

    if version is not None:
        metadata["version"] = version
    elif hasattr(settings, "API_VERSION") and settings.API_VERSION:
        metadata["version"] = settings.API_VERSION
    elif hasattr(settings, "VERSION"):
        metadata["version"] = settings.VERSION

    if terms_of_service is not None:
        metadata["terms_of_service"] = terms_of_service
    elif hasattr(settings, "API_TERMS_OF_SERVICE") and settings.API_TERMS_OF_SERVICE:
        metadata["terms_of_service"] = settings.API_TERMS_OF_SERVICE

    if contact is not None:
        metadata["contact"] = contact
    else:
        contact_dict = {}

        # From API_* settings
        if hasattr(settings, "API_CONTACT_NAME") and settings.API_CONTACT_NAME:
            contact_dict["name"] = settings.API_CONTACT_NAME
        elif hasattr(settings, "CONTACT_NAME") and settings.CONTACT_NAME:
            contact_dict["name"] = settings.CONTACT_NAME

        if hasattr(settings, "API_CONTACT_EMAIL") and settings.API_CONTACT_EMAIL:
            contact_dict["email"] = settings.API_CONTACT_EMAIL
        elif hasattr(settings, "CONTACT_EMAIL") and settings.CONTACT_EMAIL:
            contact_dict["email"] = settings.CONTACT_EMAIL

        if hasattr(settings, "API_CONTACT_URL") and settings.API_CONTACT_URL:
            contact_dict["url"] = settings.API_CONTACT_URL

        if contact_dict:
            metadata["contact"] = contact_dict

    if license_info is not None:
        metadata["license_info"] = license_info
    else:
        license_dict = {}

        if hasattr(settings, "API_LICENSE_NAME") and settings.API_LICENSE_NAME:
            license_dict["name"] = settings.API_LICENSE_NAME
        elif hasattr(settings, "LICENSE_NAME") and settings.LICENSE_NAME:
            license_dict["name"] = settings.LICENSE_NAME

        if hasattr(settings, "API_LICENSE_URL") and settings.API_LICENSE_URL:
            license_dict["url"] = settings.API_LICENSE_URL

        if hasattr(settings, "API_LICENSE_IDENTIFIER") and settings.API_LICENSE_IDENTIFIER:
            license_dict["identifier"] = settings.API_LICENSE_IDENTIFIER

        if license_dict:
            metadata["license_info"] = license_dict

    if openapi_tags is not None:
        metadata["openapi_tags"] = openapi_tags
    elif hasattr(settings, "API_TAGS_METADATA") and settings.API_TAGS_METADATA:
        try:
            tags_metadata = json.loads(settings.API_TAGS_METADATA)
            metadata["openapi_tags"] = tags_metadata
        except json.JSONDecodeError:
            pass

    _docs_url = "/docs"
    if docs_url is not None:
        _docs_url = docs_url
    elif hasattr(settings, "DOCS_URL"):
        _docs_url = settings.DOCS_URL

    _redoc_url = "/redoc"
    if redoc_url is not None:
        _redoc_url = redoc_url
    elif hasattr(settings, "REDOC_URL"):
        _redoc_url = settings.REDOC_URL

    _openapi_url = "/openapi.json"
    if openapi_url is not None:
        _openapi_url = openapi_url
    elif hasattr(settings, "OPENAPI_URL"):
        _openapi_url = settings.OPENAPI_URL

    metadata["docs_url"] = _docs_url
    metadata["redoc_url"] = _redoc_url
    metadata["openapi_url"] = _openapi_url

    kwargs.update(metadata)

    hide_docs = (
        isinstance(settings, EnvironmentSettings)
        and settings.ENVIRONMENT == EnvironmentOption.PRODUCTION
        and not _enable_docs_in_production
    )
    if hide_docs:
        kwargs.update({"docs_url": None, "redoc_url": None, "openapi_url": None})

    if lifespan is None:
        lifespan = lifespan_factory(settings, create_tables_on_startup=_create_tables_on_startup)

    application = FastAPI(lifespan=lifespan, **kwargs)

    application.include_router(router)

    if _enable_cors:
        cors_settings_dict: Dict[str, Any] = {
            "allow_origins": _cors_origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

        if hasattr(settings, "CORS_ALLOW_CREDENTIALS"):
            cors_settings_dict["allow_credentials"] = settings.CORS_ALLOW_CREDENTIALS

        if hasattr(settings, "CORS_ALLOW_METHODS"):
            methods = settings.CORS_ALLOW_METHODS
            cors_settings_dict["allow_methods"] = methods.split(",") if isinstance(methods, str) else methods

        if hasattr(settings, "CORS_ALLOW_HEADERS"):
            headers = settings.CORS_ALLOW_HEADERS
            cors_settings_dict["allow_headers"] = headers.split(",") if isinstance(headers, str) else headers

        application.add_middleware(CORSMiddleware, **cors_settings_dict)

    if _enable_gzip:
        gzip_min_size = getattr(settings, "GZIP_MINIMUM_SIZE", 1000) if hasattr(settings, "GZIP_MINIMUM_SIZE") else 1000
        application.add_middleware(GZipMiddleware, minimum_size=gzip_min_size)

    show_docs = isinstance(settings, EnvironmentSettings) and (
        settings.ENVIRONMENT != EnvironmentOption.PRODUCTION or _enable_docs_in_production
    )

    if show_docs:
        docs_router = APIRouter()

        if docs_production_dependency is not None:
            docs_router = APIRouter(dependencies=[Depends(docs_production_dependency)])

        @docs_router.get("/docs", include_in_schema=False)
        async def get_swagger_documentation() -> fastapi.responses.HTMLResponse:
            return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")

        @docs_router.get("/redoc", include_in_schema=False)
        async def get_redoc_documentation() -> fastapi.responses.HTMLResponse:
            return get_redoc_html(openapi_url="/openapi.json", title="redoc")

        @docs_router.get("/openapi.json", include_in_schema=False)
        async def openapi() -> Dict[str, Any]:
            return get_openapi(
                title=metadata.get("title", "API"),
                version=metadata.get("version", "0.1.0"),
                description=metadata.get("description", ""),
                routes=application.routes,
            )

        application.include_router(docs_router)

    return application
