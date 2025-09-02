"""Utility functions for mapping domain exceptions to HTTP exceptions."""

from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ..constants import EXCEPTION_MAPPING
from ..exceptions import DomainError


def map_exception(error: DomainError) -> HTTPException:
    """Map a domain exception to a corresponding HTTP exception."""
    for exception_class, mapper in EXCEPTION_MAPPING.items():
        if isinstance(error, exception_class):
            return mapper(str(error))

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(error)}"
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers for domain exceptions."""

    @app.exception_handler(DomainError)
    async def domain_exception_handler(request: Request, exc: DomainError) -> JSONResponse:
        """Convert domain exceptions to appropriate HTTP responses."""
        http_exception = map_exception(exc)
        return JSONResponse(
            status_code=http_exception.status_code,
            content={"detail": http_exception.detail},
        )


def handle_exception(error: Exception) -> Optional[HTTPException]:
    """
    Handle an exception and return an appropriate HTTP exception if possible.

    For use in route handlers when you want to handle exceptions manually.

    Args:
        error: The exception to handle

    Returns:
        An HTTPException if the error can be mapped, None otherwise
    """
    if isinstance(error, DomainError):
        return map_exception(error)
    elif isinstance(error, HTTPException):
        return error
    return None
