"""Common constants used across the application."""

from typing import Callable, Dict, Type

from fastapi import HTTPException, status

from .exceptions import (
    DomainError,
    PermissionDeniedError,
    ResourceExistsError,
    ResourceNotFoundError,
    ValidationError,
)

EXCEPTION_MAPPING: Dict[Type[DomainError], Callable[[str], HTTPException]] = {
    ResourceNotFoundError: lambda message: HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message),
    ResourceExistsError: lambda message: HTTPException(status_code=status.HTTP_409_CONFLICT, detail=message),
    ValidationError: lambda message: HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=message),
    PermissionDeniedError: lambda message: HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=message),
}
