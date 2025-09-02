from fastapi import APIRouter

from .chunk import router as chunk_router
from .document import router as document_router
from .embedding import router as embedding_router
from .library import router as library_router

router = APIRouter(prefix="/v1")
router.include_router(library_router)
router.include_router(document_router)
router.include_router(chunk_router)
router.include_router(embedding_router)


@router.get(
    "/health",
    summary="API Health Check",
    description="Simple health check endpoint for monitoring and container orchestration.",
    responses={
        200: {"description": "API is healthy and responding"},
    },
)
async def health_check():
    """Health check endpoint for Docker health checks."""
    return {"status": "healthy", "message": "Vector Database API is running"}
