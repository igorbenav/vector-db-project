"""API endpoints with automatic embedding generation."""

from fastapi import APIRouter, Depends, HTTPException, status

from ....modules.chunk.schemas import ChunkRead
from ....modules.document.schemas import DocumentRead
from ....modules.embedding import EmbeddingChunkService, EmbeddingDocumentService, EmbeddingInfoService, EmbeddingLibraryService
from ....modules.embedding.schemas import DocumentAutoChunk, EmbeddedChunkCreate, EmbeddingInfo, TextSearchRequest
from ....modules.library.schemas import VectorSearchResponse
from ..dependencies import (
    DbSession,
    get_embedding_chunk_service,
    get_embedding_document_service,
    get_embedding_info_service,
    get_embedding_library_service,
)

router = APIRouter(prefix="/embedding", tags=["Embedding"])


@router.get(
    "/info",
    summary="Get Embedding Model Information",
    description="""Get detailed information about the current embedding model.
    
    Returns information about the sentence-transformer model used for
    generating vector embeddings, including:
    
    - Model name and version
    - Embedding dimension size
    - Model loading status
    - Performance characteristics
    
    This endpoint is useful for understanding the embedding capabilities
    and ensuring the model is properly loaded and ready for use.
    """,
    responses={
        200: {"description": "Embedding model information returned successfully"},
        500: {"description": "Error accessing embedding model"},
    },
)
async def get_embedding_info(service: EmbeddingInfoService = Depends(get_embedding_info_service)) -> EmbeddingInfo:
    """Get information about the embedding model."""
    return await service.get_embedding_info()


@router.post(
    "/chunks",
    status_code=status.HTTP_201_CREATED,
    summary="Create Chunk with Auto-Generated Embeddings",
    description="""
    Create a new chunk with automatically generated embeddings from text content.
    
    This endpoint uses sentence-transformers to generate 768-dimensional embeddings
    from the provided text content. You can optionally provide your own embeddings
    to override the auto-generation.
    
    The embedding model used is all-mpnet-base-v2, which provides high-quality
    semantic embeddings suitable for similarity search.
    """,
    responses={
        201: {"description": "Chunk created successfully with embeddings"},
        400: {"description": "Invalid chunk data"},
        404: {"description": "Document not found"},
    },
)
async def create_chunk_with_embeddings(
    chunk_data: EmbeddedChunkCreate, db: DbSession, service: EmbeddingChunkService = Depends(get_embedding_chunk_service)
) -> ChunkRead:
    """Create chunk with auto-generated embeddings from text."""
    try:
        return await service.create_chunk_with_text(chunk_data, db)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post(
    "/libraries/{library_id}/text-search",
    summary="Search Library with Text Query",
    description="""
    Perform semantic search using natural language text queries.
    
    This endpoint automatically converts your text query into vector embeddings
    and performs k-NN similarity search across all chunks in the library.
    
    - **query_text**: Natural language search query
    - **k**: Number of results to return (1-100)
    - **metadata_filter**: Optional filters on chunk metadata
    
    The search uses cosine similarity between query and chunk embeddings,
    returning results ranked by semantic similarity.
    """,
    responses={
        200: {"description": "Search results returned"},
        404: {"description": "Library not found"},
        400: {"description": "Invalid search parameters"},
    },
)
async def text_search_library(
    library_id: int,
    search_request: TextSearchRequest,
    db: DbSession,
    service: EmbeddingLibraryService = Depends(get_embedding_library_service),
) -> VectorSearchResponse:
    """Search library using text query (auto-generates embedding)."""
    try:
        result = await service.text_search(library_id, search_request, db)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post(
    "/documents/auto-chunk",
    status_code=status.HTTP_201_CREATED,
    summary="Create Document with Auto-Chunking and Embeddings",
    description="""
    Create a document with automatic text chunking and embedding generation.
    
    This endpoint takes a large document and:
    1. Splits it into smaller, manageable chunks (with configurable size/overlap)
    2. Generates embeddings for each chunk using sentence-transformers
    3. Creates the document and all chunks in a single operation
    
    **Chunking Parameters:**
    - **chunk_size**: Target size for each chunk (100-2000 characters)
    - **chunk_overlap**: Overlap between consecutive chunks (0-200 characters)
    
    **Benefits:**
    - Handles large documents seamlessly
    - Maintains semantic coherence within chunks
    - Optimizes for vector search performance
    - Preserves chunk relationships via metadata
    """,
    responses={
        201: {"description": "Document created with auto-chunking and embeddings"},
        400: {"description": "Invalid document data"},
        404: {"description": "Library not found"},
    },
)
async def create_document_with_auto_chunks(
    doc_data: DocumentAutoChunk, db: DbSession, service: EmbeddingDocumentService = Depends(get_embedding_document_service)
) -> DocumentRead:
    """Create document with automatic chunking and embedding."""
    try:
        return await service.create_document_with_auto_chunks(doc_data, db)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
