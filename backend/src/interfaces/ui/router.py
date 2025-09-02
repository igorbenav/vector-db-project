"""HTMX UI router for interactive web interface."""

import json
import uuid
from typing import Optional, cast

from fastapi import APIRouter, BackgroundTasks, Depends, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ...infrastructure.database.session import async_session, local_session
from ...infrastructure.indexing.base import IndexType
from ...infrastructure.indexing.manager import index_manager
from ...modules.document.schemas import DocumentRead
from ...modules.document.services import DocumentService
from ...modules.embedding.schemas import DocumentAutoChunk, TextSearchRequest
from ...modules.embedding.services import EmbeddingDocumentService, EmbeddingInfoService, EmbeddingLibraryService
from ...modules.library.schemas import LibraryCreate
from ...modules.library.services import LibraryService

router = APIRouter(prefix="/ui", tags=["ui"])

processing_status = {}


async def process_document_background(doc_data: DocumentAutoChunk, document_id: str):
    """Background task for document processing."""
    try:
        async with local_session() as db:
            embedding_doc_service = EmbeddingDocumentService()
            document = await embedding_doc_service.create_document_with_auto_chunks(doc_data, db)

            processing_status[document_id] = {"status": "completed", "document": document, "error": None}
    except Exception as e:
        processing_status[document_id] = {"status": "error", "document": None, "error": str(e)}


@router.post("/libraries", response_class=HTMLResponse)
async def create_library_ui(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    index_type: str = Form("LINEAR_SEARCH"),
    db: AsyncSession = Depends(async_session),
):
    """Create library via HTMX form."""
    try:
        library_service = LibraryService()
        library_data = LibraryCreate(name=name, description=description or "")
        library = await library_service.create_library(library_data, db)

        if not library:
            return HTMLResponse("""
            <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                <strong>Error:</strong> Failed to create library
            </div>
            """)

        selected_index_type = IndexType.IVF if index_type == "IVF" else IndexType.LINEAR_SEARCH
        await index_manager.get_or_create_index(
            library_id=library.id, index_type=selected_index_type, embedding_dimension=768, db=db
        )

        index_name = "IVF (Inverted File Index)" if index_type == "IVF" else "Linear Search (Brute Force)"

        return HTMLResponse(f"""
        <div class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
            <strong>Success!</strong> Created library "{library.name}" (ID: {library.id})
            <div class="text-sm mt-1">Index algorithm: {index_name}</div>
        </div>
        """)
    except Exception as e:
        return HTMLResponse(f"""
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <strong>Error:</strong> {str(e)}
        </div>
        """)


@router.get("/libraries", response_class=HTMLResponse)
async def list_libraries_ui(db: AsyncSession = Depends(async_session)):
    """List libraries via HTMX."""
    try:
        library_service = LibraryService()
        libraries_response = await library_service.get_libraries(db)
        libraries = libraries_response["data"]

        if not libraries:
            return HTMLResponse("""
            <div class="text-gray-500 italic">No libraries found. Create one above!</div>
            """)

        html = '<div class="space-y-2">'
        for library in libraries:
            index_stats = index_manager.get_index_stats(library["id"])
            index_info = ""
            if index_stats:
                index_info = f" | {index_stats['index_type'].replace('_', ' ').title()} index"

            html += f"""
            <div class="bg-gray-50 p-3 rounded border">
                <div class="flex justify-between items-center">
                    <div class="flex-1">
                        <strong>#{library["id"]}: {library["name"]}</strong>
                        <p class="text-sm text-gray-600">{library.get("description", "")}</p>
                        <div class="text-xs text-gray-500 mt-1">
                            Created: {library["created_at"]} | {library["document_count"]} docs, {library["chunk_count"]} chunks{index_info}
                        </div>
                    </div>
                    <div class="flex gap-2 ml-4">
                        <button hx-get="/ui/libraries/{library["id"]}/documents"
                                hx-target="#library-{library["id"]}-docs"
                                hx-swap="innerHTML"
                                class="bg-blue-500 hover:bg-blue-600 text-white px-2 py-1 rounded text-xs">
                            View Docs
                        </button>
                        <button hx-delete="/ui/libraries/{library["id"]}"
                                hx-target="closest div"
                                hx-swap="outerHTML"
                                hx-confirm="Delete library '{library["name"]}'?"
                                class="bg-red-500 hover:bg-red-600 text-white px-2 py-1 rounded text-xs">
                            Delete
                        </button>
                    </div>
                </div>
                <div id="library-{library["id"]}-docs" class="mt-2"></div>
            </div>
            """  # noqa: E501
        html += "</div>"

        return HTMLResponse(html)
    except Exception as e:
        return HTMLResponse(f"""
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <strong>Error:</strong> {str(e)}
        </div>
        """)


@router.delete("/libraries/{library_id}", response_class=HTMLResponse)
async def delete_library_ui(library_id: int, db: AsyncSession = Depends(async_session)):
    """Delete library via HTMX."""
    try:
        library_service = LibraryService()
        await library_service.delete_library(library_id, db)
        return HTMLResponse("")
    except Exception as e:
        return HTMLResponse(f"""
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <strong>Error:</strong> {str(e)}
        </div>
        """)


@router.get("/libraries/{library_id}/documents", response_class=HTMLResponse)
async def list_library_documents_ui(library_id: int, db: AsyncSession = Depends(async_session)):
    """List documents in a library via HTMX."""
    try:
        document_service = DocumentService()
        documents_response = await document_service.get_documents_by_library(library_id, db)
        documents = documents_response["data"]

        if not documents:
            return HTMLResponse("""
            <div class="text-gray-500 italic text-sm mt-2">No documents in this library.</div>
            """)

        html = '<div class="mt-2 space-y-1">'
        for doc in documents:
            html += f"""
            <div class="bg-white p-2 rounded border border-gray-200 text-sm">
                <div class="flex justify-between items-center">
                    <div>
                        <strong>Doc #{doc["id"]}: {doc["title"]}</strong>
                        <span class="text-gray-500 ml-2">({doc.get("chunk_count", 0)} chunks)</span>
                    </div>
                    <button hx-delete="/ui/documents/{doc["id"]}"
                            hx-target="closest div"
                            hx-swap="outerHTML"
                            hx-confirm="Delete document '{doc["title"]}'?"
                            class="bg-red-400 hover:bg-red-500 text-white px-2 py-1 rounded text-xs">
                        Delete
                    </button>
                </div>
            </div>
            """
        html += "</div>"

        return HTMLResponse(html)
    except Exception as e:
        return HTMLResponse(f"""
        <div class="text-red-500 text-sm">Error loading documents: {str(e)}</div>
        """)


@router.delete("/documents/{document_id}", response_class=HTMLResponse)
async def delete_document_ui(document_id: int, db: AsyncSession = Depends(async_session)):
    """Delete document via HTMX."""
    try:
        document_service = DocumentService()

        success = await document_service.delete_document(document_id, db)
        if not success:
            return HTMLResponse("""
            <div class="bg-red-100 border border-red-400 text-red-700 px-2 py-1 rounded text-sm">
                Document not found
            </div>
            """)

        return HTMLResponse("")
    except Exception as e:
        return HTMLResponse(f"""
        <div class="bg-red-100 border border-red-400 text-red-700 px-2 py-1 rounded text-sm">
            Error: {str(e)}
        </div>
        """)


@router.post("/documents/auto-chunk", response_class=HTMLResponse)
async def create_document_ui(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    content: str = Form(...),
    library_id: int = Form(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    metadata: str = Form("{}"),
    db: AsyncSession = Depends(async_session),
):
    """Create document with auto-chunking via HTMX form."""
    try:
        try:
            metadata_dict = json.loads(metadata) if metadata.strip() else {}
        except json.JSONDecodeError:
            metadata_dict = {}

        doc_data = DocumentAutoChunk(
            title=title,
            content=content,
            library_id=library_id,
            metadata=metadata_dict,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        job_id = str(uuid.uuid4())
        processing_status[job_id] = {"status": "processing", "document": None, "error": None}

        background_tasks.add_task(process_document_background, doc_data, job_id)

        return HTMLResponse(f"""
        <div class="bg-blue-100 border border-blue-400 text-blue-700 px-4 py-3 rounded">
            <strong>Processing Started!</strong> Document "{title}" is being processed in the background.
            <div class="mt-2 text-sm">
                <strong>Job Details:</strong>
                <ul class="list-disc list-inside mt-1">
                    <li>Library ID: {library_id}</li>
                    <li>Chunk size: {chunk_size}</li>
                    <li>Using library's index algorithm</li>
                    <li>Job ID: {job_id}</li>
                </ul>
                <div class="mt-2">
                    <button hx-get="/ui/documents/status/{job_id}"
                            hx-target="#document-result"
                            hx-swap="innerHTML"
                            hx-trigger="click, every 2s"
                            class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm">
                        Check Status
                    </button>
                </div>
            </div>
        </div>
        """)
    except Exception as e:
        return HTMLResponse(f"""
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <strong>Error:</strong> {str(e)}
        </div>
        """)


@router.get("/documents/status/{job_id}", response_class=HTMLResponse)
async def check_document_status_ui(job_id: str):
    """Check document processing status via HTMX."""
    if job_id not in processing_status:
        return HTMLResponse("""
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <strong>Error:</strong> Job not found
        </div>
        """)

    status = processing_status[job_id]

    if status["status"] == "processing":
        return HTMLResponse(f"""
        <div class="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded">
            <div class="flex items-center">
                <div class="spinner mr-2"></div>
                <strong>Still Processing...</strong> Generating embeddings and indexing document.
            </div>
            <div class="mt-2">
                <button hx-get="/ui/documents/status/{job_id}"
                        hx-target="#document-result"
                        hx-swap="innerHTML"
                        hx-trigger="click, every 2s"
                        class="bg-yellow-500 hover:bg-yellow-600 text-white px-3 py-1 rounded text-sm">
                    Refresh Status
                </button>
            </div>
        </div>
        """)
    elif status["status"] == "completed":
        document = cast(DocumentRead, status["document"])
        del processing_status[job_id]
        return HTMLResponse(f"""
        <div class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
            <strong>Success!</strong> Document "{document.title}" (ID: {document.id}) processed with {document.chunk_count} chunks
            <div class="mt-2 text-sm">
                <strong>Final Details:</strong>
                <ul class="list-disc list-inside mt-1">
                    <li>Library ID: {document.library_id}</li>
                    <li>Chunks created: {document.chunk_count}</li>
                    <li>Created at: {document.created_at.strftime("%Y-%m-%d %H:%M:%S")}</li>
                </ul>
            </div>
        </div>
        """)  # noqa: E501
    else:
        error = status["error"]
        del processing_status[job_id]
        return HTMLResponse(f"""
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <strong>Processing Failed:</strong> {error}
        </div>
        """)


@router.post("/search/{library_id}", response_class=HTMLResponse)
async def text_search_ui(
    library_id: int,
    query_text: str = Form(...),
    total_chunks: int = Form(10),
    per_page: int = Form(5),
    page: int = Form(1),
    metadata_filter: Optional[str] = Form(None),
    db: AsyncSession = Depends(async_session),
):
    """Perform text search via HTMX."""
    try:
        filter_dict = None
        if metadata_filter and metadata_filter.strip():
            try:
                filter_dict = json.loads(metadata_filter)
            except json.JSONDecodeError:
                pass

        embedding_lib_service = EmbeddingLibraryService()
        search_request = TextSearchRequest(query_text=query_text, k=total_chunks, metadata_filter=filter_dict)
        results = await embedding_lib_service.text_search(library_id, search_request, db)

        if results and results.results:
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_results = results.results[start_idx:end_idx]
            total_available = len(results.results)
            total_pages = (total_available + per_page - 1) // per_page
            has_more = page < total_pages
            has_prev = page > 1
        else:
            paginated_results = []
            total_available = 0
            total_pages = 0
            has_more = False
            has_prev = False

        if not paginated_results or not results:
            return HTMLResponse("""
            <div class="text-gray-500 italic">No results found for your query.</div>
            """)

        html = f"""
        <div class="space-y-4">
            <div class="bg-blue-50 p-3 rounded border">
                <div class="flex justify-between items-center mb-2">
                    <div>
                        <strong>Search Results:</strong> Showing {len(paginated_results)} chunks (page {page} of {total_pages})
                    </div>
                    <div class="text-sm text-gray-600">
                        Query time: {results.query_time_ms:.2f}ms | Searched: {results.total_chunks_searched} chunks
                    </div>
                </div>

                <!-- Pagination Controls -->
                <div class="flex justify-between items-center">
                    <div class="text-sm text-gray-600">
                        Total found: {total_available} chunks | Per page: {per_page}
                    </div>
                    <div class="flex gap-1">
        """

        if page > 1:
            html += f"""
                        <button hx-post="/ui/search/{library_id}"
                                hx-include="#search-form"
                                hx-target="#search-results"
                                hx-vals='{{"page": 1}}'
                                class="bg-gray-400 hover:bg-gray-500 text-white px-2 py-1 rounded text-xs">
                            ⏮️
                        </button>
            """

        if has_prev:
            html += f"""
                        <button hx-post="/ui/search/{library_id}"
                                hx-include="#search-form"
                                hx-target="#search-results"
                                hx-vals='{{"page": {page - 1}}}'
                                class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-xs">
                            ← Prev
                        </button>
            """

        start_page = max(1, page - 2)
        end_page = min(total_pages, page + 2)

        for p in range(start_page, end_page + 1):
            if p == page:
                html += f"""
                        <button class="bg-blue-700 text-white px-3 py-1 rounded text-xs font-bold" disabled>
                            {p}
                        </button>
                """
            else:
                html += f"""
                        <button hx-post="/ui/search/{library_id}"
                                hx-include="#search-form"
                                hx-target="#search-results"
                                hx-vals='{{"page": {p}}}'
                                class="bg-gray-300 hover:bg-gray-400 text-gray-700 px-3 py-1 rounded text-xs">
                            {p}
                        </button>
                """

        if has_more:
            html += f"""
                        <button hx-post="/ui/search/{library_id}"
                                hx-include="#search-form"
                                hx-target="#search-results"
                                hx-vals='{{"page": {page + 1}}}'
                                class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-xs">
                            Next →
                        </button>
            """

        if page < total_pages:
            html += f"""
                        <button hx-post="/ui/search/{library_id}"
                                hx-include="#search-form"
                                hx-target="#search-results"
                                hx-vals='{{"page": {total_pages}}}'
                                class="bg-gray-400 hover:bg-gray-500 text-white px-2 py-1 rounded text-xs">
                            ⏭️
                        </button>
            """

        html += """
                    </div>
                </div>
            </div>
        """

        for i, result in enumerate(paginated_results, start_idx + 1):
            html += f"""
            <div class="bg-gray-50 p-4 rounded border">
                <div class="flex justify-between items-start mb-2">
                    <strong>Result #{i}</strong>
                    <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
                        Similarity: {result.similarity_score:.3f}
                    </span>
                </div>
                <p class="text-gray-800 mb-2">{result.content}</p>
                <div class="text-sm text-gray-500">
                    <span>Chunk ID: {result.chunk_id}</span> |
                    <span>Document ID: {result.document_id}</span>
                    {f" | Metadata: {result.metadata}" if result.metadata else ""}
                </div>
            </div>
            """

        html += "</div>"
        return HTMLResponse(html)

    except Exception as e:
        return HTMLResponse(f"""
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <strong>Search Error:</strong> {str(e)}
        </div>
        """)


@router.get("/embedding/info", response_class=HTMLResponse)
async def embedding_info_ui():
    """Get embedding model info via HTMX."""
    try:
        info_service = EmbeddingInfoService()
        info = await info_service.get_embedding_info()

        status_color = "green" if info.is_loaded else "yellow"
        status_text = "Loaded" if info.is_loaded else "Not Loaded"

        return HTMLResponse(f"""
        <div class="flex items-center space-x-4">
            <div class="flex items-center">
                <div class="w-3 h-3 bg-{status_color}-500 rounded-full mr-2"></div>
                <span><strong>Model:</strong> {info.model_name}</span>
            </div>
            <div>
                <span><strong>Dimension:</strong> {info.dimension}</span>
            </div>
            <div>
                <span><strong>Status:</strong> {status_text}</span>
            </div>
        </div>
        """)
    except Exception as e:
        return HTMLResponse(f"""
        <div class="text-red-600">
            <strong>Error:</strong> Unable to load embedding status - {str(e)}
        </div>
        """)
