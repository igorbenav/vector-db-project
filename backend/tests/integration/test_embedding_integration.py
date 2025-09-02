"""Integration tests for embedding functionality with both indexing algorithms."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.indexing.base import IndexType
from src.modules.library.services import LibraryService
from src.modules.library.schemas import LibraryCreate
from src.modules.document.services import DocumentService
from src.modules.document.schemas import DocumentCreate
from src.modules.chunk.services import ChunkService
from src.modules.embedding.services import EmbeddingLibraryService, EmbeddingDocumentService, EmbeddingChunkService
from src.modules.embedding.schemas import DocumentAutoChunk, EmbeddedChunkCreate, TextSearchRequest


class TestEmbeddingIntegrationWithIndexing:
    """Test embedding functionality integrated with both indexing algorithms."""

    @pytest.fixture(autouse=True)
    async def clear_index_manager(self):
        """Clear the global index manager before each test."""
        from src.infrastructure.indexing.manager import index_manager

        index_manager._indexes.clear()
        index_manager._index_types.clear()
        index_manager._locks.clear()
        yield
        index_manager._indexes.clear()
        index_manager._index_types.clear()
        index_manager._locks.clear()

    @pytest.fixture
    def library_service(self):
        """Library service instance."""
        return LibraryService()

    @pytest.fixture
    def document_service(self):
        """Document service instance."""
        return DocumentService()

    @pytest.fixture
    def chunk_service(self):
        """Chunk service instance."""
        return ChunkService()

    @pytest.fixture
    def embedding_library_service(self):
        """Embedding library service instance."""
        return EmbeddingLibraryService()

    @pytest.fixture
    def embedding_document_service(self):
        """Embedding document service instance."""
        return EmbeddingDocumentService()

    @pytest.fixture
    def embedding_chunk_service(self):
        """Embedding chunk service instance."""
        return EmbeddingChunkService()

    @pytest.mark.asyncio
    async def test_embedding_with_linear_search(
        self,
        library_service: LibraryService,
        embedding_document_service: EmbeddingDocumentService,
        embedding_library_service: EmbeddingLibraryService,
        db_session: AsyncSession,
    ):
        """Test embedding functionality with Linear Search algorithm."""

        # 1. Create a library
        library_data = LibraryCreate(name="Embedding Test Library - Linear", description="Testing embedding with Linear Search")
        library = await library_service.create_library(library_data, db_session)
        library_id = library.id

        # 2. Set indexing algorithm to Linear Search
        from src.infrastructure.indexing.manager import index_manager

        await index_manager.get_or_create_index(
            library_id=library_id, index_type=IndexType.LINEAR_SEARCH, embedding_dimension=768, db=db_session
        )

        # 3. Create a document with auto-chunking and embeddings
        doc_data = DocumentAutoChunk(
            title="AI Research Document",
            content=(
                "Artificial intelligence is transforming technology. "
                "Machine learning algorithms enable computers to learn from data. "
                "Deep learning uses neural networks with multiple layers. "
                "Natural language processing helps computers understand text. "
                "Computer vision enables image recognition and analysis."
            ),
            library_id=library_id,
            metadata={"topic": "AI", "type": "research"},
            chunk_size=150,
            chunk_overlap=30,
        )

        document = await embedding_document_service.create_document_with_auto_chunks(doc_data, db_session)
        assert document.chunk_count > 1  # Should be chunked

        # Reload vectors into the index
        await index_manager._load_library_vectors(library_id, index_manager._indexes[library_id], db_session)

        # 4. Perform text-based search using embeddings
        search_request = TextSearchRequest(
            query_text="machine learning and neural networks", k=3, metadata_filter={"topic": "AI"}
        )

        search_results = await embedding_library_service.text_search(library_id, search_request, db_session)

        # Verify search worked
        assert search_results is not None
        assert search_results.query_time_ms > 0
        assert search_results.total_chunks_searched > 0
        assert len(search_results.results) > 0

        # Verify results contain relevant content
        for result in search_results.results:
            assert result.content is not None
            assert result.similarity_score > 0.0
            assert result.metadata["topic"] == "AI"

        # 5. Verify index statistics show Linear Search algorithm
        stats = index_manager.get_index_stats(library_id)
        assert stats is not None
        assert stats["algorithm"] == "brute_force"
        assert stats["total_vectors"] > 0

    @pytest.mark.asyncio
    async def test_embedding_with_ivf_search(
        self,
        library_service: LibraryService,
        embedding_document_service: EmbeddingDocumentService,
        embedding_library_service: EmbeddingLibraryService,
        db_session: AsyncSession,
    ):
        """Test embedding functionality with IVF algorithm."""

        # 1. Create a library
        library_data = LibraryCreate(name="Embedding Test Library - IVF", description="Testing embedding with IVF algorithm")
        library = await library_service.create_library(library_data, db_session)
        library_id = library.id

        # 2. Set indexing algorithm to IVF
        from src.infrastructure.indexing.manager import index_manager

        await index_manager.get_or_create_index(
            library_id=library_id, index_type=IndexType.IVF, embedding_dimension=768, db=db_session
        )

        # 3. Create multiple documents with diverse content for better clustering
        documents_data = [
            DocumentAutoChunk(
                title="Machine Learning Fundamentals",
                content=(
                    "Machine learning is a subset of artificial intelligence that focuses on algorithms. "
                    "Supervised learning uses labeled data for training. "
                    "Unsupervised learning finds patterns in unlabeled data. "
                    "Reinforcement learning trains agents through rewards and penalties."
                ),
                library_id=library_id,
                metadata={"topic": "ML", "difficulty": "beginner"},
            ),
            DocumentAutoChunk(
                title="Deep Learning Networks",
                content=(
                    "Deep learning uses artificial neural networks with multiple hidden layers. "
                    "Convolutional neural networks excel at image processing tasks. "
                    "Recurrent neural networks handle sequential data like text and time series. "
                    "Transformer architectures have revolutionized natural language processing."
                ),
                library_id=library_id,
                metadata={"topic": "DL", "difficulty": "advanced"},
            ),
            DocumentAutoChunk(
                title="Data Science Methods",
                content=(
                    "Data science combines statistics, programming, and domain expertise. "
                    "Data preprocessing cleans and transforms raw data. "
                    "Exploratory data analysis reveals patterns and insights. "
                    "Statistical modeling helps make predictions and test hypotheses."
                ),
                library_id=library_id,
                metadata={"topic": "DS", "difficulty": "intermediate"},
            ),
        ]

        for doc_data in documents_data:
            document = await embedding_document_service.create_document_with_auto_chunks(doc_data, db_session)
            assert document.chunk_count >= 1

        # Reload vectors into the index
        await index_manager._load_library_vectors(library_id, index_manager._indexes[library_id], db_session)

        # 4. Perform text-based searches with different queries
        searches = [
            {"query_text": "neural networks and deep learning", "k": 5, "expected_topic": "DL"},
            {"query_text": "supervised learning algorithms", "k": 3, "expected_topic": "ML"},
            {"query_text": "statistical analysis and data preprocessing", "k": 4, "expected_topic": "DS"},
        ]

        for search in searches:
            search_request = TextSearchRequest(query_text=search["query_text"], k=search["k"])

            search_results = await embedding_library_service.text_search(library_id, search_request, db_session)

            # Verify search worked
            assert len(search_results.results) > 0
            assert search_results.query_time_ms > 0
            assert search_results.total_chunks_searched > 0

            # Verify semantic relevance - top result should be from expected topic
            top_result = search_results.results[0]
            assert top_result.similarity_score > 0.3  # Reasonable similarity threshold
            assert top_result.metadata["topic"] == search["expected_topic"]

        # 5. Verify index statistics show IVF algorithm
        stats = index_manager.get_index_stats(library_id)
        assert stats is not None
        assert stats["index_type"] == "ivf"
        assert stats["total_vectors"] > 0
        assert "num_clusters" in stats
        assert "num_probes" in stats

    @pytest.mark.asyncio
    async def test_embedding_individual_chunk_creation(
        self,
        library_service: LibraryService,
        document_service: DocumentService,
        embedding_chunk_service: EmbeddingChunkService,
        embedding_library_service: EmbeddingLibraryService,
        db_session: AsyncSession,
    ):
        """Test creating individual chunks with embeddings."""

        # 1. Create a library and document
        library_data = LibraryCreate(name="Individual Chunk Test", description="Testing individual chunk creation")
        library = await library_service.create_library(library_data, db_session)
        library_id = library.id

        doc_data = DocumentCreate(title="Test Document", content="Initial document content", library_id=library_id)
        document = await document_service.create_document(doc_data, db_session)
        document_id = document.id

        # 2. Set up indexing (Linear Search for simplicity)
        from src.infrastructure.indexing.manager import index_manager

        await index_manager.get_or_create_index(
            library_id=library_id, index_type=IndexType.LINEAR_SEARCH, embedding_dimension=768, db=db_session
        )

        # 3. Create chunks with automatic embedding generation
        chunk_contents = [
            "Quantum computing uses quantum mechanics principles",
            "Blockchain technology enables decentralized systems",
            "Cloud computing provides scalable infrastructure",
        ]

        chunk_ids = []
        for i, content in enumerate(chunk_contents):
            chunk_data = EmbeddedChunkCreate(
                content=content, document_id=document_id, metadata={"section": f"section_{i + 1}", "auto_embedded": True}
            )

            chunk = await embedding_chunk_service.create_chunk_with_text(chunk_data, db_session)
            chunk_ids.append(chunk.id)

        # Reload vectors into the index after all chunks are created
        await index_manager._load_library_vectors(library_id, index_manager._indexes[library_id], db_session)

        # Verify chunks have embeddings
        for i, chunk_id in enumerate(chunk_ids):
            chunk = await embedding_chunk_service.chunk_service.get_chunk(chunk_id, db_session)

            # Verify chunk has embedding
            assert chunk.embedding is not None
            assert isinstance(chunk.embedding, list)
            assert len(chunk.embedding) == 768  # all-mpnet-base-v2 dimension
            assert chunk.content == chunk_contents[i]
            assert chunk.metadata["auto_embedded"] is True

        # 4. Perform semantic search
        search_request = TextSearchRequest(query_text="quantum mechanics and computing", k=3)

        search_results = await embedding_library_service.text_search(library_id, search_request, db_session)

        # Verify search found relevant chunks
        assert len(search_results.results) > 0

        # The quantum computing chunk should be most similar
        top_result = search_results.results[0]
        assert "quantum" in top_result.content.lower()
        assert top_result.similarity_score > 0.5  # High similarity expected

    @pytest.mark.asyncio
    async def test_embedding_with_metadata_filtering(
        self,
        library_service: LibraryService,
        document_service: DocumentService,
        embedding_document_service: EmbeddingDocumentService,
        embedding_chunk_service: EmbeddingChunkService,
        embedding_library_service: EmbeddingLibraryService,
        db_session: AsyncSession,
    ):
        """Test embedding search with metadata filtering."""

        # 1. Create library and set up indexing
        library_data = LibraryCreate(name="Metadata Filter Test", description="Testing metadata filtering with embeddings")
        library = await library_service.create_library(library_data, db_session)
        library_id = library.id

        from src.infrastructure.indexing.manager import index_manager

        await index_manager.get_or_create_index(
            library_id=library_id, index_type=IndexType.LINEAR_SEARCH, embedding_dimension=768, db=db_session
        )

        # 2. Create document with mixed content types
        doc_data = DocumentAutoChunk(
            title="Mixed Content Document",
            content=(
                "This document has technical content about databases. It also has business content about market strategies. "
                "Additionally, it contains research content about academic findings."
            ),
            library_id=library_id,
            metadata={"source": "internal"},
            chunk_size=100,
            chunk_overlap=20,
        )

        document = await embedding_document_service.create_document_with_auto_chunks(doc_data, db_session)

        # Reload vectors after document creation
        await index_manager._load_library_vectors(library_id, index_manager._indexes[library_id], db_session)

        # 3. Add manual chunks with specific metadata
        tech_chunk = EmbeddedChunkCreate(
            content="Database indexing improves query performance significantly",
            document_id=document.id,
            metadata={"type": "technical", "priority": "high"},
        )

        business_chunk = EmbeddedChunkCreate(
            content="Market penetration strategies drive revenue growth",
            document_id=document.id,
            metadata={"type": "business", "priority": "medium"},
        )

        await embedding_chunk_service.create_chunk_with_text(tech_chunk, db_session)
        await embedding_chunk_service.create_chunk_with_text(business_chunk, db_session)

        # Reload vectors after adding manual chunks
        await index_manager._load_library_vectors(library_id, index_manager._indexes[library_id], db_session)

        # 4. Search with metadata filters
        tech_search = TextSearchRequest(
            query_text="database performance optimization", k=5, metadata_filter={"type": "technical"}
        )

        tech_results = await embedding_library_service.text_search(library_id, tech_search, db_session)

        # Should find technical content
        assert len(tech_results.results) > 0
        for result in tech_results.results:
            if "type" in result.metadata:
                assert result.metadata["type"] == "technical"

        # 5. Search with different metadata filter
        business_search = TextSearchRequest(
            query_text="revenue and market strategies", k=5, metadata_filter={"type": "business"}
        )

        business_results = await embedding_library_service.text_search(library_id, business_search, db_session)

        # Should find business content
        assert len(business_results.results) > 0
        for result in business_results.results:
            if "type" in result.metadata:
                assert result.metadata["type"] == "business"

    @pytest.mark.asyncio
    async def test_embedding_performance_comparison(
        self,
        library_service: LibraryService,
        embedding_document_service: EmbeddingDocumentService,
        embedding_library_service: EmbeddingLibraryService,
        db_session: AsyncSession,
    ):
        """Test and compare performance between Linear Search and IVF with embeddings."""

        # Create two libraries for comparison
        libraries = []
        algorithms = [{"name": "linear", "type": IndexType.LINEAR_SEARCH}, {"name": "ivf", "type": IndexType.IVF}]

        # Set up both libraries with same content
        for algo in algorithms:
            # Create library
            library_data = LibraryCreate(
                name=f"Performance Test {algo['name'].upper()}",
                description=f"Testing {algo['name']} performance with embeddings",
            )
            library = await library_service.create_library(library_data, db_session)
            library_id = library.id

            # Set indexing algorithm
            from src.infrastructure.indexing.manager import index_manager

            await index_manager.get_or_create_index(
                library_id=library_id, index_type=algo["type"], embedding_dimension=768, db=db_session
            )

            # Add same content to both libraries
            doc_data = DocumentAutoChunk(
                title="Technology Overview",
                content=(
                    "Artificial intelligence encompasses machine learning and deep learning. "
                    "Natural language processing enables text understanding. Computer vision processes visual information. "
                    "Robotics combines AI with physical systems. Data science extracts insights from large datasets."
                ),
                library_id=library_id,
                metadata={"category": "technology"},
                chunk_size=120,
                chunk_overlap=25,
            )

            document = await embedding_document_service.create_document_with_auto_chunks(doc_data, db_session)
            assert document.chunk_count > 0

            # Reload vectors into the index
            await index_manager._load_library_vectors(library_id, index_manager._indexes[library_id], db_session)

            libraries.append({"id": library_id, "algorithm": algo["name"]})

        # Perform same search on both libraries
        search_request = TextSearchRequest(query_text="artificial intelligence and machine learning", k=3)

        results_comparison = []
        for lib in libraries:
            search_results = await embedding_library_service.text_search(lib["id"], search_request, db_session)

            # Both should find results
            assert len(search_results.results) > 0
            assert search_results.query_time_ms > 0

            results_comparison.append(
                {
                    "algorithm": lib["algorithm"],
                    "query_time_ms": search_results.query_time_ms,
                    "results_count": len(search_results.results),
                    "top_similarity": search_results.results[0].similarity_score,
                }
            )

        # Verify both algorithms found relevant results
        for result in results_comparison:
            assert result["results_count"] > 0
            assert result["top_similarity"] > 0.3  # Reasonable similarity
            assert result["query_time_ms"] < 1000  # Should be fast (< 1 second)

        # Optional: Print performance comparison
        # print(f"Performance comparison:")
        # for result in results_comparison:
        #     print(f"  {result['algorithm'].upper()}: {result['query_time_ms']:.2f}ms, "
        #           f"{result['results_count']} results, top similarity: {result['top_similarity']:.3f}")
