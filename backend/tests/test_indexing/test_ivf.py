"""Tests for IVF (Inverted File Index) vector indexing algorithm."""

import pytest
import numpy as np

from src.infrastructure.indexing.ivf import IVFIndex
from src.infrastructure.indexing.base import ChunkVector, IndexType


class TestIVFIndex:
    """Test cases for IVF indexing algorithm."""

    @pytest.fixture
    def ivf_index(self):
        """Create an IVF index for testing."""
        return IVFIndex(dimension=3, num_clusters=4, num_probes=2)

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        return [
            ChunkVector(
                chunk_id=1, document_id=1, content="First vector content", embedding=[1.0, 0.0, 0.0], metadata={"category": "A"}
            ),
            ChunkVector(
                chunk_id=2,
                document_id=1,
                content="Second vector content",
                embedding=[0.0, 1.0, 0.0],
                metadata={"category": "B"},
            ),
            ChunkVector(
                chunk_id=3, document_id=2, content="Third vector content", embedding=[0.0, 0.0, 1.0], metadata={"category": "A"}
            ),
            ChunkVector(
                chunk_id=4,
                document_id=2,
                content="Fourth vector content",
                embedding=[0.7071, 0.7071, 0.0],
                metadata={"category": "C"},
            ),
            ChunkVector(
                chunk_id=5,
                document_id=3,
                content="Fifth vector content",
                embedding=[-1.0, 0.0, 0.0],
                metadata={"category": "A"},
            ),
            ChunkVector(
                chunk_id=6,
                document_id=3,
                content="Sixth vector content",
                embedding=[0.0, -1.0, 0.0],
                metadata={"category": "B"},
            ),
        ]

    def test_index_properties(self, ivf_index):
        """Test basic IVF index properties."""
        assert ivf_index.index_type == IndexType.IVF
        assert ivf_index.dimension == 3
        assert ivf_index.num_clusters == 4
        assert ivf_index.num_probes == 2
        assert not ivf_index.is_built()

    @pytest.mark.asyncio
    async def test_add_single_vector(self, ivf_index):
        """Test adding a single vector."""
        vector = ChunkVector(
            chunk_id=1, document_id=1, content="Test content", embedding=[1.0, 0.0, 0.0], metadata={"test": True}
        )

        await ivf_index.add_vector(vector)

        stats = ivf_index.get_stats()
        assert stats.total_vectors == 1
        assert not ivf_index.is_built()  # Single vector addition doesn't trigger build

    @pytest.mark.asyncio
    async def test_add_multiple_vectors(self, ivf_index, sample_vectors):
        """Test adding multiple vectors."""
        await ivf_index.add_vectors(sample_vectors)

        stats = ivf_index.get_stats()
        assert stats.total_vectors == 6
        assert not ivf_index.is_built()  # Bulk addition doesn't auto-build

    @pytest.mark.asyncio
    async def test_build_index(self, ivf_index, sample_vectors):
        """Test building the IVF index."""
        await ivf_index.add_vectors(sample_vectors)
        await ivf_index.build_index()

        assert ivf_index.is_built()
        stats = ivf_index.get_stats()
        assert stats.total_vectors == 6
        assert "num_clusters" in stats.algorithm_params
        assert "avg_cluster_size" in stats.algorithm_params

    @pytest.mark.asyncio
    async def test_build_index_few_vectors(self):
        """Test building index with fewer vectors than clusters."""
        ivf_index = IVFIndex(dimension=2, num_clusters=10, num_probes=3)

        # Add only 3 vectors
        vectors = [ChunkVector(i, 1, f"Content {i}", [float(i), 0.0], {}) for i in range(3)]

        await ivf_index.add_vectors(vectors)
        await ivf_index.build_index()

        assert ivf_index.is_built()
        stats = ivf_index.get_stats()
        # Should adapt to use only 3 clusters
        assert stats.algorithm_params["num_clusters"] <= 3

    @pytest.mark.asyncio
    async def test_search_empty_index(self, ivf_index):
        """Test search on empty index."""
        results = await ivf_index.search([1.0, 0.0, 0.0], k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_basic(self, ivf_index, sample_vectors):
        """Test basic IVF search functionality."""
        await ivf_index.add_vectors(sample_vectors)
        await ivf_index.build_index()

        # Search for vector similar to [1.0, 0.0, 0.0]
        results = await ivf_index.search([1.0, 0.0, 0.0], k=3)

        assert len(results) <= 3
        assert all(r.similarity_score >= 0 for r in results)
        assert all(r.similarity_score <= 1 for r in results)

        # Results should be sorted by similarity (descending)
        similarities = [r.similarity_score for r in results]
        assert similarities == sorted(similarities, reverse=True)

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, ivf_index, sample_vectors):
        """Test IVF search with metadata filtering."""
        await ivf_index.add_vectors(sample_vectors)
        await ivf_index.build_index()

        # Search with metadata filter
        results = await ivf_index.search([1.0, 0.0, 0.0], k=10, metadata_filter={"category": "A"})

        # Should only return vectors with category "A"
        for result in results:
            assert result.metadata["category"] == "A"

    @pytest.mark.asyncio
    async def test_search_k_parameter(self, ivf_index, sample_vectors):
        """Test that k parameter correctly limits results."""
        await ivf_index.add_vectors(sample_vectors)
        await ivf_index.build_index()

        # Request fewer results than available
        results = await ivf_index.search([1.0, 0.0, 0.0], k=2)
        assert len(results) <= 2

        # Request more results than available
        results = await ivf_index.search([1.0, 0.0, 0.0], k=100)
        assert len(results) <= len(sample_vectors)

    @pytest.mark.asyncio
    async def test_search_accuracy_vs_linear(self):
        """Test IVF search accuracy compared to linear search."""
        from src.infrastructure.indexing.linear_search import LinearSearchIndex

        # Create identical datasets for both indexes
        dimension = 5
        vectors = [
            ChunkVector(
                chunk_id=i,
                document_id=1,
                content=f"Content {i}",
                embedding=[float(i % 3), float((i + 1) % 3), float((i + 2) % 3), 0.0, 0.0],
                metadata={"index": i},
            )
            for i in range(20)
        ]

        # Set up both indexes
        linear_index = LinearSearchIndex(dimension=dimension)
        ivf_index = IVFIndex(dimension=dimension, num_clusters=4, num_probes=2)

        await linear_index.add_vectors(vectors)
        await ivf_index.add_vectors(vectors)
        await ivf_index.build_index()

        # Compare results for same query
        query = [1.0, 0.0, 0.0, 0.0, 0.0]
        k = 5

        linear_results = await linear_index.search(query, k)
        ivf_results = await ivf_index.search(query, k)

        # IVF should return some results (may not be identical due to approximation)
        assert len(ivf_results) > 0
        assert len(linear_results) > 0

        # Check that top result is reasonably similar between methods
        if linear_results and ivf_results:
            linear_top_score = linear_results[0].similarity_score
            ivf_top_score = ivf_results[0].similarity_score
            # IVF should be reasonably close to linear (within 20% for this test)
            assert abs(linear_top_score - ivf_top_score) < 0.2

    @pytest.mark.asyncio
    async def test_remove_vector(self, ivf_index, sample_vectors):
        """Test removing vectors from IVF index."""
        await ivf_index.add_vectors(sample_vectors)
        await ivf_index.build_index()

        # Remove a vector
        success = await ivf_index.remove_vector(1)
        assert success

        stats = ivf_index.get_stats()
        assert stats.total_vectors == 5

        # Try to remove non-existent vector
        success = await ivf_index.remove_vector(999)
        assert not success

    @pytest.mark.asyncio
    async def test_clear_index(self, ivf_index, sample_vectors):
        """Test clearing the IVF index."""
        await ivf_index.add_vectors(sample_vectors)
        await ivf_index.build_index()

        await ivf_index.clear()

        assert not ivf_index.is_built()
        stats = ivf_index.get_stats()
        assert stats.total_vectors == 0

    @pytest.mark.asyncio
    async def test_dimension_validation(self, ivf_index):
        """Test dimension validation in IVF index."""
        wrong_dim_vector = ChunkVector(
            chunk_id=1,
            document_id=1,
            content="Wrong dimension",
            embedding=[1.0, 0.0],  # 2D instead of 3D
            metadata={},
        )

        with pytest.raises(ValueError, match="Embedding dimension"):
            await ivf_index.add_vector(wrong_dim_vector)

    @pytest.mark.asyncio
    async def test_search_dimension_validation(self, ivf_index, sample_vectors):
        """Test search dimension validation."""
        await ivf_index.add_vectors(sample_vectors)
        await ivf_index.build_index()

        # Wrong dimension query
        with pytest.raises(ValueError, match="Query embedding dimension"):
            await ivf_index.search([1.0, 0.0], k=5)  # 2D instead of 3D

    @pytest.mark.asyncio
    async def test_search_result_format(self, ivf_index, sample_vectors):
        """Test that IVF search results have correct format."""
        await ivf_index.add_vectors(sample_vectors)
        await ivf_index.build_index()

        results = await ivf_index.search([1.0, 0.0, 0.0], k=3)

        for result in results:
            assert hasattr(result, "chunk_id")
            assert hasattr(result, "document_id")
            assert hasattr(result, "content")
            assert hasattr(result, "similarity_score")
            assert hasattr(result, "metadata")
            assert isinstance(result.similarity_score, float)
            assert 0 <= result.similarity_score <= 1

    @pytest.mark.asyncio
    async def test_num_probes_effect(self, sample_vectors):
        """Test that increasing num_probes improves recall."""
        dimension = 3

        # Create two indexes with different num_probes
        ivf_index_1_probe = IVFIndex(dimension=dimension, num_clusters=3, num_probes=1)
        ivf_index_2_probes = IVFIndex(dimension=dimension, num_clusters=3, num_probes=2)

        await ivf_index_1_probe.add_vectors(sample_vectors)
        await ivf_index_2_probes.add_vectors(sample_vectors)

        await ivf_index_1_probe.build_index()
        await ivf_index_2_probes.build_index()

        query = [1.0, 0.0, 0.0]
        k = 5

        results_1_probe = await ivf_index_1_probe.search(query, k)
        results_2_probes = await ivf_index_2_probes.search(query, k)

        # More probes should generally return more or equal results
        assert len(results_2_probes) >= len(results_1_probe)

    def test_cosine_similarity_calculation(self, ivf_index):
        """Test cosine similarity calculation."""
        # Test identical vectors
        sim = ivf_index._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

        # Test orthogonal vectors
        sim = ivf_index._cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert abs(sim - 0.0) < 1e-6

        # Test opposite vectors
        sim = ivf_index._cosine_similarity([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
        assert abs(sim - (-1.0)) < 1e-6

        # Test zero vectors
        sim = ivf_index._cosine_similarity([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert sim == 0.0

    def test_metadata_filter_matching(self, ivf_index):
        """Test metadata filter matching logic."""
        metadata = {"category": "A", "priority": 1, "active": True}

        # Exact match
        assert ivf_index._matches_filter(metadata, {"category": "A"})
        assert ivf_index._matches_filter(metadata, {"category": "A", "priority": 1})

        # No match
        assert not ivf_index._matches_filter(metadata, {"category": "B"})
        assert not ivf_index._matches_filter(metadata, {"category": "A", "priority": 2})
        assert not ivf_index._matches_filter(metadata, {"nonexistent": "value"})

        # Empty filter (should match everything)
        assert ivf_index._matches_filter(metadata, {})

    @pytest.mark.asyncio
    async def test_cluster_assignment_consistency(self, ivf_index, sample_vectors):
        """Test that vectors are consistently assigned to clusters."""
        await ivf_index.add_vectors(sample_vectors)
        await ivf_index.build_index()

        # Check that all vectors are assigned to clusters
        total_assigned = sum(len(cluster.vector_ids) for cluster in ivf_index._clusters)
        assert total_assigned == len(sample_vectors)

        # Check that each vector is assigned to exactly one cluster
        all_assigned_ids = []
        for cluster in ivf_index._clusters:
            all_assigned_ids.extend(cluster.vector_ids)

        assert len(all_assigned_ids) == len(set(all_assigned_ids))  # No duplicates
        assert set(all_assigned_ids) == {v.chunk_id for v in sample_vectors}

    @pytest.mark.asyncio
    async def test_ivf_performance_characteristics(self):
        """Test IVF performance characteristics with larger dataset."""
        dimension = 10
        num_vectors = 100

        # Create random vectors
        np.random.seed(42)  # For reproducibility
        vectors = []
        for i in range(num_vectors):
            embedding = np.random.randn(dimension).tolist()
            vectors.append(
                ChunkVector(
                    chunk_id=i, document_id=i // 10, content=f"Content {i}", embedding=embedding, metadata={"cluster": i % 5}
                )
            )

        # Set up IVF with reasonable parameters
        ivf_index = IVFIndex(dimension=dimension, num_clusters=10, num_probes=3)
        await ivf_index.add_vectors(vectors)
        await ivf_index.build_index()

        # Perform search
        query = np.random.randn(dimension).tolist()
        results = await ivf_index.search(query, k=10)

        assert len(results) <= 10
        assert ivf_index.is_built()

        # Check cluster distribution
        stats = ivf_index.get_stats()
        assert stats.algorithm_params["num_clusters"] == 10
        assert stats.algorithm_params["avg_cluster_size"] > 0

    @pytest.mark.asyncio
    async def test_add_vector_to_built_index(self, ivf_index, sample_vectors):
        """Test adding vector to already built index."""
        await ivf_index.add_vectors(sample_vectors)
        await ivf_index.build_index()
        assert ivf_index.is_built()

        # Add new vector to built index
        new_vector = ChunkVector(
            chunk_id=999, document_id=999, content="New vector content", embedding=[0.5, 0.5, 0.0], metadata={"category": "NEW"}
        )

        await ivf_index.add_vector(new_vector)

        # Vector should be added and assigned to nearest cluster
        stats = ivf_index.get_stats()
        assert stats.total_vectors == 7
        assert ivf_index.is_built()  # Should remain built

        # Should be able to find the new vector
        results = await ivf_index.search([0.5, 0.5, 0.0], k=1)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_kmeans_convergence(self):
        """Test k-means clustering convergence with well-separated data."""
        dimension = 2

        # Create well-separated clusters
        cluster_1 = [[1.0, 1.0], [1.1, 1.1], [0.9, 0.9], [1.0, 0.9]]
        cluster_2 = [[-1.0, -1.0], [-1.1, -1.1], [-0.9, -0.9], [-1.0, -0.9]]

        vectors = []
        for i, embedding in enumerate(cluster_1 + cluster_2):
            vectors.append(
                ChunkVector(
                    chunk_id=i,
                    document_id=1,
                    content=f"Content {i}",
                    embedding=embedding,
                    metadata={"true_cluster": 0 if i < 4 else 1},
                )
            )

        ivf_index = IVFIndex(dimension=dimension, num_clusters=2, num_probes=1)
        await ivf_index.add_vectors(vectors)
        await ivf_index.build_index()

        # Check that clusters are well-formed
        stats = ivf_index.get_stats()
        assert stats.algorithm_params["num_clusters"] == 2

        # Both clusters should have vectors
        cluster_sizes = stats.algorithm_params["cluster_distribution"]
        assert all(size > 0 for size in cluster_sizes)

    @pytest.mark.asyncio
    async def test_num_probes_parameter_limits(self):
        """Test num_probes parameter limits and adjustments."""
        dimension = 3

        # Test that num_probes can't exceed num_clusters
        ivf_index = IVFIndex(dimension=dimension, num_clusters=3, num_probes=5)
        assert ivf_index.num_probes == 3  # Should be capped at num_clusters

        # Test with minimal clusters
        ivf_index_minimal = IVFIndex(dimension=dimension, num_clusters=1, num_probes=3)
        assert ivf_index_minimal.num_probes == 1
