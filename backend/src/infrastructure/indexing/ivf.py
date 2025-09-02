"""IVF (Inverted File Index) implementation for approximate vector search.

IVF provides faster search by partitioning the vector space into clusters.
Instead of searching all vectors, it searches only the most relevant clusters.

Time Complexity: O(k*d) where k << n (number of clusters to search)
Space Complexity: O(n*d + m*d) where m is number of centroids
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .base import ChunkVector, IndexStats, IndexType, SearchResult, VectorIndex


@dataclass
class IVFCluster:
    """Represents a cluster in the IVF index."""

    centroid: np.ndarray
    vector_ids: List[int] = field(default_factory=list)


class IVFIndex(VectorIndex):
    """IVF (Inverted File Index) implementation for approximate vector search.

    This implementation uses k-means clustering to partition the vector space
    into clusters. During search, only the most relevant clusters are searched,
    providing significant speedup for large datasets.

    The algorithm works by:
    1. Clustering all vectors using k-means
    2. Assigning each vector to its nearest cluster
    3. During search, finding nearest clusters to query
    4. Searching only vectors in those clusters

    Trade-offs:
    - Speed: Much faster than linear search for large datasets
    - Accuracy: Slightly lower due to approximate nature
    - Memory: Additional overhead for storing centroids
    """

    def __init__(
        self, dimension: int, num_clusters: int = 100, num_probes: int = 1, max_iterations: int = 100, tolerance: float = 1e-4
    ):
        """Initialize IVF index.

        Args:
            dimension: Vector embedding dimension
            num_clusters: Number of clusters for partitioning (higher = more accurate, slower)
            num_probes: Number of clusters to search during query (higher = more accurate, slower)
            max_iterations: Maximum k-means iterations
            tolerance: K-means convergence tolerance
        """
        super().__init__(dimension)
        self.num_clusters = num_clusters
        self.num_probes = min(num_probes, num_clusters)
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self._vectors: Dict[int, ChunkVector] = {}
        self._clusters: List[IVFCluster] = []
        self._vector_to_cluster: Dict[int, int] = {}
        self._is_built = False

    @property
    def index_type(self) -> IndexType:
        """Get the index type."""
        return IndexType.IVF

    def is_built(self) -> bool:
        """Check if index is built and ready for search."""
        return self._is_built and len(self._clusters) > 0

    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        cluster_sizes = [len(cluster.vector_ids) for cluster in self._clusters] if self._clusters else []
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
        min_cluster_size = min(cluster_sizes) if cluster_sizes else 0

        return IndexStats(
            total_vectors=len(self._vectors),
            embedding_dimension=self.dimension,
            algorithm_params={
                "num_clusters": self.num_clusters,
                "num_probes": self.num_probes,
                "avg_cluster_size": float(avg_cluster_size),
                "max_cluster_size": int(max_cluster_size),
                "min_cluster_size": int(min_cluster_size),
                "cluster_distribution": cluster_sizes[:10] if cluster_sizes else [],
            },
        )

    async def add_vector(self, vector: ChunkVector) -> None:
        """Add a single vector to the index."""
        if len(vector.embedding) != self.dimension:
            raise ValueError(f"Embedding dimension {len(vector.embedding)} does not match index dimension {self.dimension}")

        self._vectors[vector.chunk_id] = vector

        if self._is_built and self._clusters:
            cluster_idx = self._find_nearest_cluster(vector.embedding)
            self._clusters[cluster_idx].vector_ids.append(vector.chunk_id)
            self._vector_to_cluster[vector.chunk_id] = cluster_idx
        else:
            self._is_built = False

    async def add_vectors(self, vectors: List[ChunkVector]) -> None:
        """Add multiple vectors to the index."""
        for vector in vectors:
            if len(vector.embedding) != self.dimension:
                raise ValueError(f"Embedding dimension {len(vector.embedding)} does not match index dimension {self.dimension}")
            self._vectors[vector.chunk_id] = vector

        self._is_built = False

    async def remove_vector(self, chunk_id: int) -> bool:
        """Remove a vector from the index."""
        if chunk_id not in self._vectors:
            return False

        del self._vectors[chunk_id]

        if chunk_id in self._vector_to_cluster:
            cluster_idx = self._vector_to_cluster[chunk_id]
            if cluster_idx < len(self._clusters):
                self._clusters[cluster_idx].vector_ids = [
                    vid for vid in self._clusters[cluster_idx].vector_ids if vid != chunk_id
                ]
            del self._vector_to_cluster[chunk_id]

        return True

    async def clear(self) -> None:
        """Clear all vectors from the index."""
        self._vectors.clear()
        self._clusters.clear()
        self._vector_to_cluster.clear()
        self._is_built = False

    async def build_index(self) -> None:
        """Build the IVF index using k-means clustering."""
        if not self._vectors:
            self._is_built = True
            return

        embeddings = np.array([v.embedding for v in self._vectors.values()])
        vector_ids = list(self._vectors.keys())

        effective_clusters = min(self.num_clusters, len(self._vectors))
        if effective_clusters < self.num_clusters:
            from ...modules.common.utils.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(f"Reducing clusters from {self.num_clusters} to {effective_clusters} due to insufficient vectors")
            self.num_clusters = effective_clusters

        centroids, assignments = self._kmeans(embeddings, effective_clusters)
        self._clusters = [IVFCluster(centroid=centroid) for centroid in centroids]
        self._vector_to_cluster.clear()

        for i, cluster_idx in enumerate(assignments):
            vector_id = vector_ids[i]
            self._clusters[cluster_idx].vector_ids.append(vector_id)
            self._vector_to_cluster[vector_id] = cluster_idx

        self.num_probes = min(self.num_probes, len(self._clusters))
        self._is_built = True

    async def search(
        self, query_embedding: List[float], k: int, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform approximate k-NN search using IVF."""
        if not self._is_built:
            await self.build_index()

        if not self._clusters or not self._vectors:
            return []

        if len(query_embedding) != self.dimension:
            raise ValueError(
                f"Query embedding dimension {len(query_embedding)} does not match index dimension {self.dimension}"
            )

        query_vec = np.array(query_embedding)
        cluster_indices = self._find_nearest_clusters(query_vec, self.num_probes)
        candidate_vectors = []
        for cluster_idx in cluster_indices:
            for vector_id in self._clusters[cluster_idx].vector_ids:
                if vector_id in self._vectors:
                    vector = self._vectors[vector_id]

                    if metadata_filter:
                        if not self._matches_filter(vector.metadata, metadata_filter):
                            continue

                    candidate_vectors.append(vector)

        results = []
        for vector in candidate_vectors:
            similarity = self._cosine_similarity(query_embedding, vector.embedding)
            results.append(
                SearchResult(
                    chunk_id=vector.chunk_id,
                    document_id=vector.document_id,
                    content=vector.content,
                    similarity_score=similarity,
                    metadata=vector.metadata,
                )
            )

        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:k]

    def _kmeans(self, data: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Perform k-means clustering.

        Args:
            data: Input vectors (n_samples, n_features)
            k: Number of clusters

        Returns:
            Tuple of (centroids, assignments)
        """
        n_samples, n_features = data.shape

        centroids = self._kmeans_plus_plus_init(data, k)

        for _ in range(self.max_iterations):
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            assignments = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = data[assignments == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids, atol=self.tolerance):
                break

            centroids = new_centroids

        return centroids, assignments

    def _kmeans_plus_plus_init(self, data: np.ndarray, k: int) -> np.ndarray:
        """Initialize centroids using k-means++ algorithm."""
        n_samples, n_features = data.shape
        centroids = np.zeros((k, n_features))
        centroids[0] = data[np.random.randint(n_samples)]

        for i in range(1, k):
            distances = np.array(
                [min([float(np.linalg.norm(point - centroid) ** 2) for centroid in centroids[:i]]) for point in data]
            )

            distances_sum = distances.sum()
            if distances_sum == 0:
                selected_idx = np.random.randint(len(data))
            else:
                probabilities = distances / distances_sum
                cumulative_probs = probabilities.cumsum()
                r = np.random.random()
                selected_idx = np.where(cumulative_probs >= r)[0][0]
            centroids[i] = data[selected_idx]

        return centroids

    def _find_nearest_cluster(self, query_embedding: List[float]) -> int:
        """Find the nearest cluster for a vector."""
        query_vec = np.array(query_embedding)
        distances = [np.linalg.norm(query_vec - cluster.centroid) for cluster in self._clusters]
        return int(np.argmin(distances))

    def _find_nearest_clusters(self, query_vec: np.ndarray, num_probes: int) -> List[int]:
        """Find the nearest clusters for search."""
        if not self._clusters:
            return []

        distances = [np.linalg.norm(query_vec - cluster.centroid) for cluster in self._clusters]

        nearest_indices = np.argsort(distances)[:num_probes]
        return [int(idx) for idx in nearest_indices.tolist()]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return float(dot_product / (norm_v1 * norm_v2))

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
