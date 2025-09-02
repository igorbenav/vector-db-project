"""Tests for the EmbeddingService."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
import numpy as np

from src.infrastructure.embedding.service import EmbeddingService


class TestEmbeddingService:
    """Test the EmbeddingService functionality."""

    @pytest.fixture
    def embedding_service(self):
        """Create EmbeddingService instance for testing."""
        return EmbeddingService(model_name="test-model")

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer model."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3] * 256)  # 768-dim mock
        return mock_model

    @pytest.mark.asyncio
    async def test_init(self, embedding_service):
        """Test EmbeddingService initialization."""
        assert embedding_service.model_name == "test-model"
        assert embedding_service._model is None
        assert embedding_service.embedding_dimension == 768

    @pytest.mark.asyncio
    async def test_get_model_lazy_loading(self, embedding_service, mock_sentence_transformer):
        """Test that model is loaded lazily and only once."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_sentence_transformer

            # First call should load model
            model1 = await embedding_service._get_model()
            assert model1 is mock_sentence_transformer
            assert embedding_service._model is mock_sentence_transformer
            mock_to_thread.assert_called_once()

            # Second call should use cached model
            mock_to_thread.reset_mock()
            model2 = await embedding_service._get_model()
            assert model2 is mock_sentence_transformer
            mock_to_thread.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_model_concurrent_loading(self, embedding_service, mock_sentence_transformer):
        """Test that concurrent model loading is handled correctly."""
        with patch("asyncio.to_thread") as mock_to_thread:
            # Simulate slow model loading
            async def slow_load(*args):
                await asyncio.sleep(0.01)
                return mock_sentence_transformer

            mock_to_thread.side_effect = slow_load

            # Start multiple concurrent model loads
            tasks = [embedding_service._get_model() for _ in range(3)]
            results = await asyncio.gather(*tasks)

            # All should return the same model instance
            assert all(result is mock_sentence_transformer for result in results)
            # Model should only be loaded once
            assert mock_to_thread.call_count == 1

    @pytest.mark.asyncio
    async def test_embed_text_success(self, embedding_service, mock_sentence_transformer):
        """Test successful text embedding."""
        text = "This is a test sentence."
        expected_embedding = [0.1, 0.2, 0.3] * 256  # 768-dim

        with patch("asyncio.to_thread") as mock_to_thread:
            # Mock model loading
            mock_to_thread.return_value = mock_sentence_transformer
            embedding_service._model = mock_sentence_transformer

            # Mock embedding generation
            mock_to_thread.return_value = np.array(expected_embedding)

            result = await embedding_service.embed_text(text)

            assert isinstance(result, list)
            assert len(result) == 768
            assert result == expected_embedding

    @pytest.mark.asyncio
    async def test_embed_text_empty_string(self, embedding_service):
        """Test embedding empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await embedding_service.embed_text("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            await embedding_service.embed_text("   ")

    @pytest.mark.asyncio
    async def test_embed_texts_batch(self, embedding_service, mock_sentence_transformer):
        """Test batch text embedding."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        expected_embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]

        with patch("asyncio.to_thread") as mock_to_thread:
            # Mock model loading
            mock_to_thread.return_value = mock_sentence_transformer
            embedding_service._model = mock_sentence_transformer

            # Mock batch embedding generation
            mock_to_thread.return_value = np.array(expected_embeddings)

            results = await embedding_service.embed_texts(texts)

            assert isinstance(results, list)
            assert len(results) == 3
            assert all(len(embedding) == 768 for embedding in results)
            assert results == expected_embeddings

    @pytest.mark.asyncio
    async def test_embed_texts_empty_list(self, embedding_service):
        """Test embedding empty text list returns empty list."""
        result = await embedding_service.embed_texts([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_texts_with_empty_string(self, embedding_service):
        """Test embedding list with empty string raises ValueError."""
        texts = ["Valid text", "", "Another valid text"]

        with pytest.raises(ValueError, match="All texts must be non-empty"):
            await embedding_service.embed_texts(texts)

    @pytest.mark.asyncio
    async def test_embed_texts_calls_model_correctly(self, embedding_service, mock_sentence_transformer):
        """Test that embed_texts calls the model with correct parameters."""
        texts = ["Text 1", "Text 2"]
        embedding_service._model = mock_sentence_transformer

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = np.array([[0.1] * 768, [0.2] * 768])

            await embedding_service.embed_texts(texts)

            # Verify to_thread was called with correct parameters
            mock_to_thread.assert_called_once()
            args = mock_to_thread.call_args[0]
            assert args[0] == mock_sentence_transformer.encode
            assert args[1] == texts

    @pytest.mark.asyncio
    async def test_is_loaded_before_loading(self, embedding_service):
        """Test is_loaded returns False before model is loaded."""
        result = await embedding_service.is_loaded()
        assert result is False

    @pytest.mark.asyncio
    async def test_is_loaded_after_loading(self, embedding_service, mock_sentence_transformer):
        """Test is_loaded returns True after model is loaded."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_sentence_transformer

            await embedding_service._get_model()
            result = await embedding_service.is_loaded()
            assert result is True

    def test_embedding_dimension_property(self, embedding_service):
        """Test embedding_dimension property returns correct value."""
        assert embedding_service.embedding_dimension == 768

    @pytest.mark.asyncio
    async def test_model_encode_parameters(self, embedding_service, mock_sentence_transformer):
        """Test that model.encode is called with correct parameters."""
        text = "Test text"
        embedding_service._model = mock_sentence_transformer

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = np.array([0.1] * 768)

            await embedding_service.embed_text(text)

            # Verify the encode call parameters
            mock_to_thread.assert_called_with(
                mock_sentence_transformer.encode, text, convert_to_tensor=False, normalize_embeddings=True
            )

    @pytest.mark.asyncio
    async def test_model_batch_encode_parameters(self, embedding_service, mock_sentence_transformer):
        """Test that batch model.encode is called with correct parameters."""
        texts = ["Text 1", "Text 2"]
        embedding_service._model = mock_sentence_transformer

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = np.array([[0.1] * 768, [0.2] * 768])

            await embedding_service.embed_texts(texts)

            # Verify the batch encode call parameters
            mock_to_thread.assert_called_with(
                mock_sentence_transformer.encode, texts, convert_to_tensor=False, normalize_embeddings=True, batch_size=32
            )


class TestGetEmbeddingService:
    """Test the get_embedding_service function."""

    def test_get_embedding_service_singleton(self):
        """Test that get_embedding_service returns same instance."""
        from src.infrastructure.embedding.service import get_embedding_service

        # Reset singleton for clean test
        import src.infrastructure.embedding.service as embedding_module

        embedding_module._embedding_service = None

        service1 = get_embedding_service()
        service2 = get_embedding_service()

        assert service1 is service2
        assert service1.model_name == "all-mpnet-base-v2"

    def test_get_embedding_service_instance_type(self):
        """Test that get_embedding_service returns EmbeddingService."""
        from src.infrastructure.embedding.service import get_embedding_service

        service = get_embedding_service()
        assert isinstance(service, EmbeddingService)
