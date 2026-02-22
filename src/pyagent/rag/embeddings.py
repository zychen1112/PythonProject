"""Embedding model implementations."""

import asyncio
import logging
from abc import abstractmethod
from typing import Any, Optional

from .base import BaseEmbedding

logger = logging.getLogger(__name__)


class DummyEmbedding(BaseEmbedding):
    """A dummy embedding model for testing.

    Returns zero vectors of a specified dimension.
    Useful for testing without loading actual models.
    """

    def __init__(self, dimension: int = 384):
        """Initialize the dummy embedding.

        Args:
            dimension: Dimension of the embedding vectors
        """
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dimension for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.0] * self._dimension


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding model.

    Uses OpenAI's embedding API (text-embedding-3-small/large).

    Note: Requires the 'openai' extra to be installed.
    """

    # Known model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 100,
    ):
        """Initialize the OpenAI embedding model.

        Args:
            model: Model name (text-embedding-3-small, text-embedding-3-large)
            api_key: OpenAI API key (optional, uses env var if not provided)
            base_url: Optional base URL for API
            batch_size: Batch size for embedding documents
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.batch_size = batch_size
        self._client = None

    @property
    def dimension(self) -> int:
        return self.MODEL_DIMENSIONS.get(self.model, 1536)

    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                kwargs = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.base_url:
                    kwargs["base_url"] = self.base_url

                self._client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError(
                    "OpenAI embedding requires the 'openai' package. "
                    "Install it with: pip install openai"
                )
        return self._client

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents using OpenAI API."""
        client = self._get_client()
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            response = await client.embeddings.create(
                model=self.model,
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query using OpenAI API."""
        client = self._get_client()

        response = await client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding


class LocalEmbedding(BaseEmbedding):
    """Local embedding model using sentence-transformers.

    Uses HuggingFace sentence-transformers models locally.
    No API calls required, runs entirely on the local machine.

    Note: Requires the 'vector' extra to be installed.
    """

    # Known model dimensions
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "multi-qa-mpnet-base-dot-v1": 768,
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """Initialize the local embedding model.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on (cuda, cpu, mps). Auto-detected if None.
            normalize: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None

    @property
    def dimension(self) -> int:
        return self.MODEL_DIMENSIONS.get(self.model_name, 384)

    def _get_model(self):
        """Get or load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "Local embedding requires 'sentence-transformers'. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents using local model."""
        model = self._get_model()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            ),
        )

        return embeddings.tolist()

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query using local model."""
        embeddings = await self.embed_documents([text])
        return embeddings[0]


class FakeEmbedding(BaseEmbedding):
    """Fake embedding that generates deterministic embeddings from text.

    Useful for testing when you want predictable embeddings.
    The embedding is generated from the hash of the text.
    """

    def __init__(self, dimension: int = 384, seed: int = 42):
        """Initialize the fake embedding.

        Args:
            dimension: Dimension of the embedding vectors
            seed: Random seed for reproducibility
        """
        self._dimension = dimension
        self.seed = seed

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_text(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text hash."""
        import hashlib
        import struct

        # Create a hash of the text
        text_hash = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()

        # Generate embedding values from the hash
        embedding = []
        for i in range(self._dimension):
            # Use modulo to cycle through hash bytes
            byte_idx = (i * 4) % (len(text_hash) - 4)
            value = struct.unpack("f", text_hash[byte_idx : byte_idx + 4])[0]
            # Normalize to [-1, 1] range
            embedding.append(max(-1.0, min(1.0, value / 1e38)))

        return embedding

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_text(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._hash_text(text)
