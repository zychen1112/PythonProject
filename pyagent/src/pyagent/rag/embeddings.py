"""Embedding model implementations."""

import asyncio
import logging
from typing import Optional

from .base import BaseEmbedding

logger = logging.getLogger(__name__)


class DummyEmbedding(BaseEmbedding):
    """A dummy embedding model for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dimension for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.0] * self._dimension


class FakeEmbedding(BaseEmbedding):
    """Fake embedding that generates deterministic embeddings from text."""

    def __init__(self, dimension: int = 384, seed: int = 42):
        self._dimension = dimension
        self.seed = seed

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_text(self, text: str) -> list[float]:
        import hashlib
        import struct
        text_hash = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()
        embedding = []
        for i in range(self._dimension):
            byte_idx = (i * 4) % (len(text_hash) - 4)
            value = struct.unpack("f", text_hash[byte_idx:byte_idx + 4])[0]
            embedding.append(max(-1.0, min(1.0, value / 1e38)))
        return embedding

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_text(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._hash_text(text)


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding model."""

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None, batch_size: int = 100):
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self._client = None

    @property
    def dimension(self) -> int:
        return self.MODEL_DIMENSIONS.get(self.model, 1536)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                kwargs = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                self._client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError("OpenAI embedding requires 'openai'. pip install openai")
        return self._client

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = await client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        client = self._get_client()
        response = await client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


class LocalEmbedding(BaseEmbedding):
    """Local embedding model using sentence-transformers."""

    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None, normalize: bool = True):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None

    @property
    def dimension(self) -> int:
        return self.MODEL_DIMENSIONS.get(self.model_name, 384)

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError("Local embedding requires 'sentence-transformers'. pip install sentence-transformers")
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: model.encode(texts, normalize_embeddings=self.normalize, convert_to_numpy=True)
        )
        return embeddings.tolist()

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self.embed_documents([text])
        return embeddings[0]
