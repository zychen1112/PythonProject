"""Base classes and abstract interfaces for RAG components."""

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .document import Chunk, Document, SearchResult


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add(self, chunks: list["Chunk"], embeddings: list[list[float]]) -> list[str]:
        pass

    @abstractmethod
    async def search(self, query_embedding: list[float], k: int = 5, filter: Optional[dict[str, Any]] = None) -> list["SearchResult"]:
        pass

    @abstractmethod
    async def delete(self, ids: list[str]) -> bool:
        pass

    @abstractmethod
    async def get(self, id: str) -> Optional["Chunk"]:
        pass

    @abstractmethod
    async def count(self) -> int:
        pass

    @abstractmethod
    async def clear(self) -> bool:
        pass


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    async def retrieve(self, query: str, k: int = 5, filter: Optional[dict[str, Any]] = None) -> list["SearchResult"]:
        pass


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def chunk(self, document: "Document") -> list["Chunk"]:
        pass


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    async def rerank(self, query: str, results: list["SearchResult"], top_k: int = 5) -> list["SearchResult"]:
        pass
