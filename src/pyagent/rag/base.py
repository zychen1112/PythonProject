"""Base classes and abstract interfaces for RAG components."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .document import Chunk, Document, SearchResult


class BaseEmbedding(ABC):
    """Abstract base class for embedding models.

    Embedding models convert text into dense vector representations.
    """

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        pass


class BaseVectorStore(ABC):
    """Abstract base class for vector stores.

    Vector stores persist and search document embeddings.
    """

    @abstractmethod
    async def add(
        self,
        chunks: list["Chunk"],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Add chunks with their embeddings to the store.

        Args:
            chunks: List of chunks to add
            embeddings: Corresponding embedding vectors

        Returns:
            List of added chunk IDs
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list["SearchResult"]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of search results sorted by similarity
        """
        pass

    @abstractmethod
    async def delete(self, ids: list[str]) -> bool:
        """Delete chunks by their IDs.

        Args:
            ids: List of chunk IDs to delete

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    async def get(self, id: str) -> Optional["Chunk"]:
        """Get a chunk by its ID.

        Args:
            id: Chunk ID

        Returns:
            The chunk if found, None otherwise
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """Return the number of chunks in the store."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all chunks from the store."""
        pass


class BaseRetriever(ABC):
    """Abstract base class for retrievers.

    Retrievers find relevant documents for a given query.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list["SearchResult"]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of search results
        """
        pass


class BaseChunker(ABC):
    """Abstract base class for document chunkers.

    Chunkers split documents into smaller pieces for indexing.
    """

    @abstractmethod
    def chunk(self, document: "Document") -> list["Chunk"]:
        """Split a document into chunks.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        pass


class BaseReranker(ABC):
    """Abstract base class for rerankers.

    Rerankers reorder search results to improve relevance.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list["SearchResult"],
        top_k: int = 5,
    ) -> list["SearchResult"]:
        """Rerank search results.

        Args:
            query: Original query string
            results: Search results to rerank
            top_k: Number of results to return

        Returns:
            Reranked search results
        """
        pass
