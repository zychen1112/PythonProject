"""Document and Chunk data structures for RAG."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document to be indexed and retrieved.

    Attributes:
        id: Unique identifier for the document
        content: The text content of the document
        metadata: Additional metadata about the document
        source: Optional source URL or path
    """

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.id!r}, content={content_preview!r})"


class Chunk(BaseModel):
    """A chunk of a document.

    Chunks are created by chunking strategies and stored in the vector database.

    Attributes:
        id: Unique identifier for the chunk
        document_id: ID of the parent document
        content: The text content of the chunk
        metadata: Additional metadata (inherited from document + chunk-specific)
        embedding: Optional embedding vector
        start_index: Start character index in the original document
        end_index: End character index in the original document
    """

    id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    start_index: int = 0
    end_index: int = 0

    def __repr__(self) -> str:
        content_preview = self.content[:30] + "..." if len(self.content) > 30 else self.content
        return f"Chunk(id={self.id!r}, doc_id={self.document_id!r}, content={content_preview!r})"


class SearchResult(BaseModel):
    """A search result from the RAG system.

    Attributes:
        chunk: The matching chunk
        score: Similarity score (higher is better)
        document: Optional parent document
    """

    chunk: Chunk
    score: float
    document: Optional[Document] = None

    def __repr__(self) -> str:
        return f"SearchResult(chunk_id={self.chunk.id!r}, score={self.score:.4f})"
