"""Document and Chunk data structures for RAG."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document to be indexed and retrieved."""
    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None


class Chunk(BaseModel):
    """A chunk of a document."""
    id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    start_index: int = 0
    end_index: int = 0


class SearchResult(BaseModel):
    """A search result from the RAG system."""
    chunk: Chunk
    score: float
    document: Optional[Document] = None
