"""RAG (Retrieval-Augmented Generation) system for PyAgent."""

from .document import Document, Chunk, SearchResult
from .base import (
    BaseEmbedding,
    BaseVectorStore,
    BaseRetriever,
    BaseChunker,
    BaseReranker,
)
from .embeddings import DummyEmbedding, FakeEmbedding, OpenAIEmbedding, LocalEmbedding
from .vectorstore import MemoryVectorStore, ChromaVectorStore, cosine_similarity
from .chunking import FixedSizeChunker, RecursiveChunker, SemanticChunker
from .retriever import VectorRetriever, KeywordRetriever, HybridRetriever, MultiQueryRetriever
from .reranker import IdentityReranker, LLMReranker, CrossEncoderReranker, DiversityReranker
from .pipeline import RAGPipeline, SimpleRAG, create_rag_tool

__all__ = [
    "Document", "Chunk", "SearchResult",
    "BaseEmbedding", "BaseVectorStore", "BaseRetriever", "BaseChunker", "BaseReranker",
    "DummyEmbedding", "FakeEmbedding", "OpenAIEmbedding", "LocalEmbedding",
    "MemoryVectorStore", "ChromaVectorStore", "cosine_similarity",
    "FixedSizeChunker", "RecursiveChunker", "SemanticChunker",
    "VectorRetriever", "KeywordRetriever", "HybridRetriever", "MultiQueryRetriever",
    "IdentityReranker", "LLMReranker", "CrossEncoderReranker", "DiversityReranker",
    "RAGPipeline", "SimpleRAG", "create_rag_tool",
]
