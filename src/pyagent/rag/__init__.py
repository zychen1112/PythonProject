"""RAG (Retrieval-Augmented Generation) system for PyAgent.

This module provides a complete RAG system including:
- Document and chunk data structures
- Multiple embedding providers (OpenAI, local, fake)
- Vector stores (memory, ChromaDB)
- Chunking strategies (fixed-size, recursive, semantic)
- Retrieval strategies (vector, keyword, hybrid, multi-query)
- Reranking mechanisms (LLM, cross-encoder, diversity)
- Complete RAG pipeline with tool integration

Example:
    ```python
    from pyagent.rag import (
        RAGPipeline,
        Document,
        LocalEmbedding,
        MemoryVectorStore,
        VectorRetriever,
    )

    # Create components
    embedding = LocalEmbedding()
    vectorstore = MemoryVectorStore()
    retriever = VectorRetriever(embedding, vectorstore)

    # Create pipeline
    pipeline = RAGPipeline(embedding, vectorstore, retriever)

    # Index documents
    docs = [Document(id="1", content="Python is a programming language")]
    await pipeline.index(docs)

    # Search
    results = await pipeline.retrieve("What is Python?")
    ```

For simpler use cases:
    ```python
    from pyagent.rag import SimpleRAG

    rag = SimpleRAG()
    await rag.add("Python is a programming language.")
    results = await rag.search("What is Python?")
    ```
"""

# Data structures
from .document import Document, Chunk, SearchResult

# Base classes
from .base import (
    BaseEmbedding,
    BaseVectorStore,
    BaseRetriever,
    BaseChunker,
    BaseReranker,
)

# Embedding providers
from .embeddings import (
    DummyEmbedding,
    FakeEmbedding,
    OpenAIEmbedding,
    LocalEmbedding,
)

# Vector stores
from .vectorstore import (
    MemoryVectorStore,
    ChromaVectorStore,
    cosine_similarity,
)

# Chunking strategies
from .chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
)

# Retrievers
from .retriever import (
    VectorRetriever,
    KeywordRetriever,
    HybridRetriever,
    MultiQueryRetriever,
)

# Rerankers
from .reranker import (
    IdentityReranker,
    LLMReranker,
    CrossEncoderReranker,
    DiversityReranker,
)

# Pipeline
from .pipeline import (
    RAGPipeline,
    SimpleRAG,
    create_rag_tool,
)

__all__ = [
    # Data structures
    "Document",
    "Chunk",
    "SearchResult",
    # Base classes
    "BaseEmbedding",
    "BaseVectorStore",
    "BaseRetriever",
    "BaseChunker",
    "BaseReranker",
    # Embeddings
    "DummyEmbedding",
    "FakeEmbedding",
    "OpenAIEmbedding",
    "LocalEmbedding",
    # Vector stores
    "MemoryVectorStore",
    "ChromaVectorStore",
    "cosine_similarity",
    # Chunking
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    # Retrievers
    "VectorRetriever",
    "KeywordRetriever",
    "HybridRetriever",
    "MultiQueryRetriever",
    # Rerankers
    "IdentityReranker",
    "LLMReranker",
    "CrossEncoderReranker",
    "DiversityReranker",
    # Pipeline
    "RAGPipeline",
    "SimpleRAG",
    "create_rag_tool",
]
