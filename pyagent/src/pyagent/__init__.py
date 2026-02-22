"""
PyAgent - A lightweight AI Agent framework with MCP and Skills support.
"""

from pyagent.core.agent import Agent
from pyagent.core.message import Message, Role
from pyagent.core.context import Context
from pyagent.core.tools import Tool
from pyagent.hooks import (
    BaseHook,
    HookAction,
    HookContext,
    HookExecutor,
    HookPosition,
    HookRegistry,
    HookResult,
    LoggingHook,
    TimingHook,
    ErrorHandlingHook,
    MetricsHook,
    RateLimitHook,
)
from pyagent.rag import (
    # RAG Core
    Document,
    Chunk,
    SearchResult,
    RAGPipeline,
    SimpleRAG,
    create_rag_tool,
    # Embeddings
    DummyEmbedding,
    FakeEmbedding,
    OpenAIEmbedding,
    LocalEmbedding,
    # Vector Stores
    MemoryVectorStore,
    ChromaVectorStore,
    # Chunking
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    # Retrievers
    VectorRetriever,
    KeywordRetriever,
    HybridRetriever,
    # Rerankers
    IdentityReranker,
    LLMReranker,
    DiversityReranker,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "Agent",
    "Message",
    "Role",
    "Context",
    "Tool",
    # Hooks
    "BaseHook",
    "HookAction",
    "HookContext",
    "HookExecutor",
    "HookPosition",
    "HookRegistry",
    "HookResult",
    "LoggingHook",
    "TimingHook",
    "ErrorHandlingHook",
    "MetricsHook",
    "RateLimitHook",
    # RAG
    "Document",
    "Chunk",
    "SearchResult",
    "RAGPipeline",
    "SimpleRAG",
    "create_rag_tool",
    "DummyEmbedding",
    "FakeEmbedding",
    "OpenAIEmbedding",
    "LocalEmbedding",
    "MemoryVectorStore",
    "ChromaVectorStore",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "VectorRetriever",
    "KeywordRetriever",
    "HybridRetriever",
    "IdentityReranker",
    "LLMReranker",
    "DiversityReranker",
]
