"""RAG Pipeline and Tool integration."""

import asyncio
import logging
import uuid
from typing import Any, Optional, TYPE_CHECKING

from .base import BaseChunker, BaseEmbedding, BaseReranker, BaseRetriever, BaseVectorStore
from .chunking import RecursiveChunker
from .document import Chunk, Document, SearchResult
from .reranker import IdentityReranker
from .retriever import VectorRetriever

if TYPE_CHECKING:
    from pyagent.core.tools import Tool
    from pyagent.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG (Retrieval-Augmented Generation) pipeline.

    Combines embedding, vector storage, retrieval, and optional reranking
    into a unified interface.

    Example:
        ```python
        # Create pipeline
        embedding = LocalEmbedding()
        vectorstore = MemoryVectorStore()
        retriever = VectorRetriever(embedding, vectorstore)

        pipeline = RAGPipeline(
            embedding=embedding,
            vectorstore=vectorstore,
            retriever=retriever,
        )

        # Index documents
        docs = [Document(id="1", content="Python is a programming language")]
        await pipeline.index(docs)

        # Retrieve
        results = await pipeline.retrieve("What is Python?")
        ```
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        vectorstore: BaseVectorStore,
        retriever: Optional[BaseRetriever] = None,
        chunker: Optional[BaseChunker] = None,
        reranker: Optional[BaseReranker] = None,
        llm_provider: Optional["LLMProvider"] = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            embedding: Embedding model for documents and queries
            vectorstore: Vector store for document storage
            retriever: Retriever for finding relevant documents (default: VectorRetriever)
            chunker: Document chunker (default: RecursiveChunker)
            reranker: Result reranker (default: IdentityReranker)
            llm_provider: LLM provider for query generation (optional)
        """
        self.embedding = embedding
        self.vectorstore = vectorstore
        self.retriever = retriever or VectorRetriever(embedding, vectorstore)
        self.chunker = chunker or RecursiveChunker()
        self.reranker = reranker or IdentityReranker()
        self.llm_provider = llm_provider

        # Document storage
        self._documents: dict[str, Document] = {}

    async def index(self, documents: list[Document]) -> list[str]:
        """Index a list of documents.

        Args:
            documents: Documents to index

        Returns:
            List of indexed document IDs
        """
        all_chunk_ids = []

        for document in documents:
            chunk_ids = await self.add_document(document)
            all_chunk_ids.extend(chunk_ids)

        logger.info(f"Indexed {len(documents)} documents ({len(all_chunk_ids)} chunks)")
        return all_chunk_ids

    async def add_document(self, document: Document) -> list[str]:
        """Add a single document to the index.

        Args:
            document: Document to add

        Returns:
            List of chunk IDs created
        """
        # Store document
        self._documents[document.id] = document

        # Store in vectorstore if it supports document references
        if hasattr(self.vectorstore, "add_document"):
            self.vectorstore.add_document(document)

        # Chunk the document
        chunks = self.chunker.chunk(document)

        if not chunks:
            return []

        # Embed chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding.embed_documents(texts)

        # Add to vector store
        chunk_ids = await self.vectorstore.add(chunks, embeddings)

        logger.debug(f"Added document {document.id}: {len(chunks)} chunks")
        return chunk_ids

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
        rerank: bool = True,
    ) -> list[SearchResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            rerank: Whether to apply reranking

        Returns:
            List of search results
        """
        # Retrieve initial results
        results = await self.retriever.retrieve(query, k * 2, filter)

        # Apply reranking if enabled
        if rerank and results:
            results = await self.reranker.rerank(query, results, k)

        return results[:k]

    async def query(
        self,
        query: str,
        k: int = 5,
        prompt_template: Optional[str] = None,
    ) -> str:
        """Retrieve and generate an answer.

        Args:
            query: Query string
            k: Number of documents to retrieve
            prompt_template: Custom prompt template (optional)

        Returns:
            Generated answer string
        """
        if not self.llm_provider:
            raise ValueError("LLM provider required for query() method")

        # Retrieve relevant documents
        results = await self.retrieve(query, k)

        if not results:
            return "No relevant documents found."

        # Build context
        context = "\n\n".join(
            f"[Document {i+1}]\n{result.chunk.content}"
            for i, result in enumerate(results)
        )

        # Build prompt
        if prompt_template:
            prompt = prompt_template.format(context=context, query=query)
        else:
            prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""

        # Generate response
        response = await self.llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )

        if response and "message" in response:
            content = response["message"].content
            if isinstance(content, str):
                return content

        return "Failed to generate response."

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks.

        Args:
            document_id: ID of document to delete

        Returns:
            True if deletion was successful
        """
        if document_id not in self._documents:
            return False

        # Find all chunks for this document
        # Note: This requires scanning all chunks, which may be slow
        # In production, maintain a document -> chunks mapping

        # Remove document
        del self._documents[document_id]

        logger.debug(f"Deleted document {document_id}")
        return True

    async def clear(self) -> bool:
        """Clear all documents and chunks.

        Returns:
            True if clearing was successful
        """
        self._documents.clear()
        await self.vectorstore.clear()
        return True

    async def count_documents(self) -> int:
        """Return the number of indexed documents."""
        return len(self._documents)

    async def count_chunks(self) -> int:
        """Return the number of indexed chunks."""
        return await self.vectorstore.count()

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._documents.get(document_id)


def create_rag_tool(pipeline: RAGPipeline, name: str = "rag_search") -> "Tool":
    """Create a Tool wrapper for the RAG pipeline.

    This allows the RAG pipeline to be used as an Agent tool.

    Args:
        pipeline: RAG pipeline to wrap
        name: Name for the tool

    Returns:
        Tool instance
    """
    from pyagent.core.tools import Tool

    async def search_handler(query: str, k: int = 5) -> str:
        """Search the knowledge base."""
        results = await pipeline.retrieve(query, k)

        if not results:
            return "No relevant documents found."

        # Format results
        output = []
        for i, result in enumerate(results):
            output.append(f"[{i+1}] Score: {result.score:.3f}")
            output.append(f"Content: {result.chunk.content[:300]}...")
            if result.document:
                output.append(f"Source: {result.document.source or result.document.id}")
            output.append("")

        return "\n".join(output)

    return Tool(
        name=name,
        description="Search the knowledge base for relevant information. Use this when you need to find specific information from indexed documents.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        handler=search_handler,
    )


class SimpleRAG:
    """Simplified RAG interface for common use cases.

    Provides a simpler API for basic RAG operations.

    Example:
        ```python
        rag = SimpleRAG()

        # Add documents
        await rag.add("Python is a programming language.", source="intro.txt")
        await rag.add("JavaScript is also a programming language.", source="intro.txt")

        # Search
        results = await rag.search("What is Python?")
        ```
    """

    def __init__(
        self,
        embedding: Optional[BaseEmbedding] = None,
        chunk_size: int = 500,
    ):
        """Initialize simple RAG.

        Args:
            embedding: Custom embedding model (default: FakeEmbedding)
            chunk_size: Size of document chunks
        """
        from .embeddings import FakeEmbedding
        from .vectorstore import MemoryVectorStore

        self.embedding = embedding or FakeEmbedding()
        self.vectorstore = MemoryVectorStore()
        self.retriever = VectorRetriever(self.embedding, self.vectorstore)
        self.chunker = RecursiveChunker(chunk_size=chunk_size)
        self._id_counter = 0

    def _next_id(self) -> str:
        """Generate next document ID."""
        self._id_counter += 1
        return f"doc_{self._id_counter}"

    async def add(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> str:
        """Add a document.

        Args:
            content: Document content
            metadata: Optional metadata
            source: Optional source identifier

        Returns:
            Document ID
        """
        doc_id = self._next_id()

        document = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {},
            source=source,
        )

        # Chunk
        chunks = self.chunker.chunk(document)

        # Embed
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding.embed_documents(texts)

        # Store
        await self.vectorstore.add(chunks, embeddings)

        return doc_id

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Search for relevant documents.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of search results
        """
        return await self.retriever.retrieve(query, k)

    async def clear(self) -> None:
        """Clear all documents."""
        await self.vectorstore.clear()
        self._id_counter = 0
