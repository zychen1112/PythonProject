"""RAG Pipeline and Tool integration."""

import logging
from typing import Any, Optional, TYPE_CHECKING

from .base import BaseChunker, BaseEmbedding, BaseReranker, BaseRetriever, BaseVectorStore
from .chunking import RecursiveChunker
from .document import Document, SearchResult
from .reranker import IdentityReranker
from .retriever import VectorRetriever

if TYPE_CHECKING:
    from pyagent.core.tools import Tool
    from pyagent.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG (Retrieval-Augmented Generation) pipeline."""

    def __init__(
        self,
        embedding: BaseEmbedding,
        vectorstore: BaseVectorStore,
        retriever: Optional[BaseRetriever] = None,
        chunker: Optional[BaseChunker] = None,
        reranker: Optional[BaseReranker] = None,
        llm_provider: Optional["LLMProvider"] = None,
    ):
        self.embedding = embedding
        self.vectorstore = vectorstore
        self.retriever = retriever or VectorRetriever(embedding, vectorstore)
        self.chunker = chunker or RecursiveChunker()
        self.reranker = reranker or IdentityReranker()
        self.llm_provider = llm_provider
        self._documents: dict[str, Document] = {}

    async def index(self, documents: list[Document]) -> list[str]:
        all_chunk_ids = []
        for document in documents:
            chunk_ids = await self.add_document(document)
            all_chunk_ids.extend(chunk_ids)
        logger.info(f"Indexed {len(documents)} documents ({len(all_chunk_ids)} chunks)")
        return all_chunk_ids

    async def add_document(self, document: Document) -> list[str]:
        self._documents[document.id] = document
        if hasattr(self.vectorstore, "add_document"):
            self.vectorstore.add_document(document)
        chunks = self.chunker.chunk(document)
        if not chunks:
            return []
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding.embed_documents(texts)
        chunk_ids = await self.vectorstore.add(chunks, embeddings)
        logger.debug(f"Added document {document.id}: {len(chunks)} chunks")
        return chunk_ids

    async def retrieve(self, query: str, k: int = 5, filter: Optional[dict[str, Any]] = None, rerank: bool = True) -> list[SearchResult]:
        results = await self.retriever.retrieve(query, k * 2, filter)
        if rerank and results:
            results = await self.reranker.rerank(query, results, k)
        return results[:k]

    async def query(self, query: str, k: int = 5, prompt_template: Optional[str] = None) -> str:
        if not self.llm_provider:
            raise ValueError("LLM provider required for query() method")
        results = await self.retrieve(query, k)
        if not results:
            return "No relevant documents found."
        context = "\n\n".join(f"[Document {i+1}]\n{result.chunk.content}" for i, result in enumerate(results))
        prompt = prompt_template.format(context=context, query=query) if prompt_template else f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = await self.llm_provider.complete(messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=1000)
        if response and "message" in response:
            content = response["message"].content
            if isinstance(content, str):
                return content
        return "Failed to generate response."

    async def delete_document(self, document_id: str) -> bool:
        if document_id not in self._documents:
            return False
        del self._documents[document_id]
        logger.debug(f"Deleted document {document_id}")
        return True

    async def clear(self) -> bool:
        self._documents.clear()
        await self.vectorstore.clear()
        return True

    async def count_documents(self) -> int:
        return len(self._documents)

    async def count_chunks(self) -> int:
        return await self.vectorstore.count()

    def get_document(self, document_id: str) -> Optional[Document]:
        return self._documents.get(document_id)


def create_rag_tool(pipeline: RAGPipeline, name: str = "rag_search") -> "Tool":
    """Create a Tool wrapper for the RAG pipeline."""
    from pyagent.core.tools import Tool

    async def search_handler(query: str, k: int = 5) -> str:
        results = await pipeline.retrieve(query, k)
        if not results:
            return "No relevant documents found."
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
        description="Search the knowledge base for relevant information.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "k": {"type": "integer", "description": "Number of results", "default": 5},
            },
            "required": ["query"],
        },
        handler=search_handler,
    )


class SimpleRAG:
    """Simplified RAG interface for common use cases."""

    def __init__(self, embedding: Optional[BaseEmbedding] = None, chunk_size: int = 500):
        from .embeddings import FakeEmbedding
        from .vectorstore import MemoryVectorStore

        self.embedding = embedding or FakeEmbedding()
        self.vectorstore = MemoryVectorStore()
        self.retriever = VectorRetriever(self.embedding, self.vectorstore)
        self.chunker = RecursiveChunker(chunk_size=chunk_size)
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"doc_{self._id_counter}"

    async def add(self, content: str, metadata: Optional[dict[str, Any]] = None, source: Optional[str] = None) -> str:
        doc_id = self._next_id()
        document = Document(id=doc_id, content=content, metadata=metadata or {}, source=source)
        chunks = self.chunker.chunk(document)
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding.embed_documents(texts)
        await self.vectorstore.add(chunks, embeddings)
        return doc_id

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return await self.retriever.retrieve(query, k)

    async def clear(self) -> None:
        await self.vectorstore.clear()
        self._id_counter = 0
