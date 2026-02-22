"""Vector store implementations."""

import asyncio
import logging
import math
from typing import Any, Optional

from .base import BaseVectorStore
from .document import Chunk, Document, SearchResult

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class MemoryVectorStore(BaseVectorStore):
    """In-memory vector store for testing and small datasets."""

    def __init__(self):
        self._chunks: dict[str, Chunk] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._documents: dict[str, Document] = {}

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> list[str]:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        ids = []
        for chunk, embedding in zip(chunks, embeddings):
            self._chunks[chunk.id] = chunk
            self._embeddings[chunk.id] = embedding
            ids.append(chunk.id)
        return ids

    async def search(self, query_embedding: list[float], k: int = 5, filter: Optional[dict[str, Any]] = None) -> list[SearchResult]:
        if not self._chunks:
            return []
        similarities = []
        for chunk_id, embedding in self._embeddings.items():
            chunk = self._chunks[chunk_id]
            if filter and not self._matches_filter(chunk, filter):
                continue
            score = cosine_similarity(query_embedding, embedding)
            similarities.append((chunk_id, score))
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for chunk_id, score in similarities[:k]:
            chunk = self._chunks[chunk_id]
            document = self._documents.get(chunk.document_id)
            results.append(SearchResult(chunk=chunk, score=score, document=document))
        return results

    def _matches_filter(self, chunk: Chunk, filter: dict[str, Any]) -> bool:
        for key, value in filter.items():
            if key not in chunk.metadata or chunk.metadata[key] != value:
                return False
        return True

    async def delete(self, ids: list[str]) -> bool:
        for id in ids:
            self._chunks.pop(id, None)
            self._embeddings.pop(id, None)
        return True

    async def get(self, id: str) -> Optional[Chunk]:
        return self._chunks.get(id)

    async def count(self) -> int:
        return len(self._chunks)

    async def clear(self) -> bool:
        self._chunks.clear()
        self._embeddings.clear()
        self._documents.clear()
        return True

    def add_document(self, document: Document) -> None:
        self._documents[document.id] = document


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""

    def __init__(self, collection_name: str = "default", persist_directory: Optional[str] = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
        self._documents: dict[str, Document] = {}

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
                if self.persist_directory:
                    self._client = chromadb.PersistentClient(path=self.persist_directory)
                else:
                    self._client = chromadb.Client()
            except ImportError:
                raise ImportError("ChromaDB requires 'chromadb'. pip install chromadb")
        return self._client

    def _get_collection(self):
        if self._collection is None:
            self._collection = self._get_client().get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> list[str]:
        collection = self._get_collection()
        ids, documents, metadatas = [], [], []
        for chunk, embedding in zip(chunks, embeddings):
            ids.append(chunk.id)
            documents.append(chunk.content)
            metadata = {"document_id": chunk.document_id, "start_index": chunk.start_index, "end_index": chunk.end_index, **chunk.metadata}
            metadatas.append(metadata)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas))
        return ids

    async def search(self, query_embedding: list[float], k: int = 5, filter: Optional[dict[str, Any]] = None) -> list[SearchResult]:
        collection = self._get_collection()
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(query_embeddings=[query_embedding], n_results=k, where=filter, include=["documents", "metadatas", "distances"])
        )
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                chunk = Chunk(
                    id=chunk_id, document_id=metadata.pop("document_id", ""),
                    content=results["documents"][0][i], metadata=metadata,
                    start_index=metadata.pop("start_index", 0), end_index=metadata.pop("end_index", 0)
                )
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance
                document = self._documents.get(chunk.document_id)
                search_results.append(SearchResult(chunk=chunk, score=score, document=document))
        return search_results

    async def delete(self, ids: list[str]) -> bool:
        self._get_collection().delete(ids=ids)
        return True

    async def get(self, id: str) -> Optional[Chunk]:
        results = self._get_collection().get(ids=[id], include=["documents", "metadatas"])
        if results and results["ids"]:
            metadata = results["metadatas"][0] if results["metadatas"] else {}
            return Chunk(id=results["ids"][0], document_id=metadata.pop("document_id", ""), content=results["documents"][0] if results["documents"] else "", metadata=metadata)
        return None

    async def count(self) -> int:
        return self._get_collection().count()

    async def clear(self) -> bool:
        self._get_client().delete_collection(self.collection_name)
        self._collection = None
        return True

    def add_document(self, document: Document) -> None:
        self._documents[document.id] = document
