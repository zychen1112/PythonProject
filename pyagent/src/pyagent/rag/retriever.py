"""Retriever implementations."""

import asyncio
import logging
import math
import re
from collections import Counter
from typing import Any, Optional, TYPE_CHECKING

from .base import BaseEmbedding, BaseRetriever, BaseVectorStore
from .document import Chunk, Document, SearchResult

if TYPE_CHECKING:
    from pyagent.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """Vector similarity retriever."""

    def __init__(self, embedding: BaseEmbedding, vectorstore: BaseVectorStore):
        self.embedding = embedding
        self.vectorstore = vectorstore

    async def retrieve(self, query: str, k: int = 5, filter: Optional[dict[str, Any]] = None) -> list[SearchResult]:
        query_embedding = await self.embedding.embed_query(query)
        return await self.vectorstore.search(query_embedding, k, filter)


class KeywordRetriever(BaseRetriever):
    """Keyword-based retriever using BM25-like scoring."""

    def __init__(self, documents: Optional[list[Document]] = None, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._documents, self._chunks, self._doc_lengths = {}, {}, {}
        self._term_freqs, self._doc_freqs = {}, Counter()
        self._avg_doc_length = 0
        if documents:
            for doc in documents:
                self.add_document(doc)

    def add_document(self, document: Document) -> None:
        self._documents[document.id] = document
        chunk = Chunk(id=f"{document.id}_chunk_0", document_id=document.id, content=document.content, metadata=document.metadata)
        self._chunks[chunk.id] = chunk
        tokens = self._tokenize(document.content)
        self._doc_lengths[chunk.id] = len(tokens)
        self._term_freqs[chunk.id] = Counter(tokens)
        for term in set(tokens):
            self._doc_freqs[term] += 1
        self._avg_doc_length = sum(self._doc_lengths.values()) / len(self._chunks) if self._chunks else 0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _score(self, query_tokens: list[str], chunk_id: str) -> float:
        score, doc_len = 0.0, self._doc_lengths.get(chunk_id, 0)
        doc_term_freqs, N = self._term_freqs.get(chunk_id, Counter()), len(self._chunks)
        for term in query_tokens:
            if term not in self._doc_freqs:
                continue
            df, tf = self._doc_freqs[term], doc_term_freqs.get(term, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self._avg_doc_length))
            score += idf * (numerator / denominator) if denominator > 0 else 0
        return score

    async def retrieve(self, query: str, k: int = 5, filter: Optional[dict[str, Any]] = None) -> list[SearchResult]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self._chunks:
            return []
        scores = []
        for chunk_id, chunk in self._chunks.items():
            if filter and not all(chunk.metadata.get(key) == value for key, value in filter.items()):
                continue
            scores.append((chunk_id, self._score(query_tokens, chunk_id)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [SearchResult(chunk=self._chunks[chunk_id], score=score, document=self._documents.get(self._chunks[chunk_id].document_id))
                for chunk_id, score in scores[:k]]


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector and keyword search."""

    def __init__(self, vector_retriever: VectorRetriever, keyword_retriever: Optional[KeywordRetriever] = None, alpha: float = 0.5):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.alpha = alpha

    async def retrieve(self, query: str, k: int = 5, filter: Optional[dict[str, Any]] = None) -> list[SearchResult]:
        tasks = [self.vector_retriever.retrieve(query, k * 2, filter)]
        if self.keyword_retriever:
            tasks.append(self.keyword_retriever.retrieve(query, k * 2, filter))
        results_list = await asyncio.gather(*tasks)
        combined, scores = {}, {}
        for i, results in enumerate(results_list):
            weight = self.alpha if i == 0 else (1 - self.alpha)
            if results:
                max_score = max(r.score for r in results)
                min_score = min(r.score for r in results)
                score_range = max_score - min_score if max_score != min_score else 1
                for result in results:
                    normalized = (result.score - min_score) / score_range
                    if result.chunk.id not in scores:
                        scores[result.chunk.id], combined[result.chunk.id] = 0, result
                    scores[result.chunk.id] += weight * normalized
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [SearchResult(chunk=combined[chunk_id].chunk, score=score, document=combined[chunk_id].document)
                for chunk_id, score in sorted_results[:k]]


class MultiQueryRetriever(BaseRetriever):
    """Multi-query retriever that generates query variations."""

    QUERY_VARIATION_PROMPT = """Generate 3 different search queries that would help find relevant information for the following question.

Original question: {query}

Return only the 3 queries, one per line."""

    def __init__(self, base_retriever: BaseRetriever, llm_provider: Optional["LLMProvider"] = None):
        self.base_retriever = base_retriever
        self.llm_provider = llm_provider

    async def retrieve(self, query: str, k: int = 5, filter: Optional[dict[str, Any]] = None) -> list[SearchResult]:
        queries = [query]
        if self.llm_provider:
            try:
                variations = await self._generate_variations(query)
                queries.extend(variations)
            except Exception as e:
                logger.warning(f"Failed to generate query variations: {e}")
        all_results, scores = {}, {}
        for q in queries:
            for result in await self.base_retriever.retrieve(q, k, filter):
                chunk_id = result.chunk.id
                if chunk_id not in all_results:
                    all_results[chunk_id], scores[chunk_id] = result, []
                scores[chunk_id].append(result.score)
        aggregated = [SearchResult(chunk=result.chunk, score=max(scores[chunk_id]), document=result.document)
                     for chunk_id, result in all_results.items()]
        aggregated.sort(key=lambda x: x.score, reverse=True)
        return aggregated[:k]

    async def _generate_variations(self, query: str) -> list[str]:
        if not self.llm_provider:
            return []
        response = await self.llm_provider.complete(
            messages=[{"role": "user", "content": self.QUERY_VARIATION_PROMPT.format(query=query)}],
            model="gpt-3.5-turbo", temperature=0.7, max_tokens=200
        )
        if response and "message" in response:
            content = response["message"].content
            if isinstance(content, str):
                return [line.strip() for line in content.strip().split("\n") if line.strip()][:3]
        return []
