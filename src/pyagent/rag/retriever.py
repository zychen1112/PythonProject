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
    """Vector similarity retriever.

    Retrieves documents based on embedding similarity.
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        vectorstore: BaseVectorStore,
    ):
        """Initialize the vector retriever.

        Args:
            embedding: Embedding model for queries
            vectorstore: Vector store to search
        """
        self.embedding = embedding
        self.vectorstore = vectorstore

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Retrieve documents using vector similarity."""
        # Embed the query
        query_embedding = await self.embedding.embed_query(query)

        # Search the vector store
        results = await self.vectorstore.search(query_embedding, k, filter)

        return results


class KeywordRetriever(BaseRetriever):
    """Keyword-based retriever using BM25-like scoring.

    Simple implementation of BM25 for keyword retrieval.
    Good for exact term matching.
    """

    def __init__(
        self,
        documents: Optional[list[Document]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Initialize the keyword retriever.

        Args:
            documents: Documents to index
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (document length normalization)
        """
        self.k1 = k1
        self.b = b
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}
        self._doc_lengths: dict[str, int] = {}
        self._term_freqs: dict[str, Counter] = {}
        self._doc_freqs: Counter = Counter()
        self._avg_doc_length: float = 0

        if documents:
            for doc in documents:
                self.add_document(doc)

    def add_document(self, document: Document) -> None:
        """Add a document to the index."""
        self._documents[document.id] = document

        # Create a single chunk for the document
        chunk = Chunk(
            id=f"{document.id}_chunk_0",
            document_id=document.id,
            content=document.content,
            metadata=document.metadata,
        )
        self._chunks[chunk.id] = chunk

        # Tokenize and index
        tokens = self._tokenize(document.content)
        self._doc_lengths[chunk.id] = len(tokens)
        self._term_freqs[chunk.id] = Counter(tokens)

        for term in set(tokens):
            self._doc_freqs[term] += 1

        # Update average document length
        total_docs = len(self._chunks)
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = total_length / total_docs if total_docs > 0 else 0

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase terms."""
        # Simple tokenization: lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _score(self, query_tokens: list[str], chunk_id: str) -> float:
        """Calculate BM25 score for a chunk."""
        score = 0.0
        doc_len = self._doc_lengths.get(chunk_id, 0)
        doc_term_freqs = self._term_freqs.get(chunk_id, Counter())
        N = len(self._chunks)

        for term in query_tokens:
            if term not in self._doc_freqs:
                continue

            # IDF calculation
            df = self._doc_freqs[term]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            # TF calculation
            tf = doc_term_freqs.get(term, 0)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_len / self._avg_doc_length)
            )
            score += idf * (numerator / denominator)

        return score

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Retrieve documents using BM25."""
        query_tokens = self._tokenize(query)

        if not query_tokens or not self._chunks:
            return []

        # Score all chunks
        scores = []
        for chunk_id, chunk in self._chunks.items():
            # Apply filter if provided
            if filter:
                matches = all(
                    chunk.metadata.get(key) == value
                    for key, value in filter.items()
                )
                if not matches:
                    continue

            score = self._score(query_tokens, chunk_id)
            scores.append((chunk_id, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = []
        for chunk_id, score in scores[:k]:
            chunk = self._chunks[chunk_id]
            document = self._documents.get(chunk.document_id)

            results.append(SearchResult(
                chunk=chunk,
                score=score,
                document=document,
            ))

        return results


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector and keyword search.

    Combines results from multiple retrievers using weighted scoring.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: Optional[KeywordRetriever] = None,
        alpha: float = 0.5,
    ):
        """Initialize the hybrid retriever.

        Args:
            vector_retriever: Vector similarity retriever
            keyword_retriever: Keyword retriever (optional)
            alpha: Weight for vector scores (0-1), keyword weight is (1-alpha)
        """
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.alpha = alpha

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Retrieve using hybrid search."""
        tasks = [
            self.vector_retriever.retrieve(query, k * 2, filter),
        ]

        if self.keyword_retriever:
            tasks.append(self.keyword_retriever.retrieve(query, k * 2, filter))

        results_list = await asyncio.gather(*tasks)

        # Normalize and combine scores
        combined: dict[str, SearchResult] = {}
        scores: dict[str, float] = {}

        for i, results in enumerate(results_list):
            weight = self.alpha if i == 0 else (1 - self.alpha)

            # Normalize scores
            if results:
                max_score = max(r.score for r in results)
                min_score = min(r.score for r in results)
                score_range = max_score - min_score if max_score != min_score else 1

                for result in results:
                    normalized = (result.score - min_score) / score_range
                    chunk_id = result.chunk.id

                    if chunk_id not in scores:
                        scores[chunk_id] = 0
                        combined[chunk_id] = result

                    scores[chunk_id] += weight * normalized

        # Sort by combined score
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Return top k
        return [
            SearchResult(
                chunk=combined[chunk_id].chunk,
                score=score,
                document=combined[chunk_id].document,
            )
            for chunk_id, score in sorted_results[:k]
        ]


class MultiQueryRetriever(BaseRetriever):
    """Multi-query retriever that generates query variations.

    Uses an LLM to generate multiple query variations and
    aggregates results from all variations.
    """

    QUERY_VARIATION_PROMPT = """Generate 3 different search queries that would help find relevant information for the following question. Each query should use different keywords or phrasings.

Original question: {query}

Return only the 3 queries, one per line, without numbering or additional text."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_provider: Optional["LLMProvider"] = None,
    ):
        """Initialize the multi-query retriever.

        Args:
            base_retriever: Base retriever to use for each query
            llm_provider: LLM provider for generating query variations
        """
        self.base_retriever = base_retriever
        self.llm_provider = llm_provider

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Retrieve using multiple query variations."""
        queries = [query]

        # Generate query variations if LLM is available
        if self.llm_provider:
            try:
                variations = await self._generate_variations(query)
                queries.extend(variations)
            except Exception as e:
                logger.warning(f"Failed to generate query variations: {e}")

        # Retrieve for all queries
        all_results: dict[str, SearchResult] = {}
        scores: dict[str, list[float]] = {}

        for q in queries:
            results = await self.base_retriever.retrieve(q, k, filter)

            for result in results:
                chunk_id = result.chunk.id
                if chunk_id not in all_results:
                    all_results[chunk_id] = result
                    scores[chunk_id] = []

                scores[chunk_id].append(result.score)

        # Aggregate scores (average of max scores per query)
        aggregated = []
        for chunk_id, result in all_results.items():
            # Use max score from any query
            max_score = max(scores[chunk_id])
            aggregated.append(SearchResult(
                chunk=result.chunk,
                score=max_score,
                document=result.document,
            ))

        # Sort and return top k
        aggregated.sort(key=lambda x: x.score, reverse=True)
        return aggregated[:k]

    async def _generate_variations(self, query: str) -> list[str]:
        """Generate query variations using LLM."""
        if not self.llm_provider:
            return []

        prompt = self.QUERY_VARIATION_PROMPT.format(query=query)

        response = await self.llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-3.5-turbo",  # Use cheaper model for query generation
            temperature=0.7,
            max_tokens=200,
        )

        variations = []
        if response and "message" in response:
            content = response["message"].content
            if isinstance(content, str):
                variations = [
                    line.strip()
                    for line in content.strip().split("\n")
                    if line.strip()
                ]

        return variations[:3]  # Limit to 3 variations
