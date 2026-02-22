"""Reranker implementations."""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from .base import BaseReranker
from .document import SearchResult

if TYPE_CHECKING:
    from pyagent.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class IdentityReranker(BaseReranker):
    """Identity reranker that doesn't change the order.

    Useful as a default when no reranking is needed.
    """

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return results unchanged, limited to top_k."""
        return results[:top_k]


class LLMReranker(BaseReranker):
    """Reranker using an LLM to evaluate relevance.

    Uses the LLM to score the relevance of each result to the query.
    More accurate but slower than other methods.
    """

    RERANK_PROMPT = """You are a relevance evaluator. Given a query and a document excerpt, rate how relevant the document is to the query on a scale of 0 to 10.

Query: {query}

Document: {document}

Return only a single number between 0 and 10, with no additional text."""

    def __init__(
        self,
        llm_provider: "LLMProvider",
        model: str = "gpt-3.5-turbo",
        batch_size: int = 5,
    ):
        """Initialize the LLM reranker.

        Args:
            llm_provider: LLM provider for scoring
            model: Model to use for scoring
            batch_size: Number of documents to score concurrently
        """
        self.llm_provider = llm_provider
        self.model = model
        self.batch_size = batch_size

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank results using LLM scoring."""
        if not results:
            return []

        # Score each result
        scored_results = []

        for i in range(0, len(results), self.batch_size):
            batch = results[i:i + self.batch_size]
            tasks = [self._score_result(query, result) for result in batch]
            scores = await asyncio.gather(*tasks)

            for result, score in zip(batch, scores):
                scored_results.append(SearchResult(
                    chunk=result.chunk,
                    score=score,
                    document=result.document,
                ))

        # Sort by LLM score
        scored_results.sort(key=lambda x: x.score, reverse=True)

        return scored_results[:top_k]

    async def _score_result(self, query: str, result: SearchResult) -> float:
        """Score a single result using LLM."""
        prompt = self.RERANK_PROMPT.format(
            query=query,
            document=result.chunk.content[:500],  # Limit content length
        )

        try:
            response = await self.llm_provider.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=10,
            )

            if response and "message" in response:
                content = response["message"].content
                if isinstance(content, str):
                    # Extract number from response
                    score_str = content.strip().split()[0]
                    score = float(score_str)
                    return max(0.0, min(10.0, score)) / 10.0  # Normalize to 0-1

        except Exception as e:
            logger.warning(f"Failed to score result: {e}")

        return result.score  # Fall back to original score


class CrossEncoderReranker(BaseReranker):
    """Reranker using a cross-encoder model.

    Uses a cross-encoder model from sentence-transformers for
    accurate relevance scoring.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model
            device: Device to run on (cuda, cpu, mps)
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    def _get_model(self):
        """Get or load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info(f"Loaded cross-encoder model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "CrossEncoder requires 'sentence-transformers'. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank results using cross-encoder."""
        if not results:
            return []

        model = self._get_model()

        # Prepare query-document pairs
        pairs = [(query, result.chunk.content) for result in results]

        # Score in thread pool
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: model.predict(pairs),
        )

        # Create scored results
        scored_results = []
        for result, score in zip(results, scores):
            # Normalize score to 0-1 range (cross-encoder scores vary by model)
            normalized_score = float(score)
            scored_results.append(SearchResult(
                chunk=result.chunk,
                score=normalized_score,
                document=result.document,
            ))

        # Sort by score
        scored_results.sort(key=lambda x: x.score, reverse=True)

        return scored_results[:top_k]


class DiversityReranker(BaseReranker):
    """Reranker that promotes diversity in results.

    Ensures results are diverse by penalizing similarity between
    selected documents.
    """

    def __init__(self, diversity_threshold: float = 0.8):
        """Initialize the diversity reranker.

        Args:
            diversity_threshold: Threshold for considering documents similar
        """
        self.diversity_threshold = diversity_threshold

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank to promote diversity."""
        if not results:
            return []

        selected = []
        remaining = list(results)

        while len(selected) < top_k and remaining:
            # Take the highest scoring remaining result
            best = max(remaining, key=lambda x: x.score)
            remaining.remove(best)
            selected.append(best)

            # Filter out similar results
            remaining = [
                r for r in remaining
                if not self._is_similar(best.chunk.content, r.chunk.content)
            ]

        return selected

    def _is_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar using simple overlap."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        jaccard = overlap / len(words1 | words2)

        return jaccard > self.diversity_threshold
