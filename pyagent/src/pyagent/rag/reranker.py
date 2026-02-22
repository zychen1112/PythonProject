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
    """Identity reranker that doesn't change the order."""

    async def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        return results[:top_k]


class LLMReranker(BaseReranker):
    """Reranker using an LLM to evaluate relevance."""

    RERANK_PROMPT = """You are a relevance evaluator. Rate how relevant this document is to the query on a scale of 0 to 10.

Query: {query}
Document: {document}

Return only a single number between 0 and 10."""

    def __init__(self, llm_provider: "LLMProvider", model: str = "gpt-3.5-turbo", batch_size: int = 5):
        self.llm_provider = llm_provider
        self.model = model
        self.batch_size = batch_size

    async def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not results:
            return []
        scored_results = []
        for i in range(0, len(results), self.batch_size):
            batch = results[i:i + self.batch_size]
            scores = await asyncio.gather(*[self._score_result(query, r) for r in batch])
            for result, score in zip(batch, scores):
                scored_results.append(SearchResult(chunk=result.chunk, score=score, document=result.document))
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    async def _score_result(self, query: str, result: SearchResult) -> float:
        prompt = self.RERANK_PROMPT.format(query=query, document=result.chunk.content[:500])
        try:
            response = await self.llm_provider.complete(
                messages=[{"role": "user", "content": prompt}], model=self.model, temperature=0.1, max_tokens=10
            )
            if response and "message" in response:
                content = response["message"].content
                if isinstance(content, str):
                    return max(0.0, min(1.0, float(content.strip().split()[0]) / 10.0))
        except Exception as e:
            logger.warning(f"Failed to score result: {e}")
        return result.score


class CrossEncoderReranker(BaseReranker):
    """Reranker using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
            except ImportError:
                raise ImportError("CrossEncoder requires 'sentence-transformers'. pip install sentence-transformers")
        return self._model

    async def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not results:
            return []
        model = self._get_model()
        pairs = [(query, result.chunk.content) for result in results]
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, lambda: model.predict(pairs))
        scored_results = [SearchResult(chunk=result.chunk, score=float(score), document=result.document)
                         for result, score in zip(results, scores)]
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]


class DiversityReranker(BaseReranker):
    """Reranker that promotes diversity in results."""

    def __init__(self, diversity_threshold: float = 0.8):
        self.diversity_threshold = diversity_threshold

    async def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not results:
            return []
        selected, remaining = [], list(results)
        while len(selected) < top_k and remaining:
            best = max(remaining, key=lambda x: x.score)
            remaining.remove(best)
            selected.append(best)
            remaining = [r for r in remaining if not self._is_similar(best.chunk.content, r.chunk.content)]
        return selected

    def _is_similar(self, text1: str, text2: str) -> bool:
        words1, words2 = set(text1.lower().split()), set(text2.lower().split())
        if not words1 or not words2:
            return False
        jaccard = len(words1 & words2) / len(words1 | words2)
        return jaccard > self.diversity_threshold
