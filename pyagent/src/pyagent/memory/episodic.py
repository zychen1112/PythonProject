"""Episodic Memory - Memory for experiences and events."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from pyagent.rag import BaseEmbedding, BaseVectorStore, Chunk


class Episode(BaseModel):
    """An episodic memory (event/experience)."""
    id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    summary: str
    details: str = ""
    participants: list[str] = Field(default_factory=list)
    outcome: str = ""
    emotions: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None


class TimeRange:
    """A time range for querying episodes."""

    def __init__(self, start: datetime, end: datetime):
        """Initialize time range.

        Args:
            start: Start datetime
            end: End datetime
        """
        if start > end:
            raise ValueError("Start must be before end")
        self.start = start
        self.end = end

    def contains(self, dt: datetime) -> bool:
        """Check if a datetime is within the range."""
        return self.start <= dt <= self.end


class EpisodicMemory:
    """Episodic memory for storing experiences and events.

    Stores past experiences organized by time, with optional
    semantic search capabilities.
    """

    def __init__(
        self,
        vectorstore: Optional[BaseVectorStore] = None,
        embedding: Optional[BaseEmbedding] = None,
    ):
        """Initialize episodic memory.

        Args:
            vectorstore: Vector store for semantic search (optional)
            embedding: Embedding model (optional)
        """
        self.vectorstore = vectorstore
        self.embedding = embedding
        self._episodes: dict[str, Episode] = {}
        self._id_counter = 0

    def _next_id(self) -> str:
        """Generate next episode ID."""
        self._id_counter += 1
        return f"ep_{self._id_counter}"

    async def record(self, episode: Episode) -> str:
        """Record a new episode.

        Args:
            episode: Episode to record

        Returns:
            Episode ID
        """
        if not episode.id:
            episode.id = self._next_id()

        # Generate embedding if needed
        if self.vectorstore and self.embedding and not episode.embedding:
            content = f"{episode.summary}\n{episode.details}"
            episode.embedding = await self.embedding.embed_query(content)

            # Store in vector store
            chunk = Chunk(
                id=episode.id,
                document_id=episode.id,
                content=content,
                metadata={
                    "timestamp": episode.timestamp.isoformat(),
                    "tags": episode.tags,
                    "outcome": episode.outcome,
                },
            )
            await self.vectorstore.add([chunk], [episode.embedding])

        self._episodes[episode.id] = episode
        return episode.id

    async def recall(
        self,
        query: str,
        time_range: Optional[TimeRange] = None,
        k: int = 5,
    ) -> list[Episode]:
        """Recall relevant episodes.

        Args:
            query: Search query
            time_range: Optional time range filter
            k: Maximum results

        Returns:
            List of matching episodes
        """
        results = []

        # Use vector search if available
        if self.vectorstore and self.embedding:
            query_embedding = await self.embedding.embed_query(query)
            search_results = await self.vectorstore.search(query_embedding, k * 2)

            for sr in search_results:
                episode_id = sr.chunk.id
                if episode_id in self._episodes:
                    episode = self._episodes[episode_id]
                    if time_range and not time_range.contains(episode.timestamp):
                        continue
                    results.append(episode)

        # Fall back to keyword search
        if not results:
            query_lower = query.lower()
            for episode in sorted(
                self._episodes.values(),
                key=lambda e: e.timestamp,
                reverse=True
            ):
                if time_range and not time_range.contains(episode.timestamp):
                    continue

                # Check summary and details
                if (query_lower in episode.summary.lower() or
                    query_lower in episode.details.lower() or
                    any(query_lower in tag.lower() for tag in episode.tags)):
                    results.append(episode)
                    if len(results) >= k:
                        break

        return results[:k]

    async def get_timeline(
        self,
        start: datetime,
        end: datetime,
    ) -> list[Episode]:
        """Get episodes within a time range.

        Args:
            start: Start datetime
            end: End datetime

        Returns:
            List of episodes in chronological order
        """
        episodes = [
            ep for ep in self._episodes.values()
            if start <= ep.timestamp <= end
        ]
        return sorted(episodes, key=lambda e: e.timestamp)

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get an episode by ID.

        Args:
            episode_id: Episode ID

        Returns:
            Episode or None
        """
        return self._episodes.get(episode_id)

    async def find_similar_episodes(
        self,
        episode_id: str,
        k: int = 5,
    ) -> list[Episode]:
        """Find episodes similar to a given one.

        Args:
            episode_id: Reference episode ID
            k: Maximum results

        Returns:
            List of similar episodes
        """
        if episode_id not in self._episodes:
            return []

        episode = self._episodes[episode_id]

        if not self.vectorstore or not self.embedding:
            # Fall back to tag matching
            results = []
            for ep in self._episodes.values():
                if ep.id == episode_id:
                    continue
                common_tags = set(ep.tags) & set(episode.tags)
                if common_tags:
                    results.append((ep, len(common_tags)))

            results.sort(key=lambda x: x[1], reverse=True)
            return [ep for ep, _ in results[:k]]

        # Use embedding similarity
        if not episode.embedding:
            content = f"{episode.summary}\n{episode.details}"
            episode.embedding = await self.embedding.embed_query(content)

        search_results = await self.vectorstore.search(episode.embedding, k + 1)

        results = []
        for sr in search_results:
            if sr.chunk.id != episode_id and sr.chunk.id in self._episodes:
                results.append(self._episodes[sr.chunk.id])

        return results[:k]

    async def forget(self, episode_id: str) -> bool:
        """Forget an episode.

        Args:
            episode_id: Episode ID to forget

        Returns:
            True if forgotten
        """
        if episode_id in self._episodes:
            del self._episodes[episode_id]
            if self.vectorstore:
                await self.vectorstore.delete([episode_id])
            return True
        return False

    async def clear(self) -> None:
        """Clear all episodes."""
        self._episodes.clear()
        if self.vectorstore:
            await self.vectorstore.clear()

    def count(self) -> int:
        """Get the number of episodes."""
        return len(self._episodes)

    def get_recent(self, n: int = 10) -> list[Episode]:
        """Get most recent episodes.

        Args:
            n: Number of episodes

        Returns:
            List of recent episodes
        """
        episodes = sorted(
            self._episodes.values(),
            key=lambda e: e.timestamp,
            reverse=True
        )
        return episodes[:n]
