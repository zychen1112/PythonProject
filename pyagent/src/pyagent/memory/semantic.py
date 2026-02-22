"""Semantic Memory - Long-term memory for facts and preferences."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from pyagent.rag import BaseEmbedding, BaseVectorStore, Chunk, Document, SearchResult


class MemoryItem(BaseModel):
    """A semantic memory item."""
    id: str
    key: str
    value: str
    category: str = "general"
    importance: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = 0


class SemanticMemory:
    """Long-term semantic memory for storing facts and preferences.

    Uses a vector store for semantic search and maintains an in-memory
    index for fast key-based lookups.
    """

    def __init__(
        self,
        vectorstore: Optional[BaseVectorStore] = None,
        embedding: Optional[BaseEmbedding] = None,
    ):
        """Initialize semantic memory.

        Args:
            vectorstore: Vector store for semantic search (optional)
            embedding: Embedding model for queries (optional)
        """
        self.vectorstore = vectorstore
        self.embedding = embedding
        self._items: dict[str, MemoryItem] = {}
        self._id_counter = 0

    def _next_id(self) -> str:
        """Generate next memory ID."""
        self._id_counter += 1
        return f"sem_{self._id_counter}"

    async def store(
        self,
        key: str,
        value: str,
        category: str = "general",
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store a semantic memory.

        Args:
            key: Memory key (e.g., "user_preference_theme")
            value: Memory value
            category: Category for grouping
            importance: Importance score (0-1)
            metadata: Additional metadata

        Returns:
            Memory item ID
        """
        item_id = self._next_id()
        item = MemoryItem(
            id=item_id,
            key=key,
            value=value,
            category=category,
            importance=max(0.0, min(1.0, importance)),
            metadata=metadata or {},
        )

        self._items[key] = item

        # Store in vector store if available
        if self.vectorstore and self.embedding:
            content = f"{key}: {value}"
            embedding = await self.embedding.embed_query(content)
            chunk = Chunk(
                id=item_id,
                document_id=item_id,
                content=content,
                metadata={"key": key, "category": category, "importance": importance},
            )
            await self.vectorstore.add([chunk], [embedding])

        return item_id

    async def recall(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
    ) -> list[MemoryItem]:
        """Recall relevant memories.

        Args:
            query: Search query
            k: Maximum number of results
            category: Filter by category (optional)

        Returns:
            List of matching memory items
        """
        results = []

        # Use vector search if available
        if self.vectorstore and self.embedding:
            query_embedding = await self.embedding.embed_query(query)
            filter_dict = {"category": category} if category else None
            search_results = await self.vectorstore.search(query_embedding, k, filter_dict)

            for sr in search_results:
                key = sr.chunk.metadata.get("key")
                if key and key in self._items:
                    item = self._items[key]
                    item.last_accessed = datetime.now()
                    item.access_count += 1
                    results.append(item)

        # Fall back to keyword search
        if not results:
            query_lower = query.lower()
            for item in self._items.values():
                if category and item.category != category:
                    continue
                if query_lower in item.key.lower() or query_lower in item.value.lower():
                    item.last_accessed = datetime.now()
                    item.access_count += 1
                    results.append(item)
                    if len(results) >= k:
                        break

        # Sort by importance and access count
        results.sort(key=lambda x: (x.importance, x.access_count), reverse=True)
        return results[:k]

    async def forget(self, key: str) -> bool:
        """Forget a specific memory.

        Args:
            key: Memory key to forget

        Returns:
            True if memory was forgotten
        """
        if key not in self._items:
            return False

        item = self._items.pop(key)

        # Remove from vector store
        if self.vectorstore:
            await self.vectorstore.delete([item.id])

        return True

    def get(self, key: str) -> Optional[MemoryItem]:
        """Get a memory by key.

        Args:
            key: Memory key

        Returns:
            Memory item or None
        """
        item = self._items.get(key)
        if item:
            item.last_accessed = datetime.now()
            item.access_count += 1
        return item

    async def update_importance(self, key: str, delta: float) -> None:
        """Update memory importance.

        Args:
            key: Memory key
            delta: Change in importance (-1 to 1)
        """
        if key in self._items:
            new_importance = self._items[key].importance + delta
            self._items[key].importance = max(0.0, min(1.0, new_importance))

    def get_all_keys(self) -> list[str]:
        """Get all memory keys."""
        return list(self._items.keys())

    def get_by_category(self, category: str) -> list[MemoryItem]:
        """Get all memories in a category."""
        return [item for item in self._items.values() if item.category == category]

    async def clear(self) -> None:
        """Clear all memories."""
        self._items.clear()
        if self.vectorstore:
            await self.vectorstore.clear()

    def count(self) -> int:
        """Get the number of stored memories."""
        return len(self._items)
