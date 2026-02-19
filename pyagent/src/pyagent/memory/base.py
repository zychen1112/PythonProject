"""
Base memory interface.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class MemoryEntry(BaseModel):
    """A single memory entry."""
    content: str
    metadata: dict[str, Any] = {}


class Memory(ABC):
    """
    Abstract base class for memory systems.
    """

    @abstractmethod
    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Add a memory entry.

        Args:
            content: The content to remember
            metadata: Optional metadata

        Returns:
            ID of the stored entry
        """
        pass

    @abstractmethod
    async def get(self, entry_id: str) -> MemoryEntry | None:
        """
        Get a memory entry by ID.

        Args:
            entry_id: The entry ID

        Returns:
            The memory entry, or None if not found
        """
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """
        Search for relevant memories.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching memory entries
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memories."""
        pass
