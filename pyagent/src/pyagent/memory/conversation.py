"""
Conversation memory implementation.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from pyagent.core.message import Message
from pyagent.memory.base import Memory, MemoryEntry


class ConversationMemory(Memory):
    """
    Simple in-memory conversation storage.

    Stores conversation history with optional search capability.
    """

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._entries: deque[tuple[str, MemoryEntry]] = deque(maxlen=max_entries)
        self._id_counter = 0

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Add a memory entry."""
        self._id_counter += 1
        entry_id = f"mem-{self._id_counter}"

        entry = MemoryEntry(
            content=content,
            metadata=metadata or {}
        )

        self._entries.append((entry_id, entry))
        return entry_id

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Get a memory entry by ID."""
        for eid, entry in self._entries:
            if eid == entry_id:
                return entry
        return None

    async def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """
        Search for relevant memories.

        Simple substring search implementation.
        For production use, consider using vector similarity search.
        """
        query_lower = query.lower()
        matches = []

        for eid, entry in self._entries:
            if query_lower in entry.content.lower():
                matches.append(entry)
                if len(matches) >= limit:
                    break

        return matches

    async def clear(self) -> None:
        """Clear all memories."""
        self._entries.clear()

    async def add_message(self, message: Message) -> str:
        """Add a message to memory."""
        content = message.content if isinstance(message.content, str) else str(message.content)
        return await self.add(
            content=content,
            metadata={
                "role": message.role.value,
                "name": message.name
            }
        )

    async def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Get the N most recent entries."""
        entries = list(self._entries)
        return [entry for eid, entry in entries[-n:]]

    def get_all_messages(self) -> list[Message]:
        """Get all entries as Message objects."""
        messages = []
        for eid, entry in self._entries:
            if entry.metadata.get("role"):
                messages.append(Message(
                    role=entry.metadata["role"],
                    content=entry.content,
                    name=entry.metadata.get("name")
                ))
        return messages

    def __len__(self) -> int:
        return len(self._entries)
