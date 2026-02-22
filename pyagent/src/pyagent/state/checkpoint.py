"""Checkpoint data structures and management."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class Checkpoint(BaseModel):
    """A checkpoint representing a point-in-time state.

    Checkpoints can be used to save and restore agent state,
    enabling features like pause/resume, undo, and recovery.
    """
    id: str
    thread_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class CheckpointBackend(ABC):
    """Abstract base class for checkpoint storage backends."""

    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> str:
        """Save a checkpoint.

        Args:
            checkpoint: Checkpoint to save

        Returns:
            Checkpoint ID
        """
        pass

    @abstractmethod
    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint or None if not found
        """
        pass

    @abstractmethod
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    async def list_by_thread(self, thread_id: str) -> list[Checkpoint]:
        """List checkpoints for a thread.

        Args:
            thread_id: Thread ID

        Returns:
            List of checkpoints (newest first)
        """
        pass


class CheckpointManager:
    """Manager for creating, loading, and managing checkpoints.

    Provides a high-level interface for checkpoint operations.
    """

    def __init__(self, backend: CheckpointBackend):
        """Initialize checkpoint manager.

        Args:
            backend: Storage backend
        """
        self.backend = backend
        self._id_counter = 0

    def _next_id(self) -> str:
        """Generate next checkpoint ID."""
        self._id_counter += 1
        return f"cp_{self._id_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    async def save(
        self,
        thread_id: str,
        state: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Checkpoint:
        """Create and save a checkpoint.

        Args:
            thread_id: Thread/session ID
            state: State to save
            metadata: Optional metadata

        Returns:
            Created checkpoint
        """
        checkpoint = Checkpoint(
            id=self._next_id(),
            thread_id=thread_id,
            state=state,
            metadata=metadata or {},
        )

        await self.backend.save(checkpoint)
        return checkpoint

    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint or None
        """
        return await self.backend.load(checkpoint_id)

    async def get_latest(self, thread_id: str) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a thread.

        Args:
            thread_id: Thread ID

        Returns:
            Latest checkpoint or None
        """
        checkpoints = await self.backend.list_by_thread(thread_id)
        return checkpoints[0] if checkpoints else None

    async def list(self, thread_id: str) -> list[Checkpoint]:
        """List all checkpoints for a thread.

        Args:
            thread_id: Thread ID

        Returns:
            List of checkpoints
        """
        return await self.backend.list_by_thread(thread_id)

    async def rollback(self, thread_id: str, checkpoint_id: str) -> Optional[Checkpoint]:
        """Rollback to a specific checkpoint.

        Args:
            thread_id: Thread ID
            checkpoint_id: Checkpoint to rollback to

        Returns:
            The checkpoint that was rolled back to
        """
        checkpoint = await self.backend.load(checkpoint_id)
        if checkpoint and checkpoint.thread_id == thread_id:
            return checkpoint
        return None

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            True if deleted
        """
        return await self.backend.delete(checkpoint_id)

    async def clear_thread(self, thread_id: str) -> int:
        """Clear all checkpoints for a thread.

        Args:
            thread_id: Thread ID

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = await self.backend.list_by_thread(thread_id)
        count = 0
        for cp in checkpoints:
            if await self.backend.delete(cp.id):
                count += 1
        return count
