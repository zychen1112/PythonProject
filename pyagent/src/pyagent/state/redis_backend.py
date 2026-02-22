"""Redis backend for checkpoint storage."""

import asyncio
import json
from datetime import datetime
from typing import Optional

from .checkpoint import Checkpoint, CheckpointBackend


class RedisCheckpointBackend(CheckpointBackend):
    """Redis-based checkpoint storage.

    Provides distributed storage for checkpoints using Redis.
    Supports TTL (time-to-live) for automatic expiration.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: int = 86400,  # 24 hours default
        key_prefix: str = "pyagent:checkpoint:",
    ):
        """Initialize Redis backend.

        Args:
            redis_url: Redis connection URL
            ttl: Time-to-live in seconds (0 = no expiration)
            key_prefix: Key prefix for all checkpoints
        """
        self.redis_url = redis_url
        self.ttl = ttl
        self.key_prefix = key_prefix
        self._client = None

    def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(self.redis_url)
            except ImportError:
                raise ImportError(
                    "Redis backend requires 'redis'. "
                    "Install it with: pip install redis"
                )
        return self._client

    def _get_key(self, checkpoint_id: str) -> str:
        """Get full key for a checkpoint."""
        return f"{self.key_prefix}{checkpoint_id}"

    def _get_thread_key(self, thread_id: str) -> str:
        """Get key for thread's checkpoint list."""
        return f"{self.key_prefix}thread:{thread_id}"

    async def save(self, checkpoint: Checkpoint) -> str:
        """Save a checkpoint."""
        client = self._get_client()
        key = self._get_key(checkpoint.id)
        thread_key = self._get_thread_key(checkpoint.thread_id)

        data = {
            "id": checkpoint.id,
            "thread_id": checkpoint.thread_id,
            "timestamp": checkpoint.timestamp.isoformat(),
            "state": checkpoint.state,
            "metadata": checkpoint.metadata,
        }

        # Save checkpoint
        await client.set(key, json.dumps(data))
        if self.ttl > 0:
            await client.expire(key, self.ttl)

        # Add to thread index
        await client.zadd(
            thread_key,
            {checkpoint.id: checkpoint.timestamp.timestamp()},
        )
        if self.ttl > 0:
            await client.expire(thread_key, self.ttl)

        return checkpoint.id

    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint by ID."""
        client = self._get_client()
        key = self._get_key(checkpoint_id)

        data = await client.get(key)
        if not data:
            return None

        parsed = json.loads(data)
        return Checkpoint(
            id=parsed["id"],
            thread_id=parsed["thread_id"],
            timestamp=datetime.fromisoformat(parsed["timestamp"]),
            state=parsed["state"],
            metadata=parsed["metadata"],
        )

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        client = self._get_client()

        # First get the checkpoint to find thread_id
        checkpoint = await self.load(checkpoint_id)
        if not checkpoint:
            return False

        key = self._get_key(checkpoint_id)
        thread_key = self._get_thread_key(checkpoint.thread_id)

        # Delete checkpoint
        await client.delete(key)

        # Remove from thread index
        await client.zrem(thread_key, checkpoint_id)

        return True

    async def list_by_thread(self, thread_id: str) -> list[Checkpoint]:
        """List checkpoints for a thread."""
        client = self._get_client()
        thread_key = self._get_thread_key(thread_id)

        # Get checkpoint IDs sorted by timestamp (newest first)
        ids = await client.zrevrange(thread_key, 0, -1)

        checkpoints = []
        for cp_id in ids:
            checkpoint = await self.load(cp_id)
            if checkpoint:
                checkpoints.append(checkpoint)

        return checkpoints

    async def clear(self) -> int:
        """Clear all checkpoints.

        Returns:
            Number of checkpoints deleted
        """
        client = self._get_client()

        # Find all checkpoint keys
        pattern = f"{self.key_prefix}*"
        keys = []
        async for key in client.scan_iter(match=pattern):
            keys.append(key)

        if keys:
            await client.delete(*keys)
            return len(keys)
        return 0


class MemoryCheckpointBackend(CheckpointBackend):
    """In-memory checkpoint storage for testing."""

    def __init__(self):
        """Initialize memory backend."""
        self._checkpoints: dict[str, Checkpoint] = {}
        self._thread_index: dict[str, list[str]] = {}

    async def save(self, checkpoint: Checkpoint) -> str:
        """Save a checkpoint."""
        self._checkpoints[checkpoint.id] = checkpoint

        if checkpoint.thread_id not in self._thread_index:
            self._thread_index[checkpoint.thread_id] = []
        self._thread_index[checkpoint.thread_id].append(checkpoint.id)

        return checkpoint.id

    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        if checkpoint_id not in self._checkpoints:
            return False

        checkpoint = self._checkpoints.pop(checkpoint_id)
        if checkpoint.thread_id in self._thread_index:
            try:
                self._thread_index[checkpoint.thread_id].remove(checkpoint_id)
            except ValueError:
                pass

        return True

    async def list_by_thread(self, thread_id: str) -> list[Checkpoint]:
        """List checkpoints for a thread."""
        if thread_id not in self._thread_index:
            return []

        checkpoints = [
            self._checkpoints[cp_id]
            for cp_id in self._thread_index[thread_id]
            if cp_id in self._checkpoints
        ]

        return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)

    async def clear(self) -> int:
        """Clear all checkpoints."""
        count = len(self._checkpoints)
        self._checkpoints.clear()
        self._thread_index.clear()
        return count
