"""State persistence module for PyAgent.

This module provides checkpoint-based state persistence:
- Checkpoint: Point-in-time state snapshot
- CheckpointManager: High-level checkpoint operations
- Backends: SQLite, Redis, Memory storage
- Serializer: State serialization utilities
"""

from .checkpoint import Checkpoint, CheckpointBackend, CheckpointManager
from .sqlite_backend import SQLiteCheckpointBackend
from .redis_backend import RedisCheckpointBackend, MemoryCheckpointBackend
from .serializer import StateSerializer, serialize_state, deserialize_state

__all__ = [
    # Checkpoint
    "Checkpoint",
    "CheckpointBackend",
    "CheckpointManager",
    # Backends
    "SQLiteCheckpointBackend",
    "RedisCheckpointBackend",
    "MemoryCheckpointBackend",
    # Serializer
    "StateSerializer",
    "serialize_state",
    "deserialize_state",
]
