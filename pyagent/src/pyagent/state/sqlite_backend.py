"""SQLite backend for checkpoint storage."""

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .checkpoint import Checkpoint, CheckpointBackend


class SQLiteCheckpointBackend(CheckpointBackend):
    """SQLite-based checkpoint storage.

    Provides persistent storage for checkpoints using SQLite.
    Suitable for single-machine deployments.
    """

    def __init__(self, db_path: str = "checkpoints.db"):
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self, conn: sqlite3.Connection) -> None:
        """Ensure the checkpoints table exists."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                state TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_id
            ON checkpoints(thread_id, timestamp DESC)
        """)
        conn.commit()

    async def save(self, checkpoint: Checkpoint) -> str:
        """Save a checkpoint."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._save_sync,
            checkpoint,
        )
        return checkpoint.id

    def _save_sync(self, checkpoint: Checkpoint) -> None:
        """Synchronous save implementation."""
        conn = self._get_connection()
        try:
            self._ensure_table(conn)
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (id, thread_id, timestamp, state, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.id,
                    checkpoint.thread_id,
                    checkpoint.timestamp.isoformat(),
                    json.dumps(checkpoint.state),
                    json.dumps(checkpoint.metadata),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint by ID."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._load_sync,
            checkpoint_id,
        )

    def _load_sync(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Synchronous load implementation."""
        conn = self._get_connection()
        try:
            self._ensure_table(conn)
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE id = ?",
                (checkpoint_id,),
            ).fetchone()

            if row:
                return Checkpoint(
                    id=row["id"],
                    thread_id=row["thread_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    state=json.loads(row["state"]),
                    metadata=json.loads(row["metadata"]),
                )
            return None
        finally:
            conn.close()

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._delete_sync,
            checkpoint_id,
        )

    def _delete_sync(self, checkpoint_id: str) -> bool:
        """Synchronous delete implementation."""
        conn = self._get_connection()
        try:
            self._ensure_table(conn)
            cursor = conn.execute(
                "DELETE FROM checkpoints WHERE id = ?",
                (checkpoint_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    async def list_by_thread(self, thread_id: str) -> list[Checkpoint]:
        """List checkpoints for a thread."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._list_by_thread_sync,
            thread_id,
        )

    def _list_by_thread_sync(self, thread_id: str) -> list[Checkpoint]:
        """Synchronous list implementation."""
        conn = self._get_connection()
        try:
            self._ensure_table(conn)
            rows = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE thread_id = ?
                ORDER BY timestamp DESC
                """,
                (thread_id,),
            ).fetchall()

            return [
                Checkpoint(
                    id=row["id"],
                    thread_id=row["thread_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    state=json.loads(row["state"]),
                    metadata=json.loads(row["metadata"]),
                )
                for row in rows
            ]
        finally:
            conn.close()

    async def clear(self) -> int:
        """Clear all checkpoints.

        Returns:
            Number of checkpoints deleted
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._clear_sync)

    def _clear_sync(self) -> int:
        """Synchronous clear implementation."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM checkpoints")
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
