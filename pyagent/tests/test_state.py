"""Tests for state persistence module."""

import pytest
from datetime import datetime

from pyagent.state import (
    Checkpoint,
    CheckpointManager,
    MemoryCheckpointBackend,
    SQLiteCheckpointBackend,
    StateSerializer,
    serialize_state,
    deserialize_state,
)


class TestCheckpoint:
    """Tests for Checkpoint data structure."""

    def test_checkpoint_creation(self):
        """Test creating a checkpoint."""
        cp = Checkpoint(
            id="cp_1",
            thread_id="thread_1",
            state={"key": "value"},
            metadata={"source": "test"},
        )

        assert cp.id == "cp_1"
        assert cp.thread_id == "thread_1"
        assert cp.state == {"key": "value"}
        assert isinstance(cp.timestamp, datetime)

    def test_checkpoint_defaults(self):
        """Test checkpoint default values."""
        cp = Checkpoint(id="cp_1", thread_id="t1")
        assert cp.state == {}
        assert cp.metadata == {}


class TestMemoryCheckpointBackend:
    """Tests for MemoryCheckpointBackend."""

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """Test saving and loading checkpoints."""
        backend = MemoryCheckpointBackend()

        cp = Checkpoint(
            id="cp_1",
            thread_id="thread_1",
            state={"data": "test"},
        )

        await backend.save(cp)
        loaded = await backend.load("cp_1")

        assert loaded is not None
        assert loaded.id == "cp_1"
        assert loaded.state == {"data": "test"}

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting checkpoints."""
        backend = MemoryCheckpointBackend()

        cp = Checkpoint(id="cp_1", thread_id="t1", state={})
        await backend.save(cp)

        result = await backend.delete("cp_1")
        assert result is True

        loaded = await backend.load("cp_1")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_by_thread(self):
        """Test listing checkpoints by thread."""
        backend = MemoryCheckpointBackend()

        await backend.save(Checkpoint(id="cp_1", thread_id="t1", state={"i": 1}))
        await backend.save(Checkpoint(id="cp_2", thread_id="t1", state={"i": 2}))
        await backend.save(Checkpoint(id="cp_3", thread_id="t2", state={"i": 3}))

        checkpoints = await backend.list_by_thread("t1")
        assert len(checkpoints) == 2

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing all checkpoints."""
        backend = MemoryCheckpointBackend()

        await backend.save(Checkpoint(id="cp_1", thread_id="t1", state={}))
        await backend.save(Checkpoint(id="cp_2", thread_id="t2", state={}))

        count = await backend.clear()
        assert count == 2


class TestSQLiteCheckpointBackend:
    """Tests for SQLiteCheckpointBackend."""

    @pytest.fixture
    def backend(self, tmp_path):
        """Create a SQLite backend with temp database."""
        db_path = str(tmp_path / "test_checkpoints.db")
        return SQLiteCheckpointBackend(db_path)

    @pytest.mark.asyncio
    async def test_save_and_load(self, backend):
        """Test saving and loading with SQLite."""
        cp = Checkpoint(
            id="cp_1",
            thread_id="thread_1",
            state={"key": "value"},
            metadata={"meta": "data"},
        )

        await backend.save(cp)
        loaded = await backend.load("cp_1")

        assert loaded is not None
        assert loaded.id == "cp_1"
        assert loaded.state == {"key": "value"}
        assert loaded.metadata == {"meta": "data"}

    @pytest.mark.asyncio
    async def test_list_by_thread(self, backend):
        """Test listing by thread with SQLite."""
        await backend.save(Checkpoint(id="cp_1", thread_id="t1", state={}))
        await backend.save(Checkpoint(id="cp_2", thread_id="t1", state={}))
        await backend.save(Checkpoint(id="cp_3", thread_id="t2", state={}))

        checkpoints = await backend.list_by_thread("t1")
        assert len(checkpoints) == 2

    @pytest.mark.asyncio
    async def test_delete(self, backend):
        """Test deleting with SQLite."""
        await backend.save(Checkpoint(id="cp_1", thread_id="t1", state={}))

        result = await backend.delete("cp_1")
        assert result is True

        loaded = await backend.load("cp_1")
        assert loaded is None


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.mark.asyncio
    async def test_save_checkpoint(self):
        """Test saving checkpoint through manager."""
        backend = MemoryCheckpointBackend()
        manager = CheckpointManager(backend)

        cp = await manager.save(
            thread_id="thread_1",
            state={"count": 1},
            metadata={"source": "test"},
        )

        assert cp.id is not None
        assert cp.thread_id == "thread_1"

    @pytest.mark.asyncio
    async def test_get_latest(self):
        """Test getting latest checkpoint."""
        backend = MemoryCheckpointBackend()
        manager = CheckpointManager(backend)

        await manager.save("t1", {"i": 1})
        await manager.save("t1", {"i": 2})
        await manager.save("t2", {"i": 3})

        latest = await manager.get_latest("t1")
        assert latest is not None
        assert latest.state == {"i": 2}

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """Test listing checkpoints."""
        backend = MemoryCheckpointBackend()
        manager = CheckpointManager(backend)

        await manager.save("t1", {"i": 1})
        await manager.save("t1", {"i": 2})

        checkpoints = await manager.list("t1")
        assert len(checkpoints) == 2

    @pytest.mark.asyncio
    async def test_rollback(self):
        """Test rollback to checkpoint."""
        backend = MemoryCheckpointBackend()
        manager = CheckpointManager(backend)

        cp = await manager.save("t1", {"version": 1})

        loaded = await manager.rollback("t1", cp.id)
        assert loaded is not None
        assert loaded.state == {"version": 1}

    @pytest.mark.asyncio
    async def test_clear_thread(self):
        """Test clearing thread checkpoints."""
        backend = MemoryCheckpointBackend()
        manager = CheckpointManager(backend)

        await manager.save("t1", {})
        await manager.save("t1", {})
        await manager.save("t2", {})

        count = await manager.clear_thread("t1")
        assert count == 2


class TestStateSerializer:
    """Tests for StateSerializer."""

    def test_serialize_dict(self):
        """Test serializing a dictionary."""
        state = {"key": "value", "number": 42}
        data = serialize_state(state)

        assert isinstance(data, bytes)
        loaded = deserialize_state(data)
        assert loaded == state

    def test_serialize_nested(self):
        """Test serializing nested structures."""
        state = {
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"x": 2}],
        }
        data = serialize_state(state)
        loaded = deserialize_state(data)
        assert loaded == state

    def test_serialize_datetime(self):
        """Test serializing datetime objects."""
        now = datetime.now()
        state = {"timestamp": now}

        data = serialize_state(state)
        loaded = deserialize_state(data)

        assert isinstance(loaded["timestamp"], datetime)

    def test_serialize_set(self):
        """Test serializing set objects."""
        state = {"items": {1, 2, 3}}

        data = serialize_state(state)
        loaded = deserialize_state(data)

        assert isinstance(loaded["items"], set)
        assert loaded["items"] == {1, 2, 3}

    def test_register_custom_type(self):
        """Test registering custom type encoder/decoder."""

        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        StateSerializer.register_encoder(
            Point,
            lambda p: {"x": p.x, "y": p.y},
            lambda d: Point(d["x"], d["y"]),
        )

        state = {"point": Point(10, 20)}
        data = serialize_state(state)
        loaded = deserialize_state(data)

        assert isinstance(loaded["point"], Point)
        assert loaded["point"].x == 10
        assert loaded["point"].y == 20


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_checkpoint_with_complex_state(self):
        """Test checkpoint with complex serialized state."""
        backend = MemoryCheckpointBackend()
        manager = CheckpointManager(backend)

        state = {
            "messages": ["Hello", "World"],
            "counters": {"a": 1, "b": 2},
            "timestamp": datetime.now(),
            "metadata": {"source": "test"},
        }

        cp = await manager.save("thread_1", state)
        loaded = await manager.load(cp.id)

        assert loaded is not None
        assert loaded.state["messages"] == ["Hello", "World"]
        assert isinstance(loaded.state["timestamp"], datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
