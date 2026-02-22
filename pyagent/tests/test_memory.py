"""Tests for enhanced memory system."""

import pytest
from datetime import datetime, timedelta

from pyagent.memory import (
    ConversationMemory,
    SemanticMemory,
    MemoryItem,
    ProceduralMemory,
    Workflow,
    WorkflowStep,
    ExecutionResult,
    EpisodicMemory,
    Episode,
    TimeRange,
    MemoryManager,
)
from pyagent.rag import FakeEmbedding, MemoryVectorStore


class TestSemanticMemory:
    """Tests for SemanticMemory."""

    @pytest.mark.asyncio
    async def test_store_and_recall(self):
        """Test storing and recalling memories."""
        memory = SemanticMemory()

        # Store a memory
        await memory.store(
            key="user_name",
            value="John",
            category="personal",
            importance=0.8,
        )

        # Recall it
        results = await memory.recall("name")
        assert len(results) >= 1
        assert results[0].key == "user_name"
        assert results[0].value == "John"

    @pytest.mark.asyncio
    async def test_forget(self):
        """Test forgetting memories."""
        memory = SemanticMemory()

        await memory.store("test_key", "test_value")
        assert memory.count() == 1

        result = await memory.forget("test_key")
        assert result is True
        assert memory.count() == 0

    @pytest.mark.asyncio
    async def test_get_by_key(self):
        """Test getting memory by key."""
        memory = SemanticMemory()

        await memory.store("favorite_color", "blue")
        item = memory.get("favorite_color")

        assert item is not None
        assert item.value == "blue"
        assert item.access_count >= 1

    @pytest.mark.asyncio
    async def test_update_importance(self):
        """Test updating memory importance."""
        memory = SemanticMemory()

        await memory.store("test", "value", importance=0.5)
        await memory.update_importance("test", 0.3)

        item = memory.get("test")
        assert item.importance == 0.8

    @pytest.mark.asyncio
    async def test_category_filter(self):
        """Test filtering by category."""
        memory = SemanticMemory()

        await memory.store("pref1", "value1", category="preferences")
        await memory.store("fact1", "value2", category="facts")

        prefs = memory.get_by_category("preferences")
        assert len(prefs) == 1
        assert prefs[0].key == "pref1"


class TestProceduralMemory:
    """Tests for ProceduralMemory."""

    @pytest.mark.asyncio
    async def test_learn_workflow(self):
        """Test learning a workflow."""
        memory = ProceduralMemory()

        workflow = Workflow(
            id="",
            name="Test Workflow",
            description="A test workflow",
            steps=[
                WorkflowStep(name="step1", action="do something"),
                WorkflowStep(name="step2", action="do another thing"),
            ],
        )

        wf_id = await memory.learn(workflow)
        assert wf_id is not None
        assert memory.count() == 1

    @pytest.mark.asyncio
    async def test_recall_workflow(self):
        """Test recalling a workflow."""
        memory = ProceduralMemory()

        workflow = Workflow(
            id="",
            name="Deploy Application",
            description="Steps to deploy",
            tags=["deployment", "ci"],
        )
        await memory.learn(workflow)

        recalled = await memory.recall("deploy application")
        assert recalled is not None
        assert "deploy" in recalled.name.lower()

    @pytest.mark.asyncio
    async def test_record_execution(self):
        """Test recording execution results."""
        memory = ProceduralMemory()

        workflow = Workflow(id="wf1", name="Test", description="Test")
        await memory.learn(workflow)

        # Record successful execution
        result = ExecutionResult(workflow_id="wf1", success=True)
        await memory.record_execution("wf1", result)

        wf = memory.get_workflow("wf1")
        assert wf.execution_count == 1
        assert wf.success_rate == 1.0

        # Record failed execution
        failed_result = ExecutionResult(workflow_id="wf1", success=False)
        await memory.record_execution("wf1", failed_result)

        wf = memory.get_workflow("wf1")
        assert wf.execution_count == 2
        assert wf.success_rate == 0.5

    @pytest.mark.asyncio
    async def test_forget_workflow(self):
        """Test forgetting a workflow."""
        memory = ProceduralMemory()

        workflow = Workflow(id="wf1", name="Test", description="Test")
        await memory.learn(workflow)

        result = await memory.forget("wf1")
        assert result is True
        assert memory.count() == 0


class TestEpisodicMemory:
    """Tests for EpisodicMemory."""

    @pytest.mark.asyncio
    async def test_record_episode(self):
        """Test recording an episode."""
        memory = EpisodicMemory()

        episode = Episode(
            id="",
            summary="User asked about Python",
            details="The user asked how to learn Python programming",
            tags=["python", "learning"],
        )

        ep_id = await memory.record(episode)
        assert ep_id is not None
        assert memory.count() == 1

    @pytest.mark.asyncio
    async def test_recall_episode(self):
        """Test recalling episodes."""
        memory = EpisodicMemory()

        await memory.record(Episode(
            id="",
            summary="Login issue",
            details="User had trouble logging in",
        ))

        results = await memory.recall("login")
        assert len(results) >= 1
        assert "login" in results[0].summary.lower()

    @pytest.mark.asyncio
    async def test_get_timeline(self):
        """Test getting timeline."""
        memory = EpisodicMemory()

        now = datetime.now()

        # Record episodes at different times
        ep1 = Episode(id="ep1", summary="First", timestamp=now - timedelta(hours=2))
        ep2 = Episode(id="ep2", summary="Second", timestamp=now - timedelta(hours=1))
        ep3 = Episode(id="ep3", summary="Third", timestamp=now)

        await memory.record(ep1)
        await memory.record(ep2)
        await memory.record(ep3)

        # Get timeline for last 90 minutes
        start = now - timedelta(minutes=90)
        end = now + timedelta(minutes=1)

        timeline = await memory.get_timeline(start, end)
        assert len(timeline) == 2

    @pytest.mark.asyncio
    async def test_time_range(self):
        """Test TimeRange class."""
        now = datetime.now()
        time_range = TimeRange(
            start=now - timedelta(hours=1),
            end=now + timedelta(hours=1),
        )

        assert time_range.contains(now)
        assert not time_range.contains(now - timedelta(hours=2))

    @pytest.mark.asyncio
    async def test_get_recent(self):
        """Test getting recent episodes."""
        memory = EpisodicMemory()

        for i in range(5):
            await memory.record(Episode(id=f"ep{i}", summary=f"Episode {i}"))

        recent = memory.get_recent(3)
        assert len(recent) == 3


class TestMemoryManager:
    """Tests for MemoryManager."""

    @pytest.mark.asyncio
    async def test_remember(self):
        """Test remembering from all memory types."""
        manager = MemoryManager(
            conversation_memory=ConversationMemory(),
            semantic_memory=SemanticMemory(),
            procedural_memory=ProceduralMemory(),
            episodic_memory=EpisodicMemory(),
        )

        # Store something in each memory type
        await manager.semantic.store("test_key", "test_value")
        await manager.episodic.record(Episode(id="", summary="Test episode"))

        memories = await manager.remember("test")

        assert "short_term" in memories
        assert "long_term" in memories
        assert "episodes" in memories
        assert "workflows" in memories

    @pytest.mark.asyncio
    async def test_build_context(self):
        """Test building context."""
        manager = MemoryManager(
            conversation_memory=ConversationMemory(),
            semantic_memory=SemanticMemory(),
            episodic_memory=EpisodicMemory(),
        )

        await manager.semantic.store("user_name", "Alice")
        await manager.episodic.record(Episode(id="", summary="User logged in"))

        context = await manager.build_context("user")

        assert "Alice" in context or "user_name" in context

    @pytest.mark.asyncio
    async def test_store_fact(self):
        """Test storing a fact."""
        manager = MemoryManager(
            semantic_memory=SemanticMemory(),
        )

        await manager.store_fact("favorite_color", "blue", importance=0.7)

        item = manager.semantic.get("favorite_color")
        assert item is not None
        assert item.value == "blue"

    @pytest.mark.asyncio
    async def test_store_workflow(self):
        """Test storing a workflow."""
        manager = MemoryManager(
            procedural_memory=ProceduralMemory(),
        )

        workflow = Workflow(id="", name="Test", description="Test workflow")
        await manager.store_workflow(workflow)

        assert manager.procedural.count() == 1

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting memory stats."""
        manager = MemoryManager(
            conversation_memory=ConversationMemory(),
            semantic_memory=SemanticMemory(),
            procedural_memory=ProceduralMemory(),
            episodic_memory=EpisodicMemory(),
        )

        await manager.semantic.store("test", "value")
        await manager.episodic.record(Episode(id="", summary="Test"))

        stats = manager.get_stats()

        assert stats["semantic_items"] == 1
        assert stats["episodes"] == 1


class TestIntegration:
    """Integration tests for memory system with RAG."""

    @pytest.mark.asyncio
    async def test_semantic_memory_with_vectorstore(self):
        """Test semantic memory with vector store."""
        embedding = FakeEmbedding(dimension=64)
        vectorstore = MemoryVectorStore()
        memory = SemanticMemory(vectorstore=vectorstore, embedding=embedding)

        await memory.store("python", "A programming language")
        await memory.store("java", "Another programming language")

        results = await memory.recall("programming language")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_episodic_memory_with_vectorstore(self):
        """Test episodic memory with vector store."""
        embedding = FakeEmbedding(dimension=64)
        vectorstore = MemoryVectorStore()
        memory = EpisodicMemory(vectorstore=vectorstore, embedding=embedding)

        await memory.record(Episode(
            id="",
            summary="Python tutorial completed",
            details="User finished the Python basics tutorial",
        ))

        results = await memory.recall("python tutorial")
        assert len(results) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
