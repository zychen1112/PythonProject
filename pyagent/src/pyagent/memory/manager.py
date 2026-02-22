"""Memory Manager - Unified memory management."""

from typing import Any, Optional

from .conversation import ConversationMemory
from .episodic import EpisodicMemory, Episode
from .procedural import ProceduralMemory, Workflow
from .semantic import SemanticMemory, MemoryItem


class MemoryManager:
    """Unified memory manager for all memory types.

    Provides a single interface to interact with all memory systems
    and enables cross-memory operations like consolidation.
    """

    def __init__(
        self,
        conversation_memory: Optional[ConversationMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        procedural_memory: Optional[ProceduralMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
    ):
        """Initialize memory manager.

        Args:
            conversation_memory: Short-term conversation memory
            semantic_memory: Long-term semantic memory
            procedural_memory: Procedural/skill memory
            episodic_memory: Episodic event memory
        """
        self.conversation = conversation_memory or ConversationMemory()
        self.semantic = semantic_memory
        self.procedural = procedural_memory or ProceduralMemory()
        self.episodic = episodic_memory

    async def remember(self, query: str) -> dict[str, Any]:
        """Recall from all memory types.

        Args:
            query: Search query

        Returns:
            Dictionary with results from each memory type
        """
        result = {
            "short_term": await self._get_short_term_summary(),
            "long_term": [],
            "episodes": [],
            "workflows": [],
        }

        if self.semantic:
            result["long_term"] = await self.semantic.recall(query)

        if self.episodic:
            result["episodes"] = await self.episodic.recall(query)

        if self.procedural:
            workflow = await self.procedural.recall(query)
            if workflow:
                result["workflows"] = [workflow]

        return result

    async def _get_short_term_summary(self) -> dict[str, Any]:
        """Get summary of short-term memory."""
        messages = self.conversation.get_all_messages()
        return {
            "message_count": len(messages),
            "recent_messages": messages[-5:] if messages else [],
        }

    async def consolidate(self) -> None:
        """Consolidate short-term memory into long-term memory.

        This extracts important information from conversation history
        and stores it in semantic/episodic memory.
        """
        if not self.conversation or not self.semantic:
            return

        # Get recent messages
        messages = self.conversation.get_all_messages()
        if not messages:
            return

        # This is a simplified consolidation
        # In production, you'd use an LLM to extract important info
        last_user_msg = None
        for msg in reversed(messages):
            if hasattr(msg, 'role') and msg.role.value == "user":
                last_user_msg = msg
                break

        if last_user_msg and self.episodic:
            # Create an episode
            content = str(last_user_msg.content) if hasattr(last_user_msg, 'content') else ""
            episode = Episode(
                id="",
                summary=content[:100],
                details=content,
                participants=["user", "assistant"],
            )
            await self.episodic.record(episode)

    async def build_context(self, query: str, max_tokens: int = 2000) -> str:
        """Build a context string from all memory types.

        Args:
            query: Query for retrieving relevant memories
            max_tokens: Approximate maximum tokens

        Returns:
            Context string
        """
        memories = await self.remember(query)
        context_parts = []

        # Add long-term memories
        if memories["long_term"]:
            context_parts.append("## Relevant Facts and Preferences")
            for item in memories["long_term"][:3]:
                context_parts.append(f"- {item.key}: {item.value}")

        # Add recent episodes
        if memories["episodes"]:
            context_parts.append("\n## Recent Experiences")
            for episode in memories["episodes"][:2]:
                context_parts.append(f"- {episode.summary}")

        # Add relevant workflows
        if memories["workflows"]:
            context_parts.append("\n## Relevant Skills")
            for workflow in memories["workflows"]:
                context_parts.append(f"- {workflow.name}: {workflow.description}")

        # Add recent conversation
        short_term = memories["short_term"]
        if short_term["recent_messages"]:
            context_parts.append("\n## Recent Conversation")
            for msg in short_term["recent_messages"][-3:]:
                content = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg)[:100]
                context_parts.append(f"- {content}")

        return "\n".join(context_parts)

    async def store_experience(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Store a new experience.

        Automatically determines the appropriate memory type.

        Args:
            content: Content to store
            metadata: Additional metadata
        """
        metadata = metadata or {}

        # Determine memory type from metadata
        memory_type = metadata.get("type", "episode")

        if memory_type == "fact" and self.semantic:
            await self.semantic.store(
                key=metadata.get("key", "unknown"),
                value=content,
                category=metadata.get("category", "general"),
                importance=metadata.get("importance", 0.5),
            )

        elif memory_type == "workflow" and self.procedural:
            workflow = Workflow(
                id="",
                name=metadata.get("name", "Unknown Workflow"),
                description=content,
                tags=metadata.get("tags", []),
            )
            await self.procedural.learn(workflow)

        elif self.episodic:
            episode = Episode(
                id="",
                summary=content[:100],
                details=content,
                tags=metadata.get("tags", []),
            )
            await self.episodic.record(episode)

    async def store_fact(
        self,
        key: str,
        value: str,
        category: str = "general",
        importance: float = 0.5,
    ) -> Optional[str]:
        """Store a fact in semantic memory.

        Args:
            key: Fact key
            value: Fact value
            category: Category
            importance: Importance score

        Returns:
            Memory item ID or None
        """
        if not self.semantic:
            return None
        return await self.semantic.store(key, value, category, importance)

    async def store_workflow(self, workflow: Workflow) -> Optional[str]:
        """Store a workflow in procedural memory.

        Args:
            workflow: Workflow to store

        Returns:
            Workflow ID or None
        """
        if not self.procedural:
            return None
        return await self.procedural.learn(workflow)

    async def store_episode(self, episode: Episode) -> Optional[str]:
        """Store an episode in episodic memory.

        Args:
            episode: Episode to store

        Returns:
            Episode ID or None
        """
        if not self.episodic:
            return None
        return await self.episodic.record(episode)

    async def clear_all(self) -> None:
        """Clear all memory types."""
        if self.conversation:
            await self.conversation.clear()
        if self.semantic:
            await self.semantic.clear()
        if self.procedural:
            await self.procedural.clear()
        if self.episodic:
            await self.episodic.clear()

    def get_stats(self) -> dict[str, int]:
        """Get statistics about stored memories.

        Returns:
            Dictionary with counts for each memory type
        """
        return {
            "conversation_entries": len(self.conversation._entries) if self.conversation else 0,
            "semantic_items": self.semantic.count() if self.semantic else 0,
            "workflows": self.procedural.count() if self.procedural else 0,
            "episodes": self.episodic.count() if self.episodic else 0,
        }
