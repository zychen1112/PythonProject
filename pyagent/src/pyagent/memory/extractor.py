"""Memory Extractor - Extract memories from conversations."""

import json
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from .episodic import Episode
from .procedural import Workflow, WorkflowStep

if TYPE_CHECKING:
    from pyagent.providers.base import LLMProvider


class ExtractedMemory(BaseModel):
    """A memory extracted from conversation."""
    type: str  # "fact", "preference", "skill", "episode"
    content: str
    importance: float = 0.5
    metadata: dict[str, Any] = {}


class MemoryExtractor:
    """Extract important information from conversations.

    Uses an LLM to analyze conversations and extract:
    - Facts (user-provided information)
    - Preferences (likes/dislikes)
    - Skills/Procedures (how to do things)
    - Notable events
    """

    EXTRACT_PROMPT = """Analyze the following conversation and extract important information that should be remembered for future interactions.

Conversation:
{conversation}

Extract the following types of information:
1. Facts: Important information the user shared (e.g., "user's name is John", "user works at Acme Corp")
2. Preferences: User's likes, dislikes, or preferences (e.g., "user prefers dark mode", "user dislikes long explanations")
3. Skills: Procedures or methods discussed (e.g., "how to reset the system", "the deployment process")
4. Events: Notable events or experiences (e.g., "user had an issue with login", "successful deployment")

Return a JSON array with objects containing:
- type: "fact", "preference", "skill", or "episode"
- content: The information to remember
- importance: A score from 0 to 1 indicating importance (0.1 = minor, 1.0 = critical)

Example output:
[
  {{"type": "fact", "content": "User's name is John", "importance": 0.8}},
  {{"type": "preference", "content": "User prefers concise responses", "importance": 0.6}}
]

Return ONLY the JSON array, no additional text."""

    def __init__(self, llm_provider: "LLMProvider"):
        """Initialize memory extractor.

        Args:
            llm_provider: LLM provider for extraction
        """
        self.llm_provider = llm_provider

    async def extract(self, messages: list) -> list[ExtractedMemory]:
        """Extract memories from messages.

        Args:
            messages: List of conversation messages

        Returns:
            List of extracted memories
        """
        if not messages:
            return []

        # Format conversation
        conversation = self._format_conversation(messages)

        try:
            response = await self.llm_provider.complete(
                messages=[{"role": "user", "content": self.EXTRACT_PROMPT.format(conversation=conversation)}],
                model="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=1000,
            )

            if response and "message" in response:
                content = response["message"].content
                if isinstance(content, str):
                    # Parse JSON response
                    # Remove markdown code blocks if present
                    content = content.strip()
                    if content.startswith("```"):
                        content = content.split("\n", 1)[1]
                    if content.endswith("```"):
                        content = content.rsplit("```", 1)[0]

                    data = json.loads(content)
                    return [ExtractedMemory(**item) for item in data]

        except json.JSONDecodeError:
            pass
        except Exception:
            pass

        return []

    async def extract_facts(self, content: str) -> list[str]:
        """Extract facts from content.

        Args:
            content: Content to analyze

        Returns:
            List of extracted facts
        """
        memories = await self.extract([{"role": "user", "content": content}])
        return [m.content for m in memories if m.type == "fact"]

    async def extract_preferences(self, content: str) -> dict[str, str]:
        """Extract preferences from content.

        Args:
            content: Content to analyze

        Returns:
            Dictionary of preferences
        """
        memories = await self.extract([{"role": "user", "content": content}])
        preferences = {}
        for m in memories:
            if m.type == "preference":
                # Simple key-value extraction
                parts = m.content.split(":", 1)
                if len(parts) == 2:
                    preferences[parts[0].strip()] = parts[1].strip()
                else:
                    preferences[f"pref_{len(preferences)}"] = m.content
        return preferences

    async def extract_workflow(self, content: str) -> Optional[Workflow]:
        """Extract a workflow from content.

        Args:
            content: Content describing a procedure

        Returns:
            Extracted workflow or None
        """
        WORKFLOW_PROMPT = """Analyze the following content and extract a workflow/procedure if one is described.

Content:
{content}

If a workflow is described, return a JSON object with:
- name: Short name for the workflow
- description: Brief description
- steps: Array of steps, each with "name", "action", and optionally "tool"

If no workflow is found, return null.

Example:
{{
  "name": "Deploy Application",
  "description": "Steps to deploy the application",
  "steps": [
    {{"name": "Build", "action": "Run build command", "tool": "bash"}},
    {{"name": "Test", "action": "Run test suite", "tool": "pytest"}}
  ]
}}"""

        try:
            response = await self.llm_provider.complete(
                messages=[{"role": "user", "content": WORKFLOW_PROMPT.format(content=content)}],
                model="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=500,
            )

            if response and "message" in response:
                resp_content = response["message"].content
                if isinstance(resp_content, str):
                    resp_content = resp_content.strip()
                    if resp_content.lower() == "null":
                        return None

                    if resp_content.startswith("```"):
                        resp_content = resp_content.split("\n", 1)[1]
                    if resp_content.endswith("```"):
                        resp_content = resp_content.rsplit("```", 1)[0]

                    data = json.loads(resp_content)
                    steps = [WorkflowStep(**s) for s in data.get("steps", [])]
                    return Workflow(
                        id="",
                        name=data.get("name", "Unknown"),
                        description=data.get("description", ""),
                        steps=steps,
                    )

        except Exception:
            pass

        return None

    def _format_conversation(self, messages: list) -> str:
        """Format messages into a conversation string.

        Args:
            messages: List of messages

        Returns:
            Formatted conversation string
        """
        lines = []
        for msg in messages:
            if hasattr(msg, 'role'):
                role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            else:
                role = "unknown"

            if hasattr(msg, 'content'):
                content = str(msg.content)
            else:
                content = str(msg)

            lines.append(f"{role.upper()}: {content}")

        return "\n".join(lines)
