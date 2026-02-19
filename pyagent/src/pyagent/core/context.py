"""
Context management for agent conversations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from pyagent.core.message import Message

if TYPE_CHECKING:
    from pyagent.core.tools import Tool
    from pyagent.skills.skill import Skill


class Context(BaseModel):
    """
    Agent context - holds conversation history and state.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[Message] = Field(default_factory=list)
    tools: dict[str, Any] = Field(default_factory=dict)
    skills: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_messages: int = 100

    def add_message(self, message: Message) -> None:
        """Add a message to the context."""
        self.messages.append(message)
        self._trim_if_needed()

    def add_system_message(self, content: str) -> None:
        """Add a system message."""
        self.add_message(Message.system(content))

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message(Message.user(content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self.add_message(Message.assistant(content))

    def get_messages(self) -> list[Message]:
        """Get all messages."""
        return self.messages.copy()

    def get_last_n_messages(self, n: int) -> list[Message]:
        """Get the last n messages."""
        return self.messages[-n:] if n > 0 else []

    def clear_messages(self) -> None:
        """Clear all messages."""
        self.messages.clear()

    def register_tool(self, tool: "Tool") -> None:
        """Register a tool in the context."""
        self.tools[tool.name] = tool

    def unregister_tool(self, name: str) -> "Tool | None":
        """Unregister a tool from the context."""
        return self.tools.pop(name, None)

    def get_tool(self, name: str) -> "Tool | None":
        """Get a tool by name."""
        return self.tools.get(name)

    def register_skill(self, skill: "Skill") -> None:
        """Register a skill in the context."""
        self.skills[skill.name] = skill

    def unregister_skill(self, name: str) -> "Skill | None":
        """Unregister a skill from the context."""
        return self.skills.pop(name, None)

    def get_skill(self, name: str) -> "Skill | None":
        """Get a skill by name."""
        return self.skills.get(name)

    def find_matching_skills(self, query: str) -> list["Skill"]:
        """Find skills that match a query."""
        query_lower = query.lower()
        matches = []
        for skill in self.skills.values():
            if (query_lower in skill.name.lower() or
                query_lower in skill.description.lower()):
                matches.append(skill)
        return matches

    def _trim_if_needed(self) -> None:
        """Trim messages if exceeding max_messages."""
        if len(self.messages) > self.max_messages:
            # Keep system messages and trim from the middle
            system_messages = [m for m in self.messages if m.role.value == "system"]
            other_messages = [m for m in self.messages if m.role.value != "system"]

            # Keep last N non-system messages
            keep_count = self.max_messages - len(system_messages)
            other_messages = other_messages[-keep_count:]

            self.messages = system_messages + other_messages

    def get_api_messages(self) -> list[dict[str, Any]]:
        """Get messages in API format."""
        return [msg.to_api_format() for msg in self.messages]

    def get_tools_api_format(self) -> list[dict[str, Any]]:
        """Get tools in API format."""
        return [tool.to_api_format() for tool in self.tools.values()]
