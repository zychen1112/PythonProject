"""Hook execution context."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class HookContext:
    """Hook execution context containing all relevant data.

    This context is passed to each hook during execution and provides
    access to the current state of the agent execution.

    Attributes:
        agent_id: Unique identifier for the agent instance
        position: The hook position where this context was created
        timestamp: When this context was created

        message: The current user message (if applicable)
        messages: List of all messages in the conversation

        tool_name: Name of the tool being called (if applicable)
        tool_arguments: Arguments passed to the tool (if applicable)
        tool_result: Result from tool execution (if applicable)

        llm_response: Response from LLM (if applicable)

        error: Exception that occurred (if applicable)

        iteration: Current iteration number

        metadata: Additional metadata
        modifications: Data modifications to be applied
    """

    # Required fields
    agent_id: str
    position: "HookPosition"  # type: ignore

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    # Message related
    message: Optional[str] = None
    messages: list[Any] = field(default_factory=list)

    # Tool related
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict[str, Any]] = None
    tool_result: Optional[Any] = None

    # LLM related
    llm_response: Optional[Any] = None

    # Error related
    error: Optional[Exception] = None

    # Iteration related
    iteration: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Modifications that can be applied
    modifications: dict[str, Any] = field(default_factory=dict)

    def set_modification(self, key: str, value: Any) -> None:
        """Set a modification to be applied.

        Args:
            key: The key to modify
            value: The new value
        """
        self.modifications[key] = value

    def get_modification(self, key: str, default: Any = None) -> Any:
        """Get a modification value.

        Args:
            key: The key to look up
            default: Default value if key not found

        Returns:
            The modification value or default
        """
        return self.modifications.get(key, default)

    def clear_modifications(self) -> None:
        """Clear all modifications."""
        self.modifications.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert context to a dictionary for serialization.

        Returns:
            Dictionary representation of the context
        """
        return {
            "agent_id": self.agent_id,
            "position": self.position.value if hasattr(self.position, "value") else str(self.position),
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "iteration": self.iteration,
            "metadata": self.metadata,
            "modifications": self.modifications,
            "error": str(self.error) if self.error else None,
        }
