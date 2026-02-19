"""
Message types for agent communication.
"""

from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TextContent(BaseModel):
    """Text content block."""
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content block."""
    type: Literal["image"] = "image"
    source: dict[str, Any] = Field(
        default_factory=dict,
        description="Image source with type (base64/url) and data"
    )


class ToolUseContent(BaseModel):
    """Tool use content block."""
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultContent(BaseModel):
    """Tool result content block."""
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Union[TextContent, ImageContent, ToolUseContent, ToolResultContent]


class Message(BaseModel):
    """A message in the conversation."""
    role: Role
    content: Union[str, list[ContentBlock]]
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str | list[ContentBlock]) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str | list[ContentBlock]) -> "Message":
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def tool_result(
        cls,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
        name: str | None = None
    ) -> "Message":
        """Create a tool result message."""
        return cls(
            role=Role.TOOL,
            content=[
                ToolResultContent(
                    tool_use_id=tool_use_id,
                    content=content,
                    is_error=is_error
                )
            ],
            name=name
        )

    def to_api_format(self) -> dict[str, Any]:
        """Convert to API-compatible format."""
        if isinstance(self.content, str):
            return {"role": self.role.value, "content": self.content}
        else:
            blocks = []
            for block in self.content:
                if isinstance(block, TextContent):
                    blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageContent):
                    blocks.append({"type": "image", "source": block.source})
                elif isinstance(block, ToolUseContent):
                    blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
                elif isinstance(block, ToolResultContent):
                    blocks.append({
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": block.is_error
                    })
            result: dict[str, Any] = {"role": self.role.value, "content": blocks}
            if self.name:
                result["name"] = self.name
            return result
