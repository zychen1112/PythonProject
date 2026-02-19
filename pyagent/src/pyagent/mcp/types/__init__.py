"""
MCP Type definitions.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """Definition of an MCP tool."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "required": []
    })


class ResourceDefinition(BaseModel):
    """Definition of an MCP resource."""
    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


class PromptDefinition(BaseModel):
    """Definition of an MCP prompt."""
    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = Field(default_factory=list)


class ResourceContent(BaseModel):
    """Content of a resource."""
    uri: str
    mime_type: str = "text/plain"
    text: str | None = None
    blob: bytes | None = None


class PromptMessage(BaseModel):
    """A message in a prompt."""
    role: Literal["user", "assistant"]
    content: dict[str, Any]


class PromptResult(BaseModel):
    """Result of getting a prompt."""
    description: str = ""
    messages: list[PromptMessage] = Field(default_factory=list)


class ToolResult(BaseModel):
    """Result of calling a tool."""
    content: list[dict[str, Any]] = Field(default_factory=list)
    is_error: bool = False


class ServerInfo(BaseModel):
    """Information about an MCP server."""
    name: str
    version: str
    protocol_version: str = "2024-11-05"


class ClientInfo(BaseModel):
    """Information about an MCP client."""
    name: str = "pyagent"
    version: str = "0.1.0"


class InitializeParams(BaseModel):
    """Parameters for initialization."""
    protocol_version: str = "2024-11-05"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    client_info: ClientInfo = Field(default_factory=ClientInfo)


class InitializeResult(BaseModel):
    """Result of initialization."""
    protocol_version: str
    capabilities: dict[str, Any]
    server_info: ServerInfo
