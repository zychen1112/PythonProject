"""
MCP-specific exceptions.
"""


class MCPError(Exception):
    """Base exception for MCP-related errors."""

    def __init__(self, message: str, code: int | None = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""

    def __init__(self, message: str = "Failed to connect to MCP server"):
        super().__init__(message, code=-32001)


class MCPTimeoutError(MCPError):
    """Raised when a request times out."""

    def __init__(self, message: str = "Request timed out"):
        super().__init__(message, code=-32002)


class MCPNotConnectedError(MCPError):
    """Raised when trying to use client without connection."""

    def __init__(self, message: str = "Not connected to MCP server"):
        super().__init__(message, code=-32003)


class MCPToolError(MCPError):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}", code=-32004)


class MCPResourceError(MCPError):
    """Raised when resource access fails."""

    def __init__(self, uri: str, message: str):
        self.uri = uri
        super().__init__(f"Resource '{uri}' error: {message}", code=-32005)


class MCPPromptError(MCPError):
    """Raised when prompt access fails."""

    def __init__(self, name: str, message: str):
        self.name = name
        super().__init__(f"Prompt '{name}' error: {message}", code=-32006)


class MCPProtocolError(MCPError):
    """Raised when there's a protocol-level error."""

    def __init__(self, message: str, code: int = -32600):
        super().__init__(f"Protocol error: {message}", code=code)
