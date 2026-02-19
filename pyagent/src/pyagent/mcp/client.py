"""
MCP Client implementation.
"""

from typing import Any

from pyagent.mcp.transport import Transport
from pyagent.mcp.transport.stdio import StdioTransport
from pyagent.mcp.transport.sse import SSETransport
from pyagent.mcp.types import (
    InitializeParams,
    ClientInfo,
    ToolDefinition,
    ResourceDefinition,
    PromptDefinition,
    ResourceContent,
    PromptResult,
    ToolResult,
)
from pyagent.utils.logging import get_logger

logger = get_logger(__name__)


class MCPClient:
    """
    Client for connecting to MCP servers.
    """

    def __init__(self, transport: Transport):
        self.transport = transport
        self._initialized = False
        self._server_info: dict[str, Any] = {}
        self._capabilities: dict[str, Any] = {}

    @classmethod
    def from_stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None
    ) -> "MCPClient":
        """Create a client that connects via stdio."""
        transport = StdioTransport(command, args, env)
        return cls(transport)

    @classmethod
    def from_sse(
        cls,
        url: str,
        headers: dict[str, str] | None = None
    ) -> "MCPClient":
        """Create a client that connects via SSE."""
        transport = SSETransport(url, headers)
        return cls(transport)

    async def connect(self) -> None:
        """Connect to the MCP server and initialize."""
        await self.transport.connect()
        await self._initialize()
        self._initialized = True
        logger.info(f"Connected to MCP server: {self._server_info}")

    async def close(self) -> None:
        """Close the connection."""
        await self.transport.close()
        self._initialized = False

    async def _initialize(self) -> None:
        """Initialize the MCP connection."""
        params = InitializeParams(
            protocol_version="2024-11-05",
            capabilities={
                "tools": {},
                "resources": {},
                "prompts": {}
            },
            client_info=ClientInfo()
        )

        result = await self.transport.request("initialize", params.model_dump())

        self._server_info = result.get("serverInfo", {})
        self._capabilities = result.get("capabilities", {})

        # Send initialized notification
        await self.transport.send({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        })

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools."""
        if not self._initialized:
            raise RuntimeError("Not connected")

        result = await self.transport.request("tools/list")
        tools = result.get("tools", [])

        logger.debug(f"Listed {len(tools)} tools")
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call a tool on the server."""
        if not self._initialized:
            raise RuntimeError("Not connected")

        params = {
            "name": name,
            "arguments": arguments or {}
        }

        result = await self.transport.request("tools/call", params)
        logger.debug(f"Called tool {name}")
        return result

    async def list_resources(self) -> list[dict[str, Any]]:
        """List available resources."""
        if not self._initialized:
            raise RuntimeError("Not connected")

        result = await self.transport.request("resources/list")
        resources = result.get("resources", [])

        logger.debug(f"Listed {len(resources)} resources")
        return resources

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Read a resource by URI."""
        if not self._initialized:
            raise RuntimeError("Not connected")

        params = {"uri": uri}
        result = await self.transport.request("resources/read", params)

        logger.debug(f"Read resource {uri}")
        return result

    async def list_prompts(self) -> list[dict[str, Any]]:
        """List available prompts."""
        if not self._initialized:
            raise RuntimeError("Not connected")

        result = await self.transport.request("prompts/list")
        prompts = result.get("prompts", [])

        logger.debug(f"Listed {len(prompts)} prompts")
        return prompts

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Get a prompt by name."""
        if not self._initialized:
            raise RuntimeError("Not connected")

        params = {
            "name": name,
            "arguments": arguments or {}
        }

        result = await self.transport.request("prompts/get", params)
        logger.debug(f"Got prompt {name}")
        return result

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
