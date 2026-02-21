"""
MCP Client implementation with error handling and retry support.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Callable

from pyagent.mcp.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPNotConnectedError,
    MCPPromptError,
    MCPResourceError,
    MCPTimeoutError,
    MCPToolError,
)
from pyagent.mcp.transport import Transport
from pyagent.mcp.transport.stdio import StdioTransport
from pyagent.mcp.transport.sse import SSETransport
from pyagent.mcp.types import (
    ClientInfo,
    InitializeParams,
)
from pyagent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = (
        MCPConnectionError,
        MCPTimeoutError,
        ConnectionError,
        TimeoutError,
    )


class MCPClient:
    """
    Client for connecting to MCP servers with retry support.

    Features:
    - Automatic retry with exponential backoff
    - Comprehensive error handling
    - Connection state management
    - Configurable timeouts
    """

    def __init__(
        self,
        transport: Transport,
        retry_config: RetryConfig | None = None,
        timeout: float = 30.0,
    ):
        self.transport = transport
        self._initialized = False
        self._server_info: dict[str, Any] = {}
        self._capabilities: dict[str, Any] = {}
        self._retry_config = retry_config or RetryConfig()
        self._timeout = timeout

    @classmethod
    def from_stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        retry_config: RetryConfig | None = None,
        timeout: float = 30.0,
    ) -> "MCPClient":
        """Create a client that connects via stdio."""
        transport = StdioTransport(command, args, env)
        return cls(transport, retry_config, timeout)

    @classmethod
    def from_sse(
        cls,
        url: str,
        headers: dict[str, str] | None = None,
        retry_config: RetryConfig | None = None,
        timeout: float = 30.0,
    ) -> "MCPClient":
        """Create a client that connects via SSE."""
        transport = SSETransport(url, headers)
        return cls(transport, retry_config, timeout)

    async def _retry_request(
        self,
        request_func: Callable[[], Any],
        operation_name: str = "request",
    ) -> Any:
        """
        Execute a request with retry logic and exponential backoff.

        Args:
            request_func: Async function to execute
            operation_name: Name of the operation for logging

        Returns:
            Result from the request function

        Raises:
            MCPError: If all retries are exhausted
        """
        config = self._retry_config
        last_exception: Exception | None = None

        for attempt in range(config.max_retries + 1):
            try:
                return await request_func()
            except config.retryable_exceptions as e:
                last_exception = e
                if attempt < config.max_retries:
                    delay = self._calculate_delay(attempt, config)
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"{operation_name} failed after {config.max_retries + 1} attempts: {e}"
                    )
            except MCPError:
                raise
            except Exception as e:
                logger.error(f"{operation_name} failed with unexpected error: {e}")
                raise MCPError(str(e)) from e

        raise MCPError(
            f"{operation_name} failed after {config.max_retries + 1} attempts: {last_exception}"
        )

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = config.base_delay * (config.exponential_base**attempt)
        delay = min(delay, config.max_delay)
        if config.jitter:
            delay = delay * (0.5 + random.random())
        return delay

    async def connect(self) -> None:
        """Connect to the MCP server and initialize with retry support."""
        await self._retry_request(
            self._do_connect,
            "Connection",
        )

    async def _do_connect(self) -> None:
        """Internal connection logic."""
        try:
            await self.transport.connect()
            await self._initialize()
            self._initialized = True
            logger.info(f"Connected to MCP server: {self._server_info}")
        except TimeoutError as e:
            raise MCPTimeoutError(f"Connection timeout: {e}") from e
        except ConnectionError as e:
            raise MCPConnectionError(f"Connection failed: {e}") from e

    async def close(self) -> None:
        """Close the connection."""
        try:
            await self.transport.close()
        except Exception as e:
            logger.warning(f"Error during close: {e}")
        finally:
            self._initialized = False

    async def _initialize(self) -> None:
        """Initialize the MCP connection."""
        params = InitializeParams(
            protocol_version="2024-11-05",
            capabilities={
                "tools": {},
                "resources": {},
                "prompts": {},
            },
            client_info=ClientInfo(),
        )

        result = await self.transport.request("initialize", params.model_dump())

        self._server_info = result.get("serverInfo", {})
        self._capabilities = result.get("capabilities", {})

        await self.transport.send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
        )

    def _check_connected(self) -> None:
        """Check if client is connected, raise error if not."""
        if not self._initialized:
            raise MCPNotConnectedError()

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List available tools.

        Returns:
            List of tool definitions

        Raises:
            MCPNotConnectedError: If not connected
            MCPError: If request fails
        """
        self._check_connected()

        async def _request():
            result = await self.transport.request("tools/list")
            tools = result.get("tools", [])
            logger.debug(f"Listed {len(tools)} tools")
            return tools

        return await self._retry_request(_request, "List tools")

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Call a tool on the server.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            MCPNotConnectedError: If not connected
            MCPToolError: If tool execution fails
            MCPError: If request fails
        """
        self._check_connected()

        params = {
            "name": name,
            "arguments": arguments or {},
        }

        async def _request():
            try:
                result = await self.transport.request("tools/call", params)
                logger.debug(f"Called tool {name}")
                return result
            except Exception as e:
                if "unknown tool" in str(e).lower():
                    raise MCPToolError(name, f"Unknown tool: {name}") from e
                raise

        return await self._retry_request(_request, f"Call tool '{name}'")

    async def list_resources(self) -> list[dict[str, Any]]:
        """
        List available resources.

        Returns:
            List of resource definitions

        Raises:
            MCPNotConnectedError: If not connected
            MCPError: If request fails
        """
        self._check_connected()

        async def _request():
            result = await self.transport.request("resources/list")
            resources = result.get("resources", [])
            logger.debug(f"Listed {len(resources)} resources")
            return resources

        return await self._retry_request(_request, "List resources")

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """
        Read a resource by URI.

        Args:
            uri: Resource URI

        Returns:
            Resource content

        Raises:
            MCPNotConnectedError: If not connected
            MCPResourceError: If resource access fails
            MCPError: If request fails
        """
        self._check_connected()

        params = {"uri": uri}

        async def _request():
            try:
                result = await self.transport.request("resources/read", params)
                logger.debug(f"Read resource {uri}")
                return result
            except Exception as e:
                if "not found" in str(e).lower():
                    raise MCPResourceError(uri, f"Resource not found: {uri}") from e
                raise

        return await self._retry_request(_request, f"Read resource '{uri}'")

    async def list_prompts(self) -> list[dict[str, Any]]:
        """
        List available prompts.

        Returns:
            List of prompt definitions

        Raises:
            MCPNotConnectedError: If not connected
            MCPError: If request fails
        """
        self._check_connected()

        async def _request():
            result = await self.transport.request("prompts/list")
            prompts = result.get("prompts", [])
            logger.debug(f"Listed {len(prompts)} prompts")
            return prompts

        return await self._retry_request(_request, "List prompts")

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Get a prompt by name.

        Args:
            name: Prompt name
            arguments: Prompt arguments

        Returns:
            Prompt result

        Raises:
            MCPNotConnectedError: If not connected
            MCPPromptError: If prompt access fails
            MCPError: If request fails
        """
        self._check_connected()

        params = {
            "name": name,
            "arguments": arguments or {},
        }

        async def _request():
            try:
                result = await self.transport.request("prompts/get", params)
                logger.debug(f"Got prompt {name}")
                return result
            except Exception as e:
                if "unknown prompt" in str(e).lower():
                    raise MCPPromptError(name, f"Unknown prompt: {name}") from e
                raise

        return await self._retry_request(_request, f"Get prompt '{name}'")

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
