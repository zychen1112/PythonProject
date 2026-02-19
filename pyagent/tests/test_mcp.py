"""
Tests for MCP functionality.
"""

import pytest

from pyagent.mcp.server import MCPServer
from pyagent.mcp.types import (
    ToolDefinition,
    ResourceDefinition,
    PromptDefinition,
)


@pytest.fixture
def mcp_server():
    """Create a test MCP server."""
    server = MCPServer(name="test-server", version="1.0.0")

    @server.tool(description="Add two numbers")
    def add(a: int, b: int) -> str:
        return f"Result: {a + b}"

    @server.tool(description="Echo a message")
    def echo(message: str) -> str:
        return f"Echo: {message}"

    @server.resource("file:///{path}", description="Read a file")
    def read_file(path: str) -> str:
        return f"Content of {path}"

    @server.prompt(description="Generate a greeting")
    def greeting(name: str) -> str:
        return f"Hello, {name}!"

    return server


class TestMCPServer:
    """Tests for MCP Server."""

    @pytest.mark.asyncio
    async def test_handle_initialize(self, mcp_server):
        result = await mcp_server.handle_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"}
        })

        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "test-server"
        assert "tools" in result["capabilities"]

    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_server):
        result = await mcp_server.handle_request("tools/list", {})

        assert "tools" in result
        assert len(result["tools"]) == 2

        tool_names = [t["name"] for t in result["tools"]]
        assert "add" in tool_names
        assert "echo" in tool_names

    @pytest.mark.asyncio
    async def test_call_tool(self, mcp_server):
        result = await mcp_server.handle_request("tools/call", {
            "name": "add",
            "arguments": {"a": 5, "b": 3}
        })

        assert "content" in result
        assert "Result: 8" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_call_echo_tool(self, mcp_server):
        result = await mcp_server.handle_request("tools/call", {
            "name": "echo",
            "arguments": {"message": "Hello MCP!"}
        })

        assert "Echo: Hello MCP!" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self, mcp_server):
        with pytest.raises(ValueError, match="Unknown tool"):
            await mcp_server.handle_request("tools/call", {
                "name": "nonexistent",
                "arguments": {}
            })

    @pytest.mark.asyncio
    async def test_list_resources(self, mcp_server):
        result = await mcp_server.handle_request("resources/list", {})

        assert "resources" in result
        assert len(result["resources"]) == 1
        assert result["resources"][0]["uri"] == "file:///{path}"

    @pytest.mark.asyncio
    async def test_list_prompts(self, mcp_server):
        result = await mcp_server.handle_request("prompts/list", {})

        assert "prompts" in result
        assert len(result["prompts"]) == 1
        assert result["prompts"][0]["name"] == "greeting"

    @pytest.mark.asyncio
    async def test_get_prompt(self, mcp_server):
        result = await mcp_server.handle_request("prompts/get", {
            "name": "greeting",
            "arguments": {"name": "World"}
        })

        assert "messages" in result
        assert "Hello, World!" in result["messages"][0]["content"]["text"]

    @pytest.mark.asyncio
    async def test_unknown_method(self, mcp_server):
        with pytest.raises(ValueError, match="Unknown method"):
            await mcp_server.handle_request("unknown/method", {})


class TestMCPTypes:
    """Tests for MCP type definitions."""

    def test_tool_definition(self):
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}}
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_resource_definition(self):
        resource = ResourceDefinition(
            uri="file:///test.txt",
            name="test_file",
            description="A test file",
            mime_type="text/plain"
        )

        assert resource.uri == "file:///test.txt"
        assert resource.mime_type == "text/plain"

    def test_prompt_definition(self):
        prompt = PromptDefinition(
            name="test_prompt",
            description="A test prompt",
            arguments=[{"name": "topic", "required": True}]
        )

        assert prompt.name == "test_prompt"
        assert len(prompt.arguments) == 1
