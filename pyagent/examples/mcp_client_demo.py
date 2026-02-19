"""
MCP Client example showing how to connect to MCP servers.
"""

import asyncio
from pyagent.mcp.client import MCPClient
from pyagent.mcp.server import MCPServer


# Create a simple MCP server
server = MCPServer(
    name="demo-server",
    version="1.0.0"
)


@server.tool(description="Echo back the input message")
def echo(message: str) -> str:
    """Echo the message back."""
    return f"Echo: {message}"


@server.tool(description="Add two numbers together")
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return f"Result: {a + b}"


@server.resource("config:///{name}")
def get_config(name: str) -> str:
    """Get a configuration value."""
    configs = {
        "app_name": "PyAgent Demo",
        "version": "1.0.0",
        "debug": "false"
    }
    return configs.get(name, f"Config '{name}' not found")


@server.prompt(description="Generate a greeting prompt")
def greeting_prompt(name: str) -> str:
    """Generate a greeting."""
    return f"Please greet {name} in a friendly manner."


async def demo_server():
    """Demo the server's request handling."""
    print("=== MCP Server Demo ===\n")

    # Initialize
    result = await server.handle_request("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "demo-client", "version": "1.0"}
    })
    print(f"Initialize result: {result}\n")

    # List tools
    tools = await server.handle_request("tools/list", {})
    print(f"Available tools: {tools}\n")

    # Call a tool
    result = await server.handle_request("tools/call", {
        "name": "echo",
        "arguments": {"message": "Hello, MCP!"}
    })
    print(f"Echo result: {result}\n")

    # Call add tool
    result = await server.handle_request("tools/call", {
        "name": "add",
        "arguments": {"a": 10, "b": 32}
    })
    print(f"Add result: {result}\n")

    # List resources
    resources = await server.handle_request("resources/list", {})
    print(f"Available resources: {resources}\n")

    # Read a resource
    resource = await server.handle_request("resources/read", {
        "uri": "config:///app_name"
    })
    print(f"Config resource: {resource}\n")

    # List prompts
    prompts = await server.handle_request("prompts/list", {})
    print(f"Available prompts: {prompts}\n")

    # Get a prompt
    prompt = await server.handle_request("prompts/get", {
        "name": "greeting_prompt",
        "arguments": {"name": "World"}
    })
    print(f"Greeting prompt: {prompt}\n")


async def main():
    await demo_server()

    # Example of connecting to an external MCP server via stdio
    # (uncomment and modify as needed)
    """
    async with MCPClient.from_stdio(
        command="python",
        args=["-m", "some_mcp_server"]
    ) as client:
        tools = await client.list_tools()
        print(f"Tools: {tools}")

        result = await client.call_tool("some_tool", {"arg": "value"})
        print(f"Result: {result}")
    """


if __name__ == "__main__":
    asyncio.run(main())
