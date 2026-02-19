# PyAgent

A lightweight, modular AI Agent framework with native support for **MCP (Model Context Protocol)** and **Skills** system.

## Features

- **Core Agent Framework**: Flexible agent architecture with message handling and tool execution
- **MCP Support**: Full implementation of Model Context Protocol for tool/resource/prompt interoperability
- **Skills System**: Load and execute skills following the [Agent Skills specification](https://agentskills.io/specification)
- **Multiple LLM Providers**: Built-in support for OpenAI and Anthropic APIs
- **Memory System**: Conversation and context management
- **Async-First**: Fully asynchronous design for high-performance applications

## Installation

```bash
# Basic installation
pip install pyagent

# With OpenAI support
pip install pyagent[openai]

# With Anthropic support
pip install pyagent[anthropic]

# With all features
pip install pyagent[all]
```

## Quick Start

### Basic Agent

```python
import asyncio
from pyagent import Agent
from pyagent.core.agent import AgentConfig
from pyagent.core.tools import tool
from pyagent.providers.openai import OpenAIProvider

# Define tools
@tool(description="Get weather for a city")
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 25°C"

async def main():
    # Create provider and agent
    provider = OpenAIProvider(api_key="your-api-key")
    config = AgentConfig(
        name="MyAgent",
        model="gpt-4",
        system_prompt="You are a helpful assistant."
    )

    agent = Agent(
        provider=provider,
        config=config,
        tools=[get_weather]
    )

    # Run the agent
    response = await agent.run("What's the weather in Beijing?")
    print(response)

asyncio.run(main())
```

### Using MCP

```python
from pyagent.mcp.client import MCPClient
from pyagent.mcp.server import MCPServer

# Create an MCP server
server = MCPServer(name="my-server")

@server.tool(description="Add two numbers")
def add(a: int, b: int) -> str:
    return f"Result: {a + b}"

# Handle MCP requests
result = await server.handle_request("tools/call", {
    "name": "add",
    "arguments": {"a": 5, "b": 3}
})
```

### Using Skills

```python
from pyagent.skills.loader import SkillLoader
from pyagent.skills.registry import SkillRegistry

# Load skills
loader = SkillLoader()
skill = loader.load(Path("./my-skill"))

# Register in agent
registry = SkillRegistry()
registry.register(skill)

# Use with agent
agent = Agent(provider=provider, skills=[skill])
```

## Project Structure

```
pyagent/
├── src/pyagent/
│   ├── core/           # Core agent components
│   │   ├── agent.py    # Main Agent class
│   │   ├── message.py  # Message types
│   │   ├── context.py  # Context management
│   │   └── tools.py    # Tool definitions
│   ├── mcp/            # MCP protocol support
│   │   ├── client.py   # MCP client
│   │   ├── server.py   # MCP server
│   │   └── transport/  # Transport layers
│   ├── skills/         # Skills system
│   │   ├── loader.py   # Skill loader
│   │   ├── registry.py # Skill registry
│   │   └── executor.py # Skill executor
│   ├── providers/      # LLM providers
│   │   ├── openai.py   # OpenAI provider
│   │   └── anthropic.py# Anthropic provider
│   └── memory/         # Memory systems
├── tests/              # Unit tests
├── examples/           # Example code
└── docs/               # Documentation
```

## Skill Format

Skills follow the [Agent Skills specification](https://agentskills.io/specification):

```
my-skill/
├── SKILL.md           # Required: Main skill file
├── scripts/           # Optional: Executable scripts
├── references/        # Optional: Reference docs
└── assets/            # Optional: Static assets
```

SKILL.md format:

```markdown
---
name: my-skill
description: What this skill does and when to use it
license: MIT
metadata:
  author: your-name
  version: "1.0"
  tags:
    - category
allowed-tools: Read Grep
---

# Skill Instructions

Your skill instructions here...
```

## Configuration

Create a `pyagent.yaml` file:

```yaml
name: MyAgent
model: gpt-4
temperature: 0.7
max_tokens: 4096

provider: openai
api_key: ${OPENAI_API_KEY}

skills_dir: ./skills
auto_discover_skills: true

mcp_servers:
  - command: python
    args: ["-m", "my_mcp_server"]
```

## API Reference

### Agent

```python
class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        config: AgentConfig | None = None,
        tools: list[Tool] | None = None,
        skills: list[Skill] | None = None,
        mcp_clients: list[MCPClient] | None = None
    ): ...

    async def run(self, message: str) -> str: ...
    async def run_stream(self, message: str) -> AsyncIterator[str]: ...
    def add_tool(self, tool: Tool) -> None: ...
    def add_skill(self, skill: Skill) -> None: ...
```

### Tool

```python
@tool(description="Tool description")
def my_tool(arg: str) -> str:
    return f"Result: {arg}"
```

### MCP Client

```python
async with MCPClient.from_stdio("python", ["-m", "server"]) as client:
    tools = await client.list_tools()
    result = await client.call_tool("tool_name", {"arg": "value"})
```

## Development

```bash
# Install dev dependencies
pip install pyagent[dev]

# Run tests
pytest

# Run linting
ruff check src/

# Type checking
mypy src/
```

## License

MIT License
