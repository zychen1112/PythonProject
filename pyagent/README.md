# PyAgent

A lightweight, modular AI Agent framework with native support for **MCP (Model Context Protocol)**, **Skills**, **RAG**, **Hooks**, and advanced **Memory** systems.

## Features

### Core Features
- **Core Agent Framework**: Flexible agent architecture with message handling and tool execution
- **MCP Support**: Full implementation of Model Context Protocol for tool/resource/prompt interoperability
- **Skills System**: Load and execute skills following the [Agent Skills specification](https://agentskills.io/specification)
- **Multiple LLM Providers**: Built-in support for OpenAI, Anthropic, and Zhipu APIs
- **Async-First**: Fully asynchronous design for high-performance applications

### Advanced Features
- **Hooks & Lifecycle**: Extensible hook system with 11 lifecycle positions and 5 built-in hooks
- **RAG System**: Complete Retrieval-Augmented Generation with embeddings, vector stores, and reranking
- **Enhanced Memory**: Multi-layer memory system (semantic, procedural, episodic)
- **State Persistence**: Checkpoint-based state management with SQLite and Redis backends

## Installation

```bash
# Basic installation
pip install pyagent

# With OpenAI support
pip install pyagent[openai]

# With Anthropic support
pip install pyagent[anthropic]

# With vector store support (for RAG)
pip install pyagent[vector]

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

---

## Architecture

```
pyagent/
├── src/pyagent/
│   ├── core/              # Core agent components
│   │   ├── agent.py       # Main Agent class
│   │   ├── message.py     # Message types
│   │   ├── context.py     # Context management
│   │   ├── tools.py       # Tool definitions
│   │   └── executor.py    # Tool executor
│   │
│   ├── hooks/             # Hooks & Lifecycle System
│   │   ├── base.py        # HookPosition, HookAction, BaseHook
│   │   ├── context.py     # HookContext
│   │   ├── result.py      # HookResult
│   │   ├── registry.py    # HookRegistry
│   │   ├── executor.py    # HookExecutor
│   │   └── builtin.py     # Built-in hooks
│   │
│   ├── rag/               # RAG System
│   │   ├── base.py        # Base interfaces
│   │   ├── document.py    # Document, Chunk, SearchResult
│   │   ├── embeddings.py  # Embedding providers
│   │   ├── vectorstore.py # Vector stores
│   │   ├── chunking.py    # Document chunking
│   │   ├── retriever.py   # Retrieval strategies
│   │   ├── reranker.py    # Reranking mechanisms
│   │   └── pipeline.py    # RAG Pipeline
│   │
│   ├── memory/            # Memory Systems
│   │   ├── base.py        # Memory base class
│   │   ├── conversation.py# Short-term memory
│   │   ├── semantic.py    # Long-term semantic memory
│   │   ├── procedural.py  # Skill/workflow memory
│   │   ├── episodic.py    # Event memory
│   │   ├── manager.py     # Unified memory manager
│   │   └── extractor.py   # Memory extraction
│   │
│   ├── state/             # State Persistence
│   │   ├── checkpoint.py  # Checkpoint management
│   │   ├── sqlite_backend.py
│   │   ├── redis_backend.py
│   │   └── serializer.py  # State serialization
│   │
│   ├── mcp/               # MCP Protocol
│   │   ├── client.py      # MCP client
│   │   ├── server.py      # MCP server
│   │   └── transport/     # Transport layers
│   │
│   ├── skills/            # Skills System
│   │   ├── loader.py      # Skill loader
│   │   ├── registry.py    # Skill registry
│   │   └── executor.py    # Skill executor
│   │
│   └── providers/         # LLM Providers
│       ├── openai.py      # OpenAI provider
│       ├── anthropic.py   # Anthropic provider
│       └── zhipu.py       # Zhipu AI provider
│
├── tests/                 # Unit tests
├── examples/              # Example code
└── docs/                  # Documentation
```

---

## Module Documentation

### 1. Hooks & Lifecycle System

The hooks system allows you to inject custom logic at various points in the agent execution lifecycle.

#### Hook Positions

| Position | Description |
|----------|-------------|
| `ON_INIT` | Agent initialization |
| `ON_RUN_START` | Run method start |
| `ON_RUN_END` | Run method end |
| `ON_ITERATION_START` | Each iteration start |
| `ON_ITERATION_END` | Each iteration end |
| `ON_LLM_CALL` | Before LLM API call |
| `ON_LLM_RESPONSE` | After LLM response |
| `ON_TOOL_CALL` | Before tool execution |
| `ON_TOOL_RESULT` | After tool execution |
| `ON_MESSAGE` | When message is added |
| `ON_ERROR` | When error occurs |

#### Built-in Hooks

- **LoggingHook**: Detailed execution logging
- **TimingHook**: Execution timing with slow operation warnings
- **ErrorHandlingHook**: Automatic retry with exponential backoff
- **MetricsHook**: Prometheus-style metrics collection
- **RateLimitHook**: Sliding window rate limiting

#### Usage

```python
from pyagent import Agent, HookPosition, HookResult, LoggingHook, TimingHook

# Using built-in hooks
agent = Agent(provider=provider)
agent.register_hook(LoggingHook(level="INFO"))
agent.register_hook(TimingHook(warn_threshold_ms=5000))

# Custom hook using decorator
@agent.hook(HookPosition.ON_TOOL_CALL)
async def log_tool_usage(ctx):
    print(f"Tool called: {ctx.tool_name}")
    return HookResult.continue_()

# Hook that can abort execution
@agent.hook(HookPosition.ON_LLM_CALL, priority=0)
async def check_rate_limit(ctx):
    if is_rate_limited():
        return HookResult.abort("Rate limit exceeded")
    return HookResult.continue_()
```

---

### 2. RAG System

Complete Retrieval-Augmented Generation system with embeddings, vector stores, and retrieval strategies.

#### Embedding Providers

```python
from pyagent.rag import OpenAIEmbedding, LocalEmbedding, FakeEmbedding

# OpenAI embeddings
embedding = OpenAIEmbedding(model="text-embedding-3-small")

# Local embeddings (sentence-transformers)
embedding = LocalEmbedding(model_name="all-MiniLM-L6-v2")

# Fake embeddings (for testing)
embedding = FakeEmbedding(dimension=384)
```

#### Vector Stores

```python
from pyagent.rag import MemoryVectorStore, ChromaVectorStore

# In-memory store (testing)
store = MemoryVectorStore()

# ChromaDB (persistent)
store = ChromaVectorStore(
    collection_name="my_docs",
    persist_directory="./chroma_db"
)
```

#### Document Chunking

```python
from pyagent.rag import FixedSizeChunker, RecursiveChunker, SemanticChunker

# Fixed size with overlap
chunker = FixedSizeChunker(chunk_size=500, overlap=50)

# Recursive splitting
chunker = RecursiveChunker(chunk_size=500, overlap=50)

# Semantic-aware chunking
chunker = SemanticChunker(min_chunk_size=100, max_chunk_size=1000)
```

#### Retrieval Strategies

```python
from pyagent.rag import VectorRetriever, KeywordRetriever, HybridRetriever

# Vector similarity
retriever = VectorRetriever(embedding, vectorstore)

# BM25 keyword search
retriever = KeywordRetriever(documents)

# Hybrid (vector + keyword)
retriever = HybridRetriever(
    vector_retriever=vector_ret,
    keyword_retriever=keyword_ret,
    alpha=0.5  # Weight for vector scores
)
```

#### Complete RAG Pipeline

```python
from pyagent.rag import RAGPipeline, Document, SimpleRAG

# Simple usage
rag = SimpleRAG()
await rag.add("Python is a programming language")
await rag.add("JavaScript is also a programming language")
results = await rag.search("programming")

# Full pipeline
pipeline = RAGPipeline(
    embedding=LocalEmbedding(),
    vectorstore=ChromaVectorStore(),
    retriever=VectorRetriever(embedding, vectorstore),
)

# Index documents
docs = [
    Document(id="1", content="Python is great for AI", metadata={"topic": "python"}),
    Document(id="2", content="JavaScript runs in browsers", metadata={"topic": "javascript"}),
]
await pipeline.index(docs)

# Retrieve relevant documents
results = await pipeline.retrieve("What is Python good for?", k=3)

# Use as Agent tool
from pyagent.rag import create_rag_tool
rag_tool = create_rag_tool(pipeline, name="knowledge_search")
agent.add_tool(rag_tool)
```

---

### 3. Memory System

Multi-layer memory system for storing different types of information.

#### Memory Types

| Type | Purpose | Storage |
|------|---------|---------|
| **ConversationMemory** | Short-term chat history | In-memory deque |
| **SemanticMemory** | Facts and preferences | Vector store |
| **ProceduralMemory** | Skills and workflows | Key-value |
| **EpisodicMemory** | Events and experiences | Vector store |

#### Usage

```python
from pyagent.memory import (
    ConversationMemory,
    SemanticMemory,
    ProceduralMemory,
    EpisodicMemory,
    MemoryManager,
    Workflow, WorkflowStep,
    Episode,
)

# Semantic memory - store facts
semantic = SemanticMemory(vectorstore, embedding)
await semantic.store("user_name", "Alice", category="personal", importance=0.8)
await semantic.store("preferred_language", "Python", category="preferences")

# Recall memories
facts = await semantic.recall("What does the user prefer?")

# Procedural memory - store workflows
procedural = ProceduralMemory()
workflow = Workflow(
    id="",
    name="Deploy Application",
    description="Steps to deploy",
    steps=[
        WorkflowStep(name="build", action="Run build", tool="bash"),
        WorkflowStep(name="test", action="Run tests", tool="pytest"),
        WorkflowStep(name="deploy", action="Deploy to server"),
    ],
    tags=["deployment", "ci"],
)
await procedural.learn(workflow)

# Episodic memory - store events
episodic = EpisodicMemory(vectorstore, embedding)
episode = Episode(
    id="",
    summary="User had login issue",
    details="User couldn't login due to expired password",
    outcome="Password reset successful",
    tags=["support", "login"],
)
await episodic.record(episode)

# Unified memory manager
manager = MemoryManager(
    conversation_memory=ConversationMemory(),
    semantic_memory=semantic,
    procedural_memory=procedural,
    episodic_memory=episodic,
)

# Recall from all memory types
memories = await manager.remember("What happened with login?")
# Returns: {short_term, long_term, episodes, workflows}

# Build context for LLM
context = await manager.build_context("user preferences", max_tokens=2000)
```

---

### 4. State Persistence

Checkpoint-based state management for save/restore functionality.

#### Backends

| Backend | Use Case | Features |
|---------|----------|----------|
| **MemoryCheckpointBackend** | Testing | In-memory, fast |
| **SQLiteCheckpointBackend** | Single machine | Persistent, simple |
| **RedisCheckpointBackend** | Distributed | TTL, scalable |

#### Usage

```python
from pyagent.state import (
    CheckpointManager,
    SQLiteCheckpointBackend,
    RedisCheckpointBackend,
    MemoryCheckpointBackend,
)

# SQLite backend (persistent)
backend = SQLiteCheckpointBackend(db_path="checkpoints.db")
manager = CheckpointManager(backend)

# Redis backend (distributed with TTL)
backend = RedisCheckpointBackend(
    redis_url="redis://localhost:6379",
    ttl=86400,  # 24 hours
)
manager = CheckpointManager(backend)

# Create checkpoint
checkpoint = await manager.save(
    thread_id="conversation_123",
    state={
        "messages": [...],
        "context": {...},
    },
    metadata={"source": "auto_save"},
)

# Load checkpoint
loaded = await manager.load(checkpoint.id)

# Get latest checkpoint for a thread
latest = await manager.get_latest("conversation_123")

# List all checkpoints
checkpoints = await manager.list("conversation_123")

# Rollback to specific checkpoint
await manager.rollback("conversation_123", checkpoint.id)
```

---

## MCP Support

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

# Use MCP client
async with MCPClient.from_stdio("python", ["-m", "server"]) as client:
    tools = await client.list_tools()
    result = await client.call_tool("tool_name", {"arg": "value"})
```

---

## Skills System

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

Usage:

```python
from pyagent.skills.loader import SkillLoader
from pyagent.skills.registry import SkillRegistry

# Load skills
loader = SkillLoader()
skill = loader.load(Path("./my-skill"))

# Register in agent
agent = Agent(provider=provider, skills=[skill])
```

---

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

# RAG configuration
rag:
  embedding_model: text-embedding-3-small
  vectorstore:
    type: chroma
    persist_directory: ./chroma_db

# Memory configuration
memory:
  max_entries: 1000

# State persistence
state:
  backend: sqlite
  db_path: ./state/checkpoints.db
```

---

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
        mcp_clients: list[MCPClient] | None = None,
        hooks_registry: HookRegistry | None = None,
    ): ...

    async def run(self, message: str) -> str: ...
    async def run_stream(self, message: str) -> AsyncIterator[str]: ...
    def add_tool(self, tool: Tool) -> None: ...
    def add_skill(self, skill: Skill) -> None: ...
    def register_hook(self, hook: BaseHook) -> str: ...
    def hook(self, position: HookPosition) -> Callable: ...
```

### Tool

```python
@tool(description="Tool description")
def my_tool(arg: str) -> str:
    return f"Result: {arg}"
```

---

## Development

```bash
# Install dev dependencies
pip install pyagent[dev]

# Run tests
pytest

# Run tests with coverage
pytest --cov=pyagent --cov-report=html

# Run linting
ruff check src/

# Type checking
mypy src/
```

---

## Testing

The project includes comprehensive test suites:

| Module | Tests | Coverage |
|--------|-------|----------|
| Hooks & Lifecycle | 47 | 86% |
| RAG System | 21 | - |
| Memory System | 21 | - |
| State Persistence | 20 | - |
| **Total** | **109** | - |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License
