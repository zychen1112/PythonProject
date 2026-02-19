"""
Tests for core agent functionality.
"""

import pytest

from pyagent.core.message import Message, Role, TextContent
from pyagent.core.tools import Tool, ToolResult, tool
from pyagent.core.context import Context


class TestMessage:
    """Tests for Message class."""

    def test_create_system_message(self):
        msg = Message.system("You are a helpful assistant")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are a helpful assistant"

    def test_create_user_message(self):
        msg = Message.user("Hello!")
        assert msg.role == Role.USER
        assert msg.content == "Hello!"

    def test_create_assistant_message(self):
        msg = Message.assistant("Hi there!")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"

    def test_create_tool_result_message(self):
        msg = Message.tool_result("tool-123", "Result content", is_error=False)
        assert msg.role == Role.TOOL
        assert isinstance(msg.content, list)

    def test_to_api_format_string_content(self):
        msg = Message.user("Test message")
        api_format = msg.to_api_format()
        assert api_format["role"] == "user"
        assert api_format["content"] == "Test message"


class TestTool:
    """Tests for Tool class."""

    def test_create_tool_from_function(self):
        @tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        assert add.name == "add"
        assert add.description == "Add two numbers"
        assert "a" in add.input_schema.properties
        assert "b" in add.input_schema.properties

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        test_tool = Tool.from_function(
            lambda x: f"Echo: {x}",
            name="echo",
            description="Echo input"
        )

        result = await test_tool.execute(x="hello")
        assert result.content == "Echo: hello"
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        test_tool = Tool.from_function(
            lambda x: 1/0,  # Will raise error
            name="bad_tool",
            description="A tool that errors"
        )

        result = await test_tool.execute(x="test")
        assert result.is_error
        assert "Error" in result.content


class TestContext:
    """Tests for Context class."""

    def test_add_messages(self):
        ctx = Context()
        ctx.add_user_message("Hello")
        ctx.add_assistant_message("Hi!")

        assert len(ctx.messages) == 2
        assert ctx.messages[0].role == Role.USER
        assert ctx.messages[1].role == Role.ASSISTANT

    def test_register_tool(self):
        ctx = Context()
        test_tool = Tool(name="test", description="Test tool")
        ctx.register_tool(test_tool)

        assert "test" in ctx.tools
        assert ctx.get_tool("test") == test_tool

    def test_unregister_tool(self):
        ctx = Context()
        test_tool = Tool(name="test", description="Test tool")
        ctx.register_tool(test_tool)

        removed = ctx.unregister_tool("test")
        assert removed == test_tool
        assert "test" not in ctx.tools

    def test_message_trimming(self):
        ctx = Context(max_messages=5)
        ctx.add_system_message("System prompt")

        for i in range(10):
            ctx.add_user_message(f"Message {i}")

        # Should keep system message and last 4 user messages
        assert len(ctx.messages) == 5
        assert ctx.messages[0].role == Role.SYSTEM

    def test_get_api_messages(self):
        ctx = Context()
        ctx.add_user_message("Hello")

        api_messages = ctx.get_api_messages()
        assert len(api_messages) == 1
        assert api_messages[0]["role"] == "user"
