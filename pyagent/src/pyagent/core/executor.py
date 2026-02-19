"""
Executor - handles tool execution and skill invocation.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from pyagent.core.message import Message, ToolUseContent, ToolResultContent

if TYPE_CHECKING:
    from pyagent.core.context import Context
    from pyagent.core.tools import Tool, ToolResult


class ExecutionPlan(BaseModel):
    """Plan for executing a series of actions."""
    tool_calls: list[dict[str, Any]] = []
    skill_invocations: list[dict[str, Any]] = []


class Executor:
    """
    Executes tools and skills for an agent.
    """

    def __init__(self, context: "Context"):
        self.context = context

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_use_id: str | None = None
    ) -> tuple[str, "ToolResult"]:
        """
        Execute a tool and return the result.

        Returns:
            Tuple of (tool_use_id, ToolResult)
        """
        tool_use_id = tool_use_id or str(uuid.uuid4())

        tool = self.context.get_tool(tool_name)
        if tool is None:
            return tool_use_id, type("ToolResult", (), {
                "tool_name": tool_name,
                "content": f"Tool '{tool_name}' not found",
                "is_error": True,
                "metadata": {}
            })()

        result = await tool.execute(**arguments)
        return tool_use_id, result

    async def execute_tool_uses(
        self,
        tool_uses: list["ToolUseContent"]
    ) -> list["Message"]:
        """
        Execute multiple tool uses concurrently.

        Returns:
            List of tool result messages
        """
        tasks = [
            self.execute_tool(
                tool_use.name,
                tool_use.input,
                tool_use.id
            )
            for tool_use in tool_uses
        ]

        results = await asyncio.gather(*tasks)

        messages = []
        for tool_use_id, result in results:
            messages.append(Message.tool_result(
                tool_use_id=tool_use_id,
                content=result.content,
                is_error=result.is_error,
                name=result.tool_name
            ))

        return messages

    def extract_tool_uses(self, message: Message) -> list["ToolUseContent"]:
        """Extract tool use blocks from a message."""
        if isinstance(message.content, str):
            return []

        return [
            block for block in message.content
            if isinstance(block, ToolUseContent)
        ]

    async def process_assistant_message(
        self,
        message: Message
    ) -> list[Message]:
        """
        Process an assistant message, executing any tool calls.

        Returns:
            List of tool result messages
        """
        tool_uses = self.extract_tool_uses(message)
        if not tool_uses:
            return []

        return await self.execute_tool_uses(tool_uses)
