"""
Agent - The main AI agent class.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Optional

from pydantic import BaseModel, Field

from pyagent.core.context import Context
from pyagent.core.executor import Executor
from pyagent.core.message import Message, ToolUseContent
from pyagent.core.tools import Tool
from pyagent.hooks import (
    HookAbortError,
    HookContext,
    HookExecutor,
    HookPosition,
    HookRegistry,
    HookResult,
    HookRetryError,
    HookSkipError,
)

if TYPE_CHECKING:
    from pyagent.mcp.client import MCPClient
    from pyagent.providers.base import LLMProvider
    from pyagent.skills.skill import Skill


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str = "PyAgent"
    max_iterations: int = 10
    system_prompt: str | None = None
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4096


class Agent:
    """
    Main AI Agent class that orchestrates LLM calls, tool execution, and skill invocation.

    The agent now supports a hooks system that allows injecting custom logic at
    various points in the execution lifecycle.
    """

    def __init__(
        self,
        provider: "LLMProvider",
        config: AgentConfig | None = None,
        tools: list[Tool] | None = None,
        skills: list["Skill"] | None = None,
        mcp_clients: list["MCPClient"] | None = None,
        context: Context | None = None,
        hooks_registry: HookRegistry | None = None,
    ):
        self.provider = provider
        self.config = config or AgentConfig()
        self.context = context or Context()
        self.executor = Executor(self.context)
        self.mcp_clients = mcp_clients or []

        # Initialize hooks system
        self.hooks = hooks_registry or HookRegistry()
        self.hook_executor = HookExecutor(self.hooks)

        # Register tools
        if tools:
            for tool in tools:
                self.context.register_tool(tool)

        # Register skills
        if skills:
            for skill in skills:
                self.context.register_skill(skill)

        # Set up system prompt
        if self.config.system_prompt:
            self.context.add_system_message(self.config.system_prompt)

    def hook(self, position: HookPosition, priority: int = 100, name: Optional[str] = None):
        """Decorator method to register a hook on this agent.

        Args:
            position: The hook position in the lifecycle
            priority: Execution priority (lower = earlier)
            name: Optional name for the hook

        Returns:
            Decorator function

        Example:
            @agent.hook(HookPosition.ON_TOOL_CALL)
            async def log_tools(ctx):
                print(f"Tool: {ctx.tool_name}")
                return HookResult.continue_()
        """
        return self.hooks.hook(position, priority, name)

    def register_hook(
        self,
        hook: "BaseHook",
        position: Optional[HookPosition] = None,
        priority: Optional[int] = None,
        name: Optional[str] = None,
    ) -> str:
        """Register a hook instance on this agent.

        Args:
            hook: The hook instance to register
            position: Override hook position (optional)
            priority: Override hook priority (optional)
            name: Optional name for the hook

        Returns:
            The name of the registered hook
        """
        return self.hooks.register(hook, position, priority, name)

    async def _execute_hook(
        self,
        position: HookPosition,
        context: HookContext,
    ) -> HookResult:
        """Execute hooks at a position with error handling.

        Args:
            position: The hook position
            context: The execution context

        Returns:
            HookResult from the execution
        """
        return await self.hook_executor.execute_safe(position, context)

    def _create_hook_context(
        self,
        position: HookPosition,
        **kwargs,
    ) -> HookContext:
        """Create a hook context with common fields.

        Args:
            position: The hook position
            **kwargs: Additional context fields

        Returns:
            HookContext instance
        """
        return HookContext(
            agent_id=self.config.name,
            position=position,
            messages=list(self.context.messages),
            **kwargs,
        )

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.context.register_tool(tool)

    def remove_tool(self, name: str) -> Tool | None:
        """Remove a tool from the agent."""
        return self.context.unregister_tool(name)

    def add_skill(self, skill: "Skill") -> None:
        """Add a skill to the agent."""
        self.context.register_skill(skill)

    def remove_skill(self, name: str) -> "Skill | None":
        """Remove a skill from the agent."""
        return self.context.unregister_skill(name)

    async def add_mcp_client(self, client: "MCPClient") -> None:
        """Add an MCP client and register its tools."""
        self.mcp_clients.append(client)
        await self._register_mcp_tools(client)

    async def _register_mcp_tools(self, client: "MCPClient") -> None:
        """Register tools from an MCP client."""
        tools = await client.list_tools()
        for tool_info in tools:
            # Create a wrapper tool for each MCP tool
            tool_name = tool_info["name"]
            mcp_tool = Tool(
                name=tool_name,
                description=tool_info.get("description", ""),
                input_schema=tool_info.get("input_schema", {}),
                handler=self._create_mcp_tool_handler(client, tool_name)
            )
            self.context.register_tool(mcp_tool)

    def _create_mcp_tool_handler(self, client: "MCPClient", tool_name: str):
        """Create a handler function for an MCP tool."""
        async def handler(**kwargs):
            return await self._call_mcp_tool(client, tool_name, kwargs)
        return handler

    async def _call_mcp_tool(
        self,
        client: "MCPClient",
        name: str,
        arguments: dict[str, Any]
    ) -> str:
        """Call a tool on an MCP server."""
        result = await client.call_tool(name, arguments)
        return result.get("content", "")

    async def run(
        self,
        message: str,
        on_tool_call: Callable[[str, dict], None] | None = None
    ) -> str:
        """
        Run the agent with a user message.

        Args:
            message: User input message
            on_tool_call: Optional callback when a tool is called

        Returns:
            Agent's response text
        """
        # Hook: ON_RUN_START
        ctx = self._create_hook_context(
            HookPosition.ON_RUN_START,
            message=message,
        )
        hook_result = await self._execute_hook(HookPosition.ON_RUN_START, ctx)
        if hook_result.is_abort():
            return hook_result.message or "Execution aborted by hook"

        # Add user message to context
        self.context.add_user_message(message)

        # Hook: ON_MESSAGE
        msg_ctx = self._create_hook_context(HookPosition.ON_MESSAGE, message=message)
        await self._execute_hook(HookPosition.ON_MESSAGE, msg_ctx)

        iteration = 0
        final_response = None

        while iteration < self.config.max_iterations:
            iteration += 1

            # Hook: ON_ITERATION_START
            iter_ctx = self._create_hook_context(
                HookPosition.ON_ITERATION_START,
                iteration=iteration,
            )
            hook_result = await self._execute_hook(HookPosition.ON_ITERATION_START, iter_ctx)
            if hook_result.is_abort():
                return hook_result.message or "Execution aborted by hook"

            try:
                # Hook: ON_LLM_CALL
                llm_ctx = self._create_hook_context(
                    HookPosition.ON_LLM_CALL,
                    iteration=iteration,
                )
                hook_result = await self._execute_hook(HookPosition.ON_LLM_CALL, llm_ctx)

                if hook_result.is_retry():
                    # Handle retry
                    await asyncio.sleep(hook_result.data.get("after_seconds", 0))
                    iteration -= 1  # Retry same iteration
                    continue

                if hook_result.is_abort():
                    return hook_result.message or "Execution aborted by hook"

                # Get completion from LLM
                response = await self.provider.complete(
                    messages=self.context.get_api_messages(),
                    tools=self.context.get_tools_api_format(),
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )

                # Hook: ON_LLM_RESPONSE
                resp_ctx = self._create_hook_context(
                    HookPosition.ON_LLM_RESPONSE,
                    iteration=iteration,
                    llm_response=response,
                )
                await self._execute_hook(HookPosition.ON_LLM_RESPONSE, resp_ctx)

                # Extract content and tool calls
                assistant_message = response.get("message")
                tool_calls = response.get("tool_calls", [])

                # Add assistant message to context
                if assistant_message:
                    self.context.add_message(assistant_message)

                # If no tool calls, we're done
                if not tool_calls:
                    # Return the text content
                    content = assistant_message.content if assistant_message else ""
                    if isinstance(content, str):
                        final_response = content
                    else:
                        # Extract text from content blocks
                        texts = []
                        for block in content:
                            if hasattr(block, 'text'):
                                texts.append(block.text)
                        final_response = "\n".join(texts)
                    break

                # Execute tool calls
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["input"]
                    tool_id = tool_call["id"]

                    # Hook: ON_TOOL_CALL
                    tool_ctx = self._create_hook_context(
                        HookPosition.ON_TOOL_CALL,
                        iteration=iteration,
                        tool_name=tool_name,
                        tool_arguments=tool_args,
                    )
                    hook_result = await self._execute_hook(HookPosition.ON_TOOL_CALL, tool_ctx)

                    if hook_result.is_skip():
                        continue

                    if hook_result.is_abort():
                        return hook_result.message or "Execution aborted by hook"

                    if on_tool_call:
                        on_tool_call(tool_name, tool_args)

                    # Execute tool
                    try:
                        _, result = await self.executor.execute_tool(
                            tool_name, tool_args, tool_id
                        )

                        # Hook: ON_TOOL_RESULT
                        result_ctx = self._create_hook_context(
                            HookPosition.ON_TOOL_RESULT,
                            iteration=iteration,
                            tool_name=tool_name,
                            tool_arguments=tool_args,
                            tool_result=result,
                        )
                        await self._execute_hook(HookPosition.ON_TOOL_RESULT, result_ctx)

                        # Add result to context
                        self.context.add_message(Message.tool_result(
                            tool_use_id=tool_id,
                            content=result.content,
                            is_error=result.is_error,
                            name=result.tool_name
                        ))

                    except Exception as e:
                        # Hook: ON_ERROR
                        error_ctx = self._create_hook_context(
                            HookPosition.ON_ERROR,
                            iteration=iteration,
                            error=e,
                            tool_name=tool_name,
                        )
                        hook_result = await self._execute_hook(HookPosition.ON_ERROR, error_ctx)

                        if hook_result.is_retry():
                            await asyncio.sleep(hook_result.data.get("after_seconds", 0))
                            continue

                        # Re-raise if not handled
                        raise

            finally:
                # Hook: ON_ITERATION_END
                end_ctx = self._create_hook_context(
                    HookPosition.ON_ITERATION_END,
                    iteration=iteration,
                )
                await self._execute_hook(HookPosition.ON_ITERATION_END, end_ctx)

        # Hook: ON_RUN_END
        end_ctx = self._create_hook_context(
            HookPosition.ON_RUN_END,
            message=final_response or "Max iterations reached",
        )
        await self._execute_hook(HookPosition.ON_RUN_END, end_ctx)

        return final_response or "Max iterations reached without completion."

    async def run_stream(
        self,
        message: str,
        on_tool_call: Callable[[str, dict], None] | None = None
    ) -> AsyncIterator[str]:
        """
        Run the agent with streaming response.

        Args:
            message: User input message
            on_tool_call: Optional callback when a tool is called

        Yields:
            Chunks of the agent's response
        """
        # Hook: ON_RUN_START
        ctx = self._create_hook_context(
            HookPosition.ON_RUN_START,
            message=message,
        )
        hook_result = await self._execute_hook(HookPosition.ON_RUN_START, ctx)
        if hook_result.is_abort():
            yield hook_result.message or "Execution aborted by hook"
            return

        # Add user message to context
        self.context.add_user_message(message)

        # Hook: ON_MESSAGE
        msg_ctx = self._create_hook_context(HookPosition.ON_MESSAGE, message=message)
        await self._execute_hook(HookPosition.ON_MESSAGE, msg_ctx)

        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1

            # Hook: ON_ITERATION_START
            iter_ctx = self._create_hook_context(
                HookPosition.ON_ITERATION_START,
                iteration=iteration,
            )
            hook_result = await self._execute_hook(HookPosition.ON_ITERATION_START, iter_ctx)
            if hook_result.is_abort():
                yield hook_result.message or "Execution aborted by hook"
                return

            full_response = ""
            tool_calls = []

            try:
                # Hook: ON_LLM_CALL
                llm_ctx = self._create_hook_context(
                    HookPosition.ON_LLM_CALL,
                    iteration=iteration,
                )
                hook_result = await self._execute_hook(HookPosition.ON_LLM_CALL, llm_ctx)

                if hook_result.is_retry():
                    await asyncio.sleep(hook_result.data.get("after_seconds", 0))
                    iteration -= 1
                    continue

                if hook_result.is_abort():
                    yield hook_result.message or "Execution aborted by hook"
                    return

                # Stream completion from LLM
                async for chunk in self.provider.stream(
                    messages=self.context.get_api_messages(),
                    tools=self.context.get_tools_api_format(),
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                ):
                    # Check if it's a text chunk or tool call info
                    if isinstance(chunk, dict):
                        if "text" in chunk:
                            text = chunk["text"]
                            full_response += text
                            yield text
                        elif "tool_call" in chunk:
                            tool_calls.append(chunk["tool_call"])
                    elif isinstance(chunk, str):
                        full_response += chunk
                        yield chunk

                # Hook: ON_LLM_RESPONSE
                resp_ctx = self._create_hook_context(
                    HookPosition.ON_LLM_RESPONSE,
                    iteration=iteration,
                    llm_response={"text": full_response, "tool_calls": tool_calls},
                )
                await self._execute_hook(HookPosition.ON_LLM_RESPONSE, resp_ctx)

                # Add assistant message
                if full_response:
                    self.context.add_assistant_message(full_response)

                # If no tool calls, we're done
                if not tool_calls:
                    break

                # Execute tool calls
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["input"]
                    tool_id = tool_call["id"]

                    # Hook: ON_TOOL_CALL
                    tool_ctx = self._create_hook_context(
                        HookPosition.ON_TOOL_CALL,
                        iteration=iteration,
                        tool_name=tool_name,
                        tool_arguments=tool_args,
                    )
                    hook_result = await self._execute_hook(HookPosition.ON_TOOL_CALL, tool_ctx)

                    if hook_result.is_skip():
                        continue

                    if hook_result.is_abort():
                        yield hook_result.message or "Execution aborted by hook"
                        return

                    if on_tool_call:
                        on_tool_call(tool_name, tool_args)

                    try:
                        _, result = await self.executor.execute_tool(
                            tool_name, tool_args, tool_id
                        )

                        # Hook: ON_TOOL_RESULT
                        result_ctx = self._create_hook_context(
                            HookPosition.ON_TOOL_RESULT,
                            iteration=iteration,
                            tool_name=tool_name,
                            tool_arguments=tool_args,
                            tool_result=result,
                        )
                        await self._execute_hook(HookPosition.ON_TOOL_RESULT, result_ctx)

                        self.context.add_message(Message.tool_result(
                            tool_use_id=tool_id,
                            content=result.content,
                            is_error=result.is_error,
                            name=result.tool_name
                        ))

                    except Exception as e:
                        # Hook: ON_ERROR
                        error_ctx = self._create_hook_context(
                            HookPosition.ON_ERROR,
                            iteration=iteration,
                            error=e,
                            tool_name=tool_name,
                        )
                        hook_result = await self._execute_hook(HookPosition.ON_ERROR, error_ctx)

                        if hook_result.is_retry():
                            await asyncio.sleep(hook_result.data.get("after_seconds", 0))
                            continue

                        raise

            finally:
                # Hook: ON_ITERATION_END
                end_ctx = self._create_hook_context(
                    HookPosition.ON_ITERATION_END,
                    iteration=iteration,
                )
                await self._execute_hook(HookPosition.ON_ITERATION_END, end_ctx)

            yield "\n[Tool execution completed, continuing...]\n"

        # Hook: ON_RUN_END
        end_ctx = self._create_hook_context(
            HookPosition.ON_RUN_END,
            message=full_response or "Stream completed",
        )
        await self._execute_hook(HookPosition.ON_RUN_END, end_ctx)

    def clear_history(self) -> None:
        """Clear conversation history but keep system prompt."""
        system_messages = [
            m for m in self.context.messages
            if m.role.value == "system"
        ]
        self.context.messages = system_messages

    def get_context_summary(self) -> dict[str, Any]:
        """Get a summary of the current context."""
        return {
            "message_count": len(self.context.messages),
            "tool_count": len(self.context.tools),
            "skill_count": len(self.context.skills),
            "mcp_client_count": len(self.mcp_clients)
        }
