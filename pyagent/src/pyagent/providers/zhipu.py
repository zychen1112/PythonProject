"""
Zhipu GLM LLM Provider.
"""

import json
from typing import Any, AsyncIterator

from pyagent.core.message import Message
from pyagent.providers.base import LLMProvider


class ZhipuProvider(LLMProvider):
    """
    LLM Provider for Zhipu GLM API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Get or create Zhipu client."""
        if self._client is None:
            try:
                from zhipuai import ZhipuAI
            except ImportError:
                raise ImportError(
                    "zhipuai package not installed. "
                    "Install with: pip install zhipuai"
                )

            self._client = ZhipuAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client

    def _convert_messages(
        self,
        messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert messages to Zhipu format.
        Handles tool_result messages which have different format in Zhipu API.
        """
        converted = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Handle tool result messages
            if role == "tool":
                # Extract tool_call_id and content from the content blocks
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            converted.append({
                                "role": "tool",
                                "content": block.get("content", ""),
                                "tool_call_id": block.get("tool_use_id", "")
                            })
                            break
                else:
                    converted.append(msg)
            # Handle assistant messages with tool_calls
            elif role == "assistant":
                new_msg = {"role": "assistant"}
                if isinstance(content, str):
                    new_msg["content"] = content
                elif isinstance(content, list):
                    # Extract text content
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    new_msg["content"] = "\n".join(text_parts) if text_parts else ""
                else:
                    new_msg["content"] = str(content) if content else ""
                converted.append(new_msg)
            else:
                # For user and system messages, keep as is
                converted.append(msg)

        return converted

    def _convert_tools(
        self,
        tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Convert tools to Zhipu format."""
        if not tools:
            return None

        converted = []
        for tool in tools:
            converted.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", tool.get("parameters", {}))
                }
            })
        return converted

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str = "glm-4-plus",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Get a completion from Zhipu GLM."""
        client = self._get_client()

        # Convert messages to Zhipu format
        converted_messages = self._convert_messages(messages)

        params: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            params["tools"] = converted_tools
            params["tool_choice"] = "auto"

        params.update(kwargs)

        # zhipuai SDK is synchronous, run in thread
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(**params)
        )

        # Extract content
        choice = response.choices[0]
        message = choice.message

        result: dict[str, Any] = {
            "message": None,
            "tool_calls": [],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            },
            "finish_reason": choice.finish_reason or "stop"
        }

        # Extract text content
        if message.content:
            result["message"] = Message.assistant(message.content)

        # Extract tool calls
        if message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": args
                })

        return result

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str = "glm-4-plus",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Stream a completion from Zhipu GLM."""
        client = self._get_client()

        # Convert messages to Zhipu format
        converted_messages = self._convert_messages(messages)

        params: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            params["tools"] = converted_tools
            params["tool_choice"] = "auto"

        params.update(kwargs)

        tool_calls_accumulator: dict[int, dict[str, Any]] = {}

        import asyncio
        loop = asyncio.get_event_loop()
        stream = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(**params)
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Yield text content
            if delta.content:
                yield delta.content

            # Accumulate tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if hasattr(tc, 'index') else 0
                    if idx not in tool_calls_accumulator:
                        tool_calls_accumulator[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": ""
                        }

                    if tc.id:
                        tool_calls_accumulator[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_accumulator[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_accumulator[idx]["arguments"] += tc.function.arguments

            # Check for finish
            if chunk.choices[0].finish_reason:
                # Yield accumulated tool calls
                for tc_data in tool_calls_accumulator.values():
                    try:
                        args = json.loads(tc_data["arguments"])
                    except json.JSONDecodeError:
                        args = {}

                    yield {
                        "tool_call": {
                            "id": tc_data["id"],
                            "name": tc_data["name"],
                            "input": args
                        }
                    }

    def get_available_models(self) -> list[str]:
        """Get list of available Zhipu GLM models."""
        return [
            "glm-4-plus",
            "glm-4-0520",
            "glm-4",
            "glm-4-air",
            "glm-4-airx",
            "glm-4-long",
            "glm-4-flash",
            "glm-4v-plus",
            "glm-4v",
        ]
