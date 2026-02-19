"""
Anthropic Claude LLM Provider.
"""

import json
from typing import Any, AsyncIterator

from pyagent.core.message import Message
from pyagent.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """
    LLM Provider for Anthropic Claude API.
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
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )

            self._client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client

    def _convert_messages(
        self,
        messages: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Convert messages to Anthropic format.
        Returns (system_prompt, messages)
        """
        system_prompt = ""
        converted = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_prompt += content + "\n"
                continue

            # Anthropic uses "user" and "assistant" roles
            if role in ("user", "assistant"):
                converted.append({
                    "role": role,
                    "content": content
                })
            elif role == "tool":
                # Convert tool result to user message with tool_result content
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_result":
                            converted.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": block.get("tool_use_id"),
                                    "content": block.get("content", "")
                                }]
                            })

        return system_prompt.strip(), converted

    def _convert_tools(
        self,
        tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Convert tools to Anthropic format."""
        if not tools:
            return None

        converted = []
        for tool in tools:
            converted.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", {})
            })
        return converted

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Get a completion from Anthropic."""
        client = self._get_client()

        system_prompt, converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)

        params: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            params["system"] = system_prompt

        if converted_tools:
            params["tools"] = converted_tools

        if "system" in kwargs:
            params["system"] = kwargs.pop("system")
        elif "system_prompt" in kwargs:
            params["system"] = kwargs.pop("system_prompt")

        params.update(kwargs)

        response = await client.messages.create(**params)

        result: dict[str, Any] = {
            "message": None,
            "tool_calls": [],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            "finish_reason": response.stop_reason or "stop"
        }

        # Process content blocks
        text_content = []
        for block in response.content:
            if block.type == "text":
                text_content.append(block.text)
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })

        if text_content:
            result["message"] = Message.assistant("\n".join(text_content))

        return result

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Stream a completion from Anthropic."""
        client = self._get_client()

        system_prompt, converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)

        params: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            params["system"] = system_prompt

        if converted_tools:
            params["tools"] = converted_tools

        params.update(kwargs)

        tool_calls_accumulator: dict[str, dict[str, Any]] = {}

        async with client.messages.stream(**params) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield event.delta.text
                    elif event.delta.type == "input_json_delta":
                        # Accumulate tool input
                        tool_id = event.index
                        if tool_id not in tool_calls_accumulator:
                            tool_calls_accumulator[tool_id] = {
                                "id": "",
                                "name": "",
                                "input_str": ""
                            }
                        tool_calls_accumulator[tool_id]["input_str"] += event.delta.partial_json

                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        tool_id = event.index
                        tool_calls_accumulator[tool_id] = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "input_str": ""
                        }

            # Get final message and yield tool calls
            final_message = await stream.get_final_message()

            for block in final_message.content:
                if block.type == "tool_use":
                    yield {
                        "tool_call": {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        }
                    }

    def get_available_models(self) -> list[str]:
        """Get list of available Anthropic models."""
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
