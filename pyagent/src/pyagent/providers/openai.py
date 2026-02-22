"""
OpenAI LLM Provider.
"""

from typing import Any, AsyncIterator

from pyagent.core.message import Message
from pyagent.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    LLM Provider for OpenAI API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization
            )
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Get a completion from OpenAI."""
        client = self._get_client()

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            params["tools"] = [
                {"type": "function", "function": tool}
                for tool in tools
            ]
            params["tool_choice"] = "auto"

        params.update(kwargs)

        response = await client.chat.completions.create(**params)

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
        import json
        if message.tool_calls:
            for tc in message.tool_calls:
                # Safely parse arguments - never use eval()
                if isinstance(tc.function.arguments, dict):
                    args = tc.function.arguments
                else:
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
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
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Stream a completion from OpenAI."""
        client = self._get_client()

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if tools:
            params["tools"] = [
                {"type": "function", "function": tool}
                for tool in tools
            ]
            params["tool_choice"] = "auto"

        params.update(kwargs)

        stream = await client.chat.completions.create(**params)

        tool_calls_accumulator: dict[int, dict[str, Any]] = {}

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None

            if delta is None:
                continue

            # Yield text content
            if delta.content:
                yield delta.content

            # Accumulate tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
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
                    import json
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
        """Get list of available OpenAI models."""
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]
