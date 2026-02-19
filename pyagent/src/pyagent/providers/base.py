"""
Base LLM Provider interface.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel

from pyagent.core.message import Message


class LLMResponse(BaseModel):
    """Response from an LLM."""
    message: Message | None = None
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    finish_reason: str = "stop"


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Get a completion from the LLM.

        Args:
            messages: List of messages in API format
            tools: List of available tools
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific options

        Returns:
            Dictionary with 'message', 'tool_calls', 'usage', 'finish_reason'
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> AsyncIterator[Any]:
        """
        Stream a completion from the LLM.

        Args:
            messages: List of messages in API format
            tools: List of available tools
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific options

        Yields:
            Chunks of the response (text strings or tool call dicts)
        """
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available models."""
        pass

    def count_tokens(self, text: str) -> int:
        """Estimate token count for text (simple implementation)."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
