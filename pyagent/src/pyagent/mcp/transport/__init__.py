"""
MCP Transport layer.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel


class Transport(ABC):
    """Abstract base class for MCP transports."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def send(self, message: dict[str, Any]) -> None:
        """Send a message."""
        pass

    @abstractmethod
    async def receive(self) -> AsyncIterator[dict[str, Any]]:
        """Receive messages."""
        pass

    @abstractmethod
    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a request and wait for response."""
        pass

    async def __aenter__(self) -> "Transport":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
