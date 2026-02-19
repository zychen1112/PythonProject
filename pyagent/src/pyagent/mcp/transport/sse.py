"""
SSE (Server-Sent Events) transport for MCP.
"""

import asyncio
import json
from typing import Any, AsyncIterator

import httpx

from pyagent.mcp.transport import Transport
from pyagent.utils.logging import get_logger

logger = get_logger(__name__)


class SSETransport(Transport):
    """
    Transport for MCP over Server-Sent Events.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None
    ):
        self.url = url
        self.headers = headers or {}
        self._client: httpx.AsyncClient | None = None
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._event_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Establish SSE connection."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, read=None)  # No read timeout for SSE
        )

        # Start SSE listener
        self._event_task = asyncio.create_task(self._sse_listener())

        logger.info(f"SSE transport connected to {self.url}")

    async def close(self) -> None:
        """Close the connection."""
        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()

        logger.info("SSE transport closed")

    async def send(self, message: dict[str, Any]) -> None:
        """Send a message via POST."""
        if not self._client:
            raise RuntimeError("Not connected")

        response = await self._client.post(
            self.url,
            json=message,
            headers=self.headers
        )
        response.raise_for_status()

        logger.debug(f"Sent via POST: {message}")

    async def receive(self) -> AsyncIterator[dict[str, Any]]:
        """Receive messages from SSE stream."""
        while True:
            message = await self._response_queue.get()
            yield message

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a request and wait for response."""
        if not self._client:
            raise RuntimeError("Not connected")

        self._request_id += 1
        request_id = self._request_id

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }

        # Create future for response
        future: asyncio.Future[dict[str, Any]] = asyncio.Future()
        self._pending_requests[request_id] = future

        try:
            response = await self._client.post(
                self.url,
                json=message,
                headers=self.headers
            )
            response.raise_for_status()

            # Check for immediate response
            data = response.json()
            if "id" in data and data["id"] == request_id:
                self._pending_requests.pop(request_id, None)
                if "error" in data:
                    raise Exception(data["error"].get("message", "Unknown error"))
                return data.get("result", {})

            # Wait for SSE response
            result = await asyncio.wait_for(future, timeout=30.0)
            return result

        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"Request {method} timed out")

    async def _sse_listener(self) -> None:
        """Listen for SSE events."""
        if not self._client:
            return

        try:
            async with self._client.stream(
                "GET",
                self.url,
                headers={**self.headers, "Accept": "text/event-stream"}
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                            await self._handle_message(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse SSE data: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SSE listener error: {e}")

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle an incoming message."""
        logger.debug(f"SSE received: {message}")

        if "id" in message:
            request_id = message["id"]
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if "error" in message:
                    future.set_exception(
                        Exception(message["error"].get("message", "Unknown error"))
                    )
                else:
                    future.set_result(message.get("result", {}))
            else:
                await self._response_queue.put(message)
        else:
            await self._response_queue.put(message)
