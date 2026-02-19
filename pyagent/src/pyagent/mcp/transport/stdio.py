"""
Stdio transport for MCP.
"""

import asyncio
import json
from typing import Any, AsyncIterator

from pyagent.mcp.transport import Transport
from pyagent.utils.logging import get_logger

logger = get_logger(__name__)


class StdioTransport(Transport):
    """
    Transport for MCP over stdio (stdin/stdout).
    Used for communicating with MCP servers as subprocesses.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None
    ):
        self.command = command
        self.args = args or []
        self.env = env
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}

    async def connect(self) -> None:
        """Start the subprocess and establish connection."""
        logger.info(f"Starting MCP server: {self.command} {' '.join(self.args)}")

        self._process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env
        )

        # Start reader task
        self._reader_task = asyncio.create_task(self._read_loop())

        logger.info("MCP server process started")

    async def close(self) -> None:
        """Terminate the subprocess."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()

        logger.info("MCP server process terminated")

    async def send(self, message: dict[str, Any]) -> None:
        """Send a message to the server."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Not connected")

        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

        logger.debug(f"Sent: {message}")

    async def receive(self) -> AsyncIterator[dict[str, Any]]:
        """Receive messages from the server."""
        while True:
            message = await self._response_queue.get()
            yield message

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a request and wait for response."""
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

        await self.send(message)

        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"Request {method} timed out")

    async def _read_loop(self) -> None:
        """Background task to read from stdout."""
        if not self._process or not self._process.stdout:
            return

        buffer = ""
        while True:
            try:
                data = await self._process.stdout.read(4096)
                if not data:
                    break

                buffer += data.decode()

                # Process complete lines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        message = json.loads(line)
                        await self._handle_message(message)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse message: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                break

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle an incoming message."""
        logger.debug(f"Received: {message}")

        # Check if it's a response to a request
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
                # Not a pending request, queue it
                await self._response_queue.put(message)
        else:
            # Notification, queue it
            await self._response_queue.put(message)
