"""
MCP Server base class for creating MCP servers.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from pyagent.mcp.types import (
    ServerInfo,
    ToolDefinition,
    ResourceDefinition,
    PromptDefinition,
)
from pyagent.utils.logging import get_logger

logger = get_logger(__name__)


class MCPServer:
    """
    Base class for creating MCP servers.

    Use decorators to define tools, resources, and prompts:

    @server.tool()
    def my_tool(arg: str) -> str:
        '''Tool description'''
        return f"Result: {arg}"

    @server.resource("file:///{path}")
    def my_resource(path: str) -> str:
        return open(path).read()

    @server.prompt()
    def my_prompt(topic: str) -> str:
        return f"Tell me about {topic}"
    """

    def __init__(
        self,
        name: str = "pyagent-server",
        version: str = "0.1.0"
    ):
        self.name = name
        self.version = version
        self._tools: dict[str, ToolHandler] = {}
        self._resources: dict[str, ResourceHandler] = {}
        self._prompts: dict[str, PromptHandler] = {}
        self._resource_templates: list[str] = []

    def tool(
        self,
        name: str | None = None,
        description: str | None = None
    ) -> Callable:
        """Decorator to register a tool."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or ""

            # Build input schema from function signature
            input_schema = self._build_schema_from_func(func)

            handler = ToolHandler(
                name=tool_name,
                description=tool_desc,
                input_schema=input_schema,
                handler=func
            )

            self._tools[tool_name] = handler
            logger.debug(f"Registered tool: {tool_name}")

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def resource(
        self,
        uri_template: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: str = "text/plain"
    ) -> Callable:
        """Decorator to register a resource."""
        def decorator(func: Callable) -> Callable:
            resource_name = name or func.__name__
            resource_desc = description or func.__doc__ or ""

            handler = ResourceHandler(
                uri_template=uri_template,
                name=resource_name,
                description=resource_desc,
                mime_type=mime_type,
                handler=func
            )

            self._resources[uri_template] = handler
            self._resource_templates.append(uri_template)

            logger.debug(f"Registered resource: {uri_template}")

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def prompt(
        self,
        name: str | None = None,
        description: str | None = None
    ) -> Callable:
        """Decorator to register a prompt."""
        def decorator(func: Callable) -> Callable:
            prompt_name = name or func.__name__
            prompt_desc = description or func.__doc__ or ""

            # Build arguments schema from function signature
            arguments = self._build_arguments_from_func(func)

            handler = PromptHandler(
                name=prompt_name,
                description=prompt_desc,
                arguments=arguments,
                handler=func
            )

            self._prompts[prompt_name] = handler
            logger.debug(f"Registered prompt: {prompt_name}")

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def _build_schema_from_func(self, func: Callable) -> dict[str, Any]:
        """Build JSON schema from function signature."""
        sig = inspect.signature(func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_info: dict[str, Any] = {"type": "string"}

            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_info = {"type": "integer"}
                elif param.annotation == float:
                    param_info = {"type": "number"}
                elif param.annotation == bool:
                    param_info = {"type": "boolean"}
                elif param.annotation == list:
                    param_info = {"type": "array", "items": {"type": "string"}}
                elif param.annotation == dict:
                    param_info = {"type": "object"}

            properties[param_name] = param_info

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def _build_arguments_from_func(self, func: Callable) -> list[dict[str, Any]]:
        """Build arguments list from function signature."""
        sig = inspect.signature(func)
        arguments = []

        for param_name, param in sig.parameters.items():
            arg_info: dict[str, Any] = {
                "name": param_name,
                "description": "",
                "required": param.default == inspect.Parameter.empty
            }
            arguments.append(arg_info)

        return arguments

    async def handle_request(
        self,
        method: str,
        params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle an incoming request."""
        if method == "initialize":
            return await self._handle_initialize(params)
        elif method == "tools/list":
            return await self._handle_tools_list()
        elif method == "tools/call":
            return await self._handle_tools_call(params)
        elif method == "resources/list":
            return await self._handle_resources_list()
        elif method == "resources/read":
            return await self._handle_resources_read(params)
        elif method == "prompts/list":
            return await self._handle_prompts_list()
        elif method == "prompts/get":
            return await self._handle_prompts_get(params)
        else:
            raise ValueError(f"Unknown method: {method}")

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {}
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }

    async def _handle_tools_list(self) -> dict[str, Any]:
        """Handle tools/list request."""
        tools = []
        for handler in self._tools.values():
            tools.append({
                "name": handler.name,
                "description": handler.description,
                "inputSchema": handler.input_schema
            })
        return {"tools": tools}

    async def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        handler = self._tools[name]
        result = handler.handler(**arguments)

        if asyncio.iscoroutine(result):
            result = await result

        # Format result as MCP content
        if isinstance(result, str):
            content = [{"type": "text", "text": result}]
        elif isinstance(result, dict):
            content = [{"type": "text", "text": json.dumps(result)}]
        else:
            content = [{"type": "text", "text": str(result)}]

        return {"content": content}

    async def _handle_resources_list(self) -> dict[str, Any]:
        """Handle resources/list request."""
        resources = []
        for handler in self._resources.values():
            resources.append({
                "uri": handler.uri_template,
                "name": handler.name,
                "description": handler.description,
                "mimeType": handler.mime_type
            })
        return {"resources": resources}

    async def _handle_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")

        # Find matching resource handler
        handler = None
        for template, h in self._resources.items():
            if self._match_uri_template(template, uri):
                handler = h
                break

        if not handler:
            raise ValueError(f"Unknown resource: {uri}")

        # Extract params from URI
        uri_params = self._extract_uri_params(handler.uri_template, uri)

        result = handler.handler(**uri_params)

        if asyncio.iscoroutine(result):
            result = await result

        if isinstance(result, bytes):
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": handler.mime_type,
                    "blob": result.hex()
                }]
            }
        else:
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": handler.mime_type,
                    "text": str(result)
                }]
            }

    async def _handle_prompts_list(self) -> dict[str, Any]:
        """Handle prompts/list request."""
        prompts = []
        for handler in self._prompts.values():
            prompts.append({
                "name": handler.name,
                "description": handler.description,
                "arguments": handler.arguments
            })
        return {"prompts": prompts}

    async def _handle_prompts_get(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle prompts/get request."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if name not in self._prompts:
            raise ValueError(f"Unknown prompt: {name}")

        handler = self._prompts[name]
        result = handler.handler(**arguments)

        if asyncio.iscoroutine(result):
            result = await result

        return {
            "description": handler.description,
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": str(result)}
                }
            ]
        }

    def _match_uri_template(self, template: str, uri: str) -> bool:
        """Check if URI matches template."""
        import re
        # Convert template to regex
        pattern = re.sub(r'\{(\w+)\}', r'[^/]+', template)
        return bool(re.fullmatch(pattern, uri))

    def _extract_uri_params(self, template: str, uri: str) -> dict[str, str]:
        """Extract parameters from URI based on template."""
        import re
        params = {}

        # Find parameter names
        param_names = re.findall(r'\{(\w+)\}', template)

        # Build regex pattern
        pattern = re.sub(r'\{(\w+)\}', r'([^/]+)', template)
        match = re.fullmatch(pattern, uri)

        if match:
            for i, name in enumerate(param_names):
                params[name] = match.group(i + 1)

        return params


class ToolHandler(BaseModel):
    """Handler for a tool."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable


class ResourceHandler(BaseModel):
    """Handler for a resource."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    uri_template: str
    name: str
    description: str
    mime_type: str
    handler: Callable


class PromptHandler(BaseModel):
    """Handler for a prompt."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    arguments: list[dict[str, Any]]
    handler: Callable
