"""
Tool definitions and execution.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field


class ToolSchema(BaseModel):
    """JSON Schema for a tool parameter."""
    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class Tool(BaseModel):
    """A tool that can be used by an agent."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: ToolSchema = Field(default_factory=ToolSchema)
    handler: Callable[..., Any] | None = Field(default=None, exclude=True)

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: str | None = None,
        description: str | None = None
    ) -> "Tool":
        """Create a tool from a function."""
        sig = inspect.signature(func)
        doc = description or func.__doc__ or ""

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # Default to string

            # Try to get type annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_info = {"type": "integer"}
                elif param.annotation == float:
                    param_info = {"type": "number"}
                elif param.annotation == bool:
                    param_info = {"type": "boolean"}
                elif param.annotation == list:
                    param_info = {"type": "array"}
                elif param.annotation == dict:
                    param_info = {"type": "object"}
                elif hasattr(param.annotation, "__origin__"):
                    # Handle generic types like list[str]
                    origin = param.annotation.__origin__
                    if origin == list:
                        param_info = {"type": "array", "items": {"type": "string"}}

            properties[param_name] = param_info

            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return cls(
            name=name or func.__name__,
            description=doc,
            input_schema=ToolSchema(properties=properties, required=required),
            handler=func
        )

    def to_api_format(self) -> dict[str, Any]:
        """Convert to API-compatible format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_dump()
        }

    async def execute(self, **kwargs: Any) -> "ToolResult":
        """Execute the tool with given arguments."""
        if self.handler is None:
            return ToolResult(
                tool_name=self.name,
                content="Tool has no handler",
                is_error=True
            )

        try:
            result = self.handler(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result

            return ToolResult(
                tool_name=self.name,
                content=str(result) if not isinstance(result, str) else result
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                content=f"Error executing tool: {e}",
                is_error=True
            )


class ToolResult(BaseModel):
    """Result of a tool execution."""
    tool_name: str
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


def tool(
    name: str | None = None,
    description: str | None = None
) -> Callable[[Callable], Tool]:
    """Decorator to create a tool from a function."""
    def decorator(func: Callable) -> Tool:
        return Tool.from_function(func, name=name, description=description)
    return decorator
