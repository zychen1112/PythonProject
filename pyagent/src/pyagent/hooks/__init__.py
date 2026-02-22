"""Hooks and lifecycle system for PyAgent.

This module provides a flexible hook system for extending the Agent
execution lifecycle. Hooks can be registered at various positions
to inject custom logic.

Example:
    from pyagent.hooks import HookRegistry, HookPosition, HookResult

    # Create a registry
    registry = HookRegistry()

    # Register a hook using a decorator
    @registry.hook(HookPosition.ON_TOOL_CALL)
    async def log_tool_call(context):
        print(f"Tool called: {context.tool_name}")
        return HookResult.continue_()

    # Use with an Agent
    agent = Agent(provider=provider, hooks_registry=registry)

Built-in Hooks:
    - LoggingHook: Log execution events
    - TimingHook: Track execution duration
    - ErrorHandlingHook: Handle errors with retry support
    - MetricsHook: Collect Prometheus-style metrics
    - RateLimitHook: Rate limit LLM requests
"""

from .base import BaseHook, HookAction, HookPosition
from .builtin import (
    ErrorHandlingHook,
    LoggingHook,
    MetricsHook,
    RateLimitHook,
    TimingHook,
)
from .context import HookContext
from .executor import (
    ExecutionStats,
    HookAbortError,
    HookExecutor,
    HookRetryError,
    HookSkipError,
)
from .registry import HookRegistration, HookRegistry
from .result import HookResult

__all__ = [
    # Base classes and enums
    "HookPosition",
    "HookAction",
    "BaseHook",
    "HookContext",
    "HookResult",
    # Registry
    "HookRegistry",
    "HookRegistration",
    # Executor
    "HookExecutor",
    "ExecutionStats",
    "HookAbortError",
    "HookSkipError",
    "HookRetryError",
    # Built-in hooks
    "LoggingHook",
    "TimingHook",
    "ErrorHandlingHook",
    "MetricsHook",
    "RateLimitHook",
]
