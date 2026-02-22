"""
PyAgent - A lightweight AI Agent framework with MCP and Skills support.
"""

from pyagent.core.agent import Agent
from pyagent.core.message import Message, Role
from pyagent.core.context import Context
from pyagent.core.tools import Tool
from pyagent.hooks import (
    BaseHook,
    HookAction,
    HookContext,
    HookExecutor,
    HookPosition,
    HookRegistry,
    HookResult,
    LoggingHook,
    TimingHook,
    ErrorHandlingHook,
    MetricsHook,
    RateLimitHook,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "Agent",
    "Message",
    "Role",
    "Context",
    "Tool",
    # Hooks
    "BaseHook",
    "HookAction",
    "HookContext",
    "HookExecutor",
    "HookPosition",
    "HookRegistry",
    "HookResult",
    # Built-in hooks
    "LoggingHook",
    "TimingHook",
    "ErrorHandlingHook",
    "MetricsHook",
    "RateLimitHook",
]
