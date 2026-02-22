"""Hook registry for managing hook registrations."""

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Union

from .base import BaseHook, HookPosition
from .result import HookResult


@dataclass
class HookRegistration:
    """Information about a registered hook.

    Attributes:
        hook: The hook instance
        position: Where the hook is registered
        priority: Execution priority (lower = earlier)
        name: Optional name for the hook
        enabled: Whether the hook is currently enabled
    """

    hook: BaseHook
    position: HookPosition
    priority: int
    name: Optional[str]
    enabled: bool = True


class HookRegistry:
    """Thread-safe registry for managing hooks.

    This class provides a central place to register, unregister, and
    manage hooks. It supports both programmatic registration and
    decorator-based registration.

    Example:
        registry = HookRegistry()

        # Programmatic registration
        registry.register(MyHook(), position=HookPosition.ON_TOOL_CALL)

        # Decorator registration
        @registry.hook(HookPosition.ON_RUN_START, priority=50)
        async def my_hook(context):
            print("Run starting!")
            return HookResult.continue_()
    """

    def __init__(self) -> None:
        """Initialize the hook registry."""
        self._hooks: dict[HookPosition, list[HookRegistration]] = {}
        self._lock = threading.RLock()
        self._name_counter = 0

    def register(
        self,
        hook: BaseHook,
        position: Optional[HookPosition] = None,
        priority: Optional[int] = None,
        name: Optional[str] = None,
    ) -> str:
        """Register a hook.

        Args:
            hook: The hook instance to register
            position: Override the hook's position (optional)
            priority: Override the hook's priority (optional)
            name: Optional name for the hook (for later reference)

        Returns:
            The name of the registered hook
        """
        # Use hook's position/priority if not specified
        position = position or hook.position
        priority = priority if priority is not None else hook.priority

        # Generate name if not provided
        if name is None:
            self._name_counter += 1
            name = f"{hook.__class__.__name__}_{self._name_counter}"

        with self._lock:
            if position not in self._hooks:
                self._hooks[position] = []

            registration = HookRegistration(
                hook=hook,
                position=position,
                priority=priority,
                name=name,
                enabled=True,
            )
            self._hooks[position].append(registration)
            # Sort by priority (lower = earlier)
            self._hooks[position].sort(key=lambda r: r.priority)

        return name

    def unregister(self, name: str) -> bool:
        """Unregister a hook by name.

        Args:
            name: The name of the hook to unregister

        Returns:
            True if a hook was unregistered, False otherwise
        """
        found = False
        with self._lock:
            for position in self._hooks:
                original_len = len(self._hooks[position])
                self._hooks[position] = [
                    r for r in self._hooks[position] if r.name != name
                ]
                if len(self._hooks[position]) < original_len:
                    found = True
        return found

    def get_hooks(self, position: HookPosition) -> list[HookRegistration]:
        """Get all enabled hooks for a position, sorted by priority.

        Args:
            position: The hook position to get hooks for

        Returns:
            List of enabled hook registrations, sorted by priority
        """
        with self._lock:
            return [
                r for r in self._hooks.get(position, [])
                if r.enabled
            ]

    def get_all_hooks(self) -> dict[HookPosition, list[HookRegistration]]:
        """Get all registered hooks.

        Returns:
            Dictionary mapping positions to lists of hook registrations
        """
        with self._lock:
            return {
                pos: list(hooks)
                for pos, hooks in self._hooks.items()
            }

    def hook(
        self,
        position: HookPosition,
        priority: int = 100,
        name: Optional[str] = None,
    ) -> Callable:
        """Decorator for registering a function as a hook.

        Args:
            position: The hook position
            priority: Execution priority (optional)
            name: Optional name for the hook

        Returns:
            Decorator function

        Example:
            @registry.hook(HookPosition.ON_RUN_START)
            async def my_hook(context):
                return HookResult.continue_()
        """

        def decorator(
            func: Union[Callable[[Any], HookResult], Callable[[Any], Awaitable[HookResult]]]
        ) -> Union[Callable[[Any], HookResult], Callable[[Any], Awaitable[HookResult]]]:
            """Create and register a function hook."""

            class FunctionHook(BaseHook):
                """Hook wrapper for a function."""

                def __init__(
                    self,
                    hook_position: HookPosition,
                    hook_priority: int,
                ) -> None:
                    self.position = hook_position
                    self.priority = hook_priority
                    self._func = func

                async def execute(self, context: Any) -> HookResult:
                    """Execute the wrapped function."""
                    if asyncio.iscoroutinefunction(self._func):
                        result = await self._func(context)
                    else:
                        result = self._func(context)

                    # If function returns None or no return, continue
                    if result is None:
                        return HookResult.continue_()

                    # If already a HookResult, return it
                    if isinstance(result, HookResult):
                        return result

                    # Otherwise wrap in continue result
                    return HookResult.continue_()

            # Register the function hook
            hook_name = name or func.__name__
            self.register(
                FunctionHook(position, priority),
                position=position,
                priority=priority,
                name=hook_name,
            )

            return func

        return decorator

    def enable(self, name: str) -> bool:
        """Enable a hook by name.

        Args:
            name: The name of the hook to enable

        Returns:
            True if a hook was enabled, False otherwise
        """
        found = False
        with self._lock:
            for position in self._hooks:
                for r in self._hooks[position]:
                    if r.name == name:
                        r.enabled = True
                        found = True
        return found

    def disable(self, name: str) -> bool:
        """Disable a hook by name.

        Args:
            name: The name of the hook to disable

        Returns:
            True if a hook was disabled, False otherwise
        """
        found = False
        with self._lock:
            for position in self._hooks:
                for r in self._hooks[position]:
                    if r.name == name:
                        r.enabled = False
                        found = True
        return found

    def clear(self) -> None:
        """Clear all registered hooks."""
        with self._lock:
            self._hooks.clear()

    def count(self, position: Optional[HookPosition] = None) -> int:
        """Count registered hooks.

        Args:
            position: Optional position to count hooks for

        Returns:
            Number of registered hooks
        """
        with self._lock:
            if position:
                return len(self._hooks.get(position, []))
            return sum(len(hooks) for hooks in self._hooks.values())
