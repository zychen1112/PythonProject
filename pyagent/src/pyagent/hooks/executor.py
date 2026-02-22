"""Hook execution engine."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .base import HookAction, HookPosition
from .result import HookResult

if TYPE_CHECKING:
    from .context import HookContext
    from .registry import HookRegistry

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStats:
    """Statistics for hook executions.

    Attributes:
        total_executions: Total number of execute() calls
        successful: Number of successful hook executions
        failed: Number of failed hook executions
        total_duration_ms: Total execution time in milliseconds
        by_position: Statistics broken down by position
    """

    total_executions: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration_ms: float = 0.0
    by_position: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def average_duration_ms(self) -> float:
        """Calculate average execution duration."""
        if self.total_executions == 0:
            return 0.0
        return self.total_duration_ms / self.total_executions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_executions": self.total_executions,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "total_duration_ms": self.total_duration_ms,
            "average_duration_ms": self.average_duration_ms,
            "by_position": self.by_position,
        }


class HookAbortError(Exception):
    """Exception raised when a hook aborts execution."""

    def __init__(self, reason: Optional[str] = None) -> None:
        self.reason = reason
        super().__init__(reason or "Execution aborted by hook")


class HookSkipError(Exception):
    """Exception raised when a hook skips the current operation."""

    def __init__(self, reason: Optional[str] = None) -> None:
        self.reason = reason
        super().__init__(reason or "Operation skipped by hook")


class HookRetryError(Exception):
    """Exception raised when a hook requests a retry."""

    def __init__(self, after_seconds: float = 0, reason: Optional[str] = None) -> None:
        self.after_seconds = after_seconds
        self.reason = reason
        super().__init__(reason or f"Retry requested after {after_seconds}s")


class HookExecutor:
    """Engine for executing hooks.

    This class handles the execution of hooks at each position,
    including error handling, statistics tracking, and action handling.

    Attributes:
        registry: The hook registry to use
        stats: Execution statistics
        stop_on_error: Whether to stop execution on hook errors
    """

    def __init__(
        self,
        registry: "HookRegistry",
        stop_on_error: bool = False,
    ) -> None:
        """Initialize the hook executor.

        Args:
            registry: The hook registry to use
            stop_on_error: Whether to stop execution on hook errors
        """
        self.registry = registry
        self.stats = ExecutionStats()
        self.stop_on_error = stop_on_error

    async def execute(
        self,
        position: HookPosition,
        context: "HookContext",
    ) -> HookResult:
        """Execute all hooks at a given position.

        Args:
            position: The hook position to execute
            context: The execution context

        Returns:
            The final HookResult (CONTINUE if all hooks pass)

        Raises:
            HookAbortError: If a hook returns ABORT action
            HookSkipError: If a hook returns SKIP action
            HookRetryError: If a hook returns RETRY action
        """
        registrations = self.registry.get_hooks(position)

        if not registrations:
            return HookResult.continue_()

        start_time = time.perf_counter()
        position_stats = self._get_or_create_position_stats(position)

        for registration in registrations:
            hook_start = time.perf_counter()

            try:
                result = await registration.hook.execute(context)

                # Track success
                self.stats.successful += 1
                hook_duration = (time.perf_counter() - hook_start) * 1000

                logger.debug(
                    f"Hook {registration.name} executed in {hook_duration:.2f}ms "
                    f"with action {result.action.value}"
                )

                # Handle different actions
                if result.action == HookAction.ABORT:
                    self._record_execution(start_time, position_stats)
                    raise HookAbortError(result.message)

                if result.action == HookAction.SKIP:
                    self.stats.skipped += 1
                    self._record_execution(start_time, position_stats)
                    raise HookSkipError(result.message)

                if result.action == HookAction.RETRY:
                    after_seconds = result.data.get("after_seconds", 0)
                    self._record_execution(start_time, position_stats)
                    raise HookRetryError(after_seconds, result.message)

                if result.action == HookAction.MODIFY:
                    self._apply_modifications(context, result.modified_context)

            except (HookAbortError, HookSkipError, HookRetryError):
                # Re-raise control flow exceptions
                raise

            except Exception as e:
                # Track failure
                self.stats.failed += 1
                hook_duration = (time.perf_counter() - hook_start) * 1000

                logger.error(
                    f"Hook {registration.name} failed after {hook_duration:.2f}ms: {e}"
                )

                # Store error in context for other hooks to see
                context.metadata["last_hook_error"] = e
                context.metadata["last_hook_error_name"] = registration.name

                if self.stop_on_error:
                    self._record_execution(start_time, position_stats)
                    raise

        self._record_execution(start_time, position_stats)
        return HookResult.continue_()

    async def execute_safe(
        self,
        position: HookPosition,
        context: "HookContext",
    ) -> HookResult:
        """Execute hooks without raising control flow exceptions.

        This is a convenience method that catches HookAbortError, HookSkipError,
        and HookRetryError and returns them as HookResult objects.

        Args:
            position: The hook position to execute
            context: The execution context

        Returns:
            The HookResult (never raises)
        """
        try:
            return await self.execute(position, context)
        except HookAbortError as e:
            return HookResult.abort(e.reason)
        except HookSkipError as e:
            return HookResult.skip(e.reason)
        except HookRetryError as e:
            return HookResult.retry(e.after_seconds)
        except Exception as e:
            logger.error(f"Unexpected error executing hooks: {e}")
            return HookResult.abort(str(e))

    def _get_or_create_position_stats(
        self,
        position: HookPosition,
    ) -> dict[str, Any]:
        """Get or create statistics for a position."""
        pos_key = position.value
        if pos_key not in self.stats.by_position:
            self.stats.by_position[pos_key] = {
                "count": 0,
                "duration_ms": 0.0,
                "successful": 0,
                "failed": 0,
            }
        return self.stats.by_position[pos_key]

    def _record_execution(
        self,
        start_time: float,
        position_stats: dict[str, Any],
    ) -> None:
        """Record execution statistics."""
        duration_ms = (time.perf_counter() - start_time) * 1000
        self.stats.total_executions += 1
        self.stats.total_duration_ms += duration_ms
        position_stats["count"] += 1
        position_stats["duration_ms"] += duration_ms

    def _apply_modifications(
        self,
        context: "HookContext",
        modifications: Optional[dict[str, Any]],
    ) -> None:
        """Apply context modifications from a hook."""
        if modifications:
            for key, value in modifications.items():
                context.set_modification(key, value)

    def get_stats(self) -> ExecutionStats:
        """Get execution statistics.

        Returns:
            Copy of the execution statistics
        """
        return self.stats

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.stats = ExecutionStats()


# Type alias for convenience
Any = object  # For the stats dict typing
