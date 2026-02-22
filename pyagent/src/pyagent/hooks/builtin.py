"""Built-in hook implementations."""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

from .base import BaseHook, HookPosition
from .context import HookContext
from .result import HookResult

logger = logging.getLogger(__name__)


class LoggingHook(BaseHook):
    """Hook for logging execution events.

    This hook logs information about the execution at various positions
    in the agent lifecycle.

    Attributes:
        level: Log level to use (DEBUG, INFO, WARNING, ERROR)
        include_args: Whether to include tool arguments in logs
        include_result: Whether to include tool results in logs
        include_message: Whether to include user messages in logs
    """

    def __init__(
        self,
        level: str = "DEBUG",
        include_args: bool = True,
        include_result: bool = False,
        include_message: bool = False,
    ) -> None:
        """Initialize the logging hook.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            include_args: Include tool arguments in logs
            include_result: Include tool results in logs
            include_message: Include user messages in logs
        """
        self.level = level.upper()
        self.include_args = include_args
        self.include_result = include_result
        self.include_message = include_message
        self.position = HookPosition.ON_RUN_START
        self.priority = 0  # Execute first for logging

    async def execute(self, context: HookContext) -> HookResult:
        """Execute the logging hook."""
        log_func = getattr(logger, self.level.lower(), logger.debug)

        log_data: Dict[str, Any] = {
            "position": context.position.value,
            "agent_id": context.agent_id,
            "iteration": context.iteration,
        }

        # Add tool information
        if context.tool_name:
            log_data["tool_name"] = context.tool_name

        if self.include_args and context.tool_arguments:
            log_data["tool_args"] = context.tool_arguments

        if self.include_result and context.tool_result:
            # Truncate long results
            result_str = str(context.tool_result)
            log_data["tool_result"] = (
                result_str[:500] + "..." if len(result_str) > 500 else result_str
            )

        # Add message information
        if self.include_message and context.message:
            msg_str = str(context.message)
            log_data["message"] = (
                msg_str[:200] + "..." if len(msg_str) > 200 else msg_str
            )

        # Add error information
        if context.error:
            log_data["error"] = str(context.error)

        log_func(f"[Hook] {log_data}")
        return HookResult.continue_()


class TimingHook(BaseHook):
    """Hook for timing execution duration.

    This hook records the start time at ON_RUN_START and logs the
    total duration at ON_RUN_END.

    Attributes:
        log_level: Log level for timing messages
        warn_threshold_ms: Threshold in ms to log warnings for slow operations
    """

    def __init__(
        self,
        log_level: str = "INFO",
        warn_threshold_ms: float = 5000.0,
    ) -> None:
        """Initialize the timing hook.

        Args:
            log_level: Log level for timing messages
            warn_threshold_ms: Threshold in ms to warn about slow operations
        """
        self.log_level = log_level.upper()
        self.warn_threshold_ms = warn_threshold_ms
        self.position = HookPosition.ON_RUN_START
        self.priority = 1  # Execute early
        self._start_times: Dict[str, float] = {}
        self._iteration_times: Dict[str, Dict[int, float]] = {}

    async def execute(self, context: HookContext) -> HookResult:
        """Execute the timing hook."""
        agent_id = context.agent_id
        key = f"{agent_id}:run"

        if context.position == HookPosition.ON_RUN_START:
            # Record start time
            self._start_times[key] = time.perf_counter()
            self._iteration_times[agent_id] = {}

        elif context.position == HookPosition.ON_ITERATION_START:
            # Record iteration start time
            if agent_id not in self._iteration_times:
                self._iteration_times[agent_id] = {}
            self._iteration_times[agent_id][context.iteration] = time.perf_counter()

        elif context.position == HookPosition.ON_ITERATION_END:
            # Log iteration duration
            if agent_id in self._iteration_times:
                iter_start = self._iteration_times[agent_id].get(context.iteration)
                if iter_start:
                    duration_ms = (time.perf_counter() - iter_start) * 1000
                    self._log_timing(
                        f"Iteration {context.iteration} took {duration_ms:.2f}ms",
                        duration_ms,
                    )

        elif context.position == HookPosition.ON_RUN_END:
            # Log total duration
            if key in self._start_times:
                duration_ms = (time.perf_counter() - self._start_times[key]) * 1000
                self._log_timing(
                    f"Agent {agent_id} run took {duration_ms:.2f}ms",
                    duration_ms,
                )
                # Clean up
                self._start_times.pop(key, None)
                self._iteration_times.pop(agent_id, None)

        return HookResult.continue_()

    def _log_timing(self, message: str, duration_ms: float) -> None:
        """Log timing message with appropriate level."""
        if duration_ms > self.warn_threshold_ms:
            logger.warning(f"[Timing] {message} (SLOW!)")
        else:
            log_func = getattr(logger, self.log_level.lower(), logger.info)
            log_func(f"[Timing] {message}")

    def get_durations(self) -> Dict[str, float]:
        """Get current run durations (for active runs).

        Returns:
            Dictionary of run keys to durations in ms
        """
        now = time.perf_counter()
        return {
            key: (now - start) * 1000
            for key, start in self._start_times.items()
        }


class ErrorHandlingHook(BaseHook):
    """Hook for handling errors during execution.

    This hook provides configurable error handling with retry support
    and custom error handlers.

    Attributes:
        handlers: Custom error handlers by exception type
        max_retries: Maximum number of retries
        retry_delay: Base delay between retries in seconds
    """

    def __init__(
        self,
        handlers: Optional[Dict[type, Callable]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize the error handling hook.

        Args:
            handlers: Dict mapping exception types to handler functions
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries (can be exponential)
        """
        self.position = HookPosition.ON_ERROR
        self.priority = 0  # Execute first for error handling
        self.handlers = handlers or {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._retry_count: Dict[str, int] = {}

    async def execute(self, context: HookContext) -> HookResult:
        """Execute error handling logic."""
        if not context.error:
            return HookResult.continue_()

        error = context.error
        error_type = type(error)
        key = f"{context.agent_id}:{context.iteration}"

        # Log the error
        logger.error(
            f"[ErrorHook] Error in {context.agent_id} "
            f"at iteration {context.iteration}: {error}"
        )

        # Find matching handler
        for exc_type, handler in self.handlers.items():
            if isinstance(error, exc_type):
                try:
                    result = await self._call_handler(handler, error, context)
                    if result:
                        return result
                except Exception as e:
                    logger.error(f"[ErrorHook] Handler failed: {e}")

        # Default behavior: retry with exponential backoff
        current_retry = self._retry_count.get(key, 0)
        if current_retry < self.max_retries:
            self._retry_count[key] = current_retry + 1
            delay = self.retry_delay * (2 ** current_retry)  # Exponential backoff
            logger.info(f"[ErrorHook] Retrying (attempt {current_retry + 1}) after {delay}s")
            return HookResult.retry(after_seconds=delay)

        # Max retries exceeded
        logger.error(f"[ErrorHook] Max retries ({self.max_retries}) exceeded")
        self._retry_count.pop(key, None)
        return HookResult.abort(f"Max retries exceeded: {error}")

    async def _call_handler(
        self,
        handler: Callable,
        error: Exception,
        context: HookContext,
    ) -> Optional[HookResult]:
        """Call an error handler and return its result."""
        result = handler(error, context)
        if asyncio.iscoroutine(result):
            result = await result

        if isinstance(result, HookResult):
            return result
        return None

    def reset_retries(self, agent_id: Optional[str] = None) -> None:
        """Reset retry counts.

        Args:
            agent_id: Specific agent to reset, or None for all
        """
        if agent_id:
            keys_to_remove = [k for k in self._retry_count if k.startswith(agent_id)]
            for key in keys_to_remove:
                self._retry_count.pop(key)
        else:
            self._retry_count.clear()


class MetricsHook(BaseHook):
    """Hook for collecting Prometheus-style metrics.

    This hook collects various metrics about agent execution including
    tool calls, LLM calls, errors, and timing.

    Attributes:
        prefix: Metric name prefix
    """

    def __init__(self, prefix: str = "pyagent") -> None:
        """Initialize the metrics hook.

        Args:
            prefix: Prefix for metric names
        """
        self.position = HookPosition.ON_TOOL_RESULT
        self.priority = 100
        self.prefix = prefix
        self._metrics: Dict[str, list] = {}
        self._counters: Dict[str, int] = {}

    async def execute(self, context: HookContext) -> HookResult:
        """Collect metrics based on hook position."""
        pos = context.position

        if pos == HookPosition.ON_TOOL_RESULT:
            self._record_tool_call(context)
        elif pos == HookPosition.ON_LLM_RESPONSE:
            self._record_llm_call(context)
        elif pos == HookPosition.ON_ERROR:
            self._record_error(context)
        elif pos == HookPosition.ON_RUN_END:
            self._record_run_complete(context)

        return HookResult.continue_()

    def _record_tool_call(self, context: HookContext) -> None:
        """Record a tool call metric."""
        if context.tool_name:
            metric_name = f"{self.prefix}_tool_calls_total"
            self._increment_counter(f"{metric_name}:{context.tool_name}")

            # Record detailed metric
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []

            self._metrics[metric_name].append({
                "tool": context.tool_name,
                "agent": context.agent_id,
                "timestamp": context.timestamp.isoformat(),
                "success": context.error is None,
                "iteration": context.iteration,
            })

    def _record_llm_call(self, context: HookContext) -> None:
        """Record an LLM call metric."""
        metric_name = f"{self.prefix}_llm_calls_total"
        self._increment_counter(metric_name)

        if metric_name not in self._metrics:
            self._metrics[metric_name] = []

        self._metrics[metric_name].append({
            "agent": context.agent_id,
            "timestamp": context.timestamp.isoformat(),
            "iteration": context.iteration,
        })

    def _record_error(self, context: HookContext) -> None:
        """Record an error metric."""
        if context.error:
            error_type = type(context.error).__name__
            metric_name = f"{self.prefix}_errors_total"
            self._increment_counter(f"{metric_name}:{error_type}")

    def _record_run_complete(self, context: HookContext) -> None:
        """Record run completion."""
        metric_name = f"{self.prefix}_runs_total"
        self._increment_counter(metric_name)

    def _increment_counter(self, name: str) -> None:
        """Increment a counter metric."""
        self._counters[name] = self._counters.get(name, 0) + 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "counters": self._counters.copy(),
            "detailed": self._metrics.copy(),
        }

    def get_counter(self, name: str) -> int:
        """Get a specific counter value.

        Args:
            name: Counter name

        Returns:
            Counter value
        """
        return self._counters.get(name, 0)

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._counters.clear()


class RateLimitHook(BaseHook):
    """Hook for rate limiting LLM requests.

    This hook implements a sliding window rate limiter to control
    the frequency of LLM API calls.

    Attributes:
        max_requests: Maximum requests allowed in the window
        window_seconds: Time window in seconds
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
    ) -> None:
        """Initialize the rate limit hook.

        Args:
            max_requests: Maximum requests in the window
            window_seconds: Window duration in seconds
        """
        self.position = HookPosition.ON_LLM_CALL
        self.priority = 0  # Execute first to enforce rate limit
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: list[float] = []
        self._lock = asyncio.Lock()

    async def execute(self, context: HookContext) -> HookResult:
        """Check and enforce rate limit."""
        async with self._lock:
            now = time.time()

            # Remove expired requests from window
            self._requests = [
                t for t in self._requests
                if now - t < self.window_seconds
            ]

            # Check if limit exceeded
            if len(self._requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self._requests)
                wait_time = self.window_seconds - (now - oldest_request)

                logger.warning(
                    f"[RateLimit] Rate limit exceeded. "
                    f"Wait {wait_time:.2f}s before next request."
                )

                return HookResult.retry(after_seconds=wait_time)

            # Record this request
            self._requests.append(now)

            logger.debug(
                f"[RateLimit] Request allowed. "
                f"{len(self._requests)}/{self.max_requests} in window."
            )

            return HookResult.continue_()

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status.

        Returns:
            Status dictionary with current usage
        """
        now = time.time()
        active_requests = [
            t for t in self._requests
            if now - t < self.window_seconds
        ]

        return {
            "current_requests": len(active_requests),
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "remaining": self.max_requests - len(active_requests),
        }

    def reset(self) -> None:
        """Reset the rate limiter."""
        self._requests.clear()
