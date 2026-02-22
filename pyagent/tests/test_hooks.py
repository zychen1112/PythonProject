"""Tests for the hooks and lifecycle system."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from pyagent.hooks import (
    BaseHook,
    HookAction,
    HookContext,
    HookExecutor,
    HookPosition,
    HookRegistry,
    HookResult,
    HookAbortError,
    HookSkipError,
    HookRetryError,
    LoggingHook,
    TimingHook,
    ErrorHandlingHook,
    MetricsHook,
    RateLimitHook,
)


class TestHookPosition:
    """Tests for HookPosition enum."""

    def test_all_positions_defined(self):
        """Test that all expected hook positions are defined."""
        positions = list(HookPosition)
        assert len(positions) >= 10

    def test_position_values(self):
        """Test that position values are correct."""
        assert HookPosition.ON_RUN_START.value == "on_run_start"
        assert HookPosition.ON_RUN_END.value == "on_run_end"
        assert HookPosition.ON_TOOL_CALL.value == "on_tool_call"
        assert HookPosition.ON_ERROR.value == "on_error"

    def test_position_lookup(self):
        """Test looking up positions by value."""
        position = HookPosition("on_init")
        assert position == HookPosition.ON_INIT


class TestHookAction:
    """Tests for HookAction enum."""

    def test_all_actions_defined(self):
        """Test that all expected hook actions are defined."""
        actions = list(HookAction)
        assert len(actions) == 5

    def test_action_values(self):
        """Test that action values are correct."""
        assert HookAction.CONTINUE.value == "continue"
        assert HookAction.SKIP.value == "skip"
        assert HookAction.ABORT.value == "abort"
        assert HookAction.RETRY.value == "retry"
        assert HookAction.MODIFY.value == "modify"


class TestHookContext:
    """Tests for HookContext."""

    def test_context_creation(self):
        """Test creating a hook context."""
        ctx = HookContext(
            agent_id="test_agent",
            position=HookPosition.ON_RUN_START,
        )
        assert ctx.agent_id == "test_agent"
        assert ctx.position == HookPosition.ON_RUN_START
        assert isinstance(ctx.timestamp, datetime)

    def test_context_with_all_fields(self):
        """Test context with all fields populated."""
        ctx = HookContext(
            agent_id="test_agent",
            position=HookPosition.ON_TOOL_CALL,
            message="Hello",
            tool_name="test_tool",
            tool_arguments={"arg": "value"},
            tool_result="result",
            error=ValueError("test"),
            iteration=1,
        )
        assert ctx.message == "Hello"
        assert ctx.tool_name == "test_tool"
        assert ctx.tool_arguments == {"arg": "value"}
        assert ctx.tool_result == "result"
        assert isinstance(ctx.error, ValueError)
        assert ctx.iteration == 1

    def test_modifications(self):
        """Test context modifications."""
        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_TOOL_CALL,
        )

        ctx.set_modification("key1", "value1")
        ctx.set_modification("key2", {"nested": "data"})

        assert ctx.get_modification("key1") == "value1"
        assert ctx.get_modification("key2") == {"nested": "data"}
        assert ctx.get_modification("nonexistent") is None
        assert ctx.get_modification("nonexistent", "default") == "default"

    def test_clear_modifications(self):
        """Test clearing modifications."""
        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_TOOL_CALL,
        )
        ctx.set_modification("key", "value")
        ctx.clear_modifications()
        assert ctx.get_modification("key") is None

    def test_to_dict(self):
        """Test converting context to dictionary."""
        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_RUN_START,
            message="Hello",
            iteration=1,
        )
        d = ctx.to_dict()

        assert d["agent_id"] == "test"
        assert d["position"] == "on_run_start"
        assert d["message"] == "Hello"
        assert d["iteration"] == 1
        assert "timestamp" in d


class TestHookResult:
    """Tests for HookResult."""

    def test_continue_factory(self):
        """Test continue_ factory method."""
        result = HookResult.continue_()
        assert result.action == HookAction.CONTINUE
        assert result.message is None

    def test_skip_factory(self):
        """Test skip factory method."""
        result = HookResult.skip("Skipping for reason")
        assert result.action == HookAction.SKIP
        assert result.message == "Skipping for reason"

    def test_abort_factory(self):
        """Test abort factory method."""
        result = HookResult.abort("Aborting due to error")
        assert result.action == HookAction.ABORT
        assert result.message == "Aborting due to error"

    def test_retry_factory(self):
        """Test retry factory method."""
        result = HookResult.retry(after_seconds=2.5)
        assert result.action == HookAction.RETRY
        assert result.data["after_seconds"] == 2.5

    def test_modify_factory(self):
        """Test modify factory method."""
        result = HookResult.modify({"tool_name": "new_tool"})
        assert result.action == HookAction.MODIFY
        assert result.modified_context == {"tool_name": "new_tool"}

    def test_is_methods(self):
        """Test is_* helper methods."""
        assert HookResult.continue_().is_continue()
        assert HookResult.skip().is_skip()
        assert HookResult.abort().is_abort()
        assert HookResult.retry().is_retry()
        assert HookResult.modify({}).is_modify()

    def test_repr(self):
        """Test string representation."""
        result = HookResult.abort("test")
        assert "abort" in repr(result)
        assert "test" in repr(result)


class TestHookRegistry:
    """Tests for HookRegistry."""

    def test_register_hook(self):
        """Test registering a hook."""
        registry = HookRegistry()
        hook = LoggingHook()
        name = registry.register(hook)

        hooks = registry.get_hooks(HookPosition.ON_RUN_START)
        assert len(hooks) == 1
        assert hooks[0].hook == hook

    def test_unregister_hook(self):
        """Test unregistering a hook."""
        registry = HookRegistry()
        hook = LoggingHook()
        name = registry.register(hook)

        result = registry.unregister(name)
        assert result is True

        hooks = registry.get_hooks(HookPosition.ON_RUN_START)
        assert len(hooks) == 0

    def test_priority_sorting(self):
        """Test that hooks are sorted by priority."""
        registry = HookRegistry()

        class PriorityHook(BaseHook):
            def __init__(self, priority_val):
                self.position = HookPosition.ON_RUN_START
                self.priority = priority_val

            async def execute(self, context):
                return HookResult.continue_()

        hook1 = PriorityHook(100)
        hook2 = PriorityHook(50)
        hook3 = PriorityHook(75)

        registry.register(hook1, name="hook1")
        registry.register(hook2, name="hook2")
        registry.register(hook3, name="hook3")

        hooks = registry.get_hooks(HookPosition.ON_RUN_START)
        # Should be sorted: hook2 (50), hook3 (75), hook1 (100)
        assert hooks[0].name == "hook2"
        assert hooks[1].name == "hook3"
        assert hooks[2].name == "hook1"

    def test_decorator_registration(self):
        """Test decorator-based hook registration."""
        registry = HookRegistry()
        call_count = 0

        @registry.hook(HookPosition.ON_RUN_START, priority=50)
        async def my_hook(ctx):
            nonlocal call_count
            call_count += 1
            return HookResult.continue_()

        hooks = registry.get_hooks(HookPosition.ON_RUN_START)
        assert len(hooks) == 1
        assert hooks[0].name == "my_hook"

    def test_enable_disable(self):
        """Test enabling and disabling hooks."""
        registry = HookRegistry()
        hook = LoggingHook()
        name = registry.register(hook)

        # Disable
        registry.disable(name)
        hooks = registry.get_hooks(HookPosition.ON_RUN_START)
        assert len(hooks) == 0

        # Enable
        registry.enable(name)
        hooks = registry.get_hooks(HookPosition.ON_RUN_START)
        assert len(hooks) == 1

    def test_clear(self):
        """Test clearing all hooks."""
        registry = HookRegistry()
        registry.register(LoggingHook())
        registry.register(TimingHook())

        registry.clear()
        assert registry.count() == 0

    def test_count(self):
        """Test counting hooks."""
        registry = HookRegistry()
        assert registry.count() == 0

        registry.register(LoggingHook())
        assert registry.count() == 1

        registry.register(TimingHook())
        assert registry.count() == 2


class TestHookExecutor:
    """Tests for HookExecutor."""

    @pytest.mark.asyncio
    async def test_execute_no_hooks(self):
        """Test executing with no hooks registered."""
        registry = HookRegistry()
        executor = HookExecutor(registry)

        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_RUN_START,
        )
        result = await executor.execute(HookPosition.ON_RUN_START, ctx)

        assert result.action == HookAction.CONTINUE

    @pytest.mark.asyncio
    async def test_execute_with_hooks(self):
        """Test executing with hooks registered."""
        registry = HookRegistry()
        executor = HookExecutor(registry)

        call_order = []

        class TestHook(BaseHook):
            def __init__(self, name, priority_val):
                self.position = HookPosition.ON_RUN_START
                self.priority = priority_val
                self._name = name

            async def execute(self, context):
                call_order.append(self._name)
                return HookResult.continue_()

        registry.register(TestHook("first", 10), name="first")
        registry.register(TestHook("second", 20), name="second")

        ctx = HookContext(agent_id="test", position=HookPosition.ON_RUN_START)
        await executor.execute(HookPosition.ON_RUN_START, ctx)

        assert call_order == ["first", "second"]

    @pytest.mark.asyncio
    async def test_execute_with_abort(self):
        """Test that ABORT stops execution."""
        registry = HookRegistry()
        executor = HookExecutor(registry)

        class AbortHook(BaseHook):
            def __init__(self):
                self.position = HookPosition.ON_RUN_START
                self.priority = 10

            async def execute(self, context):
                return HookResult.abort("Test abort")

        class NeverCalledHook(BaseHook):
            def __init__(self):
                self.position = HookPosition.ON_RUN_START
                self.priority = 20

            async def execute(self, context):
                raise AssertionError("Should not be called")

        registry.register(AbortHook(), name="abort")
        registry.register(NeverCalledHook(), name="never")

        ctx = HookContext(agent_id="test", position=HookPosition.ON_RUN_START)

        with pytest.raises(HookAbortError) as exc_info:
            await executor.execute(HookPosition.ON_RUN_START, ctx)

        assert "Test abort" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_with_skip(self):
        """Test that SKIP stops execution."""
        registry = HookRegistry()
        executor = HookExecutor(registry)

        class SkipHook(BaseHook):
            def __init__(self):
                self.position = HookPosition.ON_RUN_START
                self.priority = 10

            async def execute(self, context):
                return HookResult.skip("Test skip")

        registry.register(SkipHook(), name="skip")

        ctx = HookContext(agent_id="test", position=HookPosition.ON_RUN_START)

        with pytest.raises(HookSkipError):
            await executor.execute(HookPosition.ON_RUN_START, ctx)

    @pytest.mark.asyncio
    async def test_execute_with_retry(self):
        """Test that RETRY returns retry info."""
        registry = HookRegistry()
        executor = HookExecutor(registry)

        class RetryHook(BaseHook):
            def __init__(self):
                self.position = HookPosition.ON_RUN_START
                self.priority = 10

            async def execute(self, context):
                return HookResult.retry(after_seconds=5.0)

        registry.register(RetryHook(), name="retry")

        ctx = HookContext(agent_id="test", position=HookPosition.ON_RUN_START)

        with pytest.raises(HookRetryError) as exc_info:
            await executor.execute(HookPosition.ON_RUN_START, ctx)

        assert exc_info.value.after_seconds == 5.0

    @pytest.mark.asyncio
    async def test_execute_with_modify(self):
        """Test that MODIFY applies modifications."""
        registry = HookRegistry()
        executor = HookExecutor(registry)

        class ModifyHook(BaseHook):
            def __init__(self):
                self.position = HookPosition.ON_RUN_START
                self.priority = 10

            async def execute(self, context):
                return HookResult.modify({"custom_key": "custom_value"})

        registry.register(ModifyHook(), name="modify")

        ctx = HookContext(agent_id="test", position=HookPosition.ON_RUN_START)
        await executor.execute(HookPosition.ON_RUN_START, ctx)

        assert ctx.get_modification("custom_key") == "custom_value"

    @pytest.mark.asyncio
    async def test_execute_safe(self):
        """Test execute_safe doesn't raise exceptions."""
        registry = HookRegistry()
        executor = HookExecutor(registry)

        class AbortHook(BaseHook):
            def __init__(self):
                self.position = HookPosition.ON_RUN_START
                self.priority = 10

            async def execute(self, context):
                return HookResult.abort("Test abort")

        registry.register(AbortHook(), name="abort")

        ctx = HookContext(agent_id="test", position=HookPosition.ON_RUN_START)
        result = await executor.execute_safe(HookPosition.ON_RUN_START, ctx)

        assert result.action == HookAction.ABORT
        assert result.message == "Test abort"

    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test that execution statistics are tracked."""
        registry = HookRegistry()
        executor = HookExecutor(registry)

        class CountingHook(BaseHook):
            def __init__(self):
                self.position = HookPosition.ON_RUN_START
                self.priority = 10

            async def execute(self, context):
                return HookResult.continue_()

        registry.register(CountingHook(), name="count")

        ctx = HookContext(agent_id="test", position=HookPosition.ON_RUN_START)
        await executor.execute(HookPosition.ON_RUN_START, ctx)

        stats = executor.get_stats()
        assert stats.total_executions == 1
        assert stats.successful == 1
        assert stats.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in hook execution."""
        registry = HookRegistry()
        executor = HookExecutor(registry, stop_on_error=True)

        class ErrorHook(BaseHook):
            def __init__(self):
                self.position = HookPosition.ON_RUN_START
                self.priority = 10

            async def execute(self, context):
                raise ValueError("Test error")

        registry.register(ErrorHook(), name="error")

        ctx = HookContext(agent_id="test", position=HookPosition.ON_RUN_START)

        with pytest.raises(ValueError):
            await executor.execute(HookPosition.ON_RUN_START, ctx)

        stats = executor.get_stats()
        assert stats.failed == 1


class TestLoggingHook:
    """Tests for LoggingHook."""

    @pytest.mark.asyncio
    async def test_logging_hook_execute(self):
        """Test LoggingHook execution."""
        hook = LoggingHook(level="DEBUG", include_args=True)
        hook.position = HookPosition.ON_TOOL_CALL

        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_TOOL_CALL,
            tool_name="test_tool",
            tool_arguments={"arg": "value"},
        )

        result = await hook.execute(ctx)
        assert result.action == HookAction.CONTINUE


class TestTimingHook:
    """Tests for TimingHook."""

    @pytest.mark.asyncio
    async def test_timing_hook_start(self):
        """Test TimingHook at run start."""
        hook = TimingHook()

        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_RUN_START,
        )

        result = await hook.execute(ctx)
        assert result.action == HookAction.CONTINUE

    @pytest.mark.asyncio
    async def test_timing_hook_end(self):
        """Test TimingHook at run end."""
        hook = TimingHook()

        # First, start timing
        start_ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_RUN_START,
        )
        await hook.execute(start_ctx)

        # Then, end timing
        end_ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_RUN_END,
        )
        result = await hook.execute(end_ctx)
        assert result.action == HookAction.CONTINUE


class TestErrorHandlingHook:
    """Tests for ErrorHandlingHook."""

    @pytest.mark.asyncio
    async def test_error_handling_without_error(self):
        """Test ErrorHandlingHook when no error."""
        hook = ErrorHandlingHook()

        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_ERROR,
        )

        result = await hook.execute(ctx)
        assert result.action == HookAction.CONTINUE

    @pytest.mark.asyncio
    async def test_error_handling_with_error(self):
        """Test ErrorHandlingHook with error."""
        hook = ErrorHandlingHook(max_retries=2)

        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_ERROR,
            error=ValueError("Test error"),
            iteration=1,
        )

        # First retry
        result = await hook.execute(ctx)
        assert result.action == HookAction.RETRY

        # Second retry
        result = await hook.execute(ctx)
        assert result.action == HookAction.RETRY

        # Max retries exceeded
        result = await hook.execute(ctx)
        assert result.action == HookAction.ABORT

    @pytest.mark.asyncio
    async def test_custom_error_handler(self):
        """Test custom error handler."""
        handled_errors = []

        def handle_value_error(error, context):
            handled_errors.append(error)
            return HookResult.abort("Handled by custom handler")

        hook = ErrorHandlingHook(
            handlers={ValueError: handle_value_error},
            max_retries=0,
        )

        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_ERROR,
            error=ValueError("Test error"),
        )

        result = await hook.execute(ctx)
        assert result.action == HookAction.ABORT
        assert result.message == "Handled by custom handler"
        assert len(handled_errors) == 1


class TestMetricsHook:
    """Tests for MetricsHook."""

    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection."""
        hook = MetricsHook(prefix="test")

        ctx = HookContext(
            agent_id="test_agent",
            position=HookPosition.ON_TOOL_RESULT,
            tool_name="test_tool",
        )

        await hook.execute(ctx)

        metrics = hook.get_metrics()
        assert "counters" in metrics
        assert "test_tool_calls_total:test_tool" in metrics["counters"]
        assert metrics["counters"]["test_tool_calls_total:test_tool"] == 1

    @pytest.mark.asyncio
    async def test_metrics_reset(self):
        """Test metrics reset."""
        hook = MetricsHook()

        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_TOOL_RESULT,
            tool_name="tool",
        )

        await hook.execute(ctx)
        hook.reset()

        metrics = hook.get_metrics()
        assert len(metrics["counters"]) == 0


class TestRateLimitHook:
    """Tests for RateLimitHook."""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_requests(self):
        """Test that rate limiter allows requests under limit."""
        hook = RateLimitHook(max_requests=5, window_seconds=60)

        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_LLM_CALL,
        )

        # Should allow first 5 requests
        for _ in range(5):
            result = await hook.execute(ctx)
            assert result.action == HookAction.CONTINUE

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excess(self):
        """Test that rate limiter blocks excess requests."""
        hook = RateLimitHook(max_requests=2, window_seconds=60)

        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_LLM_CALL,
        )

        # Allow first 2
        await hook.execute(ctx)
        await hook.execute(ctx)

        # Third should be blocked
        result = await hook.execute(ctx)
        assert result.action == HookAction.RETRY

    @pytest.mark.asyncio
    async def test_rate_limit_status(self):
        """Test getting rate limit status."""
        hook = RateLimitHook(max_requests=5, window_seconds=60)

        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_LLM_CALL,
        )

        await hook.execute(ctx)
        status = hook.get_status()

        assert status["current_requests"] == 1
        assert status["max_requests"] == 5
        assert status["remaining"] == 4

    @pytest.mark.asyncio
    async def test_rate_limit_reset(self):
        """Test rate limiter reset."""
        hook = RateLimitHook(max_requests=2, window_seconds=60)

        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_LLM_CALL,
        )

        await hook.execute(ctx)
        await hook.execute(ctx)
        hook.reset()

        status = hook.get_status()
        assert status["current_requests"] == 0


class TestIntegration:
    """Integration tests for hooks with Agent."""

    @pytest.mark.asyncio
    async def test_hook_context_serialization(self):
        """Test that hook context can be serialized."""
        ctx = HookContext(
            agent_id="test",
            position=HookPosition.ON_RUN_START,
            message="Hello",
            tool_name="tool",
            tool_arguments={"a": 1},
            iteration=1,
            metadata={"key": "value"},
        )

        d = ctx.to_dict()

        # Should be JSON serializable
        import json
        json_str = json.dumps(d)
        assert json_str is not None

    @pytest.mark.asyncio
    async def test_multiple_hooks_same_position(self):
        """Test multiple hooks at the same position."""
        registry = HookRegistry()
        executor = HookExecutor(registry)

        execution_log = []

        class LogHook(BaseHook):
            def __init__(self, name, priority_val):
                self.position = HookPosition.ON_RUN_START
                self.priority = priority_val
                self._name = name

            async def execute(self, context):
                execution_log.append(self._name)
                return HookResult.continue_()

        registry.register(LogHook("third", 100), name="third")
        registry.register(LogHook("first", 10), name="first")
        registry.register(LogHook("second", 50), name="second")

        ctx = HookContext(agent_id="test", position=HookPosition.ON_RUN_START)
        await executor.execute(HookPosition.ON_RUN_START, ctx)

        assert execution_log == ["first", "second", "third"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
