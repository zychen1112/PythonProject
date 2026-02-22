"""Hook base classes and enums."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import HookContext
    from .result import HookResult


class HookPosition(Enum):
    """Define hook execution positions in the Agent lifecycle.

    There are 10+ hook positions covering the entire agent execution flow:
    - Initialization and run lifecycle
    - Iteration lifecycle
    - LLM call lifecycle
    - Tool execution lifecycle
    - Error handling
    """

    # Agent lifecycle
    ON_INIT = "on_init"                     # Agent initialization
    ON_RUN_START = "on_run_start"           # Run method start
    ON_RUN_END = "on_run_end"               # Run method end

    # Iteration lifecycle
    ON_ITERATION_START = "on_iteration_start"  # Each iteration start
    ON_ITERATION_END = "on_iteration_end"      # Each iteration end

    # LLM lifecycle
    ON_LLM_CALL = "on_llm_call"             # Before LLM API call
    ON_LLM_RESPONSE = "on_llm_response"     # After LLM API response

    # Tool lifecycle
    ON_TOOL_CALL = "on_tool_call"           # Before tool execution
    ON_TOOL_RESULT = "on_tool_result"       # After tool execution

    # Message lifecycle
    ON_MESSAGE = "on_message"               # When message is added

    # Error handling
    ON_ERROR = "on_error"                   # When error occurs


class HookAction(Enum):
    """Actions that can be taken after hook execution.

    - CONTINUE: Continue normal execution
    - SKIP: Skip the current operation
    - ABORT: Abort the entire execution
    - RETRY: Retry the current operation
    - MODIFY: Modify context data and continue
    """

    CONTINUE = "continue"
    SKIP = "skip"
    ABORT = "abort"
    RETRY = "retry"
    MODIFY = "modify"


class BaseHook(ABC):
    """Abstract base class for all hooks.

    Hooks are used to inject custom logic at specific points in the
    Agent execution lifecycle. Each hook must specify its position
    and priority.

    Attributes:
        position: The hook position in the lifecycle
        priority: Execution priority (lower = earlier), default 100

    Example:
        class MyHook(BaseHook):
            position = HookPosition.ON_TOOL_CALL
            priority = 50

            async def execute(self, context: HookContext) -> HookResult:
                print(f"Tool called: {context.tool_name}")
                return HookResult.continue_()
    """

    position: HookPosition
    priority: int = 100

    @abstractmethod
    async def execute(self, context: "HookContext") -> "HookResult":
        """Execute the hook logic.

        Args:
            context: The hook execution context containing all relevant data

        Returns:
            HookResult indicating what action to take next
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(position={self.position.value}, priority={self.priority})"
