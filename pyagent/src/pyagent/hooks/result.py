"""Hook execution result."""

from dataclasses import dataclass, field
from typing import Any, Optional

from .base import HookAction


@dataclass
class HookResult:
    """Result of hook execution.

    This class represents the outcome of a hook execution and tells
    the executor what action to take next.

    Attributes:
        action: The action to take after this hook
        message: Optional message describing the result
        data: Optional additional data
        modified_context: Context modifications to apply (for MODIFY action)

    Class Methods:
        continue_(): Create a CONTINUE result
        skip(): Create a SKIP result
        abort(): Create an ABORT result
        retry(): Create a RETRY result
        modify(): Create a MODIFY result
    """

    action: HookAction = HookAction.CONTINUE
    message: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    modified_context: Optional[dict[str, Any]] = None

    @classmethod
    def continue_(cls) -> "HookResult":
        """Create a result that continues normal execution.

        Returns:
            HookResult with CONTINUE action
        """
        return cls(action=HookAction.CONTINUE)

    @classmethod
    def skip(cls, reason: Optional[str] = None) -> "HookResult":
        """Create a result that skips the current operation.

        Args:
            reason: Optional reason for skipping

        Returns:
            HookResult with SKIP action
        """
        return cls(action=HookAction.SKIP, message=reason)

    @classmethod
    def abort(cls, reason: Optional[str] = None) -> "HookResult":
        """Create a result that aborts the entire execution.

        Args:
            reason: Optional reason for aborting

        Returns:
            HookResult with ABORT action
        """
        return cls(action=HookAction.ABORT, message=reason)

    @classmethod
    def retry(cls, after_seconds: float = 0) -> "HookResult":
        """Create a result that retries the current operation.

        Args:
            after_seconds: Seconds to wait before retrying

        Returns:
            HookResult with RETRY action
        """
        return cls(
            action=HookAction.RETRY,
            data={"after_seconds": after_seconds}
        )

    @classmethod
    def modify(cls, modifications: dict[str, Any]) -> "HookResult":
        """Create a result that modifies context data.

        Args:
            modifications: Dictionary of modifications to apply

        Returns:
            HookResult with MODIFY action
        """
        return cls(
            action=HookAction.MODIFY,
            modified_context=modifications
        )

    def is_continue(self) -> bool:
        """Check if action is CONTINUE."""
        return self.action == HookAction.CONTINUE

    def is_skip(self) -> bool:
        """Check if action is SKIP."""
        return self.action == HookAction.SKIP

    def is_abort(self) -> bool:
        """Check if action is ABORT."""
        return self.action == HookAction.ABORT

    def is_retry(self) -> bool:
        """Check if action is RETRY."""
        return self.action == HookAction.RETRY

    def is_modify(self) -> bool:
        """Check if action is MODIFY."""
        return self.action == HookAction.MODIFY

    def __repr__(self) -> str:
        parts = [f"HookResult(action={self.action.value}"]
        if self.message:
            parts.append(f", message={self.message!r}")
        if self.data:
            parts.append(f", data={self.data}")
        if self.modified_context:
            parts.append(f", modifications={self.modified_context}")
        parts.append(")")
        return "".join(parts)
