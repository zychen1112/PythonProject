"""Procedural Memory - Memory for skills and workflows."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class WorkflowStep(BaseModel):
    """A step in a workflow."""
    name: str
    action: str
    tool: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_output: Optional[str] = None


class Workflow(BaseModel):
    """A workflow definition."""
    id: str
    name: str
    description: str
    steps: list[WorkflowStep] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    success_rate: float = 0.0
    execution_count: int = 0


class ExecutionResult(BaseModel):
    """Result of a workflow execution."""
    workflow_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class ProceduralMemory:
    """Procedural memory for storing learned skills and workflows.

    Stores workflows that can be recalled and executed to accomplish tasks.
    Tracks success rates to improve workflow selection.
    """

    def __init__(self):
        """Initialize procedural memory."""
        self._workflows: dict[str, Workflow] = {}
        self._execution_history: dict[str, list[ExecutionResult]] = {}
        self._id_counter = 0

    def _next_id(self) -> str:
        """Generate next workflow ID."""
        self._id_counter += 1
        return f"wf_{self._id_counter}"

    async def learn(self, workflow: Workflow) -> str:
        """Learn a new workflow.

        Args:
            workflow: Workflow to learn

        Returns:
            Workflow ID
        """
        if not workflow.id:
            workflow.id = self._next_id()

        self._workflows[workflow.id] = workflow
        self._execution_history[workflow.id] = []

        return workflow.id

    async def recall(
        self,
        task_description: str,
        tags: Optional[list[str]] = None,
    ) -> Optional[Workflow]:
        """Recall a workflow for a task.

        Args:
            task_description: Description of the task
            tags: Optional tags to filter by

        Returns:
            Best matching workflow or None
        """
        candidates = []

        for workflow in self._workflows.values():
            # Filter by tags if provided
            if tags:
                if not any(tag in workflow.tags for tag in tags):
                    continue

            # Simple keyword matching
            desc_lower = task_description.lower()
            name_lower = workflow.name.lower()
            wf_desc_lower = workflow.description.lower()

            score = 0
            for word in desc_lower.split():
                if word in name_lower:
                    score += 2
                if word in wf_desc_lower:
                    score += 1

            if score > 0:
                # Boost by success rate
                score *= (1 + workflow.success_rate)
                candidates.append((workflow, score))

        if not candidates:
            return None

        # Sort by score and return best
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    async def record_execution(
        self,
        workflow_id: str,
        result: ExecutionResult,
    ) -> None:
        """Record a workflow execution result.

        Args:
            workflow_id: Workflow ID
            result: Execution result
        """
        if workflow_id not in self._workflows:
            return

        # Store result
        if workflow_id not in self._execution_history:
            self._execution_history[workflow_id] = []
        self._execution_history[workflow_id].append(result)

        # Update workflow stats
        workflow = self._workflows[workflow_id]
        workflow.execution_count += 1

        # Recalculate success rate
        history = self._execution_history[workflow_id]
        successes = sum(1 for r in history if r.success)
        workflow.success_rate = successes / len(history) if history else 0.0

    async def get_best_workflow(self, task: str) -> Optional[Workflow]:
        """Get the best workflow for a task based on success rate.

        Args:
            task: Task description

        Returns:
            Best workflow or None
        """
        workflow = await self.recall(task)
        if not workflow:
            return None

        # Check if there are better alternatives
        candidates = [workflow]
        desc_lower = task.lower()

        for wf in self._workflows.values():
            if wf.id == workflow.id:
                continue

            # Check if relevant
            if any(word in wf.name.lower() or word in wf.description.lower() for word in desc_lower.split()):
                if wf.success_rate > workflow.success_rate and wf.execution_count >= 3:
                    candidates.append(wf)

        # Return the one with best success rate
        candidates.sort(key=lambda x: x.success_rate, reverse=True)
        return candidates[0]

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID.

        Args:
            workflow_id: Workflow ID

        Returns:
            Workflow or None
        """
        return self._workflows.get(workflow_id)

    def get_all_workflows(self) -> list[Workflow]:
        """Get all workflows."""
        return list(self._workflows.values())

    def get_by_tag(self, tag: str) -> list[Workflow]:
        """Get workflows by tag."""
        return [wf for wf in self._workflows.values() if tag in wf.tags]

    async def forget(self, workflow_id: str) -> bool:
        """Forget a workflow.

        Args:
            workflow_id: Workflow ID to forget

        Returns:
            True if forgotten
        """
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            self._execution_history.pop(workflow_id, None)
            return True
        return False

    async def clear(self) -> None:
        """Clear all workflows."""
        self._workflows.clear()
        self._execution_history.clear()

    def count(self) -> int:
        """Get the number of workflows."""
        return len(self._workflows)

    def get_execution_history(self, workflow_id: str) -> list[ExecutionResult]:
        """Get execution history for a workflow."""
        return self._execution_history.get(workflow_id, [])
