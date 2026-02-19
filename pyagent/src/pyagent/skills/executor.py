"""
Skill executor - executes skills and manages their context.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyagent.skills.skill import Skill
from pyagent.utils.logging import get_logger

if TYPE_CHECKING:
    from pyagent.core.agent import Agent

logger = get_logger(__name__)


class SkillExecutionContext:
    """Context for skill execution."""

    def __init__(
        self,
        skill: Skill,
        agent: Agent,
        arguments: str | None = None
    ):
        self.skill = skill
        self.agent = agent
        self.arguments = arguments
        self.variables: dict[str, Any] = {}

    def substitute_variables(self, content: str) -> str:
        """Substitute variables in content."""
        result = content

        # Substitute arguments
        if self.arguments:
            result = result.replace("$ARGUMENTS", self.arguments)

            # Handle indexed arguments
            parts = self.arguments.split()
            for i, part in enumerate(parts):
                result = result.replace(f"$ARGUMENTS[{i}]", part)
                result = result.replace(f"${i}", part)

        # Substitute session variables
        for key, value in self.variables.items():
            result = result.replace(f"${{{key}}}", str(value))

        return result


class SkillExecutor:
    """
    Executes skills within an agent context.
    """

    def __init__(self, agent: Agent):
        self.agent = agent

    async def execute(
        self,
        skill: Skill,
        arguments: str | None = None
    ) -> str:
        """
        Execute a skill.

        Args:
            skill: Skill to execute
            arguments: Optional arguments for the skill

        Returns:
            Skill execution result
        """
        logger.info(f"Executing skill: {skill.name}")

        context = SkillExecutionContext(skill, self.agent, arguments)

        # Process dynamic context (!`command` syntax)
        instructions = await self._process_dynamic_context(
            skill.instructions,
            context
        )

        # Substitute variables
        instructions = context.substitute_variables(instructions)

        # Get full skill prompt
        full_prompt = self._build_skill_prompt(skill, instructions)

        # Execute via agent
        if skill.allowed_tools:
            # Restrict tools if specified
            return await self._execute_with_restricted_tools(skill, full_prompt)
        else:
            # Normal execution
            return await self.agent.run(full_prompt)

    async def execute_script(
        self,
        skill: Skill,
        script_name: str,
        *args: str
    ) -> str:
        """
        Execute a script from a skill.

        Args:
            skill: Skill containing the script
            script_name: Name of the script file
            *args: Arguments to pass to the script

        Returns:
            Script output
        """
        script_path = skill.get_script_path(script_name)
        if not script_path:
            raise ValueError(f"Script not found: {script_name}")

        logger.info(f"Executing script: {script_name} from skill {skill.name}")

        try:
            result = subprocess.run(
                ["python", str(script_path), *args],
                capture_output=True,
                text=True,
                cwd=str(script_path.parent)
            )

            if result.returncode != 0:
                logger.error(f"Script error: {result.stderr}")
                return f"Error: {result.stderr}"

            return result.stdout

        except Exception as e:
            logger.error(f"Failed to execute script: {e}")
            return f"Error: {e}"

    def get_reference_content(
        self,
        skill: Skill,
        reference_name: str
    ) -> str | None:
        """
        Get content of a reference file.

        Args:
            skill: Skill containing the reference
            reference_name: Name of the reference file

        Returns:
            Reference content or None
        """
        return skill.get_reference(reference_name)

    async def _process_dynamic_context(
        self,
        content: str,
        context: SkillExecutionContext
    ) -> str:
        """
        Process !`command` syntax for dynamic context.

        This executes shell commands before the skill content
        is sent to the LLM, replacing the placeholder with output.
        """
        import re

        # Find all !`command` patterns
        pattern = r'!`([^`]+)`'

        def replace_command(match):
            command = match.group(1)
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=str(context.skill.path) if context.skill.path else None
                )
                return result.stdout.strip()
            except Exception as e:
                logger.warning(f"Failed to execute command '{command}': {e}")
                return f"[Error: {e}]"

        return re.sub(pattern, replace_command, content)

    def _build_skill_prompt(self, skill: Skill, instructions: str) -> str:
        """Build the full prompt for skill execution."""
        parts = [
            f"# Skill: {skill.name}",
            "",
            f"**Description:** {skill.description}",
            ""
        ]

        if skill.license:
            parts.append(f"**License:** {skill.license}")
            parts.append("")

        parts.append("## Instructions")
        parts.append("")
        parts.append(instructions)

        return "\n".join(parts)

    async def _execute_with_restricted_tools(
        self,
        skill: Skill,
        prompt: str
    ) -> str:
        """Execute skill with restricted tool access."""
        # Store original tools
        original_tools = dict(self.agent.context.tools)

        # Filter to allowed tools
        allowed = set(skill.allowed_tools)
        self.agent.context.tools = {
            name: tool
            for name, tool in original_tools.items()
            if name in allowed
        }

        try:
            result = await self.agent.run(prompt)
            return result
        finally:
            # Restore original tools
            self.agent.context.tools = original_tools
