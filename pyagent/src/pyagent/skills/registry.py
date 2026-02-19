"""
Skill registry - manages available skills.
"""

from typing import Any

from pyagent.skills.skill import Skill
from pyagent.utils.logging import get_logger

logger = get_logger(__name__)


class SkillRegistry:
    """
    Registry for managing available skills.

    Supports registration, lookup, and discovery of skills.
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._aliases: dict[str, str] = {}

    def register(self, skill: Skill) -> None:
        """
        Register a skill.

        Args:
            skill: Skill to register
        """
        if skill.name in self._skills:
            logger.warning(f"Overwriting existing skill: {skill.name}")

        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name}")

    def unregister(self, name: str) -> Skill | None:
        """
        Unregister a skill.

        Args:
            name: Name of skill to unregister

        Returns:
            The unregistered skill, or None if not found
        """
        skill = self._skills.pop(name, None)

        # Remove any aliases
        aliases_to_remove = [k for k, v in self._aliases.items() if v == name]
        for alias in aliases_to_remove:
            del self._aliases[alias]

        if skill:
            logger.info(f"Unregistered skill: {name}")

        return skill

    def get(self, name: str) -> Skill | None:
        """
        Get a skill by name or alias.

        Args:
            name: Skill name or alias

        Returns:
            The skill, or None if not found
        """
        # Check aliases first
        if name in self._aliases:
            name = self._aliases[name]

        return self._skills.get(name)

    def has(self, name: str) -> bool:
        """Check if a skill is registered."""
        return name in self._skills or name in self._aliases

    def list_all(self) -> list[Skill]:
        """Get all registered skills."""
        return list(self._skills.values())

    def list_names(self) -> list[str]:
        """Get all registered skill names."""
        return list(self._skills.keys())

    def find_matching(self, query: str) -> list[Skill]:
        """
        Find skills that match a query.

        Args:
            query: Search query

        Returns:
            List of matching skills
        """
        matches = []
        for skill in self._skills.values():
            if skill.matches_query(query):
                matches.append(skill)

        return matches

    def find_by_tag(self, tag: str) -> list[Skill]:
        """
        Find skills by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching skills
        """
        matches = []
        tag_lower = tag.lower()

        for skill in self._skills.values():
            if any(tag_lower == t.lower() for t in skill.metadata.tags):
                matches.append(skill)

        return matches

    def add_alias(self, alias: str, skill_name: str) -> None:
        """
        Add an alias for a skill.

        Args:
            alias: Alias name
            skill_name: Target skill name
        """
        if skill_name not in self._skills:
            raise ValueError(f"Skill not found: {skill_name}")

        self._aliases[alias] = skill_name
        logger.info(f"Added alias '{alias}' for skill '{skill_name}'")

    def get_metadata_summary(self) -> list[dict[str, Any]]:
        """
        Get a summary of all skill metadata.

        Useful for generating system prompt content.
        """
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "tags": skill.metadata.tags
            }
            for skill in self._skills.values()
        ]

    def get_skills_prompt(self) -> str:
        """
        Generate a prompt listing available skills.

        Returns:
            Formatted string for system prompt
        """
        if not self._skills:
            return "No skills are currently available."

        lines = ["Available skills:"]
        for skill in sorted(self._skills.values(), key=lambda s: s.name):
            lines.append(f"  - {skill.name}: {skill.description}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all registered skills."""
        self._skills.clear()
        self._aliases.clear()
        logger.info("Cleared skill registry")

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __iter__(self):
        return iter(self._skills.values())
