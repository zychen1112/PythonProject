"""
Skill validator - validates skill structure and content.
"""

import re
from pathlib import Path
from typing import Any

from pyagent.skills.skill import Skill
from pyagent.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Error during skill validation."""
    pass


class ValidationResult:
    """Result of skill validation."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        logger.error(f"Validation error: {message}")

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning(f"Validation warning: {message}")

    def __str__(self) -> str:
        lines = []

        if self.errors:
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  - {e}")

        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        if not lines:
            lines.append("Validation passed!")

        return "\n".join(lines)


class SkillValidator:
    """
    Validates skill structure and content.

    Validates:
    - SKILL.md existence and format
    - YAML frontmatter requirements
    - Name format constraints
    - Description requirements
    - Directory structure
    """

    # Validation constants
    MAX_NAME_LENGTH = 64
    MAX_DESCRIPTION_LENGTH = 1024
    MAX_COMPATIBILITY_LENGTH = 500
    MAX_INSTRUCTIONS_LINES = 500

    # Name pattern: lowercase letters, numbers, hyphens only
    # Cannot start/end with hyphen, no consecutive hyphens
    NAME_PATTERN = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')

    def validate(self, path: Path) -> ValidationResult:
        """
        Validate a skill at the given path.

        Args:
            path: Path to skill directory

        Returns:
            ValidationResult with any errors or warnings
        """
        result = ValidationResult()

        # Check directory exists
        if not path.exists():
            result.add_error(f"Path does not exist: {path}")
            return result

        if not path.is_dir():
            result.add_error(f"Path is not a directory: {path}")
            return result

        # Check SKILL.md exists
        skill_md = path / "SKILL.md"
        if not skill_md.exists():
            result.add_error("SKILL.md file not found")
            return result

        # Validate SKILL.md content
        content = skill_md.read_text(encoding="utf-8")
        self._validate_content(content, path.name, result)

        # Validate directory structure
        self._validate_structure(path, result)

        return result

    def validate_skill(self, skill: Skill) -> ValidationResult:
        """
        Validate a loaded Skill object.

        Args:
            skill: Skill to validate

        Returns:
            ValidationResult with any errors or warnings
        """
        result = ValidationResult()

        # Validate name
        self._validate_name(skill.name, result)

        # Validate description
        self._validate_description(skill.description, result)

        # Validate compatibility
        if skill.compatibility:
            self._validate_compatibility(skill.compatibility, result)

        # Validate instructions length
        if skill.instructions:
            lines = skill.instructions.count("\n") + 1
            if lines > self.MAX_INSTRUCTIONS_LINES:
                result.add_warning(
                    f"Instructions exceed {self.MAX_INSTRUCTIONS_LINES} lines ({lines})"
                )

        return result

    def _validate_content(
        self,
        content: str,
        dir_name: str,
        result: ValidationResult
    ) -> None:
        """Validate SKILL.md content."""
        if not content.strip():
            result.add_error("SKILL.md is empty")
            return

        # Check for frontmatter
        if not content.startswith("---"):
            result.add_error("SKILL.md must start with YAML frontmatter (---)")
            return

        # Find end of frontmatter
        end_idx = content.find("---", 3)
        if end_idx == -1:
            result.add_error("YAML frontmatter not properly closed")
            return

        frontmatter_str = content[3:end_idx].strip()

        # Parse frontmatter
        try:
            import yaml
            frontmatter = yaml.safe_load(frontmatter_str) or {}
        except yaml.YAMLError as e:
            result.add_error(f"Invalid YAML frontmatter: {e}")
            return

        # Validate required fields
        name = frontmatter.get("name", "")
        description = frontmatter.get("description", "")

        # Validate name
        self._validate_name(name or dir_name, result)

        # Name should match directory
        if name and name != dir_name:
            result.add_warning(
                f"Skill name '{name}' does not match directory name '{dir_name}'"
            )

        # Validate description
        self._validate_description(description, result)

        # Validate optional fields
        if "compatibility" in frontmatter:
            self._validate_compatibility(frontmatter["compatibility"], result)

    def _validate_name(self, name: str, result: ValidationResult) -> None:
        """Validate skill name."""
        if not name:
            result.add_error("Skill name is required")
            return

        if len(name) > self.MAX_NAME_LENGTH:
            result.add_error(
                f"Skill name exceeds {self.MAX_NAME_LENGTH} characters"
            )

        if not self.NAME_PATTERN.match(name):
            result.add_error(
                f"Invalid skill name '{name}'. "
                "Must be lowercase letters, numbers, and hyphens only. "
                "Cannot start/end with hyphen or have consecutive hyphens."
            )

        # Check for reserved words
        reserved = ["anthropic", "claude"]
        if name.lower() in reserved:
            result.add_error(f"Skill name cannot be a reserved word: {name}")

    def _validate_description(self, description: str, result: ValidationResult) -> None:
        """Validate skill description."""
        if not description:
            result.add_error("Skill description is required")
            return

        if not description.strip():
            result.add_error("Skill description cannot be empty")
            return

        if len(description) > self.MAX_DESCRIPTION_LENGTH:
            result.add_error(
                f"Skill description exceeds {self.MAX_DESCRIPTION_LENGTH} characters"
            )

        # Check for XML tags (not allowed)
        if "<" in description and ">" in description:
            result.add_warning("Description contains XML-like tags")

    def _validate_compatibility(
        self,
        compatibility: str,
        result: ValidationResult
    ) -> None:
        """Validate compatibility field."""
        if len(compatibility) > self.MAX_COMPATIBILITY_LENGTH:
            result.add_error(
                f"Compatibility exceeds {self.MAX_COMPATIBILITY_LENGTH} characters"
            )

    def _validate_structure(self, path: Path, result: ValidationResult) -> None:
        """Validate directory structure."""
        # Check for optional directories
        scripts_dir = path / "scripts"
        if scripts_dir.exists():
            if not scripts_dir.is_dir():
                result.add_warning("'scripts' exists but is not a directory")
            else:
                # Check scripts are executable or valid
                for script in scripts_dir.iterdir():
                    if script.is_file() and script.suffix == ".py":
                        self._validate_python_script(script, result)

        references_dir = path / "references"
        if references_dir.exists() and not references_dir.is_dir():
            result.add_warning("'references' exists but is not a directory")

        assets_dir = path / "assets"
        if assets_dir.exists() and not assets_dir.is_dir():
            result.add_warning("'assets' exists but is not a directory")

    def _validate_python_script(
        self,
        script_path: Path,
        result: ValidationResult
    ) -> None:
        """Validate a Python script."""
        try:
            content = script_path.read_text(encoding="utf-8")
            compile(content, str(script_path), 'exec')
        except SyntaxError as e:
            result.add_warning(f"Python script has syntax error: {script_path.name}: {e}")


def validate_skill(path: str | Path) -> ValidationResult:
    """Convenience function to validate a skill."""
    validator = SkillValidator()
    return validator.validate(Path(path))
