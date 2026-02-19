"""
Skill definition.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SkillMetadata(BaseModel):
    """Metadata for a skill."""
    author: str = ""
    version: str = "1.0"
    tags: list[str] = Field(default_factory=list)
    custom: dict[str, str] = Field(default_factory=dict)


class Skill(BaseModel):
    """
    Represents an Agent Skill.

    Skills are modular capabilities that extend an agent's functionality.
    Each skill packages instructions, metadata, and optional resources.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    instructions: str = ""
    license: str = ""
    compatibility: str = ""
    metadata: SkillMetadata = Field(default_factory=SkillMetadata)
    allowed_tools: list[str] = Field(default_factory=list)

    # Filesystem paths
    path: Path | None = None
    scripts_dir: Path | None = None
    references_dir: Path | None = None
    assets_dir: Path | None = None

    @property
    def skill_md_path(self) -> Path | None:
        """Path to SKILL.md file."""
        if self.path:
            return self.path / "SKILL.md"
        return None

    def get_reference(self, name: str) -> str | None:
        """Get content of a reference file."""
        if not self.references_dir:
            return None

        ref_path = self.references_dir / name
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8")
        return None

    def get_script_path(self, name: str) -> Path | None:
        """Get path to a script file."""
        if not self.scripts_dir:
            return None

        script_path = self.scripts_dir / name
        if script_path.exists():
            return script_path
        return None

    def get_asset_path(self, name: str) -> Path | None:
        """Get path to an asset file."""
        if not self.assets_dir:
            return None

        asset_path = self.assets_dir / name
        if asset_path.exists():
            return asset_path
        return None

    def to_prompt_format(self) -> str:
        """Format skill for inclusion in system prompt."""
        return f"- **{self.name}**: {self.description}"

    def get_full_instructions(self) -> str:
        """Get full instructions including description."""
        return f"# {self.name}\n\n{self.description}\n\n{self.instructions}"

    def matches_query(self, query: str) -> bool:
        """Check if this skill matches a query."""
        query_lower = query.lower()
        return (
            query_lower in self.name.lower() or
            query_lower in self.description.lower() or
            any(query_lower in tag.lower() for tag in self.metadata.tags)
        )
