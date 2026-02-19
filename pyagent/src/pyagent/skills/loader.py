"""
Skill loader - loads skills from filesystem.
"""

from pathlib import Path
from typing import Any

import yaml

from pyagent.skills.skill import Skill, SkillMetadata
from pyagent.utils.logging import get_logger

logger = get_logger(__name__)


class SkillLoader:
    """
    Loads skills from the filesystem.

    Skills are directories containing a SKILL.md file with YAML frontmatter.
    """

    def load(self, path: Path) -> Skill:
        """
        Load a skill from a directory.

        Args:
            path: Path to the skill directory

        Returns:
            Loaded Skill object

        Raises:
            ValueError: If the skill is invalid
        """
        if not path.is_dir():
            raise ValueError(f"Skill path is not a directory: {path}")

        skill_md = path / "SKILL.md"
        if not skill_md.exists():
            raise ValueError(f"SKILL.md not found in: {path}")

        content = skill_md.read_text(encoding="utf-8")

        # Parse frontmatter and body
        frontmatter, instructions = self._parse_frontmatter(content)

        # Extract skill fields
        name = frontmatter.get("name", path.name)
        description = frontmatter.get("description", "")

        if not description:
            # Use first paragraph as description
            first_para = instructions.split("\n\n")[0].strip()
            description = first_para[:1024] if first_para else f"Skill: {name}"

        # Parse metadata
        metadata_dict = frontmatter.get("metadata", {})
        metadata = SkillMetadata(
            author=metadata_dict.get("author", ""),
            version=metadata_dict.get("version", "1.0"),
            tags=metadata_dict.get("tags", []),
            custom=metadata_dict.get("custom", {})
        )

        # Parse allowed tools
        allowed_tools_str = frontmatter.get("allowed-tools", "")
        if isinstance(allowed_tools_str, str):
            allowed_tools = allowed_tools_str.split() if allowed_tools_str else []
        else:
            allowed_tools = allowed_tools_str

        # Create skill
        skill = Skill(
            name=name,
            description=description,
            instructions=instructions,
            license=frontmatter.get("license", ""),
            compatibility=frontmatter.get("compatibility", ""),
            metadata=metadata,
            allowed_tools=allowed_tools,
            path=path,
            scripts_dir=path / "scripts" if (path / "scripts").exists() else None,
            references_dir=path / "references" if (path / "references").exists() else None,
            assets_dir=path / "assets" if (path / "assets").exists() else None
        )

        logger.info(f"Loaded skill: {name} from {path}")
        return skill

    def load_all(self, directory: Path) -> list[Skill]:
        """
        Load all skills from a directory.

        Args:
            directory: Directory containing skill subdirectories

        Returns:
            List of loaded skills
        """
        skills = []

        if not directory.exists():
            logger.warning(f"Skills directory does not exist: {directory}")
            return skills

        for item in directory.iterdir():
            if item.is_dir():
                skill_md = item / "SKILL.md"
                if skill_md.exists():
                    try:
                        skill = self.load(item)
                        skills.append(skill)
                    except Exception as e:
                        logger.error(f"Failed to load skill from {item}: {e}")

        logger.info(f"Loaded {len(skills)} skills from {directory}")
        return skills

    def discover(self) -> list[Skill]:
        """
        Discover and load skills from standard locations.

        Searches in:
        - ~/.claude/skills/ (personal skills)
        - ./.claude/skills/ (project skills)

        Returns:
            List of discovered skills
        """
        skills = []
        seen_names = set()

        # Standard locations to search
        locations = [
            Path.home() / ".claude" / "skills",  # Personal skills
            Path.cwd() / ".claude" / "skills",   # Project skills
        ]

        for location in locations:
            if location.exists():
                for skill in self.load_all(location):
                    if skill.name not in seen_names:
                        skills.append(skill)
                        seen_names.add(skill.name)

        logger.info(f"Discovered {len(skills)} skills")
        return skills

    def _parse_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """
        Parse YAML frontmatter from content.

        Args:
            content: Raw file content

        Returns:
            Tuple of (frontmatter dict, body content)
        """
        if not content.startswith("---"):
            return {}, content

        # Find the closing ---
        end_idx = content.find("---", 3)
        if end_idx == -1:
            return {}, content

        frontmatter_str = content[3:end_idx].strip()
        body = content[end_idx + 3:].strip()

        try:
            frontmatter = yaml.safe_load(frontmatter_str) or {}
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse frontmatter: {e}")
            frontmatter = {}

        return frontmatter, body


def load_skill(path: str | Path) -> Skill:
    """Convenience function to load a skill."""
    loader = SkillLoader()
    return loader.load(Path(path))


def load_skills(directory: str | Path) -> list[Skill]:
    """Convenience function to load all skills from a directory."""
    loader = SkillLoader()
    return loader.load_all(Path(directory))
