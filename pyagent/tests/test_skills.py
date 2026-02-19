"""
Tests for skills functionality.
"""

import tempfile
from pathlib import Path

import pytest

from pyagent.skills.skill import Skill, SkillMetadata
from pyagent.skills.loader import SkillLoader
from pyagent.skills.registry import SkillRegistry
from pyagent.skills.validator import SkillValidator


@pytest.fixture
def sample_skill_dir():
    """Create a temporary skill directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "test-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: test-skill
description: A test skill for unit testing
license: MIT
metadata:
  author: test
  version: "1.0"
  tags:
    - test
---

# Test Skill

This is a test skill.

## Instructions

1. Do something
2. Do something else
""")
        yield skill_dir


class TestSkillLoader:
    """Tests for SkillLoader class."""

    def test_load_skill(self, sample_skill_dir):
        loader = SkillLoader()
        skill = loader.load(sample_skill_dir)

        assert skill.name == "test-skill"
        assert skill.description == "A test skill for unit testing"
        assert skill.license == "MIT"
        assert skill.metadata.author == "test"
        assert "test" in skill.metadata.tags
        assert "Instructions" in skill.instructions

    def test_load_skill_missing_file(self):
        loader = SkillLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="SKILL.md not found"):
                loader.load(Path(tmpdir))

    def test_load_all_skills(self, sample_skill_dir):
        loader = SkillLoader()
        skills = loader.load_all(sample_skill_dir.parent)

        assert len(skills) == 1
        assert skills[0].name == "test-skill"


class TestSkillRegistry:
    """Tests for SkillRegistry class."""

    def test_register_and_get_skill(self):
        registry = SkillRegistry()
        skill = Skill(
            name="test",
            description="Test skill"
        )

        registry.register(skill)
        assert registry.get("test") == skill

    def test_unregister_skill(self):
        registry = SkillRegistry()
        skill = Skill(name="test", description="Test")

        registry.register(skill)
        removed = registry.unregister("test")

        assert removed == skill
        assert registry.get("test") is None

    def test_find_matching_skills(self):
        registry = SkillRegistry()
        skill1 = Skill(name="code-review", description="Review code")
        skill2 = Skill(name="weather", description="Get weather info")
        skill3 = Skill(
            name="code-gen",
            description="Generate code",
            metadata=SkillMetadata(tags=["code", "generation"])
        )

        registry.register(skill1)
        registry.register(skill2)
        registry.register(skill3)

        matches = registry.find_matching("code")
        assert len(matches) == 2

        matches = registry.find_matching("weather")
        assert len(matches) == 1

    def test_find_by_tag(self):
        registry = SkillRegistry()
        skill = Skill(
            name="tagged",
            description="Tagged skill",
            metadata=SkillMetadata(tags=["python", "testing"])
        )

        registry.register(skill)

        matches = registry.find_by_tag("python")
        assert len(matches) == 1

        matches = registry.find_by_tag("nonexistent")
        assert len(matches) == 0

    def test_skills_prompt(self):
        registry = SkillRegistry()
        skill = Skill(name="test", description="Test skill")
        registry.register(skill)

        prompt = registry.get_skills_prompt()
        assert "test" in prompt
        assert "Test skill" in prompt


class TestSkillValidator:
    """Tests for SkillValidator class."""

    def test_validate_valid_skill(self, sample_skill_dir):
        validator = SkillValidator()
        result = validator.validate(sample_skill_dir)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_missing_skill_md(self):
        validator = SkillValidator()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = validator.validate(Path(tmpdir))
            assert not result.is_valid
            assert "SKILL.md file not found" in str(result.errors)

    def test_validate_invalid_name(self):
        validator = SkillValidator()

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "Invalid-Name"
            skill_dir.mkdir()

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text("""---
name: Invalid-Name
description: Test
---
Content
""")
            result = validator.validate(skill_dir)
            assert not result.is_valid

    def test_validate_missing_description(self):
        validator = SkillValidator()

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir()

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text("""---
name: test-skill
---
Content
""")
            result = validator.validate(skill_dir)
            assert not result.is_valid


class TestSkill:
    """Tests for Skill class."""

    def test_skill_properties(self, sample_skill_dir):
        loader = SkillLoader()
        skill = loader.load(sample_skill_dir)

        assert skill.skill_md_path == sample_skill_dir / "SKILL.md"

    def test_matches_query(self):
        skill = Skill(
            name="code-review",
            description="Review and analyze code",
            metadata=SkillMetadata(tags=["code", "review"])
        )

        assert skill.matches_query("code")
        assert skill.matches_query("review")
        assert skill.matches_query("CODE")  # Case insensitive
        assert not skill.matches_query("weather")

    def test_to_prompt_format(self):
        skill = Skill(name="test", description="A test skill")

        prompt = skill.to_prompt_format()
        assert "test" in prompt
        assert "A test skill" in prompt
