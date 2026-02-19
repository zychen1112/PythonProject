"""
Skills demonstration example.
"""

import asyncio
from pathlib import Path

from pyagent.core.agent import Agent, AgentConfig
from pyagent.core.tools import tool
from pyagent.providers.openai import OpenAIProvider
from pyagent.skills.loader import SkillLoader
from pyagent.skills.registry import SkillRegistry
from pyagent.skills.validator import SkillValidator


async def main():
    print("=== Skills Demo ===\n")

    # 1. Validate a skill
    print("1. Validating skill...")
    validator = SkillValidator()
    skill_path = Path(__file__).parent / "skill_demo"

    result = validator.validate(skill_path)
    print(f"   Valid: {result.is_valid}")
    if not result.is_valid:
        print(f"   Errors: {result.errors}")
    if result.warnings:
        print(f"   Warnings: {result.warnings}")
    print()

    # 2. Load a skill
    print("2. Loading skill...")
    loader = SkillLoader()
    skill = loader.load(skill_path)

    print(f"   Name: {skill.name}")
    print(f"   Description: {skill.description}")
    print(f"   Allowed tools: {skill.allowed_tools}")
    print(f"   Metadata: {skill.metadata}")
    print()

    # 3. Register skill in registry
    print("3. Registering skill...")
    registry = SkillRegistry()
    registry.register(skill)

    print(f"   Skills in registry: {registry.list_names()}")
    print(f"   Skills prompt:\n{registry.get_skills_prompt()}")
    print()

    # 4. Find matching skills
    print("4. Finding matching skills...")
    matches = registry.find_matching("code")
    print(f"   Skills matching 'code': {[s.name for s in matches]}")

    matches = registry.find_by_tag("learning")
    print(f"   Skills with 'learning' tag: {[s.name for s in matches]}")
    print()

    # 5. Use skill with agent (requires API key)
    print("5. Skill ready for agent use...")
    print(f"   Skill instructions preview: {skill.instructions[:200]}...")
    print()

    # The skill can now be used with an agent:
    # agent = Agent(
    #     provider=provider,
    #     skills=[skill],
    # )
    # response = await agent.run("Explain how this function works...")


if __name__ == "__main__":
    asyncio.run(main())
