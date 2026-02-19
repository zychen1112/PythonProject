"""
Configuration utilities.
"""

from pathlib import Path
from typing import Any

import yaml

from pydantic import BaseModel


class Config(BaseModel):
    """Base configuration class."""

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load configuration from file (YAML or JSON)."""
        path = Path(path)

        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif path.suffix == ".json":
            import json
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str = "PyAgent"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_iterations: int = 10
    system_prompt: str | None = None

    # Provider settings
    provider: str = "openai"
    api_key: str | None = None
    base_url: str | None = None

    # Skills settings
    skills_dir: str | None = None
    auto_discover_skills: bool = True

    # MCP settings
    mcp_servers: list[dict[str, Any]] = []


def load_config(path: str | Path = "pyagent.yaml") -> AgentConfig:
    """
    Load agent configuration from file.

    Args:
        path: Path to config file

    Returns:
        AgentConfig instance
    """
    path = Path(path)

    if not path.exists():
        return AgentConfig()

    return AgentConfig.from_file(path)
