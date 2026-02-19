"""
Test configuration and fixtures.
"""

import pytest


@pytest.fixture
def sample_skill_content():
    """Sample skill content for testing."""
    return """---
name: sample-skill
description: A sample skill for testing
license: MIT
metadata:
  author: test
  version: "1.0"
  tags:
    - test
    - sample
allowed-tools: Read Grep
---

# Sample Skill

This is a sample skill for testing purposes.

## Instructions

1. Read files using the Read tool
2. Search using the Grep tool
3. Process results
"""
