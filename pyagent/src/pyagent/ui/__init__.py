"""
PyAgent Web UI module.

Provides a clean user interface with backend capabilities:
- Hooks (auto): Logging, Timing, Error Handling
- RAG (auto): Document indexing and retrieval
- Memory (auto): Conversation, Semantic, Episodic memory
- State (manual): Save/Load conversation checkpoints
"""

from pyagent.ui.gradio_app import create_app, launch_app
from pyagent.ui.backend import PyAgentBackend, get_backend

__all__ = [
    "create_app",
    "launch_app",
    "PyAgentBackend",
    "get_backend",
]
