"""
GuardGPT MCP Server package.

The MCP server is the architecture's main backbone: it hosts Layers 1-3, the
service registry, routing engine, LangGraph agent manager and policy hooks.
Importing this package makes the project root importable on `sys.path`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_PROJECT_ROOT_HINT = os.getenv("GUARDGPT_PROJECT_ROOT")


def project_root() -> Path:
    """Return the GuardGPT project root directory."""
    if _PROJECT_ROOT_HINT:
        return Path(_PROJECT_ROOT_HINT).resolve()
    return Path(__file__).resolve().parent.parent


def _ensure_paths() -> None:
    """Insert the project root into sys.path."""
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_paths()
