"""Runtime helpers for subprocess-based tests."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_usable(executable: Path) -> bool:
    if not executable.is_file():
        return False
    try:
        subprocess.run(
            [str(executable), "-c", "import mcp"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def resolve_test_python_executable() -> str:
    """Return the configured or currently running Python executable."""
    configured = os.getenv("GUARDGPT_TEST_PYTHON")
    if configured:
        executable = Path(configured)
        if not _is_usable(executable):
            raise RuntimeError(
                "GUARDGPT_TEST_PYTHON is not a usable project Python executable: "
                f"{executable}"
            )
        return str(executable)

    candidates = (
        Path(sys.executable),
        PROJECT_ROOT / ".venv-py314" / "Scripts" / "python.exe",
    )
    for executable in candidates:
        if _is_usable(executable):
            return str(executable)

    if not Path(sys.executable).is_file():
        raise RuntimeError(
            f"No usable project Python executable found; current interpreter is missing: "
            f"{sys.executable}"
        )
    raise RuntimeError(
        "No usable project Python executable found; checked the current interpreter "
        f"and {candidates[1]}"
    )
