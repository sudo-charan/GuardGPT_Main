"""GuardGPT tools MCP server: hosts the agent's tools over the MCP protocol.

Run by the main MCP server on demand (see :mod:`agent.tool_client`):

    python -m mcp_server.tools_server

It binds to loopback only and refuses to start without ``GUARDGPT_TOOLS_TOKEN``,
the per-launch secret that the main server presents on every call. Without that
check any local process could use the generation tools and skip Layers 1-3.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from mcp.server.mcpserver import MCPServer

    class FastMCP(MCPServer):
        def __init__(self, name: str, host: str = "127.0.0.1", port: int = 8001, **kwargs):
            super().__init__(name=name)
            self.host = host
            self.port = port

from mcp_server.tools import build_tool_router

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("GUARDGPT_PROJECT_ROOT", str(ROOT))

DEFAULT_TOOLS_URL = "http://127.0.0.1:8001/mcp"


def pid_alive(pid: int) -> bool:
    """Whether a process with this id still exists (used to exit with the parent)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _exit_with_parent() -> None:
    try:
        parent = int(os.getenv("GUARDGPT_PARENT_PID", "0"))
    except ValueError:
        parent = 0
    if parent <= 0:
        return

    def watch() -> None:
        while True:
            time.sleep(3)
            if not pid_alive(parent):
                os._exit(0)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def build_app():
    token = os.getenv("GUARDGPT_TOOLS_TOKEN", "")
    if not token:
        raise SystemExit("GUARDGPT_TOOLS_TOKEN is not set; the tools server refuses unauthenticated calls.")
    parsed = urlparse(os.getenv("GUARDGPT_TOOLS_URL", DEFAULT_TOOLS_URL))
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("GuardGPT's tools MCP server must bind to loopback.")

    from core.llama_backend import LlamaBackend

    backends: dict = {}

    def generation():
        return backends.setdefault("generate", LlamaBackend())

    def audit():
        if "audit" not in backends:
            model = os.getenv("OLLAMA_AUDIT_MODEL") or generation().model
            backends["audit"] = LlamaBackend(model=model)
        return backends["audit"]

    router = build_tool_router(generation, audit, token=token)
    mcp = FastMCP("GuardGPT-Tools", host=parsed.hostname, port=parsed.port or 8001)

    async def dispatch(name: str, arguments: dict) -> dict:
        # Model calls block for seconds; keep the event loop free.
        return await asyncio.to_thread(router.dispatch, name, arguments)

    @mcp.tool()
    async def ollama_generate(prompt: str, system: str | None = None, history: list | None = None,
                              token: str = "") -> dict:
        """Generate an answer with the local Ollama model."""
        return await dispatch("ollama_generate", {"prompt": prompt, "system": system,
                                                   "history": history, "token": token})

    @mcp.tool()
    async def ollama_support(prompt: str, system: str | None = None, history: list | None = None,
                             token: str = "") -> dict:
        """Generate a crisis-support reply with the local Ollama model."""
        return await dispatch("ollama_support", {"prompt": prompt, "system": system,
                                                  "history": history, "token": token})

    @mcp.tool()
    async def ollama_review(prompt: str, system: str | None = None, schema: dict | None = None,
                            temperature: float | None = None, token: str = "") -> dict:
        """Structured output review with the local Ollama audit model."""
        return await dispatch("ollama_review", {"prompt": prompt, "system": system, "schema": schema,
                                                 "temperature": temperature, "token": token})

    @mcp.tool()
    async def redact_text(text: str, token: str = "") -> dict:
        """Replace PII and secrets with typed placeholders."""
        return await dispatch("redact_text", {"text": text, "token": token})

    return mcp


if __name__ == "__main__":
    _exit_with_parent()
    build_app().run(transport="streamable-http")
