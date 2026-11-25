"""Client for the GuardGPT tools MCP server.

The main MCP server hosts Layers 1-3 and the LangGraph agent; the agent's tools
(Ollama generation, output review, redaction) live on a second MCP server and are
called over the MCP protocol with :func:`agent.mcp_client.call_tool`.

* The tools server is started on first use (loopback, free port, per-launch token)
  unless ``GUARDGPT_TOOLS_URL`` already points at a running one.
* It exits by itself when the main server process goes away.
* :class:`McpToolBackend` has the same ``generate`` interface as ``LlamaBackend``,
  so the response composer and the output auditor use the tools server too.
"""

from __future__ import annotations

import atexit
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent.mcp_client import MCPClientError, call_tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "llama3"


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _reachable(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.port:
        return False
    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(1.0)
            return sock.connect_ex((parsed.hostname, parsed.port)) == 0
    except OSError:
        return False


class ToolsClient:
    """Calls tools on the tools MCP server, starting it when needed."""

    def __init__(self, url: str | None = None, token: str | None = None, startup_timeout: float = 60.0) -> None:
        self._url = url or os.getenv("GUARDGPT_TOOLS_URL")
        self._token = token or os.getenv("GUARDGPT_TOOLS_TOKEN")
        self.startup_timeout = startup_timeout
        self._process: subprocess.Popen | None = None
        self._ready = False
        self._lock = threading.Lock()
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "120") or 120) + 15.0

    @property
    def url(self) -> str | None:
        return self._url

    def reachable(self) -> bool:
        return bool(self._url) and _reachable(self._url)

    def ensure_started(self) -> str:
        with self._lock:
            if self._ready and self._url:
                return self._url
            if self._url and _reachable(self._url):
                if not self._token:
                    raise MCPClientError(f"A tools server is running at {self._url} but "
                                         "GUARDGPT_TOOLS_TOKEN is not set, so calls cannot be authenticated.")
            else:
                self._start()
            self._ready = True
            return self._url  # type: ignore[return-value]

    def _start(self) -> None:
        port = (urlparse(self._url).port if self._url else None) or _free_port()
        self._url = f"http://127.0.0.1:{port}/mcp"
        self._token = self._token or secrets.token_hex(24)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(PROJECT_ROOT), GUARDGPT_PROJECT_ROOT=str(PROJECT_ROOT),
                   GUARDGPT_TOOLS_URL=self._url, GUARDGPT_TOOLS_TOKEN=self._token,
                   GUARDGPT_PARENT_PID=str(os.getpid()))
        log_path = Path(os.getenv("TEMP", "/tmp")) / f"guardgpt_tools_server_{port}.log"
        with open(log_path, "wb") as log_file:
            self._process = subprocess.Popen(
                [sys.executable, "-u", "-m", "mcp_server.tools_server"],
                cwd=str(PROJECT_ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT,
            )
        atexit.register(self.stop)
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                break
            if _reachable(self._url):
                return
            time.sleep(0.3)
        self.stop()
        raise MCPClientError(f"Tools MCP server failed to start on port {port}. Log: {log_path}")

    def stop(self) -> None:
        self._ready = False
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one tool over MCP and return its result object."""
        url = self.ensure_started()
        payload = {key: value for key, value in arguments.items() if value is not None}
        payload["token"] = self._token or ""
        try:
            return call_tool(name, payload, url=url, read_timeout_seconds=self.timeout).data
        except MCPClientError:
            self._ready = False  # re-check (and restart if needed) on the next call
            raise

    def describe(self) -> dict:
        return {"url": self._url, "reachable": self.reachable(), "protocol": "mcp/streamable-http"}


class McpToolBackend:
    """``LlamaBackend``-compatible ``generate`` that runs on the tools MCP server.

    A call with a JSON ``schema`` is an output review (``ollama_review``);
    anything else is generation (``ollama_generate``).
    """

    def __init__(self, client: ToolsClient) -> None:
        self.client = client
        self.model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    def generate(self, prompt: str, system_prompt: str | None = None, *, schema=None, temperature=None,
                 history=None) -> str:
        if schema is not None:
            result = self.client.call("ollama_review", {"prompt": prompt, "system": system_prompt,
                                                        "schema": schema, "temperature": temperature})
        else:
            result = self.client.call("ollama_generate", {"prompt": prompt, "system": system_prompt,
                                                          "history": history or None})
        return result["text"]


_client: ToolsClient | None = None
_client_lock = threading.Lock()


def tools_client() -> ToolsClient:
    """Process-wide tools client."""
    global _client
    with _client_lock:
        if _client is None:
            _client = ToolsClient()
        return _client
