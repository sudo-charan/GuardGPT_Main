"""
GuardGPT Agent <-> MCP Server client.

Responsibilities:

  - connect to the MCP Server over Streamable HTTP
  - (optionally) discover tools registered on the server
  - invoke named MCP tools with structured arguments
  - parse the structured `CallToolResult` into a clean dict
  - validate responses
  - handle connection failures and tool failures gracefully
  - expose clean sync methods for the host application (the CLI)

This is the "GuardGPT host / MCP client" box of the architecture: the CLI
talks to the MCP server only through this client and never imports the
safety layers directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
from datetime import timedelta
import httpx
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


logger = logging.getLogger(__name__)


MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"


def configured_server_url() -> str:
    """Read the endpoint at call time so runtime configuration changes apply."""
    return os.getenv("GUARDGPT_MCP_URL", MCP_SERVER_URL)


# ============================================================
# Exceptions
# ============================================================

class MCPClientError(Exception):
    """Base error raised by the GuardGPT MCP client."""


class MCPConnectionError(MCPClientError):
    """The MCP server is unreachable or did not complete initialization."""


class MCPToolError(MCPClientError):
    """The MCP tool returned an error or an invalid response."""


# ============================================================
# Result
# ============================================================

@dataclass
class ToolResult:
    """Cleaned-up tool result returned to Agent nodes."""

    tool_name: str
    data: dict[str, Any]
    raw_text: str
    is_error: bool


class SmartTimeout(float):
    """Compatible timeout wrapper for both MCP 1.x (expects timedelta.total_seconds())
    and MCP 2.x / anyio (expects float for deadline math)."""
    def total_seconds(self) -> float:
        return float(self)


class _SessionWorker:
    """One event loop and one initialised MCP session per endpoint.

    Every call used to create an HTTP client, complete the MCP handshake and tear
    it down again. The session is now held open and reused; a transport failure
    resets it so the next call reconnects.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name=f"mcp:{url}", daemon=True)
        self._lock = threading.Lock()
        self._session = None
        self._stack = None
        self._started = threading.Event()
        self._thread.start()
        self._started.wait(timeout=5)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def _submit(self, coroutine, timeout: float):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout)
        except BaseException:
            future.cancel()
            raise

    async def _ensure_session(self, timeout: float):
        if self._session is not None:
            return self._session
        stack = contextlib.AsyncExitStack()
        local = urlparse(self.url).hostname in {"127.0.0.1", "localhost", "::1"}
        try:
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=httpx.Timeout(timeout), trust_env=not local)
            )
            streams = await stack.enter_async_context(
                streamable_http_client(self.url, http_client=http_client)
            )
            session = await stack.enter_async_context(
                ClientSession(streams[0], streams[1])
            )
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        self._stack, self._session = stack, session
        return session

    async def _drop_session(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:
                pass

    async def _call(self, tool_name: str, arguments: dict, timeout: float):
        session = await self._ensure_session(timeout)
        return await session.call_tool(
            tool_name, arguments=arguments, read_timeout_seconds=SmartTimeout(timeout)
        )

    async def _list(self):
        session = await self._ensure_session(900.0)
        return await session.list_tools()

    def call(self, tool_name: str, arguments: dict, timeout: float):
        with self._lock:
            try:
                return self._submit(self._call(tool_name, arguments, timeout), timeout + 5.0)
            except BaseException:
                self._reset()
                raise

    def list(self):
        with self._lock:
            try:
                return self._submit(self._list(), 905.0)
            except BaseException:
                self._reset()
                raise

    def _reset(self) -> None:
        try:
            self._submit(self._drop_session(), 5.0)
        except Exception:
            pass

    def close(self) -> None:
        self._reset()
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass


_workers: dict[str, _SessionWorker] = {}
_workers_lock = threading.Lock()


def _worker_for(url: str) -> _SessionWorker:
    with _workers_lock:
        worker = _workers.get(url)
        if worker is None:
            worker = _SessionWorker(url)
            _workers[url] = worker
        return worker


def close_clients() -> None:
    """Close every persistent session (tests and shutdown)."""
    with _workers_lock:
        workers = list(_workers.values())
        _workers.clear()
    for worker in workers:
        worker.close()


def _timeout_seconds(read_timeout_seconds: Optional[float]) -> float:
    if isinstance(read_timeout_seconds, timedelta):
        return float(read_timeout_seconds.total_seconds())
    return float(read_timeout_seconds or 900)


def _connection_error(tool_name: str, url: str, error: Exception) -> MCPClientError:
    message = str(error).lower()
    if any(
        token in message
        for token in ("connect", "connection", "refused", "taskgroup", "all connection attempts failed")
    ):
        return MCPConnectionError(f"Cannot reach MCP server at {url}: {error}")
    return MCPClientError(f"MCP call to '{tool_name}' failed: {error}")


# ============================================================
# Result coercion
# ============================================================

def _coerce_tool_result(tool_name: str, result: Any) -> ToolResult:
    """
    Convert the MCP `CallToolResult` into a `ToolResult` with a parsed dict.

    The GuardGPT server tools return their Pydantic output models, which
    FastMCP serializes as a single JSON `TextContent` block.
    """
    if result is None:
        raise MCPToolError(f"Tool '{tool_name}' returned no result.")

    is_error = bool(getattr(result, "isError", getattr(result, "is_error", False)))
    content = getattr(result, "content", None)
    structured = getattr(result, "structuredContent", getattr(result, "structured_content", None))
    if is_error:
        raise MCPToolError(f"Tool '{tool_name}' failed. No result was accepted.")

    if structured:
        if isinstance(structured, dict):
            return ToolResult(
                tool_name=tool_name,
                data=structured,
                raw_text=json.dumps(structured, ensure_ascii=False),
                is_error=is_error,
            )

    if not content:
        if is_error:
            raise MCPToolError(f"Tool '{tool_name}' returned an error with no content.")
        raise MCPToolError(f"Tool '{tool_name}' returned empty content.")

    text_chunks: list[str] = []
    for chunk in content:
        text = getattr(chunk, "text", None)
        if text:
            text_chunks.append(text)

    raw_text = "\n".join(text_chunks).strip()

    if not raw_text:
        if is_error:
            raise MCPToolError(
                f"Tool '{tool_name}' returned an error with no text content."
            )
        raise MCPToolError(
            f"Tool '{tool_name}' returned non-text content: {content!r}"
        )

    parsed: Any
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = {"_raw_text": raw_text}

    if is_error:
        raise MCPToolError(
            f"Tool '{tool_name}' reported error: {raw_text}"
        )

    if not isinstance(parsed, dict):
        raise MCPToolError(
            f"Tool '{tool_name}' returned non-object JSON: {type(parsed).__name__}"
        )

    return ToolResult(
        tool_name=tool_name,
        data=parsed,
        raw_text=raw_text,
        is_error=False,
    )


# ============================================================
# Synchronous facade for Agent nodes (Phase 3+)
# ============================================================

def call_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    url: Optional[str] = None,
    read_timeout_seconds: Optional[float] = None,
) -> ToolResult:
    """Synchronously call an MCP tool on the persistent session for `url`."""
    target = url or configured_server_url()
    seconds = _timeout_seconds(read_timeout_seconds)
    try:
        result = _worker_for(target).call(tool_name, arguments, seconds)
    except TimeoutError as error:
        raise MCPConnectionError(
            f"Timed out calling tool '{tool_name}' at {target}"
        ) from error
    except MCPClientError:
        raise
    except Exception as error:
        raise _connection_error(tool_name, target, error) from error
    return _coerce_tool_result(tool_name, result)


def list_tools(*, url: str = MCP_SERVER_URL) -> list[str]:
    """Synchronously list tool names registered on the MCP server."""
    target = url or configured_server_url()
    try:
        listed = _worker_for(target).list()
    except MCPClientError:
        raise
    except Exception as error:
        raise MCPConnectionError(
            f"Cannot list tools on MCP server at {target}: {error}"
        ) from error
    return [tool.name for tool in listed.tools]


# ============================================================
# Backwards-compatible helpers used by the existing test_mcp.py
# ============================================================

def run_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Backwards-compatible sync helper.

    The legacy Agent code wrapped tool responses in `{"data": {...}}` and
    expected a dict back. This helper keeps that surface so the existing
    `test_mcp.py` script and Phase 3 Agent nodes can rely on it.
    """
    return {"data": call_tool(tool_name, arguments, url=url).data}


def known_tool_names() -> Iterable[str]:
    """Return the canonical GuardGPT tool names handled by the MCP server."""
    return (
        "health",
        "complete_request",
        "session_report",
        "input_guard",
        "intent_analysis",
        "jev_evaluate",
        "policy_decision",
        "review_queue",
    )


__all__ = [
    "MCP_SERVER_URL",
    "configured_server_url",
    "MCPClientError",
    "MCPConnectionError",
    "MCPToolError",
    "ToolResult",
    "call_tool",
    "list_tools",
    "run_mcp_tool",
    "known_tool_names",
    "close_clients",
]
