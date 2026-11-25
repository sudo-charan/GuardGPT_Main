"""GuardGPT MCP server - the architecture's main backbone (hosts Layers 1-3).

complete_request is the supported end-to-end entry point.

Tool names resolve through :mod:`mcp_server.registry` and dispatch through
:mod:`mcp_server.router`, so the shared policy hooks run on every call and the
registry stays the single description of what the server can do.
"""
import os
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from mcp.server.mcpserver import MCPServer
    class FastMCP(MCPServer):
        def __init__(self, name: str, host: str = "127.0.0.1", port: int = 8000, **kwargs):
            super().__init__(name=name)
            self.host = host
            self.port = port

from mcp_server.policy import default_hooks
from mcp_server.router import Router
from mcp_server.services import build_registry

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("GUARDGPT_PROJECT_ROOT", str(ROOT))
parsed = urlparse(os.getenv("GUARDGPT_MCP_URL", "http://127.0.0.1:8000/mcp"))
if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
    raise ValueError("GuardGPT's unauthenticated MCP server must bind to loopback.")
mcp = FastMCP("GuardGPT", host=parsed.hostname, port=parsed.port or 8000)

registry = build_registry()
router = Router(registry, default_hooks())


@mcp.tool()
def health() -> dict:
    """Report server version, pipeline availability and registered services."""
    return router.dispatch("health", {})


@mcp.tool()
def complete_request(
    prompt: str,
    session_id: str | None = None,
    check_only: bool = False,
    request_id: str | None = None,
) -> dict:
    """Canonical path: check input, optionally generate, audit output, and log one report."""
    return router.dispatch(
        "complete_request",
        {
            "prompt": prompt,
            "session_id": session_id,
            "check_only": check_only,
            "request_id": request_id,
        },
    )


@mcp.tool()
def session_report(session_id: str) -> dict:
    """Session intent history and decayed session risk score."""
    return router.dispatch("session_report", {"session_id": session_id})


@mcp.tool()
def input_guard(prompt: str) -> dict:
    """Layer 1 input-mode checks only."""
    return router.dispatch("input_guard", {"prompt": prompt})


@mcp.tool()
def intent_analysis(prompt: str) -> dict:
    """Layer 2 intent label, confidence and probability distribution."""
    return router.dispatch("intent_analysis", {"prompt": prompt})


@mcp.tool()
def jev_evaluate(prompt: str) -> dict:
    """JEV verdict for the Layer 2 intent + confidence."""
    return router.dispatch("jev_evaluate", {"prompt": prompt})


@mcp.tool()
def policy_decision(prompt: str) -> dict:
    """Layer 1-3 decision (ALLOW / FLAG / BLOCK) without generating an answer."""
    return router.dispatch("policy_decision", {"prompt": prompt})


@mcp.tool()
def review_queue() -> dict:
    """Pending FLAG reviews."""
    return router.dispatch("review_queue", {})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
