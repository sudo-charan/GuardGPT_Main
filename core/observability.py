"""Per-request tracing, durable metrics and error records.

The architecture diagram has an observability band that the project did not
implement: requests were logged, but nothing recorded stage latency, nothing
counted outcomes, and errors were only visible as an exception class inside a
report. Everything here writes identifiers and timings only - never prompt or
answer text.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "RequestTrace",
    "Metrics",
    "AlertRules",
    "error_log",
    "request_log",
    "metrics",
    "new_trace",
    "reset_metrics",
    "logs_dir",
    "append_jsonl",
    "audit_trail",
]


def _logs_dir() -> Path:
    """Log directory. ``GUARDGPT_LOG_DIR`` overrides it (used by the test suite
    so tests never write into the real ``logs/`` folder or its metrics)."""
    override = os.getenv("GUARDGPT_LOG_DIR")
    if override:
        return Path(override).resolve()
    root = os.getenv("GUARDGPT_PROJECT_ROOT")
    base = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    return base / "logs"


def logs_dir() -> Path:
    """Public accessor for the active log directory."""
    return _logs_dir()


class RequestTrace:
    """Collects stage timings for a single request."""

    def __init__(self, request_id: str | None = None) -> None:
        self.request_id = request_id
        self.started = time.perf_counter()
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        begin = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - begin) * 1000.0
            self.stages[name] = round(self.stages.get(name, 0.0) + elapsed, 3)

    def as_dict(self) -> dict:
        elapsed = (time.perf_counter() - self.started) * 1000.0
        return {
            "request_id": self.request_id,
            "stages_ms": dict(self.stages),
            "total_ms": round(elapsed, 3),
        }


class AlertRules:
    """Static thresholds; a tripped rule is reported next to the counters."""

    def __init__(self) -> None:
        self.max_jev_failures = 0
        self.max_audit_failures = 0
        self.max_urgent_escalations = 0   # any urgent escalation (e.g. a flagged self-harm signal) alerts
        self.max_error_rate = 0.25
        self.min_samples_for_rate = 5

    def evaluate(self, counters: dict) -> list[str]:
        alerts = []
        if counters.get("jev_failures", 0) > self.max_jev_failures:
            alerts.append("jev_failures")
        if counters.get("audit_failures", 0) > self.max_audit_failures:
            alerts.append("audit_failures")
        if counters.get("escalations_urgent", 0) > self.max_urgent_escalations:
            alerts.append("urgent_escalation")
        requests = counters.get("requests", 0)
        if requests >= self.min_samples_for_rate:
            rate = counters.get("errors", 0) / requests
            if rate > self.max_error_rate:
                alerts.append("error_rate")
        return alerts


class Metrics:
    """Thread-safe counters plus the most recent stage timings."""

    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._last_trace: dict = {}
        self.alert_rules = AlertRules()
        self.path = Path(path) if path is not None else _logs_dir() / "guardgpt_metrics.json"
        self._restore()

    def _restore(self) -> None:
        """Continue from the last snapshot; a restart must not read as zero."""
        try:
            stored = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # missing, unreadable or corrupt: start empty
            return
        counters = stored.get("counters") if isinstance(stored, dict) else None
        if isinstance(counters, dict):
            self._counters = {
                str(key): int(value)
                for key, value in counters.items()
                if isinstance(value, int) and not isinstance(value, bool)
            }

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def record_request(self, action: str, final_status: str, trace: RequestTrace | None) -> None:
        with self._lock:
            self._counters["requests"] = self._counters.get("requests", 0) + 1
            key = f"action:{str(action).lower()}"
            self._counters[key] = self._counters.get(key, 0) + 1
            status_key = f"status:{str(final_status).lower()}"
            self._counters[status_key] = self._counters.get(status_key, 0) + 1
            # Publish the error counter from the first request so a reader can
            # tell "no errors" from "key not tracked yet".
            self._counters.setdefault("errors", 0)
            if str(final_status) == "ERROR":
                self._counters["errors"] += 1
            if trace is not None:
                self._last_trace = trace.as_dict()
            snapshot = dict(self._counters)
        self._persist(snapshot)

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(self._counters)
            last_trace = dict(self._last_trace)
        return {
            "counters": counters,
            "alerts": self.alert_rules.evaluate(counters),
            "last_request": last_trace,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _persist(self, counters: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {"counters": counters, "updated_at": datetime.now(timezone.utc).isoformat()},
                ensure_ascii=False,
            )
            self.path.write_text(payload + "\n", encoding="utf-8")
        except Exception:  # metrics must never break a request
            pass

    def reset(self) -> None:
        with self._lock:
            self._counters = {}
            self._last_trace = {}
        self._persist({})


def _new_trace(request_id: str | None = None) -> RequestTrace:
    return RequestTrace(request_id)


def new_trace(request_id: str | None = None) -> RequestTrace:
    return _new_trace(request_id)


_meter = Metrics()


def metrics() -> Metrics:
    return _meter


def reset_metrics() -> None:
    _meter.reset()


_write_lock = threading.Lock()


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON line with a single O_APPEND write.

    The old writers used buffered text writes; when two processes (the CLI's
    auto-started MCP server and a test run, or two servers) appended to the
    same file, lines were interleaved and the prompt/complete logs ended up
    with fragments such as ``{"action": "BLOCK", ...}}`` on their own line.
    One ``os.write`` of the whole encoded line on an ``O_APPEND`` descriptor is
    atomic for these record sizes on POSIX and avoids the partial lines.
    """
    data = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    path = Path(path)
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags, 0o644)
        try:
            os.write(descriptor, data)
        finally:
            os.close(descriptor)


def request_log(event: dict) -> None:
    """Append one structured request record: ids, decision, stage latencies."""
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    try:
        append_jsonl(_logs_dir() / "guardgpt_requests.jsonl", entry)
    except Exception:
        pass


def error_log(event: dict) -> None:
    """Append one structured error record (no prompt text)."""
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    try:
        append_jsonl(_logs_dir() / "guardgpt_errors.jsonl", entry)
    except Exception:
        pass


def audit_trail(event_type: str, trace_id: str, path: Path | None = None, **fields) -> None:
    """Cross-cutting audit trail: one line per layer event, keyed by trace id.

    Event types follow the architecture's logging band: ``request``,
    ``security_event`` (Layer 1), ``intent`` (Layer 2), ``jev``,
    ``policy_decision`` (Layer 3), ``tool_call``, ``output_guard``,
    ``response`` and ``error``. No prompt or answer text is written here.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "event": event_type,
        **fields,
    }
    try:
        append_jsonl(Path(path) if path else _logs_dir() / "guardgpt_trace.jsonl", entry)
    except Exception:
        pass
