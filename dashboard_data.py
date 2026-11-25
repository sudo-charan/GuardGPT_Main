"""Data layer of the GuardGPT dashboard.

Everything here takes a logs directory and returns plain data: log readers, search,
request detail, sessions, time series, feedback, alert acknowledgement, system info,
log trimming, CSV exports and the summary report. Nothing here starts a model or a
server, and nothing here ever writes prompt text anywhere new: feedback stores trace
ids and labels, and exports read the (already redacted) prompt text from the prompt log.
"""

from __future__ import annotations

import csv
import html
import io
import json
import os
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LABELS = ("safe", "prompt_injection", "jailbreak", "harmful_instructions", "manipulation", "self_harm_risk")
PROMPT_LOG = "guardgpt_prompt.jsonl"
SESSION_LOG = "guardgpt_session.jsonl"
REQUEST_LOG = "guardgpt_requests.jsonl"
AUDIT_LOG = "guardgpt_audit.jsonl"
TRACE_LOG = "guardgpt_trace.jsonl"
ESCALATION_LOG = "guardgpt_escalations.jsonl"
FEEDBACK_LOG = "guardgpt_feedback.jsonl"
ALERT_ACK_FILE = "guardgpt_alerts_ack.json"
METRICS_FILE = "guardgpt_metrics.json"
EVAL_FILE = "guardgpt_eval.json"
# Logs that trimming may shorten. The review queue, escalations, feedback and metrics are never touched.
TRIMMABLE = ("guardgpt_trace.jsonl", "guardgpt_audit.jsonl", "guardgpt_prompt.jsonl", "guardgpt_session.jsonl",
             "guardgpt_complete.jsonl", "guardgpt_requests.jsonl", "guardgpt_errors.jsonl")
CONFIG_KEYS = ("OLLAMA_URL", "OLLAMA_MODEL", "OLLAMA_AUDIT_MODEL", "JEV_STYLE_MODE", "JEV_DEVICE",
               "JEV_MODEL_MIN_CONFIDENCE", "JEV_BLOCK_THRESHOLD", "JEV_FLAG_THRESHOLD", "JEV_FLAG_PEAK",
               "GUARDGPT_L2_WEIGHTS", "GUARDGPT_LOG_DIR")
PACKAGES = ("torch", "transformers", "sentence-transformers", "faiss-cpu", "numpy", "langgraph", "mcp",
            "starlette", "uvicorn", "pydantic")


class LogsBusy(RuntimeError):
    """A log was written to very recently (a chat is probably running)."""


class FeedbackError(ValueError):
    """Invalid feedback."""


# ------------------------------------------------------------------ readers
def _parse_lines(data: bytes, drop_first: bool) -> list[dict]:
    lines = data.splitlines()
    if drop_first and lines:
        lines = lines[1:]  # the first line may be cut in half
    records: list[dict] = []
    for raw in lines:
        try:
            item = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def tail_jsonl(path: Path, limit: int, max_bytes: int = 4_000_000) -> list[dict]:
    """Last ``limit`` JSON objects of a JSONL file (oldest first); bad lines are skipped."""
    path = Path(path)
    if not path.is_file() or limit <= 0:
        return []
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    lines = data.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]
    records: list[dict] = []
    for raw in lines[-limit * 3:]:
        try:
            item = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(item, dict):
            records.append(item)
    return records[-limit:]


_CACHE: dict[str, tuple[int, int, list[dict]]] = {}
_CACHE_LOCK = threading.Lock()


def load_records(path: Path, max_bytes: int = 30_000_000) -> list[dict]:
    """Every record of a JSONL file (oldest first), cached until the file changes."""
    path = Path(path)
    if not path.is_file():
        return []
    stat = path.stat()
    key = str(path)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]
    with path.open("rb") as handle:
        handle.seek(max(0, stat.st_size - max_bytes))
        records = _parse_lines(handle.read(), drop_first=stat.st_size > max_bytes)
    with _CACHE_LOCK:
        _CACHE[key] = (stat.st_mtime_ns, stat.st_size, records)
    return records


def find_trace(path: Path, prefix: str, max_bytes: int = 20_000_000) -> list[dict]:
    """Every trace event whose trace id starts with ``prefix`` (the first matching trace)."""
    path = Path(path)
    if not path.is_file():
        return []
    events: list[dict] = []
    wanted: str | None = None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - max_bytes))
        for raw in handle:
            if prefix.encode() not in raw:
                continue
            try:
                item = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            trace_id = str(item.get("trace_id", ""))
            if not trace_id.startswith(prefix):
                continue
            wanted = wanted or trace_id
            if trace_id == wanted:
                events.append(item)
    return events


def _pct(value: Any) -> float | None:
    try:
        return round(float(value) * 100, 1)
    except (TypeError, ValueError):
        return None


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _prompt_records(logs_dir: Path) -> list[dict]:
    return load_records(Path(logs_dir) / PROMPT_LOG)


# ------------------------------------------------------------------ requests
def entry_summary(record: dict) -> dict:
    final = record.get("final_result") or {}
    layer2 = record.get("layer2_intent") or {}
    jev = record.get("jev") or {}
    policy = record.get("layer3_policy") or {}
    review = record.get("review") or {}
    prompt = record.get("prompt") or {}
    return {
        "trace_id": record.get("trace_id"),
        "time": record.get("timestamp"),
        "session_id": record.get("session_id"),
        "turn": record.get("turn_index"),
        "prompt": (prompt.get("text") or "")[:160],
        "mode": prompt.get("mode"),
        "topic": (record.get("topic") or {}).get("topic"),
        "intent": layer2.get("intent"),
        "confidence": _pct(layer2.get("confidence")),
        "jev": jev.get("verdict"),
        "risk": policy.get("risk_score"),
        "action": final.get("action"),
        "status": final.get("status"),
        "review": review.get("review_id"),
    }


def recent_entries(logs_dir: Path, limit: int) -> list[dict]:
    """Newest-first request summaries from the prompt log."""
    return [entry_summary(r) for r in reversed(tail_jsonl(Path(logs_dir) / PROMPT_LOG, limit))]


def search_requests(logs_dir: Path, *, q: str = "", action: str = "", status: str = "", intent: str = "",
                    topic: str = "", session: str = "", limit: int = 25, offset: int = 0) -> dict:
    """Filtered, paged requests (newest first) with the total number of matches."""
    needle = q.strip().lower()
    action, status = action.strip().upper(), status.strip().upper()
    matches = []
    for record in reversed(_prompt_records(logs_dir)):
        final = record.get("final_result") or {}
        if action and str(final.get("action", "")).upper() != action:
            continue
        if status and str(final.get("status", "")).upper() != status:
            continue
        if intent and (record.get("layer2_intent") or {}).get("intent") != intent:
            continue
        if topic and (record.get("topic") or {}).get("topic") != topic:
            continue
        if session and not str(record.get("session_id") or "").startswith(session):
            continue
        if needle and needle not in ((record.get("prompt") or {}).get("text") or "").lower():
            continue
        matches.append(record)
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    return {"total": len(matches), "offset": offset, "limit": limit,
            "entries": [entry_summary(r) for r in matches[offset: offset + limit]]}


def find_record(logs_dir: Path, prefix: str) -> dict | None:
    for record in reversed(_prompt_records(logs_dir)):
        if str(record.get("trace_id", "")).startswith(prefix):
            return record
    return None


def request_detail(logs_dir: Path, prefix: str) -> dict | None:
    """Everything known about one request: each layer's result, timings and its session risk."""
    logs_dir = Path(logs_dir)
    record = find_record(logs_dir, prefix)
    if record is None:
        return None
    trace_id = record.get("trace_id")
    timing = next((r for r in reversed(load_records(logs_dir / REQUEST_LOG)) if r.get("trace_id") == trace_id), {})
    session = next((r for r in reversed(load_records(logs_dir / SESSION_LOG)) if r.get("trace_id") == trace_id), {})
    layer1 = record.get("layer1_input") or {}
    layer2 = record.get("layer2_intent") or {}
    sources = layer2.get("sources") or {}
    feedback = latest_feedback(logs_dir).get(trace_id)
    return {
        "summary": entry_summary(record),
        "prompt": (record.get("prompt") or {}).get("text") or "",
        "session_id": record.get("session_id"),
        "turn": record.get("turn_index"),
        "topic": record.get("topic"),
        "layer1_input": {"status": layer1.get("status"), "risk_score": layer1.get("risk_score"),
                         "findings": layer1.get("findings") or [], "pii_types": layer1.get("pii_types") or [],
                         "self_harm_level": layer1.get("self_harm_level")},
        "layer2": {"intent": layer2.get("intent"), "confidence": layer2.get("confidence"),
                   "passed_threshold": layer2.get("passed_threshold"),
                   "probabilities": layer2.get("probabilities") or {},
                   "sources": {k: sources.get(k) for k in ("head", "anchors", "dataset") if sources.get(k)},
                   "guard_evidence": sources.get("guard"), "reasons": layer2.get("reasons") or [],
                   "context": layer2.get("context")},
        "jev": record.get("jev") or {},
        "policy": record.get("layer3_policy") or {},
        "agent": record.get("agent"),
        "layer1_output": record.get("layer1_output"),
        "review": record.get("review"),
        "final": record.get("final_result") or {},
        "timings": {"stages_ms": timing.get("stages_ms") or {}, "total_ms": timing.get("total_ms")},
        "session_risk": session.get("session_risk_score"),
        "feedback": feedback,
    }


# ------------------------------------------------------------------ sessions
def sessions_summary(logs_dir: Path, limit: int = 60) -> list[dict]:
    logs_dir = Path(logs_dir)
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in _prompt_records(logs_dir):
        if record.get("session_id"):
            groups[record["session_id"]].append(record)
    last_risk: dict[str, float] = {}
    for record in load_records(logs_dir / SESSION_LOG):
        if record.get("session_id") and record.get("session_risk_score") is not None:
            last_risk[record["session_id"]] = record["session_risk_score"]
    rows = []
    for session_id, records in groups.items():
        actions = Counter((r.get("final_result") or {}).get("action") for r in records)
        intents = Counter((r.get("layer2_intent") or {}).get("intent") for r in records)
        rows.append({
            "session_id": session_id, "turns": len(records),
            "first": records[0].get("timestamp"), "last": records[-1].get("timestamp"),
            "allow": actions.get("ALLOW", 0), "flag": actions.get("FLAG", 0), "block": actions.get("BLOCK", 0),
            "unsafe_intents": {k: v for k, v in intents.items() if k not in (None, "safe")},
            "risk": last_risk.get(session_id, 0.0),
            "preview": ((records[0].get("prompt") or {}).get("text") or "")[:80],
        })
    rows.sort(key=lambda r: str(r["last"]), reverse=True)
    return rows[:limit]


def session_timeline(logs_dir: Path, session_id: str) -> list[dict] | None:
    logs_dir = Path(logs_dir)
    records = [r for r in _prompt_records(logs_dir) if str(r.get("session_id") or "").startswith(session_id)]
    if not records:
        return None
    risks = {r.get("trace_id"): r.get("session_risk_score") for r in load_records(logs_dir / SESSION_LOG)}
    timeline = []
    for record in records:
        entry = entry_summary(record)
        context = (record.get("layer2_intent") or {}).get("context") or {}
        entry.update(session_risk=risks.get(record.get("trace_id")), context=context.get("summary"),
                     session_signal=context.get("session_signal"))
        timeline.append(entry)
    return timeline


# ---------------------------------------------------------------- time series
def timeseries(logs_dir: Path, bucket: str = "hour") -> dict:
    """Requests per bucket by decision, plus average latency, ending at the newest request."""
    logs_dir = Path(logs_dir)
    day = bucket == "day"

    def floor(moment: datetime) -> datetime:
        return moment.replace(hour=0, minute=0, second=0, microsecond=0) if day else \
            moment.replace(minute=0, second=0, microsecond=0)

    cells: dict[datetime, Counter] = defaultdict(Counter)
    for record in load_records(logs_dir / AUDIT_LOG):
        moment = _parse_time(record.get("timestamp"))
        if moment is None:
            continue
        cell = cells[floor(moment)]
        status = str(record.get("final_status", "")).upper()
        action = str(record.get("action", "")).upper()
        cell["support" if status == "SUPPORT" else "error" if status == "ERROR" else action.lower()] += 1
    latency: dict[datetime, list[float]] = defaultdict(list)
    for record in load_records(logs_dir / REQUEST_LOG):
        moment = _parse_time(record.get("timestamp"))
        if moment is not None and isinstance(record.get("total_ms"), (int, float)):
            latency[floor(moment)].append(float(record["total_ms"]))
    if not cells:
        return {"bucket": "day" if day else "hour", "points": []}
    step = timedelta(days=1) if day else timedelta(hours=1)
    count = 30 if day else 48
    end = max(cells)
    points = []
    for index in range(count - 1, -1, -1):
        start = end - step * index
        if start < min(cells):
            continue
        cell, times = cells.get(start, Counter()), latency.get(start, [])
        points.append({"t": start.isoformat(), "allow": cell.get("allow", 0), "flag": cell.get("flag", 0),
                       "block": cell.get("block", 0), "support": cell.get("support", 0),
                       "error": cell.get("error", 0),
                       "avg_ms": round(sum(times) / len(times)) if times else None})
    return {"bucket": "day" if day else "hour", "points": points}


# ------------------------------------------------------------------- alerts
def _metrics_counters(logs_dir: Path) -> dict:
    try:
        counters = json.loads((Path(logs_dir) / METRICS_FILE).read_text(encoding="utf-8")).get("counters", {})
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in counters.items() if isinstance(v, int) and not isinstance(v, bool)}


def alerts_state(logs_dir: Path) -> dict:
    """Alerts raised since the last acknowledgement (counters are cumulative, so use the difference)."""
    from core.observability import AlertRules

    logs_dir = Path(logs_dir)
    counters = _metrics_counters(logs_dir)
    ack: dict = {}
    try:
        ack = json.loads((logs_dir / ALERT_ACK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    base = ack.get("counters") or {}
    delta = {key: max(0, value - int(base.get(key, 0))) for key, value in counters.items()}
    return {"active": AlertRules().evaluate(delta), "acknowledged_at": ack.get("at"), "counters": counters}


def ack_alerts(logs_dir: Path) -> dict:
    logs_dir = Path(logs_dir)
    payload = {"counters": _metrics_counters(logs_dir), "at": datetime.now(timezone.utc).isoformat()}
    target = logs_dir / ALERT_ACK_FILE
    logs_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, target)
    return {"acknowledged_at": payload["at"]}


def overview(logs_dir: Path) -> dict:
    logs_dir = Path(logs_dir)
    alerts = alerts_state(logs_dir)
    recent = tail_jsonl(logs_dir / PROMPT_LOG, 100)
    intents = Counter((r.get("layer2_intent") or {}).get("intent") or "unknown" for r in recent)
    topics = Counter((r.get("topic") or {}).get("topic") or "unknown" for r in recent)
    timings = sorted(r.get("total_ms") for r in tail_jsonl(logs_dir / REQUEST_LOG, 100)
                     if isinstance(r.get("total_ms"), (int, float)))
    return {
        "counters": alerts["counters"],
        "alerts": alerts["active"],
        "acknowledged_at": alerts["acknowledged_at"],
        "intents": dict(intents),
        "topics": dict(topics),
        "last_request": recent[-1].get("timestamp") if recent else None,
        "latency_ms": {"count": len(timings),
                       "avg": round(sum(timings) / len(timings)) if timings else None,
                       "median": round(timings[len(timings) // 2]) if timings else None,
                       "max": round(timings[-1]) if timings else None},
    }


# ------------------------------------------------------------------ feedback
def latest_feedback(logs_dir: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for record in load_records(Path(logs_dir) / FEEDBACK_LOG):
        if record.get("trace_id"):
            latest[record["trace_id"]] = record
    return latest


def add_feedback(logs_dir: Path, trace_prefix: str, verdict: str, correct_label: str | None = None,
                 note: str = "") -> dict:
    """Mark a request's safety intent as correct or wrong. Stores ids and labels only, no text."""
    logs_dir = Path(logs_dir)
    verdict = str(verdict or "").lower()
    if verdict not in ("correct", "wrong"):
        raise FeedbackError("verdict must be 'correct' or 'wrong'")
    if verdict == "wrong" and correct_label not in LABELS:
        raise FeedbackError("choose the correct label: " + ", ".join(LABELS))
    record = find_record(logs_dir, trace_prefix)
    if record is None:
        raise FeedbackError("request not found")
    predicted = (record.get("layer2_intent") or {}).get("intent")
    entry = {"at": datetime.now(timezone.utc).isoformat(), "trace_id": record["trace_id"], "verdict": verdict,
             "predicted_intent": predicted, "correct_label": correct_label if verdict == "wrong" else predicted,
             "note": str(note or "")[:300]}
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / FEEDBACK_LOG).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _CACHE.pop(str(logs_dir / FEEDBACK_LOG), None)
    return entry


def feedback_list(logs_dir: Path) -> list[dict]:
    logs_dir = Path(logs_dir)
    prompts = {r.get("trace_id"): r for r in _prompt_records(logs_dir)}
    rows = []
    for trace_id, entry in latest_feedback(logs_dir).items():
        record = prompts.get(trace_id) or {}
        rows.append({**entry, "prompt": ((record.get("prompt") or {}).get("text") or "")[:200],
                     "final_action": (record.get("final_result") or {}).get("action")})
    rows.sort(key=lambda r: str(r.get("at")), reverse=True)
    return rows


def feedback_export(logs_dir: Path) -> dict:
    """Wrong predictions as ready-to-paste lines for intent_anchors.json (the redacted prompt text)."""
    logs_dir = Path(logs_dir)
    prompts = {r.get("trace_id"): r for r in _prompt_records(logs_dir)}
    anchors: dict[str, list[str]] = {label: [] for label in LABELS}
    skipped = 0
    for trace_id, entry in latest_feedback(logs_dir).items():
        if entry.get("verdict") != "wrong":
            continue
        text = ((prompts.get(trace_id) or {}).get("prompt") or {}).get("text", "").strip()
        label = entry.get("correct_label")
        if not text or len(text) > 300 or label not in anchors or "<" in text and "_" in text and ">" in text:
            skipped += 1
            continue
        if text not in anchors[label]:
            anchors[label].append(text)
    anchors = {label: items for label, items in anchors.items() if items}
    return {"anchors": anchors, "count": sum(len(v) for v in anchors.values()), "skipped": skipped,
            "how_to": "Add each list to the matching label in intent_classifier/intent_anchors.json, then run "
                      "python evaluate_intent.py to check nothing got worse."}


# ------------------------------------------------------------------- reviews
def review_rows(records: list[dict], logs_dir: Path) -> list[dict]:
    """Review records with the (redacted) prompt text and age, for the Reviews tab."""
    prompts = {r.get("trace_id"): r for r in _prompt_records(logs_dir)}
    now = datetime.now(timezone.utc)
    keep = ("review_id", "created_at", "status", "priority", "escalated", "escalation_reasons", "intent",
            "risk_level", "reason_codes", "failed_checks", "stage", "trace_id", "resolution")
    rows = []
    for record in records:
        row = {key: record.get(key) for key in keep}
        created = _parse_time(record.get("created_at"))
        row["age_seconds"] = max(0, int((now - created).total_seconds())) if created else None
        row["prompt"] = (((prompts.get(record.get("trace_id")) or {}).get("prompt") or {}).get("text") or "")[:200]
        rows.append(row)
    return rows


# -------------------------------------------------------------------- system
def _count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return count


def ollama_models(timeout: float = 3.0) -> dict:
    import requests

    base = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    try:
        session = requests.Session()
        session.trust_env = False
        data = session.get(f"{base}/api/tags", timeout=timeout).json()
    except Exception as error:
        return {"reachable": False, "error": type(error).__name__, "models": []}
    return {"reachable": True, "models": [{"name": m.get("name"), "size_mb": round((m.get("size") or 0) / 1e6)}
                                          for m in data.get("models", [])]}


def system_info(logs_dir: Path, status: dict | None = None) -> dict:
    from importlib import metadata

    logs_dir = Path(logs_dir)
    files = []
    if logs_dir.is_dir():
        for path in sorted(logs_dir.iterdir()):
            if path.is_file() and path.suffix in (".jsonl", ".json"):
                stat = path.stat()
                files.append({"name": path.name, "bytes": stat.st_size,
                              "lines": _count_lines(path) if path.suffix == ".jsonl" else None,
                              "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                              "trimmable": path.name in TRIMMABLE})
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"status": status, "ollama": ollama_models(),
            "config": {key: os.getenv(key) for key in CONFIG_KEYS if os.getenv(key)},
            "logs": {"dir": str(logs_dir), "files": files, "total_bytes": sum(f["bytes"] for f in files)},
            "versions": versions}


def trim_logs(logs_dir: Path, keep: int = 1000, busy_seconds: float = 15.0, now: float | None = None) -> dict:
    """Keep only the newest ``keep`` records of the large logs. Review queue, escalations, feedback
    and metrics are never touched. Refuses while a log was written to very recently."""
    logs_dir = Path(logs_dir)
    keep = max(100, min(int(keep), 1_000_000))
    now = time.time() if now is None else now
    targets = [logs_dir / name for name in TRIMMABLE if (logs_dir / name).is_file()]
    for path in targets:
        if now - path.stat().st_mtime < busy_seconds:
            raise LogsBusy(f"{path.name} was written to in the last {int(busy_seconds)} seconds; "
                           "stop any running chat and try again")
    result = {}
    for path in targets:
        lines = path.read_bytes().splitlines(keepends=True)
        before = len(lines)
        if before > keep:
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_bytes(b"".join(lines[-keep:]))
            os.replace(temporary, path)
            _CACHE.pop(str(path), None)
        result[path.name] = {"before": before, "after": min(before, keep)}
    return result


# ------------------------------------------------------------------- exports
def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text  # CSV formula injection


def _csv(header: list[str], rows: list[list[Any]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([_cell(value) for value in row])
    return out.getvalue()


def requests_csv(logs_dir: Path, **filters: Any) -> str:
    filters["limit"], filters["offset"] = 200, 0
    entries: list[dict] = []
    while True:
        page = search_requests(logs_dir, **filters)
        entries += page["entries"]
        filters["offset"] += 200
        if filters["offset"] >= page["total"] or len(entries) >= 20_000:
            break
    cols = ["time", "trace_id", "session_id", "prompt", "topic", "intent", "confidence", "jev", "action",
            "status", "review"]
    return _csv(cols, [[e.get(c) for c in cols] for e in entries])


def reviews_csv(rows: list[dict]) -> str:
    cols = ["review_id", "created_at", "status", "priority", "escalated", "intent", "risk_level", "prompt",
            "trace_id"]
    return _csv(cols, [[r.get(c) for c in cols] for r in rows])


def summary_report_html(logs_dir: Path, status: dict | None = None, reviews: list[dict] | None = None) -> str:
    """A one-page, print-friendly summary (self-contained HTML)."""
    logs_dir = Path(logs_dir)
    ov = overview(logs_dir)
    counters = ov["counters"]
    records = _prompt_records(logs_dir)
    requests_log = load_records(logs_dir / REQUEST_LOG)
    stage_times: dict[str, list[float]] = defaultdict(list)
    for item in requests_log:
        for stage, ms in (item.get("stages_ms") or {}).items():
            if isinstance(ms, (int, float)):
                stage_times[stage].append(ms)
    total = counters.get("requests", 0) or 1
    esc = html.escape
    pending = [r for r in (reviews or []) if r.get("status") == "PENDING"]
    escalated = [r for r in pending if r.get("escalated")]

    def rows(pairs):
        return "".join(f"<tr><td>{esc(str(a))}</td><td>{esc(str(b))}</td></tr>" for a, b in pairs)

    def share(key):
        value = counters.get(key, 0)
        return f"{value} ({value / total:.0%})"

    first = records[0].get("timestamp") if records else "-"
    last = records[-1].get("timestamp") if records else "-"
    cfg = {k: os.getenv(k) for k in ("OLLAMA_MODEL", "JEV_STYLE_MODE") if os.getenv(k)}
    intent_rows = rows(sorted(ov["intents"].items(), key=lambda kv: -kv[1]))
    topic_rows = rows(sorted(ov["topics"].items(), key=lambda kv: -kv[1]))
    stage_rows = rows((s, f"{sum(v) / len(v) / 1000:.2f} s average over {len(v)} requests")
                      for s, v in sorted(stage_times.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])))
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>GuardGPT summary report</title>
<style>body{{font:14px/1.5 system-ui,Segoe UI,Arial,sans-serif;max-width:820px;margin:28px auto;padding:0 18px;color:#1c2430}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:15px;margin:22px 0 6px;border-bottom:1px solid #ccd3dd;padding-bottom:3px}}
table{{border-collapse:collapse;width:100%}}td{{padding:4px 8px;border-bottom:1px solid #e6eaf0}}td:first-child{{width:48%;color:#46515f}}
.muted{{color:#6b7686}}@media print{{body{{margin:0}}}}</style></head><body>
<h1>GuardGPT summary report</h1><div class="muted">Generated {esc(when)} &middot; requests logged {esc(str(first)[:19])} to {esc(str(last)[:19])}</div>
<h2>Traffic and decisions</h2><table>{rows([
        ("Requests", counters.get("requests", 0)), ("Allowed", share("action:allow")), ("Flagged for review", share("action:flag")),
        ("Blocked", share("action:block")), ("Supportive replies (self-harm)", share("status:support")),
        ("Errors", counters.get("errors", 0)), ("Escalations", counters.get("escalations", 0)),
        ("Urgent escalations", counters.get("escalations_urgent", 0))])}</table>
<h2>Reviews</h2><table>{rows([("Pending", len(pending)), ("Pending and escalated", len(escalated)),
                              ("Alerts active", ", ".join(ov["alerts"]) or "none")])}</table>
<h2>Safety intent (last 100 requests)</h2><table>{intent_rows or rows([("no data", "")])}</table>
<h2>Topics (last 100 requests)</h2><table>{topic_rows or rows([("no data", "")])}</table>
<h2>Latency</h2><table>{rows([("Median", f"{(ov['latency_ms']['median'] or 0) / 1000:.1f} s"),
                              ("Average", f"{(ov['latency_ms']['avg'] or 0) / 1000:.1f} s"),
                              ("Slowest", f"{(ov['latency_ms']['max'] or 0) / 1000:.1f} s")])}{stage_rows}</table>
<h2>Setup</h2><table>{rows(list(cfg.items()) + [("JEV mode", ((status or {}).get("jev") or {}).get("mode", "-")),
                                               ("Intent model", "ready" if (status or {}).get("intent_model") else "-")])}</table>
<p class="muted">Prompt text is not included in this report.</p></body></html>"""
