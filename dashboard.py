"""GuardGPT web dashboard: a live monitor for requests, sessions, reviews and traces.

    python dashboard.py              # serve the dashboard and open the browser
    python dashboard.py --no-browser # same, without opening a browser
    python dashboard.py --port 9000

Chat stays in the CLI (``python main.py --chat``). Run the CLI in one terminal and
the dashboard in another: the dashboard reads the same ``logs/`` folder and pushes
every new request to the open page within about a second.

The dashboard starts no model and no MCP server, so it opens instantly. The one
exception is the Evaluation tab, which runs ``evaluate_intent.py`` as a separate
background job on request. The dashboard can also resolve FLAG reviews (like
``review_cli.py``), record feedback on classifications, acknowledge alerts, and trim
old logs. Data access lives in ``dashboard_data.py``.

It binds to 127.0.0.1 only. Every /api call must carry the per-run token that is
embedded in the page, and requests whose Host header is not local are rejected, so
a web page on another site cannot drive it. It shows prompt text from ``logs/``
(already redacted); treat it like the logs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import closing
from pathlib import Path
from typing import Callable

import anyio
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

import dashboard_data as data
from dashboard_data import find_trace, overview, recent_entries, tail_jsonl  # noqa: F401  (re-exported)

ROOT = Path(__file__).resolve().parent
UI_PATH = ROOT / "dashboard_ui" / "index.html"
TOKEN_HEADER = "x-dashboard-token"
TOKEN_PLACEHOLDER = "__DASHBOARD_TOKEN__"
HEX_ID = re.compile(r"^[0-9a-fA-F]{6,64}$")
SESSION_ID = re.compile(r"^[0-9a-fA-F]{6,64}$")
REVIEW_ID = re.compile(r"^rev_[0-9a-f]{6,32}$")
STATUS_TTL = 15.0
STREAM_POLL_SECONDS = 1.0


# ----------------------------------------------------------------- local status
def local_status(root: Path = ROOT) -> dict:
    """Cheap readiness check: files on disk and an Ollama ping. Loads no model."""
    from core.layers.jev_finetuned import FineTunedJEV
    from core.llama_backend import LlamaBackend

    root = Path(root)
    backend = LlamaBackend()
    mode = (os.getenv("JEV_STYLE_MODE") or "builtin").strip().lower()
    jev = {"mode": mode, "ok": True, "problems": []}
    if mode == "finetuned":
        config = FineTunedJEV().check_config()
        jev.update(ok=bool(config["ok"]), problems=config["problems"])
    intent_files = ("best_model.pt", "tokenizer.json", "config.json", "intent_anchors.json")
    data_files = ("guardgpt_dataset.jsonl", "guardgpt_faiss.index", "guardgpt_id_map.json")
    intent_ok = all((root / "intent_classifier" / name).is_file() for name in intent_files)
    data_ok = all((root / "data" / name).is_file() for name in data_files)
    ollama = bool(backend.is_available())
    return {
        "ollama": {"reachable": ollama, "model": backend.model,
                   "audit_model": os.getenv("OLLAMA_AUDIT_MODEL") or backend.model},
        "intent_model": intent_ok,
        "data_files": data_ok,
        "jev": jev,
        "ready": ollama and intent_ok and data_ok and jev["ok"],
    }


def _int(value, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(int(value), high))
    except (TypeError, ValueError):
        return default


# -------------------------------------------------------------------------- app
class Dashboard:
    def __init__(self, logs_dir: Path, token: str, port: int,
                 status_fn: Callable[[], dict] = local_status, root: Path = ROOT) -> None:
        from core.review import ReviewQueue

        self.logs_dir, self.token, self.port, self.status_fn = Path(logs_dir), token, port, status_fn
        self.root = Path(root)
        self.queue = ReviewQueue(self.logs_dir / "guardgpt_review.jsonl")
        self._status_cache: tuple[float, dict] | None = None
        self._status_lock = threading.Lock()
        self._eval: dict | None = None
        self._eval_lock = threading.Lock()

    # -- guards
    def _host_ok(self, request: Request) -> bool:
        host = (request.headers.get("host") or "").lower()
        return host in {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}

    def _guard(self, request: Request) -> JSONResponse | None:
        if not self._host_ok(request):
            return JSONResponse({"error": "bad host"}, status_code=403)
        supplied = request.headers.get(TOKEN_HEADER, "")
        if not secrets.compare_digest(supplied.encode(), self.token.encode()):
            return JSONResponse({"error": "missing or invalid dashboard token"}, status_code=401)
        return None

    @staticmethod
    async def _json_body(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    # -- page
    async def index(self, request: Request):
        if not self._host_ok(request):
            return HTMLResponse("Forbidden", status_code=403)
        html = UI_PATH.read_text(encoding="utf-8").replace(TOKEN_PLACEHOLDER, self.token)
        return HTMLResponse(html, headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY",
                                           "Referrer-Policy": "no-referrer"})

    # -- status, overview, alerts
    def _status(self) -> dict:
        with self._status_lock:
            now = time.monotonic()
            if self._status_cache and now - self._status_cache[0] < STATUS_TTL:
                return self._status_cache[1]
            try:
                result = self.status_fn()
            except Exception as error:
                result = {"error": type(error).__name__}
            self._status_cache = (now, result)
            return result

    async def status(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        return JSONResponse(await anyio.to_thread.run_sync(self._status))

    async def get_overview(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        return JSONResponse(await anyio.to_thread.run_sync(data.overview, self.logs_dir))

    async def alerts_ack(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        return JSONResponse(await anyio.to_thread.run_sync(data.ack_alerts, self.logs_dir))

    async def get_timeseries(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        bucket = "day" if request.query_params.get("bucket") == "day" else "hour"
        return JSONResponse(await anyio.to_thread.run_sync(data.timeseries, self.logs_dir, bucket))

    # -- requests, detail, sessions
    @staticmethod
    def _filters(request: Request) -> dict:
        params = request.query_params
        return {key: params.get(key, "")[:200] for key in ("q", "action", "status", "intent", "topic", "session")}

    async def requests(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        params = request.query_params
        result = await anyio.to_thread.run_sync(lambda: data.search_requests(
            self.logs_dir, **self._filters(request), limit=_int(params.get("limit"), 25, 1, 200),
            offset=_int(params.get("offset"), 0, 0, 10_000_000)))
        return JSONResponse(result)

    async def recent(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        limit = _int(request.query_params.get("limit"), 30, 1, 200)
        return JSONResponse({"entries": await anyio.to_thread.run_sync(data.recent_entries, self.logs_dir, limit)})

    async def request_detail(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        trace_id = request.path_params["trace_id"]
        if not HEX_ID.match(trace_id):
            return JSONResponse({"error": "invalid trace id"}, status_code=400)
        detail = await anyio.to_thread.run_sync(data.request_detail, self.logs_dir, trace_id.lower())
        if detail is None:
            return JSONResponse({"error": "request not found"}, status_code=404)
        return JSONResponse(detail)

    async def sessions(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        return JSONResponse({"sessions": await anyio.to_thread.run_sync(data.sessions_summary, self.logs_dir)})

    async def session_detail(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        session_id = request.path_params["session_id"]
        if not SESSION_ID.match(session_id):
            return JSONResponse({"error": "invalid session id"}, status_code=400)
        timeline = await anyio.to_thread.run_sync(data.session_timeline, self.logs_dir, session_id.lower())
        if timeline is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({"session_id": timeline[0].get("session_id"), "turns": timeline})

    async def trace(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        trace_id = request.path_params["trace_id"]
        if not HEX_ID.match(trace_id):
            return JSONResponse({"error": "invalid trace id"}, status_code=400)
        events = await anyio.to_thread.run_sync(data.find_trace, self.logs_dir / data.TRACE_LOG, trace_id.lower())
        if not events:
            return JSONResponse({"error": "trace not found"}, status_code=404)
        return JSONResponse({"trace_id": events[0].get("trace_id"), "events": events})

    # -- reviews
    def _review_list(self, show_all: bool) -> list[dict]:
        records = self.queue.records() if show_all else self.queue.pending()
        order = {"urgent": 0, "high": 1, "normal": 2}
        records = sorted(records, key=lambda r: (r.get("status") != "PENDING", order.get(r.get("priority"), 3),
                                                 str(r.get("created_at"))))
        return data.review_rows(records[:200], self.logs_dir)

    async def reviews(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        try:
            rows = await anyio.to_thread.run_sync(self._review_list, request.query_params.get("status") == "all")
        except Exception as error:
            return JSONResponse({"error": f"review queue unreadable: {type(error).__name__}"}, status_code=500)
        return JSONResponse({"reviews": rows})

    def _resolve_one(self, review_id: str, decision: str, reviewer: str, note: str) -> dict:
        from core.review import ReviewAlreadyResolved, ReviewError, ReviewNotFound

        if not REVIEW_ID.match(review_id):
            return {"review_id": review_id, "error": "invalid review id", "code": 400}
        try:
            record = self.queue.resolve(review_id, decision, reviewer, note)
        except ReviewNotFound:
            return {"review_id": review_id, "error": "review not found", "code": 404}
        except ReviewAlreadyResolved as error:
            return {"review_id": review_id, "error": str(error), "code": 409}
        except ReviewError as error:
            return {"review_id": review_id, "error": str(error), "code": 400}
        return {"review_id": record["review_id"], "status": record["status"]}

    async def resolve(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        body = await self._json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        result = await anyio.to_thread.run_sync(
            self._resolve_one, request.path_params["review_id"], str(body.get("decision", "")).upper(),
            str(body.get("reviewer") or "dashboard")[:60], str(body.get("note") or "")[:500])
        code = result.pop("code", 200)
        return JSONResponse(result, status_code=code)

    async def resolve_bulk(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        body = await self._json_body(request)
        ids = (body or {}).get("ids")
        if not isinstance(ids, list) or not ids or len(ids) > 100 or not all(isinstance(i, str) for i in ids):
            return JSONResponse({"error": "ids must be a list of 1 to 100 review ids"}, status_code=400)
        decision = str(body.get("decision", "")).upper()
        reviewer, note = str(body.get("reviewer") or "dashboard")[:60], str(body.get("note") or "")[:500]

        def run():
            return [self._resolve_one(i, decision, reviewer, note) for i in ids]

        results = await anyio.to_thread.run_sync(run)
        for item in results:
            item.pop("code", None)
        return JSONResponse({"results": results, "resolved": sum("status" in r for r in results)})

    # -- feedback
    async def feedback_add(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        body = await self._json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        trace_id = str(body.get("trace_id", ""))
        if not HEX_ID.match(trace_id):
            return JSONResponse({"error": "invalid trace id"}, status_code=400)
        try:
            entry = await anyio.to_thread.run_sync(
                data.add_feedback, self.logs_dir, trace_id.lower(), body.get("verdict"), body.get("correct_label"),
                body.get("note") or "")
        except data.FeedbackError as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        return JSONResponse(entry)

    async def feedback_get(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        return JSONResponse({"feedback": await anyio.to_thread.run_sync(data.feedback_list, self.logs_dir)})

    async def feedback_export(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        return JSONResponse(await anyio.to_thread.run_sync(data.feedback_export, self.logs_dir))

    # -- system, trimming
    async def system(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        status = await anyio.to_thread.run_sync(self._status)
        return JSONResponse(await anyio.to_thread.run_sync(data.system_info, self.logs_dir, status))

    async def trim(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        body = await self._json_body(request)
        if not body or body.get("confirm") is not True:
            return JSONResponse({"error": "send {\"confirm\": true} to trim the logs"}, status_code=400)
        keep = _int(body.get("keep"), 1000, 100, 1_000_000)
        try:
            result = await anyio.to_thread.run_sync(data.trim_logs, self.logs_dir, keep)
        except data.LogsBusy as error:
            return JSONResponse({"error": str(error)}, status_code=409)
        return JSONResponse({"kept": keep, "files": result})

    # -- exports and report
    @staticmethod
    def _download(text: str, name: str, media: str) -> Response:
        return Response(text, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"',
                                                         "Cache-Control": "no-store"})

    async def export_requests(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        text = await anyio.to_thread.run_sync(lambda: data.requests_csv(self.logs_dir, **self._filters(request)))
        return self._download(text, "guardgpt_requests.csv", "text/csv; charset=utf-8")

    async def export_reviews(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        rows = await anyio.to_thread.run_sync(self._review_list, request.query_params.get("status") == "all")
        return self._download(data.reviews_csv(rows), "guardgpt_reviews.csv", "text/csv; charset=utf-8")

    async def report(self, request: Request):
        if (denied := self._guard(request)):
            return denied

        def build():
            return data.summary_report_html(self.logs_dir, self._status(), self._review_list(False))

        return HTMLResponse(await anyio.to_thread.run_sync(build), headers={"Cache-Control": "no-store"})

    # -- evaluation job
    def _eval_state(self) -> dict:
        with self._eval_lock:
            job = self._eval
            out = self.logs_dir / data.EVAL_FILE
            if job is None:
                result = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None
                return {"state": "idle", "result": result}
            code = job["proc"].poll()
            if code is None:
                return {"state": "running", "started": job["started"], "include_jev": job["include_jev"],
                        "seconds": round(time.time() - job["started_ts"])}
            if code == 0 and out.is_file():
                return {"state": "done", "started": job["started"], "include_jev": job["include_jev"],
                        "result": json.loads(out.read_text(encoding="utf-8"))}
            return {"state": "failed", "started": job["started"], "exit_code": code, "log": job["log"]}

    async def eval_start(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        body = await self._json_body(request) or {}
        include_jev = bool(body.get("include_jev"))

        def start() -> dict:
            with self._eval_lock:
                if self._eval is not None and self._eval["proc"].poll() is None:
                    return {"error": "an evaluation is already running", "code": 409}
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                if not include_jev:
                    env["JEV_STYLE_MODE"] = "builtin"   # quick run: rules only, no 1B model
                log = Path(os.getenv("TEMP", "/tmp")) / "guardgpt_dashboard_eval.log"
                with open(log, "wb") as handle:
                    proc = subprocess.Popen(
                        [sys.executable, "-u", str(self.root / "evaluate_intent.py"), "--json",
                         str(self.logs_dir / data.EVAL_FILE)],
                        cwd=str(self.root), env=env, stdout=handle, stderr=subprocess.STDOUT)
                self._eval = {"proc": proc, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "started_ts": time.time(), "include_jev": include_jev, "log": str(log)}
            return {"state": "running"}

        result = await anyio.to_thread.run_sync(start)
        return JSONResponse(result, status_code=result.pop("code", 200))

    async def eval_status(self, request: Request):
        if (denied := self._guard(request)):
            return denied
        return JSONResponse(await anyio.to_thread.run_sync(self._eval_state))

    def shutdown(self) -> None:
        with self._eval_lock:
            if self._eval is not None and self._eval["proc"].poll() is None:
                self._eval["proc"].terminate()

    # -- live push (server-sent events)
    def _marks(self) -> tuple:
        sizes = []
        for name in (data.PROMPT_LOG, data.ESCALATION_LOG, "guardgpt_review.jsonl"):
            try:
                sizes.append((self.logs_dir / name).stat().st_size)
            except OSError:
                sizes.append(0)
        return tuple(sizes)

    def _new_events(self, last_trace: str | None, previous: tuple, current: tuple) -> tuple[list[dict], str | None]:
        events: list[dict] = []
        if current[0] != previous[0]:
            records = tail_jsonl(self.logs_dir / data.PROMPT_LOG, 10)
            ids = [r.get("trace_id") for r in records]
            fresh = records[ids.index(last_trace) + 1:] if last_trace in ids else records[-1:]
            for record in fresh:
                events.append({"type": "request", "entry": data.entry_summary(record)})
            if records:
                last_trace = records[-1].get("trace_id")
        if current[1] != previous[1]:
            escalations = tail_jsonl(self.logs_dir / data.ESCALATION_LOG, 1)
            if escalations:
                events.append({"type": "escalation", "entry": {k: escalations[0].get(k) for k in (
                    "review_id", "priority", "escalation_reasons", "intent", "risk_level", "trace_id")}})
        if current[2] != previous[2]:
            events.append({"type": "review"})
        return events, last_trace

    async def stream(self, request: Request):
        if (denied := self._guard(request)):
            return denied

        async def generate():
            yield "retry: 3000\n\n"
            yield 'data: {"type": "hello"}\n\n'
            marks = self._marks()
            last = (tail_jsonl(self.logs_dir / data.PROMPT_LOG, 1) or [{}])[0].get("trace_id")
            idle = 0
            while not await request.is_disconnected():
                await anyio.sleep(STREAM_POLL_SECONDS)
                now = self._marks()
                if now != marks:
                    events, last = self._new_events(last, marks, now)
                    marks = now
                    for event in events:
                        yield f"data: {json.dumps(event)}\n\n"
                    idle = 0
                else:
                    idle += 1
                    if idle >= 15:
                        yield ": ping\n\n"
                        idle = 0

        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    def app(self) -> Starlette:
        return Starlette(routes=[
            Route("/", self.index),
            Route("/api/status", self.status),
            Route("/api/overview", self.get_overview),
            Route("/api/alerts/ack", self.alerts_ack, methods=["POST"]),
            Route("/api/timeseries", self.get_timeseries),
            Route("/api/requests", self.requests),
            Route("/api/recent", self.recent),
            Route("/api/request/{trace_id}", self.request_detail),
            Route("/api/sessions", self.sessions),
            Route("/api/sessions/{session_id}", self.session_detail),
            Route("/api/trace/{trace_id}", self.trace),
            Route("/api/reviews", self.reviews),
            Route("/api/reviews/bulk", self.resolve_bulk, methods=["POST"]),
            Route("/api/reviews/{review_id}/resolve", self.resolve, methods=["POST"]),
            Route("/api/feedback", self.feedback_get),
            Route("/api/feedback", self.feedback_add, methods=["POST"]),
            Route("/api/feedback/export", self.feedback_export),
            Route("/api/system", self.system),
            Route("/api/system/trim", self.trim, methods=["POST"]),
            Route("/api/export/requests", self.export_requests),
            Route("/api/export/reviews", self.export_reviews),
            Route("/api/report", self.report),
            Route("/api/eval/start", self.eval_start, methods=["POST"]),
            Route("/api/eval/status", self.eval_status),
            Route("/api/stream", self.stream),
        ])


def create_app(logs_dir, token, port, status_fn: Callable[[], dict] = local_status, root: Path = ROOT) -> Starlette:
    return Dashboard(logs_dir, token, port, status_fn, root).app()


# ------------------------------------------------------------------------- main
def _free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return sock.getsockname()[1]
    raise RuntimeError("no free port")


def main(argv: list[str] | None = None) -> int:
    # Only when the dashboard really starts: importing this module (tests) must not touch the environment.
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="GuardGPT web dashboard (live monitor).")
    parser.add_argument("--port", type=int, default=int(os.getenv("GUARDGPT_DASHBOARD_PORT", "8080")))
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window.")
    args = parser.parse_args(argv)

    import signal

    import uvicorn

    from core.observability import logs_dir

    if hasattr(signal, "SIGBREAK"):  # Ctrl+Break stops the dashboard like Ctrl+C
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    try:
        port = _free_port(args.port)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    board = Dashboard(logs_dir(), secrets.token_urlsafe(24), port)
    url = f"http://127.0.0.1:{port}"
    print(f"GuardGPT dashboard: {url}\nChat in the CLI (python main.py --chat); requests appear here live.\n"
          "Press Ctrl+C to stop.\n", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(board.app(), host="127.0.0.1", port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        board.shutdown()
    print("Dashboard stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
