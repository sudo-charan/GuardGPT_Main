"""Review queue for FLAG outcomes.

A FLAG means the pipeline produced something it will not release on its own:
the answer was withheld, but nothing in the evidence justified calling the
request unsafe. Each flagged request gets one durable record so a human can
triage it, and the record deliberately carries digests instead of prompt or
answer text, matching the compact audit log.

Resolving a record records the human decision only. It never re-releases an
answer, and it never touches the audit log or the prompt/complete reports, so
the original decision stays exactly as it was made.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "ReviewQueue",
    "ReviewError",
    "ReviewNotFound",
    "ReviewAlreadyResolved",
    "DEFAULT_QUEUE",
    "RESOLVED_STATUSES",
]

DEFAULT_QUEUE = "safety_review"
RESOLVED_STATUSES = ("APPROVED", "REJECTED")


class ReviewError(RuntimeError):
    """Base class for review queue failures."""


class ReviewNotFound(ReviewError):
    """No record with that review id exists."""


class ReviewAlreadyResolved(ReviewError):
    """The record already carries a final status."""


class ReviewQueue:
    """Append-only queue with in-place status updates for resolution."""

    def __init__(self, path=None):
        from core.observability import logs_dir

        self.path = Path(path or logs_dir() / "guardgpt_review.jsonl")
        self.lock = threading.Lock()

    # ------------------------------------------------------------ write path

    def enqueue(self, record: dict) -> dict:
        """Assign an id, mark the record pending, and persist it."""
        entry = dict(record)
        entry.setdefault("queue", DEFAULT_QUEUE)
        entry["review_id"] = "rev_" + uuid.uuid4().hex[:12]
        entry["created_at"] = datetime.now(timezone.utc).isoformat()
        entry["status"] = "PENDING"
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                file.flush()
        return entry

    # ------------------------------------------------------------- read path

    def records(self) -> list[dict]:
        """Return every queued record in file order."""
        if not self.path.is_file():
            return []
        with self.lock:
            return self._read()

    def pending(self) -> list[dict]:
        return [record for record in self.records() if record.get("status") == "PENDING"]

    def get(self, review_id: str) -> dict:
        for record in self.records():
            if record.get("review_id") == review_id:
                return record
        raise ReviewNotFound(f"No review with id {review_id!r}")

    # ------------------------------------------------------------- resolution

    def resolve(
        self,
        review_id: str,
        decision: str,
        reviewer: str = "operator",
        note: str = "",
    ) -> dict:
        """Record a human decision without altering the original evidence."""
        status = str(decision or "").upper()
        if status not in RESOLVED_STATUSES:
            raise ReviewError("decision must be APPROVED or REJECTED")

        with self.lock:
            records = self._read()
            target = next(
                (record for record in records if record.get("review_id") == review_id),
                None,
            )
            if target is None:
                raise ReviewNotFound(f"No review with id {review_id!r}")
            if target.get("status") != "PENDING":
                raise ReviewAlreadyResolved(
                    f"{review_id} is already {target.get('status')}"
                )
            target["status"] = status
            target["resolution"] = {
                "decision": status,
                "reviewer": str(reviewer or "operator"),
                "note": str(note or ""),
                "resolved_at": datetime.now(timezone.utc).isoformat(),
                "released_answer": False,
            }
            self._write(records)
        return target

    # ---------------------------------------------------------------- helpers

    def _read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        records = []
        with self.path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ReviewError(
                        f"Review queue line {line_number} is not valid JSON"
                    ) from error
                if not isinstance(record, dict):
                    raise ReviewError(
                        f"Review queue line {line_number} is not an object"
                    )
                records.append(record)
        return records

    def _write(self, records: list[dict]) -> None:
        """Replace the queue atomically so a crash cannot truncate it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.path)
