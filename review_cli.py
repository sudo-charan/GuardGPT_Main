"""Operator CLI for the GuardGPT review queue.

    python review_cli.py list
    python review_cli.py list --all
    python review_cli.py escalated
    python review_cli.py resolve rev_abc123 --approve --reviewer alice --note "ok"

Resolving only records the human decision. It does not release the withheld
answer and does not change the audit log, so the original FLAG/REVIEW decision
stays intact.
"""

from __future__ import annotations

import argparse
import sys

from core.review import ReviewError, ReviewQueue

DEFAULT_LIMIT = 20


PRIORITY_ORDER = {"urgent": 0, "high": 1, "normal": 2}


def _summary_line(record: dict) -> str:
    checks = ",".join(record.get("failed_checks") or []) or "-"
    priority = str(record.get("priority") or "normal").upper()
    mark = "!" if record.get("escalated") else " "
    return (
        f"{mark}{priority:<7} {record.get('review_id', '?'):<18} {str(record.get('created_at'))[:19]:<19} "
        f"{record.get('status', '?'):<9} {record.get('intent', '?')}/"
        f"{record.get('risk_level', '?')}  checks={checks}"
    )


def _detail_lines(record: dict) -> list[str]:
    reasons = ",".join(record.get("reason_codes") or []) or "-"
    lines = [
        f"    reasons={reasons}  request={record.get('request_id')}"
        f"  answer_sha={str(record.get('candidate_sha256'))[:12]}"
    ]
    if record.get("escalated"):
        lines.append(f"    ESCALATED: {','.join(record.get('escalation_reasons') or []) or '-'}")
    if record.get("status") != "PENDING":
        resolution = record.get("resolution") or {}
        lines.append(
            f"    resolved {str(resolution.get('resolved_at'))[:19]} by "
            f"{resolution.get('reviewer', '?')} -> {record.get('status')}"
            + (f" ({resolution['note']})" if resolution.get("note") else "")
        )
    return lines


def _cmd_list(queue: ReviewQueue, args) -> int:
    records = queue.records() if args.all else queue.pending()
    if not records:
        print("No pending reviews." if not args.all else "The review queue is empty.")
        return 0

    shown = records[: max(1, args.limit)]
    header = "ALL REVIEWS" if args.all else "PENDING REVIEWS"
    print(f"{header} ({len(records)})")
    for record in shown:
        print(_summary_line(record))
        for line in _detail_lines(record):
            print(line)
    if len(records) > len(shown):
        print(f"... {len(records) - len(shown)} more (use --limit)")
    return 0


def _cmd_escalated(queue: ReviewQueue, args) -> int:
    records = sorted(
        (r for r in queue.pending() if r.get("escalated")),
        key=lambda r: (PRIORITY_ORDER.get(r.get("priority"), 3), str(r.get("created_at"))),
    )
    if not records:
        print("No pending escalations.")
        return 0
    print(f"PENDING ESCALATIONS ({len(records)}), most urgent first")
    for record in records[: max(1, args.limit)]:
        print(_summary_line(record))
        for line in _detail_lines(record):
            print(line)
    return 0


def _cmd_resolve(queue: ReviewQueue, args) -> int:
    decision = "APPROVED" if args.approve else "REJECTED"
    record = queue.resolve(
        args.review_id, decision, reviewer=args.reviewer, note=args.note
    )
    print(f"{record['review_id']}: {decision} by {args.reviewer}")
    print(
        "Original decision unchanged: "
        f"action=FLAG status={record.get('final_status')} "
        f"request={record.get('request_id')}"
    )
    print("No answer was released; this records the review decision only.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="review_cli.py", description="List and resolve GuardGPT review records."
    )
    parser.add_argument(
        "--queue", default=None, help="Review queue path (default: logs/guardgpt_review.jsonl)"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="List review records")
    list_parser.add_argument(
        "--all", action="store_true", help="Include resolved records"
    )
    list_parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="Maximum records to print"
    )
    list_parser.set_defaults(handler=_cmd_list)

    escalated_parser = commands.add_parser("escalated", help="List pending escalated reviews, most urgent first")
    escalated_parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="Maximum records to print"
    )
    escalated_parser.set_defaults(handler=_cmd_escalated)

    resolve_parser = commands.add_parser("resolve", help="Resolve one review record")
    resolve_parser.add_argument("review_id")
    outcome = resolve_parser.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--approve", action="store_true", help="Mark as approved")
    outcome.add_argument("--reject", action="store_true", help="Mark as rejected")
    resolve_parser.add_argument("--reviewer", default="operator")
    resolve_parser.add_argument("--note", default="")
    resolve_parser.set_defaults(handler=_cmd_resolve)

    args = parser.parse_args(argv)
    queue = ReviewQueue(args.queue)
    try:
        return args.handler(queue, args)
    except ReviewError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: cannot access the review queue: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
