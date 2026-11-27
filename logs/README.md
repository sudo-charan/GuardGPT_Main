# Logs

The `.jsonl` files are JSON Lines; the `.json` files hold a single object. The pipeline's
request, trace and report logs are written with single atomic appends so concurrent
requests can no longer interleave and corrupt lines (the old logs had this).
The CLI and the optional dashboard (`python dashboard.py`) read these same files.

| File | Contents |
|---|---|
| `guardgpt_trace.jsonl` | Audit trail. One event per stage, all sharing a `trace_id`: `request`, `security_event` (Layer 1 input), `intent` (Layer 2), `jev` (with the chat's `carried_risk` before this turn), `policy_decision` (Layer 3), `tool_call` (agent tools; `via: mcp` for calls on the tools server), `output_guard` (Layer 1 output, with any `repairs` such as a closed code fence), `review_queued` and `escalation` (FLAG outcomes and explicit crisis cases), `response`, and `error` on failure. No prompt text. |
| `guardgpt_audit.jsonl` | One compact record per request: ids, topic, intent, confidence, JEV verdict, action, final status, reason codes. No prompt text. |
| `guardgpt_prompt.jsonl` | Detailed per-request report, including the prompt (redacted) and per-layer results: topic, Layer 2 intent with its `context` (task, follow-up, topic shift, session signal), JEV, policy, agent, output guard and review. |
| `guardgpt_session.jsonl` | Session reports: turns, intent history, topic history, decayed session risk, and `carried_risk` (risk remembered from earlier blocked turns, which the JEV reads). |
| `guardgpt_complete.jsonl` | Prompt report + session report + final result, linked by ids. |
| `guardgpt_review.jsonl` | Review queue: FLAG outcomes, carried-risk blocks and explicit crisis cases (`PENDING` / `APPROVED` / `REJECTED`) with `priority`, `escalated` and `escalation_reasons`; managed with `review_cli.py` or the dashboard. |
| `guardgpt_escalations.jsonl` | Escalated reviews (priority `high` / `urgent`), including explicit crisis cases: review id, priority, reasons, intent, risk level. No prompt text. An urgent one also raises the `urgent_escalation` metrics alert. |
| `guardgpt_feedback.jsonl` | Dashboard feedback: which requests you marked correct or wrong and the label they should have had. Trace ids, labels and any note you type; no prompt text is copied. |
| `guardgpt_alerts_ack.json` | The metrics counters at the moment you last acknowledged alerts in the dashboard, so an acknowledged alert stays cleared until something new happens. |
| `guardgpt_eval.json` | The last result of the dashboard's Evaluation tab (accuracy per source, per label, confusion matrix). |
| `guardgpt_requests.jsonl` | Per-request timings (per stage and total) and outcome. |
| `guardgpt_errors.jsonl` | Failed stages (for example `jev` when the fine-tuned model is missing, or `escalation_write`). |
| `guardgpt_metrics.json` | Counter snapshot: requests, actions, statuses, errors, escalations. Alerts (JEV or audit failures, error rate, urgent escalation) are derived from it. |

To follow one request: `grep <trace_id> logs/guardgpt_trace.jsonl`, or open it in the
dashboard's Traces tab.

The test suite and the evaluation scripts (`evaluate_intent.py`,
`tests/test_cases/evaluate_ablation.py`) never write here: tests use a temporary folder,
and the evaluation scripts only print results.

## Alerts

`python main.py --status` and the dashboard derive alerts from the counters in
`guardgpt_metrics.json`. The counters are cumulative, so an alert such as `jev_failures`
stays after the cause is fixed (for example, errors from a past memory shortage). Use
**Acknowledge** on the dashboard's Overview tab; the alert then returns only if something
new happens.

## Reading support replies

A supportive self-harm reply is recorded with action `BLOCK` and final status `SUPPORT`.
The `action:block` counter in `guardgpt_metrics.json` therefore includes support replies;
`status:support` counts them separately. An explicit crisis is also added to the review
queue as `urgent` (stage `support`, reason `crisis_support`) and to the escalation log;
other support replies are not.

## Sensitive data

Request prompt text is stored in `guardgpt_prompt.jsonl` and `guardgpt_complete.jsonl`.
It is redacted before it is logged, including over-long input and prompts that fail
during Layer 1. Redaction covers emails, cards, US social security numbers, US and Indian
phone numbers, Aadhaar numbers, API keys, JWTs and private keys. Other personal data,
such as names and addresses, is not recognised.

Treat this folder as sensitive runtime data, and remember the dashboard shows that prompt
text too.
