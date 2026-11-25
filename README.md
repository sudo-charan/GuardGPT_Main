# GuardGPT

A local, layered safety guard for prompts and answers. Every request goes through
an MCP server that hosts the three guard layers, the JEV model and a LangGraph
agent. The agent runs its tools (Ollama / Llama 3.2, output review, redaction) on a
second MCP server, and one trace id follows the request through every stage.

## Architecture

```text
User / CLI (main.py)
      |
      v
MCP client (agent/mcp_client.py)  ->  MCP server (mcp_server/server.py)   <- backbone
      |                                 registry, router, policy hooks
      v
Layer 1 Guard - input check (pre-generation)          core/layers/layer1_guard.py
      |   injection / jailbreak / harm / coercion patterns, PII + secrets, self-harm assessor
      |   -> SAFE | FLAG | UNSAFE + findings + risk score
      v
Layer 2 Intent Intelligence                           core/layers/layer2_intent.py
      |   ensemble: trained MiniLM (BERT) head + curated anchors + FAISS kNN + Layer 1 evidence
      |   -> label in {safe, prompt_injection, jailbreak, harmful_instructions,
      |               manipulation, self_harm_risk} + confidence + distribution
      |   + topic (what it is about) and context / task understanding
      |     (task, follow-up, topic shift, session signal); informational only
      v
JEV model (intent + confidence [+ prompt] -> verdict)  core/layers/jev.py
      |   builtin rules on intent + confidence, or your fine-tuned JEV checkpoint
      |   (JEV_STYLE_MODE=finetuned), which also reads the prompt; the rule verdict is a floor
      v
Layer 3 Guard / Policy Engine                         core/layers/layer3_policy.py
      |   ALLOW -> agent      FLAG -> review queue (priority + escalation)
      |   BLOCK -> model-written refusal
      |   self-harm -> supportive reply (explicit crisis = fixed vetted reply)
      |   emotional distress -> normal answer in a caring "wellbeing" mode
      v
LangGraph agent (agent/graph.py)
      |   route -> select_tool -> execute_tool
      |   tools run on the tools MCP server (mcp_server/tools_server.py):
      |     redact_text -> ollama_generate | ollama_support   (Ollama / Llama 3.2)
      v
Layer 1 Guard - output check (post-generation)
      |   deterministic checks + model reviewer (ollama_review)
      |   SAFE -> release   FLAG -> review   UNSAFE -> regenerate (2 attempts in total) then block
      |   support replies: never held for review; if not released, a fixed fallback is sent
      v
Response

Cross-cutting: logs/ - trace (one event per stage, same trace_id), audit, metrics, alerts.
```

Layers talk through typed contracts in `core/layers/contracts.py`
(`GuardVerdict`, `IntentResult`, `JEVResult`, `PolicyDecision`).

## Setup

The model weights (`*.pt`) are stored with [Git LFS](https://git-lfs.com). Run
`git lfs install` once before cloning, otherwise those files are only small text pointers.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS: `bash setup.sh`. Python 3.11+ is recommended. Ollama must be
running with `llama3.2` pulled for answers (`--check` does not need it).

Runtime files expected in the project:

- `data/guardgpt_dataset.jsonl`, `data/guardgpt_faiss.index`, `data/guardgpt_id_map.json`
- `intent_classifier/best_model.pt`, `tokenizer.json`, `config.json`, `intent_anchors.json`
- `intent_classifier/topic_anchors.json`, `task_anchors.json` (topic and task examples)
- for the fine-tuned JEV: `intent_classifier/jev_finetuned/best_model.pt` (see below)

The fine-tuned JEV paths in `.env` are full paths. If you move or rename the project
folder, update them (see [Troubleshooting](#troubleshooting)).

The embedding model `all-MiniLM-L6-v2` is loaded from the HuggingFace cache,
or from a folder set in `GUARDGPT_EMBEDDING_MODEL`.

## Running

```bash
python main.py --chat                 # interactive, session-aware (/session for a report)
python main.py --prompt "..."         # one audited answer
python main.py --check --prompt "..." # input layers only, no Ollama
python main.py --pipeline             # demo: one prompt per intent label
python main.py --status               # data files, JEV config, Ollama models
python main.py --chat --verbose       # full layer line, distribution, reasons
python evaluate_intent.py             # per-label accuracy on tests/data/intent_samples.json
python evaluate_intent.py --prompt "..."
python evaluate_intent.py --json out.json   # also write a machine-readable report
python review_cli.py list             # FLAG review queue (with priority)
python review_cli.py escalated        # pending escalated reviews, most urgent first
python dashboard.py                   # optional live web monitor (see below)
python run_tests.py                   # full test suite
```

Each answer prints the reply on the `GuardGPT:` line, then a short summary:

```text
[ALLOW | SAFE]
Topic: coding | Safety: safe 84% | Decision: ALLOW
Trace: 1796c4f9
```

A supportive self-harm reply currently shows as `[BLOCK | SUPPORT]` with
`Decision: BLOCK`. The message was answered: `BLOCK` here means the normal answer path
was replaced by a support reply.

A chat session remembers the last 8 answered exchanges, so a request such as "summarise
this chat" covers only those. After changing `.env`, restart the chat and the dashboard;
running programs keep the old settings.

`--verbose` prints the full line instead, plus the intent distribution and reasons:
`L1 input: SAFE | Topic: coding 98% | Context: code, follow-up | Safety intent: safe 84.1% | JEV: ALLOW (risk low) [model: safe 99%] | Policy: ALLOW | L1 output: SAFE`.
Search the trace id in `logs/guardgpt_trace.jsonl` (the short id is a prefix).

**Topic, task and safety intent.** *Safety intent* is the six-label Layer 2 result that
drives allow/flag/block. *Topic* is what the message is about (greeting, coding,
cybersecurity, technology, education, math, health_wellbeing, emotional_support,
writing, summarization, daily_life, conversation, or `general`), from
`core/layers/topic_intent.py` and `intent_classifier/topic_anchors.json`. *Context*
(`core/layers/context_understanding.py`, `task_anchors.json`) adds the task (question,
how_to, creation, code, transformation, advice, chat, personal_support), whether the
message follows up on the previous turn, a topic change, and a session trend
(clean / watch / escalating). Topic and context are informational only and never change
a decision; add a line to the anchor files to fix a wrong label, no retraining needed.

### Web dashboard (optional)

A live monitor. Chat stays in the CLI; the dashboard shows what happens.

```powershell
# terminal 1:  chat
python main.py --chat

# terminal 2:  monitor
.\.venv\Scripts\Activate.ps1
python dashboard.py              # opens http://127.0.0.1:8080
python dashboard.py --no-browser # do not open a browser window
python dashboard.py --port 9000
```

Stop it with Ctrl+C. It starts no model and no MCP server (it opens in under a second)
and reads the same `logs/` folder as the CLI. New requests are pushed to the open page
within about a second, with a toast (and an optional desktop notification) for an urgent
escalation. Tabs:

| Tab | What it does |
|---|---|
| **Overview** | Counters, alerts (with Acknowledge), traffic-over-time chart (per hour or day) with latency, safety-intent and topic charts, recent requests, and a printable **summary report**. |
| **Requests** | Every request, with search, filters (decision, safety intent, topic, session), paging and CSV export. |
| **Sessions** | One row per chat session; click for the conversation as a timeline with the session risk. |
| **Reviews** | The FLAG queue, most urgent first, with the redacted prompt, age, priority filter, notes, bulk Approve / Reject and CSV export. |
| **Feedback** | Requests you marked correct or wrong; **Export anchor lines** turns the wrong ones into lines for `intent_anchors.json`. |
| **Evaluation** | Runs `evaluate_intent.py` as a background job and shows accuracy for the head, anchors, dataset, ensemble and JEV, a per-label table and a confusion matrix. Quick = rules only; Full = with the fine-tuned JEV (minutes). |
| **System** | Readiness, installed Ollama models, configuration, package versions, log files and sizes, and **Trim old logs** (never touches reviews, escalations, feedback or metrics, and refuses while a chat is writing). |
| **Traces** | All events of one request by trace id. |

The **Block** count on the Sessions tab includes supportive replies; open the session
timeline to see which turns were `SUPPORT`.

Click any request for the **detail panel**: the five pipeline stages, a table of what the
ensemble, head, anchors and dataset each predicted, a timing bar per stage, the findings and
reasons, and buttons to mark the safety intent correct or wrong. Handy extras: **Hide text**
blurs prompt text for demos, a theme switch, deep links (`#sessions`, `#request/<trace id>`)
and keyboard shortcuts (1-8 tabs, `/` search, `r` refresh, `Esc` close).

The header shows Ollama, intent model, JEV mode and data readiness. It listens on
127.0.0.1 only, every API call needs a per-run token embedded in the page, and it shows
the redacted prompt text from `logs/`, so treat it like the logs. Feedback stores only trace
ids and labels; the evaluation job and trimming write to `logs/` (`guardgpt_feedback.jsonl`,
`guardgpt_alerts_ack.json`, `guardgpt_eval.json`).

## The JEV model

`JEV_STYLE_MODE` picks the backend:

| Mode | What it does |
|---|---|
| `builtin` (default) | Rules over the Layer 2 label, confidence and Layer 1 findings. |
| `finetuned` | Runs your fine-tuned JEV checkpoint on the prompt, and combines its verdict with the Layer 2 intent + confidence. The builtin verdict is kept as a floor, so inside JEV the model can only make a verdict stricter (Layer 3 may still relax it, see [Known issues](#known-issues)). |
| `server` / `auto` | External Jev-Style HTTP server (`auto` falls back to builtin). |

The builtin rules also use the session: after two or more blocked turns in the last
three, a FLAG becomes a BLOCK, and a safe verdict is flagged only if one single unsafe
label reaches `JEV_FLAG_PEAK`. Supportive self-harm turns are also counted as blocked
turns here.

In `finetuned` mode the question and class descriptions come from the checkpoint itself
(`jev_adapter_guardgpt.py` reads them). The model was trained with that wording, so
change thresholds or rules rather than the wording unless you retrain.

### Using the fine-tuned checkpoint

The checkpoint (`intent_classifier/jev_finetuned/best_model.pt`, about 1.5 GB, a
Qwen3.5 0.8B model) needs the base model's config, tokenizer and readout files. Use the
Jev-Style base model (`chaoliangUNSW/Jev-Style-0.8B-Decision-v3` on Hugging Face); the
folder must contain `config.json`, `tokenizer.json`, `readout_config.json` and
`jev_style_decision.py`. Point GuardGPT at them in `.env`:

```ini
JEV_STYLE_MODE=finetuned
JEV_CHECKPOINT=C:/path/to/GuardGPT/intent_classifier/jev_finetuned/best_model.pt
JEV_BASE_MODEL_DIR=C:/Users/<you>/.cache/huggingface/hub/models--chaoliangUNSW--Jev-Style-0.8B-Decision-v3/snapshots/<hash>
JEV_ADAPTER=C:/path/to/GuardGPT/jev_adapter_guardgpt.py
JEV_DEVICE=cpu                              # or cuda
```

`jev_adapter_guardgpt.py` loads the checkpoint into the official Jev-Style engine and
asks the question and class descriptions stored in the checkpoint, so scoring matches
training (about 82% exact label and 96% safe-vs-unsafe on the project's evaluation
prompts). `jev_adapter_example.py` is the template if you train a different model.
On CPU, loading takes about 20 seconds and each prompt about 2 to 3 seconds.

If the checkpoint is missing or fails to load, requests are BLOCKED with
`final_status=ERROR` and reason `jev_failed`, and the error says what is
missing. It never falls back to "safe" silently. `python main.py --status`
shows the JEV configuration and any problems.

## Review and escalation

A FLAG goes to the review queue (`logs/guardgpt_review.jsonl`) with a priority:

| Priority | When |
|---|---|
| `urgent` | critical risk, or a self-harm signal |
| `high` | high risk, two or more flagged/blocked turns in the last four, or a risky session |
| `normal` | everything else |

`high` and `urgent` are escalated: written to `logs/guardgpt_escalations.jsonl`, counted
in the metrics (an urgent one raises the `urgent_escalation` alert) and listed first by
`python review_cli.py escalated`. The user is told whether the request was sent for
review or escalated to a human reviewer. Resolving a review (`review_cli.py resolve`, or
the dashboard) records the decision only; it never releases the withheld answer.

Only FLAG outcomes are queued. A supportive self-harm reply, including an explicit
crisis, is not sent to the review queue or escalated.

## Tools MCP server

The agent never calls Ollama directly. Its tools run on a second MCP server
(`mcp_server/tools_server.py`), called over the MCP protocol by `agent/tool_client.py`:

| Tool | Purpose |
|---|---|
| `redact_text` | Security tool: replaces PII and secrets before anything reaches the model |
| `ollama_generate`, `ollama_support` | Answer and crisis-support generation |
| `ollama_review` | Structured output review (the auditor) |

The response composer (refusals, notices) and the output auditor use the same tools.
The server starts on first use (loopback, free port) and exits when the main server
exits. It refuses calls without a per-launch secret token, because raw generation must
not be reachable without Layers 1-3. Tool calls pass an allowlist and size limits.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GUARDGPT_MCP_URL` | `http://127.0.0.1:8000/mcp` | MCP endpoint (auto-started by the CLI) |
| `GUARDGPT_TOOLS_URL`, `GUARDGPT_TOOLS_TOKEN` | auto | Tools MCP server (normally started and secured automatically) |
| `GUARDGPT_DASHBOARD_PORT` | `8080` | Dashboard port |
| `GUARDGPT_EMBEDDING_MODEL` | HF cache | Local MiniLM folder |
| `GUARDGPT_ALLOW_MODEL_DOWNLOAD` | off | Set to `1` to let the embedding model download |
| `GUARDGPT_L2_WEIGHTS` | `head=0.2,anchors=0.6,dataset=0.2` | Layer 2 ensemble weights |
| `GUARDGPT_LOG_DIR` | `logs/` | Log directory |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `llama3` (`.env.example` sets `http://127.0.0.1:11434` / `llama3.2`) | Generation |
| `GUARDGPT_DATASET` | `data/guardgpt_dataset.jsonl` | Dataset file |
| `GUARDGPT_INTENT_MODEL` | `intent_classifier/best_model.pt` | Trained intent head |
| `OLLAMA_AUDIT_MODEL` | generation model | Output reviewer model |
| `OLLAMA_TIMEOUT`, `OLLAMA_TEMPERATURE` | `120`, `0.2` | Generation limits |
| `JEV_STYLE_MODE` | `builtin` | `builtin`, `finetuned`, `server`, `auto` |
| `JEV_CHECKPOINT`, `JEV_BASE_MODEL_DIR`, `JEV_ADAPTER`, `JEV_DEVICE` | unset | Fine-tuned JEV |
| `JEV_STATE_FORMAT` | `text` | How the adapter passes the prompt (`text` or `json`) |
| `JEV_MODEL_MIN_CONFIDENCE` | `0.55` | Fine-tuned label counts only above this |
| `JEV_BLOCK_THRESHOLD` | `0.75` | Uncorroborated unsafe label needs this to BLOCK |
| `JEV_FLAG_THRESHOLD` | `0.50` | Unsafe probability mass that still FLAGs |
| `JEV_FLAG_PEAK` | `0.40` | "safe" with an unsafe label this close becomes FLAG |
| `JEV_STYLE_URL`, `JEV_STYLE_TIMEOUT` | `http://127.0.0.1:8765`, `300` | External Jev-Style server (`server` / `auto`) |

## MCP services

Public (on the main server): `health`, `complete_request`, `session_report`,
`input_guard`, `intent_analysis`, `jev_evaluate`, `policy_decision`, `review_queue`.
`complete_request` is the canonical path; the layer services expose single layers for
debugging and run the same objects as the pipeline.

Internal (tools server, token-protected): `redact_text`, `ollama_generate`,
`ollama_support`, `ollama_review`.

## Layout

```text
main.py                    CLI (talks to the MCP server)
dashboard.py, dashboard_data.py, dashboard_ui/  optional live web dashboard (API, data layer, page)
mcp_server/                main MCP server: server, services, registry, router, policy hooks
mcp_server/tools.py        tool layer (registry, hooks) for the agent's tools
mcp_server/tools_server.py tools MCP server (Ollama, review, redaction)
core/layers/               Layer 1, Layer 2, topic + context, JEV, Layer 3, contracts
core/complete_pipeline.py  runs the layers and writes logs
core/escalation.py         review priority and escalation
core/output_checks.py      deterministic output checks and markdown repair
core/runtime.py            shared singletons
agent/graph.py             LangGraph agent
agent/mcp_client.py        MCP client
agent/tool_client.py       client for the tools MCP server
intent_classifier/         MiniLM head, tokenizer, anchors, jev_finetuned/
data/                      dataset + FAISS index
tests/                     unit, flow, accuracy and dashboard tests
jev_adapter_guardgpt.py    adapter for the fine-tuned JEV checkpoint
jev_adapter_example.py     template for another JEV adapter
```

## Tests

`python run_tests.py` runs the offline suite (231 tests, under a minute). It is
pinned to the builtin JEV (`tests/__init__.py`), so it does not load the 1.5 GB model;
the model-based accuracy tests need the embedding model and the trained head and are
skipped if they are missing. No Ollama server is needed. The accuracy tests also use the
dataset when it is present, so run them with the data files in place.

## Troubleshooting

**Every message is blocked with "The JEV model is unavailable"** (`final_status=ERROR`,
`jev_failed`). `JEV_CHECKPOINT`, `JEV_BASE_MODEL_DIR` or `JEV_ADAPTER` points to a path
that does not exist, often after the project folder was moved or renamed. Fix the paths in
`.env`, check them with `python main.py --status`, then restart the chat and the dashboard.

**`✗ System ready`, or `--status` reports `"dataset": false`.** A data file is missing.
Windows Defender has quarantined `data/guardgpt_dataset.jsonl` and
`data/guardgpt_id_map.json` as `Exploit:Python/Leivion.A`. Check **Windows Security →
Virus & threat protection → Protection history**, and restore them from there or from your
project archive. Defender may quarantine them again. Allow the files or exclude the
`data/` folder only if you trust where the dataset came from, since that reduces
protection. The chat still runs without them, but Layer 1 and Layer 2 lose their dataset
evidence.

## Known issues

Open problems as of 2026-11-25. They are not fixed yet.

**Safety and privacy**

- A support reply written by the model is not checked for refusals or phone numbers. In
  testing, one told the user it could not continue the conversation, and replies quoted
  country-specific helplines without knowing where the user was.
- In `finetuned` mode, Layer 3 can turn a JEV `BLOCK` into `ALLOW` when Layer 2 also sees
  a mild self-harm signal.
- A defensive word anywhere in a prompt (for example "avoid" or "prevent") switches off
  several Layer 1 harm detectors. Later layers may still catch the prompt.
- Slightly reworded injection phrases can miss the Layer 1 patterns.
- Over-long input, and prompts that fail during Layer 1, are written to the prompt and
  complete logs **without redaction**.
- Redaction covers only some phone formats: Indian formats such as 10 unbroken digits or
  `+91 98765 43210` are missed, and Aadhaar numbers are not covered.

**Wrong decisions**

- After an earlier risk in the session, the word "tonight" or "plan" alone triggers the
  fixed crisis reply (for example "help me plan my studies").
- Supportive turns count as blocked turns, so after two of them in the last three turns, a
  harmless message that would only be flagged is refused.
- Ordinary loneliness or stress can be sent to the support route, which then marks the
  session as in crisis.

**Answers withheld**

- The output reviewer accepts at most 6,000 bytes of prompt plus answer, so long answers
  are held for review.
- A safe answer that opens with "I can't..." is held, and one containing an example email
  address is blocked.

## Known limitations

- The trained intent head is weak on its own (about 58% on the project's evaluation
  prompts); the anchors carry most of Layer 2, and the dataset has a few duplicate and
  conflicting rows (see `data/README.md`).
- The fine-tuned JEV can over-flag harmless security-education prompts, and is weaker on
  jailbreak and manipulation labels.
- Local Llama 3.2 answers vary in quality; the reviewer checks safety, not facts.
- Replies may contain placeholders such as `<EMAIL_1>`, because original PII is never
  returned.

## Logs

See `logs/README.md`. Request prompt text is stored in `guardgpt_prompt.jsonl` and
`guardgpt_complete.jsonl` (and shown in the dashboard). It is normally redacted, with the
exception listed under [Known issues](#known-issues). Treat `logs/` as sensitive runtime
data.
For a plain-language explanation of how everything works and what each file does, see
`PROJECT_GUIDE.md`.
