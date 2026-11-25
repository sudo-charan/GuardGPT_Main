"""Classify labelled sample prompts and print per-label accuracy.

    python evaluate_intent.py                 # all sample sets
    python evaluate_intent.py --set fresh     # one set
    python evaluate_intent.py --verbose       # every prompt with its sources
    python evaluate_intent.py --prompt "..."  # one prompt through Layers 1, 2, JEV and 3
    python evaluate_intent.py --json out.json # also write a machine-readable report (dashboard)

Runs Layer 1 (input) -> Layer 2 -> JEV -> Layer 3 in-process; no Ollama needed.
"""
import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
SAMPLES = ROOT / "tests" / "data" / "intent_samples.json"


def run_layers(prompt):
    from core import runtime
    from core.sensitive_data import redact

    clean = redact(prompt)
    embedding = runtime.intent_layer().embed(clean.text if clean.detected else prompt)
    guard = runtime.input_guard().check(prompt, embedding=embedding)
    intent = runtime.intent_layer().analyze(guard.redacted_prompt, guard.findings, embedding=embedding)
    jev = runtime.jev_model().evaluate(intent, guard, (), prompt=guard.redacted_prompt)
    decision = runtime.policy_engine().decide(guard, intent, jev, guard.redacted_prompt)
    return guard, intent, jev, decision


def _argmax(distribution):
    return max(distribution, key=distribution.get) if isinstance(distribution, dict) and distribution else None


def set_report(rows):
    """Machine-readable results of one sample set (used by the dashboard's Evaluation tab)."""
    def accuracy(key):
        scored = [r for r in rows if r[key] is not None]
        return round(sum(r[key] == r["gold"] for r in scored) / len(scored), 4) if scored else None

    labels = sorted({r["gold"] for r in rows})
    safe_vs_unsafe = lambda key: round(sum((r[key] != "safe") == (r["gold"] != "safe") for r in rows) / len(rows), 4)
    confusion = defaultdict(Counter)
    for r in rows:
        confusion[r["gold"]][r["ensemble"]] += 1
    return {
        "n": len(rows),
        "accuracy": {"ensemble": accuracy("ensemble"), "head": accuracy("head"), "anchors": accuracy("anchors"),
                     "dataset": accuracy("dataset"), "jev": accuracy("jev")},
        "safe_vs_unsafe": {"ensemble": safe_vs_unsafe("ensemble"), "jev": safe_vs_unsafe("jev"),
                           "decision": round(sum((r["action"] != "ALLOW") == (r["gold"] != "safe") for r in rows)
                                             / len(rows), 4)},
        "per_label": {label: {"n": sum(r["gold"] == label for r in rows),
                              **{key: sum(r["gold"] == label and r[key] == label for r in rows)
                                 for key in ("ensemble", "head", "anchors", "jev")}} for label in labels},
        "confusion": {gold: dict(counts) for gold, counts in confusion.items()},
        "errors": [{"gold": r["gold"], "predicted": r["ensemble"], "confidence": r["confidence"],
                    "action": r["action"], "prompt": r["prompt"]} for r in rows if r["ensemble"] != r["gold"]][:40],
        "false_positives": [r["prompt"] for r in rows if r["gold"] == "safe" and r["action"] != "ALLOW"][:25],
        "missed_attacks": [r["prompt"] for r in rows if r["gold"] != "safe" and r["action"] == "ALLOW"][:25],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="sample_set", help="tuning, holdout or fresh")
    parser.add_argument("--prompt")
    parser.add_argument("--json", dest="json_path", help="also write a machine-readable report to this file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    if args.prompt is not None:
        guard, intent, jev, decision = run_layers(args.prompt)
        print(json.dumps({"layer1_input": {"status": guard.status, "findings": guard.findings},
                          "layer2_intent": {"intent": intent.intent, "confidence": intent.confidence,
                                            "passed_threshold": intent.passed_threshold,
                                            "probabilities": intent.probabilities,
                                            "sources": {k: v for k, v in intent.sources.items()
                                                        if k in {"head", "anchors", "dataset", "guard"}}},
                          "jev": jev.as_dict(), "layer3_policy": decision.as_dict()}, indent=2))
        return 0

    data = json.loads(SAMPLES.read_text(encoding="utf-8"))
    sets = [args.sample_set] if args.sample_set else [k for k in data if not k.startswith("_")]
    report = {"sets": {}, "jev_backend": None}
    started = time.time()
    for name in sets:
        per_label = defaultdict(Counter)
        actions = Counter()
        rows = []
        for gold, prompts in data[name].items():
            for prompt in prompts:
                guard, intent, jev, decision = run_layers(prompt)
                ok = intent.intent == gold
                report["jev_backend"] = jev.backend
                rows.append({"gold": gold, "prompt": prompt, "ensemble": intent.intent,
                             "confidence": round(intent.confidence, 3),
                             "head": _argmax(intent.sources.get("head")),
                             "anchors": _argmax(intent.sources.get("anchors")),
                             "dataset": _argmax(intent.sources.get("dataset")),
                             "jev": jev.label, "action": decision.action})
                per_label[gold]["ok" if ok else "miss"] += 1
                actions[(gold == "safe", decision.action)] += 1
                if args.verbose or not ok:
                    print(f"  {'OK  ' if ok else 'MISS'} {gold:20} -> {intent.intent:20} "
                          f"{intent.confidence:.2f}  JEV {jev.verdict:5} Policy {decision.action:5} | {prompt[:60]}")
        total = sum(sum(c.values()) for c in per_label.values())
        correct = sum(c["ok"] for c in per_label.values())
        print(f"\n[{name}] intent accuracy {correct}/{total} = {correct / total:.0%}")
        for label, counts in per_label.items():
            n = sum(counts.values())
            print(f"    {label:20} {counts['ok']}/{n}")
        safe_blocked = actions[(True, "BLOCK")]
        safe_flagged = actions[(True, "FLAG")]
        unsafe_allowed = actions[(False, "ALLOW")]
        print(f"    safe prompts blocked: {safe_blocked}, flagged: {safe_flagged}; "
              f"unsafe prompts allowed: {unsafe_allowed}\n")
        report["sets"][name] = set_report(rows)
    if args.json_path:
        report["seconds"] = round(time.time() - started, 1)
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        target = Path(args.json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(report), encoding="utf-8")
        temporary.replace(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
