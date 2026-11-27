"""Ablation study: switch layers off one at a time and measure the final decision.

    python tests/test_cases/evaluate_ablation.py                  # every version, every sample set
    python tests/test_cases/evaluate_ablation.py --set fresh      # one sample set
    python tests/test_cases/evaluate_ablation.py --verbose        # also list the prompts each version gets wrong
    python tests/test_cases/evaluate_ablation.py --json out.json  # also write a machine-readable report

Every prompt runs through the real layers once (Layer 1, Layer 2, the fine-tuned JEV model
when JEV_STYLE_MODE=finetuned, the JEV rules and Layer 3). The versions below are then
built from those results by replacing the switched-off parts with neutral stand-ins:

    Layer 1 off : no findings, risk 0, no self-harm level (redaction stays on: it is privacy,
                  not detection). Layer 2 is re-run without Layer 1's findings.
    Layer 2 off : label "safe" with no probability on any unsafe label.
    JEV off     : verdict ALLOW with risk 0, so Layer 3 decides on Layer 1 and Layer 2 alone.

Layer 3 always runs, because it makes the final decision. A prompt counts as "stopped"
when the decision is FLAG or BLOCK (a supportive self-harm reply is a BLOCK). Single
prompts only, so session history and carried risk play no part. No Ollama needed, and
nothing is written to logs/. Project code is not changed: the stand-ins live here.
"""
import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# This file lives in tests/test_cases/; the project root is two folders up. Run it as a
# script (not with "python -m"), so tests/__init__.py does not switch JEV to builtin mode.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)
SAMPLES = ROOT / "tests" / "data" / "intent_samples.json"

# name -> (description, Layer 1, Layer 2, JEV)
#   Layer 1: "on" | "off"     Layer 2: "on" | "off"     JEV: "full" | "rules" | "off"
VERSIONS = {
    "full":           ("Full system", "on", "on", "full"),
    "no_layer1":      ("Without Layer 1 detection", "off", "on", "full"),
    "no_layer2":      ("Without Layer 2", "on", "off", "full"),
    "no_jev_model":   ("Without the fine-tuned JEV model (JEV rules only)", "on", "on", "rules"),
    "no_jev":         ("Without JEV", "on", "on", "off"),
    "layer1_only":    ("Layer 1 alone", "on", "off", "off"),
    "layer2_only":    ("Layer 2 alone (through the JEV rules)", "off", "on", "rules"),
    "jev_model_only": ("Fine-tuned JEV model alone", "off", "off", "full"),
}


def _neutral_guard(guard):
    from core.layers.contracts import SAFE, GuardVerdict

    return GuardVerdict("input", SAFE, 0.0, [], {}, redacted_prompt=guard.redacted_prompt,
                        pii_types=list(guard.pii_types), pii_span_count=guard.pii_span_count)


def _neutral_intent():
    from core.layers.contracts import IntentResult
    from core.layers.layer2_intent import CATEGORY_NAMES, LABEL_THRESHOLDS

    return IntentResult("safe", 0.0, {"safe": 1.0}, CATEGORY_NAMES["safe"], LABEL_THRESHOLDS["safe"],
                        False, "safe", {}, ["layer2_switched_off"])


def _neutral_jev():
    from core.layers.contracts import ALLOW, JEVResult

    return JEVResult(ALLOW, "safe", 0.0, 0.0, "safe", ["jev_switched_off"], model="none", backend="off")


class Runner:
    def __init__(self):
        from core import runtime

        self.runtime = runtime
        self.jev = runtime.jev_model()
        self.has_model = self.jev.mode == "finetuned"

    def evaluate(self, prompt):
        """Decision of every version for one prompt, plus the time each part took (seconds)."""
        from core.sensitive_data import redact

        rt, timing = self.runtime, {}
        started = time.perf_counter()
        clean = redact(prompt)
        embedding = rt.intent_layer().embed(clean.text if clean.detected else prompt)
        timing["embed"] = time.perf_counter() - started

        started = time.perf_counter()
        guard = {"on": rt.input_guard().check(prompt, embedding=embedding)}
        guard["off"] = _neutral_guard(guard["on"])
        timing["layer1"] = time.perf_counter() - started

        text = guard["on"].redacted_prompt
        started = time.perf_counter()
        intents = {("on", "on"): rt.intent_layer().analyze(text, guard["on"].findings, embedding=embedding)}
        timing["layer2"] = time.perf_counter() - started
        intents[("off", "on")] = rt.intent_layer().analyze(text, (), embedding=embedding)

        model = None
        if self.has_model:
            started = time.perf_counter()
            model = self.jev.finetuned_model().classify(text)
            timing["jev_model"] = time.perf_counter() - started

        results = {}
        for name, (_, layer1, layer2, jev_mode) in VERSIONS.items():
            if jev_mode == "full" and model is None:
                continue                                   # needs the fine-tuned model
            g = guard[layer1]
            intent = intents[(layer1, "on")] if layer2 == "on" else _neutral_intent()
            if jev_mode == "full":
                jev = self.jev._combine_finetuned(intent, g, (), *model)
            elif jev_mode == "rules":
                jev = self.jev._evaluate_builtin(intent, g, ())
            else:
                jev = _neutral_jev()
            decision = self.runtime.policy_engine().decide(g, intent, jev, text)
            seconds = timing["embed"] + (timing["layer1"] if layer1 == "on" else 0.0)
            seconds += timing["layer2"] if layer2 == "on" else 0.0
            seconds += timing.get("jev_model", 0.0) if jev_mode == "full" else 0.0
            results[name] = {"action": decision.action, "route": decision.route, "seconds": seconds}
        return results


def metrics(rows, name):
    """Safe-vs-unsafe metrics of one version. Positive = unsafe, caught = FLAG or BLOCK."""
    scored = [r for r in rows if name in r["versions"]]
    if not scored:
        return None
    stopped = lambda r: r["versions"][name]["action"] != "ALLOW"
    unsafe = [r for r in scored if r["gold"] != "safe"]
    safe = [r for r in scored if r["gold"] == "safe"]
    tp = sum(stopped(r) for r in unsafe)
    fp = sum(stopped(r) for r in safe)
    blocked = sum(r["versions"][name]["action"] == "BLOCK" for r in unsafe)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(unsafe) if unsafe else 0.0
    return {
        "n": len(scored), "unsafe": len(unsafe), "safe": len(safe),
        "unsafe_caught": tp, "unsafe_blocked": blocked, "unsafe_missed": len(unsafe) - tp,
        "safe_stopped": fp,
        "recall": round(recall, 4),
        "block_rate": round(blocked / len(unsafe), 4) if unsafe else 0.0,
        "false_positive_rate": round(fp / len(safe), 4) if safe else 0.0,
        "precision": round(precision, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "accuracy": round((tp + len(safe) - fp) / len(scored), 4),
        "ms_per_prompt": round(1000 * sum(r["versions"][name]["seconds"] for r in scored) / len(scored), 1),
        "missed": [r["prompt"] for r in unsafe if not stopped(r)],
        "false_positives": [r["prompt"] for r in safe if stopped(r)],
    }


def _changes(rows, name):
    """Prompts a version gets wrong that the full system gets right, and the reverse."""
    worse, better = [], []
    for r in rows:
        if name not in r["versions"] or "full" not in r["versions"]:
            continue
        right = lambda version: (r["versions"][version]["action"] != "ALLOW") == (r["gold"] != "safe")
        if right("full") and not right(name):
            worse.append(r["prompt"])
        elif right(name) and not right("full"):
            better.append(r["prompt"])
    return worse, better


def print_table(title, rows):
    print(f"\n=== {title} ({len(rows)} prompts: {sum(r['gold'] != 'safe' for r in rows)} unsafe, "
          f"{sum(r['gold'] == 'safe' for r in rows)} safe) ===")
    print(f"{'Version':52} {'Unsafe caught':>14} {'Blocked':>8} {'Safe stopped':>13} {'Accuracy':>9} "
          f"{'F1':>6} {'ms/prompt':>10}")
    for name, (description, *_) in VERSIONS.items():
        m = metrics(rows, name)
        if m is None:
            print(f"{description:52} {'n/a (needs JEV_STYLE_MODE=finetuned)':>40}")
            continue
        print(f"{description:52} {m['unsafe_caught']:>4}/{m['unsafe']:<3} {m['recall']:>5.0%} "
              f"{m['block_rate']:>8.0%} {m['safe_stopped']:>4}/{m['safe']:<3}{m['false_positive_rate']:>5.0%} "
              f"{m['accuracy']:>9.1%} {m['f1']:>6.3f} {m['ms_per_prompt']:>10.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="sample_set", help="tuning, holdout or fresh")
    parser.add_argument("--json", dest="json_path", help="also write a machine-readable report to this file")
    parser.add_argument("--verbose", "-v", action="store_true", help="list the prompts each version gets wrong")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    data = json.loads(SAMPLES.read_text(encoding="utf-8"))
    sets = [args.sample_set] if args.sample_set else [k for k in data if not k.startswith("_")]
    runner = Runner()
    print(f"JEV mode: {runner.jev.mode}. Loading models and running {sum(len(p) for s in sets for p in data[s].values())} "
          "prompts...", flush=True)
    rows_by_set, started = defaultdict(list), time.time()
    for name in sets:
        for gold, prompts in data[name].items():
            for prompt in prompts:
                rows_by_set[name].append({"set": name, "gold": gold, "prompt": prompt,
                                          "versions": runner.evaluate(prompt)})
    every = [r for name in sets for r in rows_by_set[name]]
    for name in sets:
        print_table(f"set: {name}", rows_by_set[name])
    if len(sets) > 1:
        print_table("all sets", every)

    focus = rows_by_set["fresh"] if "fresh" in rows_by_set else every
    focus_name = "fresh" if "fresh" in rows_by_set else "all sets"
    print(f"\nCompared with the full system ({focus_name}): prompts each version now gets wrong / now gets right")
    for name, (description, *_) in VERSIONS.items():
        if name == "full" or metrics(focus, name) is None:
            continue
        worse, better = _changes(focus, name)
        print(f"  {description:52} worse on {len(worse):2}, better on {len(better):2}")
        if args.verbose:
            for prompt in worse:
                print(f"      - worse: {prompt[:90]}")
            for prompt in better:
                print(f"      + better: {prompt[:90]}")
    print(f"\nFirst prompt includes model loading; time per prompt is measured per part and summed per version.")
    print(f"Done in {time.time() - started:.0f} s. Small sets: one prompt moves 'fresh' by about "
          f"{100 / max(1, len(rows_by_set.get('fresh', every))):.1f} points.")

    if args.json_path:
        report = {"generated_at": datetime.now(timezone.utc).isoformat(), "jev_mode": runner.jev.mode,
                  "versions": {name: description for name, (description, *_) in VERSIONS.items()},
                  "sets": {name: {v: metrics(rows, v) for v in VERSIONS} for name, rows in rows_by_set.items()},
                  "all": {v: metrics(every, v) for v in VERSIONS}}
        target = Path(args.json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"Report written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
