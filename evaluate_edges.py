"""Run real multi-turn routing (not ASR/TTS). Gold labels never enter model context."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import urllib.request

import app


def check_turn(gold, response):
    trace = response["trace"]
    routes = [item["scenario_id"] for item in trace["scenarios"]]
    state = trace["dialog_state"]
    checks = {"primary": routes[0] == gold["routes"][0], "full_set": set(routes) == set(gold["routes"]),
              "order": routes == gold["routes"]}
    for key, value in (("language", trace["language"]), ("continuation", trace["is_continuation"]),
                       ("clarify", trace.get("needs_clarification", False)), ("handoff", response["handoff"]),
                       ("queue", state["pending_queue"])):
        if key in gold:
            checks[key] = value == gold[key]
    if "stack_contains" in gold:
        checks["stack"] = set(gold["stack_contains"]) <= {sid for group in state["stack"] for sid in group}
    if "closed" in gold:
        checks["closed"] = set(gold["closed"]) <= set(trace.get("closed_scenarios", []))
    if "slots" in gold:
        checks["slots"] = all(trace["slots"].get(k) == v for k, v in gold["slots"].items())
    if gold.get("no_execute"):
        checks["no_execute"] = not any(a["mode"] == "execute" for a in trace["actions"])
    return checks


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Optional running server URL; otherwise calls the local implementation with OPENAI_API_KEY")
    parser.add_argument("--dataset", type=Path, default=app.DATA / "edge_dialogues.json")
    parser.add_argument("--output", type=Path, default=app.DATA / "edge_report.json")
    parser.add_argument("--case", help="Run only a specific dialogue id")
    args = parser.parse_args()
    url = args.url.rstrip("/") if args.url else None
    if url:
        with urllib.request.urlopen(url + "/api/config", timeout=10) as response:
            config = json.load(response)
        if not config.get("llm_enabled") or config.get("router_version") != "edges-v4":
            parser.error("Restart the server with the new code and OPENAI_API_KEY. Demo/stale server results are not an LLM evaluation.")
    elif not os.getenv("OPENAI_API_KEY", "").strip():
        parser.error("OPENAI_API_KEY is required. Offline regressions: python -B -m unittest -v test_router")
    dialogues = json.loads(args.dataset.read_text(encoding="utf-8"))["dialogues"]
    if args.case:
        dialogues = [case for case in dialogues if case["id"] == args.case]
    if not dialogues:
        parser.error("No matching dialogues")
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "router_version": "edges-v4",
              "model": config["model"] if url else os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
              "cache": "decision cache disabled; provider prompt cache may apply",
              "scope": "text-to-route only; regression set, not a held-out estimate; no STT/TTS", "dialogues": []}
    totals, passes, tags = Counter(), Counter(), {}
    latencies = []
    expected_intents = found_intents = failed_dialogues = 0
    for case in dialogues:
        sid = None
        result = {"id": case["id"], "tags": case["tags"], "turns": []}
        for turn in case["turns"]:
            # Intentionally pass only text/session/cache policy, never expected routes.
            payload = {"text": turn["text"], "session_id": sid, "use_cache": False}
            try:
                if url:
                    request = urllib.request.Request(url + "/api/route", data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(request, timeout=90) as response:
                        response = json.load(response)
                else:
                    response = app.route_request(payload)
                # In-process responses reference mutable session slots; freeze each turn now.
                response = json.loads(json.dumps(response, ensure_ascii=False))
                sid = response["session_id"]
                checks = check_turn(turn, response)
                latencies.append(response["trace"]["latency_ms"]["router"])
                actual = {s["scenario_id"] for s in response["trace"]["scenarios"]}
                if len(turn["routes"]) > 1:
                    expected_intents += len(set(turn["routes"]))
                    found_intents += len(set(turn["routes"]) & actual)
                result["turns"].append({"input": turn, "checks": checks, "response": response})
            except Exception as exc:
                # Avoid saving upstream response bodies, which may contain sensitive data.
                checks = {"primary": False, "full_set": False, "order": False, "transport_or_output": False}
                result["turns"].append({"input": turn, "checks": checks, "error_type": type(exc).__name__})
            for name, passed in checks.items():
                totals[name] += 1
                passes[name] += int(passed)
            if "transport_or_output" in checks:
                break  # Later turns depend on the missing turn; do not fabricate context.
        result["passed"] = len(result["turns"]) == len(case["turns"]) and all(
            all(turn["checks"].values()) for turn in result["turns"])
        failed_dialogues += int(not result["passed"])
        for tag in case["tags"]:
            item = tags.setdefault(tag, {"passed": 0, "total": 0})
            item["total"] += 1
            item["passed"] += int(result["passed"])
        report["dialogues"].append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {case['id']}", flush=True)
    report["summary"] = {
        "dialogues_passed": len(dialogues) - failed_dialogues, "dialogues_total": len(dialogues),
        "turns_expected": sum(len(case["turns"]) for case in dialogues),
        "turns_attempted": sum(len(case["turns"]) for case in report["dialogues"]),
        "checks": {name: {"passed": passes[name], "total": total} for name, total in totals.items()},
        "by_tag": tags, "multi_intent_recall": found_intents / expected_intents if expected_intents else None,
        "router_ms": {"p50": percentile(latencies, .5), "p95": percentile(latencies, .95),
                      "under_500ms": sum(ms <= 500 for ms in latencies), "measured": len(latencies)},
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return int(bool(failed_dialogues))


if __name__ == "__main__":
    sys.exit(main())
