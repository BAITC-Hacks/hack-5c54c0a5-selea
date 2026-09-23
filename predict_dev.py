"""Generate predictions.json for the supplied dev set using the real LLM router."""

import json
import os
from pathlib import Path

import app


ROOT = Path(__file__).resolve().parent


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first; demo mode is not valid for evaluation.")
    utterances = json.loads((ROOT / "dev_utterances.json").read_text(encoding="utf-8"))["utterances"]
    predictions = {}
    empty_state = {
        "history": [],
        "active_scenarios": [],
        "scenario_stack": [],
        "slots": {},
        "uncertain_turns": 0,
        "pending_confirmation": None,
        "turn": 0,
    }
    for index, utterance in enumerate(utterances, 1):
        decision, latency = app.call_llm(utterance["text"], empty_state)
        decision = app.normalize_decision(decision)
        predictions[utterance["id"]] = [item["scenario_id"] for item in decision["scenarios"]]
        print(f"[{index:03}/{len(utterances)}] {utterance['id']} {predictions[utterance['id']]} {latency}ms")
    output = ROOT / "predictions.json"
    output.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
