"""Generate predictions.json for the supplied dev set using the real LLM router."""

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import app


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first; demo mode is not valid for evaluation.")
    utterances = json.loads((DATA / "dev_utterances.json").read_text(encoding="utf-8"))["utterances"]
    requested = {value.strip() for value in os.getenv("EVAL_IDS", "").split(",") if value.strip()}
    if requested:
        utterances = [item for item in utterances if item["id"] in requested]
        if not utterances:
            raise SystemExit("EVAL_IDS did not match any utterances")
    output = DATA / "predictions.json"
    predictions = json.loads(output.read_text(encoding="utf-8")) if requested and output.exists() else {}
    completed = 0

    def predict(utterance):
        empty_state = {
            "history": [], "active_scenarios": [], "scenario_stack": [], "pending_queue": [],
            "slots": {}, "action_context": {}, "uncertain_turns": 0, "pending_confirmation": None, "turn": 0,
        }
        decision, latency = app.call_llm(utterance["text"], empty_state)
        decision = app.normalize_decision(decision, utterance["text"])
        return utterance["id"], [item["scenario_id"] for item in decision["scenarios"]], latency

    workers = max(1, min(8, int(os.getenv("EVAL_WORKERS", "4"))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(predict, utterance) for utterance in utterances]
        for future in as_completed(futures):
            utterance_id, routes, latency = future.result()
            predictions[utterance_id] = routes
            completed += 1
            print(f"[{completed:03}/{len(utterances)}] {utterance_id} {routes} {latency}ms", flush=True)
    if not requested:
        predictions = {utterance["id"]: predictions[utterance["id"]] for utterance in utterances}
    output.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
