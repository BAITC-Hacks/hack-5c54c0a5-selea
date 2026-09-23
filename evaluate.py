"""Reference router evaluator for Voice Router.

Usage:
    python evaluate.py [data/predictions.json] [data/dev_utterances.json]

predictions.json:
    {"U001": ["SC01"], "U097": ["SC27", "SC04"], ...}
    Scenario IDs in the order your router returns them. Missing IDs count as errors.

Metrics:
    primary_accuracy  first predicted scenario == first expected scenario
    full_match        set of predicted scenarios == set of expected scenarios
    intent_recall     share of expected scenarios found (multi-intent only)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def evaluate_predictions(preds, utts):
    groups = defaultdict(lambda: {"n": 0, "primary": 0, "full": 0})
    recall_hit = recall_total = 0
    errors = []
    for u in utts:
        exp = u["expected"]
        got = preds.get(u["id"], [])
        if isinstance(got, str):
            got = [got]
        primary = bool(got) and got[0] == exp[0]
        full = set(got) == set(exp)
        if u["type"] == "multi_intent":
            recall_hit += len(set(exp) & set(got))
            recall_total += len(exp)
        for key in ("all", f"lang={u['lang']}", f"type={u['type']}"):
            g = groups[key]
            g["n"] += 1
            g["primary"] += primary
            g["full"] += full
        if not full:
            errors.append({"id": u["id"], "text": u["text"], "expected": exp, "predicted": got})
    overall = groups["all"]
    return {
        "total": overall["n"],
        "primary_accuracy": overall["primary"] / overall["n"],
        "full_match": overall["full"] / overall["n"],
        "multi_intent_recall": recall_hit / recall_total if recall_total else None,
        "groups": {key: {"n": value["n"], "primary_accuracy": value["primary"] / value["n"],
                         "full_match": value["full"] / value["n"]} for key, value in groups.items()},
        "error_count": len(errors),
        "errors": errors,
    }


def main():
    predictions_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "predictions.json"
    preds = json.load(open(predictions_path, encoding="utf-8"))
    dev_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / "dev_utterances.json"
    utts = json.load(open(dev_path, encoding="utf-8"))["utterances"]
    known = {u["id"] for u in utts}

    extra = sorted(set(preds) - known)
    if extra:
        print(f"Warning: {len(extra)} prediction IDs not in the dev set: {extra[:5]}")

    result = evaluate_predictions(preds, utts)

    print(f"{'group':<22}{'n':>5}{'primary_acc':>14}{'full_match':>12}")
    groups = result["groups"]
    order = ["all"] + sorted(k for k in groups if k.startswith("lang=")) + sorted(k for k in groups if k.startswith("type="))
    for k in order:
        g = groups[k]
        print(f"{k:<22}{g['n']:>5}{g['primary_accuracy']:>14.3f}{g['full_match']:>12.3f}")
    if result["multi_intent_recall"] is not None:
        print(f"\nintent_recall (multi-intent): {result['multi_intent_recall']:.3f}")
    if result["errors"]:
        print(f"\nErrors ({len(result['errors'])}):")
        for item in result["errors"]:
            print(f"  {item['id']}  expected={item['expected']}  got={item['predicted']}  | {item['text']}")
    output = Path(sys.argv[3]) if len(sys.argv) > 3 else DATA / "dev_metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to {output}")


if __name__ == "__main__":
    main()
