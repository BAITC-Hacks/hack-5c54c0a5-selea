"""Voice Router hackathon MVP.

Run with: python app.py
The server intentionally uses only the Python standard library so the project
can be started with one command on a clean machine.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"


def read_json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


SCENARIOS = read_json("scenarios.json")["scenarios"]
SLOTS = read_json("slots.json")
SCENARIO_BY_ID = {item["scenario_id"]: item for item in SCENARIOS}


def slot_catalog() -> dict:
    raw = SLOTS.get("slots", SLOTS)
    if isinstance(raw, list):
        return {item["name"]: item for item in raw}
    return raw


SLOT_BY_NAME = slot_catalog()


def compact_catalog() -> str:
    rows = []
    for item in SCENARIOS:
        boundaries = "; ".join(
            f"NOT if {rule['condition']} -> {rule['use_instead']}"
            for rule in item.get("not_this_if", [])
        )
        examples = item.get("examples", {})
        sample = (examples.get("ru", []) + examples.get("kk", []))[:3]
        rows.append(
            f"{item['scenario_id']} | {item['name']} | priority={item.get('priority', 'normal')}\n"
            f"meaning: {item['description']}\n"
            f"boundaries: {boundaries or '-'}\n"
            f"examples: {' / '.join(sample)}"
        )
    rows.extend(
        [
            "SYS_OUT_OF_SCOPE | not related to Saqta Insurance services",
            "SYS_UNCLEAR | too ambiguous; ask one short either/or question",
            "SYS_GOODBYE | conversation is ending",
        ]
    )
    return "\n\n".join(rows)


CATALOG = compact_catalog()
ALLOWED_IDS = set(SCENARIO_BY_ID) | {
    "SYS_OUT_OF_SCOPE",
    "SYS_UNCLEAR",
    "SYS_GOODBYE",
}

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "string", "enum": sorted(ALLOWED_IDS)},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["scenario_id", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "string", "enum": sorted(ALLOWED_IDS)},
                    "confidence": {"type": "number"},
                    "why_rejected": {"type": "string"},
                },
                "required": ["scenario_id", "confidence", "why_rejected"],
                "additionalProperties": False,
            },
        },
        "language": {"type": "string", "enum": ["ru", "kk", "mixed"]},
        "slots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["name", "value"],
                "additionalProperties": False,
            },
        },
        "is_continuation": {"type": "boolean"},
        "needs_clarification": {"type": "boolean"},
        "clarification_question": {"type": "string"},
    },
    "required": [
        "scenarios",
        "alternatives",
        "language",
        "slots",
        "is_continuation",
        "needs_clarification",
        "clarification_question",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = f"""You are the decision layer of Saqta Insurance's voice agent.
Route the latest customer utterance using dialogue context and the catalog below.
This is reasoning-based routing, not keyword matching.

Rules:
- Detect every intent in a multi-intent utterance. Return urgent first, then mention order.
- Distinguish neighboring scenarios using their boundaries.
- If the utterance fills slots for the active scenario, mark is_continuation=true.
- Use SYS_UNCLEAR only when one concise clarification is necessary.
- Use SYS_OUT_OF_SCOPE for unsupported products such as loans or life insurance.
- Give a short supervisor-facing reason, never private chain-of-thought.
- Confidence is calibrated from 0 to 1. Output language is ru, kk, or mixed.
- Resolve relative dates against 2026-10-01.

SCENARIO CATALOG
{CATALOG}
"""


class SessionStore:
    def __init__(self):
        self._items: dict[str, dict] = {}
        self._lock = Lock()

    def get(self, session_id: str | None) -> tuple[str, dict]:
        sid = session_id or str(uuid.uuid4())
        with self._lock:
            state = self._items.setdefault(
                sid,
                {
                    "history": [],
                    "active_scenarios": [],
                    "scenario_stack": [],
                    "slots": {},
                    "uncertain_turns": 0,
                    "pending_confirmation": None,
                    "turn": 0,
                },
            )
        return sid, state

    def reset(self, session_id: str):
        with self._lock:
            self._items.pop(session_id, None)


SESSIONS = SessionStore()


def extract_response_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                return content["text"]
    raise RuntimeError("LLM response did not contain output text")


def call_llm(text: str, state: dict) -> tuple[dict, int]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        started = time.perf_counter()
        return demo_route(text), round((time.perf_counter() - started) * 1000)

    context = {
        "active_scenarios": state["active_scenarios"],
        "known_slots": state["slots"],
        "recent_turns": state["history"][-8:],
        "latest_utterance": text,
    }
    body = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "instructions": SYSTEM_PROMPT,
        "input": json.dumps(context, ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "voice_router_decision",
                "strict": True,
                "schema": ROUTE_SCHEMA,
            }
        },
    }
    request = urllib.request.Request(
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        + "/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API returned {exc.code}: {detail[:500]}") from exc
    latency = round((time.perf_counter() - started) * 1000)
    return json.loads(extract_response_text(payload)), latency


def tokenize(value: str) -> set[str]:
    return set(re.findall(r"[\wӘәҒғҚқҢңӨөҰұҮүҺһІі]+", value.casefold()))


def demo_route(text: str) -> dict:
    """Visible no-key demo only. It is deliberately not the evaluated solution."""
    tokens = tokenize(text)
    goodbye = {"пока", "до свидания", "сау болыңыз", "рахмет"}
    if tokens & goodbye:
        primary = "SYS_GOODBYE"
        confidence = 0.96
        reason = "Клиент завершает разговор"
    else:
        ranked = []
        for scenario in SCENARIOS:
            examples = scenario.get("examples", {}).get("ru", []) + scenario.get("examples", {}).get("kk", [])
            haystack = tokenize(" ".join(examples) + " " + scenario.get("description", ""))
            score = len(tokens & haystack) / max(1, len(tokens))
            ranked.append((score, scenario["scenario_id"]))
        ranked.sort(reverse=True)
        score, primary = ranked[0]
        confidence = min(0.72, 0.34 + score)
        reason = "Демонстрационное сопоставление примеров; задайте OPENAI_API_KEY для LLM-routing"
        if score < 0.08:
            primary, confidence = "SYS_UNCLEAR", 0.4
    return {
        "scenarios": [{"scenario_id": primary, "confidence": confidence, "reason": reason}],
        "alternatives": [],
        "language": detect_language(text),
        "slots": [],
        "is_continuation": False,
        "needs_clarification": primary == "SYS_UNCLEAR",
        "clarification_question": "Уточните, пожалуйста, что именно вы хотите сделать?" if primary == "SYS_UNCLEAR" else "",
    }


def detect_language(text: str) -> str:
    kk_chars = len(re.findall(r"[ӘәҒғҚқҢңӨөҰұҮүҺһІі]", text))
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    if kk_chars and cyr > kk_chars * 2:
        return "mixed"
    return "kk" if kk_chars else "ru"


def normalize_decision(decision: dict) -> dict:
    scenarios = [item for item in decision.get("scenarios", []) if item.get("scenario_id") in ALLOWED_IDS]
    if not scenarios:
        scenarios = [{"scenario_id": "SYS_UNCLEAR", "confidence": 0.0, "reason": "Маршрут не определён"}]
    for item in scenarios:
        item["confidence"] = max(0.0, min(1.0, float(item.get("confidence", 0))))
    decision["scenarios"] = scenarios
    decision["alternatives"] = [
        item for item in decision.get("alternatives", []) if item.get("scenario_id") in ALLOWED_IDS
    ][:3]
    return decision


def slot_prompt(name: str, language: str) -> str:
    item = SLOT_BY_NAME.get(name, {}) if isinstance(SLOT_BY_NAME, dict) else {}
    prompts = item.get("prompt", {}) if isinstance(item, dict) else {}
    return prompts.get("kk" if language == "kk" else "ru", f"Уточните {name}, пожалуйста.")


def build_reply(decision: dict, state: dict, confirmed: bool = False) -> tuple[str, list[dict]]:
    primary = decision["scenarios"][0]
    scenario_id = primary["scenario_id"]
    language = decision.get("language", "ru")

    if scenario_id == "SYS_GOODBYE":
        return ("Сау болыңыз!" if language == "kk" else "До свидания!"), []
    if scenario_id == "SYS_OUT_OF_SCOPE":
        return (
            "Кешіріңіз, бұл сұрақ Saqta Insurance қызметтеріне жатпайды."
            if language == "kk"
            else "Извините, этот вопрос не относится к услугам Saqta Insurance."
        ), []
    if decision.get("needs_clarification") or primary["confidence"] < 0.75:
        return decision.get("clarification_question") or "Уточните, пожалуйста, ваш запрос.", []

    scenario = SCENARIO_BY_ID.get(scenario_id)
    if not scenario:
        return "Уточните, пожалуйста, ваш запрос.", []

    missing = [name for name in scenario.get("slots", {}).get("required", []) if name not in state["slots"]]
    if missing:
        return slot_prompt(missing[0], language), []

    actions = []
    mode = "execute" if confirmed or not scenario.get("requires_confirmation") else "preview"
    for name in scenario.get("actions", []):
        actions.append({"name": name, "mode": mode})
    if confirmed:
        state["pending_confirmation"] = None
        reply = (
            "Расталды. Әрекет тестілік жүйеде орындалды."
            if language == "kk"
            else "Подтверждение получено. Действие выполнено в тестовой системе."
        )
    elif scenario.get("requires_confirmation"):
        state["pending_confirmation"] = scenario_id
        reply = "Деректер дұрыс па? Растайсыз ба?" if language == "kk" else "Проверьте данные. Подтверждаете выполнение?"
    else:
        reply = scenario.get("responses", {}).get("kk" if language == "kk" else "ru", {}).get("opening", "Запрос принят.")
    return reply, actions


def route_request(payload: dict) -> dict:
    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("Поле text не должно быть пустым")
    sid, state = SESSIONS.get(payload.get("session_id"))
    state["turn"] += 1

    affirmative = bool(re.search(r"\b(да|верно|подтверждаю|иә|дұрыс|растаймын)\b", text.casefold()))
    confirmed = bool(state["pending_confirmation"] and affirmative)
    if confirmed:
        scenario_id = state["pending_confirmation"]
        decision = {
            "scenarios": [{"scenario_id": scenario_id, "confidence": 1.0, "reason": "Клиент явно подтвердил ранее показанное действие"}],
            "alternatives": [],
            "language": detect_language(text),
            "slots": [],
            "is_continuation": True,
            "needs_clarification": False,
            "clarification_question": "",
        }
        router_ms = 0
    else:
        decision, router_ms = call_llm(text, state)
        decision = normalize_decision(decision)
    for slot in decision.get("slots", []):
        if slot.get("name"):
            state["slots"][slot["name"]] = slot.get("value", "")

    top = decision["scenarios"][0]
    if top["confidence"] < 0.45:
        state["uncertain_turns"] += 1
    else:
        state["uncertain_turns"] = 0
    handoff = (
        state["uncertain_turns"] >= 2
        or bool(payload.get("request_operator"))
        or top["scenario_id"] == "SC37"
    )

    selected = [item["scenario_id"] for item in decision["scenarios"] if item["scenario_id"].startswith("SC")]
    if selected and selected != state["active_scenarios"]:
        if state["active_scenarios"]:
            state["scenario_stack"].append(state["active_scenarios"])
        state["active_scenarios"] = selected

    response_started = time.perf_counter()
    reply, actions = build_reply(decision, state, confirmed=confirmed)
    response_ms = round((time.perf_counter() - response_started) * 1000)
    state["history"].extend([{"role": "client", "text": text}, {"role": "bot", "text": reply}])
    state["history"] = state["history"][-20:]

    return {
        "session_id": sid,
        "reply": reply,
        "handoff": handoff,
        "handoff_summary": f"Запрос: {text}. Маршруты: {', '.join(selected) or top['scenario_id']}." if handoff else "",
        "trace": {
            "turn": state["turn"],
            "transcript": text,
            "language": decision.get("language", "ru"),
            "scenarios": decision["scenarios"],
            "alternatives": decision.get("alternatives", []),
            "slots": state["slots"],
            "actions": actions,
            "dialog_state": {
                "active_scenarios": state["active_scenarios"],
                "stack": state["scenario_stack"],
                "uncertain_turns": state["uncertain_turns"],
                "pending_confirmation": state["pending_confirmation"],
            },
            "latency_ms": {"router": router_ms, "response": response_ms, "total": router_ms + response_ms},
            "mode": "llm" if os.getenv("OPENAI_API_KEY") else "demo",
        },
    }


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean == "/":
            clean = "/index.html"
        safe_parts = [part for part in Path(clean).parts if part not in {"/", "\\", "..", "."}]
        return str(STATIC.joinpath(*safe_parts))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, status: int, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/config":
            self.send_json(
                200,
                {
                    "llm_enabled": bool(os.getenv("OPENAI_API_KEY")),
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "scenario_count": len(SCENARIOS),
                },
            )
            return
        super().do_GET()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/api/route":
                self.send_json(200, route_request(payload))
            elif self.path == "/api/reset":
                SESSIONS.reset(str(payload.get("session_id", "")))
                self.send_json(200, {"ok": True})
            else:
                self.send_json(404, {"error": "Not found"})
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main():
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Voice Router: http://127.0.0.1:{port}")
    print("Mode:", "LLM" if os.getenv("OPENAI_API_KEY") else "DEMO (set OPENAI_API_KEY for evaluation)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
