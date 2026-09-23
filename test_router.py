"""Offline policy/state regressions. These tests do not measure LLM accuracy."""
import copy
import json
import os
import unittest
from unittest.mock import patch

import app
import evaluate_edges


def decision(*ids, confidence=0.94, language="ru", **overrides):
    return {
        "scenarios": [{"scenario_id": sid, "confidence": confidence, "reason": "test"} for sid in ids],
        "alternatives": [], "language": language, "slots": [], "closed_scenarios": [],
        "is_continuation": False, "needs_clarification": False, "clarification_question": "",
        **overrides,
    }


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test-placeholder", "ROUTER_EXPERIMENTAL_FAST_PATH": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.sessions = patch.object(app, "SESSIONS", app.SessionStore())
        self.sessions.start()
        self.addCleanup(self.sessions.stop)
        app._router_cache.clear()
        self.sid, self.state = app.SESSIONS.get(None)

    def turn(self, text, result):
        with patch.object(app, "call_llm", return_value=(copy.deepcopy(result), 12)) as model:
            response = app.route_request({"text": text, "session_id": self.sid, "use_cache": False})
        return response, model

    def test_queue_survives_continuation(self):
        self.turn("Офис и способы оплаты", decision("SC33", "SC31"))
        response, _ = self.turn("Алматы", decision("SC33", is_continuation=True))
        self.assertEqual(response["trace"]["dialog_state"]["pending_queue"], ["SC31"])
        self.assertEqual(self.state["active_scenarios"], ["SC33"])
        self.assertEqual(self.state["scenario_stack"], [])

    def test_topic_switch_and_resume(self):
        self.turn("Где офис?", decision("SC33"))
        self.turn("Сейчас авария", decision("SC11"))
        self.assertEqual(self.state["scenario_stack"], [["SC33"]])
        self.turn("Вернёмся к офису", decision("SC33"))
        self.assertEqual(self.state["active_scenarios"], ["SC33"])
        self.assertEqual(self.state["scenario_stack"], [["SC11"]])

    def test_queue_deduplicates_and_removes_promoted_primary(self):
        self.turn("Два запроса", decision("SC33", "SC31"))
        self.turn("Те же", decision("SC33", "SC31"))
        self.assertEqual(self.state["pending_queue"], ["SC31"])
        self.turn("Теперь про оплату", decision("SC31"))
        self.assertEqual(self.state["pending_queue"], [])

    def test_explicit_cancel_removes_only_named_request(self):
        self.turn("Офис, оплата и копия", decision("SC33", "SC31", "SC26"))
        self.turn("Оплата уже не нужна", decision("SC33", closed_scenarios=["SC31"]))
        self.assertEqual(self.state["pending_queue"], ["SC26"])

    def test_cancelled_primary_does_not_execute(self):
        self.state["active_scenarios"] = ["SC31"]
        response, _ = self.turn("Больше не нужно", decision("SC31", closed_scenarios=["SC31"]))
        self.assertEqual(response["trace"]["actions"], [])
        self.assertEqual(self.state["active_scenarios"], [])

    def test_close_alternatives_blocks_actions_and_state_switch(self):
        self.state["active_scenarios"] = ["SC33"]
        response, _ = self.turn("Оплата?", decision("SC31", confidence=0.89,
            alternatives=[{"scenario_id": "SC30", "confidence": 0.84, "why_rejected": "unclear"}]))
        self.assertTrue(response["trace"]["needs_clarification"])
        self.assertEqual(response["trace"]["actions"], [])
        self.assertEqual(self.state["active_scenarios"], ["SC33"])
        self.assertEqual(response["trace"]["confidence_band"], "medium")
        self.assertAlmostEqual(response["trace"]["ambiguity_gap"], 0.05)

    def test_concurrent_requests_are_not_alternatives(self):
        result = app.normalize_decision(decision("SC31", "SC33", alternatives=[
            {"scenario_id": "SC33", "confidence": 0.94, "why_rejected": "duplicate"}]))
        self.assertFalse(result["needs_clarification"])
        self.assertEqual(result["alternatives"], [])

    def test_unclear_high_score_is_still_uncertain(self):
        for _ in range(2):
            response, _ = self.turn("Ну это...", decision("SYS_UNCLEAR", confidence=0.99))
        self.assertTrue(response["handoff"])
        self.assertNotEqual(response["trace"]["confidence_band"], "high")

    def test_negated_or_qualified_confirmation_never_bypasses_model(self):
        for text in ["не подтверждаю", "нет, не подтверждаю", "да, но поменяйте номер", "иә, бірақ тоқтаңыз", "да и ещё офис"]:
            with self.subTest(text=text):
                self.state["pending_confirmation"] = "SC29"
                response, model = self.turn(text, decision("SYS_UNCLEAR"))
                model.assert_called_once()
                self.assertEqual(response["trace"]["actions"], [])
                self.assertIsNone(self.state["pending_confirmation"])

    def test_explicit_confirmation_executes_previewed_scenario(self):
        self.state["pending_confirmation"] = "SC29"
        self.state["active_scenarios"] = ["SC29"]
        self.state["slots"] = {"contact_field": "phone", "new_value": "+77010000000"}
        response, model = self.turn("Да, подтверждаю!", decision("SYS_UNCLEAR"))
        model.assert_not_called()
        self.assertTrue(response["trace"]["actions"])
        self.assertTrue(all(a["mode"] == "execute" for a in response["trace"]["actions"]))
        self.assertIsNone(self.state["pending_confirmation"])

    def test_topic_shift_invalidates_old_confirmation(self):
        self.state["pending_confirmation"] = "SC29"
        self.turn("Пока расскажите про оплату", decision("SC31"))
        self.assertIsNone(self.state["pending_confirmation"])
        _, model = self.turn("Да", decision("SYS_UNCLEAR"))
        model.assert_called_once()

    def test_language_switch_preserves_queue(self):
        self.turn("Офис и оплата", decision("SC33", "SC31"))
        response, _ = self.turn("Қала — Астана", decision("SC33", language="kk", is_continuation=True,
            slots=[{"name": "city", "value": "Астана"}]))
        self.assertEqual(self.state["pending_queue"], ["SC31"])
        self.assertEqual(response["trace"]["language"], "kk")
        self.assertEqual(self.state["history"][-2]["language"], "kk")

    def test_history_keeps_ten_pairs_with_roles_and_language(self):
        for i in range(12):
            self.turn(str(i), decision("SC33"))
        self.assertEqual(len(self.state["history"]), 20)
        self.assertEqual(self.state["history"][0]["text"], "2")
        self.assertEqual({m["role"] for m in self.state["history"]}, {"user", "assistant"})
        self.assertTrue(all("language" in m for m in self.state["history"]))

    def test_invalid_slots_are_ignored(self):
        self.turn("Алматы", decision("SC33", slots=[
            {"name": "made_up", "value": "secret"}, {"name": "city", "value": "Алматы"}]))
        self.assertEqual(self.state["slots"], {"city": "Алматы"})

    def test_invalid_scores_and_duplicate_ids(self):
        result = app.normalize_decision(decision("SC31", "SC31", confidence=float("nan")))
        self.assertEqual(len(result["scenarios"]), 1)
        self.assertEqual(result["scenarios"][0]["confidence"], 0)
        self.assertTrue(result["needs_clarification"])

    def test_verified_quotes_restore_mention_order(self):
        result = decision("SC31", "SC33")
        result["scenarios"][0]["request_text"] = "способы оплаты"
        result["scenarios"][1]["request_text"] = "где офис"
        normalized = app.normalize_decision(result, "Подскажите, где офис, и способы оплаты")
        self.assertEqual([s["scenario_id"] for s in normalized["scenarios"]], ["SC33", "SC31"])

    def test_urgency_precedes_quote_order(self):
        result = decision("SC33", "SC11")
        result["scenarios"][0]["request_text"] = "офис"
        result["scenarios"][1]["request_text"] = "авария"
        normalized = app.normalize_decision(result, "Где офис? Сейчас авария!")
        self.assertEqual(normalized["scenarios"][0]["scenario_id"], "SC11")

    def test_secondary_uncertainty_is_visible_without_blocking_clear_primary(self):
        result = decision("SC33", "SC31")
        result["scenarios"][1]["confidence"] = .55
        response, _ = self.turn("Офис, а ещё оплата", result)
        self.assertEqual(response["trace"]["uncertain_scenarios"], ["SC31"])
        self.assertFalse(response["trace"]["needs_clarification"])
        self.assertEqual(response["trace"]["confidence_band"], "medium")

    def test_margin_boundary_is_not_float_roundoff(self):
        normalized = app.normalize_decision(decision("SC31", confidence=.9,
            alternatives=[{"scenario_id": "SC30", "confidence": .8, "why_rejected": "test"}]))
        self.assertNotIn("close_alternatives", normalized["uncertainty_reasons"])

    def test_old_queued_intent_is_not_replayed_as_current(self):
        self.turn("Офис и оплата", decision("SC33", "SC31"))
        result = decision("SC33", "SC31", is_continuation=True)
        result["scenarios"][1]["request_text"] = "оплата"
        response, _ = self.turn("Город Семей", result)
        self.assertEqual([s["scenario_id"] for s in response["trace"]["scenarios"]], ["SC33"])
        self.assertEqual(response["trace"]["history_only_intents"], ["SC31"])
        self.assertEqual(self.state["pending_queue"], ["SC31"])

    def test_repeated_intent_with_current_evidence_is_preserved(self):
        self.turn("Офис и оплата", decision("SC33", "SC31"))
        result = decision("SC33", "SC31")
        result["scenarios"][1]["request_text"] = "способы оплаты"
        response, _ = self.turn("Город Семей. И способы оплаты тоже расскажите", result)
        self.assertEqual(len(response["trace"]["scenarios"]), 2)

    def test_default_fast_path_never_decides_negation_or_implicit_multi_intent(self):
        for text in ["Не нужен оператор, объясните франшизу", "Где офис? Полис тоже перешлите.", "Офис қайда?"]:
            self.assertIsNone(app.try_fast_path(text, self.state, app.rank_scenarios(text)))

    def test_cache_rejects_uncertainty_multi_intent_and_slots(self):
        cases = [decision("SC31", needs_clarification=True), decision("SC31", "SC33"),
                 decision("SC33", slots=[{"name": "city", "value": "Алматы"}]),
                 decision("SC31", confidence=0.91, alternatives=[
                     {"scenario_id": "SC30", "confidence": 0.9, "why_rejected": "unclear"}])]
        for result in cases:
            app.remember_decision("text", result)
            self.assertIsNone(app.cached_decision("text"))

    def test_cache_copies_and_separates_models(self):
        app.remember_decision("Оплата", decision("SC31"))
        cached = app.cached_decision("Оплата")
        cached["scenarios"][0]["scenario_id"] = "SC33"
        self.assertEqual(app.cached_decision("Оплата")["scenarios"][0]["scenario_id"], "SC31")
        with patch.dict(os.environ, {"OPENAI_MODEL": "different-model"}):
            self.assertIsNone(app.cached_decision("Оплата"))

    def test_all_scenarios_have_ru_and_kk_candidate_examples(self):
        for scenario in app.SCENARIOS:
            text = app.candidate_catalog([scenario["scenario_id"]])
            self.assertIn(scenario["examples"]["ru"][0], text)
            self.assertIn(scenario["examples"]["kk"][0], text)

    def test_all_boundaries_are_in_stable_prompt(self):
        for scenario in app.SCENARIOS:
            for boundary in scenario.get("not_this_if", []):
                self.assertIn(boundary["condition"], app.SYSTEM_PROMPT)

    def test_llm_receives_history_stack_and_handles_incomplete(self):
        self.state["history"] = [{"role": "user", "language": "kk", "text": "test"}] * 20
        self.state["scenario_stack"] = [["SC33"]]
        with patch.object(app.urllib.request, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
                {"status": "incomplete", "output": []}).encode()
            result, _ = app.call_llm("вернёмся", self.state)
        request = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(request["temperature"], 0)
        messages = request["input"]
        context = json.loads(messages[0]["content"].split("\n", 1)[1])
        self.assertEqual(len(messages), 22)
        self.assertEqual(messages[-1], {"role": "user", "content": "вернёмся"})
        self.assertTrue(all(m == {"role": "user", "content": "test"} for m in messages[1:-1]))
        self.assertEqual(context["history_languages"], [{"message_index": i, "language": "kk"} for i in range(20)])
        self.assertEqual(context["scenario_stack"], [["SC33"]])
        normalized = app.normalize_decision(result)
        self.assertTrue(normalized["needs_clarification"])
        self.assertEqual(normalized["_router_meta"]["response_error"], "invalid_or_incomplete_output")

    def test_edge_fixture_ids_and_dialogue_lengths(self):
        cases = app.read_json("edge_dialogues.json")["dialogues"]
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({case["id"] for case in cases}), 20)
        self.assertEqual(max(len(case["turns"]) for case in cases), 10)
        for case in cases:
            for turn in case["turns"]:
                self.assertTrue(set(turn["routes"]) <= app.ALLOWED_IDS)
                self.assertTrue(turn["text"].strip())

    def test_evaluator_catches_wrong_route_and_queue(self):
        response, _ = self.turn("Оплата", decision("SC31"))
        checks = evaluate_edges.check_turn({"routes": ["SC33", "SC31"], "queue": ["SC31"]}, response)
        self.assertFalse(checks["primary"])
        self.assertFalse(checks["full_set"])
        self.assertFalse(checks["queue"])

    def test_latency_percentile_uses_nearest_rank(self):
        self.assertEqual(evaluate_edges.percentile(list(range(1, 101)), .95), 95)
        self.assertIsNone(evaluate_edges.percentile([], .5))


if __name__ == "__main__":
    unittest.main()
