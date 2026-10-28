import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from intent_classifier.common import LABELS, ROOT, read_queries, select_candidate
from intent_classifier.cascade import build_pipeline, DEFAULT_THRESHOLDS
from intent_classifier.pipeline import CascadeRouter, STAGES
from intent_classifier.routers.rules.router import RulesRouter
from intent_classifier.routers.vector.router import aggregate_similarities
from intent_classifier.validation.reporting import summarize

RULES_FILE = ROOT / "intent_classifier/routers/rules/data/rules.json"


def route_query(query, vector_predict=None, classifier_predict=None, *, enabled=STAGES, **thresholds):
    """Feed deterministic model scores through the real rules and cascade."""
    settings = {**DEFAULT_THRESHOLDS, **thresholds}
    routers = [RulesRouter(RULES_FILE)] if "rules" in enabled else []
    for name, predict in [("vector", vector_predict), ("classifier", classifier_predict)]:
        if name in enabled:
            routers.append(SimpleNamespace(name=name, route=lambda query, name=name, predict=predict:
                select_candidate(predict(query), settings[f"{name}_threshold"], settings[f"{name}_margin"])))
    return CascadeRouter(routers).route(query)


def scores(label, first=0.95, second=0.01):
    return {name: first if name == label else second for name in LABELS}


class CascadeTests(unittest.TestCase):
    def test_greeting_short_circuits_but_substantive_question_does_not(self):
        vector = Mock(return_value=scores("fund_query"))
        classifier = Mock(side_effect=AssertionError("不应调用分类器"))
        self.assertEqual(route_query("你好！", vector, classifier)["stage"], "rules")
        vector.assert_not_called()
        result = route_query("你好，查询159103的费率", vector, classifier)
        self.assertEqual(result["stage"], "vector")
        self.assertEqual(result["intent"], "fund_query")
        classifier.assert_not_called()

    def test_all_eight_enable_combinations_and_fixed_order(self):
        for size in range(4):
            for enabled in combinations(STAGES, size):
                with self.subTest(enabled=enabled):
                    vector = Mock(return_value=scores("fund_query"))
                    classifier = Mock(return_value=scores("sensitive"))
                    result = route_query("查询159103的费率", vector, classifier, enabled=enabled)
                    expected = "vector" if "vector" in enabled else "classifier" if "classifier" in enabled else "clarification"
                    self.assertEqual(result["stage"], expected)
                    self.assertEqual(vector.call_count, int("vector" in enabled))
                    self.assertEqual(classifier.call_count, int("classifier" in enabled and "vector" not in enabled))
                    self.assertEqual(result["needs_clarification"], expected == "clarification")

    def test_vector_rejection_reaches_classifier(self):
        result = route_query("一个问题", lambda _: scores("fund_query", 0.70),
                             lambda _: scores("fund_screening"))
        self.assertEqual(result["stage"], "classifier")
        self.assertEqual(result["intent"], "fund_screening")
        self.assertEqual([row["status"] for row in result["trace"]], ["rejected", "rejected", "accepted"])

    def test_low_margin_rejects_even_with_high_top_score(self):
        values = scores("fund_query", 0.95)
        values["fund_screening"] = 0.94
        result = route_query("一个问题", lambda _: values, lambda _: values)
        self.assertEqual(result["stage"], "clarification")
        self.assertIsNone(result["intent"])

    def test_low_classifier_score_requests_clarification(self):
        result = route_query("一个问题", classifier_predict=lambda _: scores("fund_query", 0.5),
                             enabled=["classifier"])
        self.assertTrue(result["needs_clarification"])
        self.assertTrue(result["clarification_question"])

    def test_similarity_margin_is_between_different_labels(self):
        refs = [{"intent": "fund_query"}, {"intent": "fund_query"}, {"intent": "fund_screening"}]
        values = aggregate_similarities([0.95, 0.94, 0.70], refs)
        result = select_candidate(values, 0.85, 0.06)
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["margin"], 0.25)

    def test_broken_predictor_is_not_disguised_as_uncertainty(self):
        vector = Mock(side_effect=RuntimeError("模型损坏"))
        with self.assertRaisesRegex(RuntimeError, "模型损坏"):
            route_query("问题", vector, enabled=["vector"])

    def test_invalid_scores_and_thresholds_fail(self):
        with self.assertRaises(ValueError):
            select_candidate(scores("fund_query"), float("nan"), 0.1)
        with self.assertRaises(ValueError):
            route_query("问题", lambda _: {"fund_query": 1.0}, enabled=["vector"])
        values = scores("fund_query", float("nan"))
        with self.assertRaises(ValueError):
            select_candidate(values, 0.8, 0.1)

    def test_disabled_models_need_no_model_paths(self):
        pipeline, device = build_pipeline(SimpleNamespace(rules_file=RULES_FILE), ["rules"])
        self.assertEqual(device, "none")
        self.assertEqual(list(pipeline.routers), ["rules"])
        self.assertEqual(pipeline.route("你好")["stage"], "rules")

    def test_pipeline_uses_fixed_order_and_rejects_duplicate_names(self):
        vector = SimpleNamespace(name="vector", route=Mock(return_value={"accepted": True, "candidate_intent": "fund_query"}))
        pipeline = CascadeRouter([vector, RulesRouter(RULES_FILE)])
        self.assertEqual(pipeline.route("你好")["stage"], "rules")
        vector.route.assert_not_called()
        with self.assertRaises(ValueError):
            CascadeRouter([vector, vector])

    def test_rule_data_can_be_changed_without_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rules.json"
            path.write_text(json.dumps([{"id": "weather", "intent": "out_of_scope", "pattern": "天气"}]), encoding="utf-8")
            router = RulesRouter(path)
            self.assertEqual(router.route("天气")["candidate_intent"], "out_of_scope")
            self.assertFalse(router.route("基金的天气风险")["accepted"])

    def test_metrics_separate_rejected_candidates_from_accepted_accuracy(self):
        result = route_query("问题", classifier_predict=lambda _: scores("fund_query", 0.5),
                             enabled=["classifier"])
        result["expected_intent"] = "fund_query"
        report = summarize([result])
        self.assertEqual(report["coverage"], 0)
        self.assertIsNone(report["accepted_accuracy"])
        self.assertEqual(report["by_stage"]["classifier"]["candidate_accuracy"], 1)

    def test_dataset_rejects_history_and_conflicting_current_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "queries.jsonl"
            path.write_text(json.dumps({"query": "问题", "intent": "fund_query", "history": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "history"):
                read_queries(path)
            path.write_text('\n'.join(json.dumps({"query": "问题", "intent": label})
                                      for label in ["fund_query", "general_finance"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "冲突标签"):
                read_queries(path)


if __name__ == "__main__":
    unittest.main()
