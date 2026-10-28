import copy
import json
import tempfile
import unittest
from pathlib import Path

from intent_classifier.common import LABELS, ROOT
from intent_classifier.routers.rules.router import RulesRouter
from intent_classifier.validation.reporting import render_report, summarize, write_reports


class ReportingTests(unittest.TestCase):
    def prediction(self):
        vector = {label: 0.40 for label in LABELS}
        vector["fund_query"] = 0.83
        probabilities = {label: 0.06 for label in LABELS}
        probabilities["fund_query"] = 0.70
        return {
            "id": "example", "query": "查询159103的费率", "intent": "fund_query",
            "expected_intent": "fund_query", "stage": "classifier",
            "needs_clarification": False, "latency_ms": 10,
            "trace": [
                {"stage": "rules", "status": "rejected", "accepted": False,
                 "scores": {label: 0 for label in LABELS}, "score_type": "rule_match"},
                {"stage": "vector", "status": "rejected", "accepted": False,
                 "candidate_intent": "fund_query", "scores": vector, "score_type": "cosine_similarity"},
                {"stage": "classifier", "status": "accepted", "accepted": True,
                 "candidate_intent": "fund_query", "scores": probabilities,
                 "score_type": "softmax_probability"},
            ],
        }

    def metrics(self, prediction, enabled):
        return {**summarize([prediction]), "enabled_routes": enabled}

    def test_all_six_scores_keep_their_different_meanings(self):
        prediction = self.prediction()
        before = copy.deepcopy(prediction)
        report = render_report(self.metrics(prediction, ["rules", "vector", "classifier"]), [prediction])
        self.assertIn("0.8300", report)
        self.assertIn("70.00%", report)
        self.assertIn("6.00%", report)
        self.assertIn("余弦相似度", report)
        self.assertIn("规则命中标记", report)
        self.assertIn("| 闲聊 | 范围外 | 敏感请求 | 基金查询 | 基金筛选 | 金融知识 |", report)
        self.assertEqual(prediction, before)

    def test_disabled_and_short_circuited_routes_have_no_fabricated_scores(self):
        prediction = self.prediction()
        prediction.update(intent="chitchat", stage="rules")
        prediction["trace"] = [{"stage": "rules", "status": "accepted", "accepted": True,
                                "candidate_intent": "chitchat", "scores": {label: int(label == "chitchat") for label in LABELS}}]
        report = render_report(self.metrics(prediction, ["rules", "vector"]), [prediction])
        self.assertIn("| 向量 | 未执行（前序路由已接受） | 余弦相似度 | — | — | — | — | — | — |", report)
        self.assertIn("| 分类模型 | 未启用 | Softmax 概率 | — | — | — | — | — | — |", report)

    def test_rule_scores_are_match_flags_including_no_match(self):
        router = RulesRouter(ROOT / "intent_classifier/routers/rules/data/rules.json")
        hit = router.route("你好")
        miss = router.route("你好，查询基金的费率")
        self.assertEqual(hit["score_type"], "rule_match")
        self.assertEqual(set(hit["scores"]), set(LABELS))
        self.assertEqual(sum(hit["scores"].values()), 1)
        self.assertEqual(miss["scores"], dict.fromkeys(LABELS, 0))

    def test_report_files_preserve_all_raw_scores(self):
        prediction = self.prediction()
        metrics = self.metrics(prediction, ["rules", "vector", "classifier"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            write_reports(path, metrics, [prediction])
            saved = json.loads((path / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved, prediction)
            self.assertEqual(json.loads((path / "metrics.json").read_text(encoding="utf-8")), metrics)
            self.assertIn("70.00%", (path / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
