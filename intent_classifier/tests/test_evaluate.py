import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from intent_classifier.routers.classifier.common import LABELS
from intent_classifier.routers.classifier.evaluate import audit_test_data, compute_report


@contextmanager
def reference_file(rows):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8",
                                     delete=False) as target:
        path = Path(target.name)
        for row in rows:
            target.write(json.dumps(row) + "\n")
    try:
        yield path
    finally:
        path.unlink()


class EvaluationTests(unittest.TestCase):
    def samples(self):
        return [{"id": f"test-{index}", "query": f"question-{index}", "history": [],
                 "intent": label, "group_id": f"test-group-{index}"}
                for index, label in enumerate(LABELS)]

    def test_metrics_and_confusion_matrix_direction(self):
        rows = [{"expected_intent": label, "predicted_intent": label,
                 "correct": True, "history": [], "tags": []} for label in LABELS]
        rows[0].update(predicted_intent=LABELS[1], correct=False)
        report = compute_report(rows)
        self.assertAlmostEqual(report["accuracy"], 5 / 6)
        self.assertAlmostEqual(report["macro_f1"], (4 + 2 / 3) / 6)
        self.assertEqual(report["error_count"], 1)
        self.assertEqual(report["confusion_matrix"]["values"][0][1], 1)
        self.assertEqual(report["confusion_matrix"]["values"][1][0], 0)
        self.assertEqual(report["classification_report"][LABELS[0]]["recall"], 0)

    def test_context_and_tag_accuracy(self):
        rows = [
            {"expected_intent": "fund_query", "predicted_intent": "fund_query",
             "correct": True, "history": [["user", "fund"]], "tags": ["reference"]},
            {"expected_intent": "general_finance", "predicted_intent": "fund_query",
             "correct": False, "history": [["user", "concept"]], "tags": ["reference"]},
            {"expected_intent": "chitchat", "predicted_intent": "chitchat",
             "correct": True, "history": [], "tags": ["greeting"]},
        ]
        report = compute_report(rows)
        self.assertEqual(report["by_context"]["with_history"]["accuracy"], 0.5)
        self.assertEqual(report["by_context"]["without_history"]["accuracy"], 1)
        self.assertEqual(report["by_tag"]["reference"]["accuracy"], 0.5)

    def test_identical_inputs_are_rejected_even_with_different_ids(self):
        rows = self.samples()
        reference = {**rows[0], "id": "training-row", "group_id": "different-group"}
        with reference_file([reference]) as path:
            with self.assertRaisesRegex(ValueError, "相同输入 1 条"):
                audit_test_data(rows, [path])

    def test_shared_groups_are_rejected_even_with_different_inputs(self):
        rows = self.samples()
        reference = {**rows[0], "query": "another question"}
        with reference_file([reference]) as path:
            with self.assertRaisesRegex(ValueError, "相同 group_id 1 个"):
                audit_test_data(rows, [path])

    def test_non_overlapping_reference_passes(self):
        reference = {"query": "training question", "intent": "fund_query"}
        with reference_file([reference]) as path:
            report = audit_test_data(self.samples(), [path, path])
        self.assertEqual(report["sample_count"], 6)
        self.assertEqual(len(report["references"]), 1)
        self.assertEqual(report["references"][0]["exact_input_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
