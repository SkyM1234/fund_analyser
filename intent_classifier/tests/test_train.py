import unittest
from types import SimpleNamespace

import numpy as np

from intent_classifier.routers.classifier.common import LABELS
from intent_classifier.routers.classifier.train import compute_metrics


class TrainingMetricsTests(unittest.TestCase):
    def test_uniform_predictions_have_expected_probability_error(self):
        count = len(LABELS)
        metrics = compute_metrics(SimpleNamespace(
            predictions=np.zeros((count, count)), label_ids=np.arange(count),
        ))
        self.assertAlmostEqual(metrics["accuracy"], 1 / count)
        self.assertAlmostEqual(metrics["mean_confidence"], 1 / count)
        self.assertAlmostEqual(metrics["brier_score"], 1 - 1 / count)

    def test_more_confident_correct_predictions_improve_probability_error(self):
        logits = np.eye(len(LABELS))
        labels = np.arange(len(LABELS))
        before = compute_metrics(SimpleNamespace(predictions=logits, label_ids=labels))
        after = compute_metrics(SimpleNamespace(predictions=logits * 5, label_ids=labels))
        self.assertEqual(before["accuracy"], after["accuracy"])
        self.assertEqual(before["macro_f1"], after["macro_f1"])
        self.assertGreater(after["mean_confidence"], before["mean_confidence"])
        self.assertLess(after["brier_score"], before["brier_score"])

    def test_more_confident_wrong_predictions_worsen_probability_error(self):
        logits = np.eye(len(LABELS))
        labels = np.roll(np.arange(len(LABELS)), 1)
        before = compute_metrics(SimpleNamespace(predictions=logits, label_ids=labels))
        after = compute_metrics(SimpleNamespace(predictions=logits * 5, label_ids=labels))
        self.assertEqual(before["accuracy"], 0)
        self.assertEqual(after["accuracy"], 0)
        self.assertGreater(after["mean_confidence"], before["mean_confidence"])
        self.assertGreater(after["brier_score"], before["brier_score"])

    def test_large_logit_offset_does_not_overflow_or_change_metrics(self):
        logits = np.eye(len(LABELS)) * 5
        labels = np.arange(len(LABELS))
        before = compute_metrics(SimpleNamespace(predictions=logits, label_ids=labels))
        after = compute_metrics(SimpleNamespace(predictions=logits + 10000, label_ids=labels))
        for key, value in before.items():
            self.assertTrue(np.isfinite(after[key]))
            self.assertAlmostEqual(after[key], value)


if __name__ == "__main__":
    unittest.main()
