import json
import tempfile
import unittest
from pathlib import Path

from intent_classifier.routers.classifier.common import (
    LABELS, check_split, normalize_history, read_jsonl, split_samples,
)


class DataTests(unittest.TestCase):
    def test_history_formats_have_the_same_meaning(self):
        self.assertEqual(
            normalize_history([["human", "基金"], ["ai", "回答"]]),
            normalize_history([
                {"role": "user", "content": "基金"},
                {"role": "assistant", "content": "回答"},
            ]),
        )

    def test_stratified_split_preserves_all_labels(self):
        samples = [
            {"query": f"{label}-{index}", "history": [], "intent": label}
            for label in LABELS for index in range(10)
        ]
        train, validation = split_samples(samples, 0.2, 42)
        check_split(train, validation)
        self.assertEqual(len(train), 48)
        self.assertEqual(len(validation), 12)

    def test_conversation_groups_do_not_cross_splits(self):
        samples = [
            {"query": f"{label}-{group}", "history": [], "intent": label,
             "group_id": f"conversation-{group}"}
            for group in range(10) for label in LABELS
        ]
        train, validation = split_samples(samples, 0.2, 42)
        check_split(train, validation)
        self.assertFalse({row["group_id"] for row in train}
                         & {row["group_id"] for row in validation})

    def test_cross_file_duplicate_is_rejected(self):
        rows = [{"query": label, "history": [], "intent": label} for label in LABELS]
        with self.assertRaisesRegex(ValueError, "相同 query/history"):
            check_split(rows, rows)

    def test_cross_file_group_leak_is_rejected(self):
        train = [{"query": "train-" + label, "history": [], "intent": label,
                  "group_id": "same-session"} for label in LABELS]
        validation = [{"query": "val-" + label, "history": [], "intent": label,
                       "group_id": "same-session"} for label in LABELS]
        with self.assertRaisesRegex(ValueError, "相同 group_id"):
            check_split(train, validation)

    def test_missing_class_is_rejected(self):
        samples = [{"query": label, "history": [], "intent": label} for label in LABELS]
        with self.assertRaisesRegex(ValueError, "缺少类别"):
            check_split(samples, samples[:-1])

    def test_duplicate_training_input_is_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8",
                                         delete=False) as target:
            path = Path(target.name)
            row = {"query": "159103的费率", "intent": "fund_query"}
            target.write(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        try:
            with self.assertRaisesRegex(ValueError, "重复或标签冲突"):
                read_jsonl(path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
