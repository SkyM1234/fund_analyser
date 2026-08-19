import unittest
from types import SimpleNamespace

from eval.evaluators import answer_metrics as metrics


def _example(key_facts: list[str]) -> SimpleNamespace:
    return SimpleNamespace(outputs={"key_facts": key_facts})


def _run(answer: str) -> SimpleNamespace:
    return SimpleNamespace(outputs={"answer": answer})


class KeyFactCoverageTests(unittest.TestCase):
    def test_matches_common_equivalent_number_and_date_formats(self) -> None:
        example = _example(
            [
                "规模为1,000万元",
                "截至2025年5月1日",
                "近一年收益率为10%",
            ]
        )
        run = _run("规模为1000万元，截至2025-05-01，近一年收益率为10。")

        result = metrics.key_fact_coverage(run, example)

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["comment"], "3/3 facts hit")

    def test_matches_reverse_equivalent_formats(self) -> None:
        example = _example(
            [
                "规模为1000万元",
                "截至2025-05-01",
                "近一年收益率为10",
            ]
        )
        run = _run("规模为1,000万元，截至2025年5月1日，近一年收益率为10%。")

        result = metrics.key_fact_coverage(run, example)

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["comment"], "3/3 facts hit")

if __name__ == "__main__":
    unittest.main()
