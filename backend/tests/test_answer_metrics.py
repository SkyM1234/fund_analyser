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
        run = _run("规模为1000万元，截至2025-05-01，近一年收益率为10％。")

        result = metrics.key_fact_coverage(run, example)

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["comment"], "3/3 facts hit")

    def test_does_not_match_partial_numbers_or_missing_units(self) -> None:
        for answer in ("115.30%", "15.31%", "15.3", "-15.3%", "15.3万元"):
            with self.subTest(answer=answer):
                self.assertEqual(metrics.key_fact_coverage(_run(answer), _example(["15.3%"]))["score"], 0)
        self.assertEqual(metrics.key_fact_coverage(_run("15.30%"), _example(["15.3%"]))["score"], 1)

    def test_date_does_not_match_another_day(self) -> None:
        self.assertEqual(metrics.key_fact_coverage(_run("2025-05-10"), _example(["2025-05-1"]))["score"], 0)

    def test_citation_uses_answer_not_tool_arguments(self) -> None:
        run = SimpleNamespace(outputs={
            "answer": "No evidence.",
            "tool_calls": [{"name": "rag_search", "args": {"filter_fund_code": "159103"}}],
        })
        example = SimpleNamespace(outputs={"expected_fund_codes": ["159103"]})
        self.assertEqual(metrics.citation_accuracy(run, example)["score"], 0)
        run.outputs["answer"] = "[159103] and [159104]"
        self.assertAlmostEqual(metrics.citation_accuracy(run, example)["score"], 2 / 3)

    def test_risk_disclaimer_is_not_a_refusal(self) -> None:
        run = _run("该基金规模为10亿元。投资有风险，请咨询专业的投资顾问。")
        example = SimpleNamespace(outputs={"should_refuse": False})
        self.assertEqual(metrics.refusal_correctness(run, example)["score"], 1)

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
