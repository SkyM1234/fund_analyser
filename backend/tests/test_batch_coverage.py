import unittest

from app.agent.retrieval_agent import _missing_batch_rag_coverage


class BatchRagCoverageTests(unittest.TestCase):
    def test_reports_unsearched_target_codes(self) -> None:
        missing = _missing_batch_rag_coverage(
            {"fund_codes": ["159103", "159299", "513130"]},
            [
                {
                    "name": "rag_search",
                    "args": {"filter_fund_code": "159103"},
                },
                {
                    "name": "rag_search",
                    "args": {"filter_fund_code": "513130"},
                },
            ],
        )

        self.assertEqual(missing, ["159299"])

    def test_single_fund_task_does_not_need_batch_coverage(self) -> None:
        self.assertEqual(
            _missing_batch_rag_coverage({"fund_codes": ["159103"]}, []),
            [],
        )
