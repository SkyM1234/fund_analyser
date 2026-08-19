import unittest

from app.agent.fund_scope_agent import _normalize_scope, _parse_scope_response
from app.tools.llm_json import extract_json_block


class FundScopeNormalizationTests(unittest.TestCase):
    def test_deduplicates_confirmed_sector_funds(self) -> None:
        scope = _normalize_scope(
            {
                "query": "ignored",
                "funds": [
                    {"fund_code": "159103", "fund_name": "fund one"},
                    {"fund_code": "159103", "fund_name": "fund one"},
                    {"fund_code": "159299", "fund_name": "fund two"},
                ],
                "total_count": 99,
                "coverage_status": "confirmed",
                "missing_or_uncertain": [],
            },
            "sector funds",
            {"159103", "159299"},
        )

        self.assertEqual(scope["total_count"], 2)
        self.assertEqual(
            [fund["fund_code"] for fund in scope["funds"]],
            ["159103", "159299"],
        )

    def test_rejects_unconfirmed_code(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_scope(
                {
                    "query": "x",
                    "funds": [{"fund_code": "159103"}],
                    "coverage_status": "confirmed",
                },
                "x",
                set(),
            )

    def test_marks_scope_incomplete_when_names_remain_uncertain(self) -> None:
        scope = _normalize_scope(
            {
                "query": "x",
                "funds": [{"fund_code": "159103"}],
                "coverage_status": "confirmed",
                "missing_or_uncertain": ["unknown fund"],
            },
            "x",
            {"159103"},
        )
        self.assertEqual(scope["coverage_status"], "incomplete")

    def test_rejects_empty_final_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "未返回 JSON"):
            _parse_scope_response("", "x", set())

    def test_extracts_fenced_json_after_explanatory_text(self) -> None:
        content = """已确认 1 只基金。

```json
{"funds":[{"fund_name":"基金一","fund_code":"159103"}],"total_count":1,"coverage_status":"confirmed","missing_or_uncertain":[]}
```"""

        self.assertEqual(
            extract_json_block(content),
            '{"funds":[{"fund_name":"基金一","fund_code":"159103"}],"total_count":1,"coverage_status":"confirmed","missing_or_uncertain":[]}',
        )
