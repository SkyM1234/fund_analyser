import unittest
from unittest.mock import AsyncMock, patch

from app.agent.multi_agent_controller import route_after_intent
from app.services.router import RouteResult, route_query


class QueryRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_standalone_greetings_short_circuit(self) -> None:
        with patch("app.services.router._llm_classify", new_callable=AsyncMock) as classify:
            self.assertEqual((await route_query("你好！")).intent, "chitchat")
            classify.assert_not_awaited()

    async def test_substantive_questions_reach_classifier(self) -> None:
        queries = ["你好，请查询159103的持仓", "不要推荐，只比较基金费率", "这只基金的外汇风险如何", "推荐一只基金", "今天天气如何"]
        for query in queries:
            with self.subTest(query=query), patch(
                "app.services.router._llm_classify", new=AsyncMock(return_value=RouteResult(intent="fund_query")),
            ) as classify:
                await route_query(query)
                classify.assert_awaited_once_with(query, None)


class RoutingBranchTests(unittest.TestCase):
    def _state(self, intent: str) -> dict:
        return {"route_result": RouteResult(intent=intent)}

    def test_non_fund_intents_short_circuit(self) -> None:
        self.assertEqual(route_after_intent(self._state("chitchat")), "direct_answer")
        self.assertEqual(
            route_after_intent(self._state("general_finance")),
            "direct_answer",
        )
        self.assertEqual(
            route_after_intent(self._state("out_of_scope")),
            "out_of_scope_refusal",
        )
        self.assertEqual(
            route_after_intent(self._state("sensitive")),
            "sensitive_refusal",
        )

    def test_fund_query_enters_scope_confirmation(self) -> None:
        self.assertEqual(
            route_after_intent(self._state("fund_query")),
            "fund_scope",
        )

    def test_fund_screening_bypasses_scope_confirmation(self) -> None:
        self.assertEqual(
            route_after_intent(self._state("fund_screening")),
            "supervisor",
        )
