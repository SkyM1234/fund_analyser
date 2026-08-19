import unittest

from app.agent.multi_agent_controller import route_after_intent
from app.services.router import RouteResult


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

    def test_specific_fund_intents_enter_scope_confirmation(self) -> None:
        for intent in ("single_fund_query", "cross_fund_query"):
            with self.subTest(intent=intent):
                self.assertEqual(
                    route_after_intent(self._state(intent)),
                    "fund_scope",
                )

    def test_fund_screening_bypasses_scope_confirmation(self) -> None:
        self.assertEqual(
            route_after_intent(self._state("fund_screening")),
            "supervisor",
        )
