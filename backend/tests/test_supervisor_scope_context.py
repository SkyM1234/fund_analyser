import unittest

from app.agent.supervisor import _scope_planning_context
from app.services.router import RouteResult


class SupervisorScopeContextTests(unittest.TestCase):
    def test_fund_screening_uses_global_retrieval_context(self) -> None:
        context = _scope_planning_context(
            RouteResult(intent="fund_screening"),
            None,
        )

        self.assertIn("未执行 fund_scope_agent", context)
        self.assertIn("fund_codes 设为 []", context)
        self.assertIn("全局检索", context)

    def test_confirmed_scope_uses_authoritative_scope_context(self) -> None:
        context = _scope_planning_context(
            RouteResult(intent="single_fund_query"),
            {"funds": [{"fund_code": "159103"}]},
        )

        self.assertIn("已确认基金范围", context)
        self.assertIn("唯一可用于计划 fund_codes", context)
