import unittest

from app.api.session import _build_execution_summaries


class SessionExecutionSummaryTests(unittest.TestCase):
    def test_restores_completed_scope_confirmation_before_plan_agents(self) -> None:
        plan, agents = _build_execution_summaries({
            "fund_scope": {"funds": [{"fund_code": "000001"}]},
            "plan": [{
                "task_id": "rag-1",
                "task_type": "rag_search",
                "description": "检索年报",
                "assigned_agent": "rag_agent",
                "fund_codes": ["000001"],
                "status": "completed",
            }],
        })

        self.assertEqual(plan[0]["task_id"], "rag-1")
        self.assertEqual(
            agents,
            [
                {
                    "agent_name": "fund_scope_agent",
                    "task_id": "fund_scope",
                    "description": "确认当前问题涉及的基金范围",
                    "status": "completed",
                    "sequence": -1,
                },
                {
                    "agent_name": "rag_agent",
                    "task_id": "rag-1",
                    "description": "检索年报",
                    "status": "completed",
                },
            ],
        )

    def test_restores_failed_scope_confirmation_without_plan(self) -> None:
        plan, agents = _build_execution_summaries({
            "fund_scope": None,
            "fund_scope_error": "未确认到基金范围",
        })

        self.assertEqual(plan, [])
        self.assertEqual(agents[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
