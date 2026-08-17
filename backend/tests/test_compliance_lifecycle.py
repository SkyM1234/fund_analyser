import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage

from app.agent.compliance_agent import compliance_agent_node
from app.agent.multi_agent_controller import commit_answer_node
from app.agent.supervisor import _new_plan_update


class ComplianceLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def test_new_plan_resets_compliance_retry_budget(self) -> None:
        update = _new_plan_update([])

        self.assertEqual(update["compliance_retry_count"], 0)
        self.assertIsNone(update["draft_answer"])
        self.assertIsNone(update["final_answer"])

    async def test_compliance_checks_draft_answer(self) -> None:
        state = {
            "draft_answer": "这是待审查的答案草稿。",
            "final_answer": "上一轮已经归档的答案。",
            "compliance_retry_count": 0,
        }

        response = type(
            "Response",
            (),
            {
                "content": '{"passed": false, "reason": "需要改写", "risk_level": "high"}',
                "usage_metadata": {},
            },
        )()
        with (
            patch("app.agent.compliance_agent.ChatOpenAI"),
            patch(
                "app.agent.compliance_agent.llm_ainvoke",
                new=AsyncMock(return_value=response),
            ) as invoke,
        ):
            update = await compliance_agent_node(state)

        self.assertIn("compliance_passed", update)
        self.assertFalse(update["compliance_passed"])
        self.assertEqual(update["compliance_retry_count"], 1)
        self.assertIn("待审查的答案草稿", invoke.await_args.args[1][1]["content"])
        self.assertNotIn("上一轮已经归档的答案", invoke.await_args.args[1][1]["content"])

    def test_commit_archives_only_the_compliant_draft(self) -> None:
        state = {
            "messages": [HumanMessage(content="查询基金信息")],
            "plan": [{"task_id": "t1", "status": "completed"}],
            "sub_results": {"t1": "结果"},
            "plan_history": [],
            "draft_answer": "通过合规检查的答案",
            "final_answer": "旧答案",
        }

        update = commit_answer_node(state)

        self.assertEqual(update["final_answer"], "通过合规检查的答案")
        self.assertEqual(len(update["plan_history"]), 1)
        self.assertEqual(update["plan_history"][0]["final_answer"], "通过合规检查的答案")
        self.assertEqual(update["plan_history"][0]["user_query"], "查询基金信息")
