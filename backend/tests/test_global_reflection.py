import unittest
from unittest.mock import AsyncMock, patch

from app.agent.reflection_agent import global_reflection_node


def make_task(task_id: str, status: str, *, reflected: bool = False) -> dict:
    return {
        "task_id": task_id,
        "task_type": "rag_search",
        "description": task_id,
        "assigned_agent": "rag_agent",
        "fund_codes": ["159103"],
        "query": task_id,
        "depends_on": [],
        "status": status,
        "result": None,
        "error": None,
        "retry_count": 0,
        "reflected": reflected,
    }


class GlobalReflectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_defers_check_until_dependent_plan_is_complete(self) -> None:
        state = {
            "plan": [
                make_task("t1", "completed"),
                make_task("t2", "pending"),
            ],
            "sub_results": {"t1": "upstream result"},
        }

        update = await global_reflection_node(state)

        self.assertEqual(update, {})
        self.assertFalse(state["plan"][0]["reflected"])

    async def test_final_check_compares_results_from_different_batches(self) -> None:
        state = {
            "plan": [
                make_task("t1", "completed"),
                {
                    **make_task("t2", "completed"),
                    "depends_on": ["t1"],
                },
            ],
            "sub_results": {
                "t1": "基金净值为 1.00 [159103]",
                "t2": "基金净值为 1.20 [159103]",
            },
            "messages": [],
            "reflection_count": 0,
            "clarification_round": 0,
            "conflict_annotations": [],
        }
        response = type(
            "Response",
            (),
            {
                "content": '{"conflicts": [], "summary": "无冲突"}',
                "response_metadata": {},
            },
        )()

        with (
            patch("app.agent.reflection_agent.ChatOpenAI"),
            patch(
                "app.agent.reflection_agent.llm_ainvoke",
                new=AsyncMock(return_value=response),
            ) as invoke,
        ):
            update = await global_reflection_node(state)

        self.assertEqual(invoke.await_count, 1)
        prompt = invoke.await_args.args[1][1]["content"]
        self.assertIn("[t1]", prompt)
        self.assertIn("[t2]", prompt)
        self.assertTrue(all(task["reflected"] for task in update["plan"]))

    async def test_clarification_completion_does_not_trigger_second_comparison(self) -> None:
        state = {
            "plan": [
                make_task("t1", "completed", reflected=True),
                make_task("t2", "completed", reflected=True),
                make_task("clarify_c1", "completed"),
            ],
            "sub_results": {"clarify_c1": "[裁决:adopt_a] 采用 t1 的结果"},
            "messages": [],
            "reflection_count": 1,
            "clarification_round": 1,
            "conflict_annotations": [
                {
                    "conflict_id": "c1",
                    "risk": "high",
                    "task_ids": ["t1", "t2"],
                    "field": "净值",
                    "description": "冲突",
                    "resolved": False,
                    "clarification_task_id": "clarify_c1",
                }
            ],
        }

        with patch(
            "app.agent.reflection_agent.llm_ainvoke",
            new=AsyncMock(),
        ) as invoke:
            update = await global_reflection_node(state)

        self.assertEqual(invoke.await_count, 0)
        self.assertTrue(update["plan"][2]["reflected"])
        self.assertTrue(update["conflict_annotations"][0]["resolved"])
