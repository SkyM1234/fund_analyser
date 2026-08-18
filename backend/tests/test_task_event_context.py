import unittest

from app.tasks.chat_tasks import _resolve_task_context


class TaskEventContextTests(unittest.TestCase):
    def test_uses_own_run_context(self) -> None:
        contexts = {
            "agent-run": {"task_id": "t1", "agent_name": "rag_agent"},
        }

        context = _resolve_task_context({"run_id": "agent-run"}, contexts)

        self.assertEqual(context["task_id"], "t1")
        self.assertEqual(context["agent_name"], "rag_agent")

    def test_uses_nearest_parent_context(self) -> None:
        contexts = {
            "agent-run": {"task_id": "t1", "agent_name": "rag_agent"},
            "nested-run": {"task_id": "t2", "agent_name": "market_agent"},
        }
        event = {
            "run_id": "tool-run",
            "parent_ids": ["agent-run", "nested-run"],
        }

        context = _resolve_task_context(event, contexts)

        self.assertEqual(context["task_id"], "t2")
        self.assertEqual(context["agent_name"], "market_agent")

    def test_returns_empty_context_without_known_parent(self) -> None:
        self.assertEqual(
            _resolve_task_context({"run_id": "tool-run", "parent_ids": ["unknown"]}, {}),
            {},
        )

