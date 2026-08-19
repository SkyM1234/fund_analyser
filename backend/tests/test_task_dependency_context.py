import unittest

from app.agent.multi_agent_controller import dispatch_tasks
from app.agent.task_context import format_dependency_results


class TaskDependencyContextTests(unittest.TestCase):
    def test_dispatch_injects_only_direct_dependency_results(self) -> None:
        state = {
            "dispatch_task_ids": ["t3"],
            "plan": [
                {
                    "task_id": "t1",
                    "status": "completed",
                    "depends_on": [],
                    "result": "first result",
                },
                {
                    "task_id": "t2",
                    "status": "completed",
                    "depends_on": [],
                    "result": "second result",
                },
                {
                    "task_id": "t3",
                    "status": "running",
                    "description": "use t1",
                    "query": "summarize",
                    "assigned_agent": "analysis_agent",
                    "fund_codes": [],
                    "depends_on": ["t1"],
                },
            ],
            "sub_results": {
                "t1": "first result",
                "t2": "unrelated result",
            },
            "messages": [],
            "route_result": None,
        }

        sends = dispatch_tasks(state)

        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].node, "analysis_agent")
        task_input = sends[0].arg["task_input"]
        self.assertEqual(task_input["dependency_results"], {"t1": "first result"})
        self.assertNotIn("t2", task_input["dependency_results"])

    def test_formats_dependency_results_as_agent_context(self) -> None:
        context = format_dependency_results(
            {"dependency_results": {"t1": "upstream finding"}}
        )

        self.assertIn("已完成上游依赖任务的结果", context)
        self.assertIn("[t1]", context)
        self.assertIn("upstream finding", context)


if __name__ == "__main__":
    unittest.main()
