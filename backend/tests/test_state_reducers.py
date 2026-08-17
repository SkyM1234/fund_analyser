import unittest

from app.agent.multi_agent_state import (
    get_blocked_tasks,
    get_ready_tasks,
    is_plan_complete,
)
from app.agent.state_reducers import PlanPatches, TaskPatch, merge_plan


class MergePlanPatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = [
            {
                "task_id": "t1",
                "status": "pending",
                "description": "first task",
                "retry_count": 0,
            },
            {
                "task_id": "t2",
                "status": "pending",
                "description": "second task",
                "retry_count": 0,
            },
        ]

    def test_concurrent_task_patches_preserve_both_completions(self) -> None:
        running = PlanPatches([
            TaskPatch("t1", {"status": "running", "started_at": 10.0}),
            TaskPatch("t2", {"status": "running", "started_at": 10.0}),
        ])
        started = merge_plan(self.plan, running)

        t1_done = TaskPatch("t1", {"status": "completed", "result": "result 1"})
        t2_done = TaskPatch("t2", {"status": "completed", "result": "result 2"})

        completed = merge_plan(merge_plan(started, t1_done), t2_done)
        self.assertEqual(
            [(task["task_id"], task["status"], task.get("result")) for task in completed],
            [("t1", "completed", "result 1"), ("t2", "completed", "result 2")],
        )

        reverse_completed = merge_plan(merge_plan(started, t2_done), t1_done)
        self.assertEqual(
            [(task["task_id"], task["status"], task.get("result")) for task in reverse_completed],
            [("t1", "completed", "result 1"), ("t2", "completed", "result 2")],
        )

    def test_task_patch_preserves_unrelated_fields_and_tasks(self) -> None:
        patched = merge_plan(
            self.plan,
            TaskPatch("t1", {"status": "failed", "error": "tool unavailable"}),
        )

        self.assertEqual(patched[0]["description"], "first task")
        self.assertEqual(patched[0]["retry_count"], 0)
        self.assertEqual(patched[0]["status"], "failed")
        self.assertEqual(patched[1], self.plan[1])

    def test_task_patch_for_unknown_task_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown task_id: missing"):
            merge_plan(self.plan, TaskPatch("missing", {"status": "completed"}))

    def test_failed_dependency_does_not_make_downstream_task_ready(self) -> None:
        state = {
            "plan": [
                {"task_id": "t1", "status": "failed", "depends_on": []},
                {"task_id": "t2", "status": "pending", "depends_on": ["t1"]},
            ],
            "completed_tasks": [],
            "failed_tasks": ["t1"],
            "blocked_tasks": [],
        }

        self.assertEqual(get_ready_tasks(state), [])
        self.assertEqual(
            [task["task_id"] for task in get_blocked_tasks(state)],
            ["t2"],
        )

    def test_failed_dependency_blocks_entire_downstream_chain(self) -> None:
        state = {
            "plan": [
                {"task_id": "t1", "status": "failed", "depends_on": []},
                {"task_id": "t2", "status": "pending", "depends_on": ["t1"]},
                {"task_id": "t3", "status": "pending", "depends_on": ["t2"]},
                {"task_id": "t4", "status": "pending", "depends_on": []},
            ],
            "completed_tasks": [],
            "failed_tasks": ["t1"],
            "blocked_tasks": [],
        }

        self.assertEqual(
            [task["task_id"] for task in get_blocked_tasks(state)],
            ["t2", "t3"],
        )
        self.assertEqual(
            [task["task_id"] for task in get_ready_tasks(state)],
            ["t4"],
        )

    def test_blocked_tasks_are_terminal_for_plan_completion(self) -> None:
        state = {
            "plan": [
                {"task_id": "t1", "status": "failed"},
                {"task_id": "t2", "status": "blocked"},
            ],
        }

        self.assertTrue(is_plan_complete(state))


if __name__ == "__main__":
    unittest.main()
