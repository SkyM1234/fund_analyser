import unittest

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


if __name__ == "__main__":
    unittest.main()
