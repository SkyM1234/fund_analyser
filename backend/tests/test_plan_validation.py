import unittest

from pydantic import ValidationError

from app.agent.multi_agent_controller import should_continue_planning
from app.agent.plan_validation import PlanValidationError, validate_supervisor_plan


def _task(task_id: str, **overrides):
    task = {
        "task_id": task_id,
        "task_type": "rag_search",
        "description": f"查询 {task_id}",
        "assigned_agent": "rag_agent",
        "fund_codes": ["159103"],
        "query": "查询基金信息",
        "depends_on": [],
        "status": "pending",
    }
    task.update(overrides)
    return task


class SupervisorPlanValidationTests(unittest.TestCase):
    def test_accepts_valid_dag(self) -> None:
        plan = {
            "plan": [
                _task("t1"),
                _task("t2", depends_on=["t1"]),
            ],
            "reasoning": "先检索基础信息，再基于结果补充分析。",
        }

        validated = validate_supervisor_plan(
            plan,
            explicit_fund_codes={"159103"},
        )

        self.assertEqual([task["task_id"] for task in validated], ["t1", "t2"])

    def test_rejects_duplicate_task_ids(self) -> None:
        plan = {
            "plan": [_task("t1"), _task("t1")],
            "reasoning": "测试重复 ID。",
        }

        with self.assertRaisesRegex(PlanValidationError, "task_id 必须唯一"):
            validate_supervisor_plan(plan, explicit_fund_codes={"159103"})

    def test_rejects_unknown_dependency(self) -> None:
        plan = {
            "plan": [_task("t1", depends_on=["missing"])],
            "reasoning": "测试不存在依赖。",
        }

        with self.assertRaisesRegex(PlanValidationError, "依赖不存在"):
            validate_supervisor_plan(plan, explicit_fund_codes={"159103"})

    def test_rejects_cycle(self) -> None:
        plan = {
            "plan": [
                _task("t1", depends_on=["t2"]),
                _task("t2", depends_on=["t1"]),
            ],
            "reasoning": "测试循环依赖。",
        }

        with self.assertRaisesRegex(
            PlanValidationError,
            r"循环依赖: t1 -> t2 -> t1",
        ):
            validate_supervisor_plan(plan, explicit_fund_codes={"159103"})

    def test_rejects_agent_task_type_mismatch(self) -> None:
        plan = {
            "plan": [
                _task(
                    "t1",
                    task_type="market_data",
                    assigned_agent="rag_agent",
                )
            ],
            "reasoning": "测试 Agent 映射。",
        }

        with self.assertRaises(ValidationError):
            validate_supervisor_plan(plan, explicit_fund_codes={"159103"})

    def test_rejects_invalid_or_unprovided_fund_code(self) -> None:
        invalid_format = {
            "plan": [_task("t1", fund_codes=["15910A"])],
            "reasoning": "测试格式。",
        }
        unprovided = {
            "plan": [_task("t1", fund_codes=["510300"])],
            "reasoning": "测试用户未提供代码。",
        }

        with self.assertRaises(ValidationError):
            validate_supervisor_plan(invalid_format, explicit_fund_codes={"159103"})
        with self.assertRaisesRegex(PlanValidationError, "未明确提供"):
            validate_supervisor_plan(unprovided, explicit_fund_codes={"159103"})

    def test_planning_error_routes_to_failure_handler(self) -> None:
        self.assertEqual(
            should_continue_planning(
                {
                    "planning_error": "循环依赖",
                    "plan": [],
                }
            ),
            "planning_failure",
        )


if __name__ == "__main__":
    unittest.main()
