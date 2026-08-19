"""Supervisor 生成计划的结构与 DAG 校验。"""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TASK_AGENT_MAPPING: dict[str, set[str]] = {
    "rag_search": {"rag_agent"},
    "market_data": {"market_agent"},
    "general_qa": {"rag_agent", "analysis_agent"},
}


class PlanValidationError(ValueError):
    """计划字段结构之外的 DAG 约束不满足。"""


class PlannedTask(BaseModel):
    """Supervisor 可提交给执行图的单个任务。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]*$",
    )
    task_type: Literal["rag_search", "market_data", "general_qa"]
    description: str = Field(min_length=1, max_length=1000)
    assigned_agent: Literal["rag_agent", "market_agent", "analysis_agent"]
    fund_codes: list[str] = Field(max_length=50)
    query: str = Field(min_length=1, max_length=2000)
    depends_on: list[str]
    status: Literal["pending"]

    @field_validator("fund_codes")
    @classmethod
    def validate_fund_codes(cls, fund_codes: list[str]) -> list[str]:
        invalid_codes = [code for code in fund_codes if not re.fullmatch(r"\d{6}", code)]
        if invalid_codes:
            raise ValueError(f"基金代码必须是 6 位数字: {invalid_codes}")
        return fund_codes

    @field_validator("depends_on")
    @classmethod
    def validate_unique_dependencies(cls, depends_on: list[str]) -> list[str]:
        if len(depends_on) != len(set(depends_on)):
            raise ValueError("depends_on 不能包含重复任务 ID")
        return depends_on

    @model_validator(mode="after")
    def validate_agent_mapping(self) -> "PlannedTask":
        allowed_agents = TASK_AGENT_MAPPING[self.task_type]
        if self.assigned_agent not in allowed_agents:
            raise ValueError(
                f"task_type={self.task_type} 只能分配给 {sorted(allowed_agents)}"
            )
        if self.assigned_agent == "analysis_agent" and not self.depends_on:
            raise ValueError(
                "analysis_agent must depend on at least one upstream task"
            )
        return self


class SupervisorPlan(BaseModel):
    """Supervisor LLM 输出的完整计划。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    plan: list[PlannedTask]
    reasoning: str = Field(min_length=1, max_length=4000)


def validate_supervisor_plan(
    raw_plan: object,
    *,
    explicit_fund_codes: set[str],
    fund_scope: dict | None = None,
    require_non_empty_plan: bool = False,
) -> list[dict]:
    """解析并验证 Supervisor 的计划，返回可写入状态的规范化任务字典。"""
    parsed_plan = SupervisorPlan.model_validate(raw_plan)
    tasks = parsed_plan.plan
    if require_non_empty_plan and not tasks:
        raise PlanValidationError("当前基金数据查询必须生成至少一个检索任务")
    scoped_fund_codes = {
        fund.get("fund_code")
        for fund in (fund_scope or {}).get("funds", [])
        if isinstance(fund, dict) and fund.get("fund_code")
    }
    allowed_fund_codes = explicit_fund_codes | scoped_fund_codes
    task_ids = [task.task_id for task in tasks]

    if len(task_ids) != len(set(task_ids)):
        raise PlanValidationError("task_id 必须唯一")

    task_id_set = set(task_ids)
    for task in tasks:
        if task.task_id in task.depends_on:
            raise PlanValidationError(f"任务 {task.task_id} 不能依赖自身")

        unknown_dependencies = set(task.depends_on) - task_id_set
        if unknown_dependencies:
            raise PlanValidationError(
                f"任务 {task.task_id} 依赖不存在的任务: {sorted(unknown_dependencies)}"
            )

        unknown_fund_codes = set(task.fund_codes) - allowed_fund_codes
        if unknown_fund_codes:
            raise PlanValidationError(
                f"任务 {task.task_id} 使用了用户未明确提供的基金代码: "
                f"{sorted(unknown_fund_codes)}"
            )

    _validate_acyclic(tasks)
    return [task.model_dump() for task in tasks]


def _validate_acyclic(tasks: list[PlannedTask]) -> None:
    dependencies = {task.task_id: task.depends_on for task in tasks}
    visiting_path: list[str] = []
    visiting_index: dict[str, int] = {}
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting_index:
            cycle = visiting_path[visiting_index[task_id]:] + [task_id]
            raise PlanValidationError(
                f"检测到循环依赖: {' -> '.join(cycle)}"
            )

        visiting_index[task_id] = len(visiting_path)
        visiting_path.append(task_id)
        for dependency_id in dependencies[task_id]:
            visit(dependency_id)
        visiting_path.pop()
        del visiting_index[task_id]
        visited.add(task_id)

    for task_id in dependencies:
        visit(task_id)
