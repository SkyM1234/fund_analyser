"""MultiAgentState 字段的 LangGraph reducer。

并发调度（多任务 fan-out）场景下，多个节点会在同一个 super-step 内并发返回
对同一 state 字段的增量更新。这些 reducer 定义"如何合并两份增量"，
使字段在并发写入下也能正确聚合，而不是后写覆盖先写。

当前流程仍是严格串行的，这些 reducer 在串行场景下退化为等价的顺序累加，
不改变现有行为；它们的价值在引入并行任务调度（Send API fan-out）之后才体现。

CLEARED 哨兵：
reducer 是"合并"语义，无法用空值（[]、{}）表达"清空重置"——空值会被当作
"这次没有更新"而保留旧值。当某个节点（如 supervisor 开始新一轮规划）需要
清空字段而不是合并时，必须显式返回 CLEARED，而不是空列表/空字典。
"""
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.agent.multi_agent_state import SubTask


class _Sentinel(Enum):
    """内部哨兵枚举。

    使用 Enum 而非自定义类，是因为 LangGraph 的 JsonPlusSerializer (_msgpack_default)
    对 Enum 有内置序列化路径（EXT_CONSTRUCTOR_SINGLE_ARG），能在 checkpoint 写入时
    正确往返，避免 "Type is not msgpack serializable" 错误。
    节点返回的更新字典会在 reducer 运行之前被 checkpoint 序列化，
    因此哨兵值必须可序列化。
    """

    CLEARED = "CLEARED"


CLEARED = _Sentinel.CLEARED


import dataclasses


@dataclasses.dataclass
class NewPlan:
    """标记一份"全新规划"（如 supervisor 开启新一轮），与增量更新区分。

    plan 字段在同一轮次内由多个并发 agent 节点按 task_id 合并更新
    （各自返回携带自己任务最新状态的完整列表快照），因此需要合并语义；
    但 supervisor 开启新一轮规划时产出的是全新的任务集合，task_id 从头编号，
    此时若仍按 task_id 合并，会把上一轮遗留的旧 task_id 永久保留在 plan 里。
    用 NewPlan 包裹新规划的列表，告诉 reducer 这是替换而不是合并。

    使用 dataclass 而非 list 子类，是因为 list 子类经 ormsgpack 序列化后会还原为
    普通 list，导致 isinstance(right, NewPlan) 在 checkpoint 往返后失效。
    dataclass 有内置序列化路径，能正确往返。
    """

    tasks: list


@dataclasses.dataclass
class TaskPatch:
    """单个任务的增量更新。

    并发 Agent 只能提交自己负责的任务变更，不能携带整份 plan 快照，
    否则后完成的分支会用旧状态覆盖其他分支已经完成的任务。
    """

    task_id: str
    changes: dict[str, Any]


@dataclasses.dataclass
class PlanPatches:
    """同一串行节点对多个任务提交的增量更新。"""

    patches: list[TaskPatch]


def _apply_task_patch(
    plan: "list[SubTask]",
    patch: TaskPatch,
) -> "list[SubTask]":
    updated = []
    found = False
    for task in plan:
        if task["task_id"] == patch.task_id:
            updated.append({**task, **patch.changes})
            found = True
        else:
            updated.append(task)
    if not found:
        raise ValueError(f"Task patch references unknown task_id: {patch.task_id}")
    return updated


def merge_plan(
    left: "list[SubTask]",
    right: "list[SubTask] | NewPlan | TaskPatch | PlanPatches | _Sentinel",
) -> "list[SubTask]":
    """按 task_id 合并两份任务列表。

    right 中的任务覆盖 left 中同 task_id 的任务（视为该任务的最新状态）；
    right 独有的 task_id 视为新任务，追加到结果中；顺序以 left 为主。
    right 为 CLEARED 时清空；right 为 NewPlan 时整体替换（新一轮规划）。
    """
    if right is CLEARED:
        return []
    if isinstance(right, NewPlan):
        return list(right.tasks)
    if isinstance(right, TaskPatch):
        return _apply_task_patch(left, right)
    if isinstance(right, PlanPatches):
        merged = left
        for patch in right.patches:
            merged = _apply_task_patch(merged, patch)
        return merged
    if not left:
        return right
    if not right:
        return left

    right_by_id = {t["task_id"]: t for t in right}
    merged = [right_by_id.pop(t["task_id"], t) for t in left]
    merged.extend(right_by_id.values())
    return merged


def merge_dict(left: dict[str, Any], right: "dict[str, Any] | _Sentinel") -> dict[str, Any]:
    """浅合并两个 dict，right 的键覆盖 left 的同名键（如 sub_results）。

    right 为 CLEARED 时直接清空。
    """
    if right is CLEARED:
        return {}
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


def merge_str_list_unique(left: list[str], right: "list[str] | _Sentinel") -> list[str]:
    """合并两个字符串列表并去重，保持首次出现的顺序（如 completed_tasks/failed_tasks）。

    right 为 CLEARED 时直接清空。
    """
    if right is CLEARED:
        return []
    if not left:
        return right
    if not right:
        return left
    merged = list(left)
    seen = set(left)
    for item in right:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def merge_conflict_list(
    left: "list[dict]", right: "list[dict] | _Sentinel"
) -> "list[dict]":
    """按 conflict_id 合并冲突标注列表，right 中的标注覆盖 left 中同 conflict_id 的条目。

    right 为 CLEARED 时清空（supervisor 开启新一轮规划时使用）。
    """
    if right is CLEARED:
        return []
    if not left:
        return list(right)
    if not right:
        return left

    right_by_id = {c["conflict_id"]: c for c in right}
    merged = [right_by_id.pop(c["conflict_id"], c) for c in left]
    merged.extend(right_by_id.values())
    return merged


def merge_token_usage(
    left: dict[str, dict[str, int]], right: "dict[str, dict[str, int]] | _Sentinel"
) -> dict[str, dict[str, int]]:
    """按调用点 key 累加 token 用量（如 {"supervisor": {...}, "rag_agent:t1": {...}}）。

    同一个 key 出现在 left 和 right 中时，对应的数值字段逐一相加，
    而不是覆盖——同一调用点在并发场景下可能被多个任务分支同时记账。
    right 为 CLEARED 时直接清空（用于 supervisor 开始新一轮规划）。
    """
    if right is CLEARED:
        return {}
    if not left:
        return right
    if not right:
        return left

    merged = {k: dict(v) for k, v in left.items()}
    for key, usage in right.items():
        if key not in merged:
            merged[key] = dict(usage)
            continue
        existing = merged[key]
        for field, value in usage.items():
            existing[field] = existing.get(field, 0) + value
    return merged


def merge_list_append(left: "list[dict]", right: "list[dict] | _Sentinel") -> "list[dict]":
    """简单拼接两份日志列表（如 tool_call_log），不去重、不按 key 合并。

    每一条都是一次独立的事件记录，直接顺序累加即可。
    right 为 CLEARED 时清空。
    """
    if right is CLEARED:
        return []
    if not left:
        return list(right)
    if not right:
        return left
    return [*left, *right]
