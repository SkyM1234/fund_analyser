"""多Agent架构的状态定义"""
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.services.router import RouteResult
from app.agent.state_reducers import (
    CLEARED,
    NewPlan,
    merge_conflict_list,
    merge_dict,
    merge_list_append,
    merge_plan,
    merge_str_list_unique,
    merge_token_usage,
)


# 子任务类型
TaskType = Literal[
    "rag_search",      # 年报检索
    "market_data",     # 实时市场数据
    "general_qa",      # 通用问答
]

# 冲突风险级别
ConflictRisk = Literal["low", "high"]


class ConflictAnnotation(TypedDict, total=False):
    """全局反思节点写入的冲突标注，Synthesizer 按此披露，不修改原始数据"""
    conflict_id: str                    # 冲突ID（如 "c1"）
    risk: ConflictRisk                  # 风险级别
    task_ids: list[str]                 # 涉及的任务ID
    field: str                          # 冲突字段描述（如 "净值"）
    description: str                    # 冲突详情（供 Synthesizer 披露）
    resolved: bool                      # 是否已通过澄清任务消解
    clarification_task_id: str | None   # 关联的澄清任务ID（高风险时填入）


class SubTask(TypedDict, total=False):
    """子任务定义"""
    task_id: str                    # 任务ID
    task_type: TaskType             # 任务类型
    description: str                # 任务描述
    assigned_agent: str             # 分配的Agent名称
    fund_codes: list[str]           # 涉及的基金代码
    query: str                      # 查询文本
    depends_on: list[str]           # 依赖的任务ID
    status: Literal["pending", "running", "completed", "failed", "blocked"]
    result: str | None              # 任务结果
    error: str | None               # 错误信息
    retry_count: int                # 重试次数
    retry_reason: str | None        # 重试原因
    reflected: bool                 # 是否已通过全局反思（防止重复评估）
    started_at: float | None        # 任务开始执行时间（time.monotonic()）
    finished_at: float | None       # 任务结束时间（time.monotonic()）
    duration_ms: float | None       # 执行耗时（毫秒），finished_at - started_at


class PlanExecution(TypedDict, total=False):
    """单次plan执行记录"""
    round_id: str                   # 轮次ID（如 "round_1", "round_2"）
    user_query: str                 # 用户问题
    plan: list[SubTask]             # 任务列表
    results: dict[str, str]         # task_id -> result
    final_answer: str | None        # 最终答案
    timestamp: str                  # 时间戳


class MultiAgentState(TypedDict):
    """多Agent系统状态"""
    # 消息历史
    messages: Annotated[list[BaseMessage], add_messages]
    
    # 路由结果
    route_result: RouteResult | None
    
    # ===== 多Agent扩展字段 =====

    # 任务规划
    # 按 task_id 合并；supervisor 开启新一轮规划时返回 CLEARED 以清空旧任务
    plan: Annotated[list[SubTask], merge_plan]
    plan_history: list[PlanExecution]  # 历史plan执行记录（累积，不清空）
    current_task_id: str | None     # 当前执行的任务ID
    task_input: SubTask | None      # Send 分支专用的不可变任务输入
    dispatch_task_ids: list[str]    # 当前调度批次的任务 ID，仅供 dispatcher 路由

    # 执行状态（按 task_id 合并去重；supervisor 新一轮规划时返回 CLEARED 清空）
    completed_tasks: Annotated[list[str], merge_str_list_unique]  # 已完成的任务ID列表
    failed_tasks: Annotated[list[str], merge_str_list_unique]     # 失败的任务ID列表
    blocked_tasks: Annotated[list[str], merge_str_list_unique]    # 被失败依赖阻断的任务ID列表

    # 子任务结果存储（按 task_id 合并；supervisor 新一轮规划时返回 CLEARED 清空）
    sub_results: Annotated[dict[str, str], merge_dict]  # {task_id: result_text}

    # Agent调度
    current_agent: str | None       # 当前执行的Agent名称
    agent_history: list[str]        # Agent调用历史

    # Token 用量统计（按调用点 key 累加，如 {"rag_agent:t1": {...}}）
    token_usage: Annotated[dict[str, dict[str, int]], merge_token_usage]

    # 工具调用记录（累积追加，如 {"agent", "task_id", "name", "args"}）
    # 与 messages 解耦：messages 是给 LLM 看的对话上下文，tool_call_log 是给评测/观测用的结构化日志
    tool_call_log: Annotated[list[dict], merge_list_append]

    # 反思与质量控制
    reflection_count: int           # 反思次数
    confidence_score: float | None  # 整体置信度
    needs_reflection: bool          # 是否需要反思

    # 全局反思冲突标注（按 conflict_id 合并去重）
    conflict_annotations: Annotated[list[ConflictAnnotation], merge_conflict_list]
    clarification_round: int        # 已触发的定向澄清轮次（上限 1）
    
    # 合规检查
    compliance_passed: bool         # 是否通过合规检查
    compliance_reason: str | None   # 合规检查结果说明
    compliance_retry_count: int     # 合规重试次数（重新汇总答案的次数）
    
    # 最终输出
    final_answer: str | None        # 最终答案
    synthesis_complete: bool        # 汇总是否完成


def create_initial_state(
    messages: list[BaseMessage],
    route_result: RouteResult | None = None
) -> MultiAgentState:
    """创建初始状态（用于向后兼容）"""
    return MultiAgentState(
        messages=messages,
        route_result=route_result,
        plan=[],
        plan_history=[],  # 初始化历史记录
        current_task_id=None,
        task_input=None,
        dispatch_task_ids=[],
        completed_tasks=[],
        failed_tasks=[],
        blocked_tasks=[],
        sub_results={},
        current_agent=None,
        agent_history=[],
        token_usage={},
        tool_call_log=[],
        reflection_count=0,
        confidence_score=None,
        needs_reflection=False,
        conflict_annotations=[],
        clarification_round=0,
        compliance_passed=True,
        compliance_reason=None,
        compliance_retry_count=0,
        final_answer=None,
        synthesis_complete=False,
    )


def get_next_pending_task(state: MultiAgentState) -> SubTask | None:
    """获取下一个可执行的待处理任务（依赖已满足）"""
    ready_tasks = get_ready_tasks(state)
    return ready_tasks[0] if ready_tasks else None


def get_ready_tasks(state: MultiAgentState) -> list[SubTask]:
    """返回所有依赖均已成功完成、可并发执行的待处理任务。"""
    completed = {
        task["task_id"]
        for task in state.get("plan", [])
        if task["status"] == "completed"
    }
    completed.update(state.get("completed_tasks", []))

    ready = []
    for task in state.get("plan", []):
        if task["status"] != "pending":
            continue
        depends_on = task.get("depends_on", [])
        if all(dep_id in completed for dep_id in depends_on):
            ready.append(task)
    return ready


def get_blocked_tasks(state: MultiAgentState) -> list[SubTask]:
    """返回因失败或已阻断依赖而不能执行的 pending 任务。

    通过固定点计算处理任意深度的依赖链：若 t1 failed，t2 依赖 t1，
    t3 又依赖 t2，则一次调用会同时返回 t2 和 t3。
    """
    plan = state.get("plan", [])
    terminal_blockers = {
        task["task_id"]
        for task in plan
        if task["status"] in ("failed", "blocked")
    }
    terminal_blockers.update(state.get("failed_tasks", []))
    terminal_blockers.update(state.get("blocked_tasks", []))

    blocked: list[SubTask] = []
    blocked_ids: set[str] = set()
    pending_tasks = [task for task in plan if task["status"] == "pending"]

    while True:
        newly_blocked = [
            task
            for task in pending_tasks
            if task["task_id"] not in blocked_ids
            and any(dep_id in terminal_blockers for dep_id in task.get("depends_on", []))
        ]
        if not newly_blocked:
            return blocked

        blocked.extend(newly_blocked)
        new_ids = {task["task_id"] for task in newly_blocked}
        blocked_ids.update(new_ids)
        terminal_blockers.update(new_ids)


def is_plan_complete(state: MultiAgentState) -> bool:
    """检查所有任务是否已进入终态。"""
    plan = state.get("plan", [])
    if not plan:
        return False
    
    return all(
        task["status"] in ["completed", "failed", "blocked"]
        for task in plan
    )
