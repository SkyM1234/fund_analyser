"""多Agent架构的主控制器 - LangGraph编排"""
import copy
import logging
import time
from typing import Literal
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.graph import Edge
from langgraph.types import Send
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agent.multi_agent_state import (
    MultiAgentState,
    get_blocked_tasks,
    get_ready_tasks,
    is_plan_complete,
)
from app.agent.state_reducers import PlanPatches, TaskPatch
from app.agent.supervisor import supervisor_node
from app.agent.rag_agent import rag_agent_node
from app.agent.market_agent import market_agent_node
from app.agent.arbiter_agent import arbiter_agent_node
from app.agent.compliance_agent import compliance_agent_node
from app.agent.synthesizer import synthesizer_node
from app.agent.reflection_agent import global_reflection_node
from app.services.router import route_query

logger = logging.getLogger(__name__)

MAX_COMPLIANCE_RETRIES = 1  # 合规不通过后最多允许 synthesizer 重新生成的次数


# ===== 路由节点 =====
async def route_node(state: MultiAgentState):
    """前置路由节点：识别意图和基金代码，写入 route_result。

    基金识别改为两级RAG方案：
    - 快速路径：正则匹配6位数字代码（在 router.py 内完成）
    - 语义路径：调用 rag_identify_funds MCP 工具（在 router.py 内完成）
    - 历史回填：当前消息识别不到基金时，从近几轮历史消息补充
    不再需要拉取全量基金注册表。
    """
    messages = state["messages"]

    if not messages:
        return {"route_result": None}

    # 获取最新用户消息
    user_msg = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_msg = msg.content
            break

    if not user_msg:
        return {"route_result": None}

    # 传入完整历史，让路由器在追问场景下能结合上下文识别基金和意图
    route_result = await route_query(user_msg, history_messages=messages)
    logger.info(f"[Route] intent={route_result.intent}")

    return {"route_result": route_result}


# ===== 路由后置分支：敏感问题硬拦截 =====
def route_after_intent(state: MultiAgentState) -> Literal["supervisor", "sensitive_refusal"]:
    """route 节点之后的硬性分支：intent == sensitive 时直接拒绝，不进入 supervisor 规划。

    route_result.intent 此前只是写入 state 供 supervisor 的 prompt 参考（软约束），
    LLM 存在概率不遵循该提示继续规划任务；敏感问题（投资建议/收益预测）属于安全
    边界，需要用图结构保证确定性拒绝，因此在此处直接短路。
    """
    route_result = state.get("route_result")
    if route_result is not None and route_result.intent == "sensitive":
        logger.info("[Router] intent=sensitive, short-circuit to sensitive_refusal")
        return "sensitive_refusal"
    return "supervisor"


# ===== 敏感问题拒绝处理 =====
def handle_sensitive_refusal(state: MultiAgentState):
    """敏感问题（投资建议/推荐/收益预测）的固定拒绝话术，由 route_after_intent 硬拦截触发。"""
    refusal_message = """抱歉，我不能提供具体的投资建议、基金推荐或收益预测。

作为基金年报分析助手，我可以帮助您：
- 查询基金的基本信息（规模、费率、基金经理等）
- 了解基金的持仓情况和投资策略
- 对比不同基金的历史数据和配置
- 解释基金相关的专业概念

有其他问题欢迎继续提问！"""

    from langchain_core.messages import AIMessage

    return {
        "final_answer": refusal_message,
        "messages": [AIMessage(content=refusal_message)],
    }


# ===== 任务调度路由函数 =====
# 注意：Send 只能从 add_conditional_edges 的路由函数返回，不能从普通 node 返回
# （普通 node 的返回值会被当作状态更新字典处理，返回 list[Send] 会触发 InvalidUpdateError: Expected dict, got [Send(...), ...]）。
# 因此 dispatch_tasks 不注册为节点，而是直接作为 supervisor / batch_reflection 之后的条件边使用。
def prepare_task_dispatch_node(state: MultiAgentState):
    """持久化当前批次的 running 状态，再交给 Send 路由并发分发。

    条件边不能写状态，因此不能在 Send 分支副本中临时修改 plan。这里先在主
    状态写入小粒度 patch，确保任务的生命周期状态不会因并发合并而丢失。
    """
    ready_tasks = get_ready_tasks(state)
    if not ready_tasks:
        return {"dispatch_task_ids": []}

    now = time.monotonic()
    ready_ids = [task["task_id"] for task in ready_tasks]
    logger.info(f"[Dispatcher] Fan-out {len(ready_tasks)} tasks in parallel: {list(ready_ids)}")
    return {
        "plan": PlanPatches([
            TaskPatch(
                task_id=task_id,
                changes={"status": "running", "started_at": now},
            )
            for task_id in ready_ids
        ]),
        "dispatch_task_ids": ready_ids,
    }


def dispatch_tasks(state: MultiAgentState) -> list[Send] | Literal["synthesizer"]:
    """将已持久化为 running 的任务作为独立不可变输入并发分发。"""
    ready_ids = state.get("dispatch_task_ids", [])
    if not ready_ids:
        logger.info("[Dispatcher] No prepared tasks, moving to synthesizer")
        return "synthesizer"

    tasks_by_id = {task["task_id"]: task for task in state.get("plan", [])}
    sub_results = state.get("sub_results", {})
    sends = []
    for task_id in ready_ids:
        task = tasks_by_id.get(task_id)
        if task is None:
            logger.error("[Dispatcher] Prepared task %s disappeared from plan", task_id)
            continue
        agent_name = task.get("assigned_agent", "")
        dependency_results = {
            dependency_id: copy.deepcopy(
                sub_results.get(
                    dependency_id,
                    tasks_by_id[dependency_id].get(
                        "result",
                        "Dependency completed without a saved result.",
                    ),
                )
            )
            for dependency_id in task.get("depends_on", [])
        }
        task_input = copy.deepcopy(task)
        task_input["dependency_results"] = dependency_results
        task_state = {
            "messages": state.get("messages", []),
            "route_result": state.get("route_result"),
            "task_input": task_input,
            "current_task_id": task_id,
            "current_agent": agent_name,
        }
        if agent_name == "rag_agent":
            sends.append(Send("rag_agent", task_state))
        elif agent_name == "market_agent":
            sends.append(Send("market_agent", task_state))
        elif agent_name == "arbiter_agent":
            sends.append(Send("arbiter_agent", task_state))
        else:
            sends.append(Send("agent_error_handler", task_state))

    return sends or "synthesizer"


def block_dependent_tasks_node(state: MultiAgentState):
    """将失败或阻断依赖的下游任务标记为 blocked，而非继续派发。"""
    blocked_tasks = get_blocked_tasks(state)
    if not blocked_tasks:
        return {}

    blocked_ids = [task["task_id"] for task in blocked_tasks]
    known_blockers = {
        task["task_id"]
        for task in state.get("plan", [])
        if task["status"] in ("failed", "blocked")
    }
    known_blockers.update(state.get("failed_tasks", []))
    known_blockers.update(state.get("blocked_tasks", []))

    patches = []
    results = {}
    for task in blocked_tasks:
        blocking_dependencies = [
            dep_id for dep_id in task.get("depends_on", [])
            if dep_id in known_blockers
        ]
        result = (
            "未执行：依赖任务 "
            f"{', '.join(blocking_dependencies)} 未成功完成，当前任务已阻断。"
        )
        patches.append(TaskPatch(
            task_id=task["task_id"],
            changes={
                "status": "blocked",
                "error": result,
                "result": result,
            },
        ))
        results[task["task_id"]] = result
        known_blockers.add(task["task_id"])

    logger.warning("[DependencyBlocker] Blocked tasks: %s", blocked_ids)
    return {
        "plan": PlanPatches(patches),
        "blocked_tasks": blocked_ids,
        "sub_results": results,
    }


# ===== 条件路由函数 =====
def should_continue_planning(state: MultiAgentState):
    """决定是否进入下一批调度。"""
    if state.get("planning_error"):
        return "planning_failure"

    plan = state.get("plan", [])

    if not plan:
        return "synthesizer"

    if is_plan_complete(state):
        logger.info("[Router] All tasks completed, moving to synthesizer")
        return "synthesizer"

    if get_blocked_tasks(state):
        return "dependency_blocker"

    if get_ready_tasks(state):
        return "task_dispatcher"

    logger.warning("[Router] Plan has no ready tasks before completion, moving to synthesizer")
    return "synthesizer"


def handle_planning_failure(state: MultiAgentState):
    """计划未通过硬校验时直接结束，不让非法 DAG 进入执行或汇总。"""
    error = state.get("planning_error", "任务规划校验失败")
    message = f"抱歉，系统未能生成可安全执行的任务计划：{error}"

    from langchain_core.messages import AIMessage

    return {
        "final_answer": message,
        "messages": [AIMessage(content=message)],
    }


def after_batch_reflection(state: MultiAgentState):
    """global_reflection 完成后：有待调度任务（含澄清任务）→ 分发，否则 → synthesizer。

    clarification_round 在 global_reflection_node 内已自增并受 MAX_CLARIFICATION_ROUNDS 上限控制，
    此处只需判断计划里是否还有 pending 任务即可，不再额外判断轮次。
    """
    plan = state.get("plan", [])
    has_pending = any(t["status"] == "pending" for t in plan)
    if not has_pending and (not plan or is_plan_complete(state)):
        logger.info("[Router] All tasks done after global reflection, moving to synthesizer")
        return "synthesizer"
    if get_blocked_tasks(state):
        logger.info("[Router] Blocking tasks with unsuccessful dependencies")
        return "dependency_blocker"
    if get_ready_tasks(state):
        logger.info("[Router] Pending tasks after global reflection, dispatching next batch")
        return "task_dispatcher"
    logger.warning("[Router] Pending tasks are not runnable after global reflection, moving to synthesizer")
    return "synthesizer"


def check_compliance(state: MultiAgentState) -> Literal["end", "synthesizer_retry", "compliance_failure"]:
    """检查合规结果

    不通过时：若合规重试次数未超过上限，回到 synthesizer 带着失败原因重新生成答案；
    超过上限则放弃重试，走固定拒绝文案兜底。
    """
    compliance_passed = state.get("compliance_passed", True)

    if compliance_passed:
        return "end"

    # compliance_agent_node 在判定不通过时已将 compliance_retry_count 加 1，
    # 此处的值即"即将进行的重试序号"，因此用 <= 判断是否还在预算内
    retry_count = state.get("compliance_retry_count", 0)
    if retry_count <= MAX_COMPLIANCE_RETRIES:
        logger.warning(
            f"[Router] Compliance check failed (retry {retry_count}/{MAX_COMPLIANCE_RETRIES}), "
            f"back to synthesizer"
        )
        return "synthesizer_retry"

    logger.warning("[Router] Compliance check failed, retry budget exhausted, using fallback refusal")
    return "compliance_failure"


# ===== 合规通过后落盘最终答案 =====
def commit_answer_node(state: MultiAgentState):
    """合规检查通过后，才将最终答案写入消息历史。

    synthesizer 每次执行（含合规重试）只覆盖 final_answer，不直接写 messages，
    避免未通过合规审查的中间版本被 add_messages 追加并持久化到对话历史里。
    """
    final_answer = state.get("final_answer", "")

    from langchain_core.messages import AIMessage

    return {
        "messages": [AIMessage(content=final_answer)],
    }


# ===== 合规拒绝处理 =====
def handle_compliance_failure(state: MultiAgentState):
    """处理合规检查失败的情况"""
    reason = state.get("compliance_reason", "内容不符合合规要求")

    from langchain_core.messages import AIMessage

    refusal_message = f"""抱歉，我不能提供具体的投资建议、基金推荐或收益预测。

{reason}

作为基金年报分析助手，我可以帮助您：
- 查询基金的基本信息（规模、费率、基金经理等）
- 了解基金的持仓情况和投资策略
- 对比不同基金的历史数据和配置
- 解释基金相关的专业概念

有其他问题欢迎继续提问！"""

    return {
        "final_answer": refusal_message,
        "messages": [AIMessage(content=refusal_message)],
    }


# ===== Agent错误处理 =====
def handle_agent_error(state: MultiAgentState):
    """处理Unknown Agent错误

    当Supervisor分配了未知的Agent类型时触发，这通常表示Supervisor的LLM输出格式错误

    处理策略：
    1. 标记当前任务为failed
    2. 记录详细错误信息
    3. 触发反思节点尝试恢复
    """
    task_input = state.get("task_input")
    current_task_id = task_input["task_id"] if task_input else state.get("current_task_id")
    current_agent = state.get("current_agent", "unknown")

    logger.error(f"[AgentErrorHandler] System error - Unknown agent: {current_agent} for task: {current_task_id}")
    if not current_task_id:
        return {}

    error = f"系统错误: 未知的Agent类型 '{current_agent}'"
    result = f"系统错误：无法识别的Agent类型 '{current_agent}'，任务无法执行"
    changes = {
        "status": "failed",
        "error": error,
        "result": result,
    }
    if task_input:
        finished_at = time.monotonic()
        changes["finished_at"] = finished_at
        started_at = task_input.get("started_at")
        if started_at is not None:
            changes["duration_ms"] = (finished_at - started_at) * 1000

    return {
        "plan": TaskPatch(current_task_id, changes),
        "sub_results": {current_task_id: result},
        "failed_tasks": [current_task_id],
    }


# ===== 构建图 =====
def build_multi_agent_graph(checkpointer: BaseCheckpointSaver):
    """构建多Agent编排图（并行 fan-out 版本）"""

    graph = StateGraph(MultiAgentState)

    # 添加节点
    graph.add_node("route", route_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("task_dispatcher", prepare_task_dispatch_node)
    graph.add_node("dependency_blocker", block_dependent_tasks_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("market_agent", market_agent_node)
    graph.add_node("arbiter_agent", arbiter_agent_node)
    graph.add_node("batch_reflection", global_reflection_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("compliance", compliance_agent_node)
    graph.add_node("commit_answer", commit_answer_node)
    graph.add_node("compliance_failure_handler", handle_compliance_failure)
    graph.add_node("planning_failure_handler", handle_planning_failure)
    graph.add_node("sensitive_refusal", handle_sensitive_refusal)
    graph.add_node("agent_error_handler", handle_agent_error)

    # 构建流程
    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        route_after_intent,
        {
            "supervisor": "supervisor",
            "sensitive_refusal": "sensitive_refusal",
        }
    )

    # Supervisor → 各专家 Agent（Send 并发 fan-out）或直接汇总
    graph.add_conditional_edges(
        "supervisor",
        should_continue_planning,
        {
            "rag_agent": "rag_agent",
            "market_agent": "market_agent",
            "arbiter_agent": "arbiter_agent",
            "agent_error_handler": "agent_error_handler",
            "task_dispatcher": "task_dispatcher",
            "dependency_blocker": "dependency_blocker",
            "planning_failure": "planning_failure_handler",
            "synthesizer": "synthesizer",
        }
    )
    graph.add_conditional_edges(
        "task_dispatcher",
        dispatch_tasks,
        {
            "rag_agent": "rag_agent",
            "market_agent": "market_agent",
            "arbiter_agent": "arbiter_agent",
            "agent_error_handler": "agent_error_handler",
            "synthesizer": "synthesizer",
        },
    )
    graph.add_conditional_edges(
        "dependency_blocker",
        should_continue_planning,
        {
            "task_dispatcher": "task_dispatcher",
            "dependency_blocker": "dependency_blocker",
            "synthesizer": "synthesizer",
        },
    )

    # 各专家 Agent 完成后汇入 batch_reflection。rag_agent/market_agent 内部已在
    # ReAct 循环耗尽轮次时原地重试（改写 query 后重跑，不跨节点/跨并行分支传状态），
    # 到这里的 failed 任务都是重试预算已用完的最终结果。
    graph.add_edge("rag_agent", "batch_reflection")
    graph.add_edge("market_agent", "batch_reflection")
    graph.add_edge("arbiter_agent", "batch_reflection")
    graph.add_edge("agent_error_handler", "batch_reflection")

    # batch_reflection 完成后：继续调度下一批（Send 并发 fan-out）or 汇总
    graph.add_conditional_edges(
        "batch_reflection",
        after_batch_reflection,
        {
            "rag_agent": "rag_agent",
            "market_agent": "market_agent",
            "arbiter_agent": "arbiter_agent",
            "agent_error_handler": "agent_error_handler",
            "task_dispatcher": "task_dispatcher",
            "dependency_blocker": "dependency_blocker",
            "synthesizer": "synthesizer",
        }
    )

    # Synthesizer → Compliance
    graph.add_edge("synthesizer", "compliance")

    # Compliance → 落盘最终答案 或 回synthesizer重新生成 或 拒绝兜底
    graph.add_conditional_edges(
        "compliance",
        check_compliance,
        {
            "end": "commit_answer",
            "synthesizer_retry": "synthesizer",
            "compliance_failure": "compliance_failure_handler",
        }
    )

    graph.add_edge("commit_answer", END)
    graph.add_edge("compliance_failure_handler", END)
    graph.add_edge("planning_failure_handler", END)
    graph.add_edge("sensitive_refusal", END)

    # 编译
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("[Graph] Multi-Agent Graph compiled successfully")
    return compiled


# ===== 可视化功能 =====
def export_graph_to_mermaid(checkpointer=None, output_file: str = None) -> str:
    """导出Graph为Mermaid图表

    Args:
        checkpointer: 检查点保存器（可选）
        output_file: 输出文件路径（可选），如果提供则写入文件

    Returns:
        Mermaid图表字符串
    """
    graph = build_multi_agent_graph(checkpointer)

    try:
        # 使用LangGraph内置的Mermaid导出
        mermaid_str = graph.get_graph().draw_mermaid()

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(mermaid_str, encoding="utf-8")
            logger.info(f"[Graph] Mermaid diagram exported to {output_file}")

        return mermaid_str
    except Exception as e:
        logger.error(f"[Graph] Failed to export Mermaid: {e}")
        return ""


def export_graph_to_png(checkpointer=None, output_file: str = "docs/graph_architecture.png") -> bool:
    """导出Graph为PNG图片

    Args:
        checkpointer: 检查点保存器（可选）
        output_file: 输出文件路径

    Returns:
        是否成功导出

    Note:
        需要安装: pip install pygraphviz
        在Windows上需要先安装Graphviz: https://graphviz.org/download/
    """
    graph = build_multi_agent_graph(checkpointer)

    try:
        from IPython.display import Image

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用LangGraph内置的PNG导出
        png_data = graph.get_graph().draw_mermaid_png()

        with open(output_path, "wb") as f:
            f.write(png_data)

        logger.info(f"[Graph] PNG diagram exported to {output_file}")
        return True
    except ImportError as e:
        logger.warning(f"[Graph] PNG export requires pygraphviz: {e}")
        logger.info("[Graph] Install with: pip install pygraphviz")
        return False
    except Exception as e:
        logger.error(f"[Graph] Failed to export PNG: {e}")
        return False


def print_graph_structure(checkpointer=None):
    """打印Graph的结构信息（用于调试）

    Args:
        checkpointer: 检查点保存器（可选）
    """
    graph = build_multi_agent_graph(checkpointer)

    try:
        graph_obj = graph.get_graph()

        print("\n" + "="*60)
        print("Multi-Agent Graph Structure")
        print("="*60)

        # 节点列表
        nodes = list(graph_obj.nodes.keys())
        print(f"\n📦 Nodes ({len(nodes)}):")
        for node in nodes:
            print(f"  - {node}")

        # 边列表
        print(f"\n🔗 Edges:")
        for item in graph_obj.edges:
            if isinstance(item, Edge):
                source = item.source
                target = item.target
                conditional = item.conditional
                print(f"  {source} → {target} (is conditional: {conditional})")

        print("\n" + "="*60 + "\n")

    except Exception as e:
        logger.error(f"[Graph] Failed to print structure: {e}")


# ===== CLI工具 =====
if __name__ == "__main__":
    """命令行工具：生成Graph可视化"""

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s"
    )

    print("\n🎨 Multi-Agent Graph Visualization Tool\n")

    # 创建graph
    print("📊 Creating graph...")

    # 打印结构
    print_graph_structure()

    # 导出Mermaid
    curr_dir = Path(__file__).parent.parent.parent
    mermaid_file = curr_dir / "docs" / "graph_mermaid.md"
    print(f"📄 Exporting Mermaid diagram to {mermaid_file}...")
    mermaid_str = export_graph_to_mermaid(output_file=mermaid_file)
    if mermaid_str:
        print(f"   ✓ Mermaid diagram exported ({len(mermaid_str)} chars)")

    # 导出PNG
    png_file = curr_dir / "docs" / "graph_architecture.png"
    print(f"🖼️  Exporting PNG diagram to {png_file}...")
    if export_graph_to_png(output_file=png_file):
        print(f"   ✓ PNG diagram exported")
    else:
        print(f"   ✗ PNG export failed (pygraphviz not installed)")

    print("\n✅ Done!\n")
