"""Arbiter Agent - 冲突仲裁，专职裁决 rag_agent / market_agent 的数据冲突。

与 retrieval_agent 的关键区别：不绑定任何 MCP 工具，不自主检索外部信息，
只基于 reflection_agent 传入的双方原始结果做一次性裁决，避免成为第三个数据源。
"""
import logging
import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from app.core.deepseek_llm import create_chat_llm
from app.core.llm_concurrency import llm_ainvoke
from app.agent.multi_agent_state import MultiAgentState
from app.agent.state_reducers import TaskPatch
from app.agent.task_context import format_dependency_results
from app.tools.llm_json import extract_json_block
from app.tools.token_usage import record_usage

logger = logging.getLogger(__name__)


ARBITER_SYSTEM_PROMPT = """你是基金分析系统的冲突仲裁专家，负责对两个业务 Agent 的结果冲突做出裁决。

你的职责（严格限制）：
1. 基于用户提供的双方原始结果和冲突描述，判断应采信哪一方，或说明无法判定
2. 给出可解释的裁决理由

绝对禁止：
- 自行补充任何未在输入中出现的数据或外部信息（你没有工具可用，也不应假装查证）
- 对双方数值类冲突取中间值、折中处理（如 A 说 10 亿、B 说 20 亿，不能输出 15 亿）
- predetermined 偏向某一方，裁决必须基于输入中的证据（数据时间戳新旧、是否标注来源、与用户问题的语义匹配度等）

裁决优先级（从高到低，供你参考，不要求机械套用）：
1. 任务状态：若某一方原始结果标注"任务状态: failed"，其内容可能只是部分/兜底数据，通常应优先采信状态正常的一方；若双方都失败或失败方缺口正是用户提问的关键信息，判为 unresolved 并说明信息缺失，不要虚构结论
2. 数据时效性：若双方是同一维度但时间戳不同，新数据优先，且这通常不构成真正冲突
3. 维度匹配：若双方本质是不同维度，应判定为不构成冲突，而非强行二选一
4. 来源明确性：有明确基金代码/数据时间标注的结果优先于无标注的
5. 证据完整性：陈述更具体、可验证的结论优先

输出格式（JSON）：
```json
{
  "verdict": "adopt_a" | "adopt_b" | "not_conflicting" | "unresolved",
  "reasoning": "裁决理由，需具体指出依据",
  "conclusion": "面向用户的结论文本，说明最终应采信的信息或未能消解时的差异说明"
}
```

verdict 取值说明：
- adopt_a / adopt_b：明确采信某一方，conclusion 写出该方的结论内容
- not_conflicting：双方本质不冲突（如维度不同），conclusion 写出两者应如何共存呈现
- unresolved：无法判定，conclusion 需并列双方结论、来源和适用前提，不得强行二选一
"""


async def arbiter_agent_node(state: MultiAgentState) -> dict[str, Any]:
    """仲裁节点：对定向澄清任务（clarify_ 前缀）做一次性裁决，不做工具调用。"""

    current_task = state.get("task_input")
    if not current_task:
        logger.error("[arbiter_agent] No task_input")
        return {}
    current_task_id = current_task["task_id"]

    logger.info(f"[arbiter_agent] Executing task: {current_task['description']}")

    query = current_task.get("query", "")

    llm = create_chat_llm(temperature=0.1)

    token_usage: dict[str, dict[str, int]] = {}

    def _finish(result_text: str, status: str = "completed", error: str | None = None) -> dict[str, Any]:
        finished_at = time.monotonic()
        changes: dict[str, Any] = {
            "status": status,
            "result": result_text,
            "finished_at": finished_at,
        }
        if error:
            changes["error"] = error
        started_at = current_task.get("started_at")
        if started_at is not None:
            changes["duration_ms"] = (finished_at - started_at) * 1000

        update: dict[str, Any] = {
            "plan": TaskPatch(current_task_id, changes),
            "sub_results": {current_task_id: result_text},
            "token_usage": token_usage,
        }
        if status == "completed":
            update["completed_tasks"] = [current_task_id]
        else:
            update["failed_tasks"] = [current_task_id]
        return update

    try:
        response = await llm_ainvoke(llm, [
            SystemMessage(content=ARBITER_SYSTEM_PROMPT),
            HumanMessage(content=query + format_dependency_results(current_task)),
        ])
        usage = record_usage(f"arbiter_agent:{current_task_id}", response)
        for k, v in usage.items():
            existing = token_usage.setdefault(k, {})
            for field, val in v.items():
                existing[field] = existing.get(field, 0) + val

        content = extract_json_block(response.content)
        verdict_data = json.loads(content)

        verdict = verdict_data.get("verdict", "unresolved")
        reasoning = verdict_data.get("reasoning", "")
        conclusion = verdict_data.get("conclusion", "")

        result_text = f"[裁决:{verdict}] {conclusion}\n（理由：{reasoning}）"
        logger.info(f"[arbiter_agent] Task completed: verdict={verdict}")
        logger.info(f"[arbiter_agent] Conclusion: {conclusion}")
        logger.info(f"[arbiter_agent] Reasoning: {reasoning}")

        return _finish(result_text)

    except Exception as e:
        logger.error(f"[arbiter_agent] Execution failed: {e}", exc_info=True)
        return _finish(f"仲裁执行失败：{str(e)}", status="failed", error=str(e))
