"""Supervisor Agent - 任务规划与调度"""
import logging
import json
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.llm_concurrency import llm_ainvoke
from app.agent.multi_agent_state import MultiAgentState, SubTask
from app.agent.state_reducers import CLEARED, NewPlan
from app.services.router import RouteResult
from app.tools.conversation_utils import format_history_for_prompt
from app.tools.llm_json import extract_json_block

logger = logging.getLogger(__name__)


SUPERVISOR_SYSTEM_PROMPT = """你是基金分析系统的 Supervisor Agent，负责任务规划与调度。

最小充分计划（必须遵守）：
- 仅规划回答用户当前问题所必需的任务；已有任务可以覆盖的问题不得拆出补充背景、泛化分析或扩展比较任务。
- 信息足以回答用户问题时，不要为追求更全面而增加额外任务或数据维度。

你的职责：
1. 分析用户问题，拆解成可独立执行的子任务
2. 为每个子任务分配合适的专家Agent
3. 识别任务间的依赖关系
4. 判断是否需要多步骤协作

可用的专家Agent：
- **rag_agent**: 检索本地基金年报内容（投资策略、持仓、业绩等历史数据）
- **market_agent**: 获取实时数据（当前净值、实时估值、最新持仓等实时数据），只有当用户问题明确涉及实时数据时才调用

任务类型(task_type)：
- "rag_search": 年报检索任务
- "market_data": 实时数据查询任务
- "general_qa": 通用问答（不需要特定工具）

规划原则：
1. **单基金查询** → 1个任务，fund_codes 只含该基金的一个代码
2. **多基金查询** → N个任务，每个任务负责一只基金，fund_codes 只含该任务对应的一个代码
3. **历史+实时数据** → 2个任务（rag_search + market_data），fund_codes 保持一致
4. **简单闲聊** → 0个任务（直接返回空列表）

⚠️ fund_codes 字段的规则：
- 每个子任务的 fund_codes **只能包含一个基金代码**
- 禁止在同一任务里放多个代码

错误示例（不能这样）：
- fund_codes=["159103","159299"]  ← 一个任务两个代码

正确示例（应该这样）：
- 任务1：fund_codes=["159103"]
- 任务2：fund_codes=["159299"]

输出格式（JSON）：
```json
{
  "plan": [
    {
      "task_id": "t1",
      "task_type": "rag_search",
      "description": "检索159103的投资策略",
      "assigned_agent": "rag_agent",
      "fund_codes": ["159103"],
      "query": "投资策略",
      "depends_on": [],
      "status": "pending"
    }
  ],
  "plan": [
    {
      "task_id": "t2",
      "task_type": "market_data",
      "description": "查询科创债ETF万家的当前净值",
      "assigned_agent": "market_agent",
      "fund_codes": [],
      "query": "当前净值",
      "depends_on": [],
      "status": "pending"
    }
  ],
  "reasoning": "t1是单基金年报查询，分配给rag_agent；t2是单基金实时数据查询，分配给market_agent"
}
```

⚠️ 注意：
- task_id 必须唯一
- depends_on 列表中的 task_id 必须在前面定义过
- 简单问题不要过度拆解
- fund_codes 字段**只能填用户消息中明确出现的6位数字代码**（如用户说了"159103"）
- 如果用户只说了基金名称/简称/别名而没有给出代码，fund_codes 必须留空列表 []，由后续 Agent 通过语义搜索自行识别代码，禁止根据已有知识猜测或推断基金代码
"""


async def supervisor_node(state: MultiAgentState) -> dict[str, Any]:
    """Supervisor 节点：生成任务规划

    Returns:
        包含 plan 的字典，会合并到 state 中
    """
    messages = state["messages"]
    route_result = state.get("route_result")

    # 获取用户最新问题
    user_query = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    if not user_query:
        logger.warning("[Supervisor] No user query found")
        return {"plan": []}

    existing_plan = state.get("plan", [])
    synthesis_complete = state.get("synthesis_complete", False)

    # 如果上一轮已完成，清空状态，准备新一轮
    if synthesis_complete and existing_plan:
        logger.info("[Supervisor] Previous round completed, starting new planning")
        result = await _generate_new_plan(user_query, route_result, messages)

        # 方案B：若 LLM 规划出空任务列表，且当前消息是追问/信息不足，
        # 用历史消息里的原始问题重新规划一次
        if not result.get("plan"):
            fallback_query = _find_last_substantive_query(messages)
            if fallback_query and fallback_query != user_query:
                logger.info(
                    f"[Supervisor] Empty plan on followup, retrying with history query: "
                    f"{fallback_query[:60]}"
                )
                result = await _generate_new_plan(fallback_query, route_result, messages)

        return result

    # 如果已有未完成的计划，跳过（避免重复规划）
    if existing_plan and not synthesis_complete:
        logger.info("[Supervisor] Plan already exists and not completed, skipping")
        return {}

    # 生成新计划
    logger.info("[Supervisor] Generating new plan")
    return await _generate_new_plan(user_query, route_result, messages)


def _find_last_substantive_query(messages: list) -> str | None:
    """从历史消息中找到最近一条信息量足够的用户问题（排除当前消息和追问类短消息）。"""
    import re
    followup_patterns = [
        r'^(再|重新|重试|再试|继续)',
        r'^(好的|ok|嗯|那|那么|然后)',
        r'.{0,10}(试一下|试试|试一试|重来)',
        r'^(现在|这次|这回)',
    ]

    found_current = False
    for msg in reversed(messages):
        if not isinstance(msg, HumanMessage):
            continue
        if not found_current:
            found_current = True
            continue  # 跳过当前消息
        text = msg.content.strip()
        # 长度足够且不是追问特征词
        if len(text) < 5:
            continue
        is_followup = any(re.search(p, text, re.IGNORECASE) for p in followup_patterns)
        if not is_followup:
            return text
    return None


async def _generate_new_plan(user_query: str, route_result: RouteResult | None, messages: list) -> dict[str, Any]:
    """生成新的任务计划

    Args:
        user_query: 用户当前问题
        route_result: 路由结果
        messages: 完整的对话历史（用于理解上下文）
    """

    # 调用 LLM 生成规划
    settings = get_settings()
    llm = ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.1,  # 规划需要更确定性
    )

    # 使用共用函数格式化对话历史
    history_text = format_history_for_prompt(messages, rounds=3, exclude_last=True)

    prompt = f"""请为以下问题生成执行计划：
{history_text}
当前用户问题：{user_query}

路由信息：
- 意图类型：{route_result.intent if route_result else "unknown"}

请输出 JSON 格式的任务规划。"""

    try:
        response = await llm_ainvoke(llm, [
            {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        # 解析 JSON
        content = extract_json_block(response.content)

        plan_data = json.loads(content)
        plan = plan_data.get("plan", [])
        reasoning = plan_data.get("reasoning", "")
        
        logger.info(f"[Supervisor] Generated plan with {len(plan)} tasks")
        logger.info(f"[Supervisor] Reasoning: {reasoning}")
        
        # 验证并标准化任务
        validated_plan = []
        for task in plan:
            raw_fund_codes = task.get("fund_codes", [])

            # 后校验：rag/market 任务的 fund_codes 不能超过一个代码
            # 如果 LLM 仍然输出了多个，自动拆开
            if (
                len(raw_fund_codes) > 1
                and task.get("task_type") in ("rag_search", "market_data")
                and task.get("assigned_agent") in ("rag_agent", "market_agent")
            ):
                logger.warning(
                    f"[Supervisor] Task {task.get('task_id')} has multiple fund_codes "
                    f"{raw_fund_codes}, splitting into separate tasks"
                )
                for i, code in enumerate(raw_fund_codes):
                    split_task = SubTask(
                        task_id=f"{task.get('task_id', 't')}_split{i+1}",
                        task_type=task.get("task_type", "rag_search"),
                        description=f"{task.get('description', '')}（{code}）",
                        assigned_agent=task.get("assigned_agent", "rag_agent"),
                        fund_codes=[code],
                        query=task.get("query", user_query),
                        depends_on=task.get("depends_on", []),
                        status="pending",
                        result=None,
                        error=None,
                    )
                    validated_plan.append(split_task)
                continue  # 跳过原始任务

            validated_task = SubTask(
                task_id=task.get("task_id", f"t{len(validated_plan)+1}"),
                task_type=task.get("task_type", "general_qa"),
                description=task.get("description", ""),
                assigned_agent=task.get("assigned_agent", "rag_agent"),
                fund_codes=raw_fund_codes,
                query=task.get("query", user_query),
                depends_on=task.get("depends_on", []),
                status="pending",
                result=None,
                error=None,
            )
            validated_plan.append(validated_task)
        
        return {
            "plan": NewPlan(tasks=validated_plan),
            # 清空之前的执行状态（reducer 管理的字段用 CLEARED 显式清空）
            "current_task_id": None,
            "completed_tasks": CLEARED,
            "failed_tasks": CLEARED,
            "blocked_tasks": CLEARED,
            "sub_results": CLEARED,
            "current_agent": None,
            "agent_history": [],
            "reflection_count": 0,
            "confidence_score": None,
            "needs_reflection": False,
            "conflict_annotations": CLEARED,
            "clarification_round": 0,
            "compliance_passed": True,
            "compliance_reason": None,
            "final_answer": None,
            "synthesis_complete": False,
        }

    except Exception as e:
        logger.error(f"[Supervisor] Failed to generate plan: {e}")
        # 降级：生成简单的单任务计划
        fallback_plan = [SubTask(
            task_id="t1",
            task_type="general_qa",
            description=f"处理用户问题: {user_query[:50]}...",
            assigned_agent="rag_agent",
            fund_codes=[],
            query=user_query,
            depends_on=[],
            status="pending",
            result=None,
            error=None,
        )]
        logger.info("[Supervisor] Using fallback single-task plan")
        return {
            "plan": NewPlan(tasks=fallback_plan),
            # 清空之前的执行状态（reducer 管理的字段用 CLEARED 显式清空）
            "current_task_id": None,
            "completed_tasks": CLEARED,
            "failed_tasks": CLEARED,
            "blocked_tasks": CLEARED,
            "sub_results": CLEARED,
            "current_agent": None,
            "agent_history": [],
            "reflection_count": 0,
            "confidence_score": None,
            "needs_reflection": False,
            "conflict_annotations": CLEARED,
            "clarification_round": 0,
            "compliance_passed": True,
            "compliance_reason": None,
            "final_answer": None,
            "synthesis_complete": False,
        }
