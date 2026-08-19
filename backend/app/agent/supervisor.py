"""Supervisor Agent - 任务规划与调度"""
import logging
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.llm_concurrency import llm_ainvoke
from app.agent.multi_agent_state import MultiAgentState, SubTask
from app.agent.plan_validation import PlanValidationError, validate_supervisor_plan
from app.agent.state_reducers import CLEARED, NewPlan
from app.services.router import RouteResult
from app.tools.conversation_utils import format_history_for_prompt
from app.tools.llm_json import extract_json_block

logger = logging.getLogger(__name__)

MAX_PLAN_VALIDATION_RETRIES = 1


SUPERVISOR_SYSTEM_PROMPT = """你是基金分析系统的 Supervisor Agent，负责将用户当前问题转换为最小且完整的执行计划。

可用 Agent：
- rag_agent：检索基金年报及历史披露数据。
- market_agent：获取当前市场数据或基金实时数据。
- analysis_agent：对上游任务结果进行分析、比较、排序、聚合或计算。

任务类型：
- rag_search：年报或历史披露数据检索。
- market_data：当前市场数据或基金实时数据检索。
- general_qa：通用基金问题回答/分析。

基金范围规则：
- 用户提示会提供 SCOPE_PLANNING_CONTEXT，并明确当前是“已确认基金范围”还是“基金筛选，无预确认范围”。
- 已确认基金范围时，CONFIRMED_FUND_SCOPE 是权威范围。禁止自行重新识别、添加、删除或推断基金代码。
- 对于用户明确点名的多基金问题，必须使用范围中所有匹配的基金。
- 对于板块、指数、主题或“所有基金”问题，必须使用已确认的完整范围，并保留其 coverage_status。
- 如果 coverage_status 是 candidate 或 incomplete，任务描述和 reasoning 不得声称已经覆盖全部基金。
- 基金筛选、持仓反查等无预确认范围的问题，必须遵守 SCOPE_PLANNING_CONTEXT 的专门约束。

批量任务规划规则：
- 如果多只基金使用相同的 Agent、数据源、报告期间和指标，必须优先合并为一个批量任务，
  将全部相关基金代码放入同一个 fund_codes 列表。
- 批量任务描述中必须明确要求执行 Agent 分别检索并分别报告每只基金。
- 不要仅因为基金数量较多，就机械地为每只基金创建一个 task。
- 只有在数据源、时间维度、指标、Agent 或依赖关系不同时才拆分任务。例如，历史年报数据
  与当前市场数据需要拆分为两个任务，但两个任务可以使用相同的 fund_codes。
- 一个批量任务可以包含多个 fund_codes，数量不能超过 schema 限制。
- 每个基金代码必须来自用户明确写出的 6 位代码，或来自已确认的基金范围。

计划要求：
1. 只创建回答当前问题所必需的任务。
2. 一个 task 对应一个完整且一致的数据操作；兼容的多个基金目标应合并处理。
3. task_id 必须唯一且格式有效。
4. depends_on 中的每个任务 ID 必须指向已定义的任务，依赖图不能存在循环。
5. 每个新任务的 status 必须为 "pending"。
6. 不得添加背景介绍、投资建议、收益预测或与问题无关的比较任务。
7. 只有通用解释且无需查询任何基金数据时，才可以返回空的 plan。
8. 涉及基金具体数据、历史披露、日期、规模、份额、排序、时间线或比较时，必须生成至少一个任务，
   不得返回空的 plan。

只输出 JSON：
{
  "plan": [
    {
      "task_id": "t1",
      "task_type": "rag_search|market_data|general_qa",
      "description": "具体任务描述",
      "assigned_agent": "rag_agent|market_agent|analysis_agent",
      "fund_codes": ["159103", "159299"],
      "query": "需要检索的指标、时间和条件",
      "depends_on": [],
      "status": "pending"
    }
  ],
  "reasoning": "简要说明任务为何合并或拆分，以及范围覆盖状态"
}
"""


SUPERVISOR_SYSTEM_PROMPT += """

补充任务规划规则：
- analysis_agent 没有工具，只能对 depends_on 中任务已返回的结果进行转换、排序、比较、汇总、计算或解释。
- 仅分析上游结果的任务，必须使用 task_type="general_qa" 和 assigned_agent="analysis_agent"，
  并至少依赖一个上游任务；不得重复上游任务的检索要求。
- 例如，可先用一个 rag_search 任务检索多只基金的日期和份额，再创建依赖该任务的 general_qa，
  根据已检索数据生成时间线，并识别最早或最晚的基金。
- 仍需获取年报、披露或市场数据的任务，必须使用 rag_search 或 market_data；
  只有独立的 general_qa 确实需要检索时，才可继续使用既有的 rag_agent 路径。
- JSON schema 中，assigned_agent 只能是
  "rag_agent"、"market_agent" 或 "analysis_agent"。
"""


async def supervisor_node(state: MultiAgentState) -> dict[str, Any]:
    """Supervisor 节点：生成任务规划

    Returns:
        包含 plan 的字典，会合并到 state 中
    """
    messages = state["messages"]
    route_result = state.get("route_result")
    fund_scope = state.get("fund_scope")
    fund_scope_error = state.get("fund_scope_error")

    # 获取用户最新问题
    user_query = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    if not user_query:
        logger.warning("[Supervisor] No user query found")
        return {"plan": []}

    if (
        route_result
        and route_result.intent in ("single_fund_query", "cross_fund_query")
        and (not fund_scope or not fund_scope.get("funds"))
    ):
        error = fund_scope_error or "未确认到可用于检索的基金范围"
        logger.warning("[Supervisor] Refusing to plan without fund scope: %s", error)
        return _planning_failure_update(f"基金范围确认失败：{error}")

    existing_plan = state.get("plan", [])
    synthesis_complete = state.get("synthesis_complete", False)

    # 如果上一轮已完成，清空状态，准备新一轮
    if synthesis_complete and existing_plan:
        logger.info("[Supervisor] Previous round completed, starting new planning")
        result = await _generate_new_plan(user_query, route_result, messages, fund_scope)

        # 方案B：若 LLM 规划出空任务列表，且当前消息是追问/信息不足，
        # 用历史消息里的原始问题重新规划一次
        if not result.get("plan"):
            fallback_query = _find_last_substantive_query(messages)
            if fallback_query and fallback_query != user_query:
                logger.info(
                    f"[Supervisor] Empty plan on followup, retrying with history query: "
                    f"{fallback_query[:60]}"
                )
                result = await _generate_new_plan(fallback_query, route_result, messages, fund_scope)

        return result

    # 如果已有未完成的计划，跳过（避免重复规划）
    if existing_plan and not synthesis_complete:
        logger.info("[Supervisor] Plan already exists and not completed, skipping")
        return {}

    # 生成新计划
    logger.info("[Supervisor] Generating new plan")
    return await _generate_new_plan(user_query, route_result, messages, fund_scope)


def _find_last_substantive_query(messages: list) -> str | None:
    """从历史消息中找到最近一条信息量足够的用户问题（排除当前消息和追问类短消息）。"""
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


def _explicit_fund_codes(messages: list) -> set[str]:
    """只允许计划使用用户消息中明确出现过的 6 位基金代码。"""
    return {
        code
        for message in messages
        if isinstance(message, HumanMessage)
        for code in re.findall(r"(?<!\d)\d{6}(?!\d)", str(message.content))
    }


def _new_plan_update(validated_plan: list[dict]) -> dict[str, Any]:
    """写入通过校验的新计划并重置本轮执行状态。"""
    return {
        "plan": NewPlan(tasks=[SubTask(**task) for task in validated_plan]),
        "current_task_id": None,
        "task_input": None,
        "dispatch_task_ids": [],
        "planning_error": None,
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
        "compliance_retry_count": 0,
        "draft_answer": None,
        "final_answer": None,
        "synthesis_complete": False,
    }


def _scope_planning_context(
    route_result: RouteResult | None,
    fund_scope: dict | None,
) -> str:
    """构建范围规划上下文，匹配路由意图，确保计划符合基金范围约束。"""
    if route_result and route_result.intent == "fund_screening":
        return (
            "模式：基金筛选（未执行 fund_scope_agent，且没有预确认的基金范围）。\n"
            "这是从条件反查基金集合的问题，例如按持仓、行业、主题或指标筛选基金。\n"
            "必须创建至少一个面向 rag_agent 的任务，并将 fund_codes 设为 []；"
            "具体使用哪些工具由 rag_agent 根据任务自主决定。不得在计划阶段猜测、补全或写入"
            "检索尚未返回的基金代码。\n"
            "任务 query 必须保留完整筛选条件，并要求执行结果逐只列出命中的基金、"
            "基金代码及可核验的依据。reasoning 不得声称已预先覆盖全部基金。"
        )

    if fund_scope:
        return (
            "模式：已确认基金范围。\n"
            "CONFIRMED_FUND_SCOPE 是唯一可用于计划 fund_codes 的确认来源；"
            "严格遵守其中的 funds 和 coverage_status。"
        )

    return (
        "模式：无预确认基金范围。\n"
        "仅当用户问题不需要特定基金数据时才可创建空 fund_codes 的全局检索任务；"
        "不得猜测基金代码。"
    )


def _planning_failure_update(error: str) -> dict[str, Any]:
    """校验重试耗尽后的硬失败结果，不向执行图提交任何任务。"""
    return {
        **_new_plan_update([]),
        "planning_error": error,
    }


async def _generate_new_plan(
    user_query: str,
    route_result: RouteResult | None,
    messages: list,
    fund_scope: dict | None = None,
) -> dict[str, Any]:
    """生成新的任务计划

    Args:
        user_query: 用户当前问题
        route_result: 路由结果
        messages: 完整的对话历史（用于理解上下文）
        fund_scope: 基金范围
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
    scope_text = json.dumps(fund_scope, ensure_ascii=False) if fund_scope else "None"

    prompt = f"""请为以下问题生成执行计划：
{history_text}
当前用户问题：{user_query}

路由信息：
- 意图类型：{route_result.intent if route_result else "unknown"}

请输出 JSON 格式的任务规划。"""

    scope_context = _scope_planning_context(route_result, fund_scope)
    prompt += (
        f"\n\nSCOPE_PLANNING_CONTEXT:\n{scope_context}\n"
        f"\nCONFIRMED_FUND_SCOPE:\n{scope_text}\n"
    )
    explicit_fund_codes = _explicit_fund_codes(messages)
    validation_feedback = ""

    for attempt in range(MAX_PLAN_VALIDATION_RETRIES + 1):
        try:
            response = await llm_ainvoke(llm, [
                {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt + validation_feedback},
            ])
            plan_data = json.loads(extract_json_block(response.content))
            validated_plan = validate_supervisor_plan(
                plan_data,
                explicit_fund_codes=explicit_fund_codes,
                fund_scope=fund_scope,
                require_non_empty_plan=bool(
                    route_result
                    and route_result.intent in (
                        "single_fund_query",
                        "cross_fund_query",
                        "fund_screening",
                    )
                ),
            )
            logger.info(
                "[Supervisor] Generated and validated plan with %s tasks",
                len(validated_plan),
            )
            return _new_plan_update(validated_plan)
        except (json.JSONDecodeError, ValidationError, PlanValidationError, ValueError) as exc:
            logger.warning(
                "[Supervisor] Invalid plan on attempt %s/%s: %s",
                attempt + 1,
                MAX_PLAN_VALIDATION_RETRIES + 1,
                exc,
            )
            validation_feedback = (
                "\n\n上一版计划未通过硬校验，错误如下：\n"
                f"{exc}\n"
                "请重新输出完整 JSON。不得省略字段；不得使用不存在或循环依赖；"
                "task_type 与 assigned_agent 必须匹配；fund_codes 只能使用用户消息中"
                "明确出现的 6 位数字代码。"
            )
        except Exception as exc:
            logger.exception("[Supervisor] Plan generation failed on attempt %s", attempt + 1)
            validation_feedback = (
                "\n\n上一版计划生成失败，请重新仅输出符合要求的完整 JSON。"
            )

    error = "计划生成或校验连续失败，未提交任何任务执行。"
    logger.error("[Supervisor] %s", error)
    return _planning_failure_update(error)
