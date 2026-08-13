"""通用检索 Agent —— rag_agent 与 market_agent 的参数化实现。

用法：
    from app.agent.retrieval_agent import make_retrieval_node, AgentConfig
    node_fn = make_retrieval_node(AgentConfig(...))
"""
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.llm_concurrency import llm_ainvoke
from app.agent.multi_agent_state import MultiAgentState
from app.tools.conversation_utils import get_recent_messages_for_agent
from app.tools.token_usage import record_usage
from app.agent.reflection_agent import agent_self_check, _improve_query, MAX_REFLECTION_RETRIES
from app.services.rag_result_parser import parse_rag_search_sections

logger = logging.getLogger(__name__)
FUND_CODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")


@dataclass
class AgentConfig:
    """参数化 retrieval agent 的配置。"""
    agent_name: str          # 与 SubTask.assigned_agent 保持一致，用于日志和 token bucket
    system_prompt: str       # 完整 system prompt 模板（含 {tool_descriptions} 占位符）
    max_iterations: int = 5  # ReAct 循环最大轮次


@dataclass
class _RagToolContext:
    """A RAG ToolMessage and its original MCP response."""

    message: ToolMessage
    output: str


def _build_tool_descriptions(tools: list) -> str:
    lines = []
    for t in tools:
        desc = (t.description or "").replace("\n", " ").strip()
        lines.append(f"- `{t.name}`: {desc}")
    return "\n".join(lines)


def _get_tools_for_agent(agent_name: str) -> list:
    """按 MCP_AGENT_TOOL_SERVERS 配置过滤工具；若无匹配则回退到全部工具。"""
    settings = get_settings()
    if not settings.MCP_ENABLED:
        return []
    try:
        from app.services.mcp_client import get_cached_mcp_tools
        all_tools = get_cached_mcp_tools()
    except Exception as e:
        logger.error(f"[{agent_name}] Failed to load cached tools: {e}")
        return []

    allowed_servers: list[str] | None = settings.MCP_AGENT_TOOL_SERVERS.get(agent_name)
    if allowed_servers is None:
        # 未配置映射时使用全量工具（向后兼容）
        logger.warning(f"[{agent_name}] No server mapping configured, using all tools")
        return all_tools

    filtered = [
        t for t in all_tools
        if (t.metadata or {}).get("server") in allowed_servers
        and t.name not in settings.MCP_EXCLUDED_TOOLS
    ]
    logger.info(
        f"[{agent_name}] Tool filter: allowed_servers={allowed_servers}, "
        f"matched {len(filtered)}/{len(all_tools)} tools"
    )
    return filtered


def _mark_finished(task: dict) -> None:
    finished_at = time.monotonic()
    task["finished_at"] = finished_at
    started_at = task.get("started_at")
    if started_at is not None:
        task["duration_ms"] = (finished_at - started_at) * 1000


def _latest_user_fund_codes(messages: list) -> set[str]:
    """提取当前用户消息中明确给出的6位基金代码。"""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return set(FUND_CODE_RE.findall(str(message.content)))
    return set()


def _normalize_filter_fund_codes(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if FUND_CODE_RE.fullmatch(value) else set()
    if isinstance(value, list):
        return {
            code
            for code in value
            if isinstance(code, str) and FUND_CODE_RE.fullmatch(code)
        }
    return set()


def _tool_output_text(output: Any) -> str:
    return output if isinstance(output, str) else str(output)


async def _execute_tool_calls(
    *,
    response,
    tools_by_name: dict[str, Any],
    agent_name: str,
    task_id: str,
    allowed_fund_codes: set[str],
    tool_call_log: list[dict],
    rag_tool_contexts: list[_RagToolContext],
) -> list[ToolMessage]:
    """执行工具调用，并校验 RAG 基金代码的来源。"""
    valid_calls: list[dict] = []
    tool_messages: list[ToolMessage] = []

    for index, tool_call in enumerate(response.tool_calls):
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args") or {}
        tool_call_id = tool_call.get("id") or f"call_{index}"

        if (
            agent_name == "rag_agent"
            and tool_name == "rag_search"
            and tool_args.get("filter_fund_code") not in (None, "", [])
        ):
            requested_codes = _normalize_filter_fund_codes(
                tool_args.get("filter_fund_code")
            )
            if not requested_codes or not requested_codes.issubset(allowed_fund_codes):
                logger.warning(
                    "[%s] 已拦截使用未确认基金代码的 rag_search：%s；允许代码=%s",
                    agent_name,
                    tool_args.get("filter_fund_code"),
                    sorted(allowed_fund_codes),
                )
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        content=(
                            "基金代码未确认，未执行 rag_search。filter_fund_code 只能来自"
                            "用户原始问题中明确给出的6位代码，或本任务 rag_identify_funds 的返回结果。"
                        ),
                        status="error",
                    )
                )
                continue

        if tool_name not in tools_by_name:
            tool_messages.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    content=f"工具不可用：{tool_name}",
                    status="error",
                )
            )
            continue

        valid_calls.append(
            {"name": tool_name, "args": tool_args, "id": tool_call_id}
        )
        tool_call_log.append(
            {
                "agent": agent_name,
                "task_id": task_id,
                "name": tool_name,
                "args": tool_args,
            }
        )

    async def invoke(tool_call: dict) -> ToolMessage:
        try:
            output = await tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
            content = _tool_output_text(output)
            if agent_name == "rag_agent" and tool_call["name"] == "rag_identify_funds":
                allowed_fund_codes.update(FUND_CODE_RE.findall(content))
            return ToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                content=content,
            )
        except Exception as exc:
            logger.exception("[%s] 工具 %s 调用失败", agent_name, tool_call["name"])
            return ToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                content=f"工具调用失败：{exc}",
                status="error",
            )

    if valid_calls:
        tool_messages.extend(await asyncio.gather(*(invoke(call) for call in valid_calls)))
        if agent_name == "rag_agent":
            for message in tool_messages:
                if message.name == "rag_search" and message.status != "error":
                    rag_tool_contexts.append(
                        _RagToolContext(message=message, output=str(message.content))
                    )
            _deduplicate_rag_tool_context(rag_tool_contexts)
    return tool_messages


def _deduplicate_rag_tool_context(contexts: list[_RagToolContext]) -> None:
    """Keep one copy of each retrieved chunk in the agent prompt context.

    A chunk keeps the position of its first occurrence. A later result with a
    higher score replaces the text at that original position without reordering
    the context.
    """
    occurrences = []
    for context_index, context in enumerate(contexts):
        for section_index, section in enumerate(parse_rag_search_sections(context.output)):
            occurrences.append((context_index, section_index, section))

    first_occurrence: dict[str, tuple[int, int]] = {}
    best_sections = {}
    for context_index, section_index, section in occurrences:
        first_occurrence.setdefault(section.chunk_id, (context_index, section_index))
        current_best = best_sections.get(section.chunk_id)
        if (
            current_best is None
            or (
                section.score is not None
                and (current_best.score is None or section.score > current_best.score)
            )
        ):
            best_sections[section.chunk_id] = section

    for context_index, context in enumerate(contexts):
        sections = parse_rag_search_sections(context.output)
        if not sections:
            continue

        kept_sections = [
            best_sections[section.chunk_id].text
            for section_index, section in enumerate(sections)
            if first_occurrence[section.chunk_id] == (context_index, section_index)
        ]
        context.message.content = (
            "\n".join(kept_sections)
            if kept_sections
            else "Duplicate RAG chunks omitted from context."
        )


def _build_task_message(current_task: dict, agent_name: str, query: str | None = None) -> str:
    task_query = query if query is not None else current_task.get("query", "")
    fund_codes = current_task.get("fund_codes", [])
    if fund_codes:
        fund_code = fund_codes[0]
        return (
            f"任务：{task_query}\n"
            f"只检索/查询基金 {fund_code}，不要查询其他基金。\n"
            f"返回检索到的原始数据，不要与其他基金对比。"
        )
    # fund_codes 为空时，注入强制识别提醒（与各 agent 的 system prompt 呼应）
    # description 通常比 query 更完整（如"查询万家科创债ETF的当前净值"），直接带给 Agent 作为基金线索
    description = current_task.get("description", "")
    task_hint = f"{task_query}（{description}）" if description else task_query
    _identify_tool = {"rag_agent": "rag_identify_funds", "market_agent": "search_fund"}.get(agent_name)
    _search_hint = {
        "rag_agent": "rag_search",
        "market_agent": "对应的数据查询工具（如 get_fund_estimate / get_fund_info 等，按任务需要选择）",
    }.get(agent_name)
    if _identify_tool and _search_hint:
        return (
            f"任务：{task_hint}\n"
            f"⚠️ 此任务未提供基金代码。若任务涉及特定基金（非全局检索），"
            f"你必须先调用 {_identify_tool} 查询基金代码，确认后再调用 {_search_hint}。\n"
            f"返回查询到的原始数据即可。"
        )
    return f"任务：{task_hint}\n返回查询到的原始数据即可。"


def make_retrieval_node(config: AgentConfig):
    """工厂函数：返回绑定到特定 AgentConfig 的 LangGraph 节点函数。"""

    async def _node(state: MultiAgentState, run_config: RunnableConfig | None = None) -> dict[str, Any]:
        label = config.agent_name
        current_task_id = state.get("current_task_id")
        if not current_task_id:
            logger.error(f"[{label}] No current_task_id")
            return {}

        current_task = next(
            (t for t in state.get("plan", []) if t["task_id"] == current_task_id),
            None,
        )
        if not current_task:
            logger.error(f"[{label}] Task {current_task_id} not found in plan")
            return {}

        logger.info(f"[{label}] Executing task: {current_task['description']}")

        tools = _get_tools_for_agent(label)
        if not tools:
            return {"sub_results": {current_task_id: f"错误：{label} 没有可用工具"}}

        system_prompt = config.system_prompt.format(
            tool_descriptions=_build_tool_descriptions(tools)
        )

        settings = get_settings()
        llm = ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            temperature=0.3,
        )
        llm_with_tools = llm.bind_tools(tools)
        tools_by_name = {tool.name: tool for tool in tools}

        history_messages = get_recent_messages_for_agent(
            state.get("messages", []), rounds=2, exclude_current_human=True
        )

        token_usage: dict[str, dict[str, int]] = {}
        tool_call_log: list[dict] = []
        rag_tool_contexts: list[_RagToolContext] = []
        task_query = current_task.get("query", "")
        allowed_fund_codes = _latest_user_fund_codes(state.get("messages", []))
        retry_count = 0

        try:
            while True:
                rag_tool_contexts.clear()
                messages = [SystemMessage(content=system_prompt)] + history_messages
                messages.append(HumanMessage(content=_build_task_message(current_task, config.agent_name, task_query)))

                iteration = 0
                while iteration < config.max_iterations:
                    iteration += 1
                    logger.info(f"[{label}] Iteration {iteration}")

                    response = await llm_ainvoke(llm_with_tools, messages)
                    messages.append(response)
                    iter_usage = record_usage(f"{label}:{current_task_id}", response)
                    for bucket, usage in iter_usage.items():
                        existing = token_usage.setdefault(bucket, {})
                        for field, value in usage.items():
                            existing[field] = existing.get(field, 0) + value

                    if not response.tool_calls:
                        result_text = response.content

                        result_text, self_check_usage, recheck_query = await agent_self_check(
                            current_task, result_text, label
                        )
                        for k, v in self_check_usage.items():
                            existing = token_usage.setdefault(k, {})
                            for field, val in v.items():
                                existing[field] = existing.get(field, 0) + val

                        # 自检发现事实矛盾 → 追加反馈消息并回到工具调用循环
                        if recheck_query and iteration < config.max_iterations:
                            logger.info(f"[{label}] Self-check triggered recheck, iter={iteration}/{config.max_iterations}")
                            messages.append(HumanMessage(
                                content=(
                                    f"自检发现以下数据矛盾需要重新查证：{recheck_query}\n"
                                    f"请调用相关工具获取准确数据，然后基于查证结果修正输出。"
                                    f"不要重复之前的所有查询，只查证上述矛盾涉及的具体数据点。"
                                )
                            ))
                            continue  # 回到 while 循环，LLM 可调用工具查证

                        logger.info(f"[{label}] Task completed: {result_text[:100]}...")

                        plan = state.get("plan", [])
                        for task in plan:
                            if task["task_id"] == current_task_id:
                                task["status"] = "completed"
                                task["result"] = result_text
                                task["retry_count"] = retry_count
                                _mark_finished(task)
                                break

                        sub_results = state.get("sub_results", {}).copy()
                        sub_results[current_task_id] = result_text

                        completed_tasks = state.get("completed_tasks", [])[:]
                        if current_task_id not in completed_tasks:
                            completed_tasks.append(current_task_id)

                        return {
                            "plan": plan,
                            "sub_results": sub_results,
                            "completed_tasks": completed_tasks,
                            "token_usage": token_usage,
                            "tool_call_log": tool_call_log,
                        }

                    logger.info(f"[{label}] Executing {len(response.tool_calls)} tool calls")
                    tool_messages = await _execute_tool_calls(
                        response=response,
                        tools_by_name=tools_by_name,
                        agent_name=label,
                        task_id=current_task_id,
                        allowed_fund_codes=allowed_fund_codes,
                        tool_call_log=tool_call_log,
                        rag_tool_contexts=rag_tool_contexts,
                    )
                    messages.extend(tool_messages)

                # 超出最大迭代：重试预算未用完时，改写 query 原地重跑一轮；用完预算才真正判定为 failed。
                # 同一分支内部完成，不涉及跨节点/跨并行分支的状态传递，避免 Send fan-out 场景下 reflection 类下游节点无法可靠定位"这次是哪个 task_id"的问题。
                logger.warning(f"[{label}] Max iterations reached (retry {retry_count}/{MAX_REFLECTION_RETRIES})")
                if retry_count >= MAX_REFLECTION_RETRIES:
                    break

                retry_count += 1

                # 通知 SSE 流式层：即将触发子Agent内部重试
                retry_cb = (run_config.get("configurable", {}) if run_config else {}).get("_sse_retry_callback")
                if retry_cb:
                    try:
                        await retry_cb(label, current_task_id, retry_count,
                                       "达到最大迭代次数，未能获取充分信息")
                    except Exception:
                        logger.exception(f"[{label}] retry callback failed (non-fatal)")

                task_query, improve_usage = await _improve_query(current_task, "达到最大迭代次数，未能获取充分信息")
                for k, v in improve_usage.items():
                    existing = token_usage.setdefault(k, {})
                    for field, val in v.items():
                        existing[field] = existing.get(field, 0) + val
                logger.info(f"[{label}] Retrying task {current_task_id} with improved query: {task_query}")

            last_ai_message = next(
                (
                    msg.content
                    for msg in reversed(messages)
                    if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls")
                ),
                None,
            )
            partial_result = last_ai_message or "任务超时：达到最大迭代次数，未能获取充分信息"

            plan = state.get("plan", [])
            for task in plan:
                if task["task_id"] == current_task_id:
                    task["status"] = "failed"
                    task["error"] = "达到最大迭代次数"
                    task["result"] = partial_result
                    task["retry_count"] = retry_count
                    _mark_finished(task)
                    break

            sub_results = state.get("sub_results", {}).copy()
            sub_results[current_task_id] = partial_result

            failed_tasks = state.get("failed_tasks", [])[:]
            if current_task_id not in failed_tasks:
                failed_tasks.append(current_task_id)

            return {
                "plan": plan,
                "sub_results": sub_results,
                "failed_tasks": failed_tasks,
                "token_usage": token_usage,
                "tool_call_log": tool_call_log,
            }

        except Exception as e:
            logger.error(f"[{label}] Execution failed: {e}", exc_info=True)

            plan = state.get("plan", [])
            for task in plan:
                if task["task_id"] == current_task_id:
                    task["status"] = "failed"
                    task["error"] = str(e)
                    _mark_finished(task)
                    break

            sub_results = state.get("sub_results", {}).copy()
            sub_results[current_task_id] = f"执行失败：{str(e)}"

            failed_tasks = state.get("failed_tasks", [])[:]
            if current_task_id not in failed_tasks:
                failed_tasks.append(current_task_id)

            return {
                "plan": plan,
                "sub_results": sub_results,
                "failed_tasks": failed_tasks,
                "token_usage": token_usage,
                "tool_call_log": tool_call_log,
            }

    _node.__name__ = f"{config.agent_name}_node"
    return _node
