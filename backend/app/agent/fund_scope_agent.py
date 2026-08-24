"""基金范围确认 Agent。

该节点只负责确认当前问题涉及的基金集合，不负责检索年报或市场数据。
"""
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.multi_agent_state import MultiAgentState
from app.core.deepseek_llm import create_chat_llm
from app.core.llm_concurrency import llm_ainvoke
from app.tools.conversation_utils import format_history_for_prompt
from app.tools.llm_json import extract_json_block

logger = logging.getLogger(__name__)
FUND_CODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")
MAX_FUND_SCOPE_TOOL_ROUNDS = 4
MAX_FUND_SCOPE_JSON_RETRIES = 2


class FundTarget(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    fund_name: str = ""
    fund_code: str = Field(pattern=r"^\d{6}$")
    confidence: float | None = None


class FundScope(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    query: str = ""
    funds: list[FundTarget] = Field(default_factory=list)
    total_count: int = 0
    coverage_status: str = "candidate"
    missing_or_uncertain: list[str] = Field(default_factory=list)


FUND_SCOPE_SYSTEM_PROMPT = """你是基金范围确认 Agent。你的唯一职责是确认用户问题涉及哪些基金。

工作规则：
1. 明确点名的基金必须逐只确认。
2. 用户给出 6 位代码时可直接使用，但不得修改代码；可使用 rag_match_fund_codes 确认基金是否存在或者基金代码对应的基金名称。
3. 用户给出的基金名称时，调用 rag_identify_funds 工具确认基金是否存在或者基金名称对应的基金代码。
4. 有关基金板块、主题问题，调用 rag_identify_funds 查询候选基金，必要时提高 top_k，因为有关基金的数量可能会大于 top_k。
5. 只能把工具返回或用户明确给出的基金写入 funds。无法确认的名称写入 missing_or_uncertain。
6. 最终只输出 JSON，不要 Markdown。原始问题由系统节点自动写入状态。
7. funds 输出 requested_name、fund_name、fund_code 和 confidence。

JSON 输出示例：
{
  "funds": [
    {"requested_name": "用户需要的基金名称", "fund_name": "工具确认的基金名称", "fund_code": "159xxx", "confidence": 0.95}
  ],
  "total_count": 4,
  "coverage_status": "confirmed|candidate|incomplete",
  "missing_or_uncertain": []
}
"""


def _get_scope_tools() -> list:
    from app.agent.retrieval_agent import _get_tools_for_agent

    return [
        tool
        for tool in _get_tools_for_agent("rag_agent")
        if tool.name in {"rag_identify_funds", "rag_match_fund_codes"}
    ]


def _latest_user_query(state: MultiAgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _codes_from_text(text: str) -> set[str]:
    return set(FUND_CODE_RE.findall(text))


def _normalize_scope(raw: object, query: str, allowed_codes: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("基金范围结果必须是 JSON 对象")
    scope = FundScope.model_validate({**raw, "query": query})
    seen: set[str] = set()
    funds = []
    for fund in scope.funds:
        if fund.fund_code not in allowed_codes:
            raise ValueError(f"范围结果包含未确认基金代码: {fund.fund_code}")
        if fund.fund_code in seen:
            continue
        seen.add(fund.fund_code)
        funds.append(fund.model_dump())
    scope.funds = [FundTarget.model_validate(fund) for fund in funds]
    scope.total_count = len(funds)
    if scope.coverage_status == "confirmed" and scope.missing_or_uncertain:
        scope.coverage_status = "incomplete"
    return scope.model_dump()


def _parse_scope_response(
    content: object,
    query: str,
    allowed_codes: set[str],
) -> dict[str, Any]:
    """Parse and validate the final scope JSON returned by the model."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("基金范围确认未返回 JSON 内容")
    raw = json.loads(extract_json_block(content))
    return _normalize_scope(raw, query, allowed_codes)


async def fund_scope_node(state: MultiAgentState) -> dict[str, Any]:
    query = _latest_user_query(state)
    if not query:
        return {"fund_scope": None, "fund_scope_error": "未找到用户问题"}

    tools = _get_scope_tools()
    if not tools:
        return {
            "fund_scope": None,
            "fund_scope_error": "基金范围确认工具不可用",
        }

    llm_base = create_chat_llm(temperature=0)
    llm = llm_base.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}
    history_text = format_history_for_prompt(
        state.get("messages", []),
        rounds=3,
        max_response_length=500,
        exclude_last=True,
    )
    scope_prompt = FUND_SCOPE_SYSTEM_PROMPT
    if history_text:
        scope_prompt += (
            "\n\n以下是最近几轮对话历史，仅用于理解当前问题中的省略、指代和基金范围。"
            "当前用户问题优先，不要把历史中未被当前问题引用的基金擅自加入范围：\n"
            f"{history_text}"
        )
    messages = [
        SystemMessage(content=scope_prompt),
        HumanMessage(content=query),
    ]
    allowed_codes = _codes_from_text(query)
    tool_log: list[dict] = []

    try:
        for _ in range(MAX_FUND_SCOPE_TOOL_ROUNDS):
            response = await llm_ainvoke(llm, messages)
            messages.append(response)
            if not response.tool_calls:
                try:
                    scope = _parse_scope_response(
                        response.content,
                        query,
                        allowed_codes,
                    )
                    return {
                        "fund_scope": scope,
                        "fund_scope_error": None,
                        "tool_call_log": tool_log,
                    }
                except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                    logger.warning(
                        "[FundScope] Invalid final JSON: %s; content=%r",
                        exc,
                        str(response.content)[:500],
                    )
                    break

            for index, call in enumerate(response.tool_calls):
                name = call.get("name", "")
                args = call.get("args") or {}
                call_id = call.get("id") or f"scope_call_{index}"
                tool_log.append({"agent": "fund_scope_agent", "name": name, "args": args})
                if name not in tools_by_name:
                    content = f"工具不可用: {name}"
                else:
                    result = await tools_by_name[name].ainvoke(args)
                    content = result if isinstance(result, str) else str(result)
                    allowed_codes.update(_codes_from_text(content))
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(
                    tool_call_id=call_id,
                    name=name,
                    content=content,
                ))

        for retry in range(MAX_FUND_SCOPE_JSON_RETRIES):
            response = await llm_ainvoke(llm_base, messages + [
                HumanMessage(content=(
                    "请根据已完成的工具调用结果，立即输出基金范围确认结果。"
                    "不要调用工具，不要解释，不要输出 Markdown；必须只输出符合系统提示词的 JSON 对象。"
                )),
            ])
            try:
                scope = _parse_scope_response(
                    response.content,
                    query,
                    allowed_codes,
                )
                logger.info(
                    "[FundScope] Recovered final JSON on retry %s/%s",
                    retry + 1,
                    MAX_FUND_SCOPE_JSON_RETRIES,
                )
                return {
                    "fund_scope": scope,
                    "fund_scope_error": None,
                    "tool_call_log": tool_log,
                }
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                logger.warning(
                    "[FundScope] Invalid JSON retry %s/%s: %s; content=%r",
                    retry + 1,
                    MAX_FUND_SCOPE_JSON_RETRIES,
                    exc,
                    str(response.content)[:500],
                )

        raise ValueError("基金范围确认未能输出有效 JSON")
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        logger.warning("[FundScope] Invalid scope result: %s", exc)
        return {
            "fund_scope": None,
            "fund_scope_error": str(exc),
            "tool_call_log": tool_log,
        }
    except Exception as exc:
        logger.exception("[FundScope] Scope confirmation failed")
        return {
            "fund_scope": None,
            "fund_scope_error": f"基金范围确认失败: {exc}",
            "tool_call_log": tool_log,
        }
