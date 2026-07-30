"""LLM-as-Judge 评测指标。

统一返回 0.0~1.0 浮点分数：
- context_relevance: 检索片段对问题的相关性（无需 ground truth）
- correctness:       回答与参考答案的语义一致性
- relevance:         回答是否切题

设计要点：
- 严格 JSON 输出，避免解析失败
- temperature=0，确定性
- 与业务 LLM 不强绑定，judge 模型可单独配置
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from eval.config import get_eval_settings

logger = logging.getLogger(__name__)

_judge_llm: ChatOpenAI | None = None


def _get_judge() -> ChatOpenAI:
    global _judge_llm
    if _judge_llm is None:
        s = get_eval_settings()
        _judge_llm = ChatOpenAI(
            base_url=s.JUDGE_LLM_BASE_URL,
            api_key=s.require_judge_api_key(),
            model=s.JUDGE_LLM_MODEL,
            temperature=s.JUDGE_LLM_TEMPERATURE,
        )
    return _judge_llm


def _parse_json(content: str) -> dict:
    """容错解析 LLM JSON 输出。"""
    content = (content or "").strip()
    # 去 fence
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except Exception:
        # 兜底：抓第一个 {...}
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def _format_chunks(results: list[dict], max_chars: int = 4000) -> str:
    """把检索结果拼成 judge 的上下文，控制长度避免超限。"""
    parts = []
    used = 0
    for i, r in enumerate(results, 1):
        content = (r.get("content") or "").strip()
        snippet = f"[{i}] (fund={r.get('fund_code','?')}) {content}\n"
        if used + len(snippet) > max_chars:
            break
        parts.append(snippet)
        used += len(snippet)
    return "".join(parts) if parts else "(无检索结果)"


# ============ 4 个 LLM judge ============

_CONTEXT_RELEVANCE_PROMPT = """你是检索质量评审员。判断"检索结果"整体上对"用户问题"的语义相关程度。

用户问题：
{query}

检索结果：
{context}

打分标准（0~1）：
- 1.0: 检索结果中多数片段与问题高度相关
- 0.5: 部分相关，多数无关或冗余
- 0.0: 几乎无相关内容

只输出 JSON：{{"score": <0~1 浮点>, "reason": "<不超过50字>"}}"""


def context_relevance(run: Any, example: Any) -> dict:
    outputs = getattr(run, "outputs", None) or {}
    inputs = getattr(example, "inputs", None) or {}
    query = inputs.get("query") or ""

    # 兼容两种格式：rag_target 用 "results"，agent_target 用 "retrieved_chunks"
    results = outputs.get("results") or outputs.get("retrieved_chunks") or []

    # 检查是否应该跳过检索评测（不需要 RAG 的场景）
    if not results:
        intent = outputs.get("intent") or ""
        NO_RAG_INTENTS = {"general_finance", "chitchat", "sensitive", "out_of_scope"}
        if intent in NO_RAG_INTENTS:
            return {
                "key": "context_relevance",
                "score": 1.0,
                "comment": f"跳过检索评测(intent={intent}, 不需要检索)",
            }
        # 如果需要 RAG 但无检索结果，则为 0.0
        return {
            "key": "context_relevance",
            "score": 0.0,
            "comment": "无检索结果",
        }

    prompt = _CONTEXT_RELEVANCE_PROMPT.format(query=query, context=_format_chunks(results))
    resp = _get_judge().invoke(prompt)
    data = _parse_json(resp.content)
    return {
        "key": "context_relevance",
        "score": float(data.get("score", 0.0)),
        "comment": data.get("reason", ""),
    }


_CORRECTNESS_PROMPT = """你是金融问答质量评审员。判断"模型回答"与"参考答案"在事实和结论上是否一致。

用户问题：
{query}

参考答案：
{reference}

模型回答：
{answer}

打分标准（0~1）：
- 1.0: 关键事实/结论完全一致
- 0.7: 主要点一致，存在小遗漏
- 0.3: 部分一致或方向偏差
- 0.0: 错误或不相关

只输出 JSON：{{"score": <0~1 浮点>, "reason": "<不超过60字>"}}"""


def correctness(run: Any, example: Any) -> dict:
    outputs = getattr(run, "outputs", None) or {}
    inputs = getattr(example, "inputs", None) or {}
    truth = getattr(example, "outputs", None) or {}
    reference = truth.get("reference_answer") or ""
    if not reference:
        return {"key": "correctness", "score": 1.0, "comment": "no reference"}

    prompt = _CORRECTNESS_PROMPT.format(
        query=inputs.get("query") or "",
        reference=reference,
        answer=outputs.get("answer") or "",
    )
    resp = _get_judge().invoke(prompt)
    data = _parse_json(resp.content)
    return {
        "key": "correctness",
        "score": float(data.get("score", 0.0)),
        "comment": data.get("reason", ""),
    }


_RELEVANCE_PROMPT = """你是问答相关性评审员。判断"模型回答"是否切题回应了"用户问题"，不评价正确性。

用户问题：
{query}

模型回答：
{answer}

打分标准（0~1）：
- 1.0: 直接回应问题
- 0.5: 部分相关或答非所重点
- 0.0: 答非所问

只输出 JSON：{{"score": <0~1 浮点>, "reason": "<不超过50字>"}}"""


def answer_relevance(run: Any, example: Any) -> dict:
    outputs = getattr(run, "outputs", None) or {}
    inputs = getattr(example, "inputs", None) or {}
    answer = outputs.get("answer") or ""

    # 检查是否是敏感问题且模型正确拒绝了（从 example.outputs 读 ground truth）
    example_outputs = getattr(example, "outputs", None) or {}
    should_refuse = example_outputs.get("should_refuse", False)

    # 简单启发式判断"拒绝"行为（检测拒绝关键词）
    refuse_keywords = ["无法", "不能", "抱歉", "不提供", "不建议", "不适合", "合规", "风险提示"]
    did_refuse = any(kw in answer for kw in refuse_keywords)

    # 如果期望拒绝且确实拒绝了，直接满分
    if should_refuse and did_refuse:
        return {
            "key": "answer_relevance",
            "score": 1.0,
            "comment": "正确拒绝敏感问题",
        }

    # 否则正常用 LLM-judge 评测
    prompt = _RELEVANCE_PROMPT.format(
        query=inputs.get("query") or "",
        answer=answer,
    )
    resp = _get_judge().invoke(prompt)
    data = _parse_json(resp.content)
    return {
        "key": "answer_relevance",
        "score": float(data.get("score", 0.0)),
        "comment": data.get("reason", ""),
    }
