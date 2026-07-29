"""检索类规则指标 evaluator。

LangSmith evaluator 约定：函数签名 (run, example) → dict / EvaluationResult
- run.outputs: target 返回值（这里是 {"results": [chunk_dict,...]}）
- example.outputs: 数据集 ground truth（RetrievalExample.dict）

每条 chunk_dict 应至少包含：
    - fund_code: str
    - content: str
    - id: 可选；若 schema 没暴露 id，可用 (fund_code, chunk_index) 拼出
"""
from __future__ import annotations

import math
import re
from typing import Any

from sympy import N



def _get_results(run: Any) -> list[dict]:
    outputs = getattr(run, "outputs", None) or {}
    # 兼容两种格式：rag_target 用 "results"，agent_target 用 "retrieved_chunks"
    return outputs.get("results") or outputs.get("retrieved_chunks") or []


def _get_truth(example: Any) -> dict:
    """从 example.outputs 取标注信息（LangSmith 约定 ground truth 在 outputs）。"""
    outputs = getattr(example, "outputs", None) or {}
    return {
        "expected_fund_codes": outputs.get("expected_fund_codes") or [],
        "relevant_chunk_ids": outputs.get("relevant_chunk_ids") or [],
        "relevant_keywords": outputs.get("relevant_keywords") or [],
        "filter_fund_code": outputs.get("filter_fund_code") or "",
    }


def _should_skip_retrieval(run: Any) -> tuple[bool, str]:
    """判断该样本是否应该跳过检索（返回 (True, reason) 或 (False, "")）。

    跳过场景：
    1. intent 为 general_finance / chitchat / sensitive / out_of_scope（不需要 RAG）
    2. retrieved_chunks 为空 + intent 满足上述条件
    """
    outputs = getattr(run, "outputs", None) or {}
    intent = outputs.get("intent") or ""
    results = outputs.get("results") or outputs.get("retrieved_chunks") or []

    # 不需要 RAG 的意图类型
    NO_RAG_INTENTS = {"general_finance", "chitchat", "sensitive", "out_of_scope"}

    if intent in NO_RAG_INTENTS and not results:
        return True, f"intent={intent}, 不需要检索"

    return False, ""


def _chunk_relevance(chunk: dict, truth: dict) -> bool:
    """判断单个检索 chunk 是否相关。

    优先用 relevant_chunk_ids 精确匹配；其次用 expected_fund_codes；
    最后兜底用 relevant_keywords 子串匹配。
    """
    relevant_ids = set(truth.get("relevant_chunk_ids") or [])

    # 当ground_truth没有，则直接返回True
    if not relevant_ids:
        return True
    
    if relevant_ids:
        chunk_id = chunk.get("id") or f"{chunk.get('fund_code')}_{chunk.get('chunk_index')}"
        if chunk_id in relevant_ids:
            return True
    

    # expected_codes = set(truth.get("expected_fund_codes") or [])
    # if expected_codes:
    #     if chunk.get("fund_code") in expected_codes:
    #         # 基金匹配只算"基金级"相关；若同时有 keywords，进一步要求 keyword 命中
    #         keywords = truth.get("relevant_keywords") or []
    #         if not keywords:
    #             return True
    #         content = chunk.get("content", "") or ""
    #         return any(kw in content for kw in keywords)

    # # 完全没有 expected_codes 时（全局/筛选），仅靠 keywords
    # keywords = truth.get("relevant_keywords") or []
    # if keywords:
    #     content = chunk.get("content", "") or ""
    #     return any(kw in content for kw in keywords)

    return False


def hit_rate(run: Any, example: Any) -> dict:
    """top-K 命中率：至少一个相关 chunk 进入结果集。"""
    skip, reason = _should_skip_retrieval(run)
    if skip:
        return {"key": "hit_rate", "score": None, "comment": f"跳过检索评测({reason})"}

    results = _get_results(run)
    truth = _get_truth(example)
    hit = any(_chunk_relevance(c, truth) for c in results)
    return {"key": "hit_rate", "score": 1.0 if hit else 0.0}


def mrr(run: Any, example: Any) -> dict:
    """Mean Reciprocal Rank：首条相关结果排名的倒数。"""
    skip, reason = _should_skip_retrieval(run)
    if skip:
        return {"key": "mrr", "score": None, "comment": f"skip({reason})"}

    results = _get_results(run)
    truth = _get_truth(example)
    for idx, c in enumerate(results, start=1):
        if _chunk_relevance(c, truth):
            return {"key": "mrr", "score": 1.0 / idx}
    return {"key": "mrr", "score": 0.0}


def ndcg(run: Any, example: Any) -> dict:
    """NDCG@K：相关性 0/1 加权排序质量。"""
    skip, reason = _should_skip_retrieval(run)
    if skip:
        return {"key": "ndcg", "score": None, "comment": f"skip({reason})"}

    results = _get_results(run)
    truth = _get_truth(example)
    if not results:
        return {"key": "ndcg", "score": 0.0}

    rels = [1.0 if _chunk_relevance(c, truth) else 0.0 for c in results]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

    ideal_count = int(sum(rels))
    if ideal_count == 0:
        return {"key": "ndcg", "score": 0.0}
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return {"key": "ndcg", "score": dcg / idcg if idcg > 0 else 0.0}


def fund_code_recall(run: Any, example: Any) -> dict:
    """目标基金代码召回率：expected_fund_codes 中有多少在结果里出现。

    针对 filter_fund_code 场景，能直接抓住"过滤失效"的问题。
    """
    skip, reason = _should_skip_retrieval(run)
    if skip:
        return {"key": "fund_code_recall", "score": None, "comment": f"skip({reason})"}

    truth = _get_truth(example)
    expected = set(truth.get("expected_fund_codes") or [])
    if not expected:
        # 没有期望基金时，不参与统计
        return {"key": "fund_code_recall", "score": None, "comment": "no expected codes"}
    
    filter_fund_code = truth.get("filter_fund_code") or ""
    if filter_fund_code:
        # 过滤基金代码存在时，不参与统计
        return {"key": "fund_code_recall", "score": None, "comment": "have filter_fund_code"}

    results = _get_results(run)
    found = {c.get("fund_code") for c in results if c.get("fund_code")}
    recall = len(expected & found) / len(expected)
    return {
        "key": "fund_code_recall",
        "score": recall,
        "comment": f"expected={sorted(expected)} found={sorted(found & expected)}",
    }
