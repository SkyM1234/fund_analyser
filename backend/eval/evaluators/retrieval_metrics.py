"""检索类规则指标 evaluator。

LangSmith evaluator 约定：函数签名 (run, example) → dict / EvaluationResult
- run.outputs: target 返回值（这里是 {"results": [chunk_dict,...]}）
- example.outputs: 数据集 ground truth（RetrievalExample.dict）

参与 chunk 排名指标的每条 chunk_dict 应至少包含：
    - fund_code: str
    - content: str
    - id: Milvus chunk 主键

没有 relevant_chunk_ids Ground Truth 的样本不参与 hit_rate / MRR / NDCG 聚合。
"""
from __future__ import annotations

import math
from typing import Any

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

    仅使用 relevant_chunk_ids 做精确匹配，确保直接 RAG 与 Agentic RAG
    使用同一套 chunk 级 Ground Truth。
    """
    relevant_ids = set(truth.get("relevant_chunk_ids") or [])
    chunk_id = chunk.get("id") or f"{chunk.get('fund_code')}_{chunk.get('chunk_index')}"
    return chunk_id in relevant_ids


def _missing_chunk_truth(truth: dict) -> bool:
    """chunk 排名类指标要求精确到 chunk 级的 ground truth。"""
    return not truth.get("relevant_chunk_ids")


def _unique_chunks(chunks: list[dict]) -> list[dict]:
    """保持排名顺序，同时防止重复的 chunk ID 抬高指标。"""
    seen_ids: set[str] = set()
    unique = []
    for chunk in chunks:
        chunk_id = chunk.get("id")
        if not chunk_id or chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        unique.append(chunk)
    return unique


def hit_rate(run: Any, example: Any) -> dict:
    """top-K 命中率：至少一个相关 chunk 进入结果集。"""
    skip, reason = _should_skip_retrieval(run)
    if skip:
        return {"key": "hit_rate", "score": None, "comment": f"跳过检索评测({reason})"}

    results = _unique_chunks(_get_results(run))
    truth = _get_truth(example)
    if _missing_chunk_truth(truth):
        return {
            "key": "hit_rate",
            "score": None,
            "comment": "no relevant_chunk_ids ground truth",
        }
    hit = any(_chunk_relevance(c, truth) for c in results)
    return {"key": "hit_rate", "score": 1.0 if hit else 0.0}


def mrr(run: Any, example: Any) -> dict:
    """Mean Reciprocal Rank：首条相关结果排名的倒数。"""
    skip, reason = _should_skip_retrieval(run)
    if skip:
        return {"key": "mrr", "score": None, "comment": f"skip({reason})"}

    results = _unique_chunks(_get_results(run))
    truth = _get_truth(example)
    if _missing_chunk_truth(truth):
        return {
            "key": "mrr",
            "score": None,
            "comment": "no relevant_chunk_ids ground truth",
        }
    for idx, c in enumerate(results, start=1):
        if _chunk_relevance(c, truth):
            return {"key": "mrr", "score": 1.0 / idx}
    return {"key": "mrr", "score": 0.0}


def ndcg(run: Any, example: Any) -> dict:
    """NDCG@K：相关性 0/1 加权排序质量。"""
    skip, reason = _should_skip_retrieval(run)
    if skip:
        return {"key": "ndcg", "score": None, "comment": f"skip({reason})"}

    results = _unique_chunks(_get_results(run))
    truth = _get_truth(example)
    if _missing_chunk_truth(truth):
        return {
            "key": "ndcg",
            "score": None,
            "comment": "no relevant_chunk_ids ground truth",
        }
    if not results:
        return {"key": "ndcg", "score": 0.0}

    rels = [1.0 if _chunk_relevance(c, truth) else 0.0 for c in results]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

    ideal_count = len(set(truth.get("relevant_chunk_ids") or []))
    if ideal_count == 0:
        return {"key": "ndcg", "score": 0.0}
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return {"key": "ndcg", "score": dcg / idcg if idcg > 0 else 0.0}


def session_hit_rate(run: Any, example: Any) -> dict:
    """相关 chunk 在所有 ground-truth chunk ID 中的覆盖率。"""
    skip, reason = _should_skip_retrieval(run)
    if skip:
        return {"key": "hit_rate", "score": None, "comment": f"skip({reason})"}

    truth = _get_truth(example)
    if _missing_chunk_truth(truth):
        return {
            "key": "hit_rate",
            "score": None,
            "comment": "no relevant_chunk_ids ground truth",
        }

    relevant_ids = set(truth["relevant_chunk_ids"])
    returned_ids = {
        chunk.get("id") for chunk in _unique_chunks(_get_results(run))
    }
    hit_count = len(relevant_ids & returned_ids)
    total = len(relevant_ids)
    return {
        "key": "hit_rate",
        "score": hit_count / total if total else 0.0,
        "comment": f"relevant_chunk_coverage={hit_count}/{total}",
    }


def session_mrr(run: Any, example: Any) -> dict:
    """每个 ground-truth chunk 的平均倒数排名；未命中的记 0 分。"""
    skip, reason = _should_skip_retrieval(run)
    if skip:
        return {"key": "session_mrr", "score": None, "comment": f"skip({reason})"}

    truth = _get_truth(example)
    if _missing_chunk_truth(truth):
        return {
            "key": "session_mrr",
            "score": None,
            "comment": "no relevant_chunk_ids ground truth",
        }

    relevant_ids = set(truth["relevant_chunk_ids"])
    ranks = {
        chunk.get("id"): index
        for index, chunk in enumerate(_unique_chunks(_get_results(run)), start=1)
        if chunk.get("id") in relevant_ids
    }
    scores = [1.0 / ranks[chunk_id] if chunk_id in ranks else 0.0
              for chunk_id in relevant_ids]
    return {
        "key": "session_mrr",
        "score": sum(scores) / len(scores) if scores else 0.0,
        "comment": f"per_chunk_rank={ranks}",
    }


def session_ndcg(run: Any, example: Any) -> dict:
    """NDCG：IDCG 使用全部 ground-truth chunk，因此未命中会拉低分数。"""
    skip, reason = _should_skip_retrieval(run)
    if skip:
        return {"key": "session_ndcg", "score": None, "comment": f"skip({reason})"}

    truth = _get_truth(example)
    if _missing_chunk_truth(truth):
        return {
            "key": "session_ndcg",
            "score": None,
            "comment": "no relevant_chunk_ids ground truth",
        }

    relevant_ids = set(truth["relevant_chunk_ids"])
    results = _unique_chunks(_get_results(run))
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, chunk in enumerate(results, start=1)
        if chunk.get("id") in relevant_ids
    )
    idcg = sum(
        1.0 / math.log2(index + 1)
        for index in range(1, len(relevant_ids) + 1)
    )
    return {
        "key": "session_ndcg",
        "score": dcg / idcg if idcg else 0.0,
        "comment": f"relevant_chunk_count={len(relevant_ids)}",
    }


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
