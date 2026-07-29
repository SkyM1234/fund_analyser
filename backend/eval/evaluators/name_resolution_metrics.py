"""两级RAG第一级（rag_identify_funds）评测指标：名称→代码识别准确率。

与 retrieval_metrics.py 解耦：这里只关心"能否识别出正确的基金代码"，
不涉及向量库内容检索质量。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _get_identified_codes(run: Any) -> list[str]:
    outputs = getattr(run, "outputs", None) or {}
    return [c for c in (outputs.get("identified_codes") or []) if c]


def _get_expected_code(example: Any) -> str | None:
    outputs = getattr(example, "outputs", None) or {}
    return outputs.get("expected_fund_code")


def top1_accuracy(run, example) -> dict:
    """置信度最高的候选是否命中期望代码。

    对于期望不命中（expected_fund_code 为 None）的样本，只要没有返回任何
    候选即算通过；只要返回了任意候选就算误命中（score=0）。
    """
    identified = _get_identified_codes(run)
    expected = _get_expected_code(example)

    if expected is None:
        score = 1.0 if not identified else 0.0
        comment = "期望无匹配" + ("，实际也无匹配" if not identified else f"，但误识别为 {identified[0]}")
        return {"key": "top1_accuracy", "score": score, "comment": comment}

    if not identified:
        return {"key": "top1_accuracy", "score": 0.0, "comment": f"期望 {expected}，但未识别到任何候选"}

    score = 1.0 if identified[0] == expected else 0.0
    comment = f"期望 {expected}，实际 top1 为 {identified[0]}"
    return {"key": "top1_accuracy", "score": score, "comment": comment}


def hit_at_k(run, example) -> dict:
    """期望代码是否出现在返回的候选列表中（不要求排第一）。"""
    identified = _get_identified_codes(run)
    expected = _get_expected_code(example)

    if expected is None:
        score = 1.0 if not identified else 0.0
        return {"key": "hit_at_k", "score": score}

    score = 1.0 if expected in identified else 0.0
    comment = f"期望 {expected}，候选列表为 {identified}"
    return {"key": "hit_at_k", "score": score, "comment": comment}


def miss_rate(run, example) -> dict:
    """未命中率：期望有匹配但完全没有返回任何候选（语义检索"漏检"）。

    仅对 expected_fund_code 非 None 的样本有意义；无匹配预期的样本跳过。
    """
    expected = _get_expected_code(example)
    if expected is None:
        return {"key": "miss_rate", "score": None, "comment": "该样本期望无匹配，不计入未命中率"}

    identified = _get_identified_codes(run)
    score = 0.0 if identified else 1.0
    comment = "已返回候选" if identified else "未返回任何候选（漏检）"
    return {"key": "miss_rate", "score": score, "comment": comment}


def false_positive_rate(run, example) -> dict:
    """误识别率：期望无匹配，但模型返回了候选（语义检索"误检"）。

    仅对 expected_fund_code 为 None 的样本有意义；有匹配预期的样本跳过。
    """
    expected = _get_expected_code(example)
    if expected is not None:
        return {"key": "false_positive_rate", "score": None, "comment": "该样本期望有匹配，不计入误识别率"}

    identified = _get_identified_codes(run)
    score = 1.0 if identified else 0.0
    comment = f"误识别为 {identified}" if identified else "正确判定为无匹配"
    return {"key": "false_positive_rate", "score": score, "comment": comment}
