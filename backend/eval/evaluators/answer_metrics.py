"""回答类规则指标：引用准确率 / 拒绝准确率 / 关键事实覆盖率。

target 返回值约定：
    {
        "answer": str,                       # 最终回答文本
        "cited_fund_codes": list[str],       # 从回答中抽取出的基金代码（target 侧负责抽）
        "intent": str | None,                # 路由识别到的意图（可选）
    }
"""
from __future__ import annotations

import re
from typing import Any

# 中国基金代码：6 位数字
FUND_CODE_RE = re.compile(r"\b\d{6}\b")

# 触发"拒绝"语义的关键词（需与 multi_agent_controller.handle_compliance_failure 的拒答文案保持兼容）
REFUSAL_MARKERS = (
    "不能提供具体的投资建议",
    "不提供投资建议",
    "无法给出具体推荐",
    "请咨询专业的投资顾问",
    "投资有风险",
)


def _get_outputs(run: Any) -> dict:
    return getattr(run, "outputs", None) or {}


def _get_truth(example: Any) -> dict:
    """从 example.outputs 取标注信息（LangSmith 约定 ground truth 在 outputs）。"""
    outputs = getattr(example, "outputs", None) or {}
    return {
        "reference_answer": outputs.get("reference_answer") or "",
        "expected_fund_codes": outputs.get("expected_fund_codes") or [],
        "key_facts": outputs.get("key_facts") or [],
        "should_refuse": outputs.get("should_refuse", False),
        "intent": outputs.get("intent"),
        "expected_tool_calls": outputs.get("expected_tool_calls") or [],
    }


def _extract_cited_codes(outputs: dict) -> set[str]:
    """优先用 target 已抽好的，其次从回答正则提取。"""
    cited = outputs.get("cited_fund_codes")
    if isinstance(cited, list):
        return {c for c in cited if c}
    answer = outputs.get("answer") or ""
    return set(FUND_CODE_RE.findall(answer))


def citation_accuracy(run: Any, example: Any) -> dict:
    """引用准确率：

    precision = (cited ∩ expected) / cited
    recall    = (cited ∩ expected) / expected
    score     = F1，两者都为 0 时返回 0
    """
    outputs = _get_outputs(run)
    truth = _get_truth(example)
    expected = {c for c in (truth.get("expected_fund_codes") or []) if c}
    cited = _extract_cited_codes(outputs)

    # 没有期望引用时不参与该指标（兼容 chitchat / sensitive 等）
    if not expected:
        return {"key": "citation_accuracy", "score": 1.0, "comment": "no expected codes"}

    if not cited:
        return {
            "key": "citation_accuracy",
            "score": 0.0,
            "comment": f"expected={sorted(expected)} cited=∅",
        }

    overlap = expected & cited
    precision = len(overlap) / len(cited)
    recall = len(overlap) / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "key": "citation_accuracy",
        "score": f1,
        "comment": f"p={precision:.2f} r={recall:.2f} cited={sorted(cited)} expected={sorted(expected)}",
    }


def refusal_correctness(run: Any, example: Any) -> dict:
    """拒绝准确率：敏感问题应拒绝、非敏感问题不应拒绝。"""
    outputs = _get_outputs(run)
    truth = _get_truth(example)
    answer = outputs.get("answer") or ""

    is_refused = any(marker in answer for marker in REFUSAL_MARKERS)
    should_refuse = bool(truth.get("should_refuse"))

    correct = is_refused == should_refuse
    return {
        "key": "refusal_correctness",
        "score": 1.0 if correct else 0.0,
        "comment": f"should_refuse={should_refuse} actually_refused={is_refused}",
    }


def key_fact_coverage(run: Any, example: Any) -> dict:
    """关键事实覆盖率：key_facts 中有多少作为子串出现在回答里。

    粗粒度但确定性强，可与 LLM-judge correctness 互补。
    """
    outputs = _get_outputs(run)
    truth = _get_truth(example)
    facts = [f for f in (truth.get("key_facts") or []) if f]
    if not facts:
        return {"key": "key_fact_coverage", "score": 1.0, "comment": "no key_facts"}

    answer = outputs.get("answer") or ""
    hit = sum(1 for f in facts if f in answer)
    return {
        "key": "key_fact_coverage",
        "score": hit / len(facts),
        "comment": f"{hit}/{len(facts)} facts hit",
    }


def intent_accuracy(run: Any, example: Any) -> dict:
    """路由意图准确率（可选指标）。"""
    outputs = _get_outputs(run)
    truth = _get_truth(example)
    expected = truth.get("intent")
    if not expected:
        return {"key": "intent_accuracy", "score": 1.0, "comment": "no expected intent"}
    actual = outputs.get("intent")
    return {
        "key": "intent_accuracy",
        "score": 1.0 if actual == expected else 0.0,
        "comment": f"expected={expected} actual={actual}",
    }


def _tool_call_matches(actual: dict, expected: dict) -> bool:
    """判断一次实际工具调用是否满足某条期望调用。

    name 必须相等；args 只要求 expected 里列出的键值对在 actual 里也存在
    （允许 expected 只标注关键参数，不要求列全 actual 的所有参数）。
    """
    if actual.get("name") != expected.get("name"):
        return False
    expected_args = expected.get("args") or {}
    actual_args = actual.get("args") or {}
    return all(actual_args.get(k) == v for k, v in expected_args.items())


def tool_call_accuracy(run: Any, example: Any) -> dict:
    """工具调用准确率：期望调用的工具是否被正确调用，是否有多余/错误的调用。

    precision = (actual 中命中 expected 的条数) / len(actual)
    recall    = (expected 中被 actual 命中的条数) / len(expected)
    score     = F1，两者都为 0 时返回 0
    """
    outputs = _get_outputs(run)
    truth = _get_truth(example)
    expected = truth.get("expected_tool_calls") or []
    actual = outputs.get("tool_calls") or []

    # 没有期望工具调用时不参与该指标（如 chitchat / sensitive 类问题）
    if not expected:
        return {"key": "tool_call_accuracy", "score": 1.0, "comment": "no expected tool_calls"}

    if not actual:
        return {
            "key": "tool_call_accuracy",
            "score": 0.0,
            "comment": f"expected {len(expected)} tool_calls, got none",
        }

    matched_expected = sum(
        1 for exp in expected if any(_tool_call_matches(act, exp) for act in actual)
    )
    matched_actual = sum(
        1 for act in actual if any(_tool_call_matches(act, exp) for exp in expected)
    )

    recall = matched_expected / len(expected)
    precision = matched_actual / len(actual)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "key": "tool_call_accuracy",
        "score": f1,
        "comment": (
            f"p={precision:.2f} r={recall:.2f} "
            f"expected={expected} actual={actual}"
        ),
    }
