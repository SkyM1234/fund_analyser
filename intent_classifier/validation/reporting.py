"""Evaluation statistics for all router combinations."""
import json
from collections import Counter

from intent_classifier.common import LABELS
from intent_classifier.pipeline import STAGES

STAGE_NAMES = {"rules": "规则", "vector": "向量", "classifier": "分类模型", "clarification": "LLM 回退"}
LABEL_NAMES = {
    "chitchat": "闲聊", "out_of_scope": "范围外", "sensitive": "敏感请求",
    "fund_query": "基金查询", "fund_screening": "基金筛选", "general_finance": "金融知识",
}
SCORE_TYPES = {
    "rules": ("rule_match", "规则命中标记"),
    "vector": ("cosine_similarity", "余弦相似度"),
    "classifier": ("softmax_probability", "Softmax 概率"),
}
STATUS_NAMES = {
    "accepted": "接受", "rejected": "未达接受条件", "disabled": "未启用",
    "skipped": "未执行（前序路由已接受）",
}


def render_report(metrics, predictions):
    """Display all six class scores without running any additional router."""
    enabled = set(metrics["enabled_routes"])
    accuracy = metrics["accepted_accuracy"]
    accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "无已接受的标注样本"
    lines = [
        "# 离线路由验证报告", "",
        f"样本数：{metrics['sample_count']}；接受：{metrics['accepted_count']}；"
        f"LLM 回退：{metrics['clarification_count']}；覆盖率：{metrics['coverage']:.2%}；"
        f"已接受样本准确率：{accuracy_text}。", "",
        "本报告仅统计本地路由；未接受的问题交给上层 LLM 识别意图，LLM 的调用与结果不计入本报告。", "",
        "规则的 0/1 表示未命中/命中；向量分数是余弦相似度；分类模型显示 Softmax 概率。"
        "相似度不代表概率，Softmax 概率未经校准，不等同于实际正确率。", "",
        "未启用或因前序命中而未执行的路由以 — 显示，不额外运行模型。", "",
    ]
    for index, prediction in enumerate(predictions, 1):
        # Quote each line so multiline queries remain visibly part of the question.
        lines.extend([f"## {index}. {prediction.get('id', index)}", ""])
        lines.extend("> " + line for line in prediction["query"].splitlines())
        final_intent = LABEL_NAMES.get(prediction["intent"], prediction["intent"]) if prediction["intent"] else "交给 LLM 识别意图"
        lines.extend(["", f"本地结果：{final_intent}；路由阶段：{STAGE_NAMES[prediction['stage']]}。", ""])
        if "expected_intent" in prediction:
            lines.extend([f"标注分类：{LABEL_NAMES[prediction['expected_intent']]}。", ""])
        lines.append("| 路由 | 状态 | 分数含义 | " + " | ".join(LABEL_NAMES[label] for label in LABELS) + " |")
        lines.append("| " + " | ".join(["---"] * (3 + len(LABELS))) + " |")
        steps = {step["stage"]: step for step in prediction["trace"]}
        for stage in STAGES:
            step = steps.get(stage, {})
            status = step.get("status", "skipped") if stage in enabled else "disabled"
            score_type, description = SCORE_TYPES[stage]
            scores = step.get("scores", {}) if status in ("accepted", "rejected") else {}
            values = []
            for label in LABELS:
                value = scores.get(label)
                if value is None:
                    values.append("—")
                elif score_type == "softmax_probability":
                    values.append(f"{value:.2%}")
                elif score_type == "rule_match":
                    values.append(str(int(value)))
                else:
                    values.append(f"{value:.4f}")
            lines.append("| " + " | ".join([STAGE_NAMES[stage], STATUS_NAMES[status], description, *values]) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_reports(output_dir, metrics, predictions):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_report(metrics, predictions), encoding="utf-8")


def summarize(predictions):
    """Summarize local coverage and accuracy; clarification_count means LLM handoffs."""
    import numpy as np
    accepted = [row for row in predictions if not row["needs_clarification"]]
    labeled = [row for row in accepted if "expected_intent" in row]
    by_stage = {}
    for stage in STAGES:
        attempts = [(row, step) for row in predictions for step in row["trace"]
                    if step["stage"] == stage and step["status"] != "disabled"]
        candidates = [(row, step) for row, step in attempts
                      if "candidate_intent" in step and "expected_intent" in row]
        by_stage[stage] = {
            "attempt_count": len(attempts),
            "accepted_count": sum(step["accepted"] for _, step in attempts),
            "candidate_accuracy": (sum(step["candidate_intent"] == row["expected_intent"]
                                       for row, step in candidates) / len(candidates)) if candidates else None,
        }
    return {
        "sample_count": len(predictions), "accepted_count": len(accepted),
        "clarification_count": len(predictions) - len(accepted),
        "coverage": len(accepted) / len(predictions),
        "accepted_correct_count": sum(row["intent"] == row["expected_intent"] for row in labeled),
        "accepted_accuracy": (sum(row["intent"] == row["expected_intent"] for row in labeled)
                              / len(labeled)) if labeled else None,
        "stage_counts": dict(Counter(row["stage"] for row in predictions)),
        "by_stage": by_stage,
        "latency_ms": {"mean": float(np.mean([row["latency_ms"] for row in predictions])),
                       "p50": float(np.percentile([row["latency_ms"] for row in predictions], 50)),
                       "p95": float(np.percentile([row["latency_ms"] for row in predictions], 95))},
    }
