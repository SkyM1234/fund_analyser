"""Shared labels, query-only data validation and score acceptance."""
import json
import math
from pathlib import Path

LABELS = [
    "chitchat", "out_of_scope", "sensitive", "fund_query",
    "fund_screening", "general_finance",
]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = Path(__file__).resolve().parent / "reports"
DEFAULT_CLASSIFIER_MODEL = ROOT / "models/intent-macbert-200-v2-epoch6/best"


def read_queries(path):
    """Both reference and evaluation JSONL files contain current queries only."""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError(f"{path}:{number}: query 必须是非空字符串")
        if "history" in row:
            raise ValueError(f"{path}:{number}: 当前离线线路不接受 history 字段")
        if row.get("intent") not in LABELS:
            raise ValueError(f"{path}:{number}: intent 必须属于 {LABELS}")
        rows.append({**row, "query": row["query"].strip()})
    if not rows:
        raise ValueError(f"数据文件为空：{path}")
    labels_by_query = {}
    for row in rows:
        previous = labels_by_query.setdefault(row["query"], row["intent"])
        if previous != row["intent"]:
            raise ValueError(f"相同 query 存在冲突标签：{row['query']}")
    return rows


def select_candidate(scores, threshold, min_margin):
    if not 0 <= threshold <= 1 or not 0 <= min_margin <= 1:
        raise ValueError("阈值和分差必须在 0 到 1 之间")
    if set(scores) != set(LABELS) or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("路由必须返回六类的有限分数")
    ranked = sorted(scores, key=scores.get, reverse=True)
    first, second = ranked[:2]
    margin = scores[first] - scores[second]
    return {
        "candidate_intent": first, "score": scores[first],
        "runner_up": second, "margin": margin,
        "threshold": threshold, "required_margin": min_margin,
        "accepted": scores[first] >= threshold and margin >= min_margin,
        "scores": scores,
    }
