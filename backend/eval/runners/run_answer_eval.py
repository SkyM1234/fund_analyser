"""跑 Agent 端到端评测。

用法：
    python -m eval.runners.run_answer_eval
    python -m eval.runners.run_answer_eval --no-judge --concurrency 2
    python -m eval.runners.run_answer_eval --experiment-prefix v2-multi-agent

Agent target 复用线上 LangGraph 配置（含路由+MCP），需要：
1. MCP_ENABLED=true 且 GPU /fund_reports/search 可用
2. LLM_API_KEY 已注入业务环境

并发建议 1~2：MCP stdio 进程对并发敏感，过高易出 race。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from langsmith import Client
from langsmith.evaluation import aevaluate

from eval.config import get_eval_settings
from eval.evaluators import answer_metrics as am
from eval.evaluators import retrieval_metrics as rm  # 新增
from eval.evaluators import llm_judge as judge
from eval.targets.agent_target import agent_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 抑制 MCP stdio 清理时的 anyio cancel scope 错误（多进程环境下的已知问题）
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# MCP 初始化已移至 agent_target.py 内部（每个 worker 独立启动）
# 这里不再需要预先 bootstrap


async def run(experiment_prefix: str, concurrency: int, use_judge: bool):
    s = get_eval_settings()
    s.export_langsmith_env()
    client = Client()

    # MCP 初始化已移至 agent_target 内部，每个 worker 独立启动

    evaluators = [
        am.citation_accuracy,
        am.refusal_correctness,
        am.key_fact_coverage,
        am.intent_accuracy,
        am.tool_call_accuracy,
    ]
    if use_judge:
        evaluators += [
            judge.correctness,
            judge.answer_relevance,
        ]

    logger.info(f"开始回答评测：dataset={s.DATASET_ANSWER_NAME} concurrency={concurrency}")
    results = await aevaluate(
        agent_target,
        data=s.DATASET_ANSWER_NAME,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        max_concurrency=concurrency,
        client=client,
    )

    rows = []
    async for r in results:
        run_obj = r["run"]
        outputs = getattr(run_obj, "outputs", {}) or {}
        eval_results = r.get("evaluation_results", {}).get("results", [])
        scores = {}
        for res in eval_results:
            key = getattr(res, "key", None) or ""
            score = getattr(res, "score", None)
            comment = getattr(res, "comment", "") or ""
            scores[key] = {"score": score, "comment": comment}

        rows.append(
            {
                "example_id": str(getattr(r["example"], "id", "")),
                "query": (getattr(r["example"], "inputs", {}) or {}).get("query"),
                "answer": outputs.get("answer"),
                "intent": outputs.get("intent"),
                "cited_fund_codes": outputs.get("cited_fund_codes"),
                "tool_calls": outputs.get("tool_calls"),
                "scores": scores,
            }
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = REPORT_DIR / f"answer-{experiment_prefix}-{stamp}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"✓ 报告已落盘: {out_path}")

    if rows:
        agg: dict[str, list[float]] = {}
        for row in rows:
            for k, v in row["scores"].items():
                if isinstance(v.get("score"), (int, float)):
                    agg.setdefault(k, []).append(float(v["score"]))
        logger.info("聚合得分：")
        for k, vs in agg.items():
            logger.info(f"  {k:24s}: mean={sum(vs)/len(vs):.3f} n={len(vs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-prefix", default="answer")
    parser.add_argument("--concurrency", type=int, default=1, help="Agent 评测并发，建议 1~2")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM-judge 指标（只跑规则指标）")
    parser.add_argument(
        "--no-cn-funds-mcp",
        action="store_true",
        help="禁用 cn-funds-mcp，仅用 rag-mcp（单独测试 RAG 部分）",
    )
    args = parser.parse_args()

    # 如果指定 --no-cn-funds-mcp，覆盖配置
    if args.no_cn_funds_mcp:
        from eval.config import get_eval_settings
        eval_s = get_eval_settings()
        eval_s.ENABLE_CN_FUNDS_MCP = False
        logger.info("✓ 已禁用 cn-funds-mcp（仅测 RAG 部分）")

    asyncio.run(
        run(
            experiment_prefix=args.experiment_prefix,
            concurrency=args.concurrency,
            use_judge=not args.no_judge,
        )
    )


if __name__ == "__main__":
    main()
